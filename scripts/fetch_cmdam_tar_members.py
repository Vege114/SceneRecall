#!/usr/bin/env python3
"""Fetch exact public MP4 members with bounded HTTP Range reads, never full TARs.

Archive names and immutable commit come from saved actual HF API metadata.
First-key ordering can guide a POSITIVE-MATCH-ONLY heuristic; a lookup miss is
always inconclusive. No tokens, cookies, gated access, or private APIs are used.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.error
import urllib.parse
import urllib.request

BLOCK = 512
MIB = 1024 * 1024
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


class Inconclusive(RuntimeError):
    pass


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def number(field):
    if field[0] & 128:
        value = int.from_bytes(bytes([field[0] & 127]) + field[1:], "big")
        return value
    text = field.rstrip(b"\x00 ").lstrip(b" ")
    return int(text or b"0", 8)


def parse_header(data, offset, archive_bytes):
    if len(data) != BLOCK or data == bytes(BLOCK) or data[257:262] != b"ustar":
        return None
    try:
        expected = number(data[148:156])
        unsigned = sum(data[:148]) + 8 * 32 + sum(data[156:])
        signed = sum(value if value < 128 else value - 256 for value in data[:148] + b" " * 8 + data[156:])
        if expected not in (unsigned, signed):
            return None
        size = number(data[124:136])
        name = data[:100].split(b"\x00", 1)[0]
        prefix = data[345:500].split(b"\x00", 1)[0]
        if prefix and data[257:263] == b"ustar\x00":
            name = prefix + b"/" + name
        name = name.decode("utf-8", errors="strict")
        if not name or any(ord(character) < 32 for character in name):
            return None
        next_offset = offset + BLOCK + ((size + BLOCK - 1) // BLOCK) * BLOCK
        if size < 0 or next_offset > archive_bytes:
            return None
        return {"header_offset": offset, "data_offset": offset + BLOCK, "next_offset": next_offset,
                "name": name, "size": size, "type": data[156:157].decode("ascii") or "0"}
    except (ValueError, UnicodeError, IndexError):
        return None


def pax_records(data):
    records, position = {}, 0
    while position < len(data):
        space = data.find(b" ", position)
        if space < 0:
            raise Inconclusive("Malformed PAX length")
        length = int(data[position:space])
        if length <= space - position + 1 or position + length > len(data):
            raise Inconclusive("Malformed PAX record boundary")
        record = data[space + 1:position + length]
        if not record.endswith(b"\n") or b"=" not in record:
            raise Inconclusive("Malformed PAX record")
        key, value = record[:-1].split(b"=", 1)
        # macOS SCHILY.xattr records can contain binary extended attributes.
        # Only path/size affect member extraction; preserve other values losslessly.
        records[key.decode("utf-8")] = value.decode("utf-8", errors="surrogateescape")
        position += length
    return records


def member_key(name):
    basename = PurePosixPath(name).name
    if basename.startswith("._"):
        basename = basename[2:]
    stem = basename.rsplit(".", 1)[0]
    return stem if ID_PATTERN.fullmatch(stem) else None


class PublicRanges:
    def __init__(self, args):
        self.args = args
        self.total_bytes = self.probe_bytes = self.requests = 0
        self.opener = urllib.request.build_opener()  # No cookie or authorization handler.

    def read(self, url, start, end, *, expected_size, probe=True):
        count = end - start + 1
        if start < 0 or count <= 0 or count >= expected_size:
            raise Inconclusive("Refusing invalid range or entire-archive request")
        if self.total_bytes + count > self.args.max_download_bytes:
            raise Inconclusive("Total Range byte budget exhausted")
        if probe and self.probe_bytes + count > self.args.max_probe_bytes:
            raise Inconclusive("Metadata/probe byte budget exhausted")
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}",
            "User-Agent": "SceneRecall-public-member-fetch/1.0", "Accept-Encoding": "identity"})
        self.requests += 1
        with self.opener.open(request, timeout=self.args.timeout) as response:
            if response.status != 206:
                raise Inconclusive(f"Range unsupported: HTTP {response.status}; body rejected")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or tuple(map(int, match.groups())) != (start, end, expected_size):
                raise Inconclusive("Content-Range does not match the exact request/archive size")
            result = bytearray()
            while len(result) < count:
                block = response.read(min(MIB, count - len(result)))
                if not block:
                    raise Inconclusive("Truncated HTTP Range body")
                result.extend(block)
                self.total_bytes += len(block)
                if probe:
                    self.probe_bytes += len(block)
            return bytes(result)


class RemoteTar:
    def __init__(self, path, size, url, client, args, anchors, local_prefix=None):
        self.path, self.size, self.url = path, size, url
        self.client, self.args, self.anchors = client, args, anchors
        self.local_prefix = local_prefix

    def read(self, start, end):
        if self.local_prefix is not None and end < len(self.local_prefix):
            return self.local_prefix[start:end + 1]
        return self.client.read(self.url, start, end, expected_size=self.size)

    def entry(self, offset):
        pax, long_name = {}, None
        for _ in range(16):
            if offset + BLOCK > self.size:
                raise Inconclusive("TAR header extends beyond archive")
            entry = parse_header(self.read(offset, offset + BLOCK - 1), offset, self.size)
            if entry is None:
                raise Inconclusive(f"No legal checksum-verified USTAR header at {offset}")
            if entry["type"] in ("x", "g", "L", "K"):
                if entry["size"] > MIB:
                    raise Inconclusive("TAR metadata extension exceeds 1 MiB")
                payload = self.read(entry["data_offset"], entry["data_offset"] + entry["size"] - 1) if entry["size"] else b""
                if entry["type"] in ("x", "g"):
                    pax.update(pax_records(payload))
                elif entry["type"] == "L":
                    long_name = payload.rstrip(b"\x00").decode("utf-8")
                offset = entry["next_offset"]
                continue
            entry["name"] = pax.get("path", long_name or entry["name"])
            if "size" in pax:
                entry["size"] = int(pax["size"])
                entry["next_offset"] = entry["data_offset"] + ((entry["size"] + BLOCK - 1) // BLOCK) * BLOCK
                if entry["size"] < 0 or entry["next_offset"] > self.size:
                    raise Inconclusive("PAX member size outside archive")
            entry["key"] = member_key(entry["name"])
            if entry["key"]:
                self.anchors[str(entry["header_offset"])] = entry
            return entry
        raise Inconclusive("Too many consecutive TAR metadata headers")

    def after_offset(self, offset):
        """Find a legal aligned header by incremental 1/2/4/.../64 MiB scans."""
        offset = (offset + BLOCK - 1) // BLOCK * BLOCK
        data, previous, limit = bytearray(), 0, MIB
        maximum = min(self.args.scan_limit_mib * MIB, self.size - offset)
        while previous < maximum:
            amount = min(limit, maximum)
            data.extend(self.read(offset + previous, offset + amount - 1))
            search = max(0, previous - BLOCK)
            while True:
                magic = data.find(b"ustar", search)
                if magic < 0:
                    break
                header_start = magic - 257
                search = magic + 1
                if header_start < 0 or (offset + header_start) % BLOCK or header_start + BLOCK > len(data):
                    continue
                if parse_header(data[header_start:header_start + BLOCK], offset + header_start, self.size):
                    entry = self.entry(offset + header_start)
                    if entry.get("key"):
                        return entry
            previous, limit = amount, limit * 2
        raise Inconclusive(f"No legal member header within {maximum} bytes after {offset}; this is NOT proof of absence")

    def matching_video(self, entry, target):
        for _ in range(8):
            if entry.get("key") != target:
                return None
            basename = PurePosixPath(entry["name"]).name
            if basename == target + ".mp4" and entry["type"] in ("0", "\x00"):
                # Re-read the exact target header online before trusting cached/probed metadata.
                prefix = self.local_prefix
                self.local_prefix = None
                try:
                    checked = self.entry(entry["header_offset"])
                finally:
                    self.local_prefix = prefix
                if checked["name"] != entry["name"] or checked["size"] != entry["size"]:
                    raise Inconclusive("Exact target header changed during verification")
                return checked
            entry = self.entry(entry["next_offset"])
        raise Inconclusive("Target key found, but its real MP4 did not follow the sidecar")

    def locate(self, target):
        first = self.entry(0)
        if not self.args.allow_sorted_heuristic:
            if not self.args.linear_max_headers:
                raise Inconclusive("Internal key sorting is unverified; enable explicit sorted heuristic or bounded scoped linear lookup")
            entry = first
            for _ in range(self.args.linear_max_headers):
                found = self.matching_video(entry, target) if entry.get("key") == target else None
                if found:
                    return found
                entry = self.entry(entry["next_offset"])
            raise Inconclusive("Bounded scoped linear header limit reached; NOT proof of absence")
        low, high = 0, self.size - BLOCK
        for entry in list(self.anchors.values()):
            key = entry.get("key")
            if key == target:
                found = self.matching_video(entry, target)
                if found:
                    return found
            elif key and key < target:
                low = max(low, entry["next_offset"])
            elif key and key > target:
                high = min(high, entry["header_offset"])
        empty_scan_ranges = []
        for iteration in range(96):
            if high <= low:
                break
            middle = ((low + high) // 2 // BLOCK) * BLOCK
            # Do not repeatedly download the same interior of a large member.
            # A header-free region changes no key bounds, but another pivot
            # nearer the known lower header can still give a valid key.
            while any(start <= middle < end for start, end in empty_scan_ranges):
                earlier = ((low + middle) // 2 // BLOCK) * BLOCK
                if earlier <= low:
                    middle = low
                    break
                middle = earlier
            try:
                entry = self.entry(low) if high - low < MIB or middle == low else self.after_offset(middle)
            except Inconclusive as error:
                # A midpoint may lie inside a large video. The lower bound is
                # an exact next-header boundary, so advance through that member
                # instead of treating a bounded scan miss as absence.
                emit("midpoint_scan_inconclusive", archive=self.path, target=target,
                     offset=middle, fallback_header=low, reason=str(error))
                empty_scan_ranges.append((middle, middle + self.args.scan_limit_mib * MIB))
                continue
            key = entry.get("key")
            emit("lookup_probe", archive=self.path, target=target, iteration=iteration,
                 offset=entry["header_offset"], key=key, probe_bytes=self.client.probe_bytes)
            if key == target:
                found = self.matching_video(entry, target)
                if found:
                    return found
            if not key:
                low = entry["next_offset"]
            elif key < target:
                low = entry["next_offset"]
            else:
                high = middle if high - low >= MIB else low
        raise Inconclusive("Sorted-key heuristic did not locate an exact MP4; internal order/coverage remains unverified, NOT proof of absence")


def download_member(tar, entry, target, args, state):
    if entry["size"] <= 0 or entry["size"] > args.max_member_bytes:
        raise Inconclusive("Matched member size exceeds permitted download size")
    destination = args.output / (target + ".mp4")
    sidecar = args.output / (target + ".json")
    identity = {"repository": state["repository"], "commit": state["commit"], "archive": tar.path,
                "member": entry["name"], "data_offset": entry["data_offset"], "bytes": entry["size"]}
    if destination.exists():
        prior = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if prior.get("identity") != identity or destination.stat().st_size != entry["size"] or prior.get("sha256") != file_sha(destination):
            raise Inconclusive("Existing destination does not match this verified member; refusing overwrite")
        return prior
    partial = destination.with_suffix(".mp4.part")
    partial_meta = destination.with_suffix(".mp4.part.json")
    if partial.exists() and (not partial_meta.exists() or json.loads(partial_meta.read_text()) != identity):
        raise Inconclusive("Partial destination belongs to another member")
    atomic_json(partial_meta, identity)
    completed = partial.stat().st_size if partial.exists() else 0
    if completed > entry["size"]:
        raise Inconclusive("Partial file exceeds member size")
    with partial.open("ab") as stream:
        while completed < entry["size"]:
            length = min(8 * MIB, entry["size"] - completed)
            data = tar.client.read(tar.url, entry["data_offset"] + completed,
                                   entry["data_offset"] + completed + length - 1,
                                   expected_size=tar.size, probe=False)
            stream.write(data)
            stream.flush()
            completed += len(data)
            emit("member_download", clip_id=target, bytes=completed, total=entry["size"])
    with partial.open("rb") as stream:
        prefix = stream.read(64)
    if b"ftyp" not in prefix:
        raise Inconclusive("Matched .mp4 payload lacks an MP4 ftyp signature near its start")
    digest = file_sha(partial)
    partial.replace(destination)
    result = {"status": "downloaded", "clip_id": target, "path": str(destination.resolve()), "identity": identity,
              "bytes": completed, "sha256": digest, "source_url": tar.url,
              "source_type": "third_party_public_huggingface_mirror", "official_dataset_host": False,
              "private": False, "gated": False, "credentials_used": False}
    atomic_json(sidecar, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--first-members", type=Path, required=True)
    parser.add_argument("--header-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", required=True)
    parser.add_argument("--tar", action="append", default=[])
    parser.add_argument("--allow-sorted-heuristic", action="store_true")
    parser.add_argument("--linear-max-headers", type=int, default=0)
    parser.add_argument("--scan-limit-mib", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=64)
    parser.add_argument("--max-probe-bytes", type=int, default=1024 ** 3)
    parser.add_argument("--max-download-bytes", type=int, default=20 * 1024 ** 3)
    parser.add_argument("--max-member-bytes", type=int, default=2 * 1024 ** 3)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--locate-only", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    if metadata.get("private") is not False or metadata.get("gated") is not False or metadata.get("disabled") is True:
        raise ValueError("Only explicitly public, ungated, enabled repositories are accepted")
    repository, commit = metadata["id"], metadata["sha"]
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository) or not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Actual repository ID and immutable commit required")
    names = {item["rfilename"] for item in metadata["siblings"] if item["rfilename"].endswith(".tar")}
    first_members = json.loads(args.first_members.read_text(encoding="utf-8-sig"))
    archives = sorted((item for item in first_members if item["archive"] in names), key=lambda item: item["archive"])
    if {item["archive"] for item in archives} != names:
        raise ValueError("First-member metadata must cover the actual API-listed TAR names")
    first_keys = [member_key(item["first_member"]) for item in archives]
    if any(key is None for key in first_keys) or first_keys != sorted(first_keys):
        raise ValueError("Saved archive first keys are invalid or not ordered")
    if any(path not in names for path in args.tar) or any(not ID_PATTERN.fullmatch(clip) for clip in args.clip_id):
        raise ValueError("Unknown archive or invalid exact clip ID")
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "fetch-report.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema_version": "cmdam-public-range-members-v1", "repository": repository, "commit": commit,
        "metadata_sha256": file_sha(args.metadata), "first_members_sha256": file_sha(args.first_members),
        "ordering_status": "first_keys_ordered_only; internal_sorting_unverified; lookup_misses_are_inconclusive",
        "anchors": {}, "clips": {}}
    if state.get("repository") != repository or state.get("commit") != commit:
        raise ValueError("Output cache belongs to another repository revision")
    client = PublicRanges(args)
    success = True
    for target in dict.fromkeys(args.clip_id):
        index = bisect.bisect_right(first_keys, target) - 1
        candidates = [item for item in archives if item["archive"] in args.tar] if args.tar else archives[max(0, index):max(0, index) + 1]
        result = None
        for item in candidates:
            path = item["archive"]
            url = "https://huggingface.co/datasets/" + repository + "/resolve/" + commit + "/" + urllib.parse.quote(path, safe="/")
            prefix = None
            if args.header_cache:
                cached = args.header_cache / (PurePosixPath(path).name + ".header.bin")
                if cached.is_file():
                    prefix = cached.read_bytes()
            tar = RemoteTar(path, int(item["archive_bytes"]), url, client, args,
                            state["anchors"].setdefault(path, {}), prefix)
            try:
                entry = tar.locate(target)
                result = {"status": "located", "clip_id": target, "archive": path, "entry": entry,
                          "source_type": "third_party_public_huggingface_mirror"} if args.locate_only else download_member(tar, entry, target, args, state)
                break
            except (Inconclusive, OSError, ValueError, urllib.error.URLError) as error:
                result = {"status": "inconclusive_not_absent", "clip_id": target, "archive": path, "reason": str(error)}
                emit("lookup_inconclusive", **result)
            finally:
                state["session_requests"] = client.requests
                state["session_bytes_received"] = client.total_bytes
                state["session_probe_bytes_received"] = client.probe_bytes
                atomic_json(state_path, state)
        state["clips"][target] = result
        success &= bool(result and result["status"] in ("downloaded", "located"))
        atomic_json(state_path, state)
        emit("clip_result", **(result or {"clip_id": target, "status": "inconclusive_not_absent"}))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
