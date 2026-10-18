#!/usr/bin/env python3
"""Index public ZIPs with verified HTTP ranges; retrieve only selected CMD-AM videos.

Python standard library only. No annotation bearer links, cookies, full archive
fallbacks, or extraction to archive-supplied paths. Public input URLs are never
printed; redirects and response headers are not written to logs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False), flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RangeClient:
    def __init__(self, url, timeout=60):
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme == "https" and parsed.hostname, "Archive URL must be public HTTPS")
        self.url, self.timeout, self.total = url, timeout, None
        self.lock = threading.Lock()

    def fetch(self, start=None, end=None, *, suffix=None):
        spec = f"bytes=-{suffix}" if suffix is not None else f"bytes={start}-{end}"
        for attempt in range(3):
            try:
                request = urllib.request.Request(self.url, headers={"Range": spec, "Accept-Encoding": "identity",
                                                                   "User-Agent": "SceneRecall-CMDAM-Range/1.0"})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    # Check before reading: HTTP 200 archive bodies are forbidden.
                    require(response.status == 206, f"Range server returned HTTP {response.status}; full body rejected")
                    value = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
                    require(match is not None, "Missing/invalid Content-Range")
                    first, last, total = map(int, match.groups())
                    require(0 <= first <= last < total, "Invalid byte-range bounds")
                    expected_first = max(0, total - suffix) if suffix is not None else start
                    expected_last = total - 1 if suffix is not None else end
                    require((first, last) == (expected_first, expected_last), "Server returned a different byte range")
                    require(response.headers.get("Content-Encoding", "identity").lower() == "identity", "Encoded range rejected")
                    wanted = last - first + 1
                    length = response.headers.get("Content-Length")
                    require(length is None or int(length) == wanted, "Content-Length differs from requested range")
                    with self.lock:
                        require(self.total is None or self.total == total, "Archive total size changed between requests")
                        self.total = total
                    data = response.read(wanted + 1)
                    require(len(data) == wanted, "Response body byte count differs from Content-Range")
                    return data, first
            except ValueError:
                raise
            except (urllib.error.HTTPError, urllib.error.URLError, http.client.IncompleteRead, TimeoutError, OSError) as error:
                if attempt == 2:
                    code = getattr(error, "code", None)
                    raise RuntimeError(f"Range network request failed after 3 attempts ({'HTTP ' + str(code) if code else type(error).__name__})") from None
                time.sleep(attempt + 1)
        raise RuntimeError("Unreachable range state")


def zip64_values(extra, usize, csize, offset, disk):
    position, found = 0, None
    while position + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, position)
        position += 4
        require(position + length <= len(extra), "Truncated ZIP extra field")
        if kind == 1:
            found = extra[position:position + length]
        position += length
    if any(value == sentinel for value, sentinel in ((usize, 0xffffffff), (csize, 0xffffffff),
                                                    (offset, 0xffffffff), (disk, 0xffff))):
        require(found is not None, "Missing required ZIP64 extra values")
        position, result = 0, []
        for value, sentinel, size in ((usize, 0xffffffff, 8), (csize, 0xffffffff, 8),
                                      (offset, 0xffffffff, 8), (disk, 0xffff, 4)):
            if value == sentinel:
                require(position + size <= len(found), "Truncated ZIP64 extra value")
                value = int.from_bytes(found[position:position + size], "little")
                position += size
            result.append(value)
        return result
    return usize, csize, offset, disk


def central_entries(data, expected_count):
    position, entries = 0, []
    while len(entries) < expected_count:
        require(position + 46 <= len(data), "Truncated ZIP central directory")
        fields = struct.unpack_from("<4s6H3I5H2I", data, position)
        require(fields[0] == b"PK\x01\x02", "Invalid central-directory signature")
        flags, method, crc, csize, usize = fields[3], fields[4], fields[7], fields[8], fields[9]
        name_length, extra_length, comment_length, disk, offset = fields[10], fields[11], fields[12], fields[13], fields[16]
        end = position + 46 + name_length + extra_length + comment_length
        require(end <= len(data), "Truncated central-directory record")
        raw_name = data[position + 46:position + 46 + name_length]
        extra = data[position + 46 + name_length:position + 46 + name_length + extra_length]
        usize, csize, offset, disk = zip64_values(extra, usize, csize, offset, disk)
        require(disk == 0, "Multi-disk ZIP archives are unsupported")
        filename = raw_name.decode("utf-8" if flags & 0x800 else "cp437")
        entries.append({"filename": filename, "filename_hex": raw_name.hex(), "flags": flags, "method": method,
                        "crc32": f"{crc:08x}", "compressed_size": csize, "uncompressed_size": usize,
                        "local_header_offset": offset})
        position = end
    return entries


def build_index(year, url, output, *, refresh=False):
    path = output / "archives" / f"{year}.central-index.json"
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        require(cached.get("schema_version") == "cmdam-zip-central-index-v1" and cached.get("public_url_sha256") == url_hash,
                "Existing index belongs to a different public archive URL; use --refresh-index")
        require(isinstance(cached.get("entries"), list) and cached.get("archive_size", 0) > 0, "Invalid cached archive index")
        emit("archive_index_cached", archive=year, entries=len(cached["entries"]), archive_bytes=cached["archive_size"])
        return cached
    client = RangeClient(url)
    tail, tail_start = client.fetch(suffix=131072)
    position = tail.rfind(b"PK\x05\x06")
    require(position >= 0 and position + 22 <= len(tail), "ZIP end record not found")
    end = struct.unpack_from("<4s4H2IH", tail, position)
    require(position + 22 + end[7] == len(tail), "ZIP end comment/trailing bytes invalid")
    require(end[1] == end[2] == 0, "Multi-disk ZIP archive rejected")
    count, central_size, central_offset = end[4], end[5], end[6]
    zip64 = count == 0xffff or central_size == 0xffffffff or central_offset == 0xffffffff
    if zip64:
        locator_position = position - 20
        if locator_position >= 0:
            locator = tail[locator_position:position]
        else:
            locator, _ = client.fetch(tail_start + position - 20, tail_start + position - 1)
        signature, disk, offset, disks = struct.unpack("<4sIQI", locator)
        require(signature == b"PK\x06\x07" and disk == 0 and disks == 1, "Invalid ZIP64 locator")
        record, _ = client.fetch(offset, offset + 55)
        z = struct.unpack("<4sQ2H2I4Q", record)
        require(z[0] == b"PK\x06\x06" and z[1] >= 44 and z[4] == z[5] == 0 and z[6] == z[7], "Invalid ZIP64 end record")
        count, central_size, central_offset = z[7], z[8], z[9]
    else:
        require(end[3] == end[4], "ZIP directory entry count differs between disks")
    require(0 < central_size <= 64 * 1024 * 1024 and central_offset + central_size <= client.total,
            "Invalid or oversized central-directory range")
    require(0 < count <= 1_000_000, "Unreasonable central-directory entry count")
    data, _ = client.fetch(central_offset, central_offset + central_size - 1)
    entries = central_entries(data, count)
    index = {"schema_version": "cmdam-zip-central-index-v1", "archive": year,
             "public_url_sha256": url_hash, "origin_host": urllib.parse.urlsplit(url).hostname,
             "archive_size": client.total, "zip64": zip64, "central_offset": central_offset,
             "central_size": central_size, "central_sha256": hashlib.sha256(data).hexdigest(),
             "entries": entries, "indexed_utc": datetime.now(timezone.utc).isoformat()}
    save_json(path, index)
    emit("archive_indexed", archive=year, entries=len(entries), archive_bytes=client.total, central_bytes=central_size)
    return index


def select_members(selection, indexes):
    wanted = {row["clip_id"]: row for row in selection["clips"]}
    matches = {clip: [] for clip in wanted}
    for year, index in indexes.items():
        for entry in index["entries"]:
            name = entry["filename"].replace("\\", "/")
            path = PurePosixPath(name)
            if path.suffix.lower() not in {".mp4", ".mkv", ".webm"} or path.stem not in wanted:
                continue
            require(not path.is_absolute() and ".." not in path.parts and ":" not in name and "\x00" not in name,
                    "Unsafe selected ZIP member path rejected")
            require(not entry["flags"] & 1 and entry["method"] in (0, 8), "Selected ZIP member is encrypted or uses unsupported compression")
            matches[path.stem].append({**entry, "clip_id": path.stem, "archive": year,
                                      "output_filename": path.stem + path.suffix.lower(),
                                      "archive_size": index["archive_size"], "central_sha256": index["central_sha256"],
                                      "central_offset": index["central_offset"]})
    selected, missing, duplicates = [], [], []
    for clip, options in sorted(matches.items()):
        if not options:
            missing.append(clip)
            continue
        signatures = {(item["crc32"], item["uncompressed_size"]) for item in options}
        require(len(signatures) == 1, "Same clip ID has conflicting contents in multiple archives: " + clip)
        options.sort(key=lambda item: (str(item["archive"]) != str(wanted[clip].get("archive_year")), str(item["archive"])))
        selected.append(options[0])
        if len(options) > 1:
            duplicates.append({"clip_id": clip, "archives": [item["archive"] for item in options],
                               "chosen_archive": options[0]["archive"], "reason": "identical CRC32 and uncompressed size"})
    return selected, missing, duplicates


def checksum(path):
    crc, count = 0, 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
            count += len(block)
    return f"{crc & 0xffffffff:08x}", count


def validate_existing(path, member):
    if not path.exists():
        return False
    require(path.is_file() and path.stat().st_size == member["uncompressed_size"], "Existing video has wrong size: " + member["clip_id"])
    crc, size = checksum(path)
    require(crc == member["crc32"] and size == member["uncompressed_size"], "Existing video failed CRC32: " + member["clip_id"])
    return True


def final_path(output, member):
    name = member["output_filename"]
    require(name in {member["clip_id"] + suffix for suffix in (".mkv", ".mp4", ".webm")}, "Unsafe output filename")
    return output / name


def local_data_offset(client, member):
    offset = member["local_header_offset"]
    header, _ = client.fetch(offset, offset + 29)
    value = struct.unpack("<4s5H3I2H", header)
    require(value[0] == b"PK\x03\x04" and value[2] == member["flags"] and value[3] == member["method"],
            "Local ZIP header differs from central directory")
    length = value[9] + value[10]
    require(0 < value[9] <= 65535 and length <= 131070, "Invalid local filename/extra length")
    details, _ = client.fetch(offset + 30, offset + 30 + length - 1)
    require(details[:value[9]].hex() == member["filename_hex"], "Local/central ZIP filenames differ")
    if not member["flags"] & 8:
        require(f"{value[6]:08x}" == member["crc32"], "Local/central CRC32 differs")
    data_offset = offset + 30 + length
    require(data_offset + member["compressed_size"] <= member["central_offset"], "Selected compressed member overlaps central directory")
    return data_offset


def inflate_file(compressed, extracted, member):
    decoder, crc, size = zlib.decompressobj(-15), 0, 0
    with compressed.open("rb") as source, extracted.open("wb") as destination:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            pending = block
            while True:
                decoded = decoder.decompress(pending, 1024 * 1024)
                pending = decoder.unconsumed_tail
                size += len(decoded)
                require(size <= member["uncompressed_size"], "Deflate expanded beyond declared uncompressed size")
                destination.write(decoded)
                crc = zlib.crc32(decoded, crc)
                if not pending and len(decoded) < 1024 * 1024:
                    break
        tail = decoder.flush()
        size += len(tail)
        require(size <= member["uncompressed_size"], "Deflate flush exceeded declared size")
        destination.write(tail)
        crc = zlib.crc32(tail, crc)
        destination.flush()
        os.fsync(destination.fileno())
    require(decoder.eof and not decoder.unused_data, "Deflate stream did not end exactly at member boundary")
    require(size == member["uncompressed_size"] and f"{crc & 0xffffffff:08x}" == member["crc32"], "Extracted video failed size/CRC32")


def partial_paths(output, member):
    root = output / ".partial"
    # Fixed validated clip IDs only; archive member paths never control output.
    return root / (member["clip_id"] + ".compressed.partial"), root / (member["clip_id"] + ".download.json")


def download_one(member, url, output, chunk_bytes):
    clip = member["clip_id"]
    final = final_path(output, member)
    if validate_existing(final, member):
        emit("clip_already_verified", clip_id=clip, bytes=member["uncompressed_size"])
        return {"clip_id": clip, "archive": member["archive"], "status": "existing_crc_verified", "path": str(final),
                "bytes": member["uncompressed_size"], "crc32": member["crc32"]}
    require(member["compressed_size"] > 0 and member["uncompressed_size"] > 0, "Empty selected video member")
    client = RangeClient(url)
    client.total = member["archive_size"]
    offset = local_data_offset(client, member)
    compressed, state_path = partial_paths(output, member)
    compressed.parent.mkdir(parents=True, exist_ok=True)
    identity = {"clip_id": clip, "archive": member["archive"], "archive_size": member["archive_size"],
                "central_sha256": member["central_sha256"], "data_offset": offset,
                "compressed_size": member["compressed_size"], "uncompressed_size": member["uncompressed_size"],
                "crc32": member["crc32"], "method": member["method"],
                "public_url_sha256": hashlib.sha256(url.encode()).hexdigest()}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        require(state.get("identity") == identity, "Partial download identity mismatch: " + clip)
        committed = state.get("committed_compressed_bytes")
        require(type(committed) is int and 0 <= committed <= member["compressed_size"], "Invalid partial committed byte count")
        require((compressed.exists() or committed == 0) and (not compressed.exists() or compressed.stat().st_size >= committed),
                "Partial file is shorter than its committed resume state")
    else:
        require(not compressed.exists(), "Unidentified partial file exists: " + clip)
        committed = 0
        save_json(state_path, {"schema_version": "cmdam-range-resume-v1", "identity": identity, "committed_compressed_bytes": 0})
    # An interrupted write may leave bytes beyond the last atomically committed
    # state. Only that uncommitted suffix of this utility's partial is discarded.
    with compressed.open("r+b" if compressed.exists() else "w+b") as stream:
        stream.truncate(committed)
        stream.seek(committed)
        last_report = time.monotonic()
        while committed < member["compressed_size"]:
            wanted = min(chunk_bytes, member["compressed_size"] - committed)
            block, _ = client.fetch(offset + committed, offset + committed + wanted - 1)
            stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
            committed += len(block)
            save_json(state_path, {"schema_version": "cmdam-range-resume-v1", "identity": identity,
                                  "committed_compressed_bytes": committed})
            if time.monotonic() - last_report >= 10 or committed == member["compressed_size"]:
                emit("clip_download_progress", clip_id=clip, archive=member["archive"], compressed_bytes=committed,
                     compressed_total=member["compressed_size"])
                last_report = time.monotonic()
    if member["method"] == 0:
        require(checksum(compressed) == (member["crc32"], member["uncompressed_size"]), "Stored video failed size/CRC32: " + clip)
        verified = compressed
    else:
        verified = compressed.with_name(clip + ".extracted.partial")
        inflate_file(compressed, verified, member)
    # Windows rename refuses to replace an existing destination. POSIX uses an
    # exclusive hard link instead, avoiding its rename-overwrite semantics.
    # Both paths live on the same output filesystem; no archive path is used.
    if os.name == "nt":
        os.rename(verified, final)
    else:
        os.link(verified, final)
        verified.unlink()
    if compressed.exists():
        compressed.unlink()
    result = {"clip_id": clip, "archive": member["archive"], "status": "downloaded_crc_verified", "path": str(final),
              "bytes": member["uncompressed_size"], "compressed_bytes": member["compressed_size"], "crc32": member["crc32"]}
    save_json(state_path, {"schema_version": "cmdam-range-resume-v1", "identity": identity,
                          "committed_compressed_bytes": committed, "completed": result})
    emit("clip_download_complete", clip_id=clip, archive=member["archive"], bytes=result["bytes"], crc32=result["crc32"])
    return result


def required_disk_bytes(members, output):
    required = 0
    for member in members:
        final = final_path(output, member)
        if final.exists():
            validate_existing(final, member)
            continue
        partial, _ = partial_paths(output, member)
        present = min(partial.stat().st_size, member["compressed_size"]) if partial.exists() else 0
        required += member["compressed_size"] - present
        if member["method"] == 8:
            required += member["uncompressed_size"]
    return required


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--years", nargs="+")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--chunk-size-mib", type=int, default=4)
    parser.add_argument("--refresh-index", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 3 or not 1 <= args.chunk_size_mib <= 16:
        parser.error("threads must be 1..3 and chunk-size-mib 1..16")
    return args


def main(args):
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    archives = json.loads(args.archives.read_text(encoding="utf-8"))
    require(isinstance(archives, dict) and archives, "Archives input must map observed year names to public HTTPS URLs")
    archives = {str(year): value["url"] if isinstance(value, dict) else value for year, value in archives.items()}
    if args.years:
        years = {year for value in args.years for year in value.split(",")}
        require(years.issubset(archives), "Requested archive year was not present in observed public URLs")
        archives = {year: url for year, url in archives.items() if year in years}
    require(all(re.fullmatch(r"\d{4}", year) and isinstance(url, str) for year, url in archives.items()), "Invalid public archive mapping")
    require(len({row["clip_id"] for row in selection["clips"]}) == len(selection["clips"])
            and all(re.fullmatch(r"[A-Za-z0-9_-]{11}", row["clip_id"]) for row in selection["clips"]), "Invalid clip selection")
    args.output.mkdir(parents=True, exist_ok=True)
    indexes = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(build_index, year, url, args.output, refresh=args.refresh_index): year for year, url in archives.items()}
        for future in as_completed(futures):
            indexes[futures[future]] = future.result()
    members, missing, duplicates = select_members(selection, indexes)
    report = {"schema_version": "cmdam-zip-selected-members-v1", "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
              "archives_indexed": sorted(indexes), "scope": "requested_archive_subset" if args.years else "all_observed_archives",
              "selected_clip_count": len(members), "missing_clip_ids": missing, "identical_duplicate_matches": duplicates,
              "compressed_bytes": sum(item["compressed_size"] for item in members),
              "uncompressed_bytes": sum(item["uncompressed_size"] for item in members), "members": members,
              "archive_year_hint_is_not_a_filter": True, "no_unobserved_2017_archive_url_was_invented": True}
    report_path = args.output / ("selection-index-" + "-".join(sorted(indexes)) + ".json")
    save_json(report_path, report)
    emit("selection_index_complete", report=str(report_path), archives=sorted(indexes), selected=len(members),
         missing=missing, compressed_bytes=report["compressed_bytes"], uncompressed_bytes=report["uncompressed_bytes"])
    if args.index_only:
        return 2 if missing and not args.years else 0
    require(members and (args.years or not missing), "Missing clips remain; do not silently download a partial unscoped selection")
    required = required_disk_bytes(members, args.output)
    free = shutil.disk_usage(args.output).free
    require(free >= required + 256 * 1024 * 1024, "Insufficient free disk for selected compressed/uncompressed members plus reserve")
    emit("download_plan", selected=len(members), required_additional_bytes=required, free_bytes=free, threads=args.threads)
    results, failures = [], []
    summary_path = args.output / ("download-summary-" + "-".join(sorted(indexes)) + ".json")
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(download_one, member, archives[member["archive"]], args.output,
                               args.chunk_size_mib * 1024 * 1024): member for member in members}
        for future in as_completed(futures):
            member = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                # All network exceptions have already been sanitized by RangeClient.
                failures.append({"clip_id": member["clip_id"], "error": f"{type(error).__name__}: {error}"})
                emit("clip_download_failed", clip_id=member["clip_id"], error_type=type(error).__name__)
            save_json(summary_path, {"schema_version": "cmdam-selected-download-v1", "selection_index": str(report_path),
                      "completed": results, "failures": failures, "selected": len(members),
                      "all_selected_crc_verified": len(results) == len(members) and not failures,
                      "video_sha256_and_ffprobe": "To be performed separately; this utility verifies ZIP CRC32 and exact uncompressed size"})
    emit("selected_download_finished", completed=len(results), failed=len(failures), summary=str(summary_path))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
