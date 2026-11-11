"""Actual FFmpeg round trips for source-backed evidence and bounded cache lifetimes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
from pathlib import Path
import shutil
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from scenerecall import keyframes, media, sample_frames
from scenerecall.library import Library, atomic_json, digest, read_json, source_snapshot
from test_retrieval_frame_capture import many_targets


@pytest.fixture
def source_frames(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for the source-backed frame fixture")
    path, metadata, frames = many_targets(tmp_path, 5)
    library = Library(tmp_path / "library")
    aid = "sample_episode"
    fingerprint = media.fingerprint(path)
    directory = library.asset_dir(aid)
    atomic_json(directory / "manifest.json", {"id": aid, "work_id": "synthetic", "title": "fixture",
        "version": "test", "fingerprint": fingerprint, "created_at": "test", "duration_ms": metadata["duration_ms"],
        "start_time_seconds": metadata["start_time_seconds"]})
    atomic_json(library.root / "private/media-locations.json", {aid: source_snapshot(path)})
    selected = [{**frame, "id": "kf_" + digest([aid, fingerprint, frame["pts"], frame["time_base"]])} for frame in frames]
    outputs = [directory / "frames" / (frame["id"] + ".jpg") for frame in selected]
    extracted = keyframes.extract_keyframes_stream(path, selected, outputs, metadata=metadata)
    library.register_frames(aid, [{**f, "sample_generated": True} for f in extracted])
    pictures = {frame["id"]: Path(frame["path"]).read_bytes() for frame in extracted}
    for output in outputs:
        output.unlink()
    return library, aid, path, selected, pictures


def test_lazy_batch_matches_archived_bytes_and_concurrent_requests_decode_once(source_frames, monkeypatch):
    library, aid, _, frames, pictures = source_frames
    ids = list(pictures)
    calls, hashes = [], []
    extract, fingerprint = keyframes.extract_keyframes, media.fingerprint

    def tracked(*args, **kwargs):
        calls.append(len(args[1]))
        return extract(*args, **kwargs)

    def hash_once(source):
        hashes.append(source)
        return fingerprint(source)

    monkeypatch.setattr(keyframes, "extract_keyframes", tracked)
    monkeypatch.setattr(media, "fingerprint", hash_once)
    library.check_frame_entries(aid, ids)
    assert not calls and not hashes
    with ThreadPoolExecutor(4) as pool:
        copies = list(pool.map(lambda _: library.frame_bytes(aid, ids[0]), range(4)))
    assert all(raw == pictures[ids[0]] for raw in copies)
    assert calls == [1] and len(hashes) == 1
    library.sample_frames.budget = max(map(len, pictures.values()))
    paths = library.frame_paths(aid, ids)
    assert {fid: path.read_bytes() for fid, path in paths.items()} == pictures
    assert calls == [1, len(frames) - 1] and len(hashes) == 1
    assert not list((library.asset_dir(aid) / "frames").glob("*.jpg"))
    # Cache eviction does not invalidate a filename already handed to an async model.
    filename = str(paths[ids[0]])
    assert filename.frame_ref == (aid, ids[0])
    assert library.sample_frames.bytes <= library.sample_frames.budget
    assert len(library.sample_frames.cache) < len(ids)
    del paths
    assert Path(filename).read_bytes() == pictures[ids[0]]


def test_legacy_plan_and_offline_metadata_backups_do_not_invent_millisecond_frames(source_frames, tmp_path):
    library, aid, source, frames, pictures = source_frames
    root = library.asset_dir(aid)
    registry = read_json(root / "frames/registry.json")
    fid = frames[-1]["id"]
    del registry[fid]["sample"]
    atomic_json(root / "frames/registry.json", registry)
    with pytest.raises(FileNotFoundError, match="精确"):
        library.frame_path(aid, fid)
    atomic_json(root / "retrieval/plans/rp_old.json", {"id": "rp_old", "source_fingerprint": media.fingerprint(source),
                "frames": frames})
    assert library.frame_bytes(aid, fid) == pictures[fid]
    source.rename(source.with_suffix(".offline"))
    library.check_frame_entries(aid, list(pictures))
    with pytest.raises(FileNotFoundError):
        library.frame_bytes(aid, fid)
    regular = frames[0]["id"]
    pinned = root / "preserved-frames" / f"{regular}.jpg"
    pinned.parent.mkdir()
    pinned.write_bytes(pictures[regular])
    registry[regular].update(path=f"preserved-frames/{regular}.jpg", pinned_sha256=registry[regular]["sample"]["jpeg_sha256"])
    atomic_json(root / "frames/registry.json", registry)
    assert library.frame_bytes(aid, regular) == pictures[regular]
    saved = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime/backups" / f"{saved['id']}.zip")
    restored.check_frame_entries(aid, list(pictures))
    assert restored.frame_bytes(aid, regular) == pictures[regular]
    with pytest.raises(FileNotFoundError):
        restored.frame_bytes(aid, fid)
    pinned.write_bytes(b"changed human evidence")
    with pytest.raises(ValueError, match="人工证据"):
        library.frame_bytes(aid, regular)


def test_changed_source_or_corrupt_expected_jpeg_never_produces_picture(source_frames):
    library, aid, source, frames, _ = source_frames
    root = library.asset_dir(aid)
    registry = read_json(root / "frames/registry.json")
    fid = frames[0]["id"]
    registry[fid]["sample"]["jpeg_sha256"] = "0" * 64
    atomic_json(root / "frames/registry.json", registry)
    with pytest.raises(ValueError, match="SHA-256"):
        library.frame_bytes(aid, fid)
    assert library.sample_frames.bytes == 0 and not library.sample_frames.cache
    source.write_bytes(b"different media")
    atomic_json(library.root / "private/media-locations.json", {aid: source_snapshot(source)})
    with pytest.raises(ValueError, match="原视频内容"):
        sample_frames.verify_source(library, aid)


def test_frame_registry_snapshot_isolated_and_invalidated(tmp_path, monkeypatch):
    import scenerecall.library as library_module
    library = Library(tmp_path / "library")
    aid = "one"
    root = library.asset_dir(aid)
    file = root / "frames/registry.json"
    atomic_json(file, {"a": {"path": "frames/a.jpg", "at_ms": 0}})
    (root / "frames/a.jpg").write_bytes(b"legacy")
    calls = []
    original = library_module.read_json

    def read(path, *args):
        if path == file:
            calls.append(path)
        return original(path, *args)

    monkeypatch.setattr(library_module, "read_json", read)
    library.frame_entry(aid, "a")["path"] = "../wrong"
    assert library.frame_bytes(aid, "a") == b"legacy" and len(calls) == 1
    atomic_json(file, {"b": {"path": "frames/a.jpg", "at_ms": 30}})
    assert library.frame_entry(aid, "b")["at_ms"] == 30
    with pytest.raises(FileNotFoundError):
        library.frame_bytes(aid, "a")
    assert len(calls) == 2


def test_widely_separated_hits_do_not_decode_the_entire_intervening_episode(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.write_bytes(b"source")
    signature = media.fingerprint(source)
    metadata = {"_sample_identity": sample_frames.source_identity(source),
                "_sample_asset": "episode", "_sample_fingerprint": signature}
    batches, requests = [], []
    for pts in (1, 3, 1800, 1801):
        fid = "kf_" + digest(["episode", signature, pts, [1, 1]])
        frame = {"id": fid, "at_ms": pts * 1000, "pts": pts, "source_pts": pts,
                 "time_base": [1, 1], "source_time_base": [1, 1]}
        sample = sample_frames.make_recipe("episode", signature, frame, hashlib.sha256(b"fixture").hexdigest())
        requests.append((fid, {"at_ms": frame["at_ms"], "sample": sample}))

    def capture(_, frames, outputs, *, metadata):
        batches.append([frame["pts"] for frame in frames])
        for output in outputs:
            output.write_bytes(b"fixture")

    monkeypatch.setattr(keyframes, "extract_keyframes", capture)
    sample_frames.restore_batch(source, metadata, requests, tmp_path)
    assert batches == [[1, 3], [1800, 1801]]


def test_uniform_cropped_sample_reuses_requested_time_and_keeps_exact_jpeg(source_frames):
    library, aid, source, frames, _ = source_frames
    at, crop = frames[2]["at_ms"] - 1, [0, 0, .5, 1]
    fid = "f_" + digest([aid, at, crop])
    output = library.asset_dir(aid) / "frames" / (fid + ".jpg")
    [generated] = media.extract_frames(source, [at], [output], crop)
    expected = output.read_bytes()
    library.register_frame(aid, {**generated, "sample_crop": crop, "sample_generated": True})
    output.unlink()
    assert library.frame_bytes(aid, fid) == expected
    assert library.frame_entry(aid, fid)["sample"]["requested_at_ms"] == at


def test_api_portable_sample_backup_restores_every_picture_without_original_media(source_frames, tmp_path):
    from scenerecall.main import create_app
    library, aid, source, frames, pictures = source_frames
    first = frames[0]["id"]
    frame_dir = library.asset_dir(aid) / "frames"
    (frame_dir / (first + ".jpg")).write_bytes(pictures[first])
    # Legacy callers may re-register a retained file without the extraction
    # fields. That must not erase the existing source recipe.
    existing = library.frame_entry(aid, first)
    library.register_frame(aid, {"id": first, "at_ms": existing["at_ms"], "path": str(frame_dir / (first + ".jpg"))})
    assert library.frame_entry(aid, first) == existing
    client = TestClient(create_app(library.root, start_worker=False, token="backup-test"))
    headers = {"authorization": "Bearer backup-test"}

    def download(options):
        response = client.post("/api/backups", json=options, headers=headers)
        assert response.status_code == 200, response.text
        data = response.json()
        return data, client.get(data["download_url"], headers=headers).content

    compact, original = download({})
    assert compact["reconstructed_sampled_frames"] == 0
    with ZipFile(BytesIO(original)) as archive:
        assert sum(name.endswith(".jpg") for name in archive.namelist()) == 1
    portable, raw = download({"include_sampled_frames": True})
    assert portable["reconstructed_sampled_frames"] == len(frames) - 1 and not portable["includes_media"]
    with ZipFile(BytesIO(raw)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and not any(n.startswith("media/") for n in names)
        for identity, expected in pictures.items():
            assert archive.read(f"assets/{aid}/frames/{identity}.jpg") == expected
    assert list(frame_dir.glob("*.jpg")) == [frame_dir / (first + ".jpg")]

    source.rename(source.with_suffix(".offline"))
    offline = TestClient(create_app(tmp_path / "offline", start_worker=False, token="backup-test"))
    restored = offline.post("/api/restore", headers=headers, files={"file": ("portable.zip", raw, "application/zip")})
    assert restored.status_code == 200, restored.text
    for identity, expected in pictures.items():
        response = offline.get(f"/api/assets/{aid}/frames/{identity}", headers=headers)
        assert response.status_code == 200 and response.content == expected
    before = set((library.root / "runtime/backups").iterdir())
    failed = client.post("/api/backups", json={"include_sampled_frames": True}, headers=headers)
    assert failed.status_code != 200 and set((library.root / "runtime/backups").iterdir()) == before
