"""Independent human annotation contracts; no predictions become human labels."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import wave

import pytest
from fastapi.testclient import TestClient

from scenerecall.library import SCHEMA, atomic_json, read_json, source_snapshot

TOOL = Path(__file__).resolve().parents[1] / "tools" / "asd-annotator"
sys.path.insert(0, str(TOOL))
spec = importlib.util.spec_from_file_location("asd_standalone_test_server", TOOL / "server.py")
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)
store = sys.modules["annotation_store"]


def media_files(directory, frames=100):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "frames.rgb").open("wb") as handle:
        handle.truncate(frames * 480 * 270 * 3)
    with wave.open(str(directory / "audio.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(frames * 640 * 2))
    (directory / "preview.mp4").write_bytes(b"fixture-video")


@pytest.fixture
def service(tmp_path, monkeypatch):
    root = tmp_path / "library"
    source = tmp_path / "source.mkv"
    source.write_bytes(b"immutable-local-source")
    asset_id = "asset_kaguya"
    asset = {"schema_version": SCHEMA, "id": asset_id, "title": "辉夜姬", "work_id": "work_kaguya",
             "duration_ms": 1300000, "fingerprint": hashlib.sha256(source.read_bytes()).hexdigest()}
    atomic_json(root / "assets" / asset_id / "manifest.json", asset)
    atomic_json(root / "private" / "media-locations.json", {asset_id: source_snapshot(source)})
    library = store.ReadOnlyLibrary(root, asset_id)
    characters = types.SimpleNamespace(catalog=lambda _: {"profiles": [{"id": "character_a", "name": "甲", "references": []}]})
    audio = store.ReadOnlyAudio(library)
    instance = store.AsdAnnotationService(library, characters, audio, tmp_path / "manual-kaguya")
    monkeypatch.setattr(store.media, "probe", lambda _: {"duration_ms": 1300000, "streams": [
        {"codec_type": "audio", "index": 1, "tags": {"language": "jpn"}}]})
    def decode(source, directory, start, end, config):
        assert config["stream_index"] == 1
        media_files(directory, (end - start) // 40)
        return 1
    monkeypatch.setattr(instance.previews, "_decode", decode)
    monkeypatch.setattr(instance.previews, "_make_clip", lambda *args: None)
    return instance


def ready(service, start=1204000, end=1208000):
    initial = service.prepare("asset_kaguya", start, end)
    thread = service._running.get(("asset_kaguya", initial["id"]))
    if thread:
        thread.join(timeout=5)
        assert not thread.is_alive()
    result = service.get("asset_kaguya", initial["id"])
    assert result["status"] == "ready", result
    return result


def track(frames=100, **changes):
    return {"id": "track_human", "character_id": None, "boxes": [[.1, .1, .4, .7]] * frames,
            "labels": [-1] * frames, "reviewed": False, "excluded": False, "note": "", **changes}


def payload(clip, tracks):
    return {"revision": clip["revision"], "note": "explicit human edit", "tracks": tracks}


def test_manual_clip_without_models_is_playable_and_has_no_labels(service):
    before = {str(p): p.read_bytes() for p in service.library.root.rglob("*") if p.is_file()}
    clip = ready(service)
    assert clip["revision"] == 0 and clip["tracks"] == [] and clip["reviewer"] is None
    assert clip["audio_stream_index"] == 1 and "手动画框" in clip["reason"]
    assert clip["server"]["model_scores_copied"] is False
    assert service.media_path("asset_kaguya", clip["id"]).read_bytes() == b"fixture-video"
    assert service.list("asset_kaguya")["default_start_ms"] == 1204000
    assert before == {str(p): p.read_bytes() for p in service.library.root.rglob("*") if p.is_file()}
    assert service.prepare("asset_kaguya", 1204000, 1208000)["id"] == clip["id"]


def test_scene_plan_binds_source_ranges_and_scene_splits(service):
    clip = ready(service)
    plan = {"schema_version": "scenerecall-asd-scene-plan-v1", "asset_id": "asset_kaguya",
            "source_fingerprint": clip["source_fingerprint"], "items": [{"id": "task_01", "clip_id": clip["id"],
            "scene_id": "scene_a", "split": "train", "title": "双人场景候选", "start_ms": clip["start_ms"], "end_ms": clip["end_ms"]}]}
    atomic_json(service.data_dir / "annotation-plan.json", plan)
    result = service.list("asset_kaguya")["annotation_plan"]
    assert result["items"][0]["status"] == "ready"
    assert result["items"][0]["title"] == "双人场景候选"
    plan["items"][0]["end_ms"] += 40
    atomic_json(service.data_dir / "annotation-plan.json", plan)
    with pytest.raises(ValueError, match="范围"):
        service.list("asset_kaguya")
    plan["items"][0]["end_ms"] -= 40
    plan["source_fingerprint"] = "wrong-source"
    atomic_json(service.data_dir / "annotation-plan.json", plan)
    with pytest.raises(ValueError, match="不匹配"):
        service.list("asset_kaguya")


def test_scene_plan_rejects_same_scene_across_splits(service):
    clip = ready(service)
    task = {"id": "task_01", "clip_id": clip["id"], "scene_id": "scene_a", "split": "train",
            "start_ms": clip["start_ms"], "end_ms": clip["end_ms"]}
    plan = {"schema_version": "scenerecall-asd-scene-plan-v1", "asset_id": "asset_kaguya",
            "source_fingerprint": clip["source_fingerprint"], "items": [task, {**task, "id": "task_02", "split": "test"}]}
    atomic_json(service.data_dir / "annotation-plan.json", plan)
    with pytest.raises(ValueError, match="跨训练验证"):
        service.list("asset_kaguya")


def test_pcm_cache_keeps_presentation_sample_alignment_and_checks_hash(service, monkeypatch):
    clip = ready(service)
    cache = service.data_dir / "pcm-cache"
    cache.mkdir()
    pcm = cache / "japanese-presentation.wav"
    samples = 1300000 * 16
    with wave.open(str(pcm), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(bytes(samples * 2))
    with pcm.open("r+b") as stream:
        stream.seek(44 + clip["start_ms"] * 32)
        stream.write(b"\x34\x12" * (clip["frame_count"] * 640))
    identity = {"source_fingerprint": clip["source_fingerprint"], "source_stat": clip["source_stat"],
                "stream_index": 1, "duration_ms": 1300000, "origin_ms": 0, "preprocessing": store.PREPROCESS_VERSION,
                "relative_path": pcm.name, "sha256": store.file_sha256(pcm), "bytes": pcm.stat().st_size}
    atomic_json(cache / "identity.json", identity)
    output = service.data_dir / "cache-test"
    output.mkdir()
    def video(arguments, **kwargs):
        assert "-copyts" in arguments and "-seek_timestamp" in arguments
        with Path(arguments[-1]).open("wb") as stream:
            stream.truncate(clip["frame_count"] * 480 * 270 * 3)
    monkeypatch.setattr(store.media, "_run", video)
    metadata = {"duration_ms": 1300000, "video_stream_index": 0}
    assert service._decode_inputs("source.mkv", output, clip, {"stream_index": 1}, metadata) == 1
    with wave.open(str(output / "audio.wav")) as audio:
        assert audio.getnframes() == clip["frame_count"] * 640
        assert audio.readframes(audio.getnframes()) == b"\x34\x12" * (clip["frame_count"] * 640)
    identity["sha256"] = "0" * 64
    atomic_json(cache / "identity.json", identity)
    with pytest.raises(ValueError, match="缓存内容"):
        service._decode_inputs("source.mkv", output, clip, {"stream_index": 1}, metadata)


def test_stable_subclip_crops_only_unreviewed_detector_inputs(service, monkeypatch):
    parent = ready(service)
    parent_dir = service._root("asset_kaguya") / parent["id"]
    proposal = track(provenance={"box_source": "detector_proposal"})
    proposal["boxes"][0] = None
    parent["tracks"] = [proposal]
    with (parent_dir / "frames.rgb").open("r+b") as stream:
        for frame in range(parent["frame_count"]):
            stream.seek(frame * 480 * 270 * 3)
            stream.write(bytes([frame]) * 3)
    parent["server"]["files"]["frames_rgb"]["sha256"] = store.file_sha256(parent_dir / "frames.rgb")
    atomic_json(parent_dir / "clip.json", parent)
    original = (parent_dir / "clip.json").read_bytes()
    monkeypatch.setattr(service.previews, "_make_clip", lambda directory, *_: (directory / "preview.mp4").write_bytes(b"new-subclip"))
    child = ready(service, start=1204040, end=1206040)
    child_dir = service._root("asset_kaguya") / child["id"]
    assert child["frame_count"] == 50 and child["tracks"][0]["boxes"] == proposal["boxes"][1:51]
    assert child["tracks"][0]["labels"] == [-1] * 50
    assert child["tracks"][0]["character_id"] is None and not child["tracks"][0]["reviewed"]
    with (child_dir / "frames.rgb").open("rb") as stream:
        assert stream.read(3) == bytes([1]) * 3
        stream.seek(49 * 480 * 270 * 3)
        assert stream.read(3) == bytes([50]) * 3
    assert child["server"]["preparation"]["human_labels_copied"] is False
    assert child["server"]["preparation"]["parent_clip_id"] == parent["id"]
    assert (parent_dir / "clip.json").read_bytes() == original


def test_subclip_reuse_never_inherits_saved_human_work(service):
    parent = ready(service)
    parent_dir = service._root("asset_kaguya") / parent["id"]
    parent["revision"] = 1
    parent["tracks"] = [track(labels=[1] * 100, character_id="character_a", reviewed=True)]
    atomic_json(parent_dir / "clip.json", parent)
    child = ready(service, start=1204040, end=1206040)
    assert child["tracks"] == []
    assert child["server"]["preparation"]["kind"] == "local_decode_manual_boxes"


def test_superseded_proposals_are_hidden_but_any_human_work_stays_visible(service):
    clip = ready(service)
    atomic_json(service.data_dir / "hidden-proposals.json", {"asset_id": "asset_kaguya",
                "source_fingerprint": clip["source_fingerprint"], "clip_ids": [clip["id"]]})
    assert not service.list("asset_kaguya")["clips"]
    saved = service.save("asset_kaguya", clip["id"], payload(clip, [track()]))
    visible = service.list("asset_kaguya")["clips"]
    assert len(visible) == 1 and visible[0]["revision"] == saved["revision"]


def test_index_uses_one_source_snapshot_but_rechecks_after_reading(service, monkeypatch):
    ready(service)
    ready(service, 1208000, 1212000)
    original = service._source
    calls = []
    def source(asset_id):
        calls.append(asset_id)
        asset, location, stat = original(asset_id)
        return asset, location, {**stat, "mtime_ns": stat["mtime_ns"] + (len(calls) > 1)}
    monkeypatch.setattr(service, "_source", source)
    with pytest.raises(ValueError, match="原片已变化"):
        service.list("asset_kaguya")
    assert len(calls) == 2


def test_preview_reuse_only_copies_boxes_and_unknown_labels(service, monkeypatch):
    asset, _, stat = service._source("asset_kaguya")
    directory = service.library.asset_dir("asset_kaguya") / "audio/lrasd-previews/preview_fixture"
    media_files(directory)
    identity = {"schema_version": SCHEMA, "preview_version": "scenerecall-lrasd-preview-v3",
                "source_fingerprint": asset["fingerprint"], "source_stat": [stat["size"], stat["mtime_ns"]],
                "start_ms": 1204000, "end_ms": 1208000, "audio_stream_index": 1,
                "preprocessing": "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1", "detector_sha256": "a" * 64}
    atomic_json(directory / "identity.json", identity)
    atomic_json(directory / "result.json", {**identity, "status": "review_required", "tracks": [
        {"id": "track_cached", "character_id": "character_a", "active_speaker_probability": .99}]})
    raw = {"id": "track_cached", "character_id": "character_a", "labels": [1] * 100,
           "active_speaker_probability": .99, "boxes": [{"at_ms": 1204000 + index * 40, "box": [.1, .1, .4, .7]} for index in range(100)]}
    atomic_json(directory / "request.json", {"start_ms": 1204000, "end_ms": 1208000, "tracks": [raw]})
    monkeypatch.setattr(service.previews, "_decode", lambda *args: pytest.fail("valid preview must be reused"))
    monkeypatch.setattr(service.previews, "_worker", lambda *args, **kwargs: pytest.fail("no model inference on cache reuse"))
    clip = ready(service)
    result = clip["tracks"][0]
    assert result["boxes"][0] == [.1, .1, .4, .7]
    assert result["labels"] == [-1] * 100 and result["character_id"] is None
    assert result["reviewed"] is False and result["reviewer"] is None
    assert "active_speaker_probability" not in result
    assert clip["server"]["preparation"]["kind"] == "reused_preview_raw_detections"
    assert service._root("asset_kaguya") != directory.parent


def test_explicit_human_save_revision_audit_stats_and_source_guard(service):
    clip = ready(service)
    row = track(labels=[1, 0] + [-1] * 98, reviewed=True, character_id="character_a")
    saved = service.save("asset_kaguya", clip["id"], payload(clip, [row]))
    assert saved["revision"] == 1 and saved["reviewer"] == {"kind": "human"}
    assert saved["tracks"][0]["reviewed_at"] and saved["tracks"][0]["provenance"]["boxes_reviewed"] is True
    stats = service.list("asset_kaguya")["stats"]
    assert stats["reviewed_tracks"] == 1 and stats["labeled_frames"] == 2 and stats["positive_frames"] == 1
    with pytest.raises(store.AnnotationConflict):
        service.save("asset_kaguya", clip["id"], payload(clip, [row]))
    folder = service._root("asset_kaguya") / clip["id"]
    assert len(list((folder / "revisions").glob("*.json"))) == 2
    assert read_json(folder / "clip.json") == saved
    exported = service.export("asset_kaguya")
    assert exported["clips"][0]["media"]["frames_rgb"].startswith("clips/asd_")
    assert not Path(exported["clips"][0]["media"]["frames_rgb"]).is_absolute()
    manifest = service.library.asset_dir("asset_kaguya") / "manifest.json"
    asset = read_json(manifest)
    asset["fingerprint"] = "changed"
    atomic_json(manifest, asset)
    with pytest.raises(ValueError, match="指纹"):
        service.save("asset_kaguya", clip["id"], payload(saved, [row]))


@pytest.mark.parametrize("change", [
    {"labels": [True] + [-1] * 99}, {"boxes": [[float("nan"), .1, .4, .7]] * 100},
    {"boxes": [[.9, .1, .4, .7]] * 100}, {"labels": [1] * 99},
    {"labels": [1] * 100, "boxes": [None] * 100}, {"character_id": "other_work"},
    {"reviewed": 1}, {"boxes": [[True, .1, .4, .7]] * 100},
])
def test_invalid_frame_annotations_rejected(service, change):
    clip = ready(service)
    with pytest.raises(ValueError):
        service.save("asset_kaguya", clip["id"], payload(clip, [track(**change)]))
    assert service.get("asset_kaguya", clip["id"])["revision"] == 0


def test_strict_range_and_small_float_roundoff():
    assert store.checked_box([.1, .1, .9000000001, .8]) == [.1, .1, .9, .8]
    for start, end in [(True, 4000), (1, 4001), (0, 639), (0, 8040), (-40, 3960)]:
        with pytest.raises(ValueError):
            store.checked_range(start, end, 1300000)


def test_standalone_auth_routes_revision_media_and_export(service):
    clip = ready(service)
    app = server.create_app(service.library.root, "asset_kaguya", service.data_dir, service=service, token="asd-test")
    base = "/api/asd-annotations"
    with TestClient(app) as client:
        assert client.get("/api/health").json()["service"] == server.SERVICE
        assert "token" not in client.get("/api/health").json()
        assert client.get(base).status_code == 401
        assert client.get(base, headers={"Authorization": "Bearer asd-test"}).status_code == 200
        assert client.post("/api/session", json={"token": "bad"}).status_code == 401
        assert client.post("/api/session", json={"token": "asd-test"}).status_code == 200
        assert server.COOKIE in client.cookies and "scenerecall_session" not in client.cookies
        assert client.get("/api/config").json()["asset_id"] == "asset_kaguya"
        assert client.get("/api/characters").json()["profiles"][0]["id"] == "character_a"
        assert client.get(clip["clip_url"], headers={"Range": "bytes=0-3"}).content == b"fixt"
        assert client.get("/server.py").status_code == 404
        assert client.get(base, headers={"Origin": "http://evil.example:8766"}).status_code == 403
        assert client.get(base, headers={"Origin": "http://[broken"}).status_code == 403
        endpoint = base + "/clips/" + clip["id"]
        row = track(reviewed=True, labels=[0] * 100)
        assert client.put(endpoint, json=payload(clip, [row])).status_code == 200
        assert client.put(endpoint, json=payload(clip, [row])).status_code == 409
        assert client.put(endpoint, json={**payload(clip, [row]), "server": {}}).status_code == 422
        exported = client.get(base + "/export")
        assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
        assert exported.json()["clips"][0]["tracks"][0]["reviewed"] is True
