"""Preview contracts with synthetic detections; these are not accuracy tests."""
import hashlib
import threading
import types
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenerecall.av_sync_worker import lrasd_track_inputs
from scenerecall.library import SCHEMA, atomic_json, digest, read_json
from scenerecall.lrasd_preview import PREVIEW_VERSION, LrasdPreviewService, candidate_matches, validate_range
from scenerecall.lrasd_preview_worker import link_tracks
from scenerecall.main import create_app
from test_character_api import seed_episode


def detection(x=20, y=10, width=100, height=180):
    return {"box_xywh": [x, y, width, height], "score": .9}


def test_preview_bounds_strict_and_effective_grid():
    assert validate_range(123, 764, 10000) == (123, 763)
    assert validate_range(0, 8000, 8000) == (0, 8000)
    for values in ((True, 1000, 10000), (0, 639, 10000), (0, 8001, 10000),
                   (-1, 1000, 10000), (9500, 10200, 10000), (0., 1000, 10000)):
        with pytest.raises(ValueError):
            validate_range(*values)


def test_tracks_end_at_gap_and_cut_without_interpolation():
    rows = [[detection()] for _ in range(49)]
    rows[16] = []
    cuts = [False] * 49
    cuts[33] = True
    tracks = link_tracks(rows, cuts, 1000)
    assert [(row["start_ms"], row["end_ms"], row["detected_frames"]) for row in tracks] == [
        (1000, 1640, 16), (1680, 2320, 16), (2320, 2960, 16)]
    assert len({row["id"] for row in tracks}) == 3
    assert all(row["interpolated_frames"] == 0 and row["character_id"] is None for row in tracks)
    for row in tracks:
        assert [box["at_ms"] for box in row["boxes"]] == list(range(row["start_ms"], row["end_ms"], 40))


def test_ambiguous_link_is_not_carried_through_crossing():
    # Wide, nearly identical proposals cannot preserve a confident identity.
    rows = [[detection()] for _ in range(16)] + [[detection(x=19), detection(x=21)]]
    tracks = link_tracks(rows, [False] * 17, 0)
    assert len(tracks) == 1 and tracks[0]["end_ms"] == 640


def test_multitrack_order_and_hard_budget():
    rows = [[detection(x=index * 52, width=40, height=100) for index in range(9)] for _ in range(16)]
    tracks = link_tracks(rows, [False] * 16, 800)
    assert len(tracks) == 8
    assert len({row["id"] for row in tracks}) == 8
    assert all(row["start_ms"] == 800 and row["end_ms"] == 1440 for row in tracks)


def test_preview_subwindow_slices_audio_and_video_on_same_grid():
    request = {"start_ms": 1000, "end_ms": 3000, "preview_only": True}
    track = {"character_id": None, "start_ms": 1400, "end_ms": 2200,
             "boxes": [{"at_ms": at, "box": [0, 0, 1, 1]} for at in range(1400, 2200, 40)]}
    frames, features, start, end = lrasd_track_inputs(list(range(50)), list(range(200)), request, track)
    assert frames == list(range(10, 30)) and features == list(range(40, 120))
    assert (start, end) == (1400, 2200)
    with pytest.raises(ValueError, match="匿名"):
        lrasd_track_inputs(list(range(50)), list(range(200)), {**request, "preview_only": False}, track)
    with pytest.raises(ValueError, match="子轨迹"):
        lrasd_track_inputs(list(range(50)), list(range(200)), request, {**track, "start_ms": 1401})
    with pytest.raises(ValueError, match="逐帧"):
        lrasd_track_inputs(list(range(50)), list(range(200)), request, {**track, "boxes": track["boxes"][1:]})
    full = {"character_id": "known", "boxes": []}
    assert lrasd_track_inputs(list(range(50)), list(range(200)), {**request, "preview_only": False}, full)[0] == list(range(50))


def test_external_candidate_matching_never_becomes_confirmed(tmp_path):
    picture = tmp_path / "reference.png"
    picture.write_bytes(b"test-reference-content")
    reference = {"id": "reference_1", "character_id": "char_a", "path": str(picture), "status": "candidate",
                 "source": {"kind": "external_cached"}, "sha256": hashlib.sha256(picture.read_bytes()).hexdigest()}
    backend = types.SimpleNamespace(model_version="fixture-ccip", detect=lambda _: [{"box": [0, 0, 100, 100]}],
                                    distances=lambda query, features: [.1] * len(features))
    calls = []
    def feature(path, version, backend, metrics):
        calls.append(Path(path).name)
        return [1., 2.]
    pipeline = types.SimpleNamespace(backend=backend, library=types.SimpleNamespace(root=tmp_path), _feature=feature,
                                    settings=lambda _: {"model": "ccip-caformer-5_fp32", "top_k": 5},
                                    gallery=types.SimpleNamespace(list=lambda _: {"references": [reference]}))
    tracks, metrics, reason = candidate_matches(pipeline, "asset", [{"id": "t", "thumbnail_file": "t.png"}], tmp_path)
    row = tracks[0]
    assert row["character_id"] is None and row["association_status"] == "candidate"
    assert row["candidate_status"] == "unverified_reference_match"
    candidate = row["candidates"][0]
    assert candidate["character_id"] == "char_a" and candidate["reference_ids"] == ["reference_1"]
    assert candidate["source_verified"] is False
    assert candidate["reference_sources"][0] == {"id": "reference_1", "kind": "external_cached", "status": "candidate",
                                                "human_confirmed": False, "source_verified": False}
    assert calls == ["reference.png", "t.png"] and "matching_ms" in metrics


def test_missing_ccip_keeps_anonymous_preview(tmp_path):
    from scenerecall.visual_identity import IdentityUnavailable
    class Backend:
        @property
        def model_version(self):
            raise IdentityUnavailable("not installed")
    pipeline = types.SimpleNamespace(backend=Backend(), settings=lambda _: {"model": "ccip-caformer-5_fp32"},
                                    gallery=types.SimpleNamespace(list=lambda _: {"references": [{"status": "candidate"}]}))
    tracks, metrics, reason = candidate_matches(pipeline, "asset", [{"id": "t", "thumbnail_file": "t.png"}], tmp_path)
    assert tracks[0]["character_id"] is None and tracks[0]["candidates"] == []
    assert "不可用" in reason


def test_background_result_is_timed_anonymous_and_persisted(tmp_path, monkeypatch):
    directory = tmp_path / "assets/asset/audio/lrasd-previews/key"
    directory.mkdir(parents=True)
    library = types.SimpleNamespace(require_source=lambda _: tmp_path / "source.mkv", get_asset=lambda _: {},
                                    asset_dir=lambda _: tmp_path / "assets/asset", root=tmp_path)
    audio = types.SimpleNamespace(library=library, _directory=lambda _: tmp_path / "assets/asset/audio")
    service = LrasdPreviewService(audio)
    monkeypatch.setattr(service, "_decode", lambda *args: 1)
    monkeypatch.setattr(service, "_make_clip", lambda directory, *args: (directory / "preview.mp4").write_bytes(b"fixture-mp4"))
    monkeypatch.setattr("scenerecall.lrasd_preview.validate_lrasd_bundle",
                        lambda path: ({"checkpoint_sha256": "a" * 64}, path / "model.pt"))
    monkeypatch.setattr("scenerecall.lrasd_preview.file_sha256", lambda path: "b" * 64)
    tracks = link_tracks([[detection()] for _ in range(16)], [False] * 16, 1400)
    tracks[0]["thumbnail_file"] = tracks[0]["id"] + ".png"
    def worker(python, module, path, **kwargs):
        if module == "lrasd_preview_worker.py":
            return {"tracks": tracks, "timings": {"detection_tracking_ms": 8}, "tracking": {"gap_interpolation": False}, "cut_count": 0}
        request = read_json(path / "request.json")
        assert request["preview_only"] is True and request["tracks"][0]["character_id"] is None
        return {"status": "review_required", "model_version": "fixture", "results": [
            {"track_id": tracks[0]["id"], "active_speaker_probability": .75,
             "frame_probabilities": [{"at_ms": at, "probability": .75} for at in range(1400, 2040, 40)]}]}
    monkeypatch.setattr(service, "_worker", worker)
    monkeypatch.setattr("scenerecall.visual_identity.VisualIdentityPipeline", lambda _: object())
    monkeypatch.setattr("scenerecall.lrasd_preview.candidate_matches", lambda pipeline, asset, rows, path: (
        [{**row, "candidates": [], "association_status": "candidate"} for row in rows], {"matching_ms": 0}, "anonymous"))
    initial = {"job_id": "key", "start_ms": 1000, "end_ms": 3000, "status": "running", "cache_hit": False}
    config = {"sync_repository_path": str(tmp_path), "sync_checkpoint_path": str(tmp_path), "sync_python_path": "python", "device": "cpu"}
    service._background("asset", directory, initial, {"checkpoint_sha256": "a" * 64, "detector_sha256": "b" * 64},
                        config, "source.mkv", "person.onnx", "visual-python")
    result = service.latest("asset")
    assert result["status"] == "review_required"
    row = result["tracks"][0]
    assert row["start_ms"] == 1400 and row["end_ms"] == 2040
    assert row["character_id"] is None and row["active_speaker_probability"] == .75
    assert row["thumbnail_url"].startswith("/api/assets/asset/audio/lrasd-previews/key/")
    assert "boxes" not in row and "thumbnail_file" not in row
    assert read_json(directory / "result.json") == result
    assert result["clip_url"] == "/api/assets/asset/audio/lrasd-previews/key/clip"
    assert not (tmp_path / "works").exists()


def test_preview_http_auth_bounds_settings_and_latest(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, token="preview-test")
    seed_episode(app.state.library, "asset", 1)
    previews = app.state.audio_identity.lrasd_previews
    calls = []
    def start(asset, first, last):
        validate_range(first, last, 10000)
        calls.append((asset, first, last))
        return {"status": "running", "start_ms": first, "end_ms": last, "tracks": [], "job_id": "fixture"}
    monkeypatch.setattr(previews, "start", start)
    endpoint = "/api/assets/asset/audio/lrasd-preview"
    with TestClient(app) as client:
        assert client.get(endpoint).status_code == 401
        clip_endpoint = "/api/assets/asset/audio/lrasd-previews/fixture/clip"
        assert client.get(clip_endpoint).status_code == 401
        client.post("/api/session", json={"token": "preview-test"})
        assert client.get(endpoint).json()["status"] == "not_started"
        result = client.post(endpoint, json={"start_ms": 1000, "end_ms": 3000})
        assert result.status_code == 200 and result.json()["status"] == "running"
        assert calls == [("asset", 1000, 3000)]
        for payload in ({"start_ms": "0", "end_ms": 1000}, {"start_ms": True, "end_ms": 1000},
                        {"start_ms": 0, "end_ms": 1000, "tracks": []}):
            assert client.post(endpoint, json=payload).status_code == 422
        assert client.post(endpoint, json={"start_ms": 0, "end_ms": 600}).status_code == 400
        saved = client.put("/api/assets/asset/audio/settings", json={"sync_detector_path": "/models/person.onnx"})
        assert saved.status_code == 200 and saved.json()["settings"]["sync_detector_path"] == "/models/person.onnx"
        directory = previews._root("asset") / "fixture"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preview.mp4").write_bytes(b"fixture-mp4")
        atomic_json(directory / "result.json", {"schema_version": SCHEMA, "preview_version": PREVIEW_VERSION,
                                                "status": "running", "clip_url": clip_endpoint})
        assert client.get(clip_endpoint).status_code == 404
        atomic_json(directory / "result.json", {"schema_version": SCHEMA, "preview_version": PREVIEW_VERSION,
                                                "status": "no_tracks", "clip_url": clip_endpoint})
        clip = client.get(clip_endpoint)
        assert clip.status_code == 200 and clip.content == b"fixture-mp4"
        assert clip.headers["content-type"] == "video/mp4"
        ranged = client.get(clip_endpoint, headers={"Range": "bytes=0-3"})
        assert ranged.status_code == 206 and ranged.content == b"fixt"
        atomic_json(directory / "result.json", {"schema_version": SCHEMA, "preview_version": "scenerecall-lrasd-preview-v2",
                                                "status": "review_required", "clip_url": clip_endpoint})
        assert client.get(clip_endpoint).status_code == 404


def test_service_start_latest_cache_thumbnail_real_persistence_roundtrip(tmp_path, monkeypatch):
    detector, source = tmp_path / "person.onnx", tmp_path / "source.mkv"
    detector.write_bytes(b"detector-v1")
    source.write_bytes(b"local-source")
    config = {"enabled": True, "sync_enabled": True, "sync_model_type": "lrasd", "device": "cpu",
              "sync_checkpoint_path": str(tmp_path / "bundle"), "sync_detector_path": str(detector),
              "sync_repository_path": str(tmp_path / "repo"), "sync_python_path": "python"}
    library = types.SimpleNamespace(root=tmp_path, get_asset=lambda _: {"duration_ms": 10000, "fingerprint": "source-sha"},
                                    require_source=lambda _: source)
    audio = types.SimpleNamespace(library=library, settings=lambda _: config,
                                  _directory=lambda _: tmp_path / "asset/audio")
    service = LrasdPreviewService(audio)
    monkeypatch.setattr("scenerecall.lrasd_preview.sync_availability", lambda _: {"available": True})
    monkeypatch.setattr("scenerecall.lrasd_preview.validate_lrasd_bundle",
                        lambda path: ({"checkpoint_sha256": "a" * 64}, path / "model.pt"))
    pipeline = types.SimpleNamespace(gallery=types.SimpleNamespace(list=lambda _: {"references": []}),
                                    settings=lambda _: {"model": "fixture"})
    monkeypatch.setattr("scenerecall.visual_identity.VisualIdentityPipeline", lambda _: pipeline)
    calls = []
    release = threading.Event()
    def decode(*args):
        assert release.wait(timeout=3)
        return 1
    monkeypatch.setattr(service, "_decode", decode)
    monkeypatch.setattr(service, "_make_clip", lambda directory, *args: (directory / "preview.mp4").write_bytes(b"fixture-mp4"))
    def worker(python, module, directory, **kwargs):
        if module == "lrasd_preview_worker.py":
            calls.append(directory.name)
            tracks = link_tracks([[detection()] for _ in range(16)], [False] * 16, 1000)
            tracks[0]["thumbnail_file"] = tracks[0]["id"] + ".png"
            (directory / tracks[0]["thumbnail_file"]).write_bytes(b"fixture-png-content")
            return {"tracks": tracks, "timings": {}, "tracking": {}, "cut_count": 0}
        request = read_json(directory / "request.json")
        track = request["tracks"][0]
        return {"status": "review_required", "model_version": "fixture", "results": [
            {"track_id": track["id"], "active_speaker_probability": .75,
             "frame_probabilities": [{"at_ms": at, "probability": .75} for at in range(1000, 1640, 40)]}]}
    monkeypatch.setattr(service, "_worker", worker)
    monkeypatch.setattr("scenerecall.lrasd_preview.candidate_matches", lambda pipeline, asset, rows, path: (
        [{**row, "candidates": [], "association_status": "candidate"} for row in rows], {"matching_ms": 0}, "anonymous"))
    first = service.start("asset", 1000, 3000)
    thread = service._running["asset"][1]
    assert first["schema_version"] == SCHEMA and first["preview_version"] == PREVIEW_VERSION
    assert service.latest("asset")["status"] == "running"
    directory = service._root("asset") / first["job_id"]
    identity = read_json(directory / "identity.json")
    assert identity["schema_version"] == SCHEMA and identity["preview_version"] == PREVIEW_VERSION
    old_identity = {**identity, "schema_version": "scenerecall-lrasd-preview-v1"}
    old_identity.pop("preview_version")
    assert digest(old_identity) != first["job_id"]
    release.set()
    thread.join(timeout=3)
    assert not thread.is_alive()
    latest = service.latest("asset")
    assert latest["status"] == "review_required" and latest["schema_version"] == SCHEMA
    assert read_json(directory / "result.json") == latest
    track = latest["tracks"][0]
    assert service.thumbnail("asset", first["job_id"], track["id"]).read_bytes() == b"fixture-png-content"
    assert service.clip("asset", first["job_id"]).read_bytes() == b"fixture-mp4"
    with pytest.raises(FileNotFoundError):
        service.thumbnail("asset", first["job_id"], "not-a-track")
    cached = service.start("asset", 1000, 3000)
    assert cached["cache_hit"] is True and cached["job_id"] == first["job_id"]
    assert service.latest("asset")["cache_hit"] is True and len(calls) == 1
    detector.write_bytes(b"detector-v2")
    second = service.start("asset", 1000, 3000)
    thread = service._running.get("asset")
    if thread:
        thread[1].join(timeout=3)
    assert second["job_id"] != first["job_id"] and len(calls) == 2


def test_clip_encoding_uses_exact_aligned_inference_inputs(tmp_path, monkeypatch):
    service = LrasdPreviewService(types.SimpleNamespace(library=None))
    with (tmp_path / "frames.rgb").open("wb") as handle:
        handle.truncate(16 * 480 * 270 * 3)
    def audio(samples):
        with wave.open(str(tmp_path / "audio.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(bytes(samples * 2))
    audio(16 * 640)
    calls = []
    monkeypatch.setattr("scenerecall.lrasd_preview.media._tool", lambda _: "ffmpeg-fixture")
    def run(arguments, timeout):
        calls.append((arguments, timeout))
        Path(arguments[-1]).write_bytes(b"encoded-fixture")
    monkeypatch.setattr("scenerecall.lrasd_preview.media._run", run)
    service._make_clip(tmp_path, 1199000, 1199640)
    arguments, timeout = calls[0]
    assert timeout == 30
    assert [arguments[index + 1] for index, value in enumerate(arguments) if value == "-i"] == [
        str(tmp_path / "frames.rgb"), str(tmp_path / "audio.wav")]
    assert arguments[arguments.index("-framerate") + 1] == "25"
    assert arguments[arguments.index("-frames:v") + 1] == "16"
    assert arguments[arguments.index("-t") + 1] == "0.640000"
    assert arguments[arguments.index("-c:v") + 1] == "libx264"
    assert arguments[arguments.index("-pix_fmt") + 1] == "yuv420p"
    assert arguments[arguments.index("-c:a") + 1] == "aac"
    assert arguments[arguments.index("-movflags") + 1] == "+faststart"
    assert (tmp_path / "preview.mp4").read_bytes() == b"encoded-fixture"
    audio(16 * 640 - 1)
    with pytest.raises(ValueError, match="音频"):
        service._make_clip(tmp_path, 1199000, 1199640)
    assert len(calls) == 1
    audio(16 * 640)
    def fail(*args, **kwargs):
        raise ValueError("encoder failed")
    monkeypatch.setattr("scenerecall.lrasd_preview.media._run", fail)
    with pytest.raises(RuntimeError, match="短片编码失败"):
        service._make_clip(tmp_path, 1199000, 1199640)
    assert not (tmp_path / "preview.partial.mp4").exists()
