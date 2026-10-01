"""Container ingestion uses generated media only and must never invoke a model."""
from __future__ import annotations

import json
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scenerecall.jobs import JobQueue
from scenerecall.library import Library, atomic_json, read_json, recommended_subtitle_track
from scenerecall.main import create_app
from scenerecall.models import AnalysisInput, AssetInput
from scenerecall.providers import AIResult
from scenerecall.search import SearchEngine


class NoModelCalls:
    def __init__(self):
        self.calls = []

    def list(self):
        return []

    def get(self, profile_id):
        return {"id": profile_id, "model": "synthetic", "base_url": "http://localhost.invalid",
                "capabilities": ["vision", "subtitle"], "provider_type": "openai_compatible"}

    async def recognize_subtitles(self, profile_id, frames):
        self.calls.append("subtitle")
        raise AssertionError("Container ingestion must not request OCR")

    async def embed(self, profile_id, texts):
        self.calls.append("embedding")
        raise AssertionError("Local extraction must not request embeddings")


@pytest.fixture(scope="module")
def container_movie(tmp_path_factory):
    directory = tmp_path_factory.mktemp("container-api-synthetic")
    first = directory / "forced.srt"
    second = directory / "default.srt"
    first.write_text("1\n00:00:00,300 --> 00:00:01,200\n另一轨独有句子\n", encoding="utf-8")
    second.write_text("1\n00:00:00,500 --> 00:00:01,500\n默认轨前往月球\n\n"
                      "2\n00:00:01,700 --> 00:00:02,600\n请带上蓝色杯子\n", encoding="utf-8")
    video = directory / "synthetic.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=navy:s=160x90:r=10:d=3", "-i", str(first), "-i", str(second),
                    "-map", "0:v", "-map", "1:s", "-map", "2:s", "-c:v", "libx264", "-bf", "0",
                    "-c:s", "srt", "-disposition:s:0", "forced", "-disposition:s:1", "default",
                    "-metadata:s:s:0", "language=zho", "-metadata:s:s:0", "title=Synthetic forced",
                    "-metadata:s:s:1", "language=zho", "-metadata:s:s:1", "title=Synthetic default",
                    str(video)], check=True)
    no_subtitles = directory / "without-subtitles.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video), "-map", "0:v",
                    "-c", "copy", str(no_subtitles)], check=True)
    return video, no_subtitles


@pytest.fixture
def local_app(tmp_path):
    providers = NoModelCalls()
    app = create_app(tmp_path / "library", start_worker=False, providers=providers, token="synthetic-token")
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer synthetic-token"
        yield app, client, providers


def test_auto_probe_import_search_and_track_switch_preserve_history(local_app, container_movie):
    app, client, providers = local_app
    video, _ = container_movie
    probe = client.post("/api/media/subtitle-tracks", json={"video_path": str(video)})
    assert probe.status_code == 200, probe.text
    assert probe.json()["recommended_stream_index"] == 2
    assert [track["index"] for track in probe.json()["tracks"]] == [1, 2]
    assert probe.json()["duration_ms"] == 3000
    response = client.post("/api/assets", json={"video_path": str(video)})
    assert response.status_code == 200, response.text
    asset = response.json()
    assert asset["subtitle_mode"] == "container" and asset["subtitle_import_mode"] == "auto"
    assert asset["subtitle_stream_index"] == 2 and asset["subtitle_count"] == 2
    assert asset["subtitle_track"]["title"] == "Synthetic default"
    asset_id = asset["id"]
    assert client.get(f"/api/assets/{asset_id}/subtitle-tracks").json() == probe.json()
    found = client.post("/api/search", json={"query": "默认轨前往月球"}).json()["results"]
    assert found and found[0]["source"] == "container" and found[0]["source_stream_index"] == 2
    assert (found[0]["start_ms"], found[0]["end_ms"]) == (500, 1500)
    cue_id = found[0]["record_id"]
    assert client.post(f"/api/assets/{asset_id}/annotations", json={
        "record_id": cue_id, "favorite": True, "note": "保存人工笔记", "summary": "人工核对过的原文"}).status_code == 200
    first_pointer = app.state.library.active(asset_id)["container_subtitles"]
    first_bytes = (app.state.library.asset_dir(asset_id) / first_pointer["path"]).read_bytes()
    switched = client.post(f"/api/assets/{asset_id}/subtitles/extract", json={"subtitle_stream_index": 1})
    assert switched.status_code == 200, switched.text
    assert switched.json()["subtitle_count"] == 1 and switched.json()["subtitle_stream_index"] == 1
    assert client.post("/api/search", json={"query": "默认轨前往月球"}).json()["results"] == []
    assert client.post("/api/search", json={"query": "另一轨独有句子"}).json()["results"]
    historical = client.get("/api/collections").json()
    assert len(historical) == 1 and historical[0]["is_historical"]
    assert historical[0]["note"] == "保存人工笔记" and historical[0]["text"] == "人工核对过的原文"
    assert (app.state.library.asset_dir(asset_id) / first_pointer["path"]).read_bytes() == first_bytes
    assert client.post(f"/api/assets/{asset_id}/subtitles/extract", json={"subtitle_stream_index": 2}).status_code == 200
    current = client.get("/api/collections").json()
    assert len(current) == 1 and current[0]["record_id"] == cue_id and not current[0].get("is_historical")
    assert current[0]["note"] == "保存人工笔记"
    assert len(app.state.library.runs(asset_id)) == 3
    assert providers.calls == []


def test_auto_without_tracks_defers_ocr_and_container_requires_tracks(local_app, container_movie):
    _, client, providers = local_app
    _, video = container_movie
    explicit = client.post("/api/assets", json={"video_path": str(video), "subtitle_mode": "container"})
    assert explicit.status_code == 400 and "没有字幕轨" in explicit.json()["detail"]
    response = client.post("/api/assets", json={"video_path": str(video)})
    assert response.status_code == 200, response.text
    asset = response.json()
    assert asset["subtitle_mode"] == "embedded" and asset["subtitle_count"] == 0
    assert "未检测到" in asset["subtitle_fallback_reason"]
    estimate = client.post("/api/jobs/estimate", json={"asset_id": asset["id"], "stages": ["subtitle"]})
    assert estimate.status_code == 400 and "绑定模型" in estimate.json()["detail"]
    assert providers.calls == []


@pytest.mark.parametrize("payload", [
    {"subtitle_stream_index": True}, {"subtitle_stream_index": -1}, {"subtitle_stream_index": "1"},
    {"subtitle_offset_ms": 0.5}, {"subtitle_mode": "embedded", "subtitle_stream_index": 1},
    {"subtitle_mode": "external", "subtitle_path": "a.srt", "subtitle_stream_index": 1},
    {"subtitle_mode": "container", "subtitle_path": "a.srt"},
    {"subtitle_mode": "embedded", "subtitle_offset_ms": 1},
])
def test_asset_subtitle_option_relationships(payload):
    with pytest.raises(ValidationError):
        AssetInput(video_path="synthetic.mkv", **payload)


def test_invalid_selection_or_extraction_keeps_current_track(local_app, container_movie, monkeypatch):
    app, client, _ = local_app
    video, _ = container_movie
    response = client.post("/api/assets", json={"video_path": str(video), "subtitle_stream_index": 0})
    assert response.status_code == 400 and app.state.library.list_assets() == []
    asset = client.post("/api/assets", json={"video_path": str(video), "subtitle_stream_index": 1}).json()
    before = app.state.library.active(asset["id"])
    for index in (0, 99):
        response = client.post(f"/api/assets/{asset['id']}/subtitles/extract", json={"subtitle_stream_index": index})
        assert response.status_code == 400
    for index in (True, "1", -1):
        response = client.post(f"/api/assets/{asset['id']}/subtitles/extract", json={"subtitle_stream_index": index})
        assert response.status_code == 422
    def broken(*args, **kwargs):
        raise ValueError("synthetic extraction failure")
    monkeypatch.setattr("scenerecall.media.extract_subtitle_track", broken)
    response = client.post(f"/api/assets/{asset['id']}/subtitles/extract", json={"subtitle_stream_index": 2})
    assert response.status_code == 400
    assert app.state.library.active(asset["id"]) == before
    assert app.state.library.subtitles(asset["id"])[0]["text"] == "另一轨独有句子"


def test_bitmap_tracks_fail_explicitly_instead_of_silent_ocr(local_app, tmp_path, monkeypatch):
    app, client, _ = local_app
    video = tmp_path / "synthetic-metadata-only.mkv"
    video.write_bytes(b"not decoded; injected bitmap metadata")
    monkeypatch.setattr("scenerecall.media.probe", lambda source: {
        "duration_ms": 3000, "width": 16, "height": 16, "streams": [
            {"index": 1, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle"}]})
    for mode in ("auto", "container"):
        response = client.post("/api/assets", json={"video_path": str(video), "subtitle_mode": mode})
        assert response.status_code == 400 and "位图" in response.json()["detail"]
    assert app.state.library.list_assets() == []


def test_default_selection_order():
    tracks = [{"index": 1, "default": False, "forced": False, "supported": True},
              {"index": 2, "default": True, "forced": True, "supported": True},
              {"index": 3, "default": True, "forced": False, "supported": True},
              {"index": 0, "default": True, "forced": False, "supported": False}]
    assert recommended_subtitle_track(tracks)["index"] == 3
    assert recommended_subtitle_track(tracks[:2])["index"] == 2
    assert recommended_subtitle_track([]) is None


def test_container_metadata_backup_and_offset_roundtrip(local_app, container_movie, tmp_path):
    app, client, providers = local_app
    video, _ = container_movie
    response = client.post("/api/assets", json={"video_path": str(video), "subtitle_mode": "container",
                                               "subtitle_stream_index": 1, "subtitle_offset_ms": 100})
    assert response.status_code == 200, response.text
    asset = response.json()
    cue = app.state.library.subtitles(asset["id"])[0]
    assert (cue["start_ms"], cue["end_ms"], cue["offset_ms"]) == (400, 1300, 100)
    app.state.library.annotate(asset["id"], {"record_id": cue["id"], "favorite": True, "note": "保留"})
    archive = app.state.library.backup(False)
    path = app.state.library.root / "runtime" / "backups" / f"{archive['id']}.zip"
    with zipfile.ZipFile(path) as source:
        assert not any(name.startswith(("media/", "private/", "indexes/")) for name in source.namelist())
        assert any(name.endswith(".srt") for name in source.namelist())
    restored = Library(tmp_path / "restored")
    restored.restore(path)
    assert restored.subtitles(asset["id"]) == app.state.library.subtitles(asset["id"])
    assert not restored.get_asset(asset["id"])["source_available"]
    assert restored.get_asset(asset["id"])["subtitle_stream_index"] == 1
    assert restored.collections()[0]["note"] == "保留"
    assert providers.calls == []


@pytest.mark.parametrize("corruption", ["path", "original_path", "stream_index", "missing_original", "retired_ocr"])
def test_restore_rejects_broken_container_reference_before_merging(local_app, container_movie, tmp_path, corruption):
    app, client, _ = local_app
    asset = client.post("/api/assets", json={"video_path": str(container_movie[0])}).json()
    directory = app.state.library.asset_dir(asset["id"])
    active = read_json(directory / "active-analysis.json")
    pointer = active["container_subtitles"]
    if corruption in {"path", "original_path"}:
        pointer[corruption] = "../../private/escaped.json"
    elif corruption == "stream_index":
        pointer["stream_index"] = 99
    elif corruption == "missing_original":
        (directory / pointer["original_path"]).unlink()
    else:
        active["subtitle_runs"] = {"old": {"path": "../../private/escaped.json"}}
    atomic_json(directory / "active-analysis.json", active)
    # Build the archive directly because the intentionally damaged live source
    # should also refuse reads of its invalid references.
    archive = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("library.json", json.dumps({"schema_version": "1.0", "id": "synthetic"}))
        for path in directory.rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(app.state.library.root))
    target = Library(tmp_path / "target")
    with pytest.raises(ValueError):
        target.restore(archive)
    assert target.list_assets() == []


@pytest.mark.asyncio
async def test_legacy_ocr_job_stops_before_requests_after_extraction(container_movie, tmp_path):
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(container_movie[0]), subtitle_mode="embedded"))
    library.save_subtitles(asset["id"], "run_old_ocr", [{"start_ms": 0, "end_ms": 500,
        "text": "历史 OCR", "language": "zh", "evidence_frame_ids": [], "review_status": "unreviewed"}], 0, 3000)
    old_cue = library.subtitles(asset["id"])[0]
    library.annotate(asset["id"], {"record_id": old_cue["id"], "favorite": True, "note": "OCR历史"})
    providers = NoModelCalls()
    queue = JobQueue(library, providers, SearchEngine(library.root / "indexes", providers))
    request = AnalysisInput(asset_id=asset["id"], stages=["subtitle"])
    _, config = queue.validate(request, {"bindings": {"subtitle": "synthetic"}})
    job = queue.add("analysis", config, asset["id"])
    library.extract_subtitles(asset["id"])
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "paused" and providers.calls == []
    assert {cue["source"] for cue in library.subtitles(asset["id"])} == {"container"}
    assert library.collections()[0]["is_historical"] and library.collections()[0]["note"] == "OCR历史"
    with pytest.raises(ValueError, match="只包含画面"):
        queue.control(job["id"], "resume")
    with pytest.raises(ValueError, match="无需调用"):
        queue.validate(request, {"bindings": {}})
    with pytest.raises(ValueError, match="旧画面 OCR"):
        library.save_subtitles(asset["id"], "run_stale", [], 0, 3000)
    assert library.active(asset["id"])["subtitle_runs"]["run_old_ocr"]


@pytest.mark.asyncio
async def test_running_ocr_result_cannot_overwrite_concurrent_track_switch(container_movie, tmp_path):
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(container_movie[0]), subtitle_mode="embedded"))

    class SwitchWhileRequestRuns(NoModelCalls):
        async def recognize_subtitles(self, profile_id, frames):
            self.calls.append("subtitle")
            library.extract_subtitles(asset["id"])
            return AIResult([{"frame_id": frame["id"], "lines": [
                {"text": "过期 OCR 结果", "language": "zh", "uncertain": False}]} for frame in frames], {}, 1, 0)

    providers = SwitchWhileRequestRuns()
    queue = JobQueue(library, providers, SearchEngine(library.root / "indexes", providers))
    request = AnalysisInput(asset_id=asset["id"], stages=["subtitle"])
    _, config = queue.validate(request, {"bindings": {"subtitle": "synthetic"}})
    job = queue.add("analysis", config, asset["id"])
    await queue.execute(job["id"])
    assert providers.calls == ["subtitle"] and queue.get(job["id"])["status"] == "paused"
    assert queue.get(job["id"])["request_count"] == 1
    assert {cue["source"] for cue in library.subtitles(asset["id"])} == {"container"}
    assert all("过期 OCR" not in cue["text"] for cue in library.subtitles(asset["id"]))


def test_container_import_storage_failure_does_not_publish_partial_asset(container_movie, tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    def broken(*args, **kwargs):
        raise OSError("synthetic full disk")
    monkeypatch.setattr("scenerecall.library.atomic_jsonl", broken)
    with pytest.raises(OSError, match="full disk"):
        library.register(AssetInput(video_path=str(container_movie[0])))
    assert library.list_assets() == []
    assert list((library.root / "assets").iterdir()) == []
    assert list((library.root / "works").iterdir()) == []
    assert list((library.root / "runtime").iterdir()) == []


def test_probe_requires_session_and_rejects_extra_fields(local_app, container_movie):
    _, client, _ = local_app
    client.headers.pop("Authorization")
    assert client.post("/api/media/subtitle-tracks", json={"video_path": str(container_movie[0])}).status_code == 401
    client.headers["Authorization"] = "Bearer synthetic-token"
    assert client.post("/api/media/subtitle-tracks", json={"video_path": str(container_movie[0]),
                                                          "unexpected": True}).status_code == 422
