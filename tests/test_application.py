"""End-to-end contracts use synthetic media and an explicitly injected test provider."""
import subprocess
import shutil
import os
import zipfile

import pytest
from fastapi.testclient import TestClient

from scenerecall.jobs import JobQueue
from scenerecall.library import Library, read_json
from scenerecall.main import create_app
from scenerecall.models import AnalysisInput, AssetInput
from scenerecall.providers import AIResult, ProviderError
from scenerecall.search import SearchEngine


class TestProviders:
    __test__ = False

    def __init__(self):
        self.calls = []
        self.fail_vision = False
        self.after_vision = None

    def get(self, id):
        if id != "test":
            raise ValueError("unknown profile")
        return {"id": "test", "name": "Explicit test double", "model": "test-double", "base_url": "http://localhost/v1",
                "capabilities": ["vision", "subtitle", "embedding", "query", "decision", "answer"],
                "input_price_per_million": 1.0, "output_price_per_million": 1.0}

    def list(self):
        return [self.get("test")]

    async def analyze(self, id, frames, start_ms, end_ms, character_context=None):
        self.calls.append("vision")
        if self.fail_vision:
            raise ProviderError("synthetic failure")
        value = {"summary": "蓝衣人物拿起红色杯子", "entities": [
            {"id": "p1", "kind": "person", "description": "蓝色外套", "name": None, "position": "left"},
            {"id": "o1", "kind": "object", "description": "红色杯子", "name": None}],
            "events": [{"actor_ids": ["p1"], "action": "拿起", "target_ids": ["o1"], "start_ms": start_ms,
                        "end_ms": end_ms, "evidence_frame_ids": [frames[0]["id"]]}],
            "evidence_frame_ids": [f["id"] for f in frames], "uncertainties": ["测试替身，不是真实识别"]}
        if self.after_vision:
            self.after_vision()
        return AIResult(value, {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}, 1, .00002)

    async def recognize_subtitles(self, id, frames):
        self.calls.append("subtitle")
        return AIResult([{"frame_id": f["id"], "lines": [{"text": "这是合成测试字幕", "language": "zh", "uncertain": False},
                                                                {"text": "テスト字幕", "language": "ja", "uncertain": False}]} for f in frames], {}, 1, .00001)

    async def embed(self, id, texts):
        self.calls.append("embedding")
        return AIResult([[1.0, float("杯子" in t), .5] for t in texts], {}, 1, .00001)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_only")
    video = root / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=navy:s=160x90:r=10:d=2.4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)], check=True)
    subtitle = root / "synthetic.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n现在出发\n\n2\n00:00:01,200 --> 00:00:02,300\n前往月球\n", encoding="utf-8")
    return video, subtitle


def register(library, synthetic, mode="external"):
    video, sub = synthetic
    return library.register(AssetInput(video_path=str(video), title="明确标记的合成测试", subtitle_mode=mode,
                                       subtitle_path=str(sub) if mode == "external" else None))


def test_api_auth_subtitles_search_and_range(tmp_path, synthetic):
    app = create_app(tmp_path / "library", start_worker=False, providers=TestProviders(), token="test-token")
    with TestClient(app) as client:
        assert client.get("/api/assets").status_code == 401
        assert client.post("/api/session", json={"token": "wrong"}).status_code == 401
        assert client.post("/api/session", json={"token": "test-token"}, headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/api/session", json={"token": "test-token"}).status_code == 200
        assert client.post("/api/assets", json={"video_path": str(synthetic[0]), "subtitle_mode": "external"}).status_code == 422
        assert app.state.library.list_assets() == []
        imported = client.post("/api/assets", json={"video_path": str(synthetic[0]), "subtitle_mode": "external", "subtitle_path": str(synthetic[1]), "title": "合成测试"})
        assert imported.status_code == 200, imported.text
        asset = imported.json()
        response = client.post("/api/search", json={"query": "现在出发"})
        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["asset_id"] == asset["id"]
        assert response.json()["degraded"]
        result = response.json()["results"][0]
        assert client.post(f"/api/assets/{asset['id']}/annotations", json={"record_id": result["record_id"], "favorite": True, "note": "测试笔记"}).status_code == 200
        assert len(client.get("/api/collections").json()) == 1
        assert "测试笔记" in client.get("/api/export/citations").text
        media = client.get(f"/api/assets/{asset['id']}/media", headers={"Range": "bytes=0-15"})
        assert media.status_code == 206 and len(media.content) == 16
        assert client.get(f"/api/assets/{asset['id']}/frames/nonexistent").status_code == 404
        backup = client.post("/api/backups", json={"include_media": False}).json()
        assert client.get(backup["download_url"]).status_code == 200


@pytest.mark.asyncio
async def test_visual_ocr_index_and_annotations_survive(tmp_path, synthetic):
    library, providers = Library(tmp_path / "library"), TestProviders()
    engine = SearchEngine(library.root / "indexes", providers)
    queue = JobQueue(library, providers, engine)
    asset = register(library, synthetic, "embedded")
    request = AnalysisInput(asset_id=asset["id"], stages=["vision", "subtitle"], max_requests=30)
    _, config = queue.validate(request, {"bindings": {"vision": "test", "subtitle": "test", "embedding": "test"}})
    job = queue.add("analysis", config, asset["id"])
    await queue.execute(job["id"])
    final = queue.get(job["id"])
    assert final["status"] == "completed", final
    assert providers.calls == ["vision", "subtitle", "embedding"]
    assert final["request_count"] == 3
    observations = library.observations(asset["id"])
    assert len(observations) == 1
    cues = library.subtitles(asset["id"])
    assert {c["language"] for c in cues} == {"zh", "ja"}
    assert all(0 <= c["start_ms"] < c["end_ms"] <= asset["duration_ms"] for c in cues)
    record_id = observations[0]["id"]
    library.annotate(asset["id"], {"record_id": record_id, "summary": "人工修正：取杯", "favorite": True})
    library.annotate(asset["id"], {"record_id": record_id, "entity_id": "p1", "character_name": "小蓝", "aliases": ["蓝同学"]})
    await engine.rebuild(library.records(), "test")
    result = await engine.search("蓝同学", filters={"character": "小蓝"})
    assert result["results"][0]["record_id"] == record_id
    before = list(providers.calls)
    await engine.search("杯子", embedding_profile_id="test")
    assert providers.calls[len(before):] == ["embedding"]
    new_job = queue.add("analysis", config, asset["id"])
    await queue.execute(new_job["id"])
    assert providers.calls.count("vision") == 1 and providers.calls.count("subtitle") == 1
    assert library.records(asset["id"])[0]["summary"] == "人工修正：取杯"
    assert len(library.runs(asset["id"])) == 2


@pytest.mark.asyncio
async def test_pause_budget_resume_and_recovery(tmp_path, synthetic):
    library, providers = Library(tmp_path / "library"), TestProviders()
    engine = SearchEngine(library.root / "indexes", providers)
    queue = JobQueue(library, providers, engine)
    asset = register(library, synthetic)
    _, config = queue.validate(AnalysisInput(asset_id=asset["id"], max_requests=1), {"bindings": {"vision": "test"}})
    job = queue.add("analysis", config, asset["id"])
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "paused"
    assert providers.calls == []
    queue.control(job["id"], "resume", {"max_requests": 10})
    providers.after_vision = lambda: queue.control(job["id"], "pause")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "paused"
    assert len(library.observations(asset["id"])) == 1
    providers.after_vision = None
    queue.control(job["id"], "resume")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "completed"
    assert providers.calls == ["vision"]
    queue.update(job["id"], status="running")
    restarted = JobQueue(library, providers, engine)
    assert restarted.get(job["id"])["status"] == "queued"


def test_backup_relocation_and_schema(tmp_path, synthetic):
    library = Library(tmp_path / "original")
    asset = register(library, synthetic)
    first_cue = library.subtitles(asset["id"])[0]
    library.annotate(asset["id"], {"record_id": first_cue["id"], "favorite": True, "note": "保留笔记"})
    archive = library.backup(include_media=True)
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / (archive["id"] + ".zip"))
    assert restored.get_asset(asset["id"])["source_available"]
    assert restored.records()[0]["note"] == "保留笔记"
    changed = tmp_path / "wrong.mp4"
    changed.write_bytes(b"not same")
    with pytest.raises(ValueError, match="不一致"):
        library.relocate(asset["id"], str(changed))
    path = tmp_path / "new-schema.json"
    path.write_text('{"schema_version":"999"}')
    with pytest.raises(ValueError, match="版本"):
        read_json(path)


def test_changed_source_blocks_playback_until_verified(tmp_path, synthetic):
    source = tmp_path / "copy.mp4"
    shutil.copy2(synthetic[0], source)
    app = create_app(tmp_path / "library", start_worker=False, providers=TestProviders(), token="test-token")
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "test-token"})
        asset = client.post("/api/assets", json={"video_path": str(source), "subtitle_mode": "external",
                                                 "subtitle_path": str(synthetic[1])}).json()
        original = source.stat()
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns + 1000000))
        assert app.state.library.get_asset(asset["id"])["source_changed"]
        assert client.get(f"/api/assets/{asset['id']}/media").status_code == 400
        verified = client.post(f"/api/assets/{asset['id']}/relocate", json={"video_path": str(source)})
        assert verified.status_code == 200 and verified.json()["source_available"]
        source.write_bytes(b"another video")
        assert not app.state.library.get_asset(asset["id"])["source_available"]
        assert client.get(f"/api/assets/{asset['id']}/media").status_code == 400
        assert client.post(f"/api/assets/{asset['id']}/relocate", json={"video_path": str(source)}).status_code == 400
        with pytest.raises(ValueError, match="重新定位"):
            app.state.library.backup(include_media=True)


@pytest.mark.asyncio
async def test_queued_analysis_rechecks_source_before_model_call(tmp_path, synthetic):
    source = tmp_path / "copy.mp4"
    shutil.copy2(synthetic[0], source)
    library, providers = Library(tmp_path / "library"), TestProviders()
    asset = register(library, (source, synthetic[1]))
    queue = JobQueue(library, providers, SearchEngine(library.root / "indexes", providers))
    _, config = queue.validate(AnalysisInput(asset_id=asset["id"]), {"bindings": {"vision": "test"}})
    job = queue.add("analysis", config, asset["id"])
    source.write_bytes(b"replacement")
    await queue.execute(job["id"])
    assert queue.get(job["id"])["status"] == "failed"
    assert providers.calls == []


def test_restore_rejects_video_with_wrong_fingerprint(tmp_path, synthetic):
    library = Library(tmp_path / "original")
    register(library, synthetic)
    backup = library.backup(include_media=True)
    archive = library.root / "runtime" / "backups" / (backup["id"] + ".zip")
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(damaged, "w") as dst:
        for info in src.infolist():
            dst.writestr(info.filename, b"wrong film" if info.filename.startswith("media/") else src.read(info))
    restored = Library(tmp_path / "restored")
    with pytest.raises(ValueError, match="指纹"):
        restored.restore(damaged)
    assert restored.list_assets() == []


def test_localhost_origin_rejection(tmp_path):
    app = create_app(tmp_path, start_worker=False, providers=TestProviders())
    with TestClient(app) as client:
        assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 403
        assert client.post("/api/session", json={"token": app.state.session_token}, headers={"Origin": "http://localhost:9999"}).status_code == 403
