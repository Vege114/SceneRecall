"""Audio routes use existing authentication/index projection and local evidence."""
import base64
import asyncio
import threading

from fastapi.testclient import TestClient
from fastapi import FastAPI
import httpx
import pytest

from scenerecall.audio_media import crop_pcm
from scenerecall.audio_identity import validate_audio_artifacts
from scenerecall.audio_identity_api import register_audio_routes
from scenerecall.library import atomic_jsonl, read_jsonl
from scenerecall.main import create_app

from test_audio_identity import Characters, Diarizer, Embedder, Library, wav_file
from test_character_api import seed_appearance, seed_episode


def test_audio_api_auth_confirm_listen_and_revoke_sample(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, token="audio-test")
    seed_episode(app.state.library, "asset", 1)
    appearance = seed_appearance(app, "asset")
    character_id = appearance["entities"][0]["character_id"]
    audio = wav_file(tmp_path / "test.wav")
    monkeypatch.setattr("scenerecall.audio_identity.media.probe", lambda _: {
        "duration_ms": 4000, "streams": [{"codec_type": "audio", "index": 1, "tags": {"language": "ja"}}]})
    monkeypatch.setattr("scenerecall.audio_identity.extract_audio",
                        lambda source, out, stream, start, end, metadata, checkpoint=None: crop_pcm(audio, out, start, end))
    app.state.audio_identity._diarizer = Diarizer()
    app.state.audio_identity._embedder = Embedder()
    endpoint = "/api/assets/asset/audio"
    with TestClient(app) as client:
        assert client.get(endpoint + "/records").status_code == 401
        assert client.post("/api/session", json={"token": "audio-test"}).status_code == 200
        assert client.put(endpoint + "/settings", json={"enabled": True, "language": "ja",
                                                       "dub_version": "jp", "stream_index": 1}).status_code == 200
        tracks = client.get(endpoint + "/tracks").json()
        assert tracks["recommended_stream_index"] == 1
        result = client.post(endpoint + "/analyze", json={"start_ms": 0, "end_ms": 4000})
        assert result.status_code == 200, result.text
        row = result.json()["records"][0]
        assert row["association_status"] == "unknown"
        clip = client.get(row["audio_url"])
        assert clip.status_code == 200 and clip.content.startswith(b"RIFF")
        confirmed = client.post(endpoint + f"/records/{row['id']}/confirm", json={"character_id": character_id})
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["association_source"] == "human"
        projected = [item for item in app.state.library.records("asset") if item["kind"] == "audio"]
        assert projected[0]["character_id"] == character_id
        samples = f"/api/assets/asset/characters/{character_id}/voice-samples"
        added = client.post(samples, json={"record_id": row["id"], "language": "ja", "dub_version": "jp",
                                           "confirmed": True})
        assert added.status_code == 200, added.text
        sample = added.json()
        assert client.get(samples + f"/{sample['id']}/audio").content.startswith(b"RIFF")
        assert client.delete(samples + f"/{sample['id']}").json()["status"] == "revoked"
        assert client.get(samples).json()["samples"][0]["status"] == "revoked"
        # Explicit unknown overrides previous human assignment and refreshes projection.
        assert client.post(endpoint + f"/records/{row['id']}/confirm", json={"character_id": None}).json()["character_id"] is None
        assert client.post(endpoint + "/cancel").json()["status"] == "not_running"


def test_external_sample_rights_and_model_path_config_validation(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="audio-test")
    seed_episode(app.state.library, "asset", 1)
    record = seed_appearance(app, "asset")
    character_id = record["entities"][0]["character_id"]
    audio = wav_file(tmp_path / "test.wav")
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "audio-test"})
        endpoint = f"/api/assets/asset/characters/{character_id}/voice-samples"
        payload = {"source_kind": "external_licensed", "audio_base64": base64.b64encode(audio.read_bytes()).decode(),
                   "language": "ja", "dub_version": "jp", "confirmed": False}
        assert client.post(endpoint, json=payload).status_code == 400
        added = client.post(endpoint, json={**payload, "rights_note": "synthetic tone generated for test"})
        assert added.status_code == 200 and added.json()["status"] == "proposed"
        settings = client.get("/api/assets/asset/audio/settings").json()
        assert not settings["availability"]["diarization"]["available"]
        assert not settings["availability"]["sync"]["available"]


def test_restore_audio_graph_rejects_escape_and_preserves_merged_historical_samples(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, token="audio-test")
    seed_episode(app.state.library, "asset", 1)
    appearance = seed_appearance(app, "asset", second_person=True)
    first, second = [entity["character_id"] for entity in appearance["entities"]]
    source = wav_file(tmp_path / "test.wav")
    service = app.state.audio_identity
    service._diarizer, service._embedder = Diarizer(), Embedder()
    monkeypatch.setattr("scenerecall.audio_identity.media.probe", lambda _: {
        "duration_ms": 4000, "streams": [{"codec_type": "audio", "index": 1}]})
    monkeypatch.setattr("scenerecall.audio_identity.extract_audio",
                        lambda source_, out, stream, start, end, metadata, checkpoint=None: crop_pcm(source, out, start, end))
    service.configure("asset", {"enabled": True, "language": "ja", "dub_version": "jp"})
    result = service.run("asset")
    record = result["records"][0]
    service.confirm("asset", record["id"], first)
    service.add_sample("asset", first, {"record_id": record["id"], "confirmed": True})
    validate_audio_artifacts(app.state.library)
    app.state.characters.merge("asset", first, second)
    assert service.samples("asset", second)[0]["character_id"] == second
    assert service.record("asset", record["id"])["character_id"] == second
    validate_audio_artifacts(app.state.library)
    bank, _ = service._bank("asset")
    original = read_jsonl(bank)
    assert original[0]["audio_ref"].startswith("voice-audio/")
    # Existing Windows libraries may have stored native relative separators.
    atomic_jsonl(bank, [{**original[0], "audio_ref": original[0]["audio_ref"].replace("/", "\\")}])
    validate_audio_artifacts(app.state.library)
    atomic_jsonl(bank, [{**original[0], "audio_ref": "../../outside.wav"}])
    with pytest.raises(ValueError, match="文件引用"):
        validate_audio_artifacts(app.state.library)
    with pytest.raises(ValueError, match="越界"):
        service._sample_file("asset", {**original[0], "audio_ref": "../../outside.wav"})
    atomic_jsonl(bank, original)
    raw_path = service._directory("asset") / "runs" / result["run_id"] / "records.jsonl"
    raw = read_jsonl(raw_path)
    atomic_jsonl(raw_path, [{**raw[0], "audio_ref": str(tmp_path / "outside.wav")}, *raw[1:]])
    with pytest.raises(ValueError, match="文件引用"):
        validate_audio_artifacts(app.state.library)
    with pytest.raises(ValueError, match="越界"):
        service.clip_path("asset", raw[0]["id"])
    atomic_jsonl(raw_path, [{**raw[0], "end_ms": 9000}, *raw[1:]])
    with pytest.raises(ValueError, match="时间范围"):
        validate_audio_artifacts(app.state.library)
    atomic_jsonl(raw_path, raw)
    app.state.characters.delete("asset", second)
    assert service.samples("asset") == []
    with pytest.raises(ValueError, match="当前作品"):
        service.samples("asset", first)
    # Historical samples remain auditable and restorable after deletion.
    validate_audio_artifacts(app.state.library)


@pytest.mark.asyncio
async def test_cancelled_http_request_drains_owned_worker_before_releasing_slot(tmp_path):
    source = wav_file(tmp_path / "test.wav")
    app = FastAPI()
    service = register_audio_routes(app, Library(tmp_path / "library", source), Characters())
    started, release = threading.Event(), threading.Event()

    def run(*args, checkpoint=None, **kwargs):
        started.set()
        assert release.wait(timeout=3)
        checkpoint()
        return {"status": "completed", "records": []}

    service.run = run
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
        request = asyncio.create_task(client.post("/api/assets/asset/audio/analyze", json={"start_ms": 0, "end_ms": 1000}))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.005)
        assert started.is_set()
        request.cancel()
        await asyncio.sleep(.01)
        another = await client.post("/api/assets/asset/audio/analyze", json={"start_ms": 0, "end_ms": 1000})
        assert another.status_code == 409
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert (await client.post("/api/assets/asset/audio/cancel")).json()["status"] == "not_running"
