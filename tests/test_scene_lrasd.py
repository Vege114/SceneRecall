"""Formal-job ASD wiring with fake local workers, never films or model inference."""
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scenerecall.library import atomic_json, read_json
from scenerecall.scene_lrasd import context_reviews, list_reviews, receipt_path, run_reviews, select_windows
from test_scene_jobs import enqueue, scene_worker  # noqa: F401


def inputs(tmp_path):
    root = tmp_path / "asset"
    root.mkdir()
    library = SimpleNamespace(asset_dir=lambda _: root, get_asset=lambda _: {"fingerprint": "fixed-source"})
    settings = {"enabled": True, "sync_enabled": True, "sync_model_type": "lrasd", "device": "cpu"}
    job = {"id": "job", "asset_id": "asset", "run_id": "run",
           "config": {"start_ms": 0, "end_ms": 30000, "identity_settings": {"audio": settings}}}
    scenes = [{"id": f"scene_{i}", "start_ms": i * 10000, "end_ms": (i + 1) * 10000,
               "frame_ids": [f"frame_{i}"]} for i in range(3)]
    cues = [{"id": f"cue_{i}", "start_ms": i * 10000 + 4000, "end_ms": i * 10000 + 6000,
             "text": "Synthetic dialogue"} for i in range(3)]
    frames = [{"id": f"frame_{i}", "at_ms": i * 10000 + 5000} for i in range(3)]
    shots = [{"start_ms": scene["start_ms"], "end_ms": scene["end_ms"]} for scene in scenes]
    return library, job, scenes, cues, frames, shots


class Previews:
    def __init__(self):
        self.calls = []

    def run_for_job(self, asset, start, end, *, expected_settings, checkpoint):
        checkpoint()
        self.calls.append((asset, start, end))
        assert expected_settings["sync_model_type"] == "lrasd"
        return {"job_id": f"preview_{start}", "status": "review_required", "timings": {"asd_ms": 12},
                "checkpoint_sha256": "a" * 64, "model_version": "lrasd:" + "a" * 64,
                "tracks": [{"id": "track", "start_ms": start, "end_ms": end, "character_id": "must_clear",
                            "association_status": "confirmed", "active_speaker_probability": .7,
                            "candidates": [{"character_id": "proposal", "distance": .1,
                                            "association_status": "confirmed", "source_verified": True}]}]}


def test_selection_is_bounded_distributed_and_within_reliable_scene_shots(tmp_path):
    library, job, scenes, cues, frames, shots = inputs(tmp_path)
    original = deepcopy((scenes, cues, frames, shots))
    selected = select_windows(scenes, cues, frames, shots, 0, 30000)
    assert len(selected) == 3
    assert [row["scene_id"] for row in selected] == ["scene_0", "scene_1", "scene_2"]
    assert all(row["end_ms"] - row["start_ms"] == 2000 for row in selected)
    assert all(row["start_ms"] % 40 == row["end_ms"] % 40 == 0 for row in selected)
    assert [row["frame_ids"] for row in selected] == [["frame_0"], ["frame_1"], ["frame_2"]]
    scenes[0]["content_kind"] = "credits"
    cues[1]["uncertain"] = True
    shots[-1]["end_ms"] = shots[-1]["start_ms"] + 1000
    assert select_windows(scenes, cues, frames, shots, 0, 30000) == []
    assert original[0][0].get("content_kind") is None  # Input copies are ordinary source records.


def test_receipts_link_evidence_force_candidates_and_resume_without_model_repetition(tmp_path):
    library, job, scenes, cues, frames, shots = inputs(tmp_path)
    previews = Previews()
    records = [{"id": "voice", "start_ms": 14000, "end_ms": 16000}]
    result = run_reviews(library, job, scenes, cues, frames, shots, previews, audio_records=records)
    assert result["status"] == "completed" and len(previews.calls) == 3
    assert result["coverage"] == {"mode": "bounded_scene_sample", "max_windows": 3,
        "selected_windows": 3, "selected_duration_ms": 6000, "analysis_duration_ms": 30000,
        "full_episode_scored": False}
    review = result["reviews"][1]
    assert review["subtitle_ids"] == ["cue_1"] and review["frame_ids"] == ["frame_1"]
    assert review["audio_record_ids"] == ["voice"] and review["review_status"] == "candidate"
    assert review["tracks"][0]["character_id"] is None
    assert review["tracks"][0]["association_status"] == "candidate"
    assert review["tracks"][0]["candidates"][0]["source_verified"] is False
    assert read_json(receipt_path(library, "asset", "run")) == result
    repeated = run_reviews(library, job, scenes, cues, frames, shots, previews, audio_records=records)
    assert len(previews.calls) == 3 and all(row["cache_hit"] for row in repeated["reviews"])
    contextual = context_reviews(library, "asset", "run", 10000, 20000)
    assert len(contextual) == 1 and contextual[0]["scene_id"] == "scene_1"
    assert "frame_probabilities" not in contextual[0]["tracks"][0]
    assert list_reviews(library, "asset")["runs"][0]["run_id"] == "run"
    assert not (tmp_path / "works").exists()  # No voice bank or trusted identity publication.


def test_disabled_missing_dependencies_and_empty_stable_window_have_explicit_outcomes(tmp_path):
    library, job, scenes, cues, frames, shots = inputs(tmp_path)
    previews = Previews()
    job["config"]["identity_settings"]["audio"]["sync_enabled"] = False
    assert run_reviews(library, job, scenes, cues, frames, shots, previews) is None
    assert not receipt_path(library, "asset", "run").exists() and previews.calls == []
    job["config"]["identity_settings"]["audio"]["sync_enabled"] = True
    class Missing:
        def run_for_job(self, *args, **kwargs):
            raise ValueError("本机检测模型未安装")
    result = run_reviews(library, job, scenes, cues, frames, shots, Missing())
    assert result["status"] == "partial" and len(result["reviews"]) == 3
    assert all("未安装" in row for row in result["degraded"])
    assert not any(row["tracks"] for row in result["reviews"])
    job["run_id"] = "empty"
    result = run_reviews(library, job, scenes, [], frames, shots, previews)
    assert result["status"] == "unavailable" and result["coverage"]["selected_windows"] == 0
    assert previews.calls == []


def test_resume_refuses_changed_input_plan_and_keeps_prior_completed_samples_on_pause(tmp_path):
    library, job, scenes, cues, frames, shots = inputs(tmp_path)
    run_reviews(library, job, scenes, cues, frames, shots, Previews())
    calls = []
    class Paused(Exception):
        pass
    def checkpoint():
        calls.append(1)
        if len(calls) == 2:
            raise Paused()
    with pytest.raises(Paused):
        run_reviews(library, job, scenes, cues, frames, shots, Previews(), checkpoint=checkpoint)
    assert len(read_json(receipt_path(library, "asset", "run"))["reviews"]) == 3
    cues[0]["start_ms"] += 1000
    with pytest.raises(ValueError, match="计划或配置"):
        run_reviews(library, job, scenes, cues, frames, shots, Previews())


def test_scene_review_endpoint_is_authenticated_and_read_only(tmp_path):
    from scenerecall.main import create_app
    from test_character_api import seed_episode
    app = create_app(tmp_path / "library", start_worker=False, token="asd-fixture")
    seed_episode(app.state.library, "asset", 1)
    endpoint = "/api/assets/asset/audio/lrasd-scene-reviews"
    with TestClient(app) as client:
        assert client.get(endpoint).status_code == 401
        client.post("/api/session", json={"token": "asd-fixture"})
        assert client.get(endpoint).json() == {"status": "not_started", "runs": []}
        path = receipt_path(app.state.library, "asset", "run")
        atomic_json(path, {"version": "scene-lrasd-samples-v1", "run_id": "run", "status": "partial", "reviews": []})
        assert client.get(endpoint).json()["runs"][0]["run_id"] == "run"
        assert client.post(endpoint, json={}).status_code == 405


def test_owned_preview_worker_is_drained_before_queue_pause(tmp_path, monkeypatch):
    from scenerecall.lrasd_preview import LrasdPreviewService
    settings = {"enabled": True, "sync_enabled": True, "sync_model_type": "lrasd"}
    service = LrasdPreviewService(SimpleNamespace(library=None, settings=lambda _: settings))
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (finished.wait(.1), finished.set()))
    def start(*args):
        thread.start()
        service._running["asset"] = ("preview", thread)
        return {"status": "running", "job_id": "preview"}
    monkeypatch.setattr(service, "start", start)
    monkeypatch.setattr(service, "latest", lambda _: {"status": "running", "job_id": "preview"})
    class Paused(Exception):
        pass
    calls = []
    def checkpoint():
        calls.append(1)
        if len(calls) == 3:
            raise Paused()
    with pytest.raises(Paused):
        service.run_for_job("asset", 0, 2000, expected_settings=settings, checkpoint=checkpoint)
    assert finished.is_set() and not thread.is_alive()


@pytest.mark.asyncio
async def test_formal_scene_job_runs_continuous_samples_after_publication_and_exposes_deep_context(
        scene_worker, monkeypatch):  # noqa: F811
    from scenerecall.audio_identity import AudioIdentityService
    from scenerecall import scene_lrasd
    worker = scene_worker(12, with_subtitles=True)
    settings = AudioIdentityService(worker.library).configure(worker.asset["id"],
        {"enabled": True, "sync_enabled": True, "sync_model_type": "lrasd"})
    monkeypatch.setattr(AudioIdentityService, "run", lambda *args, **kwargs:
                        {"status": "disabled", "records": [], "degraded": ["Synthetic fixture: no acoustic model"]})
    original = scene_lrasd.select_windows
    # Fixture shots last one second. The production selector is covered above;
    # this integration fixture uses one continuous synthetic scene for the runner.
    monkeypatch.setattr(scene_lrasd, "select_windows", lambda chunks, cues, frames, shots, first, last:
                        original(chunks, cues, frames, [], first, last))
    previews = Previews()
    original_run = previews.run_for_job
    def run(asset, first, last, **kwargs):
        assert worker.store.list(asset), "Scene publication must precede the owned ASD sample"
        assert kwargs["expected_settings"] == settings
        return original_run(asset, first, last, **kwargs)
    previews.run_for_job = run
    worker.queue.lrasd_previews = previews
    job = enqueue(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done.get("error")
    assert len(previews.calls) == 1
    receipt = read_json(receipt_path(worker.library, worker.asset["id"], job["run_id"]))
    assert receipt["status"] == "completed" and done["lrasd_status"]["status"] == "completed"
    context = worker.queue.scene_audio_context(job["id"], {}, 0, 12000)
    assert "character_id" not in context["active_speaker_reviews"][0]["tracks"][0]
    assert context["active_speaker_reviews"][0]["tracks"][0]["association_status"] == "candidate"
    assert "never confirm" in context["active_speaker_review_policy"]
