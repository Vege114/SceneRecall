"""Preflight confirmation invariants with synthetic local images/models."""
from copy import deepcopy
from pathlib import Path

from fastapi import FastAPI
import httpx
import pytest

from scenerecall.character_confirmation import CharacterConfirmation
from scenerecall.library import atomic_json, read_json
from scenerecall.visual_identity import VisualIdentityPipeline
from scenerecall.visual_identity_api import register_visual_identity_routes
from test_visual_identity import (FakeCCIP, add_reference, add_role, frame, picture, visual as visual)


def prepared(visual, times=(200, 6000, 12000, 18000)):
    library, characters, gallery = visual
    identity = add_role(characters)
    add_reference(gallery, identity)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    frames = [frame(library, f"frame_{index}", at_ms=at_ms) for index, at_ms in enumerate(times)]
    pipeline.run("asset_one", frames, {"enabled": True})
    service = CharacterConfirmation(library, pipeline)
    return service, identity


def candidate_args(candidate):
    return {"source_asset_id": candidate["asset_id"], "region_id": candidate["region_id"]}


def confirm(service, identity, candidate):
    service.decide("asset_one", identity, action="confirm", single_character=True, use_as_reference=True,
                   **candidate_args(candidate))


def test_queue_reuses_work_regions_diversifies_stored_shots_without_inference(visual, monkeypatch):
    service, identity = prepared(visual)
    library = service.library
    asset = library.get_asset("asset_one")
    second = {**asset, "id": "asset_two", "episode": 2, "title": "Episode Two"}
    atomic_json(library.asset_dir("asset_two") / "manifest.json", second)
    path = library.asset_dir("asset_two") / "frames" / "frame_other.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(picture())
    other_frame = library.register_frame("asset_two", {"id": "frame_other", "path": str(path), "at_ms": 200})
    service.pipeline.run("asset_two", [other_frame], {"enabled": True})
    atomic_json(library.asset_dir("asset_one") / "keyframe-plans" / "plan.json", {
        "fingerprint": asset["fingerprint"], "frames": [{"id": f"frame_{index}", "shot_id": "same_shot"} for index in range(4)]})
    monkeypatch.setattr(service.pipeline.backend, "detect", lambda *args: pytest.fail("queue must not invoke detection"))
    monkeypatch.setattr(service.pipeline.backend, "feature", lambda *args: pytest.fail("queue must not invoke CCIP"))
    queue = service.queue("asset_one")
    role = queue["roles"][0]
    assert role["character_id"] == identity
    assert role["available_count"] == 5
    assert len(role["candidates"]) == 2
    assert {row["asset_id"] for row in role["candidates"]} == {"asset_one", "asset_two"}
    assert all("crop_path" not in row for row in role["candidates"])


def test_confirm_is_explicit_idempotent_and_does_not_propagate(visual):
    service, identity = prepared(visual)
    original = service.queue("asset_one")["roles"][0]["candidates"]
    chosen = original[0]
    with pytest.raises(ValueError, match="同意"):
        service.decide("asset_one", identity, action="confirm", **candidate_args(chosen))
    before_refs = len(service.gallery.list("asset_one")["references"])
    calls = []
    service.search = type("Search", (), {"sync_asset": lambda _self, rows, asset: calls.append((rows, asset))})()
    confirm(service, identity, chosen)
    confirmed = service.queue("asset_one")
    assert confirmed["summary"]["confirmed"] == 1
    assert confirmed["roles"][0]["trusted_reference_count"] == 1
    reference = service.gallery.trusted("asset_one")[-1]
    assert reference["source"]["kind"] == "frame_crop"
    assert reference["source"]["asset_id"] == "asset_one"
    assert reference["source"]["region_id"] == chosen["region_id"]
    assert reference["source"]["source_fingerprint"] == "synthetic-video"
    assert calls[-1] == ([], "asset_one")  # Preflight never fabricates searchable observations.
    reviews = service.pipeline.reviews("asset_one")["regions"]
    assert sum(row["match"]["status"] == "confirmed" for row in reviews) == 1
    review_revision = read_json(service.pipeline._directory("asset_one") / "decisions.json")["revision"]
    confirm(service, identity, chosen)
    assert len(service.gallery.list("asset_one")["references"]) == before_refs + 1
    assert read_json(service.pipeline._directory("asset_one") / "decisions.json")["revision"] == review_revision


def test_reject_unclear_skip_survive_restart_without_altering_identity(visual):
    service, identity = prepared(visual)
    original = service.queue("asset_one")["roles"][0]["candidates"]
    service.decide("asset_one", identity, action="reject", **candidate_args(original[0]))
    service.decide("asset_one", identity, action="unclear", **candidate_args(original[1]))
    service.decide("asset_one", identity, action="skip")
    fresh = CharacterConfirmation(service.library)
    queue = fresh.queue("asset_one")
    assert queue["summary"]["skipped"] == 1
    assert queue["roles"][0]["available_count"] == 2
    assert all(row["region_id"] not in {original[0]["region_id"], original[1]["region_id"]}
               for row in queue["roles"][0]["candidates"])
    assert not (service.pipeline._directory("asset_one") / "decisions.json").exists()
    fresh.decide("asset_one", identity, action="resume")
    assert fresh.queue("asset_one")["summary"]["pending"] == 1


def test_confirmation_rejects_scope_conflicts_and_changed_pixels(visual):
    service, identity = prepared(visual)
    chosen = service.queue("asset_one")["roles"][0]["candidates"][0]
    other_role = add_role(visual[1], "Other")
    service.pipeline.review("asset_one", chosen["region_id"], status="confirmed", character_id=other_role, reason="human")
    with pytest.raises(ValueError, match="另一角色"):
        confirm(service, identity, chosen)
    service.pipeline.review("asset_one", chosen["region_id"], status="unknown", reason="reset")
    service.library.frame_path("asset_one", chosen["frame_id"]).write_bytes(picture(190))
    with pytest.raises(ValueError, match="已变化"):
        confirm(service, identity, chosen)
    alien = {**service.library.get_asset("asset_one"), "id": "asset_alien", "series": "Different", "work_id": "work_alien"}
    alien.pop("character_scope_id", None)
    alien.pop("project_id", None)
    atomic_json(service.library.asset_dir("asset_alien") / "manifest.json", alien)
    with pytest.raises(ValueError, match="同一作品"):
        service.decide("asset_one", identity, action="reject", source_asset_id="asset_alien", region_id=chosen["region_id"])


def test_failed_search_projection_retains_recoverable_candidate_and_reuses_reference(visual):
    service, identity = prepared(visual)
    chosen = service.queue("asset_one")["roles"][0]["candidates"][0]
    class BrokenSearch:
        def sync_asset(self, *args):
            raise RuntimeError("search interrupted")
    service.search = BrokenSearch()
    with pytest.raises(RuntimeError, match="interrupted"):
        confirm(service, identity, chosen)
    queue = service.queue("asset_one")
    assert queue["roles"][0]["status"] == "pending"
    assert any(row["recovery_required"] for row in queue["roles"][0]["candidates"])
    assert len(service.gallery.trusted("asset_one")) == 2
    service.search = None
    confirm(service, identity, chosen)
    assert service.queue("asset_one")["summary"]["confirmed"] == 1
    assert len(service.gallery.trusted("asset_one")) == 2


def test_retry_never_overwrites_later_manual_correction(visual):
    service, identity = prepared(visual)
    chosen = service.queue("asset_one")["roles"][0]["candidates"][0]
    confirm(service, identity, chosen)
    service.pipeline.review("asset_one", chosen["region_id"], status="unknown", reason="later human correction")
    with pytest.raises(ValueError, match="人工修改"):
        confirm(service, identity, chosen)
    assert service.pipeline.reviews("asset_one")["regions"][0]["match"].get("character_id") is None


def test_normal_review_unknown_and_revoked_reference_do_not_reappear(visual):
    service, identity = prepared(visual, times=(200, 6000))
    chosen = service.queue("asset_one")["roles"][0]["candidates"][0]
    service.pipeline.review("asset_one", chosen["region_id"], status="unknown", reason="not identifiable")
    assert service.queue("asset_one")["roles"][0]["available_count"] == 1
    reference = service.gallery.trusted("asset_one")[0]
    service.gallery.update("asset_one", identity, reference["id"], status="revoked", reason="wrong character")
    assert service.queue("asset_one")["roles"][0]["available_count"] == 0


def test_sampling_is_bounded_local_candidate_only_and_reuses_existing_frame(visual, monkeypatch):
    service, _identity = prepared(visual, times=(2500,))
    before = service.pipeline.settings("asset_one")
    extracted = []
    monkeypatch.setattr(service.library, "require_source", lambda asset: Path("synthetic.mp4"))
    def extract(_source, at_ms, output):
        extracted.append(at_ms)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(picture())
        return {"id": output.stem, "path": str(output), "at_ms": at_ms}
    monkeypatch.setattr("scenerecall.media.extract_frame", extract)
    value = service.start_sampling("asset_one", 3)
    service._running[value["scope_id"]].join(10)
    result = service.sampling("asset_one")
    assert result["status"] == "completed" and result["completed"] == result["total"] == 3
    assert 2500 not in extracted and len(extracted) == 2
    assert service.pipeline.settings("asset_one") == before
    assert all(row["match"]["status"] != "confirmed" for row in service.pipeline.reviews("asset_one")["regions"])
    second = service.start_sampling("asset_one", 3)
    service._running[second["scope_id"]].join(10)
    assert second["round"] == 1 and second["requested_times"] != result["requested_times"]
    with pytest.raises(ValueError, match="24"):
        service.start_sampling("asset_one", 25)


def test_sampling_restores_interrupted_progress_and_retries_only_unfinished_frames(visual, monkeypatch):
    service, _identity = prepared(visual, times=(2500, 5000, 7500))
    monkeypatch.setattr(service.library, "require_source", lambda asset: Path("synthetic.mp4"))
    real_run = service.pipeline.run
    calls = []
    def fails_after_first(asset, frames, config):
        calls.append(frames[0]["at_ms"])
        if len(calls) == 2:
            raise RuntimeError("interrupted local worker")
        return real_run(asset, frames, config)
    monkeypatch.setattr(service.pipeline, "run", fails_after_first)
    value = service.start_sampling("asset_one", 3)
    service._running[value["scope_id"]].join(10)
    prior = service.sampling("asset_one")
    assert prior["status"] == "failed" and prior["completed"] == 1
    restart = CharacterConfirmation(service.library, service.pipeline)
    stale = {**prior, "status": "running"}
    atomic_json(service._root(value["scope_id"]) / "sampling.json", stale)
    assert restart.sampling("asset_one")["status"] == "interrupted"
    monkeypatch.setattr(service.pipeline, "run", real_run)
    resumed = restart.start_sampling("asset_one", 3)
    restart._running[value["scope_id"]].join(10)
    assert resumed["id"] == prior["id"]
    assert restart.sampling("asset_one")["completed"] == 3


def test_sampling_reuses_pixels_without_downgrading_calibrated_canonical_region(visual, monkeypatch):
    service, identity = prepared(visual, times=(5000,))
    add_reference(service.gallery, add_role(visual[1], "Other"), red=220)
    source_frame = {"id": "frame_0", "at_ms": 5000}
    result = service.pipeline.run("asset_one", [source_frame], {"enabled": True, "calibration": {
        "model_version": "synthetic-ccip-v1", "dataset_id": "synthetic", "evaluation_id": "calibration_fixture",
        "aggregation": "nearest-two-mean-v1", "max_distance": .1, "min_margin": .1}})
    canonical = result["regions"][0]
    assert canonical["match"]["character_id"] == identity
    receipt = service.pipeline._directory("asset_one") / "regions" / f"{canonical['id']}.json"
    original_bytes = receipt.read_bytes()
    monkeypatch.setattr(service.library, "require_source", lambda asset: Path("synthetic.mp4"))
    monkeypatch.setattr("scenerecall.media.extract_frame", lambda *args: pytest.fail("registered pixels must be reused"))
    sample = service.start_sampling("asset_one", 1)
    service._running[sample["scope_id"]].join(10)
    completed = service.sampling("asset_one")
    assert completed["status"] == "completed"
    assert completed["frame_ids"] != ["frame_0"]
    assert receipt.read_bytes() == original_bytes
    assert service.library.frame_path("asset_one", completed["frame_ids"][0]) == service.library.frame_path("asset_one", "frame_0")
    sampled = [row for row in service.pipeline.reviews("asset_one")["regions"] if row["frame_id"] in completed["frame_ids"]]
    assert len(sampled) == 1 and sampled[0]["match"]["status"] == "candidate"


def test_retrieval_plan_shots_prevent_repeated_same_shot_candidates(visual):
    service, _identity = prepared(visual)
    atomic_json(service.library.asset_dir("asset_one") / "retrieval" / "plans" / "plan.json", {
        "source_fingerprint": "synthetic-video", "frames": [],
        "shots": [{"id": "long_shot", "start_ms": 0, "end_ms": 19000}]})
    candidates = service.queue("asset_one")["roles"][0]["candidates"]
    assert len(candidates) == 1 and candidates[0]["shot_id"] == "long_shot"


@pytest.mark.asyncio
async def test_confirmation_api_contract_and_limits(visual):
    service, identity = prepared(visual)
    app = FastAPI()
    register_visual_identity_routes(app, service.library, None)
    base = "/api/assets/asset_one/identity/confirmation"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        queue = (await client.get(base)).json()
        assert queue["summary"]["total"] == 1
        assert (await client.get(base + "/sampling")).json() == {"sampling": None}
        chosen = queue["roles"][0]["candidates"][0]
        response = await client.post(base + f"/characters/{identity}/decision", json={
            "action": "confirm", "single_character": True, "use_as_reference": True, **candidate_args(chosen)})
        assert response.status_code == 200 and response.json()["summary"]["confirmed"] == 1
        assert (await client.post(base + "/sample", json={"max_frames": 25})).status_code == 422
        assert (await client.post(base + "/sample", json={"max_frames": True})).status_code == 422


def test_confirm_rechecks_quality_and_does_not_upgrade_group_crop(visual):
    service, identity = prepared(visual)
    chosen = service.queue("asset_one")["roles"][0]["candidates"][0]
    path = service.pipeline._directory("asset_one") / "regions" / f"{chosen['region_id']}.json"
    row = deepcopy(read_json(path))
    row["quality"].update(accepted=False, single_character=False, person_count=2)
    atomic_json(path, row)
    with pytest.raises(ValueError, match="单人"):
        confirm(service, identity, chosen)
    assert len(service.gallery.trusted("asset_one")) == 1
