"""Work-wide bounded candidate generation with deterministic synthetic backends."""
from copy import deepcopy
from pathlib import Path
import threading

import pytest

from scenerecall.character_confirmation import CharacterConfirmation, ConfirmationSample
from scenerecall.characters import CharacterStore
from scenerecall.library import atomic_json, read_json
from scenerecall.visual_identity import VisualIdentityPipeline
from test_character_confirmation import choices, confirm, prepared
from test_visual_identity import FakeCCIP, add_reference, add_role, frame, picture, visual as visual


def add_episode(service, name, episode):
    asset = {**service.library.get_asset("asset_one"), "id": name, "episode": episode, "duration_ms": 120000}
    atomic_json(service.library.asset_dir(name) / "manifest.json", asset)
    return asset


def fake_extract(monkeypatch):
    calls = []
    monkeypatch.setattr("scenerecall.library.Library.require_source", lambda self, asset: Path(asset + ".mp4"))
    def extract(source, at_ms, output):
        calls.append((source.stem, at_ms))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(picture())
        return {"id": output.stem, "path": str(output), "at_ms": at_ms}
    monkeypatch.setattr("scenerecall.media.extract_frame", extract)
    return calls


def completed(service, **request):
    initial = service.start_sampling("asset_one", **request)
    service._running[initial["scope_id"]].join(20)
    assert not service._running[initial["scope_id"]].is_alive()
    return service.sampling("asset_one")


def test_uncalibrated_pool_has_unique_pixels_raw_topk_and_explicit_any_role_confirmation(visual):
    service, first = prepared(visual, times=(200,))
    second = add_role(visual[1], "Second without reference")
    queue = service.queue("asset_one")
    assert len(queue["review_pool"]) == 1 and not any(role["candidates"] for role in queue["roles"])
    crop = queue["review_pool"][0]
    assert crop["confidence_state"] == "uncalibrated" and crop["assignment_state"] == "pending"
    assert crop["raw_candidates"][0]["character_id"] == first
    assert second in crop["eligible_character_ids"]
    confirm(service, second, crop)
    assert service.queue("asset_one")["review_pool"] == []
    with pytest.raises(ValueError):
        confirm(service, first, crop)
    assert service.pipeline.reviews("asset_one")["regions"][0]["match"]["character_id"] == second


def test_valid_calibration_assigns_only_closest_and_ambiguous_or_low_distance_stay_pending(visual):
    service, first = prepared(visual, times=(200,))
    second = add_role(visual[1], "Other")
    other_reference = add_reference(visual[2], second, red=220)
    calibration = {"model_version": "synthetic-ccip-v1", "dataset_id": "synthetic", "evaluation_id": "heldout",
                   "aggregation": "nearest-two-mean-v1", "max_distance": .1, "min_margin": .1}
    service.pipeline.configure("asset_one", {"calibration": calibration})
    service.pipeline.run("asset_one", [{"id": "frame_0", "at_ms": 200}], {"enabled": True, "calibration": calibration})
    queue = service.queue("asset_one")
    roles = {row["character_id"]: row for row in queue["roles"]}
    assert len(roles[first]["candidates"]) == 1 and roles[second]["candidates"] == []
    assert queue["review_pool"] == []
    crop = roles[first]["candidates"][0]
    assert crop["assigned_character_id"] == first and crop["confidence_state"] == "calibrated"
    # Rejecting the winner does not promote the runner-up to an owner.
    service.decide("asset_one", first, action="reject", source_asset_id="asset_one", region_id=crop["region_id"])
    queue = service.queue("asset_one")
    assert not any(role["candidates"] for role in queue["roles"])
    assert queue["review_pool"][0]["eligible_character_ids"] == [second]

    # Equal reference distance is unresolved even with otherwise valid calibration.
    service.gallery.update("asset_one", second, other_reference["id"], status="revoked", reason="fixture replaced")
    add_reference(visual[2], second, red=90)
    service.pipeline.run("asset_one", [frame(service.library, "frame_tie", at_ms=6000)],
                         {"enabled": True, "calibration": calibration})
    tied = next(row for row in service.queue("asset_one")["review_pool"] if row["frame_id"] == "frame_tie")
    assert tied["confidence_state"] == "ambiguous"
    service.pipeline.run("asset_one", [frame(service.library, "frame_far", red=10, at_ms=8000)],
                         {"enabled": True, "calibration": calibration})
    far = next(row for row in service.queue("asset_one")["review_pool"] if row["frame_id"] == "frame_far")
    assert far["confidence_state"] == "low_confidence"


def test_same_pixels_alias_cannot_escape_reject_unclear_or_human_unknown(visual):
    service, first = prepared(visual, times=(200,))
    second = add_role(visual[1], "Other")
    original = choices(service.queue("asset_one"))[0]
    service.decide("asset_one", first, action="reject", source_asset_id="asset_one", region_id=original["region_id"])
    service.library.register_sample_alias("asset_one", "frame_alias", "frame_0")
    service.pipeline.run("asset_one", [{"id": "frame_alias", "at_ms": 200}], {"enabled": True})
    pool = service.queue("asset_one")["review_pool"]
    assert len(pool) == 1 and pool[0]["eligible_character_ids"] == [second]
    service.pipeline.review("asset_one", original["region_id"], status="unknown", reason="manual lock")
    assert service.queue("asset_one")["review_pool"] == []


def test_work_sampling_global_budget_is_fair_and_never_touches_other_work(visual, monkeypatch):
    service, _ = prepared(visual, times=(200,))
    add_episode(service, "asset_two", 2)
    add_episode(service, "asset_three", 3)
    outsider = {**service.library.get_asset("asset_one"), "id": "asset_alien", "work_id": "work_other", "series": "Other"}
    outsider.pop("character_scope_id", None)
    outsider.pop("project_id", None)
    atomic_json(service.library.asset_dir("asset_alien") / "manifest.json", outsider)
    calls = fake_extract(monkeypatch)
    before = deepcopy(service._read(service.queue("asset_one")["scope_id"]))
    result = completed(service, scope="work", max_frames=5, max_assets=3, frames_per_asset=4)
    assert result["status"] == "completed" and result["completed"] == result["total"] == 5
    assert [row["asset_id"] for row in result["plan"]][:3] == ["asset_one", "asset_two", "asset_three"]
    assert sorted(child["total"] for child in result["assets"]) == [1, 2, 2]
    assert {asset for asset, _ in calls} <= {"asset_one", "asset_two", "asset_three"}
    assert service._read(result["scope_id"]) == before
    assert not (service.pipeline._directory("asset_alien") / "regions").exists()


def test_repeated_work_rounds_do_not_resample_same_time_and_reuse_registered_pixels(visual, monkeypatch):
    service, _ = prepared(visual, times=(200, 6000))
    fake_extract(monkeypatch)
    first = completed(service, scope="work", max_frames=3, frames_per_asset=3)
    second = completed(service, scope="work", max_frames=3, frames_per_asset=3)
    a = {row["at_ms"] for row in first["completed_frames"]}
    b = {row["at_ms"] for row in second["completed_frames"]}
    assert not a & b
    assert first["reused_frames"] >= 2


def test_no_reference_does_not_force_matches_and_empty_detection_has_clear_coverage(visual, monkeypatch):
    library, characters, _ = visual
    identity = add_role(characters)
    service = CharacterConfirmation(library, VisualIdentityPipeline(library, FakeCCIP(boxes=[])))
    fake_extract(monkeypatch)
    result = completed(service, scope="work", max_frames=2, frames_per_asset=2)
    assert result["accepted_regions"] == 0 and service.queue("asset_one")["review_pool"] == []
    role = service.queue("asset_one")["roles"][0]
    assert role["character_id"] == identity and role["coverage"]["status"] == "missing_reference"
    add_reference(service.gallery, identity)
    result = completed(service, scope="work", max_frames=2, frames_per_asset=2)
    assert result["completed"] == 2
    assert service.queue("asset_one")["roles"][0]["coverage"]["status"] == "detection_invalid"


def test_work_partial_failure_is_resumable_without_repeating_successful_frames(visual, monkeypatch):
    service, _ = prepared(visual, times=(200,))
    add_episode(service, "asset_two", 2)
    fake_extract(monkeypatch)
    real_run = service.pipeline.run
    calls = []
    def fail_second(asset_id, frames, config):
        calls.append((asset_id, frames[0]["id"]))
        if asset_id == "asset_two":
            raise RuntimeError("synthetic unavailable")
        return real_run(asset_id, frames, config)
    monkeypatch.setattr(service.pipeline, "run", fail_second)
    prior = completed(service, scope="work", max_frames=4, max_assets=2, frames_per_asset=2)
    assert prior["status"] == "partial" and prior["completed"] == prior["failed_frames"] == 2
    succeeded = {row["frame_id"] for row in prior["completed_frames"]}
    monkeypatch.setattr(service.pipeline, "run", real_run)
    restart = CharacterConfirmation(service.library, service.pipeline)
    result = completed(restart, scope="work", max_frames=4, max_assets=2, frames_per_asset=2)
    assert result["id"] == prior["id"] and result["completed"] == 4 and result["failed_frames"] == 0
    assert succeeded <= set(result["frame_ids"])
    assert all(child["status"] == "completed" and "error" not in child for child in result["assets"])
    assert result["attempt_history"][-1]["status"] == "partial"
    assert len(result["attempt_history"][-1]["failures"]) == 2
    assert all("synthetic unavailable" in item["error"] for item in result["attempt_history"][-1]["failures"])


def test_cancel_preserves_progress_then_restart_resumes_remaining_plan(visual, monkeypatch):
    service, _ = prepared(visual, times=(200,))
    fake_extract(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    run = service.pipeline.run
    def blocked(asset, frames, config):
        entered.set()
        assert release.wait(10)
        return run(asset, frames, config)
    monkeypatch.setattr(service.pipeline, "run", blocked)
    value = service.start_sampling("asset_one", 3, scope="work", frames_per_asset=3)
    assert entered.wait(10)
    assert service.cancel_sampling("asset_one")["status"] == "cancelling"
    release.set()
    service._running[value["scope_id"]].join(20)
    prior = service.sampling("asset_one")
    assert prior["status"] == "cancelled" and prior["completed"] == 1
    monkeypatch.setattr(service.pipeline, "run", run)
    restart = CharacterConfirmation(service.library, service.pipeline)
    result = completed(restart, scope="work", max_frames=3, frames_per_asset=3)
    assert result["id"] == prior["id"] and result["completed"] == 3


def test_selection_changed_while_generation_running_invalidates_without_altering_human_history(visual, monkeypatch):
    service, identity = prepared(visual, times=(200,))
    fake_extract(monkeypatch)
    run = service.pipeline.run
    def change_selection(asset, frames, config):
        result = run(asset, frames, config)
        roster = CharacterStore(service.library).roster("asset_one")
        CharacterStore(service.library).set_recognition_selection("asset_one", [], expected_revision=roster["revision"],
                                                                expected_scope=roster["scope_id"])
        return result
    monkeypatch.setattr(service.pipeline, "run", change_selection)
    prior = deepcopy(service._read(service.queue("asset_one")["scope_id"]))
    result = completed(service, scope="work", max_frames=3, frames_per_asset=3)
    assert result["status"] == "invalidated" and result["completed"] == 0
    assert service._read(result["scope_id"]) == prior
    assert service.queue("asset_one")["review_pool"] == []


def test_offline_source_can_reuse_cached_frame_and_reports_partial_new_frame_failures(visual, monkeypatch):
    service, _ = prepared(visual, times=(200,))
    def offline(asset):
        raise FileNotFoundError("synthetic offline")
    monkeypatch.setattr(service.library, "require_source", offline)
    result = completed(service, scope="work", max_frames=2, frames_per_asset=2)
    assert result["status"] == "partial" and result["completed"] == result["failed_frames"] == 1
    assert result["reused_frames"] == 1
    assert "offline" in result["error"]


def test_request_budgets_are_strict_and_default_asset_limit_is_compatible():
    assert ConfirmationSample.model_validate({"scope": "work", "max_frames": 120}).max_frames == 120
    for payload in ({"max_frames": 25}, {"scope": "work", "max_frames": 121},
                    {"scope": "work", "max_assets": 25}, {"scope": "work", "max_frames": True}):
        with pytest.raises(ValueError):
            ConfirmationSample.model_validate(payload)


def test_review_pool_pagination_keeps_every_region_reachable_and_later_page_confirmable(visual):
    service, first = prepared(visual, times=(200,))
    second = add_role(visual[1], "Explicit role absent from TopK")
    source = read_json(next((service.pipeline._directory("asset_one") / "regions").glob("*.json")))
    registry = read_json(service.library.asset_dir("asset_one") / "frames" / "registry.json")
    for index in range(124):
        identifier = f"frame_page_{index}"
        registry[identifier] = {**registry["frame_0"], "at_ms": 250 + index * 40}
        region = {**deepcopy(source), "id": f"region_page_{index}", "frame_id": identifier, "at_ms": registry[identifier]["at_ms"]}
        atomic_json(service.pipeline._directory("asset_one") / "regions" / f"{region['id']}.json", region)
        (service.pipeline._directory("asset_one") / "crops" / f"{region['id']}.png").write_bytes(picture())
    atomic_json(service.library.asset_dir("asset_one") / "frames" / "registry.json", registry)
    page = service.queue("asset_one")
    assert page["review_pool_count"] == 125 and len(page["review_pool"]) == 20
    seen, last = set(), None
    while True:
        current = {(row["asset_id"], row["region_id"]) for row in page["review_pool"]}
        assert not seen & current
        seen |= current
        last = page["review_pool"][-1]
        cursor = page["review_pool_next_cursor"]
        if cursor is None:
            break
        page = service.queue("asset_one", review_pool_cursor=cursor)
    assert len(seen) == 125 and page["review_pool_previous_cursor"] is not None
    confirm(service, second, last)
    assert service.queue("asset_one")["review_pool_count"] == 124
    with pytest.raises(ValueError):
        confirm(service, first, last)


def test_budget_change_preserves_sampling_dedup_history(visual, monkeypatch):
    service, _ = prepared(visual, times=(200, 6000))
    fake_extract(monkeypatch)
    first = completed(service, scope="work", max_frames=2, frames_per_asset=2)
    second = completed(service, scope="work", max_frames=3, frames_per_asset=3)
    assert not {row["at_ms"] for row in first["completed_frames"]} & {row["at_ms"] for row in second["completed_frames"]}


def test_calibrated_role_choices_keep_existing_shot_and_time_diversity(visual):
    service, first = prepared(visual, times=(200, 6000, 8000))
    add_reference(visual[2], add_role(visual[1], "Other"), red=220)
    calibration = {"model_version": "synthetic-ccip-v1", "dataset_id": "synthetic", "evaluation_id": "heldout",
                   "aggregation": "nearest-two-mean-v1", "max_distance": .1, "min_margin": .1}
    service.pipeline.configure("asset_one", {"calibration": calibration})
    service.pipeline.run("asset_one", [{"id": f"frame_{index}", "at_ms": at} for index, at in enumerate((200, 6000, 8000))],
                         {"enabled": True, "calibration": calibration})
    atomic_json(service.library.asset_dir("asset_one") / "keyframe-plans" / "same-shot.json", {
        "fingerprint": "synthetic-video", "frames": [{"id": f"frame_{index}", "shot_id": "same_shot"} for index in range(3)]})
    role = next(row for row in service.queue("asset_one")["roles"] if row["character_id"] == first)
    assert role["available_count"] == 3 and len(role["candidates"]) == 1
    assert role["candidates"][0]["shot_id"] == "same_shot"


@pytest.mark.parametrize("has_reference", [False, True])
def test_human_assigned_and_recovering_role_never_inherit_another_roles_distance(visual, has_reference):
    service, first = prepared(visual, times=(200,))
    second = add_role(visual[1], "Human selected role")
    if has_reference:
        add_reference(service.gallery, second, red=220)
    result = service.pipeline.run("asset_one", [{"id": "frame_0", "at_ms": 200}], {"enabled": True, "top_k": 20})
    region = result["regions"][0]
    raw = region["match"]["candidates"]
    assert raw[0]["character_id"] == first
    service.pipeline.review("asset_one", region["id"], status="candidate", character_id=second, reason="explicit human choice")
    def assigned():
        return next(role for role in service.queue("asset_one")["roles"] if role["character_id"] == second)["candidates"][0]
    candidate = assigned()
    expected = next((row for row in raw if row["character_id"] == second), {})
    assert candidate["distance"] == expected.get("distance")
    assert candidate["candidate_rank"] == (2 if has_reference else None)
    assert candidate["raw_candidates"] == raw
    class BrokenSearch:
        def sync_asset(self, *args):
            raise RuntimeError("synthetic publication interruption")
    service.search = BrokenSearch()
    with pytest.raises(RuntimeError):
        confirm(service, second, candidate)
    recovered = assigned()
    assert recovered["recovery_required"] and recovered["assigned_character_id"] == second
    assert recovered["distance"] == expected.get("distance")
    assert recovered["candidate_rank"] == (2 if has_reference else None)


@pytest.mark.parametrize("mutation", ["model", "reference_version", "reference_sha256", "reference_status"])
def test_generation_invalidates_mutated_inputs_before_recording_old_signature_history(visual, monkeypatch, mutation):
    service, _ = prepared(visual, times=(200,))
    fake_extract(monkeypatch)
    run = service.pipeline.run
    calls = []
    def mutate_during_run(asset_id, frames, config):
        result = run(asset_id, frames, config)
        calls.append(frames[0]["id"])
        if mutation == "model":
            service.pipeline.configure(asset_id, {"model": "ccip-caformer-5_fp32"})
        else:
            scope = CharacterStore(service.library).roster(asset_id)["scope_id"]
            with service.library.lock:
                gallery = service.gallery._read(scope)
                reference = gallery["references"][0]
                if mutation == "reference_version":
                    reference["version"] += 1
                elif mutation == "reference_sha256":
                    reference["sha256"] = "0" * 64
                else:
                    reference["status"] = "revoked"
                atomic_json(service.gallery._directory(scope) / "gallery.json", gallery)
        return result
    monkeypatch.setattr(service.pipeline, "run", mutate_during_run)
    result = completed(service, scope="work", max_frames=3, frames_per_asset=3)
    assert result["status"] == "invalidated" and result["completed"] == 0
    assert result["attempted"] == 0 and len(calls) == 1
    assert read_json(service._root(result["scope_id"]) / "sampled-history.json", {"frames": []})["frames"] == []
