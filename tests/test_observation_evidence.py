"""Sparse model evidence can be normalized without broadening current-frame access."""
from copy import deepcopy

import pytest

from scenerecall.models import Observation, normalize_observation_evidence


def sparse_observation():
    return {"id": "observation", "asset_id": "asset", "run_id": "run", "window_id": "window",
            "shot_id": "shot", "start_ms": 0, "end_ms": 5000, "summary": "一名角色按灯后转身",
            "evidence_frame_ids": ["frame-2"],
            "entities": [{"id": "actor", "kind": "person", "description": "角色",
                          "evidence_frame_ids": ["frame-0"]}],
            "events": [{"actor_ids": ["actor"], "action": "按灯", "start_ms": 1000, "end_ms": 1000,
                        "evidence_frame_ids": ["frame-1"]}],
            "spatial_observations": [{"subject_id": "actor", "relation": "位于左侧",
                                      "evidence_frame_ids": ["frame-3"]}],
            "metadata": {"receipt": "paid-result"}}


def test_normalization_unions_all_cited_current_frames_in_input_order_without_mutation():
    original = sparse_observation()
    before = deepcopy(original)
    input_ids = [f"frame-{number}" for number in range(5)]
    normalized = normalize_observation_evidence(original, input_ids)
    assert normalized["evidence_frame_ids"] == input_ids[:4]
    assert normalized["entities"] == original["entities"]
    assert normalized["events"] == original["events"]
    assert normalized["spatial_observations"] == original["spatial_observations"]
    assert original == before
    assert Observation.model_validate(normalized).evidence_frame_ids == input_ids[:4]
    replay = normalize_observation_evidence(normalized, input_ids)
    assert replay == normalized and replay is not normalized
    normalized["entities"][0]["description"] = "changed returned copy"
    normalized["metadata"]["receipt"] = "changed returned copy"
    assert original == before


def test_normalization_preserves_existing_root_only_evidence_and_deduplicates():
    original = sparse_observation()
    original["evidence_frame_ids"] = ["frame-2", "frame-2", "frame-0"]
    original["events"][0]["evidence_frame_ids"] = ["frame-1", "frame-1"]
    normalized = normalize_observation_evidence(original, ["frame-3", "frame-2", "frame-1", "frame-0", "unused"])
    assert normalized["evidence_frame_ids"] == ["frame-3", "frame-2", "frame-1", "frame-0"]
    assert original["evidence_frame_ids"] == ["frame-2", "frame-2", "frame-0"]


def test_legacy_single_spatial_frame_reference_is_included_without_rewriting_position():
    original = sparse_observation()
    original["spatial_observations"] = [{"entity_id": "actor", "relation": "位于左侧", "frame_id": "frame-3"}]
    normalized = normalize_observation_evidence(original, [f"frame-{number}" for number in range(5)])
    assert normalized["evidence_frame_ids"] == [f"frame-{number}" for number in range(4)]
    assert normalized["spatial_observations"] == original["spatial_observations"]
    assert Observation.model_validate(normalized).spatial_observations[0]["frame_id"] == "frame-3"


@pytest.mark.parametrize("part", ["root", "entities", "events", "spatial_observations", "spatial_frame_id"])
@pytest.mark.parametrize("foreign_id", ["unknown-frame", "past-frame", "cue-0", "reference:character"])
def test_normalization_rejects_foreign_evidence_in_every_location_without_dropping_it(part, foreign_id):
    original = sparse_observation()
    if part == "root":
        original["evidence_frame_ids"] = [foreign_id]
    elif part == "spatial_frame_id":
        original["spatial_observations"][0]["frame_id"] = foreign_id
    else:
        original[part][0]["evidence_frame_ids"] = [foreign_id]
    before = deepcopy(original)
    with pytest.raises(ValueError):
        normalize_observation_evidence(original, [f"frame-{number}" for number in range(4)])
    assert original == before
