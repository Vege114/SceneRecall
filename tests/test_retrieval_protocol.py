from copy import deepcopy

import pytest
from pydantic import ValidationError

from scenerecall.retrieval_protocol import output_schema, validate_batch


def inputs():
    units = [{"id": "window", "start_ms": 100, "end_ms": 1000, "local_frame_ids": ["before", "after"]}]
    frames = [{"id": "before", "at_ms": 100}, {"id": "after", "at_ms": 900}]
    data = {"observations": [{"unit_id": "window", "summary": "人物递出物品", "evidence_frame_ids": ["before", "after"],
        "entities": [{"id": "p", "kind": "person", "description": "红衣人物", "evidence_frame_ids": ["before"]}],
        "events": [{"action": "递出", "actor_ids": ["p"], "start_ms": 100, "end_ms": 900,
                    "temporal": True, "evidence_frame_ids": ["before", "after"]}]}]}
    return data, units, frames


def test_temporal_events_preserve_roles_and_order():
    data, units, frames = inputs()
    result = validate_batch(data, units, frames)
    assert result["observations"][0]["events"][0]["actor_ids"] == ["p"]
    data["observations"][0]["events"][0]["evidence_frame_ids"] = ["before"]
    with pytest.raises(ValueError, match="两帧"):
        validate_batch(data, units, frames)
    # A model's false temporal flag cannot turn an explicit transfer into a
    # single-frame event. The original observation is not silently rewritten.
    data["observations"][0]["events"][0]["temporal"] = False
    with pytest.raises(ValueError, match="两帧"):
        validate_batch(data, units, frames)


def test_unknown_fields_candidates_and_context_cannot_become_facts():
    original, units, frames = inputs()
    for mutation, error in ((lambda d: d["observations"][0]["entities"][0].update(concept_id="invented"), ValidationError),
                            (lambda d: d["observations"][0]["entities"][0].update(candidate_character_ids=["other_work"]), ValueError),
                            (lambda d: d["observations"][0].update(evidence_frame_ids=["context_frame"]), ValueError)):
        data = deepcopy(original)
        mutation(data)
        with pytest.raises(error):
            validate_batch(data, units, frames)


def test_schema_is_strict_and_new_fields_survive_roundtrip():
    data, units, frames = inputs()
    data["observations"][0]["entities"][0]["candidate_concept_ids"] = ["prop"]
    result = validate_batch(data, units, frames, concept_ids={"prop"})
    assert result["observations"][0]["entities"][0]["candidate_concept_ids"] == ["prop"]
    schema = output_schema(units, concept_ids={"prop"})
    assert schema["additionalProperties"] is False
    for definition in schema["$defs"].values():
        assert definition["required"] == list(definition["properties"])
        assert definition["additionalProperties"] is False
