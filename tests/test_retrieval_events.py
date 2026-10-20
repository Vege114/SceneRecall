import copy

import pytest

from scenerecall.retrieval_events import event_seek


def record():
    return {"id": "window", "record_id": "window", "asset_id": "episode", "kind": "visual",
            "evidence_type": "retrieval_window", "source_fingerprint": "synthetic-source",
            "provenance": {"evidence_kind": "current_visual"},
            "start_ms": 0, "end_ms": 6000, "hit_frame_id": "last", "seek_ms": 2000, "match_type": "partial",
            "evidence_frame_ids": ["first", "last"],
            "ordered_frames": [
                {"id": "first", "at_ms": 1000, "pts": 1, "time_base": [1, 1]},
                {"id": "last", "at_ms": 2000, "pts": 2, "time_base": [1, 1]}],
            "entities": [
                {"id": "p1", "character_id": "alice", "character_name": "阿岚", "aliases": ["兰"],
                 "binding_status": "human_confirmed", "description": "红衣人物"},
                {"id": "p2", "character_id": "bob", "character_name": "阿夏", "aliases": [],
                 "binding_status": "auto_bound", "description": "蓝衣人物"},
                {"id": "cup", "description": "红杯"}],
            "concept_bindings": [{"entity_id": "cup", "concept_id": "magic-cup", "concept_name": "星杯",
                                  "aliases": ["圣杯"], "status": "confirmed"}],
            "events": [{"action": "递给", "actor_ids": ["p1"], "target_ids": [], "object_id": "cup",
                        "recipient_id": "p2", "polarity": "positive", "temporal": True, "uncertainty": "",
                        "start_ms": 1000, "end_ms": 2200, "evidence_frame_ids": ["first", "last"]}]}


@pytest.mark.parametrize("query", ["阿岚把红杯递给阿夏", "找到兰将圣杯交给阿夏的画面", "红衣人物把星杯递给蓝衣人物", "递给"])
def test_event_seek_uses_supported_onset_and_preserves_hit_frame_and_partial(query):
    source = record()
    before = copy.deepcopy(source)
    result = event_seek(query, source)
    assert result["seek_ms"] == 1000 and result["seek_basis"] == "local_event"
    assert result["event_evidence"]["actor_ids"] == ["p1"]
    assert result["event_evidence"]["participants"][0]["character_id"] == "alice"
    assert result["event_evidence"]["ordered_frames"] == source["ordered_frames"]
    assert source == before and source["hit_frame_id"] == "last" and source["match_type"] == "partial"
    assert "match_type" not in result and "hit_frame_id" not in result


@pytest.mark.parametrize("query", [
    "阿岚没有把红杯递给阿夏", "阿夏把红杯递给阿岚", "阿岚把蓝杯递给阿夏",
    "阿岚把红杯递给阿红", "阿岚在厨房把红杯递给阿夏", "阿岚先转身再把红杯递给阿夏",
    "阿岚把红杯递给阿夏并且离开", "阿岚迅速把红杯递给阿夏", "阿岚把红杯递给阿夏两次",
])
def test_unresolved_query_constraints_never_retime_a_hit(query):
    assert event_seek(query, record()) is None


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(kind="scene"),
    lambda row: row.update(provenance={}),
    lambda row: row["events"][0].update(temporal=False),
    lambda row: row["events"][0].update(polarity="negative"),
    lambda row: row["events"][0].update(uncertainty="无法确定交给谁"),
    lambda row: row["events"][0].update(start_ms=0),
    lambda row: row["events"][0].update(end_ms=7000),
    lambda row: row["events"][0].update(evidence_frame_ids=["last"]),
    lambda row: row["events"][0].update(evidence_frame_ids=["last", "first"]),
    lambda row: row["events"][0].update(evidence_frame_ids=["first", "outside"]),
    lambda row: row["ordered_frames"][1].update(pts=1),
    lambda row: row["ordered_frames"][1].update(pts=2000, time_base=[1, 2000]),
    lambda row: row["events"].append(copy.deepcopy(row["events"][0])),
])
def test_current_temporal_evidence_and_unique_event_are_required(mutation):
    source = record()
    mutation(source)
    assert event_seek("阿岚把红杯递给阿夏", source) is None


def test_candidate_name_or_revoked_concept_cannot_satisfy_query_role():
    source = record()
    source["entities"][0].update(binding_status="candidate", character_id=None)
    assert event_seek("阿岚把红杯递给阿夏", source) is None
    source = record()
    source["concept_bindings"][0]["status"] = "revoked"
    assert event_seek("阿岚把星杯递给阿夏", source) is None


def test_turn_and_enter_queries_keep_actor_and_place_roles():
    source = record()
    source["events"][0].update(action="转身", object_id=None, recipient_id=None)
    assert event_seek("请帮我找到阿岚正在转身的片段", source)["seek_ms"] == 1000
    assert event_seek("阿夏转身", source) is None
    source["entities"].append({"id": "room", "description": "房间"})
    source["events"][0].update(action="走进", target_ids=["room"])
    assert event_seek("阿岚进入房间", source)["seek_ms"] == 1000
    assert event_seek("阿岚进入驾驶舱", source) is None


def test_merged_result_cannot_move_event_outside_its_original_source_span():
    source = record()
    source["member_ids"] = ["previous", "window"]
    source["source_spans"] = [{"record_id": "previous", "start_ms": 0, "end_ms": 3000},
                             {"record_id": "window", "start_ms": 3000, "end_ms": 6000}]
    assert event_seek("递给", source) is None


@pytest.mark.parametrize("action", ["试图递给", "似乎递给", "想要递给", "递给阿岚", "把蓝杯递给阿夏"])
def test_action_prose_cannot_hide_unresolved_qualifiers_or_conflicting_roles(action):
    source = record()
    source["events"][0]["action"] = action
    assert event_seek("阿岚把红杯递给阿夏", source) is None


def test_complete_action_phrase_must_agree_with_its_structured_roles():
    source = record()
    source["events"][0]["action"] = "阿岚把红杯递给阿夏"
    assert event_seek("兰将圣杯交给阿夏", source)["seek_ms"] == 1000
