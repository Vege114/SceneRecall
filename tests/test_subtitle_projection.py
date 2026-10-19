import copy

import pytest

from scenerecall.subtitle_projection import compact_subtitle_records


def cue(identity, start, end, **extra):
    return {"id": identity, "record_id": identity, "kind": "subtitle", "asset_id": "film",
            "language": "zh", "text": "同一句台词", "start_ms": start, "end_ms": end,
            "evidence_frame_ids": [], "favorite": False, "note": "", "review_status": "unreviewed", **extra}


def test_compaction_is_search_only_and_preserves_all_original_time_boundaries():
    rows = [cue("b", 100, 500, source_start_ms=-100, source_end_ms=300,
                evidence_frame_ids=["frame-2", "frame-1"]),
            cue("a", 400, 800, source_start_ms=200, source_end_ms=600,
                evidence_frame_ids=["frame-1", "frame-3"])]
    original = copy.deepcopy(rows)

    [merged] = compact_subtitle_records(rows)

    assert rows == original
    assert (merged["id"], merged["record_id"], merged["start_ms"], merged["end_ms"]) == ("a", "a", 100, 800)
    assert merged["source_record_ids"] == ["b", "a"]
    assert merged["source_spans"] == [
        {"record_id": "b", "start_ms": 100, "end_ms": 500, "source_start_ms": -100, "source_end_ms": 300},
        {"record_id": "a", "start_ms": 400, "end_ms": 800, "source_start_ms": 200, "source_end_ms": 600},
    ]
    assert (merged["source_start_ms"], merged["source_end_ms"]) == (200, 600)
    assert merged["evidence_frame_ids"] == ["frame-2", "frame-1", "frame-3"]
    assert merged["thumbnail"] == "frame-2"


@pytest.mark.parametrize("metadata", [
    {"note": "我做的注释"}, {"favorite": True}, {"user_confirmed": True}, {"review_status": "user_confirmed"},
])
def test_every_annotated_cue_remains_individually_addressable(metadata):
    rows = [cue("a", 100, 500, **metadata), cue("b", 100, 500, **metadata), cue("c", 100, 500)]
    assert compact_subtitle_records(rows) == rows


@pytest.mark.parametrize("metadata", [
    {"asset_id": "other-film"}, {"language": "ja"}, {"text": "另一句台词"},
    {"entities": [{"id": "person-1", "name": "小明"}]}, {"source_stream_index": 3},
    {"character": "小明"}, {"review_status": "reviewed"}, {"summary": "人工解释"},
])
def test_distinct_semantics_languages_and_source_tracks_are_not_merged(metadata):
    rows = [cue("a", 100, 500), cue("b", 100, 500, **metadata)]
    assert compact_subtitle_records(rows) == rows


def test_chain_merges_only_strict_overlap_and_keeps_touching_and_later_utterances():
    rows = [cue("a", 0, 1000), cue("b", 900, 1900), cue("c", 1800, 2000),
            cue("d", 2000, 2200), cue("e", 2201, 2500)]
    result = compact_subtitle_records(rows)
    assert [(row["id"], row["start_ms"], row["end_ms"]) for row in result] == [
        ("a", 0, 2000), ("d", 2000, 2200), ("e", 2201, 2500)]
    assert result[1:] == rows[3:]


def test_simultaneous_different_text_and_non_subtitle_order_are_preserved():
    visual = cue("visual", 0, 3000, kind="visual")
    audio = cue("audio", 0, 3000, kind="audio")
    other = cue("other", 100, 800, text="同时出现的另一句")
    rows = [visual, cue("b", 100, 500), other, audio, cue("a", 400, 800)]
    result = compact_subtitle_records(rows)
    assert [row["id"] for row in result] == ["visual", "a", "other", "audio"]
    assert result[0] == visual and result[2] == other and result[3] == audio


def test_representative_is_deterministic_and_projection_is_idempotent():
    rows = [cue("c", 300, 800), cue("a", 100, 500), cue("b", 100, 400)]
    result = compact_subtitle_records(rows)
    assert result == compact_subtitle_records(list(reversed(rows)))
    assert result == compact_subtitle_records(result)
    assert result == compact_subtitle_records(compact_subtitle_records(rows[:2]) + rows[2:])


def test_partial_compaction_keeps_chronological_original_source_ids():
    rows = [cue("a", 100, 500), cue("c", 400, 800), cue("b", 200, 600)]
    result = compact_subtitle_records(rows)
    assert result == compact_subtitle_records(compact_subtitle_records(rows[:2]) + rows[2:])
    assert result[0]["source_record_ids"] == ["a", "b", "c"]


def test_original_record_identity_is_preferred_to_projection_identity():
    [merged] = compact_subtitle_records([
        cue("projection-1", 100, 500, record_id="original-b"),
        cue("projection-2", 100, 500, record_id="original-a"),
    ])
    assert merged["id"] == "projection-2" and merged["record_id"] == "original-a"
    assert merged["source_record_ids"] == ["original-a", "original-b"]


def test_singletons_and_empty_input_do_not_gain_projection_fields():
    rows = [cue("a", 100, 500), cue("b", 100, 500, language="ja")]
    assert compact_subtitle_records(rows) == rows
    assert compact_subtitle_records([]) == []


def test_many_overlapping_effect_fragments_form_one_projection():
    # A sweep over one large overlap chain must not compare every pair of cues.
    rows = [cue(f"effect-{i:05}", i, i + 1000, source_cue_index=i) for i in range(20_000)]
    [merged] = compact_subtitle_records(rows)
    assert (merged["start_ms"], merged["end_ms"]) == (0, 20_999)
    assert len(merged["source_record_ids"]) == len(merged["source_spans"]) == 20_000
