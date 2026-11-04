"""Named-character metadata cannot borrow identities from other source frames."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from scenerecall.character_query import name_index, resolve_text
from scenerecall.character_ranking import character_frame_support, grounded_rrf_score, summarize_character_match
from scenerecall.visual_index import evidence_digest
from test_character_query import profile
from test_search import local, record  # noqa: F401


TARGETS = [{"scope_id": "cast", "character_id": "a", "name": "王小明"},
           {"scope_id": "cast", "character_id": "b", "name": "李小红"}]


def person(identity, region, frame, *, confirmed=False, scope="cast", mapped=True):
    binding = {"id": "binding-" + region, "policy_version": "identity-binding-v2", "character_scope_id": scope,
               "status": "human_confirmed" if confirmed else "candidate",
               "character_id" if confirmed else "candidate_character_id": identity}
    value = {"id": region, "kind": "person", "source": "retrieval-local-identity-v1",
             "identity_region_id": region, "identity_support_region_ids": [], "evidence_frame_ids": [frame],
             "identity_binding": binding, "character_id" if confirmed else "candidate_character_id": identity}
    if mapped:
        value["identity_region_evidence"] = [{"region_id": region, "frame_id": frame}]
    return value


def window(identity, frames=("empty", "people"), entities=(), **extra):
    return record(identity, "候选姓名", kind="visual", summary="", source_fingerprint="synthetic",
                  frame_ids=list(frames), evidence_frame_ids=list(frames), entities=list(entities),
                  ordered_frames=[{"id": frame, "at_ms": index * 100, "pts": index * 100, "time_base": [1, 1000]}
                                  for index, frame in enumerate(frames)], **extra)


class Names:
    def revision(self, *_args):
        return "stable-names"

    def resolve(self, query, *_args):
        return resolve_text(query, name_index([profile("a", "王小明", "cast"), profile("b", "李小红", "cast")]))


def test_exact_same_frame_coverage_preserves_candidate_strength_and_scope():
    row = window("w", entities=[person("a", "a", "people", confirmed=True), person("b", "b", "people")])
    match = summarize_character_match(row, TARGETS, "people")
    assert match["status"] == "candidate" and match["confirmed_count"] == match["candidate_count"] == 1
    assert match["required_count"] == 2
    assert summarize_character_match(row, TARGETS, "empty")["status"] == "unverified"
    row["entities"][1]["identity_binding"]["character_scope_id"] = "another-cast"
    match = summarize_character_match(row, TARGETS, "people")
    assert match["status"] == "partial" and match["candidate_count"] == 0


def test_regions_cannot_be_shared_by_two_names_or_borrow_broad_support_frames():
    row = window("w", entities=[person("a", "shared", "people"), person("b", "shared", "people")])
    assert character_frame_support(row, TARGETS) == {}
    broad = person("a", "a", "empty", mapped=False)
    broad.update(source="generative-observation", evidence_frame_ids=["empty", "people"])
    row["entities"] = [broad]
    assert character_frame_support(row, TARGETS) == {}
    broad["identity_region_evidence"] = [{"region_id": "a", "frame_id": "empty"}]
    assert summarize_character_match(row, TARGETS, "people")["confirmed_count"] == 0
    assert summarize_character_match(row, TARGETS, "people")["candidate_count"] == 0
    broad["is_character"] = False
    assert character_frame_support(row, TARGETS) == {}


def test_old_single_region_projection_is_safe_but_explicit_empty_mapping_is_authoritative():
    row = window("w", entities=[person("a", "a", "people", mapped=False)])
    assert summarize_character_match(row, TARGETS, "people")["candidate_count"] == 1
    row["entities"][0]["identity_region_evidence"] = []
    assert summarize_character_match(row, TARGETS, "people")["candidate_count"] == 0


def test_frame_gate_keeps_visual_but_window_text_cannot_restore_identity_votes():
    row = window("w", entities=[person("a", "a", "people"), person("b", "b", "people")])
    votes = {"visual": .01, "lexical": .02, "semantic": .03, "concept": .04}
    assert grounded_rrf_score(row, TARGETS, votes, "empty") == .01
    assert grounded_rrf_score(row, TARGETS, votes, "people") == pytest.approx(.01 + .5 * .09)
    row["summary"] = "两个人站在塔上，背后亮着灯。"
    assert grounded_rrf_score(row, TARGETS, votes, "empty") == .01
    assert summarize_character_match(row, TARGETS, "empty")["status"] == "unverified"
    row["kind"] = "subtitle"
    assert grounded_rrf_score(row, TARGETS, votes, "empty") == pytest.approx(.1)


@pytest.mark.asyncio
async def test_fusion_weights_before_75_and_full_short_names_agree(local, monkeypatch):  # noqa: F811
    local.character_resolver = Names()
    background = window("background", ("empty", "other-a", "other-b"),
                        entities=[person("a", "a", "other-a"), person("b", "b", "other-b")])
    complete = window("complete", entities=[person("a", "ca", "people"), person("b", "cb", "people")])
    background.update(hit_frame_id="empty", visual_similarity=.99)
    complete.update(hit_frame_id="people", visual_similarity=.95)

    async def recall(_query, _search_text, filters, *_args):
        assert [(row["scope_id"], row["character_id"]) for row in filters["_character_targets"]] == [("cast", "a"), ("cast", "b")]
        return [([background, complete], []), ([], []), ([background, complete], []), ([], [])]

    monkeypatch.setattr(local, "_recall_parallel", recall)
    full = await local.search("王小明与李小红站在塔上", min_match_score=75)
    short = await local.search("小明与小红站在塔上", min_match_score=75)
    assert full == short
    assert full["results"] == []  # Candidate deductions are not normalized away.
    lower = await local.search("小明与小红站在塔上", min_match_score=70)
    assert [row["id"] for row in lower["results"]] == ["complete"]
    result = lower["results"][0]
    assert result["character_match"]["candidate_count"] == 2
    assert result["character_match"]["confirmed_count"] == 0
    assert result["score"] == pytest.approx(1.5 / 62)
    assert result["raw_rrf_score"] == pytest.approx(2 / 62)
    assert result["_match_score_reference"] == pytest.approx(2 / 61)
    assert result["match_score"] == pytest.approx(100 * (1.5 / 62) / (2 / 61))
    assert result["match_score"] < 75
    assert full["selection"]["score_basis"] == "character_grounded_rrf_v1"
    assert local.completed_character_targets() == TARGETS
    copied = local.completed_character_targets()
    copied.clear()
    assert local.completed_character_targets() == TARGETS


@pytest.mark.asyncio
async def test_missing_identity_penalty_can_filter_all_at_75_but_visual_fallback_survives_lower_threshold(local, monkeypatch):  # noqa: F811
    local.character_resolver = Names()
    rows = [window(identity) for identity in ("a", "b")]
    for row in rows:
        row.update(hit_frame_id="empty", visual_similarity=.9)

    async def recall(*_args):
        return [(rows, []), ([], []), (rows, []), ([], [])]

    monkeypatch.setattr(local, "_recall_parallel", recall)
    assert (await local.search("王小明与李小红站在塔上", min_match_score=75))["results"] == []
    lower = await local.search("王小明与李小红站在塔上", min_match_score=49)
    assert [row["id"] for row in lower["results"]] == ["a", "b"]
    assert lower["results"][0]["match_score"] == pytest.approx(50)
    assert all(row["character_match"]["status"] == "unverified" for row in lower["results"])
    assert all(row["score"] == pytest.approx(1 / (61 + index)) for index, row in enumerate(lower["results"]))
    assert all(row["_match_score_reference"] == pytest.approx(2 / 61) for row in lower["results"])
    assert (await local.search("王小明与李小红站在塔上", min_match_score=50))["results"] == []
    legacy = await local.search("王小明与李小红站在塔上", limit=1)
    assert [row["id"] for row in legacy["results"]] == ["a"]
    assert "match_score" not in legacy["results"][0]
    assert legacy["results"][0]["_match_score_reference"] == legacy["results"][0]["score"]


@pytest.mark.asyncio
async def test_no_names_keeps_old_rrf_and_private_filter_is_not_injectable(local, monkeypatch):  # noqa: F811
    local.character_resolver = Names()
    row = window("visual").copy()
    row.update(hit_frame_id="empty")

    async def recall(_query, _search_text, filters, *_args):
        assert "_character_targets" not in filters
        return [([row], []), ([], []), ([row], []), ([], [])]

    monkeypatch.setattr(local, "_recall_parallel", recall)
    result = await local.search("塔上", {"_character_targets": TARGETS}, min_match_score=75)
    assert result["results"][0]["score"] == pytest.approx(2 / 61)
    assert "character_match" not in result["results"][0]
    assert result["selection"]["score_basis"] == "relative_rrf_v1"
    assert local.completed_character_targets() == []


@pytest.mark.asyncio
async def test_model_cannot_upgrade_other_frame_identities_before_composition(local, monkeypatch):  # noqa: F811
    local.character_resolver = Names()
    row = window("w", entities=[person("a", "a", "people", confirmed=True), person("b", "b", "people", confirmed=True)])
    row.update(hit_frame_id="empty")
    seen = []

    async def recall(*_args):
        return [([row], []), ([], []), ([row], []), ([], [])]

    async def decide(*_args):
        return SimpleNamespace(data={"candidates": [{"id": "w", "score": 1., "match_type": "full"}]})

    async def compose(_profile, _query, candidates):
        seen.extend(candidates)
        return SimpleNamespace(data={"text": "只支持部分条件"})

    monkeypatch.setattr(local, "_recall_parallel", recall)
    monkeypatch.setattr(local.providers, "decide", decide)
    monkeypatch.setattr(local.providers, "compose", compose)
    result = await local.search("王小明与李小红站在塔上", min_match_score=0,
                                decision_profile_id="judge", answer_profile_id="answer")
    assert result["results"][0]["match_type"] == "partial"
    assert seen[0]["match_type"] == "partial"
    assert seen[0]["character_match"]["status"] == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("branch", ["lexical", "concept"])
async def test_nonvisual_only_identity_recall_locates_real_support_without_claiming_visual_similarity(local, monkeypatch, branch):  # noqa: F811
    local.character_resolver = Names()
    row = window("w", entities=[person("a", "a", "people", confirmed=True), person("b", "b", "people", confirmed=True)])

    async def recall(*_args):
        return [([row] if branch == name else [], []) for name in ("lexical", "semantic", "visual", "concept")]

    monkeypatch.setattr(local, "_recall_parallel", recall)
    result = await local.search("王小明与李小红站在塔上", min_match_score=75)
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["hit_frame_id"] == "people" and hit["seek_ms"] == 100
    assert hit["hit_frame_basis"] == "character_evidence" and hit["seek_basis"] == "character_frame"
    assert hit["character_match"]["status"] == "confirmed"
    assert hit["match_type"] == "partial"  # Identity does not establish the action/location.
    assert "visual_similarity" not in hit
    assert "hit_frame_id" not in row


def test_cached_and_streamed_best_frame_keeps_high_visual_missed_detection_and_breaks_exact_ties(local):  # noqa: F811
    rows = [window("missed", ("high-missed", "low-person"), [person("a", "region-a", "low-person")]),
            window("tie", ("tie-empty", "tie-people"), [person("a", "region-b", "tie-people")]),
            window("no-regions", ("no-region-high", "no-region-low"))]
    local.upsert(rows)
    vectors = [[1., 0.], [.2, float(np.sqrt(.96))], [1., 0.], [1., 0.], [.9, float(np.sqrt(.19))], [.1, float(np.sqrt(.99))]]
    with local._db() as db:
        db.execute("INSERT INTO visual_namespaces VALUES ('fixture','{}',2,0)")
        index = 0
        for row in rows:
            db.execute("INSERT INTO visual_records VALUES (?,?,?,?)", ("fixture", row["id"], row["asset_id"], evidence_digest(row)))
            for frame in row["ordered_frames"]:
                db.execute("INSERT INTO visual_links VALUES (?,?,?,?,?)", ("fixture", row["id"], frame["id"], frame["id"], frame["at_ms"]))
                db.execute("INSERT INTO visual_cache VALUES (?,?,?,?)", ("fixture", frame["id"], 2, np.array(vectors[index], dtype="<f4").tobytes()))
                index += 1
    vector = np.array([1., 0.], dtype="<f4")
    baseline = local.visual_index._rank("fixture", vector, 2, "1=1", [], 50)
    named = local.visual_index._rank("fixture", vector, 2, "1=1", [], 50, targets=TARGETS)
    with local._db() as db:
        streamed = local.visual_index._rank_stream(db, "fixture", vector, 2, "1=1", [], 50, targets=TARGETS)
    assert named == streamed
    selected = {row["id"]: row["hit_frame_id"] for row in named}
    assert selected == {"missed": "high-missed", "tie": "tie-people", "no-regions": "no-region-high"}
    assert next(row for row in baseline if row["id"] == "tie")["hit_frame_id"] == "tie-empty"
    assert local.visual_index.matrix_cache.stats()["builds"] == 1
    assert local.visual_index.matrix_cache.stats()["hits"] == 1
    assert rows[0]["entities"] == [person("a", "region-a", "low-person")]


def test_region_mapping_must_stay_inside_own_entity_evidence():
    entity = person("a", "a", "empty")
    entity["identity_region_evidence"] = [{"region_id": "a", "frame_id": "people"}]
    original = copy.deepcopy(entity)
    assert character_frame_support(window("w", entities=[entity]), TARGETS) == {}
    assert entity == original
