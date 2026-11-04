"""Threshold selection covers complete local recall, with one explicit score scale."""
from types import SimpleNamespace

import pytest

from scenerecall.search import SearchEngine
from test_query_matrix_cache import seeded  # noqa: F401
from test_search import local, record, vector_provider  # noqa: F401


@pytest.mark.asyncio
async def test_threshold_lexical_is_complete_strict_and_legacy_remains_bounded(local):  # noqa: F811
    local.upsert([record(f"v{number:03}", "相同杯子", kind="visual") for number in range(90)])
    full = await local.search("杯子", min_match_score=0, limit=1)
    assert len(full["results"]) == 90
    assert full["selection"] == {"mode": "threshold", "min_match_score": 0,
        "score_basis": "relative_rrf_v1", "candidate_count": 90, "returned_count": 90}
    assert full["results"][0]["match_score"] == 100
    assert full["results"][0]["score"] == pytest.approx(1 / 61)
    boundary = full["results"][60]["match_score"]
    selected = await local.search("杯子", min_match_score=boundary)
    assert [row["id"] for row in selected["results"]] == [row["id"] for row in full["results"][:60]]
    assert (await local.search("杯子", min_match_score=100))["results"] == []
    legacy = await local.search("杯子")
    assert len(legacy["results"]) == 10 and "selection" not in legacy
    assert "match_score" not in legacy["results"][0]
    assert len((await local.search("杯子", limit=50))["results"]) == 50
    # Private recall control is not accepted from caller-supplied filter keys.
    assert len((await local.search("杯子", {"_all_candidates": True}, limit=50))["results"]) == 50
    assert local._where({"_all_candidates": True}) == ("1=1", [])


@pytest.mark.asyncio
async def test_75_percent_can_return_over_50_after_complete_multibranch_fusion(local, monkeypatch):  # noqa: F811
    rows = [record(f"v{number:03}", "杯子", kind="visual") for number in range(100)]
    local.upsert(rows)
    local.visual_encoder = object()

    async def visual(_query, filters, _degraded):
        assert filters["_all_candidates"] is True
        return list(reversed(rows))

    monkeypatch.setattr(local, "_visual", visual)
    result = await local.search("杯子", min_match_score=75)
    assert len(result["results"]) == 100
    assert result["selection"]["candidate_count"] == 100
    assert result["results"] == sorted(result["results"], key=lambda row: (-row["match_score"], row["id"]))
    assert result["results"][0]["id"] == "v000"
    assert all(row["match_score"] > 75 for row in result["results"])
    assert any(row["id"] == "v080" for row in result["results"])


@pytest.mark.asyncio
async def test_real_semantic_threshold_reads_all_valid_scoped_vectors(tmp_path):
    providers, _calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    rows = [record(f"v{number:03}", "杯子", kind="visual") for number in range(80)]
    rows.append(record("other", "杯子", kind="visual", asset_id="film-two"))
    await engine.rebuild(rows, "embed")
    result = await engine.search("喝水", {"asset_id": "film-one"}, embedding_profile_id="embed", min_match_score=0)
    assert [row["id"] for row in result["results"]] == [row["id"] for row in rows[:-1]]
    assert result["selection"]["candidate_count"] == 80
    engine.upsert([record("v079", "已经改为飞鸟", kind="visual")])
    stale = await engine.search("喝水", {"asset_id": "film-one"}, embedding_profile_id="embed", min_match_score=0)
    assert len(stale["results"]) == 79 and all(row["id"] != "v079" for row in stale["results"])
    assert any("旧向量已跳过" in warning for warning in stale["degraded"])


@pytest.mark.asyncio
async def test_visual_threshold_cached_and_stream_paths_return_all_positive_rows(seeded, monkeypatch):  # noqa: F811
    engine, _rows, vectors = seeded
    index = engine.visual_index
    monkeypatch.setattr(index, "namespace", lambda _encoder: ("fixture", {}))

    async def encode(_texts):
        return [vectors[0]]

    encoder = SimpleNamespace(encode_texts=encode)
    cached = await index.search("synthetic", {"_all_candidates": True}, encoder, [])
    assert len(cached) > 50
    with engine._db() as db:
        streamed = index._rank_stream(db, "fixture", vectors[0], 64, "1=1", [], None)
    assert cached == streamed
    assert len(await index.search("synthetic", {}, encoder, [])) == 50


@pytest.mark.asyncio
@pytest.mark.parametrize("expand", [False, True])
async def test_model_budget_does_not_drop_unjudged_or_replace_relative_scores(local, monkeypatch, expand):  # noqa: F811
    local.upsert([record(f"v{number:03}", "杯子", kind="visual") for number in range(90)])
    baseline = await local.search("杯子", min_match_score=0)
    calls, composed = [], []

    async def decide(_profile, _query, candidates):
        calls.append(candidates)
        return SimpleNamespace(data={"candidates": [
            {"id": candidates[0]["id"], "score": .99, "match_type": "none"},
            {"id": candidates[1]["id"], "score": .01, "match_type": "partial"},
        ], "expand_search": expand})

    async def compose(_profile, _query, candidates):
        composed.extend(candidates)
        return SimpleNamespace(data={"text": "合成测试摘要"})

    monkeypatch.setattr(local.providers, "decide", decide)
    monkeypatch.setattr(local.providers, "compose", compose)
    result = await local.search("杯子", min_match_score=0, decision_profile_id="judge", answer_profile_id="answer")
    rejected = {batch[0]["id"] for batch in calls}
    expected = [row for row in baseline["results"] if row["id"] not in rejected]
    assert [row["id"] for row in result["results"]] == [row["id"] for row in expected]
    assert [(row["score"], row["match_score"]) for row in result["results"]] == [
        (row["score"], row["match_score"]) for row in expected]
    assert len(calls) == (2 if expand else 1) and all(len(batch) == 20 for batch in calls)
    assert result["results"][-1]["id"] == "v089"
    assert result["results"][0]["decision_score"] == .01
    assert len(composed) == 10 and "仅覆盖排名前 10" in result["explanation"]
    assert result["selection"]["candidate_count"] == 90


@pytest.mark.asyncio
async def test_visual_merge_keeps_highest_score_and_full_candidate_denominator(local, monkeypatch):  # noqa: F811
    rows = [record(f"v{number}", "", kind="visual", start=number * 6000, end=number * 6000 + 4000,
                   evidence_type="retrieval_window", source_fingerprint="source", frame_ids=["frame"],
                   hit_frame_id="frame", evidence_frame_ids=["frame"]) for number in range(2)]
    # Adjacent windows sharing one frame remain one result with the leading score.
    rows[0]["end_ms"] = 6000

    async def recall(*_args, **_kwargs):
        return [([], []), ([], []), (rows, []), ([], [])]

    monkeypatch.setattr(local, "_recall_parallel", recall)
    result = await local.search("画面", min_match_score=0)
    assert len(result["results"]) == 1
    assert result["results"][0]["match_score"] == 100
    assert result["results"][0]["member_ids"] == ["v0", "v1"]
    assert result["selection"]["candidate_count"] == 2
    assert result["selection"]["returned_count"] == 1
    strict = await local.search("画面", min_match_score=99)
    assert strict["results"][0].get("member_ids", ["v0"]) == ["v0"]
    assert strict["results"][0]["end_ms"] == 6000


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ["threshold", "decision"])
async def test_context_is_not_suppressed_by_an_ineligible_source(local, monkeypatch, rejection):  # noqa: F811
    source = record("source", "杯子")
    context = record("context", "杯子后续", is_context=True, source_ids=["source"])

    async def recall(*_args, **_kwargs):
        return [([source, context], []), ([], []), ([context] if rejection == "threshold" else [], []), ([], [])]

    async def decide(*_args):
        return SimpleNamespace(data={"candidates": [{"id": "source", "score": 0, "match_type": "none"}]})

    monkeypatch.setattr(local, "_recall_parallel", recall)
    monkeypatch.setattr(local.providers, "decide", decide)
    result = await local.search("杯子", min_match_score=75 if rejection == "threshold" else 0,
                                decision_profile_id="judge" if rejection == "decision" else None)
    assert [row["id"] for row in result["results"]] == ["context"]


@pytest.mark.asyncio
async def test_concept_callback_all_candidates_survive_outer_fence(local):  # noqa: F811
    rows = [record(f"v{number:03}", "概念出场", kind="visual", concept_bindings=[
        {"concept_id": "target", "entity_id": "entity", "status": "confirmed"}]) for number in range(75)]
    local.upsert(rows)

    def recall(_query, filters):
        assert filters["_all_candidates"] is True
        return [{**row, "match_basis": "setting_concept_appearance", "matched_concept_ids": ["target"]}
                for row in rows]

    local.concept_lookup = recall
    result = await local.search("异质请求", min_match_score=0)
    assert len(result["results"]) == 75
    assert all(row["match_type"] == "partial" for row in result["results"])


@pytest.mark.asyncio
async def test_interpreted_lexical_union_is_not_capped(local, monkeypatch):  # noqa: F811
    local.upsert([record(f"v{number:03}", "杯子", kind="visual") for number in range(80)])

    async def interpret(_profile, _query):
        return SimpleNamespace(data={"search_text": "杯子", "filters": {}})

    monkeypatch.setattr(local.providers, "interpret", interpret)
    result = await local.search("喝水", min_match_score=0, query_profile_id="query")
    assert len(result["results"]) == 80


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [-1, 101, float("nan"), float("inf")])
async def test_threshold_rejects_invalid_scores_before_recall(local, threshold):  # noqa: F811
    with pytest.raises(ValueError, match="阈值"):
        await local.search("杯子", min_match_score=threshold)
