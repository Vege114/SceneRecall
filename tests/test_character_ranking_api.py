"""Current evidence must determine named-person ranking at HTTP publication."""
from copy import deepcopy

import httpx
import pytest

from scenerecall.main import create_app


TARGETS = [{"scope_id": "scope_one", "character_id": key, "name": key}
           for key in ("Aoi", "Hikari")]


def person(identity, region, frame, *, confirmed=True):
    field = "character_id" if confirmed else "candidate_character_id"
    return {"id": region, "kind": "person", "is_character": True,
            field: identity, "identity_region_id": region,
            "identity_support_region_ids": [], "evidence_frame_ids": [frame],
            "identity_region_evidence": [{"region_id": region, "frame_id": frame}],
            "identity_binding": {"id": "binding_" + region, "policy_version": "identity-binding-v2",
                                 "character_scope_id": "scope_one", field: identity,
                                 "status": "human_confirmed" if confirmed else "candidate"}}


async def publish(tmp_path, monkeypatch, current, *, contributions, reference, threshold=75, hit="frame_hit",
                  hit_basis=None, match_type="partial", explanation=None, visual_verification=False):
    app = create_app(tmp_path / "library", start_worker=False, token="character-rank-test")
    engine, library = app.state.search, app.state.library
    stale = {**deepcopy(current), "score": reference, "raw_rrf_score": .03,
             "match_score": 100., "match_score_basis": "character_grounded_rrf_v1",
             "hit_frame_id": hit, "seek_ms": 1000, "match_type": match_type,
             "_rrf_contributions": contributions, "_match_score_reference": reference,
             "character_match": {"status": "confirmed", "required_count": 2,
                                 "confirmed_count": 2, "candidate_count": 0, "frame_id": hit}}
    if hit_basis is not None:
        stale["hit_frame_basis"] = hit_basis

    async def search(*_args, **_kwargs):
        return {"results": [stale], "degraded": [], "explanation": explanation,
                "selection": {"mode": "threshold", "min_match_score": threshold,
                              "score_basis": "character_grounded_rrf_v1",
                              "candidate_count": 1, "returned_count": 1}}

    monkeypatch.setattr(engine, "search", search)
    monkeypatch.setattr(engine, "completed_character_targets", lambda: deepcopy(TARGETS))
    monkeypatch.setattr(engine, "result_asset_ids", lambda _candidates: ["asset_one"])
    monkeypatch.setattr(library, "records", lambda *_args: [deepcopy(current)])
    if visual_verification:
        async def verify(*args, **kwargs):
            response = args[-1]
            response["results"][0].update(match_type="full", match_reason="模型判为完整匹配")
            response["explanation"] = "模型声称人物已确认"
            return response
        monkeypatch.setattr("scenerecall.retrieval_verification.verify_candidates", verify)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post("/api/search", headers={"Authorization": "Bearer character-rank-test"},
                                     json={"query": "Aoi与Hikari站在塔上", "filters": {"kind": "visual"},
                                           "min_match_score": threshold, "visual_verification": visual_verification})
    assert response.status_code == 200, response.text
    return response.json()


def evidence(entities):
    return {"id": "visual_one", "asset_id": "asset_one", "kind": "visual",
            "text": "当前画面候选", "summary": "当前画面候选", "start_ms": 0, "end_ms": 3000,
            "evidence_frame_ids": ["frame_hit", "frame_other"],
            "ordered_frames": [{"id": "frame_hit", "at_ms": 1000}, {"id": "frame_other", "at_ms": 2000}],
            "entities": entities}


@pytest.mark.asyncio
async def test_current_cross_frame_names_cannot_keep_stale_bonus_above_threshold(tmp_path, monkeypatch):
    current = evidence([person("Aoi", "region_a", "frame_other"), person("Hikari", "region_b", "frame_other")])
    result = await publish(tmp_path, monkeypatch, current,
                           contributions={"visual": .01, "lexical": .02}, reference=.03)
    assert result["results"] == []
    assert result["selection"]["returned_count"] == 0
    assert result["selection"]["candidate_count"] == 1


@pytest.mark.asyncio
async def test_missing_detection_retains_visual_fallback_and_reports_unknown(tmp_path, monkeypatch):
    result = await publish(tmp_path, monkeypatch, evidence([]), contributions={"visual": .03}, reference=.03)
    row = result["results"][0]
    assert row["match_score"] == 100
    assert row["character_match"]["status"] == "unverified"
    assert row["character_match"]["confirmed_count"] == row["character_match"]["candidate_count"] == 0
    assert row["match_type"] == "partial"
    assert not any(key.startswith("_rrf") or key.startswith("_match_score") for key in row)


@pytest.mark.asyncio
async def test_current_candidates_remain_candidates_and_use_original_reference(tmp_path, monkeypatch):
    current = evidence([person("Aoi", "region_a", "frame_hit", confirmed=False),
                        person("Hikari", "region_b", "frame_hit", confirmed=False)])
    result = await publish(tmp_path, monkeypatch, current,
                           contributions={"visual": .01, "lexical": .02}, reference=.03, threshold=60)
    row = result["results"][0]
    assert row["match_score"] == pytest.approx(100 * .02 / .03)
    assert row["raw_rrf_score"] == .03
    assert row["character_match"]["status"] == "candidate"
    assert row["character_match"]["confirmed_count"] == 0
    assert row["character_match"]["candidate_count"] == 2
    assert row["character_match"]["frame_id"] == "frame_hit"
    assert row["match_type"] == "partial"


@pytest.mark.asyncio
async def test_invalid_hit_frame_cannot_inherit_current_identity_bonus(tmp_path, monkeypatch):
    current = evidence([person("Aoi", "region_a", "frame_hit"), person("Hikari", "region_b", "frame_hit")])
    result = await publish(tmp_path, monkeypatch, current,
                           contributions={"visual": .01, "lexical": .02}, reference=.03, hit="foreign_frame")
    assert result["results"] == []


@pytest.mark.asyncio
async def test_identity_only_location_is_not_published_as_visual_similarity_hit(tmp_path, monkeypatch):
    current = evidence([person("Aoi", "region_a", "frame_hit"), person("Hikari", "region_b", "frame_hit")])
    result = await publish(tmp_path, monkeypatch, current, contributions={"lexical": .03}, reference=.03,
                           hit_basis="character_evidence")
    row = result["results"][0]
    assert row["hit_frame_basis"] == "character_evidence"
    assert row["seek_basis"] == "character_frame"
    assert row["seek_ms"] == 1000
    assert row["character_match"]["status"] == "confirmed"
    assert row["match_type"] == "partial"


@pytest.mark.asyncio
async def test_current_unverified_identity_clears_stale_full_match_explanation(tmp_path, monkeypatch):
    result = await publish(tmp_path, monkeypatch, evidence([]), contributions={"visual": .03}, reference=.03,
                           match_type="full", explanation="两人已在塔上")
    assert result["results"][0]["match_type"] == "partial"
    assert result["explanation"] is None


@pytest.mark.asyncio
async def test_visual_verifier_cannot_upgrade_unverified_hit_frame_identities(tmp_path, monkeypatch):
    result = await publish(tmp_path, monkeypatch, evidence([]), contributions={"visual": .03}, reference=.03,
                           visual_verification=True)
    row = result["results"][0]
    assert row["character_match"]["status"] == "unverified"
    assert row["match_type"] == "partial"
    assert result["explanation"] is None
