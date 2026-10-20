"""Local identity jobs and candidate names must remain separate from VLM claims."""
import asyncio

from scenerecall.retrieval_store import STRATEGY, RetrievalStore
from scenerecall.visual_identity import VisualIdentityPipeline
from test_retrieval_verification import VerificationTransport, verification_input  # noqa: F401
from test_scene_api import scene_app  # noqa: F401
from test_search import local, record  # noqa: F401
from test_visual_identity import visual  # noqa: F401


def test_identity_backfill_api_is_local_only_idempotent_and_rejects_model_options(scene_app):  # noqa: F811
    app, client, _provider, _scene = scene_app
    library = app.state.library
    asset_id = "scene_asset"
    frame = {"id": "frame_original", "at_ms": 0, "pts": 0, "time_base": [1, 1000]}
    window = {"id": "identity_window", "asset_id": asset_id, "source_fingerprint": "synthetic-scene",
              "strategy": STRATEGY, "evidence_type": "retrieval_window", "kind": "visual",
              "start_ms": 0, "end_ms": 6000, "summary": "", "text": "", "entities": [], "events": [],
              "frame_ids": [frame["id"]], "local_frame_ids": [frame["id"]],
              "evidence_frame_ids": [frame["id"]], "ordered_frames": [frame]}
    RetrievalStore(library).publish(asset_id, "old_core_run", window)
    VisualIdentityPipeline(library).configure(asset_id, {"enabled": True})
    endpoint = f"/api/assets/{asset_id}/retrieval/identity"
    assert client.post(endpoint, json={"model_overrides": {}}).status_code == 422
    response = client.post(endpoint, json={})
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "queued" and job["request_count"] == 0
    assert job["retrieval_mode"] == "core" and job["retrieval_contract_version"] == 2
    assert job["retrieval_operation"] == "identity_only" and job["task_models"] == {}
    assert client.post(endpoint, json={}).json()["id"] == job["id"]
    assert len(app.state.jobs.list()) == 1
    frozen = app.state.jobs.get(job["id"])["config"]
    assert "encoder_identity" not in frozen and frozen["profile_snapshots"] == {}


def test_identity_backfill_requires_existing_windows_and_enabled_local_recognizer(scene_app):  # noqa: F811
    _app, client, _provider, _scene = scene_app
    response = client.post("/api/assets/scene_asset/retrieval/identity", json={})
    assert response.status_code == 400 and "画面检索索引" in response.text


def test_candidate_alias_search_is_partial_and_excluded_from_confirmed_character_filter(local):  # noqa: F811
    entity = {"id": "region", "character_id": None, "candidate_character_id": "role_a",
              "candidate_character_name": "星见纯那", "candidate_character_aliases": ["純那"],
              "binding_status": "candidate", "description": "本地人物区域"}
    row = record("candidate", "", kind="visual", entities=[entity])
    asyncio.run(local.rebuild([row]))
    for name in ("星见纯那", "純那"):
        results = asyncio.run(local.search(name))["results"]
        assert results[0]["id"] == "candidate" and results[0]["match_type"] == "partial"
        assert results[0]["unverified_character_names"] == [name]
        assert asyncio.run(local.search(name, {"character": name}))["results"] == []
    assert asyncio.run(local.search("纯那", {"character_id": "role_a"}))["results"] == []


def test_visual_verification_cannot_promote_a_candidate_name_to_confirmed_identity(verification_input):  # noqa: F811
    from scenerecall.retrieval_verification import verify_candidates

    library, response = verification_input
    response["results"][0]["entities"] = [{"id": "p", "character_id": None,
        "candidate_character_id": "role_a", "candidate_character_name": "星见纯那",
        "candidate_character_aliases": [], "binding_status": "candidate"}]
    provider = VerificationTransport()
    result = asyncio.run(verify_candidates(library, provider, "vision", "星见纯那", response))
    candidate = next(row for row in result["results"] if row["id"] == "unit_0")
    assert candidate["match_type"] == "partial" and candidate["unverified_character_names"] == ["星见纯那"]
    assert candidate["entities"][0]["character_id"] is None
