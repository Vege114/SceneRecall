"""A bounded selection publishes once per source and recovers each item independently."""
from copy import deepcopy
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.visual_identity_api import register_visual_identity_routes
from test_character_confirmation import candidate_args, prepared
from test_organization import bangumi_archive
from test_visual_identity import visual as visual


def selection(service):
    return [candidate_args(row) for row in service.queue("asset_one")["roles"][0]["candidates"]]


def batch(service, identity, candidates):
    return service.confirm_batch("asset_one", identity, candidates=candidates,
                                 single_character=True, use_as_reference=True)


def decisions(service):
    scope = service.library.get_asset("asset_one")["character_scope_id"]
    return service._read(scope)["decisions"]


def second_source(service):
    library = service.library
    atomic_json(library.asset_dir("asset_two") / "manifest.json", {
        **library.get_asset("asset_one"), "id": "asset_two", "episode": 2})
    path = library.asset_dir("asset_two") / "frames" / "frame_second.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(library.frame_path("asset_one", "frame_0").read_bytes())
    frame = library.register_frame("asset_two", {"id": "frame_second", "path": str(path), "at_ms": 200})
    region = service.pipeline.run("asset_two", [frame], {"enabled": True})["regions"][0]
    return {"source_asset_id": "asset_two", "region_id": region["id"]}


def test_three_selected_crops_publish_once_and_retry_without_duplicate_writes(visual, monkeypatch):
    service, identity = prepared(visual)
    candidates = selection(service)
    before_refs = len(service.gallery.trusted("asset_one"))
    calls = []
    record_calls = []
    original_records = service.library.records

    def records(asset_id):
        record_calls.append(asset_id)
        return original_records(asset_id)

    def sync(rows, asset_id):
        calls.append((rows, asset_id))
        assert {row["action"] for row in decisions(service).values()} == {"confirming"}

    monkeypatch.setattr(service.library, "records", records)
    service.search = SimpleNamespace(sync_asset=sync)
    result = batch(service, identity, candidates)
    assert result["summary"] == {"requested": 3, "confirmed": 3, "failed": 0}
    assert [{key: row[key] for key in ("source_asset_id", "region_id")} for row in result["results"]] == candidates
    assert record_calls == ["asset_one"] and calls == [([], "asset_one")]
    assert len(service.gallery.trusted("asset_one")) == before_refs + 3
    assert sum(row["match"]["status"] == "confirmed" for row in service.pipeline.reviews("asset_one")["regions"]) == 3
    assert {row["action"] for row in decisions(service).values()} == {"confirm"}
    before_review = read_json(service.pipeline._directory("asset_one") / "decisions.json")
    before_state = deepcopy(decisions(service))
    service.search = SimpleNamespace(sync_asset=lambda rows, asset: calls.append((rows, asset)))
    assert batch(service, identity, candidates)["summary"] == result["summary"]
    assert len(service.gallery.trusted("asset_one")) == before_refs + 3
    assert read_json(service.pipeline._directory("asset_one") / "decisions.json") == before_review
    assert decisions(service) == before_state


@pytest.mark.parametrize("write_review", [False, True])
def test_item_review_failure_preserves_other_successes_and_is_independently_retryable(visual, monkeypatch, write_review):
    service, identity = prepared(visual)
    candidates = selection(service)
    failed_id = candidates[1]["region_id"]
    original_review = service.pipeline.review
    calls = []

    def fails(asset_id, region_id, **kwargs):
        if region_id != failed_id or write_review:
            result = original_review(asset_id, region_id, **kwargs)
        if region_id == failed_id:
            raise OSError("review interrupted")
        return result

    monkeypatch.setattr(service.pipeline, "review", fails)
    service.search = SimpleNamespace(sync_asset=lambda rows, asset: calls.append(asset))
    result = batch(service, identity, candidates)
    assert result["summary"] == {"requested": 3, "confirmed": 2, "failed": 1}
    assert [row["status"] for row in result["results"]] == ["confirmed", "failed", "confirmed"]
    assert result["results"][1]["recovery_required"] and "review interrupted" in result["results"][1]["error"]
    assert calls == ["asset_one"]
    assert sorted(row["action"] for row in decisions(service).values()) == ["confirm", "confirm", "confirming"]
    assert len(service.gallery.trusted("asset_one")) == 4
    monkeypatch.setattr(service.pipeline, "review", original_review)
    assert batch(service, identity, [candidates[1]])["summary"]["confirmed"] == 1
    assert len(service.gallery.trusted("asset_one")) == 4
    assert read_json(service.pipeline._directory("asset_one") / "decisions.json")["revision"] == 3


def test_shared_search_failure_keeps_all_journals_pending_and_retry_reuses_reviews(visual):
    service, identity = prepared(visual)
    candidates = selection(service)

    def fails(*args):
        raise OSError("search interrupted")

    service.search = SimpleNamespace(sync_asset=fails)
    result = batch(service, identity, candidates)
    assert result["summary"] == {"requested": 3, "confirmed": 0, "failed": 3}
    assert all(row["recovery_required"] and "search interrupted" in row["error"] for row in result["results"])
    assert {row["action"] for row in decisions(service).values()} == {"confirming"}
    shown = service.queue("asset_one")["roles"][0]["candidates"]
    assert {row["region_id"] for row in shown} == {row["region_id"] for row in candidates}
    assert all(row["recovery_required"] for row in shown)
    before = read_json(service.pipeline._directory("asset_one") / "decisions.json")
    service.search = None
    assert batch(service, identity, candidates)["summary"]["confirmed"] == 3
    assert len(service.gallery.trusted("asset_one")) == 4
    assert read_json(service.pipeline._directory("asset_one") / "decisions.json") == before


def test_final_journal_failure_does_not_erase_other_successes(visual, monkeypatch):
    service, identity = prepared(visual)
    candidates = selection(service)
    failed_id = candidates[1]["region_id"]
    original_save = service._save

    def fails(scope, state, key, decision):
        if decision["action"] == "confirm" and decision["region_id"] == failed_id:
            raise OSError("finalize interrupted")
        return original_save(scope, state, key, decision)

    monkeypatch.setattr(service, "_save", fails)
    result = batch(service, identity, candidates)
    assert [row["status"] for row in result["results"]] == ["confirmed", "failed", "confirmed"]
    assert result["results"][1]["recovery_required"]
    assert {row["region_id"]: row["action"] for row in decisions(service).values()} == {
        row["region_id"]: "confirming" if row["region_id"] == failed_id else "confirm" for row in candidates}
    monkeypatch.setattr(service, "_save", original_save)
    assert batch(service, identity, [candidates[1]])["summary"]["confirmed"] == 1
    assert len(service.gallery.trusted("asset_one")) == 4


def test_retry_of_pending_confirmation_never_overwrites_a_later_manual_edit(visual):
    service, identity = prepared(visual)
    candidate = selection(service)[0]

    def fails(*args):
        raise OSError("search interrupted")

    service.search = SimpleNamespace(sync_asset=fails)
    assert batch(service, identity, [candidate])["results"][0]["recovery_required"]
    service.pipeline.review("asset_one", candidate["region_id"], status="unknown", reason="later human correction")
    before = deepcopy(decisions(service))
    before_review = read_json(service.pipeline._directory("asset_one") / "decisions.json")
    calls = []
    service.search = SimpleNamespace(sync_asset=lambda rows, asset: calls.append(asset))
    result = batch(service, identity, [candidate])
    assert result["summary"] == {"requested": 1, "confirmed": 0, "failed": 1}
    assert result["results"][0]["recovery_required"] and "人工修改" in result["results"][0]["error"]
    assert calls == []  # A blocked retry cannot publish or invalidate this source.
    assert decisions(service) == before
    assert read_json(service.pipeline._directory("asset_one") / "decisions.json") == before_review


def test_selected_human_candidate_is_explicitly_upgraded_to_confirmation(visual):
    service, identity = prepared(visual)
    candidate = selection(service)[0]
    service.pipeline.review("asset_one", candidate["region_id"], status="candidate", character_id=identity,
                            reason="human tentative identity")
    assert batch(service, identity, [candidate])["summary"]["confirmed"] == 1
    region = next(row for row in service.pipeline.reviews("asset_one")["regions"] if row["id"] == candidate["region_id"])
    assert region["match"]["status"] == "confirmed" and region["match"]["source"] == "human"


def test_source_search_failure_is_isolated_and_each_source_is_published_once(visual):
    service, identity = prepared(visual)
    candidates = selection(service)[:2] + [second_source(service)]
    calls = []

    def sync(rows, asset_id):
        calls.append(asset_id)
        if asset_id == "asset_two":
            raise OSError("second source interrupted")

    service.search = SimpleNamespace(sync_asset=sync)
    result = batch(service, identity, candidates)
    assert calls == ["asset_one", "asset_two"]
    assert [row["status"] for row in result["results"]] == ["confirmed", "confirmed", "failed"]
    assert result["results"][2]["recovery_required"]
    service.search = None
    assert batch(service, identity, [candidates[2]])["summary"]["confirmed"] == 1
    assert len(service.gallery.trusted("asset_one")) == 4


def test_batch_checks_each_source_cast_policy_and_protects_later_manual_corrections(visual):
    service, identity = prepared(visual)
    candidates = selection(service)[:2]
    other_source = second_source(service)
    organization = service.library.organization._read()
    organization["assets"]["asset_two"]["anime_profile"] = {"mode": "bangumi", "dossier": bangumi_archive(200)}
    atomic_json(service.library.organization.path, organization)
    result = batch(service, identity, [candidates[0], other_source])
    assert [row["status"] for row in result["results"]] == ["confirmed", "failed"]
    assert "同一作品名单" in result["results"][1]["error"]
    assert not result["results"][1]["recovery_required"]
    assert not (service.pipeline._directory("asset_two") / "decisions.json").exists()
    assert len(decisions(service)) == 1
    service.pipeline.review("asset_one", candidates[0]["region_id"], status="unknown", reason="later human correction")
    before = read_json(service.pipeline._directory("asset_one") / "decisions.json")
    result = batch(service, identity, candidates)
    assert [row["status"] for row in result["results"]] == ["failed", "confirmed"]
    assert "人工修改" in result["results"][0]["error"]
    latest = read_json(service.pipeline._directory("asset_one") / "decisions.json")
    assert latest["decisions"][candidates[0]["region_id"]] == before["decisions"][candidates[0]["region_id"]]


@pytest.mark.asyncio
async def test_batch_api_returns_partial_results_and_builds_queue_once(visual, monkeypatch):
    service, identity = prepared(visual)
    candidates = selection(service)
    candidates[1] = {"source_asset_id": "asset_one", "region_id": "region_missing"}
    app = FastAPI()
    register_visual_identity_routes(app, service.library, None)
    api_service = app.state.character_confirmation
    queue = api_service.queue
    calls = []

    def counted(asset_id):
        calls.append(asset_id)
        return queue(asset_id)

    monkeypatch.setattr(api_service, "queue", counted)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(f"/api/assets/asset_one/identity/confirmation/characters/{identity}/confirm-batch", json={
            "candidates": candidates, "single_character": True, "use_as_reference": True})
    assert response.status_code == 200
    result = response.json()
    assert result["summary"] == {"requested": 3, "confirmed": 2, "failed": 1}
    assert [row["status"] for row in result["results"]] == ["confirmed", "failed", "confirmed"]
    assert result["queue"]["summary"]["confirmed"] == 1
    assert calls == ["asset_one"]


@pytest.mark.asyncio
async def test_batch_api_rejects_invalid_selections_before_any_write(visual):
    service, identity = prepared(visual)
    candidates = selection(service)
    app = FastAPI()
    register_visual_identity_routes(app, service.library, None)
    base = {"candidates": candidates, "single_character": True, "use_as_reference": True}
    invalid = [
        {**base, "candidates": []},
        {**base, "candidates": candidates + [{"source_asset_id": "asset_one", "region_id": "region_four"}]},
        {**base, "candidates": [candidates[0], candidates[0]]},
        {**base, "candidates": [{"source_asset_id": "../escape", "region_id": "region_one"}]},
        {**base, "single_character": False}, {**base, "use_as_reference": False},
        {**base, "single_character": "true"}, {**base, "use_as_reference": 1},
        {key: value for key, value in base.items() if key != "use_as_reference"},
        {**base, "reason": "x" * 4001}, {**base, "extra": True},
    ]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for payload in invalid:
            response = await client.post(f"/api/assets/asset_one/identity/confirmation/characters/{identity}/confirm-batch", json=payload)
            assert response.status_code == 422, response.text
    assert decisions(service) == {}
    assert len(service.gallery.trusted("asset_one")) == 1
    assert not (service.pipeline._directory("asset_one") / "decisions.json").exists()
