"""Cross-module retrieval regressions use only local synthetic evidence."""
import copy

import pytest
from PIL import Image

from scenerecall.providers import ProviderManager
from scenerecall.retrieval_store import RetrievalStore, STRATEGY
from scenerecall.search import SearchEngine
from scenerecall.work_entities import WorkEntityStore
from test_retrieval_complete import complete_worker, enqueue_complete  # noqa: F401
from test_scene_api import scene_app  # noqa: F401
from test_scene_jobs import scene_worker  # noqa: F401
from test_visual_retrieval import Encoder, evidence
from test_work_entities import image_bytes, prop


@pytest.mark.asyncio
async def test_silent_complete_job_selects_picture_dossiers_without_confirming_presence(complete_worker):  # noqa: F811
    worker = complete_worker()
    asset_id = worker.asset["id"]
    worker.library.save_subtitles(asset_id, "silent-test", [], 0, worker.asset["duration_ms"])
    bank = WorkEntityStore(worker.library)
    bank.create_scope("合成作品", "TV", [asset_id])
    concept = bank.create(asset_id, prop())
    reference = bank.add_reference(asset_id, concept["concept_id"], image_bytes(), confirmed=True)
    assert worker.library.subtitles(asset_id) == []
    job = enqueue_complete(worker)
    await worker.queue.execute(job["id"])
    done = worker.queue.get(job["id"])
    assert done["status"] == "completed", done
    payloads = [call["payload"] for call in worker.provider.calls if call["stage"] == "local_records"]
    assert payloads and all(not payload["subtitles"] for payload in payloads)
    assert all(concept["concept_id"] in {row["concept_id"] for row in payload["work_entities"]} for payload in payloads)
    assert worker.queue.search.visual_encoder.images.count(reference["id"]) == 1
    assert bank.scope(asset_id)["bindings"] == {}
    assert all(not row.get("concept_bindings") for row in worker.library.records(asset_id))
    assert all(not entity.get("concept_id") for row in RetrievalStore(worker.library).records(asset_id)
               for entity in row.get("entities", []))


def publish_window(library, identity, start=0, end=6000, frame_id="frame_original"):
    record = {"id": identity, "asset_id": "scene_asset", "source_fingerprint": "synthetic-scene",
              "strategy": STRATEGY, "evidence_type": "retrieval_window", "kind": "visual",
              "start_ms": start, "end_ms": end, "summary": "", "text": "", "entities": [], "events": [],
              "frame_ids": [frame_id], "evidence_frame_ids": [frame_id],
              "ordered_frames": [{"id": frame_id, "at_ms": start, "pts": start, "time_base": [1, 1000]}]}
    RetrievalStore(library).publish("scene_asset", "review-run", record)
    return record


def test_api_rejected_merge_cannot_keep_excluded_member_evidence(scene_app, monkeypatch):  # noqa: F811
    app, client, _, _ = scene_app
    library = app.state.library
    first = publish_window(library, "review-first")
    path = library.asset_dir("scene_asset") / "frames/frame_second.jpg"
    Image.new("RGB", (32, 32), "blue").save(path)
    library.register_frame("scene_asset", {"id": "frame_second", "path": str(path), "at_ms": 6000})
    second = publish_window(library, "review-second", 6000, 10000, "frame_second")

    async def stale_search(*args, **kwargs):
        return {"results": [{**first, "member_ids": [first["id"], second["id"]], "end_ms": 10000,
                             "hit_frame_id": "frame_original", "seek_ms": 9999,
                             "evidence_ids": ["frame_original", "frame_second", "invented"]}],
                "degraded": [], "explanation": ""}

    app.state.search.sync_asset(library.records("scene_asset"), "scene_asset")
    monkeypatch.setattr(app.state.search, "search", stale_search)
    result = client.post("/api/search", json={"query": "背景"}).json()["results"][0]
    assert not result.get("member_ids")
    assert result["end_ms"] == 6000 and result["seek_ms"] == 0
    assert result["evidence_ids"] == ["frame_original"]


def test_api_retains_validated_setting_appearance_basis_without_asserting_event(scene_app):  # noqa: F811
    app, client, _, _ = scene_app
    library = app.state.library
    record = publish_window(library, "review-setting")
    bank = app.state.work_entities
    bank.create_scope("合成作品", "TV", ["scene_asset"])
    concept = bank.create("scene_asset", prop())
    bank.bind("scene_asset", {"record_id": record["id"], "concept_id": concept["concept_id"], "status": "confirmed",
                             "evidence_frame_ids": ["frame_original"], "reason": "合成测试人工确认"})
    app.state.search.sync_asset(library.records("scene_asset"), "scene_asset")
    result = client.post("/api/search", json={"query": "能停止时间", "filters": {"kind": "visual"}}).json()["results"][0]
    assert result["match_basis"] == "setting_concept_appearance"
    assert result["event_verified"] is False and result["match_type"] == "partial"
    assert result["matched_concept_ids"] == [concept["concept_id"]]
    assert result["events"] == []


@pytest.mark.asyncio
async def test_visual_query_overflow_preserves_original_constraints_and_degrades_to_existing_recall(tmp_path):
    engine = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))
    query = "甲没有把红杯递给乙，" * 20 + "最后不是蓝杯，也不是丙。"
    row, frame = evidence(tmp_path, "long-query", text=query)
    original = copy.deepcopy(row)

    class LimitedEncoder(Encoder):
        async def encode_texts(self, texts):
            self.text_calls.append(texts)
            raise ValueError("查询超过编码器 token 上限；不能静默截断")

    encoder = engine.visual_encoder = LimitedEncoder()
    await engine.index_visual([row], [frame])
    result = await engine.search(query)
    assert encoder.text_calls == [[query]]
    assert result["results"][0]["id"] == row["id"] and result["results"][0]["match_type"] == "partial"
    assert "visual" not in result["results"][0]["retrieval_branches"]
    assert any("不能静默截断" in message for message in result["degraded"])
    assert row == original
