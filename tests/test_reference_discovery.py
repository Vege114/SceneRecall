# ruff: noqa: F811
"""Core-only reference discovery must retain exact evidence and human authority."""
import copy
import io

import pytest
from PIL import Image

from scenerecall.jobs import JobQueue
from scenerecall.knowledge_jobs import confirm_discovery, discovery_inputs, discovery_path, discovery_view
from scenerecall.knowledge_recall import WorkEntityRecall
from scenerecall.knowledge_store import KnowledgeStore
from scenerecall.library import atomic_json, read_json
from scenerecall.providers import ProviderManager
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.search import SearchEngine
from test_visual_retrieval import Encoder
from test_work_entities import image_bytes, observation, prop, work  # noqa: F401


def different_image():
    stream = io.BytesIO()
    Image.new("RGB", (96, 64), color=(10, 140, 20)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
async def discovery(work, tmp_path):
    library, entities = work
    record, first = observation(library)
    second = {"id": "last-frame", "at_ms": 700, "pts": 700, "time_base": [1, 1000],
              "path": str(library.asset_dir("episode-one") / "frames" / "last-frame.png")}
    from pathlib import Path
    Path(second["path"]).write_bytes(different_image())
    library.register_frame("episode-one", second)
    record = copy.deepcopy(record)
    # A core window does not require local generative entities or events.
    record.update(entities=[], events=[], frame_ids=[first["id"], second["id"]],
                  evidence_frame_ids=[first["id"], second["id"]],
                  ordered_frames=[{key: frame[key] for key in ("id", "at_ms", "pts", "time_base")} for frame in (first, second)])
    RetrievalStore(library).publish("episode-one", "core-no-vlm", record)
    concept = entities.create("episode-one", prop())
    ref = entities.add_reference("episode-one", concept["concept_id"], image_bytes(), confirmed=True)
    scope = entities.scope("episode-one")["work_scope_id"]
    providers = ProviderManager(tmp_path / "providers")
    search = SearchEngine(tmp_path / "search", providers)
    encoder = Encoder(vectors={first["id"]: [0, 1, 0], second["id"]: [1, 0, 0], ref["id"]: [1, 0, 0]})
    search.visual_encoder = encoder
    await search.index_visual(library.records("episode-one"), [first, second])
    knowledge = KnowledgeStore(entities)
    queue = JobQueue(library, providers, search)
    queue.knowledge = knowledge
    return knowledge, search, queue, scope, concept, ref, first, second


async def run_discovery(fixture):
    knowledge, search, queue, scope, *_ = fixture
    job = queue.add("reference_discovery", {"work_scope_id": scope})
    queue.update(job["id"], status="running")
    await queue.discover_references(job["id"])
    return discovery_view(knowledge, search, scope), job


def test_reference_inputs_satisfy_real_encoder_metadata_contract(discovery):
    from scenerecall.visual_encoder import VisualEncoder
    knowledge, search, _, scope, *_ = discovery
    references, _, _, _ = discovery_inputs(knowledge, search, scope)
    assert references and all(row["asset_id"] == scope for row in references)
    # This validates the real adapter without loading a model or a GPU.
    assert len(VisualEncoder._image_values(references)) == len(references)


@pytest.mark.asyncio
async def test_core_discovery_reuses_film_vectors_and_reports_winning_original_frame(discovery):
    knowledge, search, queue, scope, concept, ref, first, second = discovery
    encoder = search.visual_encoder
    before = knowledge.store.scope("episode-one")["bindings"]
    result, job = await run_discovery(discovery)
    assert result["stale"] is False
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["concept_id"] == concept["concept_id"]
    assert candidate["reference_ids"] == [ref["id"]]
    assert candidate["frame_id"] == second["id"] and candidate["at_ms"] == 700
    assert candidate["record_id"] == "window-one" and candidate["asset_id"] == "episode-one"
    assert candidate["similarity"] == pytest.approx(1.)
    assert candidate["status"] == "candidate"
    assert knowledge.store.scope("episode-one")["bindings"] == before == {}
    assert encoder.image_calls == [[first["id"], second["id"]], [ref["id"]]]
    assert encoder.text_calls == []
    assert queue.get(job["id"])["request_count"] == 0
    await run_discovery(discovery)
    assert encoder.image_calls == [[first["id"], second["id"]], [ref["id"]]]


@pytest.mark.asyncio
async def test_confirmed_core_candidate_immediately_enters_knowledge_recall(discovery):
    knowledge, search, _, scope, concept, _, _, _ = discovery
    result, _ = await run_discovery(discovery)
    candidate = result["candidates"][0]
    recall = WorkEntityRecall(knowledge.store)
    assert recall.search("停止时间", {"asset_id": "episode-one"}, search) == []
    with pytest.raises(ValueError):
        confirm_discovery(knowledge, search, scope, candidate["candidate_id"], "")
    binding = confirm_discovery(knowledge, search, scope, candidate["candidate_id"], "核对原帧中的金色表盘")
    assert binding["status"] == "confirmed" and binding["decision_source"] == "human"
    assert binding["evidence_frame_ids"] == [candidate["frame_id"]]
    rows = recall.search("停止时间", {"asset_id": "episode-one"}, search)
    assert [row["id"] for row in rows] == [candidate["record_id"]]
    assert rows[0]["matched_concept_ids"] == [concept["concept_id"]]
    assert rows[0]["event_verified"] is False and rows[0]["match_type"] == "partial"
    refreshed = discovery_view(knowledge, search, scope)
    assert refreshed["stale"] is False
    assert refreshed["candidates"][0]["status"] == "confirmed"
    assert refreshed["candidates"][0]["binding_id"] == binding["id"]


@pytest.mark.parametrize("change", ["reference_revoke", "reference_pixels", "source", "encoder"])
@pytest.mark.asyncio
async def test_changed_reference_source_or_encoder_cannot_confirm_old_candidate(discovery, change):
    knowledge, search, _, scope, concept, ref, _, _ = discovery
    result, _ = await run_discovery(discovery)
    candidate = result["candidates"][0]
    if change == "reference_revoke":
        knowledge.store.update_reference("episode-one", concept["concept_id"], ref["id"], status="revoked", reason="参考错误")
    elif change == "reference_pixels":
        knowledge.store.reference_path(scope, concept["concept_id"], ref["id"]).write_bytes(different_image())
    elif change == "source":
        path = knowledge.library.asset_dir("episode-one") / "manifest.json"
        saved = read_json(path)
        atomic_json(path, {**saved, "fingerprint": "replacement-source"})
    else:
        search.visual_encoder = Encoder(model="different-encoder")
    with pytest.raises((ValueError, FileNotFoundError)):
        confirm_discovery(knowledge, search, scope, candidate["candidate_id"], "不能确认过期候选")
    assert knowledge.store.scope("episode-one")["bindings"] == {}


@pytest.mark.asyncio
async def test_reference_changed_during_encoding_does_not_publish_candidate(discovery):
    knowledge, search, _, scope, concept, ref, _, _ = discovery
    search.visual_encoder.on_images = lambda: knowledge.store.update_reference(
        "episode-one", concept["concept_id"], ref["id"], status="revoked", reason="编码时撤销")
    with pytest.raises(ValueError, match="变化"):
        await run_discovery(discovery)
    assert not discovery_path(knowledge, scope).exists()
    assert knowledge.store.scope("episode-one")["bindings"] == {}


@pytest.mark.asyncio
async def test_reference_discovery_does_not_scan_foreign_work(discovery):
    knowledge, search, _, scope, _, _, _, _ = discovery
    foreign, frame = observation(knowledge.library, "other-work")
    foreign = {**foreign, "id": "foreign-window"}
    RetrievalStore(knowledge.library).publish("other-work", "foreign-run", foreign)
    await search.index_visual(knowledge.library.records("other-work"), [frame])
    _, records, _, _ = discovery_inputs(knowledge, search, scope)
    assert {record["asset_id"] for record in records} == {"episode-one"}
    result, _ = await run_discovery(discovery)
    assert all(row["asset_id"] == "episode-one" for row in result["candidates"])


@pytest.mark.asyncio
async def test_identical_reference_pixels_keep_concept_specific_reference_ids(discovery):
    knowledge, search, _, scope, first_concept, first_ref, _, _ = discovery
    second_concept = knowledge.store.create("episode-one", prop("另一只相似道具"))
    second_ref = knowledge.store.add_reference("episode-one", second_concept["concept_id"], image_bytes(), confirmed=True)
    references, records, _, _ = discovery_inputs(knowledge, search, scope)
    result = await search.match_visual_references(records, references, detailed=True)
    winners = {row["concept_id"]: row for row in result["window-one"]}
    assert winners[first_concept["concept_id"]]["reference_ids"] == [first_ref["id"]]
    assert winners[second_concept["concept_id"]]["reference_ids"] == [second_ref["id"]]
