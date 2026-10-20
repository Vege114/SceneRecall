import copy
import io
import json
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

from scenerecall.library import Library, atomic_json, read_json
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.work_entities import WorkEntityStore
from scenerecall.work_entity_api import register_work_entity_routes


def image_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (96, 64), color=(90, 30, 150)).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def work(tmp_path):
    library = Library(tmp_path / "library")
    for identity in ("episode-one", "episode-two", "other-work"):
        atomic_json(library.asset_dir(identity) / "manifest.json", {
            "schema_version": "1.0", "id": identity, "work_id": "legacy-" + identity,
            "title": identity, "kind": "animation", "created_at": "2026-10-20T00:00:00Z",
            "duration_ms": 5000, "version": "test", "fingerprint": "source-" + identity})
    store = WorkEntityStore(library)
    store.create_scope("测试作品", "TV", ["episode-one", "episode-two"])
    store.create_scope("另一部作品", "movie", ["other-work"])
    return library, store


def prop(name="时间道具"):
    return {"type": "prop", "name": name, "original_name": "時の道具", "aliases": ["秒表"],
            "appearance": "金色圆形表盘", "sources": [{"id": "manual", "kind": "manual", "title": "人工整理"}],
            "facts": [{"id": "stop", "kind": "function", "text": "能停止时间", "source_ids": ["manual"]}]}


def observation(library, asset_id="episode-one"):
    frame_id = "registered-frame"
    path = library.asset_dir(asset_id) / "frames" / (frame_id + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes())
    frame = {"id": frame_id, "path": str(path), "at_ms": 200, "pts": 200, "time_base": [1, 1000]}
    library.register_frame(asset_id, frame)
    record = {"schema_version": "1.0", "id": "window-one", "asset_id": asset_id,
              "source_fingerprint": "source-" + asset_id, "strategy": "retrieval-first-v1",
              "kind": "visual", "evidence_type": "retrieval_window", "text": "圆形金色物体",
              "start_ms": 0, "end_ms": 1000, "frame_ids": [frame_id], "evidence_frame_ids": [frame_id],
              "ordered_frames": [{key: frame[key] for key in ("id", "at_ms", "pts", "time_base")}],
              "entities": [{"id": "local-object", "name": "圆形物体", "description": "金色表盘"}]}
    RetrievalStore(library).publish(asset_id, "test-run", record)
    return record, frame


def binding(concept, status="confirmed"):
    return {"record_id": "window-one", "entity_id": "local-object", "concept_id": concept["concept_id"],
            "status": status, "evidence_frame_ids": ["registered-frame"], "reason": "人工核对可见外观"}


def test_scopes_are_explicit_stable_and_cross_episode_reusable(work):
    library, store = work
    one = store.scope("episode-one")["work_scope_id"]
    assert one == store.scope("episode-two")["work_scope_id"]
    assert one != store.scope("other-work")["work_scope_id"]
    assert not one.endswith("episode-one")
    concept = store.create("episode-one", prop())
    assert store.get("episode-two", concept["concept_id"])["name"] == "时间道具"
    project = library.organization.save_project({"name": "随意整理的项目", "kind": "animation"})
    library.organization.assign(["episode-one"], project["id"])
    assert store.scope("episode-one")["work_scope_id"] == one
    assert store.get("episode-one", concept["concept_id"])["concept_id"] == concept["concept_id"]
    with pytest.raises(ValueError, match="另一作品"):
        store.attach("episode-one", store.scope("other-work")["work_scope_id"])


def test_cross_work_id_relations_and_binding_are_rejected(work):
    library, store = work
    local = store.create("episode-one", prop())
    foreign = store.create("other-work", prop("同名道具"))
    with pytest.raises(FileNotFoundError):
        store.get("episode-one", foreign["concept_id"])
    data = prop("机体")
    data["relations"] = [{"predicate": "内含", "target_concept_id": foreign["concept_id"], "source_ids": ["manual"]}]
    with pytest.raises(ValueError, match="同一作品"):
        store.create("episode-one", data)
    observation(library)
    with pytest.raises(ValueError, match="不属于当前作品"):
        store.bind("episode-one", binding(foreign))
    assert not store.occurrences("episode-one", local["concept_id"])


def test_setting_function_never_asserts_current_event_and_only_confirmed_occurrences_return(work):
    library, store = work
    raw, _ = observation(library)
    original = copy.deepcopy(raw)
    concept = store.create("episode-one", prop())
    context = store.context("episode-one", "停止时间")
    assert context["profiles"][0]["concept_id"] == concept["concept_id"]
    assert context["facts_are_not_current_events"]
    assert store.occurrences("episode-one", concept["concept_id"]) == []
    store.bind("episode-one", binding(concept, "candidate"))
    assert store.occurrences("episode-one", concept["concept_id"]) == []
    store.bind("episode-one", binding(concept))
    rows = store.occurrences("episode-one", concept["concept_id"])
    assert len(rows) == 1 and rows[0]["event_verified"] is False
    assert "停止时间" not in store.project_record("episode-one", raw)["text"]
    assert raw == original
    store.bind("episode-one", binding(concept, "revoked"))
    assert store.occurrences("episode-one", concept["concept_id"]) == []


def test_rename_updates_projection_and_revoke_preserves_immutable_observation(work):
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    store.update("episode-one", concept["concept_id"], {"name": "时停计时器", "aliases": ["计时器"]}, reason="校对译名")
    projected = store.project_record("episode-one", raw)
    assert projected["concept_bindings"][0]["concept_name"] == "时停计时器"
    assert "时停计时器" not in raw["text"]
    stored = RetrievalStore(library).records("episode-one")[0]
    assert stored["text"] == raw["text"]
    store.revoke("episode-one", concept["concept_id"], "识别有误")
    assert not store.project_record("episode-one", raw)["concept_bindings"]
    assert len(store.scope("episode-one")["audit"]) >= 4


def test_bindings_require_real_frame_and_local_entity_evidence(work):
    library, store = work
    observation(library)
    concept = store.create("episode-one", prop())
    with pytest.raises(ValueError, match="证据必须属于"):
        store.bind("episode-one", {**binding(concept), "evidence_frame_ids": ["unseen-frame"]})
    with pytest.raises(ValueError, match="局部实体"):
        store.bind("episode-one", {**binding(concept), "entity_id": "nonexistent"})
    with pytest.raises(ValueError):
        store.bind("episode-one", {**binding(concept), "status": "auto_bound", "confidence": 0.99})


def test_stale_observation_source_cannot_keep_confirmed_binding(work):
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    changed = {**raw, "end_ms": 2000}
    assert not store.project_record("episode-one", changed)["concept_bindings"]


def test_references_use_safe_pixels_category_views_and_inert_attribution_urls(work, monkeypatch):
    library, store = work
    _, frame = observation(library)
    concept = store.create("episode-one", prop())
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: pytest.fail("attribution must not fetch"))
    ref = store.add_reference("episode-one", concept["concept_id"], image_bytes(),
                              confirmed=True, source_url="https://example.org/reference.png")
    assert ref["reference_only"] and ref["status"] == "confirmed" and "path" not in ref
    scope = store.scope("episode-one")["work_scope_id"]
    assert store.reference_path(scope, concept["concept_id"], ref["id"]).is_file()
    crop = store.reference_from_frame("episode-one", concept["concept_id"], frame["id"],
                                     box=[0, 0, 0.5, 1], confirmed=True)
    assert crop["view"] == "object_region" and crop["source"]["frame_id"] == frame["id"]
    place = store.create("episode-one", {"type": "place", "name": "驾驶舱"})
    with pytest.raises(ValueError, match="地点参考"):
        store.reference_from_frame("episode-one", place["concept_id"], frame["id"], box=[0, 0, 0.5, 1])
    store.reference_from_frame("episode-one", place["concept_id"], frame["id"], view="whole_scene")
    store.validate_all()


def test_context_is_bounded_to_five_profiles_and_eight_confirmed_images(work):
    _, store = work
    for index in range(7):
        concept = store.create("episode-one", prop(f"时间道具{index}"))
        for _ in range(3):
            store.add_reference("episode-one", concept["concept_id"], image_bytes(), confirmed=True)
    context = store.context("episode-one", "时间道具")
    assert len(context["profiles"]) == 5
    assert sum(len(row["references"]) for row in context["profiles"]) == 8


def package(payload, files=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("manifest.json", json.dumps(payload, ensure_ascii=False))
        for name, data in (files or {}).items():
            archive.writestr(name, data)
    return stream.getvalue()


def test_package_import_is_bounded_local_atomic_and_never_promotes_references(work):
    _, store = work
    data = {**prop(), "reference_files": [{"file": "refs/clock.png", "confirmed": True,
                                          "source_url": "https://example.org/clock.png"}]}
    imported = store.import_package("episode-one", package({"entities": [data]}, {"refs/clock.png": image_bytes()}))
    assert len(imported) == 1 and imported[0]["references"][0]["status"] == "candidate"
    ref = imported[0]["references"][0]
    store.update_reference("episode-one", imported[0]["concept_id"], ref["id"], status="confirmed", reason="人工核对图片")
    assert store.get("episode-one", imported[0]["concept_id"])["references"][0]["status"] == "confirmed"
    before = store.list("episode-one")
    with pytest.raises(ValueError):
        store.import_package("episode-one", package({"entities": [prop("有效"), {"type": "character", "name": "冒充角色"}]}))
    assert store.list("episode-one") == before
    with pytest.raises(ValueError, match="越界路径"):
        store.import_package("episode-one", package({"entities": [prop()]}, {"../escape.png": image_bytes()}))
    store.validate_all()


def test_restore_validation_rejects_reference_path_escape(work):
    _, store = work
    concept = store.create("episode-one", prop())
    store.add_reference("episode-one", concept["concept_id"], image_bytes())
    scope = store.scope("episode-one")["work_scope_id"]
    saved = read_json(store._path(scope))
    saved["entities"][concept["concept_id"]]["references"][0]["path"] = "../../escape.png"
    atomic_json(store._path(scope), saved)
    with pytest.raises(ValueError, match="越界"):
        store.validate_all()


@pytest.mark.asyncio
async def test_confirmed_concept_names_are_searchable_without_asserting_setting_functions(work, tmp_path):
    from scenerecall.providers import ProviderManager
    from scenerecall.search import SearchEngine
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    search = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))
    search.upsert([store.project_record("episode-one", raw)])
    assert (await search.search("秒表"))["results"][0]["id"] == raw["id"]
    assert not (await search.search("停止"))["results"]


def test_batch_projection_reads_memberships_scopes_and_assets_once_and_skips_subtitles(work, monkeypatch):
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    calls = {"memberships": 0, "catalogs": 0, "assets": 0}
    read_memberships, read_catalog, read_asset = store._memberships, store._read, library.get_asset

    def memberships():
        calls["memberships"] += 1
        return read_memberships()

    def catalog(scope):
        calls["catalogs"] += 1
        return read_catalog(scope)

    def asset(*args, **kwargs):
        calls["assets"] += 1
        return read_asset(*args, **kwargs)

    monkeypatch.setattr(store, "_memberships", memberships)
    monkeypatch.setattr(store, "_read", catalog)
    monkeypatch.setattr(library, "get_asset", asset)
    subtitles = [{"id": f"cue-{index}", "asset_id": "episode-one", "kind": "subtitle", "text": "字幕"}
                 for index in range(100)]
    rows = store.project_records([raw, {**raw, "id": "second-window", "asset_id": "episode-two"}, *subtitles])
    assert rows[0]["concept_bindings"][0]["concept_id"] == concept["concept_id"]
    assert rows[2:] == subtitles
    assert calls == {"memberships": 1, "catalogs": 1, "assets": 2}


def test_concept_occurrence_lookup_never_reads_library_records_and_preserves_event_boundary(work, monkeypatch):
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    monkeypatch.setattr(library, "records", lambda *args, **kwargs: pytest.fail("recursive evidence lookup"))
    assert store.search_occurrences("停止时间") == []
    result = store.search_occurrences("正在停止时间", eligible_records=[raw])
    assert len(result) == 1
    assert result[0]["event_verified"] is False and result[0]["match_type"] == "partial"
    assert result[0]["match_basis"] == "setting_concept_appearance" and result[0]["retrieval_branches"] == ["concept"]
    assert "events" not in result[0] and result[0]["text"] == raw["text"]
    assert not store.search_occurrences("时间", {"asset_id": "other-work"}, eligible_records=[raw])
    store.bind = lambda *args: pytest.fail("lookup must not change bindings")
    assert store.search_occurrences("秒表", eligible_records=[raw])


@pytest.mark.asyncio
async def test_concept_recall_is_fourth_partial_branch_with_stable_id_filters(work, tmp_path):
    from scenerecall.providers import ProviderManager
    from scenerecall.search import SearchEngine
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    row = store.project_record("episode-one", raw)
    row["entities"][0]["character_id"] = "stable-character"
    search = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))
    search.upsert([row])
    queries = []

    def lookup(query, filters):
        queries.append(query)
        return store.search_occurrences(query, filters, eligible_records=search.filtered_records(filters))

    search.concept_lookup = lookup
    query = "正在停止时间而非读取时间"
    results = await search.search(query, {"concept_id": concept["concept_id"], "character_id": "stable-character"})
    assert queries == [query] and len(results["results"]) == 1
    hit = results["results"][0]
    assert hit["match_basis"] == "setting_concept_appearance" and hit["match_type"] == "partial"
    assert hit["event_verified"] is False and "concept" in hit["retrieval_branches"]
    assert "未证明当前功能或动作" in hit["match_reason"]
    assert not (await search.search("时间", {"concept_id": "foreign-concept"}))["results"]
    assert not (await search.search("时间", {"character_id": "foreign-character"}))["results"]


@pytest.mark.asyncio
async def test_concept_callback_cannot_escape_sql_filter_or_republish_removed_record(work, tmp_path):
    from scenerecall.providers import ProviderManager
    from scenerecall.search import SearchEngine
    library, store = work
    raw, _ = observation(library)
    concept = store.create("episode-one", prop())
    store.bind("episode-one", binding(concept))
    row = store.project_record("episode-one", raw)
    candidates = store.search_occurrences("停止时间", eligible_records=[row])
    search = SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))
    search.upsert([row])
    search.concept_lookup = lambda query, filters: candidates
    assert not (await search.search("停止", {"asset_id": "other-work"}))["results"]

    def remove_then_return(query, filters):
        search.remove_assets(["episode-one"])
        return candidates

    search.concept_lookup = remove_then_return
    assert not (await search.search("停止"))["results"]


def test_api_strict_schema_crud_and_scope_registration(work):
    library, _ = work
    app = FastAPI()

    @app.exception_handler(ValueError)
    async def invalid(_request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    register_work_entity_routes(app, library)
    with TestClient(app) as client:
        url = "/api/assets/episode-one/work-entities"
        response = client.post(url, json=prop())
        assert response.status_code == 200
        concept_id = response.json()["concept_id"]
        assert client.post(url, json={**prop(), "character_id": "not-a-character"}).status_code == 422
        edit = client.patch(f"{url}/{concept_id}", json={"changes": {"aliases": ["新别名"]}, "reason": "补充别名"})
        assert edit.status_code == 200 and edit.json()["aliases"] == ["新别名"]
        assert client.get("/api/assets/episode-two/work-entities").json()["entities"][0]["concept_id"] == concept_id
