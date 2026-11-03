# ruff: noqa: F811
"""Bangumi knowledge preparation through real import jobs and HTTP boundaries."""
import copy

from test_bangumi_workflow import archive, client, execute, import_request, video  # noqa: F401

from scenerecall.anime_profiles import AnimeProfileStore


def test_video_import_prepares_sources_and_candidate_cards_without_building(client, video):
    job = client.post("/api/imports", json=import_request(
        video, selection={"mode": "bangumi", "subject_id": 42})).json()
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    item = done["items"][0]
    receipt = item["knowledge_import"]
    assert receipt["source_count"] == 2 and receipt["created_cards"] >= 1
    scope = client.get(f"/api/assets/{item['asset_id']}/work-entity-scope").json()["work_scope_id"]
    base = f"/api/work-entity-scopes/{scope}/knowledge"
    sources = client.get(base + "/sources").json()["sources"]
    assert {row["url"] for row in sources} == {"https://bgm.tv/subject/42", "https://bgm.tv/character/7"}
    cards = client.get(base + "/cards").json()["cards"]
    assert cards and all(row["effective_status"] == "candidate" for row in cards)
    assert client.app.state.knowledge.projection(scope)["cards"] == []
    assert client.app.state.work_entities.scope(item["asset_id"])["bindings"] == {}
    assert not any(job["type"] == "knowledge_build" for job in client.app.state.jobs.list())


def test_inherited_project_archive_also_prepares_knowledge(client, video):
    project = client.post("/api/projects", json={"name": "合成动画", "kind": "animation"}).json()
    client.app.state.bangumi.apply_project(project["id"], {"mode": "bangumi", "subject_id": 42}, archive())
    job = client.post("/api/imports", json=import_request(video, project_id=project["id"])).json()
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    assert done["items"][0]["knowledge_import"]["created_sources"] == 2


def test_existing_archive_can_prepare_and_repeat_locally_without_network(client, video, monkeypatch):
    original = AnimeProfileStore._prepare_knowledge
    monkeypatch.setattr(AnimeProfileStore, "_prepare_knowledge", lambda *_: None)
    job = client.post("/api/imports", json=import_request(
        video, selection={"mode": "bangumi", "subject_id": 42})).json()
    done = execute(client, job["id"])
    asset_id = done["items"][0]["asset_id"]
    assert client.app.state.work_entities.scope(asset_id) is None
    monkeypatch.setattr(AnimeProfileStore, "_prepare_knowledge", original)
    calls = copy.deepcopy(client.app.state.bangumi.client.calls)
    client.app.state.bangumi.client.fail = True
    prepared = client.post(f"/api/assets/{asset_id}/knowledge/bangumi", json={})
    assert prepared.status_code == 200, prepared.text
    first = prepared.json()
    assert first["created_sources"] == 2 and first["created_cards"] >= 1
    base = f"/api/work-entity-scopes/{first['work_scope_id']}/knowledge"
    again = client.post(base + "/bangumi", json={"asset_id": asset_id})
    assert again.status_code == 200, again.text
    assert again.json()["created_sources"] == again.json()["created_cards"] == 0
    assert client.app.state.bangumi.client.calls == calls
    assert client.post(base + "/bangumi", json={"asset_id": "missing"}).status_code == 404
    foreign = client.app.state.work_entities.create_scope("另一作品", "TV", [])
    conflict = client.post(f"/api/work-entity-scopes/{foreign['work_scope_id']}/knowledge/bangumi",
                           json={"asset_id": asset_id})
    assert conflict.status_code == 400, conflict.text
    assert client.app.state.work_entities.scope(asset_id)["work_scope_id"] == first["work_scope_id"]


def test_knowledge_preparation_error_does_not_undo_video_and_cast_import(client, video, monkeypatch):
    def failed(*_, **__):
        raise ValueError("synthetic storage failure")

    monkeypatch.setattr("scenerecall.bangumi_knowledge.sync_bangumi_knowledge", failed)
    job = client.post("/api/imports", json=import_request(
        video, selection={"mode": "bangumi", "subject_id": 42})).json()
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    item = done["items"][0]
    assert item["knowledge_import"]["status"] == "error"
    assert any("知识资料整理失败" in warning for warning in item["warnings"])
    assert client.app.state.characters.catalog(item["asset_id"])["profiles"]


def test_scratch_archive_does_not_generate_external_knowledge(client, video):
    job = client.post("/api/imports", json=import_request(video, selection={"mode": "scratch"})).json()
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    item = done["items"][0]
    assert "knowledge_import" not in item
    assert client.app.state.work_entities.scope(item["asset_id"]) is None


def test_bangumi_knowledge_preparation_requires_session(client):
    client.cookies.clear()
    assert client.post("/api/assets/missing/knowledge/bangumi", json={}).status_code == 401
