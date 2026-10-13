from fastapi.testclient import TestClient

from scenerecall.main import create_app
from test_organization import appearance, seed


def test_organization_crud_assignment_recursive_search_and_session(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="test-organization")
    seed(app.state.library)
    appearance(app.state.library, app.state.characters)
    with TestClient(app) as client:
        assert client.get("/api/projects").status_code == 401
        client.post("/api/session", json={"token": "test-organization"})
        parent = client.post("/api/folders", json={"name": "动画"}).json()
        child = client.post("/api/folders", json={"name": "秋季", "parent_id": parent["id"]}).json()
        project = client.post("/api/projects", json={"name": "新动画", "folder_id": child["id"], "tags": ["精彩"]}).json()
        response = client.post("/api/projects/assign", json={"asset_ids": ["asset_one"], "project_id": project["id"]})
        assert response.status_code == 200 and response.json()["moved_count"] == 1
        for filters in ({"project_id": project["id"]}, {"folder_id": parent["id"]}, {"tag": "精彩"}):
            result = client.post("/api/search", json={"query": "杯子", "filters": filters}).json()
            assert result["results"][0]["asset_id"] == "asset_one"
        assert client.post("/api/search", json={"query": "杯子", "filters": {"project_id": "other"}}).json()["results"] == []
        assert client.patch(f"/api/folders/{parent['id']}", json={"parent_id": child["id"]}).status_code == 400
        assert client.patch(f"/api/projects/{project['id']}", json={"name": "新名称"}).json()["name"] == "新名称"
        assert client.get("/api/assets/asset_one").json()["asset"]["project_name"] == "新名称"
        assert client.delete(f"/api/folders/{child['id']}").status_code == 200
        assert client.get("/api/projects").json()[-1]["folder_id"] == parent["id"]
        assert client.delete(f"/api/projects/{project['id']}").status_code == 200
        assert client.get("/api/assets/asset_one").json()["asset"]["project_id"] is None


def test_invalid_organization_payloads_do_not_mutate(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="test-organization")
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "test-organization"})
        assert client.post("/api/projects", json={"name": " "}).status_code == 400
        assert client.post("/api/projects", json={"name": "项目", "folder_id": "missing"}).status_code == 400
        assert client.post("/api/projects", json={"name": "项目", "character_scope_id": "forged"}).status_code == 422
        assert client.post("/api/folders", json={"name": "文件夹", "parent_id": "missing"}).status_code == 400
        assert client.get("/api/projects").json() == []
        assert client.get("/api/folders").json() == []
