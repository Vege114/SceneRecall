from scenerecall.search import SearchSnapshotChanged
from test_scene_api import scene_app  # noqa: F401


def test_http_only_reads_indexed_candidate_owners(scene_app, monkeypatch):  # noqa: F811
    app, client, _provider, _scene = scene_app
    library, search = app.state.library, app.state.search
    original = library.records
    rows = original("scene_asset")
    search.sync_asset(rows, "scene_asset")
    calls = []

    def records(asset_id=None, **kwargs):
        calls.append(asset_id)
        assert asset_id == "scene_asset", "HTTP search must not project the full library"
        return original(asset_id, **kwargs)

    async def recalled(*args, **kwargs):
        return {"results": [{**rows[0], "asset_id": "forged-owner", "score": 1}],
                "explanation": "", "degraded": []}

    monkeypatch.setattr(library, "records", records)
    monkeypatch.setattr(search, "search", recalled)
    response = client.post("/api/search", json={"query": "画面"})
    assert response.status_code == 200
    assert response.json()["results"][0]["asset_id"] == "scene_asset"
    assert calls == ["scene_asset"]
    assert app.state.search_http_diagnostics["canonical_ms"] >= 0


def test_http_changed_snapshot_has_retryable_status(scene_app, monkeypatch):  # noqa: F811
    app, client, _provider, _scene = scene_app

    async def changed(*args, **kwargs):
        raise SearchSnapshotChanged("资料正在更新，请重试搜索")

    monkeypatch.setattr(app.state.search, "search", changed)
    response = client.post("/api/search", json={"query": "画面"})
    assert response.status_code == 409
    assert "重试" in response.json()["detail"]


def test_http_empty_candidates_skip_canonical_reads(scene_app, monkeypatch):  # noqa: F811
    app, client, _provider, _scene = scene_app

    async def empty(*args, **kwargs):
        return {"results": [], "explanation": "", "degraded": []}

    def forbidden(*args, **kwargs):
        raise AssertionError("no canonical records needed")

    monkeypatch.setattr(app.state.search, "search", empty)
    monkeypatch.setattr(app.state.library, "records", forbidden)
    assert client.post("/api/search", json={"query": "不存在"}).json()["results"] == []


def test_http_rechecks_revision_after_authoritative_projection(scene_app, monkeypatch):  # noqa: F811
    app, client, _provider, _scene = scene_app
    library, search = app.state.library, app.state.search
    original_records, original_search = library.records, search.search
    rows = original_records("scene_asset")
    search.sync_asset(rows, "scene_asset")

    async def recalled(*args, **kwargs):
        await original_search("画面")
        return {"results": [rows[0]], "explanation": "", "degraded": []}

    def changed(asset_id=None, **kwargs):
        result = original_records(asset_id, **kwargs)
        search.remove_assets(["scene_asset"])
        return result

    monkeypatch.setattr(search, "search", recalled)
    monkeypatch.setattr(library, "records", changed)
    assert client.post("/api/search", json={"query": "画面"}).status_code == 409
