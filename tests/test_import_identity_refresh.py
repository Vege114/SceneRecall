"""Import roster propagation keeps both retrieval paths aligned with live groups."""
import asyncio

import pytest
from fastapi.testclient import TestClient

from scenerecall.library import atomic_json, atomic_jsonl, digest, read_json
from scenerecall.main import create_app
from scenerecall.models import ImportInput
from test_organization import bangumi_archive, seed
from test_search import vector_provider


@pytest.mark.parametrize("assigned", [True, False])
def test_import_binding_refreshes_earlier_episodes_without_rebuilding_vectors(tmp_path, assigned):
    providers, calls = vector_provider(tmp_path)
    app = create_app(tmp_path / "library", providers=providers, start_worker=False, token="identity-test")
    library, queue, search = app.state.library, app.state.jobs, app.state.search
    series = "系列" if assigned else ""
    first = seed(library, title="同一动画", series=series, episode=1)
    path = library.asset_dir(first["id"]) / "manifest.json"
    atomic_json(path, {**read_json(path), "subtitle_mode": "external"})
    atomic_jsonl(library.asset_dir(first["id"]) / "subtitles" / "external.jsonl", [
        {"schema_version": "1.0", "id": "cue_first", "asset_id": first["id"], "start_ms": 0,
         "end_ms": 900, "text": "人物拿着杯子", "source": "external"}])
    immutable = {path: path.read_bytes() for path in library.asset_dir(first["id"]).rglob("*") if path.is_file()}
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "identity-test"})
        asyncio.run(search.rebuild(library.records(), "embed"))
        original_namespace = search.status()["embedding_namespaces"]
        request = ImportInput.model_validate({"request_id": "bind-next-episode", "project_id": first["project_id"],
                                             "items": [{"item_id": "two", "video_path": "not-accessed.mp4",
                                                        "title": "同一动画", "episode": 2,
                                                        "anime_profile": {"mode": "bangumi", "subject_id": 100}}]})
        job = queue.add_import(request, anime_dossiers={"two": bangumi_archive()})
        # Recover a committed second episode, exercising the same import binding
        # and publication path without manufacturing another media fixture.
        second = seed(library, "asset_two", title="同一动画", series=series, episode=2)
        path = library.asset_dir(second["id"]) / "manifest.json"
        atomic_json(path, {**read_json(path), "import_item_key": digest([job["id"], "two"])})
        asyncio.run(queue.execute(job["id"]))
        assert queue.get(job["id"])["status"] == "completed"
        live = library.get_asset(first["id"])
        assert live["anime_id"] != first["anime_id"]
        assert live["anime_id"] == library.get_asset(second["id"])["anime_id"]
        assert search.status()["embedding_namespaces"] == original_namespace
        assert len(calls) == 1, "Metadata refresh must reuse existing vectors"
        for query, profile in (("杯子", None), ("喝水", "embed")):
            result = asyncio.run(search.search(query, {"anime_id": live["anime_id"]}, embedding_profile_id=profile))
            assert [row["asset_id"] for row in result["results"]] == [first["id"]]
            assert {row["anime_id"] for row in result["results"]} == {live["anime_id"]}
            assert asyncio.run(search.search(query, {"anime_id": first["anime_id"]},
                                             embedding_profile_id=profile))["results"] == []
        assert all(path.read_bytes() == data for path, data in immutable.items())
