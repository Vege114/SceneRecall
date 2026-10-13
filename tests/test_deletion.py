"""Persistent archive/restore preserves evidence, identity and queue boundaries."""
import asyncio
import copy
import zipfile

import pytest
from fastapi.testclient import TestClient

from scenerecall.characters import CharacterStore
from scenerecall.library import DeletedAssetError, Library, atomic_json, read_json
from scenerecall.main import create_app
from scenerecall.models import AssetInput
from scenerecall.organization import OrganizationConflict
from test_organization import appearance, seed


@pytest.fixture
def library(tmp_path):
    result = Library(tmp_path / "library")
    seed(result)
    return result


def test_asset_archive_keeps_immutable_evidence_favorites_and_scope(library):
    store = CharacterStore(library)
    first = appearance(library, store)
    seed(library, "asset_two", episode=2)
    appearance(library, store, "asset_two", character_id=first)
    library.annotate("asset_one", {"record_id": "obs_asset_one_1", "favorite": True, "note": "收藏笔记"})
    original = {path: path.read_bytes() for path in library.asset_dir("asset_one").rglob("*") if path.is_file()}
    before = library.get_asset("asset_one")
    result = library.deletions.delete(kind="assets", asset_ids=["asset_one"])
    assert [item["id"] for item in library.list_assets()] == ["asset_two"]
    assert library.collections() == []
    assert len(store.context("asset_two")["profiles"]) == 1
    assert all(ref["asset_id"] != "asset_one" for ref in store.catalog("asset_two")["profiles"][0]["appearances"])
    assert all(path.read_bytes() == data for path, data in original.items())
    store.validate_all()
    library.deletions.restore(result["trash"]["id"])
    assert library.collections()[0]["note"] == "收藏笔记"
    after = library.get_asset("asset_one")
    assert after["character_scope_id"] == before["character_scope_id"]
    assert after["membership_revision"] > before["membership_revision"]
    assert len(store.catalog("asset_two")["profiles"]) == 1
    store.validate_all()


def test_nested_folder_content_archive_and_stable_id_restore(library):
    organization = library.organization
    parent = organization.save_folder({"name": "总分类"})
    child = organization.save_folder({"name": "子分类", "parent_id": parent["id"]})
    project = organization.projects()[0]
    organization.save_project({"folder_id": child["id"]}, project["id"])
    appearance(library, CharacterStore(library))
    preview = library.deletions.preview(kind="folder", id=parent["id"], delete_contents=True)
    assert (preview["asset_count"], preview["project_count"], preview["folder_count"]) == (1, 1, 2)
    result = library.deletions.delete(kind="folder", id=parent["id"], delete_contents=True, expected_token=preview["token"])
    assert organization.folders() == organization.projects() == library.list_assets() == []
    library.deletions.restore(result["trash"]["id"])
    assert library.get_asset("asset_one")["project_id"] == project["id"]
    assert library.get_asset("asset_one")["folder_ids"] == [child["id"], parent["id"]]
    CharacterStore(library).validate_all()


def test_category_only_delete_restore_and_conflict_are_non_destructive(library):
    organization = library.organization
    folder = organization.save_folder({"name": "分类"})
    child = organization.save_folder({"name": "子分类", "parent_id": folder["id"]})
    result = library.deletions.delete(kind="folder", id=folder["id"])
    assert organization.folders()[0]["parent_id"] is None
    organization.save_folder({"name": "用户新名字"}, child["id"])
    with pytest.raises(OrganizationConflict, match="修改"):
        library.deletions.restore(result["trash"]["id"])
    assert organization.folders()[0]["name"] == "用户新名字"
    assert len(library.deletions.list()) == 1
    project = organization.projects()[0]
    appearance(library, CharacterStore(library))
    archived = library.deletions.delete(kind="project", id=project["id"])
    assert library.get_asset("asset_one")["project_id"] is None
    library.deletions.restore(archived["trash"]["id"])
    assert library.get_asset("asset_one")["project_id"] == project["id"]
    CharacterStore(library).validate_all()


def test_project_restore_does_not_overwrite_new_analysis_or_membership(library):
    store = CharacterStore(library)
    appearance(library, store)
    project = library.organization.projects()[0]
    archived = library.deletions.delete(kind="project", id=project["id"])
    appearance(library, store, number=2)
    before = library.organization.path.read_bytes()
    with pytest.raises(OrganizationConflict, match="更新"):
        library.deletions.restore(archived["trash"]["id"])
    assert library.organization.path.read_bytes() == before
    assert len(library.observations("asset_one")) == 2
    store.validate_all()


def test_preview_scope_change_is_rejected_and_new_asset_survives(library):
    project = library.organization.projects()[0]
    request = {"kind": "project", "id": project["id"], "delete_contents": True}
    preview = library.deletions.preview(**request)
    seed(library, "asset_two", episode=2)
    with pytest.raises(OrganizationConflict, match="变化"):
        library.deletions.delete(**request, expected_token=preview["token"])
    assert len(library.list_assets()) == 2 and not library.deletions.list()


def test_deleted_import_identity_is_not_duplicated(library):
    manifest_path = library.asset_dir("asset_one") / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["import_item_key"] = "stable_key"
    atomic_json(manifest_path, manifest)
    library.deletions.delete(kind="assets", asset_ids=["asset_one"])
    with pytest.raises(DeletedAssetError, match="回收站"):
        library.register(AssetInput(video_path="/not-accessed.mp4"), import_item_key="stable_key")
    assert len(library.list_assets(include_deleted=True)) == 1


def test_delete_after_image_recovers_before_old_queued_job_can_run(tmp_path, monkeypatch):
    app = create_app(tmp_path / "library", start_worker=False, token="test")
    library, queue = app.state.library, app.state.jobs
    seed(library)
    job = queue.add("proxy", {}, "asset_one")
    def interrupted(_jobs):
        raise OSError("synthetic crash before queue update")
    monkeypatch.setattr(library.deletions, "_invalidate_jobs", interrupted)
    with pytest.raises(OSError):
        library.deletions.delete(kind="assets", asset_ids=["asset_one"])
    assert queue.get(job["id"])["status"] == "queued"
    asyncio.run(queue.execute(job["id"]))
    assert queue.get(job["id"])["status"] == "cancelled"
    trash_id = library.deletions.list()[0]["id"]
    # Restoring metadata must not make pre-deletion tasks executable again.
    monkeypatch.setattr(library.deletions, "_invalidate_jobs", lambda jobs: None)
    library.deletions.restore(trash_id)
    with pytest.raises(ValueError, match="旧任务"):
        queue.control(job["id"], "resume")


def test_deleted_backup_roundtrip_includes_archive_and_roles(library, tmp_path):
    appearance(library, CharacterStore(library))
    library.annotate("asset_one", {"record_id": "obs_asset_one_1", "favorite": True})
    deleted = library.deletions.delete(kind="assets", asset_ids=["asset_one"])
    backup = library.backup()
    archive_path = library.root / "runtime" / "backups" / f"{backup['id']}.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert all(not name.startswith("private/") for name in archive.namelist())
    restored = Library(tmp_path / "restored")
    restored.restore(archive_path)
    assert restored.list_assets() == []
    assert restored.deletions.list()[0]["id"] == deleted["trash"]["id"]
    restored.deletions.restore(deleted["trash"]["id"])
    assert restored.collections()[0]["favorite"]
    CharacterStore(restored).validate_all()


def test_trash_validation_rejects_broken_recovery_graph(library):
    library.deletions.delete(kind="assets", asset_ids=["asset_one"])
    valid = library.organization._read()
    for mutate in (
        lambda catalog: catalog.pop("trash"),
        lambda catalog: catalog["trash"].clear(),
        lambda catalog: next(iter(catalog["trash"].values()))["patches"]["assets"].clear(),
        lambda catalog: next(iter(catalog["trash"].values()))["patches"]["assets"]["asset_one"]["before"].update(project_id="missing"),
    ):
        corrupt = copy.deepcopy(valid)
        mutate(corrupt)
        with pytest.raises(ValueError):
            library.organization.validate(corrupt)
    library.organization.validate(valid)


def test_deleted_public_links_search_and_annotations_are_hidden(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="test")
    library = app.state.library
    seed(library)
    appearance(library, app.state.characters)
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "test"})
        assert client.post("/api/search", json={"query": "杯子"}).json()["results"]
        deleted = client.delete("/api/assets/asset_one").json()
        for path in ("/api/assets/asset_one", "/api/assets/asset_one/frames/frame_test", "/api/assets/asset_one/media",
                     "/api/assets/asset_one/characters"):
            assert client.get(path).status_code == 404
        assert client.post("/api/assets/asset_one/annotations", json={"record_id": "obs_asset_one_1", "favorite": True}).status_code == 404
        assert client.post("/api/search", json={"query": "杯子"}).json()["results"] == []
        assert app.state.search.status()["records"] == 0
        assert client.post(f"/api/trash/{deleted['trash']['id']}/restore").status_code == 200
        assert client.post("/api/search", json={"query": "杯子"}).json()["results"]


def test_new_legacy_import_after_project_deletion_gets_fresh_identity_scope(library):
    original = library.get_asset("asset_one")
    appearance(library, CharacterStore(library))
    library.deletions.delete(kind="project", id=original["project_id"], delete_contents=True)
    imported = seed(library, "asset_two", episode=2)
    assert imported["project_id"] != original["project_id"]
    assert imported["character_scope_id"] != original["character_scope_id"]
    assert CharacterStore(library).catalog("asset_two")["profiles"] == []
    CharacterStore(library).validate_all()
