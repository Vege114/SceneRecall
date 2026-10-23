"""Projection reads scale with operations rather than individual entities."""
import copy
from pathlib import Path

import pytest

from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.characters import CharacterStore
from scenerecall.library import Library, atomic_json
from scenerecall.scenes import SceneStore
from test_organization import bangumi_archive, seed


@pytest.fixture
def large_catalog(tmp_path):
    library = Library(tmp_path / "library")
    for number in range(1, 13):
        seed(library, f"asset_{number}", title="规模测试动画", episode=number)
    archive = bangumi_archive()
    archive["subject"]["summary"] = "规模回归的公开动画简介。" * 8000
    AnimeProfileStore(library).apply_asset("asset_1", {"mode": "bangumi", "subject_id": 100}, archive)
    library.deletions.delete(kind="assets", asset_ids=["asset_12"])
    assert library.organization.path.stat().st_size > 1024 * 1024
    store = CharacterStore(library)
    role = store.catalog("asset_1")["profiles"][0]
    return library, store, role


def records(role, count):
    return [{"id": f"obs_{number}", "asset_id": "asset_1", "run_id": "run_test",
             "start_ms": 0, "end_ms": 1000, "summary": "人物站立", "evidence_frame_ids": ["frame_test"],
             "entities": [{"id": f"entity_{index}", "character_id": role["id"], "name": "旧名字",
                           "evidence_frame_ids": ["frame_test"]} for index in range(2)]}
            for number in range(count)]


def count_catalog_reads(monkeypatch, library):
    reads = []
    read_text = Path.read_text

    def counted(path, *args, **kwargs):
        if path == library.organization.path:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    return reads


def test_decorating_more_entities_reads_the_large_catalog_once(large_catalog, monkeypatch):
    library, store, role = large_catalog
    reads = count_catalog_reads(monkeypatch, library)
    for count in (10, 1000):
        reads.clear()
        source = records(role, count)
        projected = store.decorate_observations("asset_1", source)
        assert len(reads) == 1
        assert len(projected) == count
        assert all(entity["character_name"] == role["name"] and entity["aliases"] == role["aliases"]
                   for row in projected for entity in row["entities"])
        assert source[0]["entities"][0]["name"] == "旧名字"


def test_cast_appearance_projection_does_not_reread_catalog_per_entity(large_catalog, monkeypatch):
    library, store, role = large_catalog
    reads = count_catalog_reads(monkeypatch, library)
    counts = []
    for count in (10, 1000):
        source = records(role, count)
        monkeypatch.setattr(library, "observations", lambda identity: source if identity == "asset_1" else [])
        # The injected observation collection changes without real files;
        # advance its source pointer just as publishing a real run would.
        atomic_json(library.asset_dir("asset_1") / "active-analysis.json",
                    {"observations": {}, "scenes": {}, "fixture_revision": count})
        reads.clear()
        cast = store.catalog("asset_1")
        counts.append(len(reads))
        projected = next(profile for profile in cast["profiles"] if profile["id"] == role["id"])
        assert projected["appearance_count"] == count * 2
    assert counts[0] == counts[1]
    assert counts[0] <= 12  # one cast snapshot plus one scene check per active asset


def test_asset_list_has_one_snapshot_and_sees_the_next_external_atomic_update(large_catalog, monkeypatch):
    library, _store, _role = large_catalog
    catalog = library.organization._read()
    project = next(iter(catalog["projects"].values()))
    updated = copy.deepcopy(catalog)
    updated["projects"][project["id"]]["name"] = "外部更新的项目名称"
    read_text = Path.read_text
    reads = []

    def replace_after_read(path, *args, **kwargs):
        value = read_text(path, *args, **kwargs)
        if path == library.organization.path:
            reads.append(path)
            if len(reads) == 1:
                atomic_json(path, updated)
        return value

    monkeypatch.setattr(Path, "read_text", replace_after_read)
    before = library.list_assets()
    assert len(reads) == 1
    assert {asset["project_name"] for asset in before} == {project["name"]}
    after = library.list_assets()
    assert len(reads) == 2
    assert {asset["project_name"] for asset in after} == {"外部更新的项目名称"}


def test_asset_list_recovers_a_pending_catalog_transaction_before_snapshot(large_catalog):
    library, _store, _role = large_catalog
    catalog = library.organization._read()
    project = next(iter(catalog["projects"].values()))
    project["name"] = "事务恢复后的名称"
    atomic_json(library.organization.journal, {"schema_version": "1.0", "id": "organization_tx_test",
                                             "writes": {"organization/catalog.json": catalog}})
    assert {asset["project_name"] for asset in library.list_assets()} == {"事务恢复后的名称"}
    assert not library.organization.journal.exists()


def test_long_scene_provenance_chain_reads_metadata_per_asset_not_reference(large_catalog, monkeypatch):
    library, _characters, _role = large_catalog
    scenes = SceneStore(library)
    asset_id = "asset_1"
    base = scenes.validate(asset_id, {"run_id": "run_scale", "start_ms": 0, "end_ms": 50,
                                     "summary": "合成场景", "dependencies": scenes.dependencies(asset_id)})
    previous, pointers = [], {}
    for number in range(200):
        start = number * 50
        row = {**base, "id": scenes.chunk_id(asset_id, start, start + 50),
               "chunk_id": scenes.chunk_id(asset_id, start, start + 50),
               "start_ms": start, "end_ms": start + 50, "summary": f"合成场景 {number}",
               "memory_before": {"provenance": [{"asset_id": asset_id, "chunk_id": source["id"],
                                                  "revision_digest": scenes.revision_digest(source)}
                                                 for source in previous[-12:]]}}
        relative = scenes._relative(row)
        atomic_json(library.asset_dir(asset_id) / relative, row)
        pointers[row["id"]] = relative
        previous.append(row)
    reads = count_catalog_reads(monkeypatch, library)
    counts = []
    for count in (10, 200):
        atomic_json(library.asset_dir(asset_id) / "active-analysis.json",
                    {"schema_version": "1.0", "observations": {}, "subtitle_runs": {},
                     "scenes": dict(list(pointers.items())[:count])})
        reads.clear()
        projected = scenes.list(asset_id)
        counts.append(len(reads))
        assert len(projected) == count
        assert not any(row["context_stale"] for row in projected)
    assert counts == [3, 3]


def test_cached_deleted_scene_source_still_invalidates_continuous_memory(large_catalog):
    library, _characters, _role = large_catalog
    scenes = SceneStore(library)
    source = scenes.publish("asset_1", "run_scale_source", [
        {"start_ms": 0, "end_ms": 10000, "summary": "前集场景"}], 0, 10000)[0]
    scenes.publish("asset_2", "run_scale_target", [
        {"start_ms": 0, "end_ms": 10000, "summary": "后集场景", "memory_before": {"provenance": [
            {"asset_id": "asset_1", "chunk_id": source["id"], "revision_digest": scenes.revision_digest(source)}]}}],
        0, 10000)
    assert not scenes.list("asset_2")[0]["context_stale"]
    library.deletions.delete(kind="assets", asset_ids=["asset_1"])
    result = scenes.list("asset_2")[0]
    assert result["context_stale"] and "前序场景上下文已不可用" in result["stale_reasons"]


def test_empty_annotation_list_does_not_project_unrelated_scene_dependencies(large_catalog, monkeypatch):
    library, _characters, _role = large_catalog
    reads = count_catalog_reads(monkeypatch, library)
    assert library.annotations("asset_1") == []
    assert reads == []
