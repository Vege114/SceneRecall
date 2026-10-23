"""Portrait reuse stays fresh across active evidence and identity mutations."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scenerecall.library import atomic_json, read_json
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.scenes import SceneStore
from test_characters import cast, observation, save, seed  # noqa: F401
from test_identity_bindings import calibrated  # noqa: F401
from test_visual_identity import visual  # noqa: F401


def test_catalog_and_concurrent_portraits_reuse_projection_without_stale_walk(cast, monkeypatch):  # noqa: F811
    library, store = cast
    identity = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .5, 1]))

    def forbidden(*args, **kwargs):
        pytest.fail("cached portraits must not reread episode evidence or narrative dependencies")

    monkeypatch.setattr(SceneStore, "list", forbidden)
    first = store.catalog("asset_one")
    expected = first["profiles"][0]["representative"]
    first["profiles"][0]["appearances"].clear()
    monkeypatch.setattr(SceneStore, "_raw", forbidden)
    monkeypatch.setattr(RetrievalStore, "records", forbidden)
    monkeypatch.setattr(library, "observations", forbidden)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: store.portrait_reference("asset_one", identity), range(3)))
    assert results == [expected] * 3
    assert store.catalog("asset_one")["profiles"][0]["appearance_count"] == 1
    assert len(store.catalog("asset_one")["profiles"][0]["appearances"]) == 1


def test_new_run_active_pointer_and_current_file_replacement_refresh_next_portrait(cast):  # noqa: F811
    library, store = cast
    identity = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .5, 1]))
    assert store.portrait_reference("asset_one", identity)["box"] == [0, 0, .5, 1]
    replacement = observation(character_id=identity, portrait_frame_id="frame_one", portrait_box=[.5, 0, .5, 1])
    replacement["run_id"] = "new_run"
    library.save_observation("asset_one", replacement)
    assert store.portrait_reference("asset_one", identity)["box"] == [.5, 0, .5, 1]
    path = library.asset_dir("asset_one") / library.active("asset_one")["observations"][replacement["id"]]
    updated = read_json(path)
    updated["entities"][0]["portrait_box"] = [0, 0, 1, 1]
    atomic_json(path, updated)  # external atomic replacement, same active pointer
    assert store.portrait_reference("asset_one", identity)["box"] == [0, 0, 1, 1]
    updated["entities"][0]["character_id"] = None
    atomic_json(path, updated)
    with pytest.raises(FileNotFoundError, match="无可用图片"):
        store.portrait_reference("asset_one", identity)


def test_missing_region_restoration_reference_pixels_and_human_unbinding_invalidate(calibrated):  # noqa: F811
    library, characters, gallery, pipeline, _backend, identity, _other, references, raw = calibrated
    library.save_observation("asset_one", raw)
    expected = characters.portrait_reference("asset_one", identity)
    region_id = raw["entities"][0]["identity_region_id"]
    path = pipeline._directory("asset_one") / "regions" / f"{region_id}.json"
    region = read_json(path)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="无可用图片"):
        characters.portrait_reference("asset_one", identity)
    atomic_json(path, region)
    assert characters.portrait_reference("asset_one", identity) == expected
    # Reference hashes remain authoritative even without a gallery JSON edit.
    image = gallery._directory(characters.roster("asset_one")["scope_id"]) / f"{references[0]['id']}.png"
    pixels = image.read_bytes()
    image.write_bytes(b"externally changed reference pixels")
    with pytest.raises(FileNotFoundError, match="无可用图片"):
        characters.portrait_reference("asset_one", identity)
    image.write_bytes(pixels)
    assert characters.portrait_reference("asset_one", identity) == expected
    pipeline.review("asset_one", region_id, status="unknown", reason="Human detached current person")
    with pytest.raises(FileNotFoundError, match="无可用图片"):
        characters.portrait_reference("asset_one", identity)


def test_merge_rename_new_episode_and_scope_move_refresh_cached_appearances(cast):  # noqa: F811
    library, store = cast
    first = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .5, 1]))
    seed(library, "asset_two", episode=2)
    second = save(library, store, observation("asset_two", number=2,
                                            portrait_frame_id="frame_one", portrait_box=[.5, 0, .5, 1]))
    original = deepcopy(store.catalog("asset_one"))
    store.merge("asset_one", second, first)
    expected = {"asset_id": "asset_two", "frame_id": "frame_one", "box": [.5, 0, .5, 1]}
    assert store.portrait_reference("asset_one", first) == expected
    assert store.portrait_reference("asset_one", second) == expected
    store.update("asset_one", first, {"name": "Renamed after merge"})
    assert store.catalog("asset_one")["profiles"][0]["name"] == "Renamed after merge"
    project = library.organization.save_project({"name": "Moved episode", "kind": "animation"})
    library.organization.assign(["asset_two"], project["id"])
    assert store.portrait_reference("asset_one", first)["asset_id"] == "asset_one"
    assert len(original["profiles"]) == 2
    seed(library, "asset_three", episode=3)
    save(library, store, observation("asset_three", character_id=first))
    assert store.portrait_reference("asset_one", first)["asset_id"] == "asset_three"


def test_pointer_switch_during_dependency_collection_does_not_cache_old_run_dependencies(cast, monkeypatch):  # noqa: F811
    library, store = cast
    identity = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .5, 1]))
    root = library.asset_dir("asset_one")
    active = library.active("asset_one")
    replacement = read_json(root / next(iter(active["observations"].values())))
    replacement["run_id"] = "concurrent_run"
    replacement["entities"][0]["portrait_box"] = [.5, 0, .5, 1]
    relative = f"analyses/concurrent_run/observations/{replacement['id']}.json"
    atomic_json(root / relative, replacement)
    original_active = library.active
    switched = False

    def concurrent_active(asset_id):
        nonlocal switched
        old = original_active(asset_id)
        if not switched:
            switched = True
            current = deepcopy(old)
            current["observations"] = {replacement["id"]: relative}
            atomic_json(root / "active-analysis.json", current)
        return old

    monkeypatch.setattr(library, "active", concurrent_active)
    assert store.portrait_reference("asset_one", identity)["box"] == [.5, 0, .5, 1]
    replacement["entities"][0]["portrait_box"] = [0, 0, 1, 1]
    atomic_json(root / relative, replacement)
    assert store.portrait_reference("asset_one", identity)["box"] == [0, 0, 1, 1]


def test_active_parent_symlink_retarget_is_checked_at_original_path(cast):  # noqa: F811
    library, store = cast
    identity = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .5, 1]))
    root = library.asset_dir("asset_one")
    relative = next(iter(library.active("asset_one")["observations"].values()))
    original = root / Path(relative).parent
    first = original.with_name("first_observations")
    second = original.with_name("second_observations")
    original.rename(first)
    try:
        original.symlink_to(first, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this host")
    replacement = read_json(root / relative)
    replacement["entities"][0]["portrait_box"] = [.5, 0, .5, 1]
    atomic_json(second / Path(relative).name, replacement)
    assert store.portrait_reference("asset_one", identity)["box"] == [0, 0, .5, 1]
    original.unlink()
    original.symlink_to(second, target_is_directory=True)
    assert store.portrait_reference("asset_one", identity)["box"] == [.5, 0, .5, 1]


def test_new_deep_scene_supersedes_observation_in_cached_portrait(cast):  # noqa: F811
    library, store = cast
    identity = save(library, store, observation(portrait_frame_id="frame_one", portrait_box=[0, 0, .5, 1]))
    assert store.portrait_reference("asset_one", identity)["box"] == [0, 0, .5, 1]
    scene = observation(character_id=identity, portrait_frame_id="frame_one", portrait_box=[.5, 0, .5, 1])
    scene.update(id=SceneStore.chunk_id("asset_one", 0, 1000), run_id="deep_run", understanding_level="deep")
    relative = SceneStore._relative(scene)
    root = library.asset_dir("asset_one")
    atomic_json(root / relative, scene)
    active = library.active("asset_one")
    active["scenes"] = {scene["id"]: relative}
    atomic_json(root / "active-analysis.json", active)
    assert store.portrait_reference("asset_one", identity)["box"] == [.5, 0, .5, 1]
    assert store.catalog("asset_one")["profiles"][0]["appearance_count"] == 1
