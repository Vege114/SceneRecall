"""Concrete animations remain distinct within a series, including old trash."""
import copy

import pytest
from fastapi.testclient import TestClient

from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.characters import CharacterStore
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.main import create_app
from scenerecall.models import AssetInput, ImportInput
from scenerecall.organization import anime_metadata
from scenerecall.scenes import SceneStore
from scenerecall.search import SearchEngine
from test_organization import bangumi_archive, seed
from test_scene_storage import chunk, seed as scene_seed
from test_search import record, vector_provider


def test_concrete_work_groups_ignore_episode_and_version_and_keep_seasons_separate(tmp_path):
    library = Library(tmp_path / "library")
    first = seed(library, "asset_one", title="同一动画", episode=1)
    seed(library, "asset_two", title="同一动画", episode=2)
    path = library.asset_dir("asset_two") / "manifest.json"
    atomic_json(path, {**read_json(path), "version": "另一视频版本"})
    assert library.get_asset("asset_two")["anime_id"] == first["anime_id"]
    seed(library, "asset_season", title="同一动画", episode=1)
    path = library.asset_dir("asset_season") / "manifest.json"
    atomic_json(path, {**read_json(path), "season": 2})
    seed(library, "asset_film", title="同一动画 剧场版", episode=None)
    entries = library.organization.anime(first["project_id"])
    assert sorted(entry["asset_count"] for entry in entries) == [1, 1, 2]
    assert library.get_asset("asset_one")["anime_id"] == Library(library.root).get_asset("asset_one")["anime_id"]
    assert AssetInput(video_path="synthetic.mp4").kind == "animation"


def test_series_tv_and_film_keep_independent_rosters_and_do_not_mutate_on_view(tmp_path):
    library = Library(tmp_path / "library")
    tv = seed(library, title="TV 动画")
    seed(library, "asset_two", title="TV 动画", episode=2)
    film = seed(library, "asset_film", title="剧场版", episode=None)
    profiles = AnimeProfileStore(library)
    profiles.apply_asset(tv["id"], {"mode": "bangumi", "subject_id": 100}, bangumi_archive(100))
    film_archive = bangumi_archive(101)
    for character in film_archive["characters"]:
        character["id"] += 1000
    profiles.apply_asset(film["id"], {"mode": "bangumi", "subject_id": 101}, film_archive, for_import=True)
    first = library.get_asset(tv["id"])
    second = library.get_asset("asset_two")
    feature = library.get_asset(film["id"])
    assert first["anime_id"] == second["anime_id"] != feature["anime_id"]
    assert second["anime_profile"]["dossier"]["subject"]["id"] == 100
    assert feature["anime_profile"]["dossier"]["subject"]["id"] == 101
    characters = CharacterStore(library)
    revision = characters._read(first["character_scope_id"])["revision"]
    for identity, subject in ((tv["id"], 100), (film["id"], 101), ("asset_two", 100)):
        cast = characters.catalog(identity)
        assert {source["external_source"]["subject_id"] for source in cast["profiles"]} == {subject}
        assert characters._read(first["character_scope_id"])["revision"] == revision
    with pytest.raises(ValueError, match="具体动画"):
        profiles.apply_project(first["project_id"], {"mode": "bangumi", "subject_id": 100}, bangumi_archive(100))
    seed(library, "asset_scratch", title="另一个从零建档动画")
    assert characters.catalog("asset_scratch")["cast_policy"]["mode"] == "named_only"


def test_release_migration_groups_named_bangumi_and_old_movie_episodes_and_restores(tmp_path):
    library = Library(tmp_path / "library")
    for number in range(1, 13):
        filename = f"[DMG&MH&VCB-Studio] Synthetic Anime [{number:02d}][Ma10p_1080p][x265_flac].mkv"
        asset = {"schema_version": "1.0", "id": f"asset_{number}", "work_id": f"work_{number}",
                 "title": "手填中文动画名" if number == 1 else filename, "file_name": filename,
                 "kind": "animation" if number == 1 else "movie", "series": "", "project_id": None,
                 "season": None, "episode": None, "version": "原始版本", "duration_ms": 10000,
                 "fingerprint": f"fingerprint-{number}", "created_at": "2026-10-14T00:00:00Z"}
        atomic_json(library.asset_dir(asset["id"]) / "manifest.json", asset)
        library.get_asset(asset["id"])
    AnimeProfileStore(library).apply_asset("asset_1", {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    removed = library.organization.library.deletions.delete(kind="assets", asset_ids=[f"asset_{n}" for n in range(1, 13)])
    catalog = copy.deepcopy(library.organization._read())
    catalog.pop("anime_membership_version", None)
    # Recreate the 0.2 catalog layout and its exact recovery snapshots.
    for member in [*catalog["assets"].values(),
                   *(phase for entry in catalog["trash"].values() for patch in entry["patches"]["assets"].values()
                     for phase in patch.values())]:
        for key in list(member):
            if key.startswith("anime_") and key != "anime_profile" or key in {"kind", "season", "episode"}:
                member.pop(key)
        if (member.get("anime_profile") or {}).get("mode") == "scratch":
            member.pop("anime_profile", None)
    atomic_json(library.organization.path, catalog)
    originals = {path: path.read_bytes() for path in (library.root / "assets").glob("*/manifest.json")}
    reopened = Library(library.root)
    archived = reopened.list_assets(include_deleted=True)
    assert len({asset["anime_id"] for asset in archived}) == 1
    assert {asset["anime_title"] for asset in archived} == {"手填中文动画名"}
    assert {asset["kind"] for asset in archived} == {"animation"}
    assert {asset["episode"] for asset in archived} == set(range(1, 13))
    reopened.deletions.restore(removed["trash"]["id"])
    assert len(reopened.organization.anime()) == 1
    assert reopened.organization.anime()[0]["asset_count"] == 12
    assert all(path.read_bytes() == content for path, content in originals.items())
    # Raw movie inputs without animation evidence remain movie archives.
    assert anime_metadata({"title": "真人电影", "kind": "movie", "episode": None})["kind"] == "movie"


@pytest.mark.asyncio
async def test_anime_filters_restrict_lexical_and_semantic_recall(tmp_path):
    providers, _calls = vector_provider(tmp_path)
    engine = SearchEngine(tmp_path / "index", providers)
    await engine.rebuild([record("tv", "拿杯子", kind="visual", anime_id="anime_tv", project_id="project_series"),
                          record("film", "拿杯子", kind="visual", anime_id="anime_film", project_id="project_series")], "embed")
    for query in ("杯子", "喝水"):
        result = await engine.search(query, {"anime_id": "anime_tv"}, embedding_profile_id="embed")
        assert [row["id"] for row in result["results"]] == ["tv"]
    assert (await engine.search("喝水", {"anime_id": "' OR 1=1 --"}, embedding_profile_id="embed"))["results"] == []


def test_prior_memory_is_concrete_animation_scoped_inside_the_same_series(tmp_path):
    library = Library(tmp_path / "library")
    tv = scene_seed(library, "asset_tv", series="系列", episode=1, season=None)
    next_episode = scene_seed(library, "asset_next", series="系列", episode=2, season=None)
    other = scene_seed(library, "asset_other", series="系列", episode=1, season=None)
    path = library.asset_dir(other["id"]) / "manifest.json"
    atomic_json(path, {**read_json(path), "title": "系列的另一个 TV 动画"})
    store = SceneStore(library)
    store.publish(tv["id"], "run_tv", [chunk()], 0, 10000)
    store.publish(other["id"], "run_other", [chunk()], 0, 10000)
    assert {row["asset_id"] for row in store.prior_memory(next_episode["id"], 0)["summaries"]} == {tv["id"]}


def test_explicit_scratch_archive_is_not_overwritten_by_release_migration(tmp_path):
    library = Library(tmp_path / "library")
    for number in (1, 2):
        filename = f"[VCB-Studio] Synthetic Anime [{number:02d}][1080p].mkv"
        member = seed(library, f"asset_{number}", series="", title=filename, episode=number)
        path = library.asset_dir(member["id"]) / "manifest.json"
        atomic_json(path, {**read_json(path), "file_name": filename})
    profiles = AnimeProfileStore(library)
    profiles.apply_asset("asset_1", {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    # Emulate a new import that inherited a matching release before its explicit
    # scratch selection was applied, then view projects and reopen the library.
    catalog = library.organization._read()
    catalog["assets"]["asset_2"]["anime_profile_inherited"] = True
    library.organization._commit(catalog)
    profiles.apply_asset("asset_2", {"mode": "scratch"}, for_import=True)
    library.organization.projects()
    assert Library(library.root).get_asset("asset_2")["anime_profile"] == {"mode": "scratch"}


def test_generic_movie_can_still_be_assigned_to_a_movie_project(tmp_path):
    library = Library(tmp_path / "library")
    asset = {"id": "asset_movie", "work_id": "work_movie", "title": "真人电影", "kind": "movie",
             "project_id": None, "duration_ms": 10000, "version": "原版", "fingerprint": "movie-source",
             "created_at": "2026-10-14T00:00:00Z"}
    atomic_json(library.asset_dir(asset["id"]) / "manifest.json", asset)
    library.get_asset(asset["id"])
    project = library.organization.save_project({"name": "真人电影项目", "kind": "movie"})
    library.organization.assign([asset["id"]], project["id"])
    assert library.get_asset(asset["id"])["kind"] == "movie"


def test_concrete_animation_api_and_membership_profiles_are_local_session_protected(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="hierarchy-test")
    asset = seed(app.state.library)
    with TestClient(app) as client:
        assert client.get("/api/anime").status_code == 401
        client.post("/api/session", json={"token": "hierarchy-test"})
        result = client.get("/api/anime", params={"project_id": asset["project_id"]})
        assert result.status_code == 200
        assert result.json()[0]["id"] == asset["anime_id"]
        assert result.json()[0]["asset_ids"] == [asset["id"]]
        assert client.get("/api/anime", params={"project_id": "project_missing"}).status_code == 404


@pytest.mark.asyncio
async def test_same_series_import_prepares_two_distinct_subjects(tmp_path):
    from test_bangumi_workflow import FakeBangumi
    library = Library(tmp_path / "library")
    project = library.organization.save_project({"name": "系列"})
    client = FakeBangumi()
    store = AnimeProfileStore(library, client)
    request = ImportInput.model_validate({"request_id": "distinct-animations", "project_id": project["id"], "items": [
        {"item_id": "tv", "video_path": "synthetic-tv.mp4", "title": "TV", "episode": 1,
         "anime_profile": {"mode": "bangumi", "subject_id": 100}},
        {"item_id": "film", "video_path": "synthetic-film.mp4", "title": "Film",
         "anime_profile": {"mode": "bangumi", "subject_id": 101}}]})
    assert set(await store.prepare_import(request)) == {"tv", "film"}
    assert client.calls == [("dossier", 100), ("dossier", 101)]
