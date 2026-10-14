"""Archive orchestration tests use explicit fake network data and generated media."""
import asyncio
import copy
import subprocess
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from scenerecall.bangumi import BangumiError
from scenerecall.anime_profiles import merge_dossier
from scenerecall.library import atomic_json, now, read_json
from scenerecall.main import create_app
from scenerecall.providers import ProviderManager


def archive(subject_id=42):
    return {"source": "bangumi", "fetched_at": now(),
            "subject": {"id": subject_id, "name": "テスト", "name_cn": "合成动画",
                        "summary": "仅用于测试的作品简介", "url": f"https://bgm.tv/subject/{subject_id}",
                        "image_url": "", "date": "2026-01-01", "infobox": []},
            "characters": [{"id": 7, "name": "テスト角色", "name_cn": "合成角色", "aliases": ["测试别名"],
                            "description": "仅用于测试的角色简介", "image_url": "",
                            "url": "https://bgm.tv/character/7", "role": "主角"}]}


class FakeBangumi:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.candidates = [archive()["subject"]]

    async def search(self, query):
        self.calls.append(("search", query))
        if self.fail:
            raise BangumiError("测试网络不可用")
        return copy.deepcopy(self.candidates)

    async def dossier(self, subject_id):
        self.calls.append(("dossier", subject_id))
        if self.fail:
            raise BangumiError("测试网络不可用")
        return archive(subject_id)


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="bangumi-test")
    app.state.bangumi.client = FakeBangumi()
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "bangumi-test"})
        yield client


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("bangumi-generated-media") / "test.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=96x64:r=12:d=1.2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def import_request(path, *, project_id=None, selection=None, request_id="import-anime"):
    item = {"item_id": "episode-one", "video_path": str(path), "kind": "animation", "title": "合成动画",
            "subtitle_mode": "auto"}
    if selection is not None:
        item["anime_profile"] = selection
    return {"request_id": request_id, "project_id": project_id, "items": [item]}


def execute(client, job_id):
    asyncio.run(client.app.state.jobs.execute(job_id))
    return client.get(f"/api/jobs/{job_id}").json()


def test_authenticated_discovery_distinguishes_no_match_from_failure(client):
    client.cookies.clear()
    assert client.post("/api/bangumi/search", json={"query": "test"}).status_code == 401
    assert client.app.state.bangumi.client.calls == []
    client.post("/api/session", json={"token": "bangumi-test"})
    result = client.post("/api/bangumi/search", json={"query": "合成动画"}).json()
    assert result["status"] == "matches" and result["candidates"][0]["id"] == 42
    client.app.state.bangumi.client.candidates = []
    assert client.post("/api/bangumi/search", json={"query": "其他"}).json()["status"] == "not_found"
    client.app.state.bangumi.client.fail = True
    result = client.post("/api/bangumi/search", json={"query": "test"}).json()
    assert result["status"] == "error" and "网络" in result["error"]


def test_link_reads_cached_server_data_and_rejects_arbitrary_urls(client):
    assert client.post("/api/bangumi/subject", json={"reference": "https://example.com/subject/42"}).status_code == 400
    assert client.app.state.bangumi.client.calls == []
    first = client.post("/api/bangumi/subject", json={"reference": "https://bgm.tv/subject/42"})
    assert first.status_code == 200, first.text
    client.app.state.bangumi.client.fail = True
    second = client.post("/api/bangumi/subject", json={"reference": "42"})
    assert second.json() == first.json()
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]


def test_fresh_legacy_cache_upgrades_names_locally_and_retains_conversion_source(client):
    path = client.app.state.library.root / "runtime" / "bangumi" / "42.json"
    old = archive()
    old["characters"] = [{"id": 150045, "name": "櫻木麗", "name_cn": "", "aliases": ["原始别名"],
                          "infobox": [{"key": "别名", "values": [{"k": "罗马字", "v": "Sakuragi Rei"}]}]}]
    atomic_json(path, old)
    client.app.state.bangumi.client.fail = True
    result = client.post("/api/bangumi/subject", json={"reference": "42"})
    assert result.status_code == 200, result.text
    upgraded = result.json()
    assert upgraded["characters"][0]["name_cn"] == "樱木丽"
    assert upgraded["characters"][0]["name_en"] == "Sakuragi Rei"
    assert upgraded["fetched_at"] == old["fetched_at"]
    assert read_json(path) == upgraded
    assert client.app.state.bangumi.client.calls == []
    assert client.post("/api/bangumi/subject", json={"reference": "42"}).json() == upgraded


def test_dossier_refresh_keeps_current_roster_and_merges_new_name_variants():
    first = archive()
    first["characters"].append({"id": 8, "name": "已离开来源名单", "aliases": []})
    second = archive()
    second["characters"][0]["infobox"] = [{"key": "别名", "values": [
        {"k": "英文名", "v": "Example Character"},
    ]}]
    merged = merge_dossier(first, second)
    assert [value["id"] for value in merged["characters"]] == [7]
    character = merged["characters"][0]
    assert character["name_en"] == "Example Character"
    assert "Example Character" in character["aliases"] and "测试别名" in character["aliases"]
    assert any(value["name"] == "Example Character" and value["language"] == "en"
               for value in character["name_variants"])
    assert merge_dossier(merged, second) == merged


def test_source_names_replace_converted_names_and_preserve_the_conversion_history():
    first = archive()
    first["characters"] = [{"id": 150045, "name": "櫻木麗", "aliases": [], "infobox": [
        {"key": "别名", "values": [{"k": "纯假名", "v": "さくらぎ れい"}]},
    ]}]
    derived = merge_dossier(first, first)
    assert derived["characters"][0]["name_cn"] == "樱木丽"
    assert derived["characters"][0]["name_en"] == "Sakuragi Rei"
    second = copy.deepcopy(first)
    second["characters"][0]["infobox"] = [
        {"key": "简体中文名", "value": "樱木玲"},
        {"key": "别名", "value": [{"k": "英文名", "v": "Rei Sakuragi"},
                                   {"k": "纯假名", "v": "さくらぎ れい"}]},
    ]
    merged = merge_dossier(derived, second)
    character = merged["characters"][0]
    assert character["name_cn"] == "樱木玲" and character["name_en"] == "Rei Sakuragi"
    assert set(character["aliases"]) >= {"樱木丽", "樱木玲", "Sakuragi Rei", "Rei Sakuragi"}
    assert any(value["name"] == "樱木丽" and value["source"] == "opencc" for value in character["name_variants"])
    assert any(value["name"] == "樱木玲" and value["source"] == "bangumi" for value in character["name_variants"])
    assert merge_dossier(merged, second) == merged


def test_source_can_confirm_a_name_with_the_same_text_as_a_conversion():
    first = archive()
    first["characters"] = [{"id": 150045, "name": "櫻木麗", "aliases": []}]
    derived = merge_dossier(first, first)
    second = copy.deepcopy(first)
    second["characters"][0]["infobox"] = [{"key": "简体中文名", "value": "樱木丽"}]
    merged = merge_dossier(derived, second)
    variants = merged["characters"][0]["name_variants"]
    assert {value["source"] for value in variants if value["name"] == "樱木丽"} == {"bangumi", "opencc"}
    assert merge_dossier(merged, second) == merged


@pytest.mark.parametrize("older_time", ["2026-10-01T00:00:00Z", "2026-10-01 00:00:00", "invalid-date"])
def test_older_or_undated_snapshot_cannot_roll_back_the_roster_but_can_fill_metadata(older_time):
    latest = archive()
    latest["fetched_at"] = "2026-10-14T08:00:00+08:00"
    latest["characters"][0]["description"] = ""
    latest["characters"].append({"id": 9, "name": "Current", "aliases": []})
    older = archive()
    older["fetched_at"] = older_time
    older["characters"][0]["aliases"].append("旧资料有源别名")
    older["characters"].append({"id": 8, "name": "Removed", "aliases": []})
    merged = merge_dossier(latest, older)
    assert [value["id"] for value in merged["characters"]] == [7, 9]
    assert merged["fetched_at"] == latest["fetched_at"]
    assert merged["characters"][0]["description"] == older["characters"][0]["description"]
    assert "旧资料有源别名" in merged["characters"][0]["aliases"]
    assert merge_dossier(merged, older) == merged


def test_latest_roster_date_is_recorded_even_when_no_metadata_changed():
    first = archive()
    first["fetched_at"] = "2026-10-01T00:00:00Z"
    refresh = copy.deepcopy(first)
    refresh["fetched_at"] = "2026-10-14T00:00:00Z"
    merged = merge_dossier(first, refresh)
    assert merged["fetched_at"] == refresh["fetched_at"]
    middle = copy.deepcopy(first)
    middle["fetched_at"] = "2026-10-07T00:00:00Z"
    middle["characters"].append({"id": 8, "name": "Older-only", "aliases": []})
    assert [value["id"] for value in merge_dossier(merged, middle)["characters"]] == [7]


def test_legacy_unknown_dates_keep_the_saved_roster_until_a_valid_refresh_arrives():
    saved = archive()
    saved["fetched_at"] = "old"
    invalid = copy.deepcopy(saved)
    invalid["fetched_at"] = "not-a-date"
    invalid["characters"].append({"id": 8, "name": "Unverified refresh", "aliases": []})
    assert [value["id"] for value in merge_dossier(saved, invalid)["characters"]] == [7]
    invalid["fetched_at"] = "2026-10-14"
    assert [value["id"] for value in merge_dossier(saved, invalid)["characters"]] == [7, 8]


def test_moving_old_asset_into_refreshed_project_keeps_the_newest_roster(client, video):
    payload = import_request(video, selection={"mode": "bangumi", "subject_id": 42})
    job = client.post("/api/imports", json=payload).json()
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    asset_id = done["items"][0]["asset_id"]
    old = client.app.state.library.get_asset(asset_id)["anime_profile"]["dossier"]
    latest = copy.deepcopy(old)
    latest["fetched_at"] = (datetime.fromisoformat(old["fetched_at"]) + timedelta(seconds=1)).isoformat()
    latest["characters"] = [{"id": 9, "name": "Current", "name_cn": "现有角色", "aliases": []}]
    project = client.post("/api/projects", json={"name": "较新档案项目", "kind": "animation"}).json()
    client.app.state.bangumi.apply_project(project["id"], {"mode": "bangumi", "subject_id": 42}, latest)
    client.app.state.library.organization.assign([asset_id], project["id"])
    profile = client.app.state.library.get_asset(asset_id)["anime_profile"]["dossier"]
    assert [value["id"] for value in profile["characters"]] == [9]
    assert profile["fetched_at"] == latest["fetched_at"]
    assert [value["external_source"]["character_id"]
            for value in client.app.state.characters.catalog(asset_id)["profiles"]] == [9]


def test_frozen_import_snapshot_cannot_roll_back_a_newer_project_roster(client, video):
    project = client.post("/api/projects", json={"name": "已刷新档案项目", "kind": "animation"}).json()
    payload = import_request(video, project_id=project["id"], selection={"mode": "bangumi", "subject_id": 42})
    job = client.post("/api/imports", json=payload).json()
    frozen = client.app.state.jobs.get(job["id"])["config"]["items"][0]["anime_dossier"]
    latest = copy.deepcopy(frozen)
    latest["fetched_at"] = (datetime.fromisoformat(frozen["fetched_at"]) + timedelta(seconds=1)).isoformat()
    latest["characters"] = [{"id": 9, "name": "Current", "name_cn": "现有角色", "aliases": []}]
    client.app.state.bangumi.apply_project(project["id"], {"mode": "bangumi", "subject_id": 42}, latest)
    client.app.state.bangumi.client.fail = True
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    asset_id = done["items"][0]["asset_id"]
    profile = client.app.state.library.get_asset(asset_id)["anime_profile"]["dossier"]
    assert [value["id"] for value in profile["characters"]] == [9]
    assert profile["fetched_at"] == latest["fetched_at"]
    assert [value["external_source"]["character_id"]
            for value in client.app.state.characters.catalog(asset_id)["profiles"]] == [9]
    assert client.app.state.jobs.get(job["id"])["config"]["items"][0]["anime_dossier"] == frozen


def test_import_seeds_shared_cast_before_first_recognition_and_replay_is_offline(client, video):
    project = client.post("/api/projects", json={"name": "合成动画", "kind": "animation"}).json()
    payload = import_request(video, project_id=project["id"], selection={"mode": "bangumi", "subject_id": 42})
    response = client.post("/api/imports", json=payload)
    assert response.status_code == 200, response.text
    job = response.json()
    client.app.state.bangumi.client.fail = True
    assert client.post("/api/imports", json=payload).json()["id"] == job["id"]
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    asset_id = done["items"][0]["asset_id"]
    asset = client.app.state.library.get_asset(asset_id)
    assert asset["anime_profile"]["dossier"]["subject"]["id"] == 42
    cast = client.app.state.characters.catalog(asset_id)
    assert cast["profiles"][0]["appearance_count"] == 0
    assert cast["profiles"][0]["representative"] is None
    context = client.app.state.jobs.character_context(asset_id)
    assert context["profiles"][0]["name"] == "合成角色"
    assert context["reference_frames"] == []
    assert "仅用于测试的作品简介" in ProviderManager.character_context_text(context)
    public_config = client.get(f"/api/imports/{job['id']}").json()["items"][0]
    assert public_config["anime_profile"] == {"mode": "bangumi", "subject_id": 42}
    assert "anime_dossier" not in public_config


def test_scratch_archive_has_no_network_calls_and_standalone_bangumi_has_summary(client, video):
    payload = import_request(video, selection={"mode": "scratch"})
    job = client.post("/api/imports", json=payload).json()
    done = execute(client, job["id"])
    assert done["status"] == "completed", done
    asset_id = done["items"][0]["asset_id"]
    assert client.app.state.library.get_asset(asset_id)["anime_profile"] == {"mode": "scratch"}
    assert client.app.state.bangumi.client.calls == []
    added = client.post(f"/api/assets/{asset_id}/anime-profile", json={"mode": "bangumi", "subject_id": 42})
    assert added.status_code == 200, added.text
    context = client.app.state.characters.context(asset_id)
    assert context["work_summary"]["id"] == 42
    assert context["profiles"][0]["external_source"]["source"] == "bangumi"


def test_failed_import_retries_using_durable_archive_snapshot(client, video, tmp_path):
    payload = import_request(tmp_path / "missing.mp4", selection={"mode": "bangumi", "subject_id": 42})
    job = client.post("/api/imports", json=payload).json()
    assert execute(client, job["id"])["status"] == "partial"
    client.app.state.bangumi.client.fail = True
    retry = {"items": [{**payload["items"][0], "video_path": str(video)}]}
    result = client.post(f"/api/imports/{job['id']}/retry", json=retry)
    assert result.status_code == 200, result.text
    assert execute(client, job["id"])["status"] == "completed"
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]


def test_conflicting_subjects_do_not_mix_project_cast_or_publish_jobs(client, video):
    project = client.post("/api/projects", json={"name": "合成动画", "kind": "animation"}).json()
    payload = import_request(video, project_id=project["id"], selection={"mode": "bangumi", "subject_id": 42})
    payload["items"].append({**payload["items"][0], "item_id": "other", "anime_profile": {"mode": "scratch"}})
    assert client.post("/api/imports", json=payload).status_code == 400
    assert client.get("/api/jobs").json() == []
    assert client.app.state.bangumi.client.calls == []
    bound = client.post(f"/api/projects/{project['id']}/anime-profile", json={"mode": "bangumi", "subject_id": 42})
    assert bound.status_code == 200, bound.text
    before = read_json(client.app.state.library.organization.path)
    response = client.post(f"/api/projects/{project['id']}/anime-profile", json={"mode": "bangumi", "subject_id": 99})
    assert response.status_code == 400
    assert read_json(client.app.state.library.organization.path) == before
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]


def test_profile_attachment_rejects_non_animation_and_untrusted_dossier(client):
    project = client.post("/api/projects", json={"name": "电影", "kind": "movie"}).json()
    assert client.post(f"/api/projects/{project['id']}/anime-profile",
                       json={"mode": "bangumi", "subject_id": 42}).status_code == 400
    assert client.post("/api/projects", json={"name": "伪造", "anime_profile": archive()}).status_code == 422
    assert client.app.state.bangumi.client.calls == []


def test_legacy_failed_import_can_select_archive_on_retry(client, video, tmp_path):
    payload = import_request(tmp_path / "missing.mp4")
    job = client.post("/api/imports", json=payload).json()
    assert execute(client, job["id"])["status"] == "partial"
    retry = {"items": [{**payload["items"][0], "video_path": str(video),
                        "anime_profile": {"mode": "bangumi", "subject_id": 42}}]}
    result = client.post(f"/api/imports/{job['id']}/retry", json=retry)
    assert result.status_code == 200, result.text
    assert execute(client, job["id"])["status"] == "completed"
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]


def test_same_subject_refresh_adds_missing_metadata_and_roles_without_resetting_cast(client):
    project = client.post("/api/projects", json={"name": "合成动画", "kind": "animation"}).json()
    store = client.app.state.bangumi
    first = archive()
    first["characters"][0]["description"] = ""
    store.apply_project(project["id"], {"mode": "bangumi", "subject_id": 42}, first)
    scope = client.app.state.library.organization.get_project(project["id"])["character_scope_id"]
    before = client.app.state.characters._read(scope)
    second = archive()
    second["characters"].append({**second["characters"][0], "id": 8, "name": "新增角色"})
    result = store.apply_project(project["id"], {"mode": "bangumi", "subject_id": 42}, second)
    after = client.app.state.characters._read(scope)
    assert len(result["dossier"]["characters"]) == len(after["profiles"]) == 2
    assert set(before["profiles"]).issubset(after["profiles"])
    assert next(iter(after["profiles"].values()))["description"] == "仅用于测试的角色简介"
    assert after["revision"] == before["revision"] + 1
    store.apply_project(project["id"], {"mode": "bangumi", "subject_id": 42}, second)
    assert client.app.state.characters._read(scope) == after
