"""Archive orchestration tests use explicit fake network data and generated media."""
import asyncio
import copy
import importlib.util
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from scenerecall.bangumi import BangumiError
from scenerecall.anime_profiles import merge_dossier
from scenerecall.library import atomic_json, now, read_json
from scenerecall.main import create_app
from scenerecall.providers import ProviderManager

spec = importlib.util.spec_from_file_location("cast_verifier_test_module", Path(__file__).parents[1] / "scripts/verify_bangumi_cast.py")
cast_verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cast_verifier)


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


def test_batch_original_ass_import_warns_and_retries_only_failure_offline(client, video, tmp_path, monkeypatch):
    """Exercise the selected video + original ASS + Bangumi path without a cleaned derivative."""
    second_video = tmp_path / "episode-two.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=96x64:r=12:d=1.2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(second_video)], check=True)
    header = ("[Script Info]\r\nScriptType: v4.00+\r\n[Events]\r\n"
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n")
    original = tmp_path / "episode-one.sc.ass"
    original_bytes = ("\ufeff" + header
        + "Dialogue: 0,0:00:00.10,0:00:00.80,Default,,0,0,0,,第一集原字幕\r\n"
        + "Dialogue: 0,0:00:00.90,0:00:00.60,OP-effect,,0,0,0,,反向特效\r\n"
        + "Dialogue: 0,0:00:00.50,0:00:00.50,OP-effect,,0,0,0,,零时长特效\r\n").encode("utf-8")
    original.write_bytes(original_bytes)
    wrong = tmp_path / "episode-two.sc.ass"
    wrong_bytes = (header + "Dialogue: 0,0:00:00.10,0:00:03.00,Default,,0,0,0,,版本不匹配\r\n").encode()
    wrong.write_bytes(wrong_bytes)
    fixed = tmp_path / "episode-two-correct.sc.ass"
    fixed_bytes = (header + "Dialogue: 0,0:00:00.20,0:00:00.90,Default,,0,0,0,,第二集正确字幕\r\n").encode()
    fixed.write_bytes(fixed_bytes)

    model_calls = []
    async def forbidden_model_call(*args, **kwargs):
        model_calls.append((args, kwargs))
        raise AssertionError("local import must not call a model")
    for method in ("analyze", "recognize_subtitles", "prepare_subtitles", "prepare_visual", "embed"):
        monkeypatch.setattr(client.app.state.providers, method, forbidden_model_call)

    project = client.post("/api/projects", json={"name": "原字幕批量回归", "kind": "animation"}).json()
    first_item = {"item_id": "one", "video_path": str(video), "episode": 1, "kind": "animation",
                  "subtitle_mode": "external", "subtitle_path": str(original), "subtitle_offset_ms": 100,
                  "anime_profile": {"mode": "bangumi", "subject_id": 42}}
    second_item = {**first_item, "item_id": "two", "video_path": str(second_video), "episode": 2,
                   "subtitle_path": str(wrong), "subtitle_offset_ms": 0}
    payload = {"request_id": "original-ass-batch", "project_id": project["id"],
               "items": [first_item, second_item]}
    response = client.post("/api/imports", json=payload)
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    frozen = copy.deepcopy(client.app.state.jobs.get(job_id)["config"]["items"])
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]
    client.app.state.bangumi.client.fail = True
    # Remove the adapter cache: execution and retry must use the durable job snapshot.
    (client.app.state.library.root / "runtime" / "bangumi" / "42.json").unlink(missing_ok=True)
    assert client.post("/api/imports", json=payload).json()["id"] == job_id

    partial = execute(client, job_id)
    assert partial["status"] == "partial", partial
    assert [item["status"] for item in partial["items"]] == ["completed", "failed"]
    assert "超出视频范围" in partial["items"][1]["error"]
    assert len(client.get("/api/assets").json()) == 1
    assert partial["request_count"] == 0
    assert str(tmp_path) not in str(partial)
    first_asset_id = partial["items"][0]["asset_id"]
    first_asset = client.app.state.library.get_asset(first_asset_id)
    asset_directory = client.app.state.library.asset_dir(first_asset_id)
    assert (asset_directory / "subtitles" / "originals" / "original.ass").read_bytes() == original_bytes
    diagnostics_path = asset_directory / "subtitles" / "import-diagnostics.json"
    diagnostics = read_json(diagnostics_path)
    assert diagnostics["ignored_ass_events"] == [
        {"source_cue_index": 1, "start_ms": 900, "end_ms": 600,
         "reason": "non_positive_duration", "style": "OP-effect"},
        {"source_cue_index": 2, "start_ms": 500, "end_ms": 500,
         "reason": "non_positive_duration", "style": "OP-effect"},
    ]
    assert first_asset["subtitle_import_warnings"] == partial["items"][0]["warnings"]
    assert len(first_asset["subtitle_import_warnings"]) == 1
    assert "2" in first_asset["subtitle_import_warnings"][0]
    cues = client.app.state.library.subtitles(first_asset_id)
    assert [(cue["start_ms"], cue["end_ms"], cue["text"]) for cue in cues] == [(200, 900, "第一集原字幕")]

    retry = client.post(f"/api/imports/{job_id}/retry", json={"items": [{**second_item, "subtitle_path": str(fixed)}]})
    assert retry.status_code == 200, retry.text
    completed = execute(client, job_id)
    assert completed["status"] == "completed", completed
    assert completed["items"][0] == partial["items"][0]
    assert completed["items"][1]["warnings"] == []
    assert completed["request_count"] == 0 and model_calls == []
    assets = client.get("/api/assets").json()
    assert len(assets) == 2 and {asset["project_id"] for asset in assets} == {project["id"]}
    second_asset_id = completed["items"][1]["asset_id"]
    assert client.app.state.library.source_path(first_asset_id) == video
    assert client.app.state.library.source_path(second_asset_id) == second_video
    assert (client.app.state.library.asset_dir(second_asset_id) / "subtitles" / "originals" / "original.ass").read_bytes() == fixed_bytes
    assert read_json(diagnostics_path) == diagnostics
    assert original.read_bytes() == original_bytes and wrong.read_bytes() == wrong_bytes and fixed.read_bytes() == fixed_bytes
    assert client.app.state.bangumi.client.calls == [("dossier", 42)]
    assert [item["anime_dossier"] for item in client.app.state.jobs.get(job_id)["config"]["items"]] == [
        item["anime_dossier"] for item in frozen]
    for asset_id in (first_asset_id, second_asset_id):
        cast = client.app.state.characters.catalog(asset_id)
        assert len(cast["profiles"]) == 1 and cast["profiles"][0]["name"] == "合成角色"
        assert cast["profiles"][0]["appearance_count"] == 0

    # A new import creates an independent archive and rechecks its own subtitles.
    duplicate = client.post("/api/imports", json={"request_id": "duplicate-ass", "items": [
        {key: value for key, value in payload["items"][0].items() if key != "anime_profile"}]})
    assert duplicate.status_code == 200, duplicate.text
    duplicate_result = execute(client, duplicate.json()["id"])
    assert duplicate_result["items"][0]["status"] == "completed"
    assert duplicate_result["items"][0]["asset_id"] != first_asset_id
    assert duplicate_result["items"][0]["warnings"] == partial["items"][0]["warnings"]
    assert len(client.get("/api/assets").json()) == 3


@pytest.mark.parametrize("legacy_filter", [False, True])
def test_cast_verifier_counts_automatic_omissions_and_checks_original_bytes(video, tmp_path, legacy_filter):
    """The verification report must disclose product omissions, including with the old flag set."""
    media_directory = tmp_path / "media"
    media_directory.mkdir()
    (media_directory / "episode[01].mp4").symlink_to(video)
    source = ("[Script Info]\nScriptType: v4.00+\n[Events]\n"
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
              "Dialogue: 0,0:00:00.10,0:00:00.80,Default,,0,0,0,,正常台词\n"
              "Dialogue: 0,0:00:00.90,0:00:00.60,OP,,0,0,0,,反向特效\n"
              "Dialogue: 0,0:00:00.50,0:00:00.50,OP,,0,0,0,,零时长特效\n").encode()
    for suffix in (".sc.ass", ".tc.ass"):
        (media_directory / f"episode[01]{suffix}").write_bytes(source)
    dossier = archive()
    dossier["subject"].update(eps=1, platform={"type": "tv"})
    dossier["characters"][0].update(name_ja="テスト角色", name_en="Test Character")
    dossier_cache = tmp_path / "public-dossier.json"
    atomic_json(dossier_cache, dossier)
    output = tmp_path / "verification"
    args = SimpleNamespace(media_dir=media_directory, output=output, resume=False,
                           episode_pattern=r"\[(\d{2})\]", expected_episodes=1, subject=42,
                           bangumi_timeout=1, dossier_cache=dossier_cache, provider_settings=None,
                           vision_times_ms=[], filter_invalid_ass_events=legacy_filter)
    report = asyncio.run(cast_verifier.verify(args))
    assert report["status"] == "passed", report
    validation = report["source_validation"]
    assert validation["unmodified_ass_passed"] is True
    assert validation["excluded_events"] == validation["automatic_ignored_events"] == 4
    assert validation["imported_automatic_ignored_events"] == 2
    assert validation["explicitly_filtered_events"] == validation["rejected_ass_files"] == 0
    episode = report["episodes"][0]
    assert episode["saved_input_subtitle_bytes_preserved"] is True
    assert episode["unmodified_source_subtitle_imported"] is True
    assert episode["import_diagnostics_match_source_parse"] is True
    assert episode["derived_subtitle"] is False
    assert len(episode["subtitle_import_diagnostics"]["ignored_ass_events"]) == 2
    assert len(episode["subtitle_import_warnings"]) == 1
    assert not (output / "derived-subtitles").exists()


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
