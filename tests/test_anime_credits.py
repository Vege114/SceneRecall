"""Credits use fabricated public metadata, never OCR or model inference."""
import copy
import zipfile

import pytest
from fastapi.testclient import TestClient

from scenerecall.anime_credits import AnimeCreditsStore, episode_match
from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.bangumi import BangumiClient, BangumiError
from scenerecall.library import Library, atomic_json
from scenerecall.main import create_app
from test_organization import bangumi_archive, seed


class CreditsClient:
    timeout_s = 1
    _items = staticmethod(BangumiClient._items)

    def __init__(self):
        self.calls = []
        self.error = None

    async def _run(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return {"total": 3, "data": [
            {"staff": {"id": 11, "name": "合成画师", "nameCN": "合成画师"}, "positions": [
                {"type": {"cn": "分镜"}, "appearEps": "1,3-5"},
                {"type": {"cn": "演出"}, "appearEps": "2"}]},
            {"staff": {"id": 12, "name": "合成监督"}, "positions": [
                {"type": {"cn": "监督"}, "appearEps": ""}]},
            {"staff": {"id": 13, "name": "合成演出"}, "positions": [
                {"type": {"cn": "演出"}, "appearEps": "OP、特别篇"}]},
        ]}


@pytest.mark.parametrize(("scope", "episode", "expected"), [
    ("1,3-5", 4, True), ("1,3-5", 2, False), ("6,11", 11, True),
    ("第1话、EP02、3～5", 5, True), ("全话", 12, True),
    ("", 1, None), ("OP,1", 1, None), ("4-1", 2, None),
    ("1-12", None, None), ("1/2/3", 2, True),
])
def test_only_explicit_episode_scopes_establish_participation(scope, episode, expected):
    assert episode_match(scope, episode) is expected


@pytest.fixture
def store(tmp_path):
    library = Library(tmp_path / "library")
    seed(library, "asset_first", episode=1)
    seed(library, "asset_second", episode=2)
    AnimeProfileStore(library).apply_asset("asset_first", {"mode": "bangumi", "subject_id": 100}, bangumi_archive())
    return AnimeCreditsStore(library, CreditsClient())


async def test_shared_source_is_projected_by_episode_and_kept_out_of_narrative(store):
    assert store.get("asset_first")["status"] == "missing"
    first = await store.fetch("asset_first")
    assert [(row["position"], row["scope"]) for row in first["rows"]] == [
        ("分镜", "episode"), ("监督", "subject"), ("演出", "unparsed")]
    second = store.get("asset_second")
    assert [row["position"] for row in second["rows"] if row["scope"] == "episode"] == ["演出"]
    assert len(store.client.calls) == 1
    assert store.library.observations("asset_first") == []
    assert store.library.subtitles("asset_first") == []
    first["rows"][0]["name"] = "mutated"
    assert store.get("asset_first")["rows"][0]["name"] == "合成画师"
    backup = store.library.backup()
    path = store.library.root / "runtime" / "backups" / f"{backup['id']}.zip"
    with zipfile.ZipFile(path) as archive:
        assert "organization/bangumi-credits/100.json" in archive.namelist()
    restored = Library(store.library.root.parent / "restored")
    restored.restore(path)
    assert AnimeCreditsStore(restored, CreditsClient()).get("asset_second")["rows"] == second["rows"]


async def test_failed_refresh_keeps_last_successful_table(store):
    saved = await store.fetch("asset_first")
    store.client.error = BangumiError("合成网络失败")
    with pytest.raises(BangumiError, match="网络"):
        await store.fetch("asset_first")
    assert store.get("asset_first") == saved


async def test_scratch_archive_never_fetches_credits(store):
    seed(store.library, "scratch", series="其他作品", title="其他作品")
    assert store.get("scratch")["status"] == "unavailable"
    with pytest.raises(ValueError, match="关联 Bangumi"):
        await store.fetch("scratch")
    assert not store.client.calls


def test_credit_routes_are_session_protected_and_get_stays_local(store):
    app = create_app(store.library.root, start_worker=False, token="credits-test")
    app.state.anime_credits.client = store.client
    with TestClient(app) as client:
        assert client.post("/api/assets/asset_first/credits").status_code == 401
        client.post("/api/session", json={"token": "credits-test"})
        assert client.get("/api/assets/asset_first/credits").json()["status"] == "missing"
        assert not store.client.calls
        assert client.post("/api/assets/asset_first/credits").status_code == 200
        assert client.get("/api/assets/asset_second/credits").json()["status"] == "completed"
        assert len(store.client.calls) == 1


def test_public_filename_remains_usable_offline_and_omits_directories(store):
    manifest = store.library.asset_dir("asset_first") / "manifest.json"
    asset = copy.deepcopy(store.library.get_asset("asset_first"))
    asset["file_name"] = r"C:\private-folder\EP01.mkv"
    atomic_json(manifest, asset)
    public = store.library.get_asset("asset_first")
    assert public["source_filename"] == "EP01.mkv"
    assert not public["source_available"]
