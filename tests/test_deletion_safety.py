"""Deletion safety contracts use generated media and never access user videos."""
import asyncio
import copy
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from scenerecall.library import Library, atomic_json, read_json
from scenerecall.main import create_app


class NoRemoteProvider:
    def list(self):
        return []

    def get(self, _profile_id):
        raise AssertionError("Deletion must not call a model provider")


@pytest.fixture(scope="module")
def generated_sources(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for synthetic deletion tests")
    root = tmp_path_factory.mktemp("deletion-generated")
    sources = []
    for color in ("red", "blue"):
        path = root / f"{color}.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        f"color={color}:s=96x64:r=10:d=0.4", "-c:v", "libx264", str(path)], check=True)
        sources.append(path)
    return sources


@pytest.fixture
def sources(tmp_path, generated_sources):
    directory = tmp_path / "owned-test-sources"
    directory.mkdir()
    paths = []
    for original in generated_sources:
        path = directory / original.name
        shutil.copy2(original, path)
        paths.append(path)
    return paths


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, providers=NoRemoteProvider(), token="test")
    with TestClient(app) as result:
        result.post("/api/session", json={"token": "test"})
        yield result


def register(client, source, **extra):
    result = client.post("/api/assets", json={"video_path": str(source), "subtitle_mode": "auto", **extra})
    assert result.status_code == 200, result.text
    return result.json()


def preview(client, assets, *, delete_source=False):
    result = client.post("/api/deletions/preview", json={
        "kind": "assets", "asset_ids": [asset["id"] for asset in assets], "delete_source": delete_source})
    assert result.status_code == 200, result.text
    return result.json()


def delete(client, assets, prepared, *, delete_source=False):
    return client.post("/api/assets/delete", json={"asset_ids": [asset["id"] for asset in assets],
                       "expected_token": prepared["token"], "delete_source": delete_source})


def assert_rejected(result):
    assert result.status_code in {400, 409}, result.text


def test_metadata_only_delete_keeps_source_and_restores_same_asset(client, sources):
    asset = register(client, sources[0])
    original = sources[0].read_bytes()
    result = delete(client, [asset], preview(client, [asset]))
    assert result.status_code == 200, result.text
    assert sources[0].read_bytes() == original
    assert client.get(f"/api/assets/{asset['id']}").status_code == 404
    restored = client.post(f"/api/trash/{result.json()['trash']['id']}/restore")
    assert restored.status_code == 200, restored.text
    assert client.get(f"/api/assets/{asset['id']}").json()["asset"]["source_available"]


def test_permanent_source_delete_retains_subtitle_and_metadata_restore_is_offline(client, sources):
    subtitle = sources[0].with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:00,300\n合成字幕\n", encoding="utf-8")
    asset = register(client, sources[0], subtitle_mode="external", subtitle_path=str(subtitle))
    prepared = preview(client, [asset], delete_source=True)
    assert prepared["source_file_count"] == 1
    assert prepared["source_files"][0]["status"] == "ready"
    result = client.delete(f"/api/assets/{asset['id']}", params={
        "expected_token": prepared["token"], "delete_source": "true"})
    assert result.status_code == 200, result.text
    trash = result.json()["trash"]
    assert [entry["status"] for entry in trash["source_results"]] == ["deleted"]
    assert not sources[0].exists()
    assert subtitle.is_file() and subtitle.parent.is_dir()
    restored = client.post(f"/api/trash/{trash['id']}/restore")
    assert restored.status_code == 200, restored.text
    detail = client.get(f"/api/assets/{asset['id']}").json()
    assert detail["asset"]["source_available"] is False
    assert detail["subtitles"][0]["text"] == "合成字幕"
    assert not sources[0].exists()


def test_full_sha_detects_changed_content_with_preserved_size_and_mtime(client, sources):
    asset = register(client, sources[0])
    before = sources[0].stat()
    content = bytearray(sources[0].read_bytes())
    content[-1] ^= 1
    sources[0].write_bytes(content)
    os.utime(sources[0], ns=(before.st_atime_ns, before.st_mtime_ns))
    # The ordinary cheap availability check alone cannot detect this change.
    assert client.app.state.library.get_asset(asset["id"])["source_changed"] is False
    prepared = preview(client, [asset], delete_source=True)
    assert prepared["source_files"][0]["status"] == "changed"
    assert_rejected(delete(client, [asset], prepared, delete_source=True))
    assert sources[0].read_bytes() == bytes(content)
    assert client.get(f"/api/assets/{asset['id']}").status_code == 200


def test_source_stat_change_is_not_silently_deleted(client, sources):
    asset = register(client, sources[0])
    before = sources[0].stat()
    os.utime(sources[0], ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    prepared = preview(client, [asset], delete_source=True)
    assert prepared["source_files"][0]["status"] == "changed"
    assert_rejected(delete(client, [asset], prepared, delete_source=True))
    assert sources[0].is_file()


def test_missing_source_does_not_silently_downgrade_permanent_deletion(client, sources):
    asset = register(client, sources[0])
    moved = sources[0].with_suffix(".moved")
    sources[0].rename(moved)
    prepared = preview(client, [asset], delete_source=True)
    assert prepared["source_files"][0]["status"] == "missing"
    assert_rejected(delete(client, [asset], prepared, delete_source=True))
    assert moved.is_file()
    assert client.get(f"/api/assets/{asset['id']}").status_code == 200


def test_preview_token_rejects_identical_content_replacement_inode(client, sources):
    asset = register(client, sources[0])
    prepared = preview(client, [asset], delete_source=True)
    before = sources[0].stat()
    replacement = sources[0].with_suffix(".replacement")
    shutil.copy2(sources[0], replacement)
    assert replacement.stat().st_ino != before.st_ino
    os.replace(replacement, sources[0])
    assert sources[0].stat().st_mtime_ns == before.st_mtime_ns
    assert_rejected(delete(client, [asset], prepared, delete_source=True))
    assert sources[0].is_file()
    assert client.get(f"/api/assets/{asset['id']}").status_code == 200


def test_symlink_replacement_cannot_delete_its_target(client, sources):
    asset = register(client, sources[0])
    target = sources[0].with_name("unselected-target.mp4")
    shutil.copy2(sources[0], target)
    original = target.read_bytes()
    sources[0].unlink()
    sources[0].symlink_to(target)
    prepared = preview(client, [asset], delete_source=True)
    assert prepared["source_files"][0]["status"] == "unsafe"
    assert_rejected(delete(client, [asset], prepared, delete_source=True))
    assert sources[0].is_symlink() and target.read_bytes() == original


def test_shared_source_with_unselected_historical_asset_blocks_removal(client, sources):
    first = register(client, sources[0])
    library = client.app.state.library
    # An old/merged library can contain two immutable asset IDs for one source.
    second_id = "asset_shared_history"
    manifest = copy.deepcopy(read_json(library.asset_dir(first["id"]) / "manifest.json"))
    manifest["id"] = second_id
    atomic_json(library.asset_dir(second_id) / "manifest.json", manifest)
    locations_path = library.root / "private" / "media-locations.json"
    locations = read_json(locations_path)
    locations[second_id] = copy.deepcopy(locations[first["id"]])
    atomic_json(locations_path, locations)
    library.get_asset(second_id)
    prepared = preview(client, [first], delete_source=True)
    assert prepared["source_files"][0]["status"] == "shared"
    assert_rejected(delete(client, [first], prepared, delete_source=True))
    assert sources[0].is_file()
    assert client.get(f"/api/assets/{second_id}").status_code == 200


def test_repeat_import_shared_source_blocks_partial_removal(client, sources):
    first = register(client, sources[0])
    second = register(client, sources[0])
    assert first["id"] != second["id"]
    prepared = preview(client, [first], delete_source=True)
    assert prepared["source_files"][0]["status"] == "shared"
    assert_rejected(delete(client, [first], prepared, delete_source=True))
    assert sources[0].is_file()
    assert client.get(f"/api/assets/{second['id']}").json()["asset"]["source_available"]


def test_new_shared_reference_during_final_hash_prevents_unlink(client, sources, monkeypatch):
    from scenerecall import media

    asset = register(client, sources[0])
    library = client.app.state.library
    second_id = "asset_shared_during_hash"
    manifest = copy.deepcopy(read_json(library.asset_dir(asset["id"]) / "manifest.json"))
    manifest["id"] = second_id
    locations_path = library.root / "private" / "media-locations.json"
    fingerprint = media.fingerprint
    calls = 0

    def add_reference_after_final_hash(path):
        nonlocal calls
        checksum = fingerprint(path)
        calls += 1
        # Preview and commit preflight are already finished. This simulates a
        # concurrent publication while the final long hash runs without the lock.
        if calls == 3:
            with library.lock:
                atomic_json(library.asset_dir(second_id) / "manifest.json", manifest)
                locations = read_json(locations_path)
                locations[second_id] = copy.deepcopy(locations[asset["id"]])
                atomic_json(locations_path, locations)
                library.get_asset(second_id)
        return checksum

    monkeypatch.setattr(media, "fingerprint", add_reference_after_final_hash)
    prepared = preview(client, [asset], delete_source=True)
    assert prepared["source_files"][0]["status"] == "ready"
    result = delete(client, [asset], prepared, delete_source=True)
    assert result.status_code == 200, result.text
    assert calls == 3
    assert result.json()["trash"]["source_results"][0]["status"] == "error"
    assert sources[0].is_file()
    assert client.get(f"/api/assets/{asset['id']}").status_code == 404
    assert client.get(f"/api/assets/{second_id}").json()["asset"]["source_available"]


def test_partial_unlink_failure_keeps_persistent_per_source_outcomes(client, sources, monkeypatch):
    from scenerecall.deletions import DeletionStore

    assets = [register(client, source) for source in sources]
    prepared = preview(client, assets, delete_source=True)
    unlink = DeletionStore._unlink_source
    calls = []

    def fail_second(path):
        calls.append(Path(path))
        if len(calls) == 2:
            raise PermissionError("synthetic source cannot be unlinked")
        return unlink(path)

    monkeypatch.setattr(DeletionStore, "_unlink_source", staticmethod(fail_second))
    result = delete(client, assets, prepared, delete_source=True)
    assert result.status_code == 200, result.text
    trash = result.json()["trash"]
    assert len(calls) == 2 and not calls[0].exists() and calls[1].exists()
    assert sorted(entry["status"] for entry in trash["source_results"]) == ["deleted", "error"]
    assert set(result.json()["deleted_asset_ids"]) == {asset["id"] for asset in assets}
    # A new application instance reads durable outcomes and never retries unlink.
    reopened = create_app(client.app.state.library.root, start_worker=False,
                          providers=NoRemoteProvider(), token="reopened")
    with TestClient(reopened) as other:
        other.post("/api/session", json={"token": "reopened"})
        persisted = next(entry for entry in other.get("/api/trash").json() if entry["id"] == trash["id"])
        assert persisted["source_results"] == trash["source_results"]
        assert len(calls) == 2
        restored = other.post(f"/api/trash/{trash['id']}/restore")
        assert restored.status_code == 200, restored.text
        availability = [other.get(f"/api/assets/{asset['id']}").json()["asset"]["source_available"]
                        for asset in assets]
        assert sorted(availability) == [False, True]


@pytest.mark.parametrize("replace_after_crash", [False, True])
def test_interrupted_unlink_reconciliation_never_deletes_new_or_pending_sources(
        client, sources, monkeypatch, replace_after_crash):
    from scenerecall.deletions import DeletionStore

    assets = [register(client, source) for source in sources]
    store = client.app.state.library.deletions
    request = {"kind": "assets", "asset_ids": [asset["id"] for asset in assets], "delete_source": True}
    prepared = store.preview(**request)
    deleted_path = Path(prepared["source_files"][0]["path"])
    remaining_path = Path(prepared["source_files"][1]["path"])
    original_remaining = remaining_path.read_bytes()
    original_result = DeletionStore._set_source_result

    def crash_after_unlink(self, trash_id, index, status, message):
        raise OSError("synthetic crash after unlink, before result checkpoint")

    monkeypatch.setattr(DeletionStore, "_set_source_result", crash_after_unlink)
    with pytest.raises(OSError, match="synthetic crash"):
        store.delete(expected_token=prepared["token"], **request)
    assert not deleted_path.exists() and remaining_path.is_file()
    trash_id = store.list()[0]["id"]
    assert {entry["status"] for entry in store.list()[0]["source_results"]} == {"pending"}
    if replace_after_crash:
        deleted_path.write_bytes(b"a new unrelated file now occupies the former source path")
    monkeypatch.setattr(DeletionStore, "_set_source_result", original_result)

    def never_replay_unlink(_path):
        raise AssertionError("Startup must inspect interrupted source deletions, never replay unlink")

    monkeypatch.setattr(DeletionStore, "_unlink_source", staticmethod(never_replay_unlink))
    restored = Library(client.app.state.library.root)
    record = next(entry for entry in restored.deletions.list() if entry["id"] == trash_id)
    assert [entry["status"] for entry in record["source_results"]] == ["error", "error"]
    if not replace_after_crash:
        assert "不确定" in record["source_results"][0]["message"]
    assert remaining_path.read_bytes() == original_remaining
    if replace_after_crash:
        assert deleted_path.read_bytes() == b"a new unrelated file now occupies the former source path"


@pytest.mark.parametrize("job_type,status", [
    ("proxy", "running"), ("keyframe_scan", "pausing"), ("analysis", "cancelling"), ("index", "running"),
])
def test_running_or_stopping_jobs_block_deletion(client, sources, job_type, status):
    asset = register(client, sources[0])
    queue = client.app.state.jobs
    config = {"stages": ["vision"]} if job_type == "analysis" else {}
    job = queue.add(job_type, config, None if job_type == "index" else asset["id"])
    queue.update(job["id"], status=status)
    prepared = preview(client, [asset], delete_source=True)
    assert job["id"] in str(prepared["blocked_jobs"])
    assert_rejected(delete(client, [asset], prepared, delete_source=True))
    assert sources[0].is_file()
    assert client.get(f"/api/assets/{asset['id']}").status_code == 200


def test_other_episode_inflight_character_context_blocks_deletion(client, sources):
    project = client.post("/api/projects", json={"name": "合成动画", "kind": "animation"}).json()
    assets = [register(client, source, project_id=project["id"]) for source in sources]
    queue = client.app.state.jobs
    job = queue.add("analysis", {"stages": ["vision"]}, assets[1]["id"])
    queue.update(job["id"], status="running")
    prepared = preview(client, [assets[0]])
    assert job["id"] in str(prepared["blocked_jobs"])
    assert_rejected(delete(client, [assets[0]], prepared))


def test_deleted_asset_old_analysis_cannot_resume_or_write_run(client, sources):
    asset = register(client, sources[0])
    queue = client.app.state.jobs
    job = queue.add("analysis", {"stages": ["vision"]}, asset["id"])
    queue.update(job["id"], status="paused")
    result = delete(client, [asset], preview(client, [asset]))
    assert result.status_code == 200, result.text
    response = client.post(f"/api/jobs/{job['id']}/resume")
    assert response.status_code in {400, 404, 409}, response.text
    asyncio.run(queue.execute(job["id"]))
    assert queue.get(job["id"])["status"] != "completed"
    assert not (queue.library.asset_dir(asset["id"]) / "analyses" / job["run_id"] / "run.json").exists()
