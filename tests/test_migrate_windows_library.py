"""Migration fixtures contain fabricated evidence and no user media or secrets."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from contextlib import closing

import pytest

from scenerecall.library import Library, atomic_json, source_snapshot


spec = importlib.util.spec_from_file_location(
    "migrate_windows_library", Path(__file__).parents[1] / "scripts/migrate_windows_library.py")
migration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = migration
spec.loader.exec_module(migration)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def fixture(tmp_path):
    root = tmp_path / "library"
    library = Library(root)
    source = tmp_path / "source with spaces.mp4"
    source.write_bytes(b"synthetic original source, never altered")
    asset = "asset_fixture"
    directory = root / "assets" / asset
    atomic_json(directory / "manifest.json", {
        "id": asset, "work_id": "work_fixture", "title": "synthetic", "created_at": "2026-10-18T00:00:00Z",
        "duration_ms": 1000, "version": "1", "fingerprint": digest(source.read_bytes()), "subtitle_mode": "embedded",
    })
    atomic_json(root / "private/media-locations.json", {
        asset: {**source_snapshot(source), "path": "/mnt/c/source with spaces.mp4"},
    })
    atomic_json(directory / "analyses/run/observations/obs.json", {"summary": "Keep /mnt/c/prose unchanged"})
    atomic_json(directory / "active-analysis.json", {
        "observations": {"obs": r"analyses\run\observations\obs.json"}, "subtitle_runs": {},
    })
    atomic_json(directory / "audio/settings.json", {"sync_python_path": "/mnt/d/worker/.venv/bin/python",
                                                 "sync_checkpoint_path": "/mnt/d/models/bundle",
                                                 "asr_model_path": "", "asr_enabled": False})
    executable = tmp_path / "native-python.exe"
    executable.write_bytes(b"synthetic worker marker")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    atomic_json(directory / "audio/lrasd-previews/old/request.json", {
        "repository": "/mnt/d/history", "checkpoint": "/mnt/d/history", "tracks": [],
    })
    atomic_json(root / "cache/model-feature.json", {"feature": [1, 2, 3], "model_version": "frozen"})
    mappings = {"/mnt/c/source with spaces.mp4": str(source), "/mnt/d/worker/.venv/bin/python": str(executable),
                "/mnt/d/models/bundle": str(bundle), "/mnt/c/queued.mp4": str(source)}
    # The migration runs on Windows; explicit identity mappings keep its
    # fabricated filesystem targets portable when exercising it on Linux.
    mappings.update({str(path): str(path) for path in (source, executable, bundle)})
    with closing(sqlite3.connect(root / "runtime/jobs.sqlite")) as db, db:
        db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        db.execute("CREATE TABLE requests (kind TEXT,id TEXT,payload_hash TEXT,result TEXT)")
        for identity, status in (("queued", "queued"), ("done", "completed")):
            job = {"status": status, "config": {"items": [{"video_path": "/mnt/c/queued.mp4",
                   "subtitle_path": None, "note": "Keep /mnt/c/prose unchanged"}]},
                   "raw_output": "/mnt/c/raw text", "error": "/mnt/c/error prose"}
            db.execute("INSERT INTO jobs VALUES (?,?)", (identity, json.dumps(job)))
        db.execute("INSERT INTO requests VALUES ('import','request','immutable-hash','{}')")
    library.get_asset(asset)
    return root, source, mappings, library


def test_mount_mapping_is_whole_value_and_python_is_never_guessed():
    assert migration.map_path("/mnt/d/目录 with spaces/model.onnx", {}) == r"D:\目录 with spaces\model.onnx"
    assert migration.map_path("A sentence mentioning /mnt/c/clip", {}) == "A sentence mentioning /mnt/c/clip"
    with pytest.raises(migration.MigrationError, match="Python requires"):
        migration.map_path("/mnt/d/.venv/bin/python", {}, interpreter=True)
    with pytest.raises(migration.MigrationError, match="explicit --path-map"):
        migration.map_path("/home/user/model", {})


@pytest.mark.parametrize("value", ["/mnt/c/../outside", "/mnt/c/nested/../../outside", "/mnt/c/bad\\name"])
def test_mount_mapping_rejects_ambiguous_or_escaping_paths(value):
    with pytest.raises(migration.MigrationError):
        migration.map_path(value, {})


def test_dry_run_does_not_write_library_or_rewrite_historical_evidence(tmp_path):
    root, source, mappings, _ = fixture(tmp_path)
    before = migration.inventory(root)
    plan = migration.build_plan(root, mappings)
    assert plan.summary() == {"assets": 1, "active_references_verified": 1, "sqlite_jobs": 2,
                              "json_files_to_update": 3, "sqlite_jobs_to_update": 1, "source_sha256_verified": 1}
    assert migration.inventory(root) == before
    assert digest(source.read_bytes()) == plan.sources[0][1]
    assert all("request.json" not in path and "cache/" not in path for path in plan.files)
    queued = json.loads(plan.jobs[0][2])
    assert queued["config"]["items"][0]["video_path"] == str(source)
    assert queued["config"]["items"][0]["note"] == "Keep /mnt/c/prose unchanged"
    assert queued["raw_output"] == "/mnt/c/raw text"


def test_apply_validates_sources_preserves_evidence_and_is_idempotent_and_reversible(tmp_path):
    root, source, mappings, library = fixture(tmp_path)
    before = migration.inventory(root)
    backup = tmp_path / "backup"
    plan = migration.build_plan(root, mappings)
    migration.apply_plan(plan, backup)
    assert library.get_asset("asset_fixture")["source_available"]
    assert not library.get_asset("asset_fixture")["source_changed"]
    assert migration.verify_backup(backup)["status"] == "applied"
    after = migration.inventory(root)
    changed = set(plan.files) | {"runtime/jobs.sqlite"}
    assert all(after[key] == value for key, value in before.items() if key not in changed)
    assert not migration.build_plan(root, mappings).files
    assert not migration.build_plan(root, mappings).jobs
    with closing(sqlite3.connect(root / "runtime/jobs.sqlite")) as db:
        assert db.execute("SELECT payload_hash FROM requests").fetchone()[0] == "immutable-hash"
        completed = json.loads(db.execute("SELECT data FROM jobs WHERE id='done'").fetchone()[0])
    assert completed["config"]["items"][0]["video_path"] == "/mnt/c/queued.mp4"
    migration.rollback(backup)
    restored = migration.inventory(root)
    assert all(restored[key] == value for key, value in before.items() if key != "runtime/jobs.sqlite")
    assert migration.rollback(backup) == {"status": "already_rolled_back"}
    assert digest(source.read_bytes()) == plan.sources[0][1]


def test_fingerprint_failure_changes_nothing(tmp_path):
    root, source, mappings, _ = fixture(tmp_path)
    source.write_bytes(b"replaced source")
    before = migration.inventory(root)
    with pytest.raises(migration.MigrationError, match="fingerprint"):
        migration.build_plan(root, mappings)
    assert migration.inventory(root) == before


def test_failed_apply_restores_both_json_and_sqlite(tmp_path, monkeypatch):
    root, _, mappings, _ = fixture(tmp_path)
    before = migration.inventory(root)
    plan = migration.build_plan(root, mappings)
    monkeypatch.setattr(migration, "verify_sources", lambda _: (_ for _ in ()).throw(RuntimeError("failure")))
    with pytest.raises(RuntimeError, match="failure"):
        migration.apply_plan(plan, tmp_path / "backup")
    after = migration.inventory(root)
    assert all(after[key] == value for key, value in before.items() if key != "runtime/jobs.sqlite")
    with closing(sqlite3.connect(root / "runtime/jobs.sqlite")) as db:
        config = json.loads(db.execute("SELECT data FROM jobs WHERE id='queued'").fetchone()[0])["config"]
    assert config["items"][0]["video_path"] == "/mnt/c/queued.mp4"


def test_rollback_refuses_to_overwrite_post_migration_edits(tmp_path):
    root, _, mappings, _ = fixture(tmp_path)
    backup = tmp_path / "backup"
    migration.apply_plan(migration.build_plan(root, mappings), backup)
    locations = root / "private/media-locations.json"
    locations.write_text("{}", encoding="utf-8")
    with pytest.raises(migration.MigrationError, match="edited after"):
        migration.rollback(backup)
    assert locations.read_text(encoding="utf-8") == "{}"


def test_queued_linux_python_without_explicit_mapping_or_live_job_blocks_migration(tmp_path):
    root, _, mappings, _ = fixture(tmp_path)
    mappings.pop("/mnt/d/worker/.venv/bin/python")
    with pytest.raises(migration.MigrationError, match="Python requires"):
        migration.build_plan(root, mappings)
    mappings["/mnt/d/worker/.venv/bin/python"] = str(tmp_path / "native-python.exe")
    with closing(sqlite3.connect(root / "runtime/jobs.sqlite")) as db, db:
        db.execute("UPDATE jobs SET data=? WHERE id='queued'", (json.dumps({"status": "running"}),))
    with pytest.raises(migration.MigrationError, match="in-flight"):
        migration.build_plan(root, mappings)


def test_active_reference_escape_or_missing_file_blocks_before_mutation(tmp_path):
    root, _, mappings, _ = fixture(tmp_path)
    active = root / "assets/asset_fixture/active-analysis.json"
    atomic_json(active, {"observations": {"obs": r"..\outside.json"}})
    with pytest.raises(migration.MigrationError, match="Unsafe internal"):
        migration.build_plan(root, mappings)
    atomic_json(active, {"observations": {"obs": "missing.json"}})
    with pytest.raises(migration.MigrationError, match="unavailable"):
        migration.build_plan(root, mappings)


def test_backup_inside_library_is_rejected(tmp_path):
    root, _, mappings, _ = fixture(tmp_path)
    with pytest.raises(migration.MigrationError, match="outside the library"):
        migration.apply_plan(migration.build_plan(root, mappings), root / "backup")
