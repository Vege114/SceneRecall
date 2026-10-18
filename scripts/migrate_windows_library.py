"""Offline, allowlisted WSL -> Windows library migration (dry-run by default).

Run with the native Windows Python. The library is already shared through its
Windows drive: media and historical evidence stay in place. Only operational
path fields are rewritten. Stop both application instances before --apply or
--rollback. A complete byte-preserving backup and a recoverable journal are
written before the first mutation; no secrets or media titles are printed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone


AUDIO_PATH_FIELDS = frozenset({
    "diarization_model_path", "embedding_model_path", "asr_model_path", "calibration_path",
    "sync_python_path", "sync_repository_path", "sync_checkpoint_path", "sync_detector_path",
})
SOURCE_PATH_FIELDS = frozenset({"video_path", "subtitle_path"})
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


class MigrationError(ValueError):
    pass


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, content: bytes):
    temporary = path.with_name(path.name + ".migration-tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_reference(value: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise MigrationError("Invalid internal reference")
    windows = PureWindowsPath(value)
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if windows.drive or windows.root or posix.is_absolute() or ".." in posix.parts:
        raise MigrationError("Unsafe internal reference")
    return normalized


def map_path(value: str, overrides: dict[str, str], *, interpreter=False) -> str:
    """Convert a whole path value, never substrings of prose or raw outputs."""
    if not value:
        return value
    if value in overrides:
        return overrides[value]
    matched = re.fullmatch(r"/mnt/([a-zA-Z])/(.*)", value)
    if matched:
        if interpreter:
            raise MigrationError("Linux Python requires an explicit --python-map old=new")
        tail = PurePosixPath(matched[2])
        if ".." in tail.parts or "\0" in value or "\\" in value:
            raise MigrationError("Unsafe WSL mount path")
        return str(PureWindowsPath(matched[1].upper() + ":/") / tail)
    if value.startswith("/"):
        raise MigrationError("Non-drive Linux path requires an explicit --path-map old=new")
    return value


def operational_config(value: dict, overrides: dict[str, str]) -> dict:
    """Only current config containers and known fields can carry live paths."""
    result = copy.deepcopy(value)
    for key in AUDIO_PATH_FIELDS | SOURCE_PATH_FIELDS:
        if isinstance(result.get(key), str):
            result[key] = map_path(result[key], overrides, interpreter=key == "sync_python_path")
    for key in ("identity_settings", "audio", "audio_settings"):
        if isinstance(result.get(key), dict):
            result[key] = operational_config(result[key], overrides)
    if isinstance(result.get("items"), list):
        result["items"] = [operational_config(item, overrides) if isinstance(item, dict) else item
                           for item in result["items"]]
    return result


@dataclass
class Plan:
    root: Path
    files: dict[str, bytes] = field(default_factory=dict)
    original_hashes: dict[str, str] = field(default_factory=dict)
    jobs: list[tuple[str, str, str]] = field(default_factory=list)
    sources: list[tuple[Path, str]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, path: Path, original, updated):
        if original != updated:
            relative = path.relative_to(self.root).as_posix()
            self.files[relative] = encode(updated)
            self.original_hashes[relative] = sha256(path)

    def summary(self):
        return {**self.counts, "json_files_to_update": len(self.files), "sqlite_jobs_to_update": len(self.jobs),
                "source_sha256_verified": len(self.sources)}


def database(root: Path) -> Path:
    return root / "runtime" / "jobs.sqlite"


def readonly_database(path: Path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def build_plan(root: Path, overrides: dict[str, str] | None = None) -> Plan:
    root = root.resolve(strict=True)
    overrides = overrides or {}
    if not (root / "library.json").is_file():
        raise MigrationError("Not a SceneRecall library")
    plan = Plan(root, counts={"assets": 0, "active_references_verified": 0, "sqlite_jobs": 0})
    locations_path = root / "private" / "media-locations.json"
    locations = load(locations_path) if locations_path.is_file() else {}
    updated_locations = copy.deepcopy(locations)
    for asset_id, location in locations.items():
        if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,180}", asset_id) or asset_id in {".", ".."}:
            raise MigrationError("Invalid asset ID")
        original = location["path"] if isinstance(location, dict) else location
        target = Path(map_path(original, overrides))
        # An unavailable source remains offline. Never fabricate a source snapshot.
        if not target.is_file():
            raise MigrationError("A mapped original source is unavailable; no files were changed")
        manifest = load(root / "assets" / asset_id / "manifest.json")
        checksum = sha256(target)
        if checksum != manifest["fingerprint"]:
            raise MigrationError("Original source SHA256 differs from its imported fingerprint")
        stat = target.stat()
        updated_locations[asset_id] = {"path": str(target), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        plan.sources.append((target, checksum))
    plan.update(locations_path, locations, updated_locations)
    for manifest_path in (root / "assets").glob("*/manifest.json"):
        plan.counts["assets"] += 1
        directory = manifest_path.parent
        settings = directory / "audio" / "settings.json"
        if settings.is_file():
            original = load(settings)
            updated = operational_config(original, overrides)
            for key in AUDIO_PATH_FIELDS:
                if updated.get(key) and updated.get(key) != original.get(key) and not Path(updated[key]).exists():
                    raise MigrationError("A mapped local model or interpreter is unavailable")
            plan.update(settings, original, updated)
        for name in ("active-analysis.json", "frames/registry.json"):
            path = directory / name
            if not path.is_file():
                continue
            original = load(path)
            updated = copy.deepcopy(original)
            references = []
            if name == "active-analysis.json":
                observations = updated.get("observations", {})
                for record_id, reference in observations.items():
                    observations[record_id] = relative_reference(reference)
                    references.append(observations[record_id])
                references.extend(_reference_dicts(updated.get("subtitle_runs", {}).values()))
                if updated.get("container_subtitles"):
                    references.extend(_reference_dicts([updated["container_subtitles"]]))
            else:
                references.extend(_reference_dicts(updated.values()))
            for reference in references:
                target = (directory / reference).resolve()
                if not target.is_relative_to(directory.resolve()) or not target.is_file():
                    raise MigrationError("An active internal evidence reference is unavailable")
                plan.counts["active_references_verified"] += 1
            plan.update(path, original, updated)
    db_path = database(root)
    if db_path.is_file():
        with closing(readonly_database(db_path)) as db:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise MigrationError("Job SQLite integrity check failed")
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'").fetchone():
                for job_id, data in db.execute("SELECT id,data FROM jobs"):
                    job = json.loads(data)
                    plan.counts["sqlite_jobs"] += 1
                    if job.get("status") in {"running", "pausing", "cancelling"}:
                        raise MigrationError("An in-flight job must settle before migration")
                    # Finished jobs, model receipts, errors and request hashes are historical evidence.
                    if job.get("status") not in {"queued", "paused"}:
                        continue
                    updated = copy.deepcopy(job)
                    if isinstance(updated.get("config"), dict):
                        updated["config"] = operational_config(updated["config"], overrides)
                    if updated != job:
                        plan.jobs.append((job_id, data, json.dumps(updated, ensure_ascii=False, allow_nan=False)))
    return plan


def _reference_dicts(rows):
    references = []
    for row in rows:
        for key in ("path", "original_path"):
            if key in row:
                row[key] = relative_reference(row[key])
                references.append(row[key])
    return references


def inventory(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise MigrationError("Library symlinks must be reviewed before migration")
        if path.is_file() and not (path.parent.name == "runtime" and path.name.startswith("jobs.sqlite-")):
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def verify_sources(plan: Plan):
    for path, expected in plan.sources:
        if sha256(path) != expected:
            raise MigrationError("Original source changed during migration")


def apply_plan(plan: Plan, backup: Path):
    backup = backup.resolve()
    if backup == plan.root or backup.is_relative_to(plan.root):
        raise MigrationError("Backup must be outside the library")
    if backup.exists():
        raise MigrationError("Backup directory already exists; choose a fresh directory")
    before = inventory(plan.root)
    for relative, expected in plan.original_hashes.items():
        if before.get(relative) != expected:
            raise MigrationError("Library changed after planning; rebuild the plan")
    backup.mkdir(parents=True)
    shutil.copytree(plan.root, backup / "files")
    db_path = database(plan.root)
    if db_path.is_file():
        target_path = database(backup / "files")
        for suffix in SQLITE_SIDECARS:
            target_path.with_name(target_path.name + suffix).unlink(missing_ok=True)
        with closing(readonly_database(db_path)) as source, closing(sqlite3.connect(target_path)) as target:
            source.backup(target)
    backup_hashes = inventory(backup / "files")
    # A stable SQLite backup may have different container bytes after WAL consolidation.
    for relative, expected in before.items():
        if relative != "runtime/jobs.sqlite" and backup_hashes.get(relative) != expected:
            raise MigrationError("Backup verification failed before mutation")
    journal = {"version": 1, "library": str(plan.root), "status": "prepared",
               "created_at": datetime.now(timezone.utc).isoformat(), "before": before,
               "backup_hashes": backup_hashes, "changed": list(plan.files),
               "planned_hashes": {relative: hashlib.sha256(content).hexdigest()
                                  for relative, content in plan.files.items()},
               "sqlite_changed": bool(plan.jobs), "summary": plan.summary()}
    journal_path = backup / "migration.json"
    atomic_bytes(journal_path, encode(journal))
    try:
        for relative, content in plan.files.items():
            atomic_bytes(plan.root / relative, content)
        if plan.jobs:
            with closing(sqlite3.connect(db_path)) as db, db:
                db.execute("BEGIN IMMEDIATE")
                for job_id, previous, updated in plan.jobs:
                    cursor = db.execute("UPDATE jobs SET data=? WHERE id=? AND data=?", (updated, job_id, previous))
                    if cursor.rowcount != 1:
                        raise MigrationError("A SQLite job changed after planning")
        verify_sources(plan)
        after = inventory(plan.root)
        changed = set(plan.files) | ({"runtime/jobs.sqlite"} if plan.jobs else set())
        for relative, expected in before.items():
            if relative not in changed and after.get(relative) != expected:
                raise MigrationError("An unplanned library file changed during migration")
        journal.update(status="applied", after=after)
        atomic_bytes(journal_path, encode(journal))
    except Exception:
        rollback(backup, recovery=True)
        raise
    return plan.summary()


def rollback(backup: Path, *, recovery=False):
    journal_path = backup / "migration.json"
    journal = load(journal_path)
    if journal.get("version") != 1:
        raise MigrationError("Unsupported migration journal")
    if journal["status"] == "rolled_back":
        return {"status": "already_rolled_back"}
    root = Path(journal["library"]).resolve(strict=True)
    backup_root = backup / "files"
    for relative in journal["changed"] + (["runtime/jobs.sqlite"] if journal["sqlite_changed"] else []):
        relative_reference(relative)
        source = backup_root / relative
        if sha256(source) != journal["backup_hashes"][relative]:
            raise MigrationError("Backup integrity check failed")
        current = root / relative
        if not recovery:
            allowed = ({journal["after"][relative]} if journal["status"] == "applied" else
                       {journal["before"][relative], journal["planned_hashes"].get(relative)})
            if sha256(current) not in allowed:
                raise MigrationError("Library was edited after migration; automatic rollback refused")
    for relative in journal["changed"]:
        atomic_bytes(root / relative, (backup_root / relative).read_bytes())
    if journal["sqlite_changed"]:
        destination = database(root)
        atomic_bytes(destination, database(backup_root).read_bytes())
        for suffix in SQLITE_SIDECARS:
            destination.with_name(destination.name + suffix).unlink(missing_ok=True)
    journal["status"] = "rolled_back"
    atomic_bytes(journal_path, encode(journal))
    return {"status": "rolled_back", "json_files_restored": len(journal["changed"])}


def verify_backup(backup: Path):
    journal = load(backup / "migration.json")
    if inventory(backup / "files") != journal["backup_hashes"]:
        raise MigrationError("Full backup integrity check failed")
    return {"status": journal["status"], "backup_files_verified": len(journal["backup_hashes"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--apply", action="store_true", help="Requires both application instances stopped")
    parser.add_argument("--dry-run", action="store_true", help="Default: read, hash and plan only")
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--python-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--rollback", type=Path, metavar="BACKUP_DIR")
    parser.add_argument("--verify-backup", type=Path, metavar="BACKUP_DIR")
    args = parser.parse_args(argv)
    if args.rollback:
        result = rollback(args.rollback.resolve())
    elif args.verify_backup:
        result = verify_backup(args.verify_backup.resolve())
    else:
        if not args.library or (args.apply and (args.dry_run or not args.backup_dir)):
            parser.error("--library is required; --apply requires --backup-dir and excludes --dry-run")
        overrides = {}
        for entry in args.python_map + args.path_map:
            if "=" not in entry:
                parser.error("Mappings use OLD=NEW")
            old, new = entry.split("=", 1)
            if not old or not new or old in overrides:
                parser.error("Mappings must be nonempty and unique")
            overrides[old] = new
        plan = build_plan(args.library, overrides)
        result = {"mode": "apply" if args.apply else "dry_run", **plan.summary()}
        if args.apply:
            apply_plan(plan, args.backup_dir)
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (MigrationError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        # OS messages can contain private filenames. Do not echo their arguments.
        print(json.dumps({"error": str(exc) if isinstance(exc, MigrationError) else type(exc).__name__}), file=sys.stderr)
        sys.exit(1)
