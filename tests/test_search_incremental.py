"""Incremental FTS work uses stable integer rows and preserves old-index upgrades."""
import asyncio
import re
import threading
from unittest.mock import patch

import pytest

from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine
import scenerecall.search as search_module


def record(identity, text, *, asset="asset", kind="visual", start=0):
    return {"id": identity, "asset_id": asset, "work_id": "work", "kind": kind,
            "text": text, "start_ms": start, "end_ms": start + 900}


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))


def snapshot(engine):
    with engine._db() as db:
        return {"records": [tuple(row) for row in db.execute("SELECT rowid,* FROM records ORDER BY rowid")],
                "fts": [tuple(row) for row in db.execute("SELECT rowid,* FROM evidence_fts ORDER BY rowid")],
                "checkpoints": [tuple(row) for row in db.execute("SELECT * FROM build_checkpoints")]}


def trace(engine, monkeypatch):
    statements = []
    original = engine._db

    def connection():
        db = original()
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(engine, "_db", connection)
    return statements


def assert_aligned(engine):
    with engine._db() as db:
        assert not db.execute("""SELECT r.id FROM records r LEFT JOIN evidence_fts f ON f.rowid=r.rowid
                                 WHERE f.id IS NULL OR f.id<>r.id""").fetchall()
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == db.execute(
            "SELECT COUNT(*) FROM evidence_fts").fetchone()[0]


def test_legacy_migration_aligns_rows_once_and_preserves_tokens_and_vector_metadata(engine, monkeypatch):
    rows = [record("first", "firsttoken"), record("second", "secondtoken"), record("missing", "missingtoken")]
    engine.upsert(rows)
    with engine._db() as db:
        tokens = {row[0]: row[1] for row in db.execute("SELECT id,tokens FROM evidence_fts")}
        db.execute("DELETE FROM evidence_fts")
        db.executemany("INSERT INTO evidence_fts(rowid,id,tokens) VALUES (?,?,?)",
                       [(100, "first", tokens["first"]), (200, "second", tokens["second"]),
                        (300, "first", tokens["first"]), (400, "orphan", "orphanword")])
        db.execute("INSERT INTO namespaces VALUES ('sig','embed','model','vectors_saved',3,3,1)")
        db.execute("INSERT INTO vector_cache VALUES ('sig','hash','[1,0,0]')")
        db.execute("INSERT INTO build_checkpoints VALUES ('vector:sig','saved-digest')")
        db.execute("PRAGMA user_version=0")
    original = search_module.tokenize
    tokenized = []

    def counted(text):
        tokenized.append(text)
        return original(text)

    monkeypatch.setattr(search_module, "tokenize", counted)
    upgraded = SearchEngine(engine.index_dir, engine.providers)

    assert tokenized == ["missingtoken"]
    assert_aligned(upgraded)
    with upgraded._db() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute("SELECT table_name FROM namespaces").fetchone()[0] == "vectors_saved"
        assert db.execute("SELECT vector FROM vector_cache").fetchone()[0] == "[1,0,0]"
        assert db.execute("SELECT records_digest FROM build_checkpoints").fetchone()[0] == "saved-digest"
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'orphanword'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'firsttoken'").fetchone()[0] == 1
    before = snapshot(upgraded)
    tokenized.clear()
    reopened = SearchEngine(engine.index_dir, engine.providers)
    assert snapshot(reopened) == before and tokenized == []


def test_failed_legacy_migration_rolls_back_original_fts_and_version(engine, monkeypatch):
    engine.upsert([record("kept", "retainedtoken"), record("missing", "missingtoken")])
    with engine._db() as db:
        db.execute("DELETE FROM evidence_fts WHERE rowid=(SELECT rowid FROM records WHERE id='missing')")
        db.execute("PRAGMA user_version=0")
    before = snapshot(engine)

    def fail(_text):
        raise RuntimeError("migration interrupted")

    monkeypatch.setattr(search_module, "tokenize", fail)
    with pytest.raises(RuntimeError, match="migration interrupted"):
        SearchEngine(engine.index_dir, engine.providers)
    assert snapshot(engine) == before
    with engine._db() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='evidence_fts_rowids'").fetchone()


@pytest.mark.asyncio
async def test_identical_upsert_is_noop_preserving_context_rows_checkpoint_and_generation(engine, monkeypatch):
    rows = [record("s1", "firsttoken", kind="subtitle"),
            record("s2", "secondtoken", kind="subtitle", start=1000)]
    await engine.rebuild(rows)
    before, generation = snapshot(engine), engine._generation
    statements = trace(engine, monkeypatch)
    with patch.object(engine, "_write_record", wraps=engine._write_record) as write:
        engine.upsert(rows)
    assert write.call_count == 0
    assert snapshot(engine) == before and engine._generation == generation
    assert not any(re.match(r"^\s*(?:INSERT|DELETE|UPDATE)\b", sql, re.I) for sql in statements)
    assert (await engine.rebuild(rows))["resumed"] is True


def test_new_rows_skip_fts_deletes_and_updates_keep_integer_row_identity(engine, monkeypatch):
    statements = trace(engine, monkeypatch)
    engine.upsert([record("same", "obsoleteword")])
    assert not any(re.match(r"^\s*DELETE FROM evidence_fts\b", sql, re.I) for sql in statements)
    with engine._db() as db:
        old_rowid = db.execute("SELECT rowid FROM records WHERE id='same'").fetchone()[0]
    engine.upsert([record("same", "replacementword")])
    with engine._db() as db:
        assert db.execute("SELECT rowid FROM records WHERE id='same'").fetchone()[0] == old_rowid
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'obsoleteword'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'replacementword'").fetchone()[0] == 1
    assert_aligned(engine)
    assert not any(re.match(r"^\s*DELETE FROM evidence_fts WHERE id\b", sql, re.I) for sql in statements)


def test_contexts_change_only_near_edited_cue_and_upsert_retains_other_sources(engine):
    rows = [record("s" + str(i), "word" + str(i), kind="subtitle", start=i * 1000) for i in range(6)]
    engine.upsert(rows)
    before = snapshot(engine)
    with patch.object(engine, "_write_record", wraps=engine._write_record) as write:
        engine.upsert([{**rows[0], "text": "replacementword"}])
    # Only the first cue and its two/three-cue contexts change.
    assert write.call_count == 3
    assert engine.status()["records"] == 6 and engine.status()["contexts"] == 9
    after = snapshot(engine)
    assert [row[0:2] for row in before["records"]] == [row[0:2] for row in after["records"]]
    assert_aligned(engine)


def test_sync_retirement_and_asset_removal_use_rowid_constraints(engine, monkeypatch):
    rows = [record("s1", "retiredword", kind="subtitle"),
            record("s2", "retainedword", kind="subtitle", start=1000)]
    engine.upsert([*rows, record("other", "otherword", asset="other")])
    statements = trace(engine, monkeypatch)
    result = engine.sync_asset([rows[1]])
    assert result == {"upserted": 0, "removed": 1, "contexts_changed": 1}
    with engine._db() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'retiredword'").fetchone()[0] == 0
    engine.remove_assets(["asset"])
    assert_aligned(engine)
    with engine._db() as db:
        assert [row[0] for row in db.execute("SELECT id FROM records")] == ["other"]
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'otherword'").fetchone()[0] == 1
        for sql, args in [
            ("DELETE FROM evidence_fts WHERE rowid=?", (1,)),
            ("DELETE FROM evidence_fts WHERE rowid=(SELECT rowid FROM records WHERE id=?)", ("other",)),
            ("DELETE FROM evidence_fts WHERE rowid IN (SELECT rowid FROM records WHERE asset_id=?)", ("other",)),
        ]:
            plan = [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + sql, args)]
            fts_plan = [detail for detail in plan if "evidence_fts" in detail]
            assert len(fts_plan) == 1 and fts_plan[0].endswith(":="), plan
    assert not any(re.match(r"^\s*DELETE FROM evidence_fts WHERE id\b", sql, re.I) for sql in statements)


def test_failed_incremental_sync_rolls_back_retired_sources_contexts_and_rowids(engine, monkeypatch):
    rows = [record("s1", "retainedword", kind="subtitle"),
            record("s2", "secondword", kind="subtitle", start=1000)]
    engine.upsert(rows)
    before, generation = snapshot(engine), engine._generation

    def fail(_text):
        raise RuntimeError("incremental interrupted")

    monkeypatch.setattr(search_module, "tokenize", fail)
    with pytest.raises(RuntimeError, match="incremental interrupted"):
        engine.sync_asset([{**rows[1], "text": "replacementword"}])
    assert snapshot(engine) == before and engine._generation == generation
    assert_aligned(engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "remove"])
async def test_source_snapshot_rejects_edits_or_deletions_before_rebuild_starts(engine, change):
    old = record("kept", "oldword")
    await engine.rebuild([old])
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def source():
        rows = [dict(old)]
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5), "source release timed out"
        return rows

    rebuilding = asyncio.create_task(engine.rebuild_from_source(source))
    try:
        await asyncio.wait_for(started.wait(), 5)
        if change == "edit":
            engine.sync_asset([{**old, "text": "newword"}])
        else:
            engine.remove_assets(["asset"])
        after_change = snapshot(engine)
    finally:
        release.set()
    with pytest.raises(ValueError, match="索引构建期间资料发生变化"):
        await rebuilding
    assert snapshot(engine) == after_change
    assert_aligned(engine)


@pytest.mark.asyncio
async def test_stale_expected_generation_cannot_reuse_matching_checkpoint(engine):
    rows = [record("kept", "retainedword")]
    await engine.rebuild(rows)
    old_generation = engine._generation
    # A publication may retain an identical lexical checkpoint, while still
    # advancing generation (for example, publishing a vector namespace).
    engine._publish(rows, None, old_generation)
    before = snapshot(engine)
    with pytest.raises(ValueError, match="索引构建期间资料发生变化"):
        await engine.rebuild(rows, expected_generation=old_generation)
    assert snapshot(engine) == before
    assert (await engine.rebuild_from_source(lambda: rows))["resumed"] is True


@pytest.mark.asyncio
async def test_source_wrapper_forwards_embedding_budget_callbacks(engine):
    engine.providers.upsert({"id": "embed", "name": "Synthetic embedding", "model": "synthetic",
                             "base_url": "https://fixture.invalid/v1", "capabilities": ["embedding"],
                             "secret_mode": "none"})
    requests = []

    def stop(count):
        requests.append(count)
        raise ValueError("budget exhausted")

    with pytest.raises(ValueError, match="budget exhausted"):
        await engine.rebuild_from_source(lambda: [record("new", "newword")], "embed", before_batch=stop)
    assert requests == [1]
    assert engine.status()["records"] == 0
