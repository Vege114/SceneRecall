"""Full publications avoid redundant FTS scans; incremental replacement stays safe."""
import json
import re

import pytest

from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine
import scenerecall.search as search_module


def record(identity, text, *, kind="visual", start=0):
    return {"id": identity, "asset_id": "asset", "work_id": "work", "kind": kind,
            "text": text, "start_ms": start, "end_ms": start + 900}


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))


def trace_connections(engine, monkeypatch):
    statements = []
    original = engine._db

    def traced_db():
        db = original()
        db.set_trace_callback(statements.append)
        return db

    monkeypatch.setattr(engine, "_db", traced_db)
    return statements


def row_deletes(statements):
    # FTS5's id column is UNINDEXED; incremental deletes must constrain rowid.
    return [statement for statement in statements
            if re.match(r"^\s*DELETE FROM evidence_fts\s+WHERE\s+id\b", statement, re.I)]


@pytest.mark.asyncio
async def test_full_rebuild_clears_fts_once_without_per_record_delete(engine, monkeypatch):
    engine.upsert([record("retired", "retiredtoken")])
    generation = engine._generation
    statements = trace_connections(engine, monkeypatch)
    source = [record("cue-" + str(index), "dialogue " + str(index), kind="subtitle", start=index * 1000)
              for index in range(64)]

    result = await engine.rebuild(source)

    assert result["records"] == 64 and result["contexts"] == 125
    assert not row_deletes(statements)
    assert sum(statement.strip().upper() == "DELETE FROM EVIDENCE_FTS" for statement in statements) == 1
    assert engine._generation == generation + 1
    with engine._db() as db:
        assert db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 189
        assert db.execute("SELECT COUNT(*),COUNT(DISTINCT id) FROM evidence_fts").fetchone()[:] == (189, 189)
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE id='retired'").fetchone()[0] == 0
        published = [json.loads(row[0]) for row in db.execute("SELECT data FROM records")]
        assert db.execute("SELECT records_digest FROM build_checkpoints WHERE kind='lexical'").fetchone()[0] == engine._records_digest(published)

    statements.clear()
    resumed = await engine.rebuild(source)
    assert resumed["resumed"] is True
    assert not any(statement.lstrip().upper().startswith("DELETE FROM EVIDENCE_FTS") for statement in statements)
    assert engine._generation == generation + 1


def test_normal_upsert_still_replaces_one_fts_row_and_removes_old_tokens(engine, monkeypatch):
    engine.upsert([record("same-id", "obsoleteword")])
    statements = trace_connections(engine, monkeypatch)

    engine.upsert([record("same-id", "replacementword")])

    assert not row_deletes(statements)
    assert sum(bool(re.match(r"^DELETE FROM evidence_fts WHERE rowid=", statement))
               for statement in statements) == 1
    with engine._db() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE id='same-id'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'obsoleteword'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM evidence_fts WHERE evidence_fts MATCH 'replacementword'").fetchone()[0] == 1
        assert json.loads(db.execute("SELECT data FROM records WHERE id='same-id'").fetchone()[0])["text"] == "replacementword"


@pytest.mark.asyncio
async def test_failed_bulk_publication_rolls_back_clear_rows_and_checkpoint(engine, monkeypatch):
    await engine.rebuild([record("kept", "retainedtoken")])
    generation = engine._generation
    with engine._db() as db:
        checkpoint = db.execute("SELECT records_digest FROM build_checkpoints WHERE kind='lexical'").fetchone()[0]
    original = search_module.tokenize

    def failing_tokenize(text):
        if text == "explode":
            raise RuntimeError("tokenization interrupted")
        return original(text)

    monkeypatch.setattr(search_module, "tokenize", failing_tokenize)
    with pytest.raises(RuntimeError, match="tokenization interrupted"):
        await engine.rebuild([record("new", "newtoken"), record("broken", "explode")])

    assert engine._generation == generation
    with engine._db() as db:
        assert [row[0] for row in db.execute("SELECT id FROM records")] == ["kept"]
        assert [row[0] for row in db.execute("SELECT id FROM evidence_fts")] == ["kept"]
        assert db.execute("SELECT records_digest FROM build_checkpoints WHERE kind='lexical'").fetchone()[0] == checkpoint


@pytest.mark.asyncio
async def test_duplicate_ids_are_rejected_before_bulk_clear(engine):
    await engine.rebuild([record("kept", "retainedtoken")])
    with pytest.raises(ValueError, match="ID"):
        await engine.rebuild([record("duplicate", "first"), record("duplicate", "second")])
    with engine._db() as db:
        assert db.execute("SELECT id FROM records").fetchone()[0] == "kept"
        assert db.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0] == 1
