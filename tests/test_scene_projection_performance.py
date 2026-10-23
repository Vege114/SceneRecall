"""Dependency walks must batch identity reads without retaining stale decisions."""
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

from scenerecall.scenes import SceneStore


def test_dependency_walk_reuses_each_scene_and_refreshes_identity_next_read(monkeypatch):
    rows = []
    for index in range(40):
        rows.append({"id": f"scene_{index}", "asset_id": "asset_one", "start_ms": index * 1000,
                     "summary": str(index), "inferences": [{"text": "inference"}],
                     "entities": [{"id": "person", "identity_region_id": f"region_{index}",
                                   "character_id": "character_old"}],
                     "memory_before": {"provenance": [
                         {"asset_id": "asset_one", "chunk_id": row["id"],
                          "revision_digest": SceneStore.revision_digest(row)} for row in rows[-8:]]}})
    original = deepcopy(rows)
    calls = Counter()
    decision = {"character_id": "character_old"}

    def active(asset_id):
        calls["active"] += 1
        return {}

    library = SimpleNamespace(get_asset=lambda *args, **kwargs: {"anime_id": "anime"}, active=active)
    store = SceneStore(library)

    def raw(asset_id):
        calls["raw"] += 1
        return deepcopy(rows)

    def decorate(library, asset_id, records):
        calls["identity_batch"] += 1
        assert len(records) == len(rows)
        result = deepcopy(records)
        result[0]["entities"][0]["character_id"] = decision["character_id"]
        return result

    project = store._project_semantics

    def semantics(*args):
        calls["semantics"] += 1
        return project(*args)

    monkeypatch.setattr(store, "_raw", raw)
    monkeypatch.setattr(store, "_annotations", lambda asset_id: [])
    monkeypatch.setattr(store, "dependencies", lambda asset_id: {})
    monkeypatch.setattr(store, "_project_semantics", semantics)
    monkeypatch.setattr("scenerecall.visual_identity.decorate_identity_records", decorate)

    assert not any(row["context_stale"] for row in store.list("asset_one"))
    assert calls == {"raw": 1, "active": 1, "identity_batch": 1, "semantics": len(rows)}
    calls.clear()
    decision["character_id"] = "character_corrected"
    result = store.list("asset_one")
    assert "关联身份已纠正，相关剧情推断待复核" in result[0]["stale_reasons"]
    assert all("前序场景上下文已变化" in row["stale_reasons"] for row in result[1:])
    assert calls == {"raw": 1, "active": 1, "identity_batch": 1, "semantics": len(rows)}
    assert rows == original
    assert store.list("asset_one", include_stale=False) == []
