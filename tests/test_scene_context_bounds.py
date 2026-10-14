from copy import deepcopy

import pytest

from scenerecall.jobs import CharacterContextChanged
from scenerecall.models import AnalysisInput
from scenerecall.providers import ProviderError
from scenerecall.scene_jobs import encoded_size
from test_scene_jobs import scene_worker  # noqa: F401


def test_scene_bounds_large_cast_before_legacy_validation_and_preserves_full_dossiers(scene_worker, monkeypatch):  # noqa: F811
    worker = scene_worker(1)
    context = worker.queue.characters.context(worker.asset["id"])
    context["profiles"] = [{"id": f"character_{index}", "name": "人物", "aliases": [],
        "description": "人工描述" * 500, "notes": "人工备注" * 800,
        "appearance": ["可见外观" * 200] * 30, "observed_traits": ["短发"] * 20,
        "edited_fields": ["notes", "description", "appearance"], "representative": None,
        "updated_at": f"{index:05d}"} for index in range(100)]
    original = deepcopy(context)
    monkeypatch.setattr(worker.queue.characters, "context", lambda asset_id: deepcopy(context))
    with pytest.raises(ProviderError, match="上下文超过"):
        worker.queue.character_context(worker.asset["id"])
    # Real new-job validation used to hit the whole-cast legacy limit here.
    _, config = worker.queue.validate(AnalysisInput(asset_id=worker.asset["id"], stages=["vision"],
        analysis_strategy="scene-chunks-v1"), {"bindings": {"preparation": "fast", "vision": "deep"}})
    bounded = worker.queue._scene_context(worker.asset["id"], config)
    assert 0 < len(bounded["profiles"]) <= 16
    assert bounded["profiles"][0]["id"] == "character_99"
    assert encoded_size(bounded) <= 18000
    assert "完整人工档案仍保留" in bounded["profiles"][0]["notes"]
    assert context == original
    context["revision"] += 1
    with pytest.raises(CharacterContextChanged):
        worker.queue._scene_context(worker.asset["id"], config)
