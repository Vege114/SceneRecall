"""Optional deep input selection never discards the complete keyframe index."""
import json

import pytest

from scenerecall.scene_preparation import PreparedChunk, validate_visual_preparation
from test_scene_providers import chunk, images, manager, reply


def validate_selection(selected=None, **changes):
    frames = [{"id": f"frame-{index}", "at_ms": index * 1000} for index in range(4)]
    value = chunk([frame["id"] for frame in frames], needs_deep=True, reason="中段动作需要核查", **changes)
    if selected is not None:
        value["deep_frame_ids"] = selected
    return validate_visual_preparation({"chunks": [value], "unresolved": []}, frames, set(), has_context=False)


def test_deep_selection_keeps_all_indexed_frames_and_legacy_fallback():
    value = validate_selection(["frame-0", "frame-2", "frame-3"])["chunks"][0]
    assert value["frame_ids"] == ["frame-0", "frame-1", "frame-2", "frame-3"]
    assert value["deep_frame_ids"] == ["frame-0", "frame-2", "frame-3"]
    assert validate_selection([])["chunks"][0]["deep_frame_ids"] == []
    old = validate_selection()["chunks"][0]
    assert "deep_frame_ids" not in old
    assert PreparedChunk.model_validate(old).deep_frame_ids == []


@pytest.mark.parametrize("selected", [
    ["frame-0", "frame-3", "frame-2"],
    ["frame-0", "frame-2"],
    ["frame-1", "frame-3"],
    ["frame-0", "frame-3", "frame-3"],
    ["frame-0", "remembered-frame", "frame-3"],
    ["frame-0", "contact-sheet-1", "frame-3"],
    ["frame-0", None, "frame-3"],
    ["frame-0"] * 57,
    "frame-0",
])
def test_deep_selection_rejects_wrong_order_missing_representatives_and_invalid_ids(selected):
    with pytest.raises(ValueError):
        validate_selection(selected)


def test_deep_selection_retains_middle_representative():
    with pytest.raises(ValueError, match="全部代表帧"):
        validate_selection(["frame-0", "frame-3"], representative_frame_ids=["frame-0", "frame-2", "frame-3"])


@pytest.mark.asyncio
async def test_preparation_deep_selection_is_returned_without_an_additional_request(tmp_path):
    frames = images(tmp_path, 4)
    data = {"chunks": [chunk([frame["id"] for frame in frames], needs_deep=True, reason="手部动作需核查",
                            deep_frame_ids=["frame-0", "frame-2", "frame-3"])], "unresolved": []}
    requests = []
    providers = manager(tmp_path, lambda request: requests.append(request) or reply(data))
    result = await providers.prepare_visual("scene", frames, [], None, None)
    assert result.data == data
    assert result.request_count == len(requests) == 1
    prompt = json.loads(requests[0].content)["messages"][0]["content"]
    assert "scene-visual-preparation-v5" in prompt
    assert "For fast action or changes that cannot safely be summarized, select every core frame" in prompt
    assert "frame_ids must still retain every core frame" in prompt
