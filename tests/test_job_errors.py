"""Job diagnostics should identify the invariant without dumping source content."""
import pytest
from pydantic import ValidationError

from scenerecall.jobs import error_message
from scenerecall.models import Observation


def test_validation_failure_reports_reason_without_raw_observation():
    record = {"id": "fixture", "asset_id": "fixture", "run_id": "fixture", "window_id": "fixture",
              "shot_id": "fixture", "start_ms": 0, "end_ms": 10, "summary": "private synthetic source text",
              "evidence_frame_ids": ["current"], "events": [{"evidence_frame_ids": ["foreign-secret-frame"]}]}
    with pytest.raises(ValidationError) as caught:
        Observation.model_validate(record)
    message = error_message(caught.value)
    assert message == "分析结果校验失败：事件引用不存在的证据帧"
    for private in ("private synthetic", "foreign-secret-frame", "input_value", "input_type", "https://"):
        assert private not in message


def test_safe_provider_errors_remain_readable_and_unexpected_errors_do_not_expose_details():
    assert error_message(ValueError("模型连接请求超时")) == "模型连接请求超时"
    assert "sensitive exception text" not in error_message(RuntimeError("sensitive exception text"))
