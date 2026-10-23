import cv2
import pytest

from scenerecall.opencv_runtime import configure_opencv_threads


def test_opencv_thread_configuration(monkeypatch):
    configured = []
    monkeypatch.setattr(cv2, "setNumThreads", configured.append)
    monkeypatch.delenv("SCENERECALL_OPENCV_THREADS", raising=False)
    assert configure_opencv_threads() == 1
    monkeypatch.setenv("SCENERECALL_OPENCV_THREADS", "4")
    assert configure_opencv_threads() == 4
    assert configured == [1, 4]


@pytest.mark.parametrize("value", ["0", "-1", "33", "auto", "1.5"])
def test_invalid_opencv_threads_do_not_modify_runtime(monkeypatch, value):
    configured = []
    monkeypatch.setattr(cv2, "setNumThreads", configured.append)
    monkeypatch.setenv("SCENERECALL_OPENCV_THREADS", value)
    with pytest.raises(ValueError, match="SCENERECALL_OPENCV_THREADS"):
        configure_opencv_threads()
    assert not configured
