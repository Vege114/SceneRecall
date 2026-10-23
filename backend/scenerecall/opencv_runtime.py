"""Configure OpenCV once at application startup, before local jobs run."""
from __future__ import annotations

import os


def configure_opencv_threads() -> int:
    import cv2

    try:
        threads = int(os.environ.get("SCENERECALL_OPENCV_THREADS", "1"))
    except ValueError as exc:
        raise ValueError("SCENERECALL_OPENCV_THREADS 必须是 1–32 的整数") from exc
    if not 1 <= threads <= 32:
        raise ValueError("SCENERECALL_OPENCV_THREADS 必须是 1–32 的整数")
    cv2.setNumThreads(threads)
    return threads
