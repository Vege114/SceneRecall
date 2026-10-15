"""One local inference admission gate shared by optional vision/audio backends.

Model inference is bounded independently of remote provider concurrency. Decoding,
cache reads and network model calls must not hold this lock.
"""
import threading

LOCAL_MODEL_GATE = threading.RLock()
