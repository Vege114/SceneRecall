from fractions import Fraction

import numpy as np

from scenerecall.retrieval_sampling import RetrievalSampler, build_windows


def frame(at, shot="shot"):
    return {"id": f"f_{at}", "at_ms": at, "pts": at, "time_base": [1, 1000],
            "source_pts": at, "source_time_base": [1, 1000], "shot_id": shot}


def test_sampler_retains_short_shot_edges_and_local_transient():
    sampler = RetrievalSampler()
    for at in (0, 40, 80, 120, 160, 200):
        image = np.zeros((36, 64), dtype=np.uint8)
        if at == 80:
            image[:9, :16] = 255
        sampler.observe({**frame(at), "gray": image, "time": Fraction(at, 1000)})
    sampler.finish_shot(0, 240)
    times = {f["at_ms"] for f in sampler.frames}
    assert {0, 80, 200}.issubset(times)
    assert all(f["shot_id"] == "shot_000000000000_000000000240" for f in sampler.frames)


def test_dense_shots_split_units_without_losing_index_frames():
    frames = [frame(at, f"shot_{at}") for at in range(0, 12000, 250)]
    shots = [{"id": f"shot_{at}", "start_ms": at, "end_ms": at + 250} for at in range(0, 12000, 250)]
    units = build_windows("asset", "fingerprint", 0, 12000, frames, shots)
    assert units[0]["start_ms"] == 0 and units[-1]["end_ms"] == 12000
    assert all(a["end_ms"] == b["start_ms"] for a, b in zip(units, units[1:]))
    assert [f for unit in units for f in unit["frame_ids"]] == [f["id"] for f in frames]
    assert all(len(unit["local_frame_ids"]) <= 6 for unit in units)


def test_held_source_timestamp_is_not_replaced_with_window_start():
    units = build_windows("asset", "fingerprint", 0, 15000, [frame(0)],
                          [{"id": "shot", "start_ms": 0, "end_ms": 15000}])
    assert len(units) == 3
    assert units[-1]["start_ms"] == 12000
    assert units[-1]["ordered_frames"][0]["at_ms"] == 0
