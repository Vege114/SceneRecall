"""Versioned retrieval sampling alongside, never in place of, the legacy plan."""
from __future__ import annotations

import copy
from fractions import Fraction

import numpy as np

from .library import digest

VERSION = "retrieval-sampling-v1"
PARAMETERS = {"interval_ms": 500, "core_ms": 6000, "context_ms": 1000,
              "region_change_min": .12, "grid_rows": 4, "grid_columns": 4,
              "local_max_frames": 6}


def metadata(frame):
    return {key: copy.deepcopy(frame[key]) for key in
            ("id", "at_ms", "pts", "time_base", "source_pts", "source_time_base", "shot_id", "reason")
            if key in frame}


class RetrievalSampler:
    """Inspect the already-decoded low-resolution frames with bounded memory."""

    def __init__(self):
        self.frames = []
        self.selected = {}
        self.first = self.last = None
        self.previous = None
        self.previous_change = 0.0
        self.pending_peak = None
        self.last_periodic = None

    def _keep(self, frame, reason):
        item = metadata(frame)
        item["reason"] = reason
        self.selected[(item["pts"], tuple(item["time_base"]))] = item

    def observe(self, frame):
        if self.first is None:
            self.first = frame
        self.last = frame
        at = frame["at_ms"]
        if self.last_periodic is None or at - self.last_periodic >= PARAMETERS["interval_ms"]:
            self._keep(frame, "retrieval_periodic")
            self.last_periodic = at
        if self.previous is not None:
            delta = np.abs(frame["gray"].astype(np.int16) - self.previous.astype(np.int16)) / 255
            tiles = [float(tile.mean()) for row in np.array_split(delta, PARAMETERS["grid_rows"])
                     for tile in np.array_split(row, PARAMETERS["grid_columns"], axis=1)]
            change = max(tiles, default=0)
            if self.pending_peak is not None and self.previous_change >= change:
                self._keep(self.pending_peak, "retrieval_region_peak")
            self.pending_peak = frame if change >= PARAMETERS["region_change_min"] else None
            self.previous_change = change
        self.previous = frame["gray"].copy()

    def finish_shot(self, start_ms, end_ms):
        if self.first is not None:
            self._keep(self.first, "retrieval_shot_first")
            self._keep(self.last, "retrieval_shot_last")
            if self.pending_peak is not None:
                self._keep(self.pending_peak, "retrieval_region_peak")
            shot = f"shot_{start_ms:012d}_{end_ms:012d}"
            self.frames.extend({**item, "shot_id": shot} for item in self.selected.values())
        self.selected = {}
        self.first = self.last = self.previous = self.pending_peak = None
        self.previous_change = 0.0
        self.last_periodic = None


def build_windows(asset_id, fingerprint, start_ms, end_ms, frames, shots, cues=()):
    """Cover the entire range; VLM units and image-index frames stay distinct."""
    ordered = sorted(frames, key=lambda item: (Fraction(item["pts"]) * Fraction(*item["time_base"]), item["id"]))
    if not ordered or len({f["id"] for f in ordered}) != len(ordered):
        raise ValueError("检索采样没有有效帧或包含重复帧")
    if any(not start_ms <= f["at_ms"] < end_ms for f in ordered):
        raise ValueError("检索证据帧超出范围")
    units = []

    def add(a, b):
        current = [f for f in ordered if a <= f["at_ms"] < b]
        held_spans = []
        if not current:
            # VFR held pictures may begin before this core. Keep that exact source
            # timestamp and identify it as held evidence, not a fabricated new frame.
            previous = [f for f in ordered if f["at_ms"] < a]
            if not previous:
                raise ValueError("局部范围缺少可见源帧")
            current = [previous[-1]]
            held_spans = [{"frame_id": current[0]["id"], "start_ms": a, "end_ms": b}]
        shot_frames = {}
        for f in current:
            shot_frames.setdefault(f.get("shot_id"), f)
        if len(shot_frames) > PARAMETERS["local_max_frames"] and b - a > 1:
            midpoint = current[len(current) // 2]["at_ms"]
            if a < midpoint < b:
                add(a, midpoint)
                add(midpoint, b)
                return
        indices = np.linspace(0, len(current) - 1, min(4, len(current)), dtype=int)
        selected = {current[int(i)]["id"]: current[int(i)] for i in indices}
        selected.update({f["id"]: f for f in shot_frames.values()})
        if len(selected) > PARAMETERS["local_max_frames"]:
            # All shot representatives win over uniform samples.
            selected = {f["id"]: f for f in shot_frames.values()}
            for i in indices:
                if len(selected) >= PARAMETERS["local_max_frames"]:
                    break
                selected[current[int(i)]["id"]] = current[int(i)]
        context = [f for f in ordered if max(start_ms, a - PARAMETERS["context_ms"]) <= f["at_ms"]
                   < min(end_ms, b + PARAMETERS["context_ms"]) and not a <= f["at_ms"] < b]
        identifier = "rw_" + digest([asset_id, fingerprint, VERSION, a, b, [f["id"] for f in current]])
        units.append({"id": identifier, "asset_id": asset_id, "start_ms": a, "end_ms": b,
                      "source_fingerprint": fingerprint, "sampling_version": VERSION,
                      "shot_ids": [s["id"] for s in shots if s["start_ms"] < b and s["end_ms"] > a],
                      "frame_ids": [f["id"] for f in current],
                      "evidence_frame_ids": [f["id"] for f in current],
                      "ordered_frames": [metadata(f) for f in current],
                      "local_frame_ids": [f["id"] for f in sorted(selected.values(), key=lambda f: f["at_ms"])],
                      "context_frame_ids": [f["id"] for f in context],
                      "held_frame_spans": held_spans,
                      "subtitle_ids": [c["id"] for c in cues if c["start_ms"] < b and c["end_ms"] > a]})

    for a in range(start_ms, end_ms, PARAMETERS["core_ms"]):
        add(a, min(end_ms, a + PARAMETERS["core_ms"]))
    return units
