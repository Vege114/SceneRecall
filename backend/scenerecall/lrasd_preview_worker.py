"""Offline per-frame anime body detection for bounded LR-ASD previews.

This worker needs only NumPy, OpenCV and ONNX Runtime in the visual environment.
Every emitted box was detected in that exact frame; gaps and cuts end tracks.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

DETECTION_VERSION = "anime-person-onnx640-perframe-iou-v1"
TRACKING = {"threshold": .35, "iou_min": .30, "ambiguity_margin": .10,
            "cut_mean_absolute_difference": 32., "minimum_frames": 16, "maximum_tracks": 8,
            "gap_interpolation": False}


def iou(left, right):
    x, y = max(left[0], right[0]), max(left[1], right[1])
    end_x = min(left[0] + left[2], right[0] + right[2])
    end_y = min(left[1] + left[3], right[1] + right[3])
    intersection = max(0., end_x - x) * max(0., end_y - y)
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.


def link_tracks(detections, cuts, start_ms, *, width=480, height=270, minimum_frames=16, maximum_tracks=8):
    """Consecutive, mutual unique IoU matches; ambiguous crossings start anew."""
    if len(detections) != len(cuts):
        raise ValueError("Detection and cut grids differ")
    active, finished = [], []
    for index, observed in enumerate(detections):
        current = []
        for row in observed:
            box = row["box_xywh"]
            if (len(box) != 4 or not all(math.isfinite(float(value)) for value in box)
                    or min(box[0], box[1]) < 0 or min(box[2:]) < 2
                    or box[0] + box[2] > width + .001 or box[1] + box[3] > height + .001):
                continue
            current.append({"box": list(map(float, box)), "score": float(row["score"])})
        # A broad crop containing another detected person's centre is unsafe
        # for both ASD and CCIP. Keep the smaller isolated person's own box.
        current = [row for row in current if not any(
            other is not row and row["box"][0] < other["box"][0] + other["box"][2] / 2 < row["box"][0] + row["box"][2]
            and row["box"][1] < other["box"][1] + other["box"][3] / 2 < row["box"][1] + row["box"][3]
            for other in current)]
        current.sort(key=lambda row: tuple(row["box"]))
        if cuts[index]:
            finished.extend(active)
            active = []
        matrix = [[iou(track["observations"][-1]["pixel_box"], row["box"]) for row in current] for track in active]

        def unique_best(values):
            ranked = sorted(enumerate(values), key=lambda pair: (-pair[1], pair[0]))
            if not ranked or ranked[0][1] < TRACKING["iou_min"]:
                return None
            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < TRACKING["ambiguity_margin"]:
                return None
            return ranked[0][0]

        forward = [unique_best(values) for values in matrix]
        reverse = [unique_best([matrix[t][d] for t in range(len(active))]) for d in range(len(current))]
        linked, used = [], set()
        for t, track in enumerate(active):
            d = forward[t]
            if d is None or reverse[d] != t:
                finished.append(track)
                continue
            used.add(d)
            row = current[d]
            track["observations"].append({"frame": index, "pixel_box": row["box"], "score": row["score"]})
            linked.append(track)
        for d, row in enumerate(current):
            if d not in used:
                linked.append({"observations": [{"frame": index, "pixel_box": row["box"], "score": row["score"]}]})
        active = linked
    finished.extend(active)
    accepted = [track for track in finished if len(track["observations"]) >= minimum_frames]
    accepted.sort(key=lambda track: (-len(track["observations"]), track["observations"][0]["frame"],
                                      tuple(track["observations"][0]["pixel_box"])))
    accepted = accepted[:maximum_tracks]
    accepted.sort(key=lambda track: (track["observations"][0]["frame"], tuple(track["observations"][0]["pixel_box"])))
    result = []
    for track in accepted:
        observations = track["observations"]
        first, last = observations[0]["frame"], observations[-1]["frame"]
        if len(observations) != last - first + 1:
            raise ValueError("Refusing a track with interpolated or missing frames")
        identifier = "track_" + hashlib.sha256(json.dumps([start_ms, first, last, observations[0]["pixel_box"]]).encode()).hexdigest()[:20]
        result.append({"id": identifier, "character_id": None, "association_status": "candidate",
                       "start_ms": start_ms + first * 40, "end_ms": start_ms + (last + 1) * 40,
                       "boxes": [{"at_ms": start_ms + row["frame"] * 40,
                                  "box": [row["pixel_box"][0] / width, row["pixel_box"][1] / height,
                                          row["pixel_box"][2] / width, row["pixel_box"][3] / height],
                                  "detection_score": row["score"]} for row in observations],
                       "detected_frames": len(observations), "interpolated_frames": 0})
    return result


class LocalPersonDetector:
    def __init__(self, path, threshold=.35):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.threshold = threshold

    def __call__(self, rgb):
        import cv2
        import numpy as np
        height, width = rgb.shape[:2]
        side, scale = 640, min(640 / width, 640 / height)
        resized = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
        top, left = (side - resized.shape[0]) // 2, (side - resized.shape[1]) // 2
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        raw = self.session.run(None, {self.input.name: tensor})[0][0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        scores = raw[:, 4:].max(axis=1)
        selected, scores = raw[scores >= self.threshold], scores[scores >= self.threshold]
        boxes = []
        for cx, cy, bw, bh, *_ in selected:
            x0, y0 = max(0., (float(cx - bw / 2) - left) / scale), max(0., (float(cy - bh / 2) - top) / scale)
            x1, y1 = min(float(width), (float(cx + bw / 2) - left) / scale), min(float(height), (float(cy + bh / 2) - top) / scale)
            boxes.append([x0, y0, max(0., x1 - x0), max(0., y1 - y0)])
        kept = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.threshold, .5) if boxes else []
        return [{"box_xywh": boxes[int(i)], "score": float(scores[int(i)])} for i in np.asarray(kept).reshape(-1)]


def main(directory):
    import cv2
    import numpy as np
    request = json.loads((directory / "preview-request.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    frames = np.fromfile(directory / "frames.rgb", dtype=np.uint8).reshape(-1, 270, 480, 3)
    expected = (request["end_ms"] - request["start_ms"]) // 40
    if not 16 <= expected <= 200 or len(frames) != expected:
        raise ValueError("Preview frame count does not match the bounded 25fps grid")
    detector = LocalPersonDetector(Path(request["detector_path"]), TRACKING["threshold"])
    loaded = time.monotonic()
    detections, cuts = [], []
    previous = None
    for frame in frames:
        cuts.append(previous is not None and float(np.abs(frame.astype(np.int16) - previous.astype(np.int16)).mean()) > TRACKING["cut_mean_absolute_difference"])
        detections.append(detector(frame))
        previous = frame
    tracks = link_tracks(detections, cuts, request["start_ms"])
    for track in tracks:
        middle = len(track["boxes"]) // 2
        box = track["boxes"][middle]
        frame = frames[(box["at_ms"] - request["start_ms"]) // 40]
        x, y, w, h = box["box"]
        crop = frame[round(y * 270):round((y + h) * 270), round(x * 480):round((x + w) * 480)]
        target = directory / (track["id"] + ".png")
        if not cv2.imwrite(str(target), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)):
            raise ValueError("Cannot save detected body thumbnail")
        track["thumbnail_file"] = target.name
    return {"tracks": tracks, "tracking": {"version": DETECTION_VERSION, **TRACKING},
            "cut_count": sum(cuts), "frames_detected": len(frames),
            "timings": {"detector_load_ms": round((loaded - started) * 1000),
                        "detection_tracking_ms": round((time.monotonic() - loaded) * 1000)}}


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), allow_nan=False))
