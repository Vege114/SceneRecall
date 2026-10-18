"""Bounded offline smoke of local anime ONNX detectors on authorized media."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort


class LocalDetector:
    def __init__(self, path, threshold=.45):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.input = self.session.get_inputs()[0]

    def __call__(self, rgb):
        height, width = rgb.shape[:2]
        side = 640
        scale = min(side / width, side / height)
        resized = cv2.resize(rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
        top, left = (side - resized.shape[0]) // 2, (side - resized.shape[1]) // 2
        canvas = np.full((side, side, 3), 114, dtype=np.uint8)
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        raw = self.session.run(None, {self.input.name: tensor})[0][0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        scores = raw[:, 4:].max(axis=1)
        selected = raw[scores >= self.threshold]
        scores = scores[scores >= self.threshold]
        boxes = []
        for center_x, center_y, box_width, box_height, *_ in selected:
            x0 = max(0., (float(center_x - box_width / 2) - left) / scale)
            y0 = max(0., (float(center_y - box_height / 2) - top) / scale)
            x1 = min(float(width), (float(center_x + box_width / 2) - left) / scale)
            y1 = min(float(height), (float(center_y + box_height / 2) - top) / scale)
            boxes.append([x0, y0, max(0., x1 - x0), max(0., y1 - y0)])
        kept = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.threshold, .5) if boxes else []
        return [{"box_xywh": boxes[int(index)], "score": float(scores[int(index)])}
                for index in np.asarray(kept).reshape(-1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("/mnt/d/SceneRecall/training/lrasd/models"))
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    detectors = {name: LocalDetector(args.models / (name + ".onnx")) for name in ("face", "person")}
    rows = []
    for at in (120, 600, 1200, 2400, 3600, 4800, 6000, 7200):
        image = args.output / ("frame-%04d.jpg" % at)
        if not image.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(at), "-i", source["video_path"],
                            "-map", "0:v:0", "-frames:v", "1", "-vf", "scale=640:360:flags=bilinear", str(image)],
                           check=True, timeout=60)
        bgr = cv2.imread(str(image))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        result = {name: detector(rgb) for name, detector in detectors.items()}
        row = {"at_seconds": at, "path": str(image), "detections": result,
               "inference_seconds": time.perf_counter() - started}
        rows.append(row)
        for name, detections in result.items():
            for detection in detections:
                x, y, width, height = map(round, detection["box_xywh"])
                color = (0, 0, 255) if name == "face" else (255, 120, 0)
                cv2.rectangle(bgr, (x, y), (x + width, y + height), color, 2)
                cv2.putText(bgr, name, (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        cv2.imwrite(str(args.output / ("annotated-%04d.jpg" % at)), bgr)
        print(json.dumps({"at_seconds": at, "faces": len(result["face"]), "persons": len(result["person"]),
                          "inference_seconds": row["inference_seconds"]}), flush=True)
    report = {"purpose": "detection smoke only; not active speaker labels", "frames": rows,
              "models": {name: {"sha256": hashlib.sha256((args.models / (name + ".onnx")).read_bytes()).hexdigest(),
                                "input": str(detector.input.shape)} for name, detector in detectors.items()}}
    (args.output / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
