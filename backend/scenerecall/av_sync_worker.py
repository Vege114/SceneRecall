"""Isolated bridge to user-provided Animated_AD code; no upstream code copied."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main(directory: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    from .audio_media import pcm_samples

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    package = Path(request["repository"]) / "character_recognition/audio_recognition"
    sys.path.insert(0, str(package))
    from model.lwtnet import LWTNet_temporal_adapter
    from model.sync_scorer import DemoEvalTrainerScoreOnly

    device = request["device"]
    if device not in {"cpu", "cuda"}:
        raise ValueError("Animated_AD scorer currently supports CPU or CUDA")
    model = LWTNet_temporal_adapter()
    checkpoint = torch.load(request["checkpoint"], map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    if not any(key.startswith("vid_adapter.") for key in state) or not any(key.startswith("aud_adapter.") for key in state):
        raise ValueError("Missing trained temporal adapters; refusing random initialization")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    opts = SimpleNamespace(gpu_id=0, sample_rate=16000, fps=25, n_negative_samples=0)
    trainer = DemoEvalTrainerScoreOnly(model, opts)
    trainer.device = torch.device(device)
    trainer.model.to(device)
    frames = np.fromfile(directory / "frames.rgb", dtype=np.uint8).reshape(-1, 270, 480, 3)
    # Animated_AD's audio frontend consumes float32 containing PCM16 magnitudes,
    # unlike ECAPA's [-1,1] input. Preserve that model-specific preprocessing.
    audio = np.asarray(pcm_samples(directory / "audio.wav"), dtype=np.float32)
    count = min(len(frames), len(audio) // 640)
    if count < 16:
        raise ValueError("Not enough continuous source frames")
    frames, audio = frames[:count], audio[:count * 640]
    results = []
    for track in request["tracks"]:
        boxes = track["boxes"]
        regions = []
        for index, frame in enumerate(frames):
            at = request["start_ms"] + index * 40
            row = min(boxes, key=lambda row: abs(row["at_ms"] - at))
            if abs(row["at_ms"] - at) > 40:
                raise ValueError("Track does not cover continuous media grid")
            x, y, width, height = row["box"]
            region = Image.fromarray(frame).crop((round(x * 480), round(y * 270),
                                                  round((x + width) * 480), round((y + height) * 270)))
            regions.append(np.asarray(region.resize((480, 270)), dtype=np.float32))
        video_tensor = torch.from_numpy(np.stack(regions)).permute(3, 0, 1, 2).unsqueeze(0)
        with torch.inference_mode():
            scores = trainer.eval([{"video": video_tensor, "audio": torch.from_numpy(audio).unsqueeze(0)}])
        score = float(scores.mean())
        if not np.isfinite(score):
            raise ValueError("Invalid synchronization score")
        results.append({"character_id": track["character_id"], "track_id": track.get("id"),
                        "synchronization_score": score, "association_status": "candidate"})
    return {"status": "review_required", "results": results, "model_version": request["model_version"],
            "preprocessing": request["preprocessing"], "start_ms": request["start_ms"], "end_ms": request["end_ms"],
            "reason": "同步分数没有动画测试集校准，不自动覆盖人工身份或进入可信声音库"}


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        result = main(Path(sys.argv[1]))
    print(json.dumps(result, allow_nan=False))
