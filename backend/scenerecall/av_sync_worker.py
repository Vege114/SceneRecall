"""Isolated bridges to user-provided LWTNet/LR-ASD code; no upstream code copied."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import math
import re
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

LRASD_SCHEMA = "scenerecall-lrasd-bundle-v1"
LRASD_ARCHITECTURE = "lr-asd-original-168-v1"
LRASD_PREPROCESS = "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1"
LRASD_SCORE_DOMAIN = "active_speaker_probability"


def validate_lrasd_bundle(directory: Path) -> tuple[dict, Path]:
    """Only a finished, evaluated, content-addressed local export is usable."""
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    required = {"schema_version": LRASD_SCHEMA, "model_type": "lrasd", "architecture": LRASD_ARCHITECTURE,
                "preprocessing": LRASD_PREPROCESS, "score_domain": LRASD_SCORE_DOMAIN, "checkpoint": "model.pt"}
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("LR-ASD bundle 架构、分数域或预处理版本不匹配")
    if metadata.get("visual_canvas") != {"width": 480, "height": 270, "resize": "ffmpeg-bilinear"}:
        raise ValueError("LR-ASD 训练与推理的 480x270 人物裁图画布不匹配")
    dataset, training, validation = (metadata.get(key, {}) for key in ("dataset", "training", "validation"))
    if not all(isinstance(value, dict) for value in (dataset, training, validation)):
        raise ValueError("LR-ASD bundle 元数据格式无效")
    if (dataset.get("crop_type") != "full_body_bbox" or not dataset.get("name") or not dataset.get("version")
            or not re.fullmatch(r"[0-9a-f]{64}", str(dataset.get("split_manifest_sha256", "")))):
        raise ValueError("LR-ASD bundle 缺少版本化人物全身框数据来源")
    if (training.get("status") != "completed" or any(type(training.get(key)) is not int or training[key] <= 0
                                                    for key in ("steps", "epochs_completed"))):
        raise ValueError("LR-ASD 训练尚未完成，不能启用中途 checkpoint")
    metrics = validation.get("metrics")
    if (validation.get("status") != "completed" or not validation.get("split") or not validation.get("version")
            or not isinstance(metrics, dict) or not metrics
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values())):
        raise ValueError("LR-ASD bundle 缺少已完成的独立验证指标")
    checkpoint = directory / "model.pt"
    expected = metadata.get("checkpoint_sha256", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("LR-ASD checkpoint 缺少 SHA-256")
    with checkpoint.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
        if actual is None:  # Standalone LR-ASD's Python 3.10 environment.
            value = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
            actual = value.hexdigest()
    if actual != expected:
        raise ValueError("LR-ASD checkpoint 哈希与导出元数据不符")
    return metadata, checkpoint


@contextlib.contextmanager
def upstream_modules(repository: Path, names=("model", "loss")):
    """Temporarily isolate upstream's generic imports from the host process."""
    def belongs(name):
        return any(name == root or name.startswith(root + ".") for root in names)
    saved = {name: module for name, module in sys.modules.items() if belongs(name)}
    original_path = list(sys.path)
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(repository))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if belongs(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = original_path


def load_lrasd_model(repository: Path, checkpoint: Path, device: str):
    import torch
    # Match training and teacher evaluation's FP32 math on Ampere/Ada GPUs.
    # Float32 tensors alone do not disable cuDNN's optional TF32 kernels.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Avoid ASD.__init__, which always creates a CUDA optimizer. Register the
    # original three modules so strict loading includes both trained FC heads.
    with upstream_modules(repository):
        architecture = importlib.import_module("model.Model").ASD_Model
        losses = importlib.import_module("loss")
        model = torch.nn.Module()
        model.model = architecture()
        model.lossAV = losses.lossAV()
        model.lossV = losses.lossV()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 168 or len(model.state_dict()) != 168:
        raise ValueError("LR-ASD 需要包含两个分类头的原始完整 168 项 state_dict")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def read_pcm(path: Path):
    import numpy as np
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("同步音频必须为 16kHz 单声道 PCM16")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").copy()


def aligned_lrasd_inputs(directory: Path, request: dict):
    import numpy as np
    import python_speech_features
    frames = np.fromfile(directory / "frames.rgb", dtype=np.uint8).reshape(
        -1, request.get("frame_height", 270), request.get("frame_width", 480), 3)
    audio = read_pcm(directory / "audio.wav")
    if frames.shape[1:3] != (270, 480):
        raise ValueError("LR-ASD 仅接受已按版本化方法缩放的 480x270 连续画布")
    expected = (request["end_ms"] - request["start_ms"]) // 40
    if expected < 16 or len(frames) != expected or len(audio) != expected * 640:
        raise ValueError("LR-ASD 连续 25fps 帧与同时间 16kHz 音频长度不一致")
    # Match upstream dataLoader: PCM16 magnitudes, 13 MFCCs at 100 Hz. The
    # upstream frontend owns gray normalization; inputs remain raw 0..255.
    mfcc = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=.025, winstep=.010)
    if len(mfcc) < expected * 4:
        mfcc = np.pad(mfcc, ((0, expected * 4 - len(mfcc)), (0, 0)), "wrap")
    return frames, np.asarray(mfcc[:expected * 4], dtype=np.float32)


def lrasd_visual(frames, boxes: list[dict], start_ms: int):
    import cv2
    import numpy as np
    if not boxes:
        raise ValueError("LR-ASD 缺少连续人物框")
    regions = []
    for index, frame in enumerate(frames):
        at = start_ms + index * 40
        row = min(boxes, key=lambda row: abs(row["at_ms"] - at))
        if abs(row["at_ms"] - at) > 40:
            raise ValueError("人物框未覆盖连续 25fps 时间网格")
        x, y, width, height = row["box"]
        if (not all(math.isfinite(v) for v in (x, y, width, height)) or min(x, y) < 0
                or min(width, height) <= 0 or x + width > 1.000001 or y + height > 1.000001):
            raise ValueError("LR-ASD 人物全身框无效")
        frame_height, frame_width = frame.shape[:2]
        left, top = max(0, round(x * frame_width)), max(0, round(y * frame_height))
        right, bottom = min(frame_width, round((x + width) * frame_width)), min(frame_height, round((y + height) * frame_height))
        if right <= left or bottom <= top:
            raise ValueError("LR-ASD 人物全身框不足一个像素")
        gray = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
        regions.append(cv2.resize(gray, (112, 112), interpolation=cv2.INTER_LINEAR))
    return np.asarray(regions, dtype=np.float32)


def lrasd_track_inputs(frames, mfcc, request: dict, track: dict):
    """Slice an explicitly bounded preview track on the shared 40 ms grid.

    Existing confirmed reviews omit track times and still consume the complete
    request. Anonymous detections are admitted only by the preview entry point.
    """
    if not track.get("character_id") and request.get("preview_only") is not True:
        raise ValueError("匿名人物轨迹仅允许用于独立 LR-ASD 预览")
    start = track.get("start_ms", request["start_ms"])
    end = track.get("end_ms", request["end_ms"])
    if (type(start) is not int or type(end) is not int
            or start < request["start_ms"] or end > request["end_ms"]
            or end - start < 640 or (start - request["start_ms"]) % 40 or (end - start) % 40):
        raise ValueError("LR-ASD 子轨迹必须为范围内至少 16 帧的连续 25fps 窗口")
    if request.get("preview_only") is not True and (start != request["start_ms"] or end != request["end_ms"]):
        raise ValueError("既有身份同步复核必须覆盖完整请求窗口")
    first, last = (start - request["start_ms"]) // 40, (end - request["start_ms"]) // 40
    if request.get("preview_only") is True:
        boxes = track.get("boxes", [])
        if (len(boxes) != last - first
                or any(row.get("at_ms") != start + index * 40 for index, row in enumerate(boxes))):
            raise ValueError("预览人物框必须逐帧真实覆盖子窗口；拒绝补帧或稀疏插值")
    return frames[first:last], mfcc[first * 4:last * 4], start, end


def lrasd_main(directory: Path, request: dict) -> dict:
    import numpy as np
    import torch
    metadata, checkpoint = validate_lrasd_bundle(Path(request["checkpoint"]))
    if request["preprocessing"] != LRASD_PREPROCESS:
        raise ValueError("LR-ASD 请求预处理版本不匹配")
    device = request["device"]
    if device not in {"cpu", "cuda"}:
        raise ValueError("LR-ASD 仅支持 CPU 或 CUDA")
    model = load_lrasd_model(Path(request["repository"]), checkpoint, device)
    frames, mfcc = aligned_lrasd_inputs(directory, request)
    results = []
    for track in request["tracks"]:
        track_frames, track_mfcc, start, end = lrasd_track_inputs(frames, mfcc, request, track)
        audio_tensor = torch.from_numpy(track_mfcc).unsqueeze(0).to(device)
        visual = lrasd_visual(track_frames, track["boxes"], start)
        with torch.inference_mode():
            av, _ = model.model(audio_tensor, torch.from_numpy(visual).unsqueeze(0).to(device))
            # lossAV.forward(labels=None) returns unnormalized logits upstream.
            # Use both trained logits for an explicit class-1 probability.
            probabilities = torch.softmax(model.lossAV.FC(av), dim=-1)[:, 1].cpu().numpy()
        if len(probabilities) != len(track_frames) or not np.isfinite(probabilities).all():
            raise ValueError("LR-ASD 返回无效帧级概率")
        results.append({"character_id": track["character_id"], "track_id": track.get("id"),
                        "active_speaker_probability": float(probabilities.mean()),
                        "start_ms": start, "end_ms": end,
                        "frame_probabilities": [{"at_ms": start + index * 40,
                                                 "probability": float(value)}
                                                for index, value in enumerate(probabilities)],
                        "score_domain": LRASD_SCORE_DOMAIN, "association_status": "candidate"})
    return {"status": "review_required", "results": results, "model_type": "lrasd",
            "architecture": metadata["architecture"], "model_version": request["model_version"],
            "preprocessing": LRASD_PREPROCESS, "score_domain": LRASD_SCORE_DOMAIN,
            "dataset": metadata["dataset"], "validation": metadata["validation"],
            "domain_validation": metadata.get("domain_validation", {"status": "not_claimed"}),
            "calibration": metadata.get("calibration", {"status": "uncalibrated"}),
            "start_ms": request["start_ms"], "end_ms": request["end_ms"],
            "reason": "LR-ASD 仅提供人物正在说话的待复核概率；不自动覆盖人工身份或进入可信声音库"}


def main(directory: Path) -> dict:
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request.get("model_type") == "lrasd":
        return lrasd_main(directory, request)
    import numpy as np
    import torch
    from PIL import Image

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
    audio = np.asarray(read_pcm(directory / "audio.wav"), dtype=np.float32)
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
