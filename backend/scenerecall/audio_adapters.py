"""Optional resident local audio models. No model downloads at application startup."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path

from .audio_media import SAMPLE_RATE, pcm_samples
from .local_resources import LOCAL_MODEL_GATE


class AudioUnavailable(RuntimeError):
    pass


_MODELS: dict[tuple, object] = {}


def model_identity(path: str, family: str) -> str:
    """Fingerprint checkpoint metadata without hashing gigabytes on every segment."""
    root = Path(path).expanduser()
    if not path or not root.exists():
        return f"{family}:unconfigured"
    entries = sorted(root.rglob("*")) if root.is_dir() else [root]
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode())
    for entry in entries:
        if entry.is_file():
            stat = entry.stat()
            digest.update(f"{entry.relative_to(root) if root.is_dir() else entry.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return f"{family}:{digest.hexdigest()[:24]}"


def availability(module: str, path: str) -> dict:
    try:
        installed = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    if not installed:
        return {"available": False, "reason": f"未安装可选依赖 {module}"}
    if not path or not Path(path).expanduser().exists():
        return {"available": False, "reason": "未配置本机模型权重；应用不会自动下载"}
    return {"available": True, "reason": "本机权重已配置，实际加载与动画效果仍需验证"}


def _local_directory(path: str) -> Path:
    root = Path(path).expanduser()
    if not path or not root.is_dir():
        raise AudioUnavailable("请先手动下载并配置完整的本机模型目录；不会自动下载权重")
    return root.resolve()


def validate_pyannote_config(root: Path, config: dict) -> None:
    if (not isinstance(config, dict) or not isinstance(config.get("pipeline"), dict)
            or config["pipeline"].get("name") != "pyannote.audio.pipelines.SpeakerDiarization"
            or config.get("preprocessors")):
        raise AudioUnavailable("仅接受完整的本机 community-1 SpeakerDiarization 配置")
    params = config["pipeline"].get("params", {})
    if not isinstance(params, dict):
        raise AudioUnavailable("community-1 模型配置无效")
    for name in ("segmentation", "embedding", "plda"):
        reference = params.get(name)
        subfolder = None
        if isinstance(reference, dict):
            subfolder = reference.get("subfolder")
            reference = reference.get("checkpoint")
        if not isinstance(reference, str) or not reference.strip() or "://" in reference:
            raise AudioUnavailable(f"community-1 {name} 必须显式指定本机权重，不允许默认远程模型")
        local = root / reference.removeprefix("$model/") if reference.startswith("$model/") else Path(reference)
        if not local.is_absolute():
            raise AudioUnavailable(f"community-1 {name} 不能使用远程模型 ID")
        if subfolder is not None:
            if not isinstance(subfolder, str) or Path(subfolder).is_absolute() or ".." in Path(subfolder).parts:
                raise AudioUnavailable("community-1 本机模型子目录无效")
            local = local / subfolder
        if not local.exists():
            raise AudioUnavailable(f"community-1 本机 {name} 权重缺失；拒绝联网获取")


def validate_ecapa_config(root: Path) -> None:
    """Inspect YAML nodes without executing HyperPyYAML object constructors."""
    import yaml

    def mapping(node):
        if not isinstance(node, yaml.MappingNode):
            raise AudioUnavailable("ECAPA 本机 hyperparams.yaml 格式无效")
        return {key.value: value for key, value in node.value}

    try:
        config = mapping(yaml.compose((root / "hyperparams.yaml").read_text()))
    except yaml.YAMLError as exc:
        raise AudioUnavailable("ECAPA 本机 hyperparams.yaml 格式无效") from exc
    paths = mapping(mapping(config.get("pretrainer")).get("paths"))
    expected = {"embedding_model": "embedding_model.ckpt", "mean_var_norm_emb": "mean_var_norm_emb.ckpt",
                "classifier": "classifier.ckpt", "label_encoder": "label_encoder.txt"}
    if set(paths) != set(expected):
        raise AudioUnavailable("ECAPA 预训练文件列表不属于已支持的本机模型配置")
    for key, filename in expected.items():
        value = paths[key]
        if (not isinstance(value, yaml.ScalarNode)
                or value.value not in {f"<pretrained_path>/{filename}", str(root / filename)}):
            raise AudioUnavailable("ECAPA 预训练文件必须指向本机已验证权重；拒绝远程下载")


class PyannoteDiarizer:
    """community-1's regular annotation preserves overlap; never use exclusive output."""

    def __init__(self, path: str, device: str = "cpu"):
        self.path, self.device = path, device
        self.version = model_identity(path, "pyannote-community-1")

    def diarize(self, path: Path) -> list[dict]:
        root = _local_directory(self.path)
        with LOCAL_MODEL_GATE:
            try:
                # This model offers telemetry independently of inference. SceneRecall
                # is local-only: never send its durations or model usage remotely.
                os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
                import numpy as np
                import torch
                import yaml
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise AudioUnavailable("自动说话人分段需要 audio 可选依赖") from exc
            key = (self.version, self.device)
            if key not in _MODELS:
                config_path = root / "config.yaml"
                if not config_path.is_file():
                    raise AudioUnavailable("community-1 本机目录缺少 config.yaml")
                try:
                    config = yaml.safe_load(config_path.read_text())
                except yaml.YAMLError as exc:
                    raise AudioUnavailable("community-1 本机 config.yaml 格式无效") from exc
                validate_pyannote_config(root, config)
                pipeline = Pipeline.from_pretrained(str(root))
                pipeline.to(torch.device(self.device))
                _MODELS[key] = pipeline
            values = np.asarray(pcm_samples(path), dtype="float32") / 32768
            result = _MODELS[key]({"waveform": torch.from_numpy(values).unsqueeze(0), "sample_rate": SAMPLE_RATE})
            annotation = getattr(result, "speaker_diarization", result)
            if hasattr(annotation, "itertracks"):
                turns = [(turn, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True)]
            else:
                turns = list(annotation)
            return [{"start_ms": round(turn.start * 1000), "end_ms": round(turn.end * 1000),
                     "speaker": str(speaker) if speaker is not None else None} for turn, speaker in turns]


class ECAPAEmbedder:
    def __init__(self, path: str, device: str = "cpu"):
        self.path, self.device = path, device
        self.version = model_identity(path, "speechbrain-ecapa-voxceleb")

    def embed(self, path: Path) -> list[float]:
        root = _local_directory(self.path)
        # Reject incomplete snapshots before SpeechBrain can fetch missing files.
        for name in ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.txt"):
            if not (root / name).is_file():
                raise AudioUnavailable(f"ECAPA 本机模型不完整，缺少 {name}")
        with LOCAL_MODEL_GATE:
            try:
                import numpy as np
                import torch
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError as exc:
                raise AudioUnavailable("声音特征提取需要 audio 可选依赖") from exc
            key = (self.version, self.device)
            if key not in _MODELS:
                validate_ecapa_config(root)
                # Official YAML has a remote pretrained_path. Override it to the
                # validated local snapshot, including its pretrainer loadables.
                _MODELS[key] = EncoderClassifier.from_hparams(
                    source=str(root), savedir=str(root), overrides={"pretrained_path": str(root)},
                    use_auth_token=False, run_opts={"device": self.device})
            values = np.asarray(pcm_samples(path), dtype="float32") / 32768
            with torch.inference_mode():
                embedding = _MODELS[key].encode_batch(torch.from_numpy(values).unsqueeze(0))
            return normalized(embedding.detach().cpu().reshape(-1).tolist())


class FasterWhisperASR:
    def __init__(self, path: str, device: str = "cpu"):
        self.path, self.device = path, device
        self.version = model_identity(path, "faster-whisper")

    def transcribe(self, path: Path, language: str) -> list[dict]:
        root = _local_directory(self.path)
        with LOCAL_MODEL_GATE:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise AudioUnavailable("无可靠字幕时的 ASR 需要 asr 可选依赖") from exc
            key = (self.version, self.device)
            if key not in _MODELS:
                _MODELS[key] = WhisperModel(str(root), device=self.device, local_files_only=True,
                                           compute_type="int8" if self.device == "cpu" else "float16")
            segments, _ = _MODELS[key].transcribe(str(path), language=None if language == "und" else language,
                                                 word_timestamps=False, vad_filter=True)
            return [{"start_ms": round(segment.start * 1000), "end_ms": round(segment.end * 1000),
                     "text": segment.text, "language": language, "source": "asr"} for segment in segments]


def normalized(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(float(x)) for x in vector):
        raise ValueError("声音模型返回无效特征")
    norm = math.sqrt(sum(float(x) ** 2 for x in vector))
    if norm <= 1e-12:
        raise ValueError("声音模型返回零特征")
    return [float(x) / norm for x in vector]


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("不同声音模型的特征不能混用")
    return max(0.0, min(2.0, 1 - sum(a * b for a, b in zip(normalized(left), normalized(right), strict=True))))
