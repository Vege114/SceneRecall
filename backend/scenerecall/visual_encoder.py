"""Offline image/text encoding in an isolated, supervised Python worker.

Configuration lives in ``settings_dir/retrieval-encoder.json``. Model snapshots
are installed explicitly, never fetched by this module. ``box`` uses normalized
left/top/right/bottom coordinates; source metadata is preserved by the caller.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from fractions import Fraction

ENCODERS = {
    "siglip2-base-p16-256": "google/siglip2-base-patch16-256",
    "openclip-multilingual-b32": "laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
}
IMPLEMENTATION = "retrieval-encoder-v1"
MAX_REPLY_BYTES = 16 * 1024 * 1024
_PACKAGES = ("torch", "torchvision", "transformers", "tokenizers", "sentencepiece", "safetensors", "Pillow", "numpy")


class VisualEncoderUnavailable(RuntimeError):
    """Configuration, dependencies or the requested inference device are unavailable."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _local_path(value, label, *, directory=False) -> Path:
    if not isinstance(value, str) or not value.strip() or value.startswith(("\\\\", "//")):
        raise VisualEncoderUnavailable(f"{label}须为本地绝对路径")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise VisualEncoderUnavailable(f"{label}须为本地绝对路径")
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise VisualEncoderUnavailable(f"{label}不存在；请先准备本地依赖与权重")
    return path


def _settings(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise VisualEncoderUnavailable("尚未配置 retrieval-encoder.json；不会自动下载模型") from None
    except (OSError, ValueError):
        raise VisualEncoderUnavailable("retrieval-encoder.json 不是有效的本地配置") from None
    allowed = {"encoder_id", "model_dir", "text_model_dir", "revision", "python", "device", "batch_size", "timeout_s"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise VisualEncoderUnavailable("视觉编码配置包含未声明字段")
    encoder_id = value.get("encoder_id")
    if encoder_id not in ENCODERS:
        raise VisualEncoderUnavailable("不支持的视觉编码器")
    revision = value.get("revision", "")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise VisualEncoderUnavailable("视觉权重 revision 须为固定提交 SHA，不能使用 main 或标签")
    device = value.get("device", "cuda")
    if device != "cpu" and (not isinstance(device, str) or not re.fullmatch(r"cuda(?::\d+)?", device)):
        raise VisualEncoderUnavailable("视觉 device 须为 cuda、cuda:N 或显式 cpu")
    batch = value.get("batch_size", 16)
    timeout = value.get("timeout_s", 180)
    if type(batch) is not int or not 1 <= batch <= 256:
        raise VisualEncoderUnavailable("视觉 batch_size 须为 1–256")
    if type(timeout) not in {float, int} or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise VisualEncoderUnavailable("视觉 timeout_s 须为 0–3600 秒")
    executable = value.get("python") or os.environ.get("SCENERECALL_RETRIEVAL_PYTHON")
    if not isinstance(executable, str) or not executable:
        raise VisualEncoderUnavailable("请指定独立视觉编码 Python 或 SCENERECALL_RETRIEVAL_PYTHON")
    resolved = shutil.which(executable)
    if not resolved:
        raise VisualEncoderUnavailable("视觉编码 Python 不可用")
    model = _local_path(value.get("model_dir"), "模型目录", directory=True)
    text_model = None
    if encoder_id == "openclip-multilingual-b32":
        text_model = _local_path(value.get("text_model_dir", str(model / "text_encoder")),
                                 "XLM-R 配置与 tokenizer 目录", directory=True)
    return {**value, "encoder_id": encoder_id, "revision": revision, "model_dir": str(model),
            "text_model_dir": str(text_model) if text_model else None, "python": str(Path(resolved).resolve()),
            "device": device, "dtype": "float32" if device == "cpu" else "float16",
            "batch_size": batch, "timeout_s": float(timeout)}


def snapshot_files(config: dict) -> dict[str, Path]:
    """Return exactly the weights/configuration families consumed by the worker."""
    root = Path(config["model_dir"])
    if config["encoder_id"] == "siglip2-base-p16-256":
        if (root / "model.safetensors").is_file():
            weights = [root / "model.safetensors"]
        elif (root / "model.safetensors.index.json").is_file():
            try:
                index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
                names = set(index["weight_map"].values())
                if not names or any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
                    raise ValueError
                weights = [root / name for name in sorted(names)]
            except (OSError, ValueError, KeyError, TypeError):
                raise VisualEncoderUnavailable("本地分片权重索引无效") from None
        elif (root / "pytorch_model.bin").is_file():
            weights = [root / "pytorch_model.bin"]
        else:
            raise VisualEncoderUnavailable("缺少本地 SigLIP 2 权重；不会自动下载")
        required = [root / "config.json", root / "preprocessor_config.json", root / "tokenizer_config.json"]
        tokenizer_root = root
    else:
        names = ("open_clip_model.safetensors", "open_clip_pytorch_model.safetensors",
                 "open_clip_pytorch_model.bin", "open_clip_pytorch_model.pth",
                 "model.safetensors", "pytorch_model.bin", "pytorch_model.pth", "model.pth")
        weights = [root / name for name in names if (root / name).is_file()][:1]
        if not weights:
            raise VisualEncoderUnavailable("缺少本地 OpenCLIP 权重；不会自动下载")
        tokenizer_root = Path(config["text_model_dir"])
        required = [root / "open_clip_config.json", tokenizer_root / "config.json",
                    tokenizer_root / "tokenizer_config.json"]
    if not any((tokenizer_root / name).is_file() for name in
               ("tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "spiece.model")):
        raise VisualEncoderUnavailable("缺少本地 tokenizer 词表")
    if any(not path.is_file() for path in [*weights, *required]):
        raise VisualEncoderUnavailable("本地模型 snapshot 不完整")
    result = {"weights/" + path.name: path for path in weights}
    for prefix, directory in (("model", root), ("text", tokenizer_root)):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in {".json", ".model", ".txt"}:
                result[f"{prefix}/{path.name}"] = path
    return result


def _worker_command(config: dict) -> list[str]:
    command = [config["python"], "-u", "-m", "scenerecall.visual_encoder_worker"]
    if os.name == "nt":
        # The base interpreter, rather than Windows' venv redirector, owns the
        # kill-on-close Job Object. Its descendants include the actual ML worker.
        command = [getattr(sys, "_base_executable", sys.executable), "-I",
                   str(Path(__file__).with_name("_windows_process.py")), *command]
    return command


class VisualEncoder:
    def __init__(self, settings_dir: Path):
        self.path = Path(settings_dir) / "retrieval-encoder.json"
        self._hashes = {}
        self._runtime_cache = {}
        self._process = None
        self._process_key = None
        self._process_lock = threading.RLock()
        self._request_lock = threading.Lock()
        self._identity_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._pending = set()
        self._closed = False
        self.metadata: dict = {}

    def settings(self) -> dict:
        """Safe UI status; no local paths, environment values or snapshot contents."""
        result = {"available": False, "offline": True}
        try:
            config = _settings(self.path)
            result.update({key: config[key] for key in ("encoder_id", "revision", "device", "dtype", "batch_size")})
            identity = self.identity()
            result.update(available=True, signature=identity["signature"], repo=identity["repo"])
        except VisualEncoderUnavailable as exc:
            result["reason"] = str(exc)
        return result

    @property
    def batch_size(self) -> int:
        return _settings(self.path)["batch_size"]

    def _runtime_versions(self, config: dict) -> dict:
        packages = (*_PACKAGES, *(("open_clip_torch", "timm", "ftfy", "regex") if
                                config["encoder_id"] == "openclip-multilingual-b32" else ()))
        executable = Path(config["python"])
        # Directory mtimes invalidate this cheap probe after an environment install.
        root = executable.parent.parent if executable.parent.name.lower() in {"scripts", "bin"} else executable.parent
        sites = [root / "Lib" / "site-packages", *root.glob("lib/python*/site-packages")]
        stamp = (str(executable), executable.stat().st_mtime_ns, packages,
                 tuple((str(path), path.stat().st_mtime_ns) for path in sites if path.exists()))
        if stamp not in self._runtime_cache:
            source = ("import json,importlib.metadata as m,sys;"
                      "print(json.dumps({p:m.version(p) for p in sys.argv[1:]}))")
            try:
                result = subprocess.run([str(executable), "-I", "-c", source, *packages], capture_output=True,
                                        text=True, encoding="utf-8", timeout=15, check=True,
                                        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
                versions = json.loads(result.stdout)
                if set(versions) != set(packages) or not all(isinstance(v, str) and v for v in versions.values()):
                    raise ValueError
            except (OSError, ValueError, subprocess.SubprocessError):
                raise VisualEncoderUnavailable("独立视觉编码环境缺少依赖或无法读取版本；不会安装依赖") from None
            self._runtime_cache = {stamp: versions}
        return self._runtime_cache[stamp]

    def identity(self) -> dict:
        """Strict, path-free cache identity, without loading a model or fetching files."""
        with self._identity_lock:
            config = _settings(self.path)
            files = snapshot_files(config)
            hashes = {}
            for name, path in files.items():
                stat = path.stat()
                key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if key not in self._hashes:
                    with path.open("rb") as stream:
                        self._hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
                hashes[name] = self._hashes[key]
            versions = dict(self._runtime_versions(config))
            implementation_files = [Path(__file__), Path(__file__).with_name("visual_encoder_worker.py")]
            implementation_hash = _digest([hashlib.sha256(path.read_bytes()).hexdigest() for path in implementation_files])
            identity = {
                "encoder_id": config["encoder_id"], "repo": ENCODERS[config["encoder_id"]],
                "revision": config["revision"], "weight_hashes": {k: v for k, v in hashes.items() if k.startswith("weights/")},
                "preprocessor_hash": _digest({k: v for k, v in hashes.items() if not k.startswith("weights/")}),
                "tokenizer_hash": _digest({k: v for k, v in hashes.items() if k.startswith("text/")}),
                "implementation": IMPLEMENTATION, "implementation_hash": implementation_hash,
                "runtime_packages": versions, "device": config["device"], "dtype": config["dtype"],
                "normalization": "l2-float32", "crop": "normalized-xyxy-floor-ceil-v1",
            }
            return {**identity, "signature": _digest(identity)}

    def _stop(self):
        with self._process_lock:
            process, self._process = self._process, None
            self._process_key = None
            if process is None:
                return
            if process.poll() is None:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    with contextlib.suppress(OSError, ValueError):
                        stream.close()

    def _start(self, config: dict, identity: dict):
        with self._process_lock:
            key = _digest([config, identity["signature"]])
            if self._process is not None and self._process.poll() is None and self._process_key == key:
                return self._process
            self._stop()
            environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               HF_HUB_DISABLE_TELEMETRY="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="2")
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
                str(Path(__file__).resolve().parent.parent), environment.get("PYTHONPATH")]))
            self._process = subprocess.Popen(
                _worker_command(config), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=environment, start_new_session=os.name == "posix",
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
            self._process_key = key
            return self._process

    @staticmethod
    def _vectors(value, count):
        if not isinstance(value, list) or len(value) != count:
            raise VisualEncoderUnavailable("视觉编码返回的向量数量不一致")
        dimensions = set()
        for row in value:
            if (not isinstance(row, list) or not row or
                    any(type(x) not in {int, float} or not math.isfinite(x) for x in row)):
                raise VisualEncoderUnavailable("视觉编码返回了无效向量")
            norm = math.sqrt(sum(x * x for x in row))
            if not .999 <= norm <= 1.001:
                raise VisualEncoderUnavailable("视觉编码向量未正确归一化")
            dimensions.add(len(row))
        if len(dimensions) > 1:
            raise VisualEncoderUnavailable("视觉编码向量维度不一致")
        return value

    def _request(self, action, values, cancelled):
        from .local_resources import LOCAL_MODEL_GATE
        acquired_request = acquired_gate = False
        try:
            while not cancelled.is_set():
                acquired_request = self._request_lock.acquire(timeout=.05)
                if acquired_request:
                    break
            while not cancelled.is_set() and acquired_request:
                acquired_gate = LOCAL_MODEL_GATE.acquire(timeout=.05)
                if acquired_gate:
                    break
            if cancelled.is_set() or self._closed:
                raise VisualEncoderUnavailable("视觉编码已取消")
            config = _settings(self.path)
            identity = self.identity()
            process = self._start(config, identity)
            request = {"action": action, "values": values, "config": config, "identity": identity}
            reply = queue.Queue(maxsize=1)

            def exchange():
                try:
                    process.stdin.write((json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
                    process.stdin.flush()
                    reply.put(process.stdout.readline(MAX_REPLY_BYTES + 1))
                except (OSError, ValueError) as exc:
                    reply.put(exc)

            reader = threading.Thread(target=exchange, name="scenerecall-encoder-output", daemon=True)
            reader.start()
            deadline = time.monotonic() + config["timeout_s"]
            while True:
                if cancelled.is_set() or self._closed:
                    raise VisualEncoderUnavailable("视觉编码已取消")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VisualEncoderUnavailable("视觉编码超时；已停止 worker")
                try:
                    raw = reply.get(timeout=min(.05, remaining))
                    break
                except queue.Empty:
                    pass
            if isinstance(raw, Exception) or not raw or len(raw) > MAX_REPLY_BYTES or not raw.endswith(b"\n"):
                raise VisualEncoderUnavailable("视觉编码 worker 返回无效协议")
            response = json.loads(raw)
            if response.get("error"):
                raise VisualEncoderUnavailable(response["error"])
            if response.get("signature") != identity["signature"]:
                raise VisualEncoderUnavailable("视觉编码 worker 模型签名不一致")
            self.metadata = response.get("metadata", {})
            return self._vectors(response.get("vectors"), len(values))
        except VisualEncoderUnavailable:
            if acquired_request:
                self._stop()
            raise
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self._stop()
            raise VisualEncoderUnavailable("视觉编码 worker 不可用；请检查本地依赖与权重") from exc
        finally:
            if acquired_gate:
                LOCAL_MODEL_GATE.release()
            if acquired_request:
                self._request_lock.release()

    async def _encode(self, action, values):
        if self._closed:
            raise VisualEncoderUnavailable("视觉编码器已关闭")
        if not values:
            return []
        cancelled = threading.Event()
        with self._pending_lock:
            self._pending.add(cancelled)
        task = asyncio.create_task(asyncio.to_thread(self._request, action, values, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # _request owns the process/gate; it stops promptly even when only
            # part of a JSON line has arrived. Wait until it releases the slot.
            with contextlib.suppress(VisualEncoderUnavailable):
                await asyncio.shield(task)
            raise
        finally:
            with self._pending_lock:
                self._pending.discard(cancelled)

    async def encode_images(self, items: list[dict]) -> list[list[float]]:
        values = []
        for item in items:
            if not isinstance(item, dict) or any(not item.get(key) for key in ("id", "asset_id", "source_fingerprint")):
                raise ValueError("视觉帧缺少稳定 ID 或源指纹")
            if type(item.get("pts")) is not int:
                raise ValueError("视觉帧须提供原始整数 PTS")
            try:
                base = item.get("time_base")
                if isinstance(base, (list, tuple)):
                    if len(base) != 2 or any(type(part) is not int for part in base):
                        raise ValueError
                    base = Fraction(*base)
                else:
                    base = Fraction(str(base))
                if base <= 0:
                    raise ValueError
            except (ValueError, ZeroDivisionError):
                raise ValueError("视觉帧 time_base 无效") from None
            path = _local_path(str(item.get("path", "")), "证据帧")
            if "bbox" in item and "box" in item and item["bbox"] != item["box"]:
                raise ValueError("视觉区域 bbox 与 box 不一致")
            box = item.get("bbox", item.get("box"))
            if box is not None and (not isinstance(box, (list, tuple)) or len(box) != 4
                                    or any(type(x) not in {float, int} or not math.isfinite(x) for x in box)
                                    or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                raise ValueError("视觉区域须为归一化 left/top/right/bottom")
            values.append({"path": str(path), "box": list(box) if box is not None else None})
        return await self._encode("images", values)

    async def encode_texts(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() or len(text) > 2000 for text in texts):
            raise ValueError("视觉检索文本须为 1–2000 字符")
        return await self._encode("texts", texts)

    async def close(self):
        self._closed = True
        with self._pending_lock:
            for cancelled in self._pending:
                cancelled.set()
        await asyncio.to_thread(self._stop)
