"""Private offline-only image/text worker; optional ML imports stay out of the app."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


def encode_batches(values, batch_size, encode, is_oom, clear_cache):
    """Retry only the failing batch, preserving order and successful work."""
    result, offset, retries = [], 0, 0
    while offset < len(values):
        size = min(batch_size, len(values) - offset)
        try:
            rows = encode(values[offset:offset + size])
        except Exception as exc:
            if not is_oom(exc) or size <= 1:
                raise
            batch_size = max(1, size // 2)
            retries += 1
            clear_cache()
            continue
        result.extend(rows)
        offset += size
    return result, batch_size, retries


class EncoderRuntime:
    def __init__(self, config, identity):
        started = time.monotonic()
        from .visual_encoder import snapshot_files, _digest
        hashes = {}
        for name, path in snapshot_files(config).items():
            with path.open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if ({k: v for k, v in hashes.items() if k.startswith("weights/")} != identity["weight_hashes"]
                or _digest({k: v for k, v in hashes.items() if not k.startswith("weights/")})
                != identity["preprocessor_hash"]):
            raise RuntimeError("编码环境依赖或模型文件已变化，请重新读取模型身份")
        import torch
        from PIL import Image
        self.torch, self.Image = torch, Image
        self.config = config
        self.identity = identity
        self.batch_size = config["batch_size"]
        for name, expected in identity["runtime_packages"].items():
            if importlib.metadata.version(name) != expected:
                raise RuntimeError("编码环境依赖已变化，请重新读取模型身份")
        if config["device"] != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("请求的 CUDA 不可用；拒绝静默回退 CPU")
        self.device = torch.device(config["device"])
        self.dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.processor = self.tokenizer = self.transform = None
        self.clean_text = lambda text: text
        model_dir = config["model_dir"]
        if config["encoder_id"] == "siglip2-base-p16-256":
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                                            use_fast=False)
            self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                                   torch_dtype=self.dtype).to(self.device).eval()
            self.tokenizer = self.processor.tokenizer
            self.clean_text = str.lower
            self.context_length = 64
        else:
            import open_clip
            from open_clip.tokenizer import get_clean_fn
            from transformers import AutoTokenizer
            local = json.loads((Path(model_dir) / "open_clip_config.json").read_text(encoding="utf-8"))
            text_cfg = copy.deepcopy(local["model_cfg"]["text_cfg"])
            text_cfg.update(hf_model_name=config["text_model_dir"], hf_tokenizer_name=config["text_model_dir"],
                            hf_model_pretrained=False)
            self.model, _, self.transform = open_clip.create_model_and_transforms(
                "local-dir:" + model_dir, device=self.device, precision="fp32", pretrained_text=False,
                require_pretrained=True, text_cfg=text_cfg)
            # Whole-model casting also handles the HF text tower consistently.
            self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
            self.context_length = int(text_cfg.get("context_length", 77))
            self.tokenizer = AutoTokenizer.from_pretrained(config["text_model_dir"], local_files_only=True,
                                                           trust_remote_code=False)
            self.clean_text = get_clean_fn("whitespace")
        actual_device = next(self.model.parameters()).device
        if actual_device.type != self.device.type:
            raise RuntimeError("视觉编码实际 device 与配置不一致")
        self.metadata = {
            "provider": "pytorch-cuda" if actual_device.type == "cuda" else "pytorch-cpu",
            "device": str(actual_device), "dtype": config["dtype"], "offline": True,
            "cpu_explicit": config["device"] == "cpu", "runtime_packages": identity["runtime_packages"],
            "model_load_ms": round((time.monotonic() - started) * 1000), "effective_batch_size": self.batch_size,
            "oom_retries": 0,
        }

    def _image(self, item):
        with self.Image.open(item["path"]) as source:
            image = source.convert("RGB")
        if item["box"] is not None:
            x1, y1, x2, y2 = item["box"]
            image = image.crop((math.floor(x1 * image.width), math.floor(y1 * image.height),
                                math.ceil(x2 * image.width), math.ceil(y2 * image.height)))
        return image

    def _batch(self, action, values):
        torch = self.torch
        with torch.inference_mode():
            if action == "images":
                images = [self._image(item) for item in values]
                try:
                    if self.processor is not None:
                        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
                        vectors = self.model.get_image_features(**inputs)
                    else:
                        pixels = torch.stack([self.transform(image) for image in images]).to(self.device, self.dtype)
                        vectors = self.model.encode_image(pixels)
                finally:
                    for image in images:
                        image.close()
            else:
                texts = [self.clean_text(text) for text in values]
                # Never discard a role, negation or action at the encoder boundary.
                # The search layer may split the original query into bounded clauses.
                raw = self.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
                if any(len(ids) > self.context_length for ids in raw["input_ids"]):
                    raise ValueError("查询超过编码器 token 上限；请保留约束分句检索，不能静默截断")
                if self.processor is not None:
                    inputs = self.processor(text=texts, padding="max_length", max_length=self.context_length,
                                            truncation=False, return_tensors="pt").to(self.device)
                    vectors = self.model.get_text_features(**inputs)
                else:
                    tokens = self.tokenizer(texts, padding="max_length", max_length=self.context_length,
                                            truncation=False, return_tensors="pt")["input_ids"].to(self.device)
                    vectors = self.model.encode_text(tokens)
            vectors = vectors.float()
            norms = vectors.norm(dim=-1, keepdim=True)
            if not torch.isfinite(vectors).all() or not (norms > 0).all():
                raise ValueError("模型返回非有限或零向量")
            return (vectors / norms).cpu().tolist()

    def encode(self, action, values):
        if action not in {"images", "texts"}:
            raise ValueError("未知编码操作")
        started = time.monotonic()
        vectors, self.batch_size, retries = encode_batches(
            values, self.batch_size, lambda batch: self._batch(action, batch),
            lambda exc: isinstance(exc, self.torch.cuda.OutOfMemoryError) and self.device.type == "cuda",
            self.torch.cuda.empty_cache)
        self.metadata.update(effective_batch_size=self.batch_size,
                             oom_retries=self.metadata["oom_retries"] + retries,
                             inference_ms=round((time.monotonic() - started) * 1000),
                             item_count=len(values), dimensions=len(vectors[0]) if vectors else 0)
        if self.device.type == "cuda":
            self.metadata["peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated(self.device)
        return vectors


def main():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    output = sys.stdout
    runtime = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if runtime is None:
                    runtime = EncoderRuntime(request["config"], request["identity"])
                if runtime.identity["signature"] != request["identity"]["signature"]:
                    raise ValueError("worker 模型签名不一致")
                vectors = runtime.encode(request["action"], request["values"])
            response = {"vectors": vectors, "signature": runtime.identity["signature"], "metadata": runtime.metadata}
        except Exception as exc:
            # Paths and private input content must not be copied into UI errors.
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) and str(exc).startswith(
                ("请求的 CUDA", "查询超过", "视觉编码实际", "编码环境依赖", "worker 模型")) else type(exc).__name__
            response = {"error": "本地视觉编码失败：" + message + "；请检查离线模型环境，不会自动下载。"}
        output.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
