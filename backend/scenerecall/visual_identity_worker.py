"""Private JSON-line worker for the optional imgutils extra (offline only)."""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def configure_onnx_runtime():
    """Opt-in CUDA DLL loading and FP32 parity controls for an isolated worker.

    Do not import optional ML dependencies or change imgutils defaults unless a
    runtime option is explicitly set. Model/cache identity remains unchanged.
    """
    directory = os.environ.get("SCENERECALL_CUDA_DLL_DIRECTORY", "").strip()
    disable_tf32 = os.environ.get("SCENERECALL_ORT_DISABLE_TF32", "").strip().lower() in {"1", "true", "yes"}
    if not directory and not disable_tf32:
        return None
    with contextlib.redirect_stdout(sys.stderr):
        import onnxruntime as ort

        if directory:
            path = Path(directory).expanduser().resolve()
            if not path.is_dir():
                raise ValueError("SCENERECALL_CUDA_DLL_DIRECTORY 不是本机 DLL 目录")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("当前视觉环境未安装 CUDA ONNX Runtime")
            # ORT can print dependency diagnostics to stdout; keep the worker's
            # stdout exclusively available for one JSON reply per input line.
            ort.preload_dlls(directory=str(path))
        import imgutils.utils.onnxruntime as img_ort

    runtime = {"onnxruntime_version": ort.__version__, "cuda_dll_directory": directory or None,
               "disable_tf32": disable_tf32, "sessions": []}
    original_session = img_ort.InferenceSession

    def open_session(*args, **kwargs):
        providers = kwargs.get("providers", [])
        requested_cuda = any((item[0] if isinstance(item, tuple) else item) == "CUDAExecutionProvider"
                             for item in providers)
        if disable_tf32:
            configured = []
            for item in providers:
                provider, options = item if isinstance(item, tuple) else (item, {})
                configured.append((provider, {**options, "use_tf32": 0})
                                  if provider == "CUDAExecutionProvider" else item)
            kwargs["providers"] = configured
        session = original_session(*args, **kwargs)
        actual = session.get_providers()
        if directory and requested_cuda and "CUDAExecutionProvider" not in actual:
            raise RuntimeError("视觉模型未成功启用 CUDA provider；拒绝静默回退 CPU")
        runtime["sessions"].append({"model": Path(str(args[0] if args else kwargs.get("path_or_bytes", ""))).name,
                                    "providers": actual, "provider_options": session.get_provider_options()})
        return session

    img_ort.InferenceSession = open_session
    return runtime


def main():
    # These flags must be set before importing Hugging Face or imgutils.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    output = sys.stdout
    model = sys.argv[1]
    backend = None
    runtime = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if backend is None:
                    runtime = configure_onnx_runtime()
                    import numpy as np
                    from huggingface_hub import hf_hub_download
                    from imgutils.detect import detect_person
                    from imgutils.metrics import ccip_extract_feature, ccip_difference
                    from PIL import Image
                    filenames = [("deepghs/ccip_onnx", f"{model}/{name}")
                                 for name in ("model_feat.onnx", "model_metrics.onnx")]
                    filenames.append(("deepghs/anime_person_detection", "person_detect_v1.1_m/model.onnx"))
                    snapshots = []
                    for repo_id, filename in filenames:
                        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True))
                        # Cache commit directory is part of the feature key. Do
                        # not hash model names alone across mutable snapshots.
                        parts = path.parts
                        commit = parts[parts.index("snapshots") + 1] if "snapshots" in parts else str(path.stat().st_mtime_ns)
                        snapshots.append(f"{repo_id}@{commit}")
                    version = f"imgutils-{importlib.metadata.version('dghs-imgutils')}:{model}:" + ";".join(snapshots)
                    # Load all three ONNX sessions here so model-load time is
                    # charged to startup, not hidden in the first video frame.
                    warm_image = Image.new("RGB", (384, 384))
                    warm_feature = ccip_extract_feature(warm_image, size=384, model=model)
                    ccip_difference(warm_feature, warm_feature, size=384, model=model)
                    detect_person(warm_image, level="m", version="v1.1")
                    backend = (np, detect_person, ccip_extract_feature, ccip_difference, version)
                np, detect_person, extract_feature, difference, version = backend
                if request["action"] == "metadata":
                    result = {"model_version": version, "offline": True}
                    if runtime is not None:
                        result["runtime"] = runtime
                elif request["action"] == "detect":
                    result = [{"box": list(box), "label": label, "detection_score": float(score)}
                              for box, label, score in detect_person(request["path"], level="m", version="v1.1")]
                elif request["action"] == "feature":
                    result = extract_feature(request["path"], size=384, model=model).tolist()
                elif request["action"] == "distances":
                    query = np.asarray(request["query"], dtype=np.float32)
                    result = [float(difference(query, np.asarray(feature, dtype=np.float32), size=384, model=model))
                              for feature in request["features"]]
                else:
                    raise ValueError("未知 worker 操作")
            response = {"result": result}
        except Exception as exc:
            response = {"error": f"imgutils 未完成识别 ({type(exc).__name__}): {str(exc)[:500]}。请安装 visual-identity extra 并预先缓存合法权重；不会自动下载。"}
        output.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        output.flush()


if __name__ == "__main__":
    main()
