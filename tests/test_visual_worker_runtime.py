"""Opt-in worker runtime controls with fake sessions, without ML or downloads."""
import json
import sys
from types import ModuleType

import pytest

from scenerecall.visual_identity_worker import configure_onnx_runtime


def fake_runtime(monkeypatch, providers):
    ort = ModuleType("onnxruntime")
    ort.__version__ = "test-ort"
    ort.get_available_providers = lambda: list(providers)
    calls = []
    ort.preload_dlls = lambda **kwargs: print("preload diagnostic")
    imgutils = ModuleType("imgutils")
    utils = ModuleType("imgutils.utils")
    img_ort = ModuleType("imgutils.utils.onnxruntime")

    class Session:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))

        def get_providers(self):
            return list(providers)

        def get_provider_options(self):
            return {provider: {} for provider in providers}

    img_ort.InferenceSession = Session
    imgutils.utils = utils
    utils.onnxruntime = img_ort
    for name, module in (("onnxruntime", ort), ("imgutils", imgutils), ("imgutils.utils", utils),
                         ("imgutils.utils.onnxruntime", img_ort)):
        monkeypatch.setitem(sys.modules, name, module)
    return ort, img_ort, calls


def test_default_worker_does_not_import_or_modify_optional_onnx_runtime(monkeypatch):
    monkeypatch.delenv("SCENERECALL_CUDA_DLL_DIRECTORY", raising=False)
    monkeypatch.delenv("SCENERECALL_ORT_DISABLE_TF32", raising=False)
    assert configure_onnx_runtime() is None


def test_opt_in_preload_keeps_stdout_clean_and_disables_tf32_only_for_cuda(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SCENERECALL_CUDA_DLL_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("SCENERECALL_ORT_DISABLE_TF32", "1")
    _, img_ort, calls = fake_runtime(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])
    runtime = configure_onnx_runtime()
    marker = object()
    img_ort.InferenceSession("model_feat.onnx", sess_options=marker,
                             providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "preload diagnostic\n"
    assert calls[0][1] == {"sess_options": marker,
                           "providers": [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"]}
    assert runtime["sessions"][0]["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime["sessions"][0]["model"] == "model_feat.onnx"
    assert json.loads(json.dumps(runtime))["disable_tf32"] is True


def test_requested_cuda_cannot_silently_fall_back_to_cpu(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENERECALL_CUDA_DLL_DIRECTORY", str(tmp_path))
    monkeypatch.delenv("SCENERECALL_ORT_DISABLE_TF32", raising=False)
    ort, img_ort, _ = fake_runtime(monkeypatch, ["CPUExecutionProvider"])
    ort.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    configure_onnx_runtime()
    with pytest.raises(RuntimeError, match="拒绝静默回退"):
        img_ort.InferenceSession("model_feat.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_missing_gpu_package_is_reported_before_attempting_a_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENERECALL_CUDA_DLL_DIRECTORY", str(tmp_path))
    monkeypatch.delenv("SCENERECALL_ORT_DISABLE_TF32", raising=False)
    _, _, calls = fake_runtime(monkeypatch, ["AzureExecutionProvider", "CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="未安装 CUDA"):
        configure_onnx_runtime()
    assert calls == []
