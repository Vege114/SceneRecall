"""Opt-in worker runtime controls with fake sessions, without ML or downloads."""
import io
import json
import sys
from types import ModuleType

import pytest

from scenerecall import visual_identity_worker
from scenerecall.visual_identity_worker import cached_weight_snapshots, configure_onnx_runtime


def fake_weight_cache(monkeypatch, tmp_path, missing=None):
    hub = ModuleType("huggingface_hub")
    constants = ModuleType("huggingface_hub.constants")
    errors = ModuleType("huggingface_hub.errors")
    constants.HF_HUB_CACHE = str(tmp_path / "hub")

    class LocalEntryNotFoundError(FileNotFoundError):
        pass

    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    calls = []

    def download(*, repo_id, filename, local_files_only):
        calls.append((repo_id, filename, local_files_only))
        if filename == missing:
            raise LocalEntryNotFoundError("not cached")
        commit = "ccip-commit" if repo_id.endswith("ccip_onnx") else "person-commit"
        path = tmp_path / "hub" / repo_id.replace("/", "--") / "snapshots" / commit / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cached weight")
        return str(path)

    hub.hf_hub_download = download
    for name, module in (("huggingface_hub", hub), ("huggingface_hub.constants", constants),
                         ("huggingface_hub.errors", errors)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_weight_preflight_uses_selected_model_and_offline_cache_commits(monkeypatch, tmp_path):
    calls = fake_weight_cache(monkeypatch, tmp_path)
    assert cached_weight_snapshots("ccip-caformer-5_fp32") == [
        "deepghs/ccip_onnx@ccip-commit", "deepghs/ccip_onnx@ccip-commit",
        "deepghs/anime_person_detection@person-commit"]
    assert calls == [("deepghs/ccip_onnx", "ccip-caformer-5_fp32/model_feat.onnx", True),
                     ("deepghs/ccip_onnx", "ccip-caformer-5_fp32/model_metrics.onnx", True),
                     ("deepghs/anime_person_detection", "person_detect_v1.1_m/model.onnx", True)]


@pytest.mark.parametrize("filename", ["ccip-caformer-24-randaug-pruned/model_feat.onnx",
                                      "ccip-caformer-24-randaug-pruned/model_metrics.onnx",
                                      "person_detect_v1.1_m/model.onnx"])
def test_missing_weight_reports_exact_file_and_cache_without_fallback(monkeypatch, tmp_path, filename):
    calls = fake_weight_cache(monkeypatch, tmp_path, missing=filename)
    with pytest.raises(RuntimeError) as error:
        cached_weight_snapshots("ccip-caformer-24-randaug-pruned")
    assert filename in str(error.value)
    assert str(tmp_path / "hub") in str(error.value)
    assert "不会联网或自动切换模型" in str(error.value)
    assert all(call[2] is True and "5_fp32" not in call[1] for call in calls)


def test_missing_weight_fails_before_importing_or_starting_ml_runtime(monkeypatch, tmp_path, capsys):
    filename = "ccip-caformer-24-randaug-pruned/model_feat.onnx"
    fake_weight_cache(monkeypatch, tmp_path, missing=filename)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setattr(sys, "argv", ["visual_identity_worker", "ccip-caformer-24-randaug-pruned"])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"action":"metadata"}\n'))

    def forbidden_runtime():
        pytest.fail("missing cache must fail before loading ML dependencies")

    monkeypatch.setattr(visual_identity_worker, "configure_onnx_runtime", forbidden_runtime)
    visual_identity_worker.main()
    response = json.loads(capsys.readouterr().out)
    assert filename in response["error"]
    assert "result" not in response


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
