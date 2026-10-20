"""Offline encoder contracts and real pipe supervision, without ML or downloads."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from types import ModuleType, SimpleNamespace

from PIL import Image
import psutil
import pytest

from scenerecall import visual_encoder
from scenerecall.visual_encoder import VisualEncoder, VisualEncoderUnavailable
from scenerecall.visual_encoder_worker import EncoderRuntime, encode_batches


@pytest.fixture
def configured(tmp_path, monkeypatch):
    model = tmp_path / "private-model"
    model.mkdir()
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"test-only weights, never loaded")
    config = {"encoder_id": "siglip2-base-p16-256", "model_dir": str(model), "python": sys.executable,
              "revision": "a" * 40, "timeout_s": 5}
    path = tmp_path / "retrieval-encoder.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(VisualEncoder, "_runtime_versions", lambda self, config: {"torch": "test-only"})
    return VisualEncoder(tmp_path), config, path, model


def write_config(path, config, **changes):
    path.write_text(json.dumps({**config, **changes}), encoding="utf-8")


def fake_command(monkeypatch, source):
    def command(config):
        child = [sys.executable, "-u", "-c", textwrap.dedent(source)]
        if os.name == "nt":
            return [getattr(sys, "_base_executable", sys.executable), "-I",
                    str(Path(visual_encoder.__file__).with_name("_windows_process.py")), *child]
        return child
    monkeypatch.setattr(visual_encoder, "_worker_command", command)


ECHO_WORKER = """
import json, os, sys
for line in sys.stdin:
    request = json.loads(line)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    print(json.dumps({"signature": request["identity"]["signature"],
          "vectors": [[.6, .8] for _ in request["values"]],
          "metadata": {"worker_pid": os.getpid(), "offline": True}}), flush=True)
"""


def test_unconfigured_status_is_explicit_and_never_starts_process(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Unexpected subprocess"))
    encoder = VisualEncoder(tmp_path)
    assert encoder.settings()["available"] is False
    assert "不会自动下载" in encoder.settings()["reason"]
    with pytest.raises(VisualEncoderUnavailable, match="尚未配置"):
        encoder.identity()


def test_identity_tracks_actual_weights_processor_tokenizer_and_device_without_paths(configured):
    encoder, config, path, model = configured
    original = encoder.identity()
    assert original == encoder.identity()
    assert original["repo"] == "google/siglip2-base-patch16-256"
    assert original["dtype"] == "float16"
    assert str(model) not in json.dumps(original)
    assert str(model) not in json.dumps(encoder.settings())
    (model / "model.safetensors").write_bytes(b"changed weights")
    weight_changed = encoder.identity()
    assert weight_changed["signature"] != original["signature"]
    assert weight_changed["weight_hashes"] != original["weight_hashes"]
    (model / "preprocessor_config.json").write_text('{"size": 256}', encoding="utf-8")
    processor_changed = encoder.identity()
    assert processor_changed["preprocessor_hash"] != weight_changed["preprocessor_hash"]
    (model / "tokenizer.json").write_text('{"vocab": "changed"}', encoding="utf-8")
    assert encoder.identity()["tokenizer_hash"] != processor_changed["tokenizer_hash"]
    write_config(path, config, device="cpu")
    cpu = encoder.identity()
    assert cpu["device"] == "cpu" and cpu["dtype"] == "float32"
    assert cpu["signature"] != original["signature"]


@pytest.mark.parametrize("change, message", [
    ({"revision": "main"}, "固定提交"),
    ({"encoder_id": "third-encoder"}, "不支持"),
    ({"device": "auto"}, "device"),
    ({"batch_size": True}, "batch_size"),
    ({"timeout_s": float("inf")}, "timeout_s"),
    ({"api_key": "private"}, "未声明"),
])
def test_invalid_or_floating_configuration_is_rejected(configured, change, message):
    encoder, config, path, _ = configured
    write_config(path, config, **change)
    with pytest.raises(VisualEncoderUnavailable, match=message):
        encoder.identity()
    assert "private" not in json.dumps(encoder.settings())


def test_missing_weights_and_unsafe_shard_names_fail_before_runtime_probe(configured, monkeypatch):
    encoder, _, _, model = configured
    (model / "model.safetensors").unlink()
    monkeypatch.setattr(encoder, "_runtime_versions", lambda config: pytest.fail("Unexpected runtime probe"))
    with pytest.raises(VisualEncoderUnavailable, match="缺少"):
        encoder.identity()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "../outside.safetensors"}}), encoding="utf-8")
    with pytest.raises(VisualEncoderUnavailable, match="分片"):
        encoder.identity()


def test_openclip_requires_local_text_configuration_and_separate_namespace(configured):
    encoder, config, path, model = configured
    original = encoder.identity()
    write_config(path, config, encoder_id="openclip-multilingual-b32")
    with pytest.raises(VisualEncoderUnavailable, match="XLM-R"):
        encoder.identity()
    text = model / "text_encoder"
    text.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (text / name).write_text("{}", encoding="utf-8")
    (model / "open_clip_config.json").write_text("{}", encoding="utf-8")
    other = encoder.identity()
    assert other["signature"] != original["signature"]
    assert other["repo"].startswith("laion/")


async def test_resident_pipe_preserves_batch_order_metadata_and_environment(configured, monkeypatch, tmp_path):
    encoder, _, _, _ = configured
    original_offline = os.environ.get("HF_HUB_OFFLINE")
    fake_command(monkeypatch, ECHO_WORKER)
    picture = tmp_path / "evidence.png"
    Image.new("RGB", (12, 8)).save(picture)
    item = {"id": "f1", "path": str(picture), "asset_id": "a1", "source_fingerprint": "source",
            "pts": 90, "time_base": "1/30", "box": [0, 0, 1, 1]}
    try:
        assert await encoder.encode_images([item, {**item, "id": "f2", "time_base": [1, 30]}]) == [[.6, .8], [.6, .8]]
        pid = encoder.metadata["worker_pid"]
        assert await encoder.encode_texts(["谁递给谁", "背景"]) == [[.6, .8], [.6, .8]]
        assert encoder.metadata["worker_pid"] == pid
        assert os.environ.get("HF_HUB_OFFLINE") == original_offline
    finally:
        await encoder.close()
    with contextlib.suppress(psutil.NoSuchProcess):
        psutil.Process(pid).wait(timeout=3)
    assert not psutil.pid_exists(pid)


async def test_timeout_kills_supervised_worker_and_allows_fresh_retry(configured, monkeypatch):
    encoder, config, path, _ = configured
    fake_command(monkeypatch, """
        import json, os, sys, time
        for line in sys.stdin:
            request = json.loads(line)
            if request["values"] == ["stall"]:
                sys.stdout.write("{")
                sys.stdout.flush()
                time.sleep(30)
            print(json.dumps({"signature": request["identity"]["signature"],
                  "vectors": [[1., 0.]], "metadata": {"worker_pid": os.getpid()}}), flush=True)
    """)
    write_config(path, config, timeout_s=1)
    await encoder.encode_texts(["first"])
    process = psutil.Process(encoder.metadata["worker_pid"])
    try:
        with pytest.raises(VisualEncoderUnavailable, match="超时"):
            await encoder.encode_texts(["stall"])
        assert encoder._process is None
        process.wait(timeout=3)
        assert await encoder.encode_texts(["retry"]) == [[1., 0.]]
        assert encoder.metadata["worker_pid"] != process.pid
    finally:
        await encoder.close()


async def test_cancel_stalled_call_releases_process_and_model_gate(configured, monkeypatch):
    from scenerecall.local_resources import LOCAL_MODEL_GATE
    encoder, _, _, _ = configured
    fake_command(monkeypatch, """
        import sys, time
        sys.stdin.readline()
        sys.stdout.write("{")
        sys.stdout.flush()
        time.sleep(30)
    """)
    call = asyncio.create_task(encoder.encode_texts(["stall"]))
    for _ in range(100):
        if encoder._process is not None:
            break
        await asyncio.sleep(.01)
    assert encoder._process is not None
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=3)
    assert encoder._process is None
    assert LOCAL_MODEL_GATE.acquire(blocking=False)
    LOCAL_MODEL_GATE.release()
    await encoder.close()


async def test_cancelling_waiting_request_does_not_kill_active_request(configured, monkeypatch):
    encoder, _, _, _ = configured
    entered, release = threading.Event(), threading.Event()
    original_request = encoder._request

    def request(action, values, cancelled):
        if values == ["active"]:
            encoder._request_lock.acquire()
            entered.set()
            release.wait(timeout=5)
            encoder._request_lock.release()
            return [[1., 0.]]
        return original_request(action, values, cancelled)

    stopped = []
    monkeypatch.setattr(encoder, "_request", request)
    monkeypatch.setattr(encoder, "_stop", lambda: stopped.append(True))
    first = asyncio.create_task(encoder.encode_texts(["active"]))
    await asyncio.to_thread(entered.wait, 2)
    second = asyncio.create_task(encoder.encode_texts(["queued"]))
    await asyncio.sleep(.1)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert stopped == []
    release.set()
    assert await first == [[1., 0.]]


@pytest.mark.parametrize("vectors", [[[0., 0.]], [[float("nan"), 1.]], [[1., 1.]], [[1., 0.], [1.]]])
def test_nonfinite_zero_unnormalized_or_mixed_dimension_vectors_rejected(vectors):
    with pytest.raises(VisualEncoderUnavailable):
        VisualEncoder._vectors(vectors, len(vectors))


async def test_wrong_worker_signature_is_rejected_and_stopped(configured, monkeypatch):
    encoder, _, _, _ = configured
    fake_command(monkeypatch, """
        import json, sys
        sys.stdin.readline()
        print(json.dumps({"signature": "wrong", "vectors": [[1., 0.]]}), flush=True)
    """)
    try:
        with pytest.raises(VisualEncoderUnavailable, match="签名"):
            await encoder.encode_texts(["text"])
        assert encoder._process is None
    finally:
        await encoder.close()


async def test_bad_frame_provenance_or_region_never_reaches_worker(configured, tmp_path, monkeypatch):
    encoder, _, _, _ = configured
    picture = tmp_path / "frame.png"
    Image.new("RGB", (4, 4)).save(picture)
    item = {"id": "f", "asset_id": "a", "source_fingerprint": "s", "pts": 0, "time_base": "1/24",
            "path": str(picture)}
    monkeypatch.setattr(encoder, "_encode", lambda *args: pytest.fail("Unexpected worker"))
    for changes in ({"pts": 1.2}, {"time_base": "0"}, {"source_fingerprint": ""},
                    {"time_base": [1, 0]}, {"time_base": [1.5, 30]},
                    {"box": [0, 0, 2, 1]}, {"box": [0, 0, float("nan"), 1]},
                    {"bbox": [0, 0, .5, .5], "box": [0, 0, 1, 1]}):
        with pytest.raises(ValueError):
            await encoder.encode_images([{**item, **changes}])


def test_oom_halves_only_failed_batch_without_losing_or_repeating_successes():
    calls, clears = [], []

    class OOM(RuntimeError):
        pass

    def encode(batch):
        calls.append(list(batch))
        if len(batch) > 2:
            raise OOM
        return [[value] for value in batch]

    result, size, retries = encode_batches(list(range(7)), 4, encode, lambda exc: isinstance(exc, OOM),
                                           lambda: clears.append(True))
    assert result == [[i] for i in range(7)]
    assert calls == [[0, 1, 2, 3], [0, 1], [2, 3], [4, 5], [6]]
    assert size == 2 and retries == 1 and len(clears) == 1
    with pytest.raises(OOM):
        encode_batches([0], 1, lambda batch: (_ for _ in ()).throw(OOM()),
                       lambda exc: isinstance(exc, OOM), lambda: None)


def test_region_crops_full_scene_with_normalized_xyxy(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (20, 10), "red").save(image)
    runtime = EncoderRuntime.__new__(EncoderRuntime)
    runtime.Image = Image
    picture = runtime._image({"path": str(image), "box": [.25, .2, .75, .8]})
    try:
        assert picture.size == (10, 6)
    finally:
        picture.close()


def test_requested_cuda_unavailable_does_not_attempt_cpu_or_load_model(configured, monkeypatch):
    encoder, _, path, _ = configured
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-only")
    with pytest.raises(RuntimeError, match="拒绝静默回退"):
        EncoderRuntime(visual_encoder._settings(path), encoder.identity())


def test_explicit_cpu_runtime_reports_actual_provider_and_local_only_loading(configured, monkeypatch):
    encoder, config, path, _ = configured
    write_config(path, config, device="cpu")
    torch = ModuleType("torch")

    class Device:
        def __init__(self, value):
            self.type = value

        def __str__(self):
            return self.type

    class Model:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([SimpleNamespace(device=self.device)])

    calls = []
    torch.device = Device
    torch.float32 = "float32"
    torch.float16 = "float16"
    transformers = ModuleType("transformers")

    def model_load(*args, **kwargs):
        calls.append(kwargs)
        return Model()

    def processor_load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(tokenizer=lambda *args, **kwargs: None)

    transformers.AutoModel = SimpleNamespace(from_pretrained=model_load)
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=processor_load)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-only")
    runtime = EncoderRuntime(visual_encoder._settings(path), encoder.identity())
    assert runtime.metadata["provider"] == "pytorch-cpu"
    assert runtime.metadata["dtype"] == "float32" and runtime.metadata["cpu_explicit"] is True
    assert all(call["local_files_only"] and call["trust_remote_code"] is False for call in calls)


def test_query_token_overflow_cannot_silently_discard_final_constraints():
    runtime = EncoderRuntime.__new__(EncoderRuntime)
    runtime.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    runtime.clean_text = str.lower
    runtime.context_length = 64
    runtime.tokenizer = lambda *args, **kwargs: {"input_ids": [list(range(65))]}
    with pytest.raises(ValueError, match="不能静默截断"):
        runtime._batch("texts", ["role one gives role two an object but not the red one"])
