"""Bounded CPU image lookahead; no model downloads or GPU dependency."""
from __future__ import annotations

import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scenerecall.visual_encoder_worker import EncoderRuntime
from scenerecall import visual_encoder_worker


def runtime(batch_size=2):
    result = EncoderRuntime.__new__(EncoderRuntime)
    result.Image = Image
    result.config = {"batch_size": batch_size}
    result.batch_size = batch_size
    result.device = SimpleNamespace(type="cuda")
    result.torch = SimpleNamespace(cuda=SimpleNamespace(
        OutOfMemoryError=MemoryError, empty_cache=lambda: None, max_memory_allocated=lambda device: 0))
    result.metadata = {"oom_retries": 0}
    return result


def image_items(tmp_path, count=4):
    items = []
    for i in range(count):
        path = tmp_path / f"frame-{i}.png"
        image = Image.new("RGB", (17 + i, 13 + i), (i * 30, 255 - i * 20, i * 10))
        image.save(path)
        image.close()
        items.append({"path": str(path), "box": None if i % 2 == 0 else [.2, .1, .9, .8]})
    return items


class PixelProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": np.stack([np.asarray(image.resize((8, 8))) for image in images])}


def test_parallel_decode_keeps_original_order_crops_and_batch_preprocessing(tmp_path):
    worker = runtime()
    worker.processor = PixelProcessor()
    items = image_items(tmp_path)
    images = [worker._image(item) for item in items]
    try:
        expected = worker.processor(images=images, return_tensors="pt")["pixel_values"]
    finally:
        for image in images:
            image.close()
    try:
        actual = worker._prepare_images(items)["pixel_values"]
        np.testing.assert_array_equal(actual, expected)
    finally:
        worker.close()
    assert not hasattr(worker, "_image_read_pool")


def test_next_cpu_batch_overlaps_current_gpu_and_is_consumed_once(tmp_path):
    worker = runtime()
    items = image_items(tmp_path)
    current_gpu = threading.Event()
    next_preparing = threading.Event()
    next_ready = threading.Event()
    prepared, inferred = [], []
    main_thread = threading.get_ident()

    def prepare(values):
        numbers = [int(Path(item["path"]).stem.split('-')[-1]) for item in values]
        prepared.append(numbers)
        if numbers == [2, 3]:
            assert threading.get_ident() != main_thread
            next_preparing.set()
            assert current_gpu.wait(3)
            next_ready.set()
        return {"pixel_values": np.array(numbers)}

    def infer(action, values, prepared=None):
        assert action == "images" and threading.get_ident() == main_thread
        numbers = prepared["pixel_values"].tolist()
        inferred.append(numbers)
        if numbers == [0, 1]:
            assert next_preparing.wait(3)
            current_gpu.set()
            assert next_ready.wait(3)
        return [[number, 1] for number in numbers]

    worker._prepare_images, worker._batch = prepare, infer
    try:
        assert worker.encode("images", items[:2], items[2:]) == [[0, 1], [1, 1]]
        assert worker.encode("images", items[2:]) == [[2, 1], [3, 1]]
        assert prepared == [[0, 1], [2, 3]]
        assert inferred == prepared
        assert worker.metadata["image_prefetch_hits"] == 1
        assert worker._image_prefetch is None
    finally:
        worker.close()


def test_mismatched_lookahead_cannot_supply_other_images_or_crops(tmp_path):
    worker = runtime()
    items = image_items(tmp_path)
    calls = []

    def prepare(values):
        calls.append([dict(item) for item in values])
        return {"pixel_values": np.arange(len(values))}

    worker._prepare_images = prepare
    worker._batch = lambda action, values, prepared=None: [[1] for _ in values]
    try:
        worker.encode("images", items[:2], items[2:])
        other = [{**items[2], "box": [.1, .1, .8, .8]}, items[3]]
        worker.encode("images", other)
        assert calls[-1] == other
        assert worker.metadata.get("image_prefetch_hits", 0) == 0
    finally:
        worker.close()


def test_replaced_file_is_prepared_again_instead_of_using_stale_pixels(tmp_path):
    worker = runtime()
    worker.processor = PixelProcessor()
    items = image_items(tmp_path)
    decoded = []
    original = worker._prepare_images

    def prepare(values):
        decoded.append([item["path"] for item in values])
        return original(values)

    worker._prepare_images = prepare
    worker._batch = lambda action, values, prepared=None: [[float(pixel.mean())]
                                                          for pixel in prepared["pixel_values"]]
    try:
        worker.encode("images", items[:2], items[2:])
        future = worker._image_prefetch[1]
        future.result(timeout=3)
        Image.new("RGB", (40, 30), "blue").save(items[2]["path"])
        worker.encode("images", items[2:])
        assert decoded.count([item["path"] for item in items[2:]]) == 2
        assert worker.metadata.get("image_prefetch_hits", 0) == 0
    finally:
        worker.close()


def test_oom_splits_only_failed_gpu_batch_with_corresponding_cpu_slices(tmp_path):
    worker = runtime(batch_size=4)
    items = image_items(tmp_path, count=6)
    prepared_calls, gpu_calls = [], []

    def prepare(values):
        numbers = [int(Path(item["path"]).stem.split('-')[-1]) for item in values]
        prepared_calls.append(numbers)
        return {"pixel_values": np.array(numbers)}

    def infer(action, values, prepared=None):
        numbers = prepared["pixel_values"].tolist()
        gpu_calls.append(numbers)
        if len(numbers) > 2:
            raise MemoryError()
        return [[number, 1] for number in numbers]

    worker._prepare_images, worker._batch = prepare, infer
    try:
        assert worker.encode("images", items) == [[number, 1] for number in range(6)]
        assert gpu_calls == [[0, 1, 2, 3], [0, 1], [2, 3], [4, 5]]
        assert prepared_calls == [[0, 1, 2, 3], [4, 5]]
        assert worker.batch_size == 2 and worker.metadata["oom_retries"] == 1
    finally:
        worker.close()


def test_decode_error_closes_every_successful_image(tmp_path):
    worker = runtime()
    worker.processor = PixelProcessor()
    items = image_items(tmp_path)
    decoded = []
    original = worker._image

    def image(item):
        if item == items[1]:
            raise OSError("decode failed")
        result = original(item)
        decoded.append(result)
        return result

    worker._image = image
    try:
        with pytest.raises(OSError, match="decode failed"):
            worker._prepare_images(items)
        assert len(decoded) == 3
        for picture in decoded:
            with pytest.raises(ValueError, match="closed image"):
                picture.getpixel((0, 0))
    finally:
        worker.close()


def test_failed_model_call_releases_prefetch_and_threads_on_close(tmp_path):
    worker = runtime()
    items = image_items(tmp_path)
    worker._prepare_images = lambda values: {"pixel_values": np.arange(len(values))}
    worker._batch = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed"))
    with pytest.raises(RuntimeError, match="model failed"):
        worker.encode("images", items[:2], items[2:])
    assert worker._image_prefetch is None
    worker.close()
    worker.close()
    assert not hasattr(worker, "_image_prepare_pool")


def test_text_request_discards_unused_image_lookahead(tmp_path):
    worker = runtime()
    items = image_items(tmp_path)
    worker._prepare_images = lambda values: {"pixel_values": np.arange(len(values))}
    worker._batch = lambda action, values, prepared=None: [[1] for _ in values]
    try:
        worker.encode("images", items[:2], items[2:])
        assert worker._image_prefetch is not None
        worker.encode("texts", ["query"])
        assert worker._image_prefetch is None
    finally:
        worker.close()


@pytest.mark.parametrize("interrupt", [False, True])
def test_worker_protocol_passes_lookahead_and_closes_runtime_on_eof_or_interrupt(monkeypatch, interrupt):
    request = {"config": {}, "identity": {"signature": "test-signature"},
               "action": "images", "values": [{"path": "current", "box": None}],
               "prefetch_values": [{"path": "following", "box": None}]}
    events = []

    class Runtime:
        def __init__(self, config, identity):
            self.identity, self.metadata = identity, {}

        def encode(self, action, values, prefetch_values=None):
            events.append((action, values, prefetch_values))
            return [[1]]

        def close(self):
            events.append("closed")

    def lines():
        yield json.dumps(request) + '\n'
        if interrupt:
            raise KeyboardInterrupt()

    output = io.StringIO()
    monkeypatch.setattr(visual_encoder_worker, "EncoderRuntime", Runtime)
    monkeypatch.setattr(visual_encoder_worker.sys, "stdin", lines())
    monkeypatch.setattr(visual_encoder_worker.sys, "stdout", output)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            visual_encoder_worker.main()
    else:
        visual_encoder_worker.main()
    response = json.loads(output.getvalue())
    assert response == {"vectors": [[1]], "signature": "test-signature", "metadata": {}}
    assert events == [("images", request["values"], request["prefetch_values"]), "closed"]
