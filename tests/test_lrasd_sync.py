"""LR-ASD bundle and preprocessing contracts; not an accuracy benchmark."""
import hashlib
import json
import math
import os
import sys
import types
import wave
from pathlib import Path

import pytest

from scenerecall.av_sync import sync_availability, validate_track
from scenerecall.av_sync_worker import (LRASD_ARCHITECTURE, LRASD_PREPROCESS, LRASD_SCHEMA,
                                       LRASD_SCORE_DOMAIN, aligned_lrasd_inputs, load_lrasd_model,
                                       lrasd_main, lrasd_visual, upstream_modules, validate_lrasd_bundle)


def write_bundle(directory, **overrides):
    directory.mkdir(exist_ok=True)
    (directory / "model.pt").write_bytes(b"test checkpoint; metadata validation does not unpickle")
    metadata = {"schema_version": LRASD_SCHEMA, "model_type": "lrasd", "architecture": LRASD_ARCHITECTURE,
                "checkpoint": "model.pt", "checkpoint_sha256": hashlib.sha256((directory / "model.pt").read_bytes()).hexdigest(),
                "preprocessing": LRASD_PREPROCESS, "score_domain": LRASD_SCORE_DOMAIN,
                "visual_canvas": {"width": 480, "height": 270, "resize": "ffmpeg-bilinear"},
                "dataset": {"name": "test", "version": "v1", "crop_type": "full_body_bbox",
                            "split_manifest_sha256": "1" * 64},
                "training": {"status": "completed", "epochs_completed": 1, "steps": 20},
                "validation": {"status": "completed", "version": "v1", "split": "heldout-videos",
                               "metrics": {"ap": .7}}, "calibration": {"status": "uncalibrated"}}
    metadata.update(overrides)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def test_bundle_requires_finished_training_and_content_hash(tmp_path):
    expected = write_bundle(tmp_path)
    assert validate_lrasd_bundle(tmp_path) == (expected, tmp_path / "model.pt")
    (tmp_path / "model.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="哈希"):
        validate_lrasd_bundle(tmp_path)
    write_bundle(tmp_path, training={"status": "running", "epochs_completed": 1, "steps": 20})
    with pytest.raises(ValueError, match="尚未完成"):
        validate_lrasd_bundle(tmp_path)


@pytest.mark.parametrize("overrides", [
    {"architecture": "other-architecture"}, {"score_domain": "cosine_similarity"},
    {"preprocessing": "normalized-gray"}, {"checkpoint": "../other.pt"},
    {"visual_canvas": {"width": 480, "height": 270, "resize": "bicubic"}},
    {"validation": {"status": "completed", "version": "v1", "split": "heldout", "metrics": {"ap": float("nan")}}},
    {"validation": {"status": "not_run"}},
])
def test_incompatible_or_unvalidated_bundle_is_rejected(tmp_path, overrides):
    write_bundle(tmp_path, **overrides)
    with pytest.raises(ValueError):
        validate_lrasd_bundle(tmp_path)


def test_lrasd_availability_is_explicit_and_keeps_lwtnet_behavior(tmp_path):
    bundle, repository = tmp_path / "bundle", tmp_path / "repo"
    write_bundle(bundle)
    for relative in ("model/Model.py", "model/Encoder.py", "model/Classifier.py", "loss.py"):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    config = {"sync_enabled": True, "sync_model_type": "lrasd", "sync_checkpoint_path": str(bundle),
              "sync_repository_path": str(repository), "sync_python_path": sys.executable}
    assert sync_availability(config)["available"]
    config["sync_model_type"] = "animated_ad_lwtnet"
    assert not sync_availability(config)["available"]


def test_unverified_teacher_bundle_is_available_only_with_weak_label_notice(tmp_path):
    bundle, repository = tmp_path / "bundle", tmp_path / "repo"
    write_bundle(bundle, domain_validation={"status": "unverified", "label_quality": "machine_teacher_consensus_weak"})
    for relative in ("model/Model.py", "model/Encoder.py", "model/Classifier.py", "loss.py"):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    config = {"sync_enabled": True, "sync_model_type": "lrasd", "sync_checkpoint_path": str(bundle),
              "sync_repository_path": str(repository), "sync_python_path": sys.executable}
    availability = sync_availability(config)
    assert availability["available"]
    assert "弱标签" in availability["reason"] and "尚未完成人工效果验证" in availability["reason"]


def test_external_generic_module_names_are_restored(tmp_path, monkeypatch):
    original = types.ModuleType("model")
    original_path = list(sys.path)
    monkeypatch.setitem(sys.modules, "model", original)
    with upstream_modules(tmp_path):
        assert "model" not in sys.modules
        sys.modules["model"] = types.ModuleType("model")
        sys.modules["model.temporary"] = types.ModuleType("temporary")
    assert sys.modules["model"] is original
    assert "model.temporary" not in sys.modules
    assert sys.path == original_path


def test_strict_state_loading_preserves_all_heads_and_rejects_wrong_shape(tmp_path):
    torch = pytest.importorskip("torch")
    class Architecture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for index in range(166):
                self.register_parameter(f"p{index}", torch.nn.Parameter(torch.zeros(1)))
    fake_model = types.SimpleNamespace(ASD_Model=Architecture)
    fake_loss = types.SimpleNamespace(lossAV=lambda: torch.nn.Linear(1, 1, bias=False),
                                     lossV=lambda: torch.nn.Linear(1, 1, bias=False))
    state = {f"model.p{index}": torch.ones(1) for index in range(166)}
    state.update({"lossAV.weight": torch.full((1, 1), 2.), "lossV.weight": torch.full((1, 1), 3.)})
    path = tmp_path / "model.pt"
    torch.save(state, path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scenerecall.av_sync_worker.importlib.import_module",
                      lambda name: fake_model if name == "model.Model" else fake_loss)
        model = load_lrasd_model(tmp_path, path, "cpu")
        assert model.lossAV.weight.item() == 2 and model.lossV.weight.item() == 3
        state["lossAV.weight"] = torch.ones(2, 2)
        torch.save(state, path)
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_lrasd_model(tmp_path, path, "cpu")
        del state["lossV.weight"]
        torch.save(state, path)
        with pytest.raises(ValueError, match="168"):
            load_lrasd_model(tmp_path, path, "cpu")


@pytest.mark.skipif(not os.environ.get("SCENERECALL_LRASD_REPOSITORY"), reason="Local LR-ASD checkout not specified")
def test_original_architecture_cpu_forward_and_full_state(tmp_path):
    torch = pytest.importorskip("torch")
    repository = Path(os.environ["SCENERECALL_LRASD_REPOSITORY"])
    with upstream_modules(repository):
        from model.Model import ASD_Model
        from loss import lossAV, lossV
        original = torch.nn.Module()
        original.model, original.lossAV, original.lossV = ASD_Model(), lossAV(), lossV()
    assert len(original.state_dict()) == 168
    checkpoint = tmp_path / "model.pt"
    torch.save(original.state_dict(), checkpoint)
    loaded = load_lrasd_model(repository, checkpoint, "cpu")
    with torch.inference_mode():
        audiovisual, _ = loaded.model(torch.zeros((1, 64, 13)), torch.zeros((1, 16, 112, 112)))
        probabilities = torch.softmax(loaded.lossAV.FC(audiovisual), dim=-1)
    assert probabilities.shape == (16, 2)
    assert torch.isfinite(probabilities).all()


def test_raw_gray_is_not_normalized_twice_and_boxes_use_source_time():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frames = np.zeros((16, 270, 480, 3), dtype=np.uint8)
    frames[:, :, 240:] = 255
    boxes = [{"at_ms": 1000 + index * 40, "box": [.5 if index % 2 else 0, 0, .5, 1]}
             for index in range(16)]
    visual = lrasd_visual(frames, boxes, 1000)
    assert visual.shape == (16, 112, 112)
    assert visual.dtype == np.float32
    assert visual[0].max() == 0 and visual[1].min() == 255
    # Normalization belongs to original ASD_Model.forward_visual_frontend.
    assert visual[1, 0, 0] / 255 == 1
    boxes[5]["at_ms"] += 4000
    with pytest.raises(ValueError, match="时间网格"):
        lrasd_visual(frames, boxes[5:], 1000)


def test_mfcc_pcm_magnitudes_and_exact_audio_video_alignment(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    frames = np.zeros((16, 270, 480, 3), dtype=np.uint8)
    frames.tofile(tmp_path / "frames.rgb")
    audio = np.full(16 * 640, 12000, dtype="<i2")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as source:
        source.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        source.writeframes(audio.tobytes())
    def mfcc(values, samplerate, **kwargs):
        assert values.dtype == np.int16 and values[0] == 12000
        assert samplerate == 16000 and kwargs == {"numcep": 13, "winlen": .025, "winstep": .01}
        return np.ones((63, 13))
    monkeypatch.setitem(sys.modules, "python_speech_features", types.SimpleNamespace(mfcc=mfcc))
    video, features = aligned_lrasd_inputs(tmp_path, {"start_ms": 1000, "end_ms": 1640})
    assert len(video) == 16 and features.shape == (64, 13)
    with pytest.raises(ValueError, match="长度不一致"):
        aligned_lrasd_inputs(tmp_path, {"start_ms": 1000, "end_ms": 1680})


def test_probability_uses_both_logits_and_stays_review_candidate(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    metadata = write_bundle(tmp_path)
    class Classifier(torch.nn.Module):
        def forward(self, value):
            return torch.tensor([[0., math.log(3)]]).repeat(len(value), 1)
    model = types.SimpleNamespace(model=lambda audio, visual: (torch.zeros((16, 128)), None),
                                  lossAV=types.SimpleNamespace(FC=Classifier()))
    monkeypatch.setattr("scenerecall.av_sync_worker.load_lrasd_model", lambda *args: model)
    monkeypatch.setattr("scenerecall.av_sync_worker.aligned_lrasd_inputs",
                        lambda *args: (np.zeros((16, 270, 480, 3)), np.zeros((64, 13), dtype=np.float32)))
    monkeypatch.setattr("scenerecall.av_sync_worker.lrasd_visual",
                        lambda *args: np.zeros((16, 112, 112), dtype=np.float32))
    result = lrasd_main(tmp_path, {"checkpoint": str(tmp_path), "preprocessing": LRASD_PREPROCESS,
                                 "repository": str(tmp_path), "device": "cpu", "model_version": "test",
                                 "start_ms": 1000, "end_ms": 1640,
                                 "tracks": [{"character_id": "a", "boxes": []}]})
    assert result["status"] == "review_required"
    assert result["dataset"] == metadata["dataset"]
    row = result["results"][0]
    assert row["association_status"] == "candidate"
    assert row["active_speaker_probability"] == pytest.approx(.75)
    assert row["score_domain"] == "active_speaker_probability" and "synchronization_score" not in row
    assert row["frame_probabilities"][-1]["at_ms"] == 1600


def test_nonfinite_timestamps_and_boxes_are_rejected():
    track = {"association_status": "confirmed", "character_id": "a",
             "boxes": [{"at_ms": index * 40, "box": [0, 0, 1, 1]} for index in range(16)]}
    track["boxes"][5]["at_ms"] = float("nan")
    with pytest.raises(ValueError):
        validate_track(track, 0, 640)
    track["boxes"][5]["at_ms"] = 200
    track["boxes"][5]["box"][0] = float("nan")
    with pytest.raises(ValueError):
        validate_track(track, 0, 640)
