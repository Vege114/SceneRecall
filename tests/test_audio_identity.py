"""Audio identity contract tests. Injected embeddings are NOT recognition benchmarks."""
import base64
import copy
import math
import shutil
import subprocess
import threading
import wave
from array import array

import pytest

from scenerecall.audio_adapters import AudioUnavailable, validate_pyannote_config
from scenerecall.audio_identity import AudioIdentityService, match_voice, validate_turns
from scenerecall.audio_media import audio_tracks, crop_pcm, extract_audio, pcm_samples
from scenerecall.av_sync import sync_availability, validate_track
from scenerecall.library import atomic_json, read_jsonl


def wav_file(path, duration_ms=4000, frequency=220):
    values = array("h", (int(7000 * math.sin(2 * math.pi * frequency * n / 16000))
                         for n in range(duration_ms * 16)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(values.tobytes())
    return path


class Library:
    def __init__(self, root, source):
        self.root, self.source, self.lock = root, source, threading.RLock()
        self.cues = [{"id": "cn", "start_ms": 0, "end_ms": 4000, "text": "这是中文译文", "language": "zh"}]

    def get_asset(self, identity):
        return {"id": identity, "fingerprint": identity, "source_available": True, "duration_ms": 4000}

    def asset_dir(self, identity):
        return self.root / "assets" / identity

    def require_source(self, identity):
        return self.source

    def subtitles(self, identity):
        return self.cues

    def active(self, identity):
        return {}


class Characters:
    def __init__(self):
        self.name = "甲"

    def catalog(self, identity):
        return {"scope_id": "work", "profiles": [{"id": "char_a", "name": self.name}, {"id": "char_b", "name": "乙"}],
                "assets": [{"id": identity}]}


class Diarizer:
    version = "test-diarizer-v1"

    def __init__(self, overlap=False):
        self.calls, self.overlap = 0, overlap

    def diarize(self, path):
        self.calls += 1
        return [{"start_ms": 0, "end_ms": 2500 if self.overlap else 2000, "speaker": "S0"},
                {"start_ms": 2000, "end_ms": 4000, "speaker": "S1"}]


class Embedder:
    version = "test-ecapa-contract-v1"

    def __init__(self):
        self.calls = 0

    def embed(self, path):
        self.calls += 1
        return [1., 0.]


@pytest.fixture
def service(tmp_path, monkeypatch):
    source = wav_file(tmp_path / "source.wav")
    library = Library(tmp_path / "library", source)
    instance = AudioIdentityService(library, Characters(), diarizer=Diarizer(), embedder=Embedder())
    monkeypatch.setattr("scenerecall.audio_identity.media.probe", lambda _: {
        "duration_ms": 4000, "start_time_ms": 0, "streams": [{"codec_type": "audio", "index": 1}]})

    def extract(source, destination, stream, start, end, metadata, checkpoint=None):
        crop_pcm(source, destination, start, end)

    monkeypatch.setattr("scenerecall.audio_identity.extract_audio", extract)
    instance.configure("asset", {"enabled": True, "language": "ja", "dub_version": "japanese-tv"})
    return instance


def add_external(service, *, character="char_a", dub="japanese-tv", confirmed=True):
    return service.add_sample("asset", character, {"source_kind": "external_licensed", "rights_note": "synthetic fixture",
                               "audio_base64": base64.b64encode(service.library.source.read_bytes()).decode(),
                               "confirmed": confirmed, "language": "ja", "dub_version": dub})


def test_no_reference_bank_keeps_unknown_and_offscreen_audio_is_not_visual(service):
    result = service.run("asset", frames=[{"id": "f1", "at_ms": 1000}])
    assert result["status"] == "completed"
    assert len(result["records"]) == 2
    for row in result["records"]:
        assert row["association_status"] == "unknown"
        assert row["character_id"] is None
        assert row["evidence_frame_ids"] == []
        assert row["visual_evidence"] is False
        assert row["subtitle_association"] == "time_overlap_only"
        assert row["phonetic_alignment"] is False
        assert row["dialogue_attribution"] == "ambiguous"
        assert row["text"] == "这是中文译文"


def test_unknown_overlap_and_speaker_ids_scoped_to_asset(service):
    service._diarizer.overlap = True
    first = service.run("asset")
    service.configure("another", {"enabled": True, "language": "ja", "dub_version": "japanese-tv"})
    second = service.run("another")
    assert all(row["overlap"] for row in first["records"])
    assert first["records"][0]["speaker_id"] != second["records"][0]["speaker_id"]
    with pytest.raises(ValueError, match="重叠"):
        service.add_sample("asset", "char_a", {"record_id": first["records"][0]["id"], "confirmed": True})


def test_trusted_bank_dub_isolation_and_revoke_without_reencoding(service):
    wrong = add_external(service, dub="english-redub")
    proposed = add_external(service, confirmed=False)
    assert all(row["association_status"] == "unknown" for row in service.run("asset")["records"])
    service.update_sample("asset", "char_a", proposed["id"], "confirmed")
    rows = service.run("asset")["records"]
    assert all(row["association_status"] == "candidate" for row in rows)
    assert all(row["candidate_character_id"] == "char_a" for row in rows)
    calls = service._embedder.calls
    service.update_sample("asset", "char_a", proposed["id"], "revoked")
    assert all(row["association_status"] == "unknown" for row in service.records("asset"))
    assert service._embedder.calls == calls
    assert wrong["id"] != proposed["id"]
    assert len(read_jsonl(service.library.root / "works/work/voice-audit.jsonl")) == 4


def test_manual_correction_and_rename_are_projection_only(service):
    rows = service.run("asset")["records"]
    original = copy.deepcopy(service._raw_records("asset"))
    service.confirm("asset", rows[0]["id"], "char_b", "核对原音确认")
    service.characters.name = "修改后的甲"
    current = service.confirm("asset", rows[1]["id"], "char_a")
    assert current["character_name"] == "修改后的甲"
    assert current["association_source"] == "human"
    assert service._raw_records("asset") == original
    assert service.confirm("asset", rows[0]["id"], None)["character_id"] is None
    assert len(read_jsonl(service._directory("asset") / "identity-audit.jsonl")) == 3


def test_resume_does_not_repeat_diarization_or_feature_calls(service):
    first = service.run("asset")
    counts = service._diarizer.calls, service._embedder.calls
    second = service.run("asset")
    assert counts == (service._diarizer.calls, service._embedder.calls)
    assert second["cache_hits"] >= 4
    assert [row["id"] for row in first["records"]] == [row["id"] for row in second["records"]]
    assert "total_s" in second["timings"] and "diarization_s" in first["timings"]


def test_cancelled_run_resumes_saved_acoustic_features(service):
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if service._embedder.calls:
            raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        service.run("asset", checkpoint=cancel)
    assert service.records("asset") == []
    service.run("asset")
    assert service._diarizer.calls == 1
    # Identical synthetic segments share one content-addressed embedding.
    assert service._embedder.calls == 1


def test_no_audio_and_missing_dependency_are_explicit(service, monkeypatch):
    monkeypatch.setattr("scenerecall.audio_identity.media.probe", lambda _: {"duration_ms": 4000, "streams": []})
    assert service.run("asset")["status"] == "no_audio"
    assert service._diarizer.calls == 0


def test_unavailable_diarizer_does_not_claim_completion(service):
    def missing(_):
        raise AudioUnavailable("未配置模型")
    service._diarizer.diarize = missing
    result = service.run("asset")
    assert result["status"] == "unavailable"
    assert result["degraded"] == ["未配置模型"]


def test_local_budget_does_not_fail_legacy_visual_work_and_disabled_does_no_work(service):
    service.configure("asset", {"max_duration_ms": 1000})
    result = service.run("asset")
    assert result["status"] == "budget_exceeded"
    assert service._diarizer.calls == 0
    service.configure("asset", {"enabled": False})
    assert service.run("asset")["status"] == "disabled"
    assert service._diarizer.calls == 0


def test_failed_optional_embedding_can_be_retried_and_old_result_is_audited(service):
    original_embed = service._embedder.embed

    def unavailable(_):
        raise AudioUnavailable("model temporarily unavailable")

    service._embedder.embed = unavailable
    result = service.run("asset")
    assert result["status"] == "partial"
    assert all(row["embedding"] is None for row in service._raw_records("asset"))
    service._embedder.embed = original_embed
    result = service.run("asset")
    assert result["status"] == "completed"
    assert all(row["embedding"] for row in service._raw_records("asset"))
    assert list((service._directory("asset") / "runs" / result["run_id"] / "history").glob("*.jsonl"))


def test_changed_dub_or_subtitle_does_not_overwrite_old_evidence_or_reextract_features(service):
    first = service.run("asset")
    calls = service._diarizer.calls, service._embedder.calls
    service.configure("asset", {"dub_version": "different-dub"})
    service.library.cues[0]["text"] = "人工修订译文"
    second = service.run("asset")
    assert first["run_id"] != second["run_id"]
    assert calls == (service._diarizer.calls, service._embedder.calls)
    original = read_jsonl(service._directory("asset") / "runs" / first["run_id"] / "records.jsonl")
    assert original[0]["text"] == "这是中文译文"
    assert second["records"][0]["text"] == "人工修订译文"


def test_runtime_budget_and_subtitle_changes_preserve_human_identity(service):
    row = service.run("asset")["records"][0]
    service.confirm("asset", row["id"], "char_b")
    service.configure("asset", {"max_duration_ms": 120000, "top_k": 3, "sync_enabled": True})
    service.library.cues[0]["text"] = "更正字幕"
    updated = service.run("asset")["records"][0]
    assert updated["id"] == row["id"]
    assert updated["character_id"] == "char_b" and updated["association_source"] == "human"


def test_calibration_is_snapshotted_and_read_does_not_follow_external_paths(service, tmp_path):
    add_external(service)
    add_external(service, character="char_b")
    path = tmp_path / "calibration.json"
    atomic_json(path, {"id": "calibration-v1", "evaluation_set_id": "fixture", "model_version": service._embedder.version,
                      "scope_id": "work", "language": "ja", "dub_version": "japanese-tv",
                      "candidate_max_distance": .3, "confirmed_max_distance": .1, "min_margin": .2})
    service.configure("asset", {"calibration_path": str(path)})
    result = service.run("asset")
    path.unlink()
    counts = service._embedder.calls, service._diarizer.calls
    assert [row["association_status"] for row in service.records("asset")] == [row["association_status"] for row in result["records"]]
    assert counts == (service._embedder.calls, service._diarizer.calls)


def test_partial_reanalysis_keeps_cross_boundary_evidence_and_exact_audio_fragments(service):
    original = service.run("asset")["records"]
    service.confirm("asset", original[0]["id"], "char_a")
    service._diarizer.diarize = lambda path: [{"start_ms": 0, "end_ms": 2000, "speaker": "S0"}]
    service.run("asset", 1000, 3000)
    rows = service.records("asset")
    assert [(row["start_ms"], row["end_ms"]) for row in rows] == [(0, 1000), (1000, 3000), (3000, 4000)]
    assert rows[0]["source_record_id"] == original[0]["id"]
    assert rows[0]["character_id"] == "char_a"
    assert len({row["id"] for row in rows}) == 3
    for row in (rows[0], rows[-1]):
        assert len(pcm_samples(service.clip_path("asset", row["id"]))) == 16000


def test_asr_only_without_reliable_subtitle_and_cached(service):
    class ASR:
        version = "test-asr"
        calls = 0

        def transcribe(self, path, language):
            self.calls += 1
            return [{"start_ms": 0, "end_ms": 1000, "text": "hello", "language": language}]

    service._asr = ASR()
    service.configure("asset", {"asr_enabled": True})
    service.run("asset")
    assert service._asr.calls == 0
    service.library.cues = []
    assert service.run("asset")["records"][0]["text"] == "hello"
    service.run("asset")
    assert service._asr.calls == 1


def test_calibration_not_raw_similarity_controls_confirmation_and_overlap_blocks_it():
    samples = [{"id": "v1", "character_id": "a", "embedding": [1, 0]},
               {"id": "v2", "character_id": "b", "embedding": [0, 1]}]
    policy = {"id": "heldout-v1", "candidate_max_distance": .3, "confirmed_max_distance": .1, "min_margin": .2}
    assert match_voice([1, 0], samples)["association_status"] == "candidate"
    matched = match_voice([1, 0], samples, calibration=policy)
    assert matched["association_status"] == "confirmed" and matched["distance"] == 0
    assert "match_confidence" not in matched
    assert match_voice([1, 0], samples, calibration=policy, blocked=True)["association_status"] == "candidate"
    assert match_voice([-1, 0], samples, calibration=policy)["association_status"] == "unknown"


def test_invalid_calibration_and_unknown_speakers(service, tmp_path):
    path = tmp_path / "calibration.json"
    atomic_json(path, {"id": "bad"})
    service.configure("asset", {"calibration_path": str(path)})
    with pytest.raises(ValueError, match="评测集"):
        service.run("asset")
    assert validate_turns([{"start_ms": 0, "end_ms": 1000, "speaker": None}], 1000)[0]["speaker"].startswith("unknown_")
    with pytest.raises(ValueError, match="越界"):
        validate_turns([{"start_ms": 0, "end_ms": 1002, "speaker": "s"}], 1000)


def test_sync_missing_weights_and_sparse_keyframes_are_rejected():
    assert not sync_availability({"sync_enabled": True})["available"]
    with pytest.raises(ValueError, match="连续"):
        validate_track({"association_status": "confirmed", "character_id": "a",
                        "boxes": [{"at_ms": n * 1000, "box": [0, 0, 1, 1]} for n in range(15)]}, 0, 15000)
    with pytest.raises(ValueError, match="视觉身份"):
        validate_track({"association_status": "candidate", "character_id": "a"}, 0, 1000)


@pytest.mark.parametrize("reference", [None, 7, {}, {"checkpoint": None}, "pyannote/segmentation", "https://remote/model"])
def test_pyannote_cannot_fall_back_to_default_or_remote_models(tmp_path, reference):
    with pytest.raises(AudioUnavailable):
        validate_pyannote_config(tmp_path, {"pipeline": {"name": "pyannote.audio.pipelines.SpeakerDiarization",
                                                         "params": {"segmentation": reference}}})


def test_pyannote_accepts_only_complete_local_snapshot(tmp_path):
    params = {}
    for name in ("segmentation", "embedding", "plda"):
        (tmp_path / name).mkdir()
        params[name] = "$model/" + name
    validate_pyannote_config(tmp_path, {"pipeline": {"name": "pyannote.audio.pipelines.SpeakerDiarization", "params": params}})


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_real_ffmpeg_track_delay_clip_offset_and_resampling(tmp_path):
    from scenerecall.media import probe
    source = tmp_path / "offset.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=s=64x64:r=25:d=3",
                    "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=600:sample_rate=48000:duration=2",
                    "-map", "0:v", "-map", "1:a", "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)], check=True)
    metadata = probe(source)
    tracks = audio_tracks(metadata)
    assert tracks[0]["start_ms"] == 500
    output = tmp_path / "selected.wav"
    extract_audio(source, output, tracks[0]["index"], 250, 2000, metadata)
    samples = pcm_samples(output)
    assert len(samples) == 1750 * 16
    assert max(abs(x) for x in samples[:3500]) == 0  # original 250..469 ms is still silent
    assert max(abs(x) for x in samples[5000:]) > 1000
    with pytest.raises(ValueError, match="音轨"):
        extract_audio(source, output, 999, 0, 1000, metadata)
