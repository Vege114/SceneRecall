"""Official annotation/timebase conversion checks; no video downloads or training."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest


@pytest.fixture
def prepare():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cmdam_lrasd.py"
    spec = importlib.util.spec_from_file_location("cmdam_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_other_is_literal_and_uncertainty_never_becomes_negative(prepare):
    visible = {"Other Father": [(0, 0, 10, 10)], "Coraline Jones": [(10, 0, 20, 10)]}
    known = [(0., 2., "Other Father")]
    assert prepare.agreed_speaker(1., known, known, visible) == "Other Father"
    assert prepare.agreed_speaker(.04, known, known, visible) is None
    assert prepare.agreed_speaker(3., known, known, visible) is None
    assert prepare.agreed_speaker(1., [(0., 2., "Other")], [(0., 2., "Other")], visible) is None
    assert prepare.agreed_speaker(1., known, [(0., 2., "Coraline Jones")], visible) is None
    assert prepare.agreed_speaker(1., known + [(0., 2., "Coraline Jones")], known, visible) is None
    visible["Other Father"].append((20, 0, 30, 10))
    assert prepare.agreed_speaker(1., known, known, visible) is None
    assert prepare.canonical_character("The Beldam (Other Mother)", "Coraline") == "The Beldam (Other Mother)"


def test_frame_mapping_matches_fps_pts_duplication_without_rewriting_source_clock(prepare):
    # Explicit round=near quantizes source frame 12 to output PTS 13, so
    # output frame 12 repeats source frame 11, rather than guessing source 12.
    mapping = prepare.resampled_indices(list(range(24)), "1/24", 25)
    assert mapping[11:14] == [11, 11, 12]
    assert mapping[-1] == 23 and len(mapping) == 25
    assert mapping == prepare.resampled_indices(list(range(42, 66)), "1/24", 25)


def test_probe_preserves_42ms_video_to_audio_offset_and_requires_audited_prefix(prepare, monkeypatch):
    metadata = {"streams": [{"index": 0, "codec_type": "video", "width": 640, "height": 360,
                  "avg_frame_rate": "25/1", "r_frame_rate": "25/1", "time_base": "1/1000"},
                 {"index": 1, "codec_type": "audio"}], "format": {"start_time": "0.000"}}
    frames = {"frames": [{"best_effort_timestamp": value} for value in (42, 82, 122, 162)]}

    def probe(command, **kwargs):
        return json.dumps(frames if "-show_frames" in command else metadata).encode()

    monkeypatch.setattr(prepare.subprocess, "check_output", probe)
    canvas = {"width": 640, "height": 360}
    result = prepare.probe_video(Path("clip.mkv"), 4, canvas, "ffprobe", 10)
    assert result["video_offset_seconds"] == .042
    assert result["video_first_pts"] == 42 and result["native_pts"] == [42, 82, 122, 162]
    with pytest.raises(ValueError, match="no verified prefix"):
        prepare.probe_video(Path("clip.mkv"), 2, canvas, "ffprobe", 10)
    result = prepare.probe_video(Path("clip.mkv"), 2, {"width": 640, "height": 344}, "ffprobe", 10,
                                 {"crop": [0, 8, 640, 344], "annotation_prefix_frames": 2})
    assert result["source_native_frames"] == 4 and result["native_frames"] == 2
    assert result["source_tail_frames_dropped"] == 2 and result["static_crop"] == [0, 8, 640, 344]
    assert result["native_pts"] == [42, 82]
    with pytest.raises(ValueError, match="dimensions differ"):
        prepare.probe_video(Path("clip.mkv"), 4, {"width": 640, "height": 344}, "ffprobe", 10)


def test_pcm_crop_uses_video_offset_and_only_audited_small_tail_padding(prepare, tmp_path, monkeypatch):
    target = tmp_path / "aligned.wav"
    observed = {}

    def run(command, **kwargs):
        observed["filter"] = command[command.index("-af") + 1]
        with wave.open(str(target), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * (32000 - 8))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare.subprocess, "run", run)
    probe = {"video_offset_fraction": "21/500", "format_origin": "0", "audio_stream_index": 1}
    audit = prepare.extract_aligned_audio(Path("real-local.mkv"), target, probe, 50,
                                           SimpleNamespace(ffmpeg="ffmpeg", timeout=10))
    assert "atrim=start=0.042000000000:end=2.042000000000" in observed["filter"]
    assert audit["tail_difference_samples"] == -8 and audit["tail_adjustment"] == "zero_pad"
    output = tmp_path / "window.wav"
    prepare.window_audio(target, output, 0, 32000)
    with wave.open(str(output), "rb") as wav:
        assert wav.getnframes() == 32000
        assert wav.readframes(32000)[-16:] == b"\0" * 16


def test_native_initial_and_tail_jitter_preserves_exact_frame_identity(prepare, monkeypatch):
    metadata = {"streams": [{"index": 0, "codec_type": "video", "width": 640, "height": 360,
                  "avg_frame_rate": "24000/1001", "r_frame_rate": "24000/1001", "time_base": "1/1000"},
                 {"index": 1, "codec_type": "audio"}], "format": {"start_time": "0"}}
    native = [42, 86, 128, 170, 212, 253, 295, 337, 378, 420, 503]

    def probe(command, **kwargs):
        return json.dumps({"frames": [{"best_effort_timestamp": value} for value in native]}
                          if "-show_frames" in command else metadata).encode()

    monkeypatch.setattr(prepare.subprocess, "check_output", probe)
    result = prepare.probe_video(Path("official.mkv"), len(native), {"width": 640, "height": 360}, "ffprobe", 10)
    assert result["native_pts"] == native
    assert result["pts_audit"]["maximum_interval_seconds"] == .083
    assert result["native_duration_seconds"] == .461
    assert result["guaranteed_canonical_frames"] == 12
    assert prepare.resampled_indices(native, "1/1000", 12)[-2:] == [9, 9]
    native[-1] = native[-2]
    with pytest.raises(ValueError, match="nonmonotonic"):
        prepare.probe_video(Path("official.mkv"), len(native), {"width": 640, "height": 360}, "ffprobe", 10)
    native[-1] = 2000
    with pytest.raises(ValueError, match="implausible"):
        prepare.probe_video(Path("official.mkv"), len(native), {"width": 640, "height": 360}, "ffprobe", 10)


def test_short_real_audio_discards_complete_tail_window_instead_of_fabricating_pcm(prepare, tmp_path, monkeypatch):
    target = tmp_path / "short-real.wav"
    original_frames = 56 * 25
    real_samples = original_frames * 640 - 347

    def run(command, **kwargs):
        with wave.open(str(target), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x01\x00" * real_samples)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare.subprocess, "run", run)
    probe = {"video_offset_fraction": "0", "format_origin": "0", "audio_stream_index": 1}
    audio = prepare.extract_aligned_audio(Path("short-audio.mkv"), target, probe, original_frames,
                                         SimpleNamespace(ffmpeg="ffmpeg", timeout=10))
    assert audio["decoded_samples"] == real_samples
    assert audio["tail_adjustment"] == "drop_complete_tail_windows"
    common = prepare.common_av_window_audit(original_frames, real_samples)
    assert common["retained_canonical_frames"] == original_frames - 50
    assert common["discarded_video_frames"] == 50
    assert common["discarded_video_seconds"] == 2
    assert common["retained_audio_rounding_pad_samples"] == 0
    assert common["unused_decoded_audio_tail_samples"] == 32000 - 347
    assert prepare.common_av_window_audit(50, 0)["retained_canonical_frames"] == 0
    assert prepare.common_av_window_audit(50, 31967)["retained_canonical_frames"] == 0
    rounding = prepare.common_av_window_audit(50, 31968)
    assert rounding["retained_canonical_frames"] == 50
    assert rounding["retained_audio_rounding_pad_samples"] == 32
    assert prepare.common_av_window_audit(50, 64000)["retained_canonical_frames"] == 50
    with pytest.raises(ValueError, match="nonnegative"):
        prepare.common_av_window_audit(50, -1)


def test_masked_other_tail_is_audited_but_named_speaker_tail_is_rejected(prepare):
    probe = {"video_offset_seconds": 0., "native_duration_seconds": 153.9871667, "fps": "24000/1001"}
    ap = [(140., 145.668, "Ash")]
    der = [(80.978, 154.09714, "Other"), (80., 156., "Minion")]
    audit = prepare.speaker_extent_audit(probe, ap, der)
    ignored = audit["ignored_nontraining_annotation_extent"]
    assert audit["maximum_named_speaker_end_seconds"] == 145.668
    assert ignored["maximum_end_seconds"] == 156.
    assert ignored["seconds_beyond_prefix"] == pytest.approx(2.0128333)
    assert [row["character"] for row in ignored["overflow_segments"]] == ["Other", "Minion"]
    with pytest.raises(ValueError, match="named-speaker"):
        prepare.speaker_extent_audit(probe, ap, [(80., 154.09714, "Ash")])
    # Coraline's Other Father is a named individual, not the literal open set.
    with pytest.raises(ValueError, match="named-speaker"):
        prepare.speaker_extent_audit(probe, ap, [(80., 154.09714, "Other Father")])


def test_exact_video_id_rejects_multiple_formats_and_missing_files(prepare, tmp_path):
    clip = "8dXF7y1QUxU"
    with pytest.raises(ValueError, match="found 0"):
        prepare.locate_video(tmp_path, clip)
    (tmp_path / (clip + ".mkv")).write_bytes(b"local")
    assert prepare.locate_video(tmp_path, clip).suffix == ".mkv"
    (tmp_path / (clip + ".mp4")).write_bytes(b"local")
    with pytest.raises(ValueError, match="found 2"):
        prepare.locate_video(tmp_path, clip)


def test_official_all_movie_holdout_is_fixed_not_replaced_for_missing_test(prepare):
    movies = ["Kung Fu Panda", "Coraline", "Despicable Me", "Ice Age", "The Polar Express", "Sing",
              "Spider Man: Into the Spider-Verse", "The Secret Life of Pets"]
    plan = prepare.movie_split_plan(movies)
    assert plan["The Secret Life of Pets"] == "test"
    assert plan["Spider Man: Into the Spider-Verse"] == "val"
    assert list(plan.values()).count("train") == 6
    with pytest.raises(ValueError, match="eight movies"):
        prepare.movie_split_plan(movies[:-1])


def test_alignment_override_requires_bound_source_and_explicit_review(prepare, tmp_path):
    clip = "8dXF7y1QUxU"
    path = tmp_path / "overrides.json"
    value = {"schema_version": "cmdam-alignment-overrides-v1", "clips": {clip: {
        "verified": True, "source_sha256": "a" * 64, "evidence": ["frame-overlay-review.json"],
        "crop": [0, 8, 640, 344], "annotation_prefix_frames": 2950}}}
    path.write_text(json.dumps(value), encoding="utf-8")
    assert prepare.alignment_overrides(path, [{"clip_id": clip}])[clip]["annotation_prefix_frames"] == 2950
    value["clips"][clip]["time_offset_seconds"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot change FPS"):
        prepare.alignment_overrides(path, [{"clip_id": clip}])
