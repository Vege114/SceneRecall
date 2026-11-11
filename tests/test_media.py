"""All media are generated during testing. Existing user films are never accessed."""
import hashlib
import shutil
import subprocess

import pytest
from PIL import Image
from scenerecall import media
from scenerecall.media import (
    MediaError,
    create_proxy,
    detect_shots,
    extract_frame,
    extract_frames,
    fingerprint,
    make_windows,
    probe,
    sample_times,
)


@pytest.fixture(scope="module")
def movie(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for synthetic-media integration tests")
    directory = tmp_path_factory.mktemp("synthetic-media")
    path = directory / "two colors ; $(nothing).mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x96:r=10:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=160x96:r=10:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    return path


def test_probe_and_fingerprint_only_local_source(movie):
    metadata = probe(movie)
    assert metadata["duration_ms"] == 2000
    assert (metadata["width"], metadata["height"]) == (160, 96)
    assert metadata["video_codec"] == "h264"
    assert metadata["audio_codec"] is None
    assert fingerprint(movie) == hashlib.sha256(movie.read_bytes()).hexdigest()


def test_shots_cover_requested_interval_and_windows_never_cross_cuts(movie):
    shots = detect_shots(movie, 0, 2000)
    assert [(s["start_ms"], s["end_ms"]) for s in shots] == [(0, 1000), (1000, 2000)]
    partial = detect_shots(movie, 500, 1500)
    assert [(s["start_ms"], s["end_ms"]) for s in partial] == [(500, 1000), (1000, 1500)]
    windows = make_windows(shots, max_ms=700)
    assert [(w["start_ms"], w["end_ms"]) for w in windows] == [(0, 700), (700, 1000), (1000, 1700), (1700, 2000)]
    assert all(w["start_ms"] <= t < w["end_ms"] for w in windows for t in sample_times(w["start_ms"], w["end_ms"]))


def test_extract_uses_decoded_timestamp_crop_and_literal_paths(movie, tmp_path):
    output = tmp_path / "frame;name.jpg"
    frame = extract_frame(movie, 1155, output, crop=[0, .5, 1, .5])
    assert frame["id"] == "frame;name"
    assert frame["requested_at_ms"] == 1155
    assert frame["at_ms"] == 1200
    with Image.open(output) as image:
        assert image.size == (160, 48)
        red, _green, blue = image.convert("RGB").getpixel((30, 20))
        assert blue > 200 and red < 30
    assert not list(tmp_path.glob(".media-*"))


def test_proxy_preserves_duration_and_content(movie, tmp_path):
    output = tmp_path / "proxy.mp4"
    create_proxy(movie, output)
    assert abs(probe(output)["duration_ms"] - probe(movie)["duration_ms"]) <= 100
    frame = extract_frame(output, 1200, tmp_path / "proxy-frame.jpg")
    assert frame["at_ms"] == 1200


def test_batch_extract_preserves_each_requested_id_and_actual_time(movie, tmp_path):
    times = [155, 500, 1155, 1190, 1700]
    outputs = [tmp_path / f"frame-{at}.jpg" for at in times]
    frames = extract_frames(movie, times, outputs, crop=[0, .5, 1, .5])
    assert [f["at_ms"] for f in frames] == [200, 500, 1200, 1200, 1700]
    assert [f["requested_at_ms"] for f in frames] == times
    assert [f["id"] for f in frames] == [p.stem for p in outputs]
    assert all(p.is_file() for p in outputs)
    assert outputs[2].read_bytes() == outputs[3].read_bytes()


def test_invalid_media_inputs_and_windows_fail_clearly(movie, tmp_path):
    with pytest.raises(MediaError):
        probe(tmp_path / "missing.mp4")
    with pytest.raises(MediaError):
        detect_shots(movie, 0, 2500)
    with pytest.raises(MediaError):
        extract_frame(movie, 2500, tmp_path / "bad.jpg")
    with pytest.raises(MediaError):
        extract_frame(movie, 0, tmp_path / "bad.jpg", [0, .8, 1, .4])
    with pytest.raises(MediaError):
        make_windows([{"id": "1", "start_ms": 0, "end_ms": 10},
                      {"id": "2", "start_ms": 9, "end_ms": 20}])
    assert sample_times(0, 1) == [0]


def test_variable_frame_rate_cuts_use_presentation_time(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg missing")
    # 10 fps for the first second, 5 fps for the second; the second cut is at 2 s,
    # not at frame 15 / 10 = 1.5 s. Synthetic colors create unambiguous cuts.
    path = tmp_path / "vfr.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x96:r=10:d=1",
        "-f", "lavfi", "-i", "color=c=blue:s=160x96:r=5:d=1",
        "-f", "lavfi", "-i", "color=c=green:s=160x96:r=10:d=1",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]", "-map", "[out]",
        "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True, capture_output=True)
    assert probe(path)["duration_ms"] == 3000
    shots = detect_shots(path, 0, 3000)
    assert [s["start_ms"] for s in shots] == [0, 1000, 2000]
    frame = extract_frame(path, 1155, tmp_path / "vfr.jpg")
    assert frame["at_ms"] == 1200


def test_nonzero_container_origin_is_not_added_to_playback_time(movie, tmp_path):
    path = tmp_path / "offset.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(movie), "-c", "copy",
                    "-output_ts_offset", "5", str(path)], check=True, capture_output=True)
    info = probe(path)
    assert info["start_time_ms"] == 5000
    assert info["duration_ms"] == 2000
    shots = detect_shots(path, 0, 2000)
    assert [s["start_ms"] for s in shots] == [0, 1000]
    assert extract_frame(path, 1155, tmp_path / "offset.jpg")["at_ms"] == 1200


def test_empty_last_second_probe_retries_earlier_for_a_real_presented_frame(movie, tmp_path, monkeypatch):
    metadata = probe(movie)
    original_run = media._run
    intervals = []

    def run(args, **options):
        if "-read_intervals" in args:
            intervals.append(args[args.index("-read_intervals") + 1])
            if len(intervals) == 1:
                return subprocess.CompletedProcess(args, 0, b'{"frames":[]}', b"")
        return original_run(args, **options)

    monkeypatch.setattr(media, "_run", run)
    output = tmp_path / "last.jpg"
    at = media._last_visible_frame(movie, output, None, metadata)
    assert len(intervals) == 2 and intervals[0] != intervals[1]
    assert 0 < at < metadata["duration_ms"]
    assert output.read_bytes().startswith(b"\xff\xd8")


@pytest.mark.parametrize("offset", [0, 5])
def test_low_fps_final_held_frame_keeps_source_pts_and_rejects_stream_end(tmp_path, offset):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg missing")
    path = tmp_path / ("low-fps.mkv" if offset else "low-fps.mp4")
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=red:s=160x96:r=1:d=2",
        "-c:v", "libx264", "-output_ts_offset", str(offset), str(path),
    ], check=True, capture_output=True)
    assert probe(path)["duration_ms"] == 2000
    single = extract_frame(path, 1500, tmp_path / "single.jpg", crop=[0, .5, 1, .5])
    assert single["at_ms"] == 1000 and single["requested_at_ms"] == 1500
    with Image.open(single["path"]) as image:
        assert image.size == (160, 48)
    for times in ([0, 500, 1000, 1500, 1999], [1500, 1999]):
        records = extract_frames(path, times, [tmp_path / f"batch-{at}.jpg" for at in times])
        assert [frame["at_ms"] for frame in records] == [0 if at == 0 else 1000 for at in times]
        assert [frame["requested_at_ms"] for frame in records] == times
    for at in (2000, 2500):
        with pytest.raises(MediaError):
            extract_frame(path, at, tmp_path / "invalid.jpg")
        with pytest.raises(MediaError):
            extract_frames(path, [0, at], [tmp_path / "invalid-first.jpg", tmp_path / "invalid-last.jpg"])
    assert not (tmp_path / "invalid-first.jpg").exists()
    if offset:
        proxy = tmp_path / "normalized-proxy.mp4"
        create_proxy(path, proxy)
        assert probe(proxy)["duration_ms"] == 2000
        assert probe(proxy)["start_time_ms"] == 0


def test_fractional_final_frame_is_not_lost_by_rounding_seek(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg missing")
    path = tmp_path / "fractional.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=160x96:r=30:d=2",
        "-c:v", "libx264", str(path),
    ], check=True, capture_output=True)
    # Last encoded PTS is 1.966666..., held until the 2-second media end.
    frame = extract_frame(path, 1999, tmp_path / "last.jpg")
    assert frame["at_ms"] == 1967
    assert frame["requested_at_ms"] == 1999
