"""Backend comparisons use generated media, never a user's video library."""
from fractions import Fraction
from itertools import count
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter
import pytest

from scenerecall import keyframes, media
from test_keyframes import assert_plan, picture, video


@pytest.fixture(autouse=True)
def require_media_tools():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for generated backend fixtures")


def make_video(path: Path, *, offset=0, audio_first=False):
    images = []
    for index in range(72):
        image = np.zeros((72, 128, 3), dtype=np.uint8)
        image[:] = (20, 30, 45)
        image[8:62, 8 + index % 72:40 + index % 72] = (240, 120, 50)
        image[8:18, 8:120] = (20, 230, 100)
        images.append(image)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    if audio_first:
        command += ["-f", "lavfi", "-i", "sine=frequency=400:duration=3"]
    command += ["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "128x72",
                "-framerate", "24", "-i", "pipe:0"]
    if audio_first:
        command += ["-map", "0:a", "-map", "1:v", "-c:a", "aac"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "12", "-bf", "2",
                "-output_ts_offset", str(offset), str(path)]
    subprocess.run(command, input=b"".join(image.tobytes() for image in images),
                   capture_output=True, check=True)
    return path


def decoded(path, metadata, backend, start=0, end=None):
    decode = keyframes._decode if backend == "ffmpeg" else keyframes._decode_pyav
    return list(decode(path, start, end or metadata["duration_ms"], metadata, None))


def test_default_checksum_optimization_preserves_exact_plan(tmp_path, monkeypatch):
    path = make_video(tmp_path / "default.mp4")
    metadata = media.probe(path)
    optimized = keyframes.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata)
    real_popen = keyframes.subprocess.Popen
    commands = []

    def checksummed(command, *args, **kwargs):
        commands.append(command)
        command = [part.replace("showinfo=checksum=0", "showinfo=checksum=1") for part in command]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(keyframes.subprocess, "Popen", checksummed)
    old_checksums = keyframes.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata)
    assert old_checksums == optimized
    assert any("showinfo=checksum=0" in part for command in commands for part in command)


def test_probe_metadata_can_be_reused_by_scan_and_exact_extraction(tmp_path, monkeypatch):
    path = make_video(tmp_path / "reuse.mp4")
    metadata = media.probe(path)

    def redundant_probe(_path):
        pytest.fail("a current metadata snapshot should avoid a second ffprobe")

    monkeypatch.setattr(media, "probe", redundant_probe)
    plan = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata)
    targets = [tmp_path / f"frame-{index}.jpg" for index in range(len(plan["frames"]))]
    result = media.extract_keyframes(path, plan["frames"], targets, metadata=metadata)
    assert [item["source_pts"] for item in result] == [item["source_pts"] for item in plan["frames"]]
    assert all(path.is_file() for path in targets)


def test_reused_metadata_rejects_wrong_or_changed_source(tmp_path):
    path = make_video(tmp_path / "first.mp4")
    metadata = media.probe(path)
    other = tmp_path / "second.mp4"
    shutil.copy2(path, other)
    with pytest.raises(media.MediaError, match="元数据"):
        media.scan_keyframes(other, 0, 1000, metadata=metadata)
    with path.open("ab") as output:
        output.write(b"changed")
    with pytest.raises(media.MediaError, match="元数据"):
        media.scan_keyframes(path, 0, 1000, metadata=metadata)
    assert "path" not in metadata["_source_identity"]


def test_unknown_decoder_is_rejected(tmp_path):
    path = make_video(tmp_path / "unknown.mp4")
    with pytest.raises(media.MediaError, match="解码后端"):
        media.scan_keyframes(path, 0, 1000, decoder="typo")


def test_pyav_is_optional_and_never_an_implicit_fallback(tmp_path, monkeypatch):
    path = make_video(tmp_path / "optional.mp4")
    metadata = media.probe(path)
    monkeypatch.setitem(sys.modules, "av", None)
    assert media.scan_keyframes(path, 0, 300, metadata=metadata)["frames"]
    with pytest.raises(media.MediaError, match="未安装"):
        media.scan_keyframes(path, 0, 300, metadata=metadata, decoder="pyav")


@pytest.mark.parametrize("offset,audio_first", [(0, False), (5, True)])
def test_pyav_preserves_native_pts_stream_selection_and_partial_seek(tmp_path, offset, audio_first):
    pytest.importorskip("av")
    path = make_video(tmp_path / "native.mp4", offset=offset, audio_first=audio_first)
    metadata = media.probe(path)
    if audio_first:
        assert metadata["video_stream_index"] == 1
    expected = decoded(path, metadata, "ffmpeg", start=2200, end=2800)
    actual = decoded(path, metadata, "pyav", start=2200, end=2800)
    assert [item[1:] for item in actual] == [item[1:] for item in expected]
    assert actual
    for left, right in zip(expected, actual, strict=True):
        assert right[0].shape == (180, 320, 3)
        # FFmpeg CLI and PyAV may bundle different swscale builds. Timing must
        # match exactly; small scaling-rounding differences are not hidden.
        assert np.abs(left[0].astype(float) - right[0]).mean() < 2
    plan = media.scan_keyframes(path, 2200, 2800, metadata=metadata, decoder="pyav")
    assert plan == media.scan_keyframes(path, 2200, 2800, metadata=metadata, decoder="ffmpeg")
    assert plan["frames"]
    assert all(2200 <= frame["at_ms"] < 2800 for frame in plan["frames"])


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_pyav_uses_the_same_display_rotation_and_letterboxing(tmp_path, rotation):
    pytest.importorskip("av")
    source = make_video(tmp_path / "source.mp4")
    path = tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-map", "0", "-c", "copy", "-metadata:s:v:0", f"rotate={rotation}", str(path)],
                   capture_output=True, check=True)
    metadata = media.probe(path)
    expected = decoded(path, metadata, "ffmpeg", end=300)
    actual = decoded(path, metadata, "pyav", end=300)
    assert [item[1:] for item in actual] == [item[1:] for item in expected]
    assert actual
    for left, right in zip(expected, actual, strict=True):
        assert np.abs(left[0].astype(float) - right[0]).mean() < 2


def test_pyav_vfr_with_offset_retains_exact_source_times(tmp_path):
    pytest.importorskip("av")
    path = tmp_path / "variable.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "color=red:s=128x72:r=10:d=1", "-f", "lavfi", "-i", "color=blue:s=128x72:r=5:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
        "-fps_mode", "vfr", "-c:v", "libx264", "-output_ts_offset", "5", str(path),
    ], capture_output=True, check=True)
    metadata = media.probe(path)
    expected = decoded(path, metadata, "ffmpeg")
    actual = decoded(path, metadata, "pyav")
    assert [item[1:] for item in actual] == [item[1:] for item in expected]
    steps = {right[3] - left[3] for left, right in zip(actual, actual[1:])}
    assert Fraction(1, 10) in steps and Fraction(1, 5) in steps
    expected_plan = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="ffmpeg")
    actual_plan = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="pyav")
    assert actual_plan == expected_plan
    assert [frame["at_ms"] for frame in actual_plan["frames"]] == [0, 1000]


@pytest.mark.parametrize("scene", ["short_shots", "held_short", "held_long", "blur_transition", "motion"])
def test_pyav_content_scenes_match_the_complete_ffmpeg_plan(tmp_path, scene):
    """Use the original regression suite's pictures and lossless video fixture."""
    pytest.importorskip("av")
    if scene == "short_shots":
        colors = [Image.new("RGB", (128, 72), color) for color in ("red", "blue", "green")]
        pictures = [color for color in colors for _ in range(2)]
    elif scene.startswith("held_"):
        pictures = [picture()] * (24 if scene == "held_short" else 240)
    elif scene == "blur_transition":
        first, second = picture(7), picture(77)
        transition = [Image.blend(first, second, alpha).filter(ImageFilter.GaussianBlur(3))
                      for alpha in (.2, .4, .6, .8)]
        pictures = [first] * 12 + transition + [second] * 12 + transition[::-1] + [first] * 12
    else:
        pictures = [picture(at, color=(230, 120, 50)) for at in list(range(0, 91, 3)) * 3]
    path = video(tmp_path / f"{scene}.mp4", pictures)
    metadata = media.probe(path)
    expected = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="ffmpeg")
    actual = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="pyav")
    # This includes every shot boundary, selected source PTS/time base, reason,
    # ID and batch boundary. Similar frame counts alone are not acceptance.
    assert actual == expected
    assert_plan(actual)
    if scene == "short_shots":
        assert len(actual["shots"]) == len(actual["frames"]) == 3
        assert [shot["start_ms"] for shot in actual["shots"]] == [0, 83, 167]
    elif scene.startswith("held_"):
        assert len(actual["frames"]) == 1
        assert actual["frames"][0]["reason"] == "stable_pose"
    elif scene == "blur_transition":
        times = [frame["at_ms"] for frame in actual["frames"]]
        assert any(at < 500 for at in times)
        assert any(667 <= at < 1167 for at in times)
        assert any(at >= 1333 for at in times)
        assert sum(500 <= at < 667 or 1167 <= at < 1333 for at in times) < 4
    else:
        assert 2 < len(actual["frames"]) < len(pictures)
        assert any(frame["reason"] == "content_change" for frame in actual["frames"])


@pytest.mark.parametrize("offset", [None, 5])
def test_pyav_selected_fractional_pts_round_trip_through_exact_ffmpeg_extraction(tmp_path, offset):
    pytest.importorskip("av")
    colors = [Image.new("RGB", (128, 72), color) for color in ("red", "blue", "green", "yellow")]
    pictures = [color for color in colors for _ in range(2)]
    path = video(tmp_path / "fractional-round-trip.mp4", pictures, rate="30000/1001", offset=offset)
    metadata = media.probe(path)
    expected = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="ffmpeg")
    actual = media.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="pyav")
    assert actual == expected
    assert_plan(actual)
    assert len(actual["frames"]) == 4
    assert any(frame["pts"] * Fraction(*frame["time_base"]) * 1000 != frame["at_ms"]
               for frame in actual["frames"])
    outputs = [tmp_path / f"pyav-{index}.jpg" for index in range(len(actual["frames"]))]
    references = [tmp_path / f"ffmpeg-{index}.jpg" for index in range(len(expected["frames"]))]
    extracted = media.extract_keyframes(path, actual["frames"], outputs, metadata=metadata)
    media.extract_keyframes(path, expected["frames"], references, metadata=metadata)
    for selected, result, output, reference in zip(actual["frames"], extracted, outputs, references, strict=True):
        assert result["source_pts"] == selected["source_pts"]
        assert result["source_time_base"] == selected["source_time_base"]
        assert result["at_ms"] == result["requested_at_ms"] == selected["at_ms"]
        assert output.read_bytes() == reference.read_bytes()
        with Image.open(output) as image:
            observed_color = np.array(image).astype(float).mean(axis=(0, 1))
        color_index = min(3, round(selected["at_ms"] / (1001 / 30)) // 2)
        assert np.abs(observed_color - np.array(colors[color_index]).mean(axis=(0, 1))).max() < 8

    # A precise PTS between pictures must fail, even when derived from a valid
    # PyAV plan. It must never silently fall back to a neighboring real frame.
    missing = dict(actual["frames"][1])
    missing["source_pts"] += 1
    relative = missing["source_pts"] * Fraction(*missing["source_time_base"]) - keyframes._origin(metadata)
    missing.update(pts=relative.numerator, time_base=[1, relative.denominator], at_ms=round(relative * 1000))
    rejected = tmp_path / "missing-exact-pyav-frame.jpg"
    with pytest.raises(media.MediaError, match="精确画面"):
        media.extract_keyframes(path, [missing], [rejected], metadata=metadata)
    assert not rejected.exists()


@pytest.mark.parametrize("pixel_format,color_range,sar", [
    pytest.param("yuv420p", "tv", "1/1", id="bt709-limited-8bit"),
    pytest.param("yuv444p", "pc", "1/1", id="bt709-full-8bit"),
    pytest.param("yuv420p10le", "pc", "1/1", id="bt709-full-10bit"),
    pytest.param("yuv420p", "tv", "4/3", id="bt709-nonsquare-sar"),
    pytest.param("yuv420p10le", "pc", "4/3", id="bt709-full-10bit-nonsquare-sar"),
])
def test_pyav_color_depth_range_and_sample_aspect_match_ffmpeg(tmp_path, pixel_format, color_range, sar):
    """Check decoded pixels, not just source PTS, for graph property handling."""
    pytest.importorskip("av")
    path = tmp_path / "color-properties.mkv"
    filters = (f"scale=in_color_matrix=bt601:out_color_matrix=bt709:in_range=tv:out_range={color_range},"
               f"setsar={sar},format={pixel_format},"
               f"setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range={color_range}")
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=128x72:rate=12:duration=0.5", "-vf", filters,
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-threads", "1", "-pix_fmt", pixel_format,
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", color_range, str(path),
    ], capture_output=True, check=True)
    metadata = media.probe(path)
    stream = next(item for item in metadata["streams"] if item["index"] == metadata["video_stream_index"])
    # Prove the fixture actually carries the property under test; encoder
    # fallback or missing tags must not turn this into an ordinary 8-bit test.
    # H.264's full-range 8-bit output can use the equivalent legacy yuvj name.
    assert stream["pix_fmt"].replace("yuvj", "yuv") == pixel_format
    assert stream["color_space"] == stream["color_primaries"] == stream["color_transfer"] == "bt709"
    assert stream["color_range"] == color_range
    assert stream["sample_aspect_ratio"] == sar.replace("/", ":")
    expected = decoded(path, metadata, "ffmpeg")
    actual = decoded(path, metadata, "pyav")
    assert len(actual) == len(expected) == 6
    assert [item[1:] for item in actual] == [item[1:] for item in expected]
    for left, right in zip(expected, actual, strict=True):
        assert right[0].shape == left[0].shape == (180, 320, 3)
        assert right[0].dtype == left[0].dtype == np.uint8
        assert np.abs(left[0].astype(float) - right[0]).mean() < 2


@pytest.mark.parametrize("termination", ["checkpoint", "close", "decode_error", "missing_pts"])
def test_pyav_closes_container_on_cancellation_close_and_decode_failure(tmp_path, monkeypatch, termination):
    av = pytest.importorskip("av")
    path = make_video(tmp_path / "release.mp4")
    metadata = media.probe(path)
    original_open = av.open
    opened = []

    class TrackedContainer:
        def __init__(self, *args, **kwargs):
            self.inner = original_open(*args, **kwargs)
            self.streams = self.inner.streams
            self.closed = False
            opened.append(self)

        def decode(self, stream):
            if termination == "decode_error":
                raise av.error.InvalidDataError(1, "generated failure")
            if termination == "missing_pts":
                frame = next(self.inner.decode(stream))
                frame.pts = None
                return iter([frame])
            return self.inner.decode(stream)

        def close(self):
            self.inner.close()
            self.closed = True

    monkeypatch.setattr(av, "open", TrackedContainer)
    if termination == "checkpoint":
        clock = count(start=1, step=.25)
        monkeypatch.setattr(keyframes.time, "monotonic", lambda: next(clock))
    calls = 0

    class Cancelled(Exception):
        pass

    def checkpoint():
        nonlocal calls
        calls += 1
        if termination == "checkpoint" and calls >= 4:
            raise Cancelled

    frames = keyframes._decode_pyav(path, 0, metadata["duration_ms"], metadata, checkpoint)
    if termination == "checkpoint":
        with pytest.raises(Cancelled):
            list(frames)
    elif termination == "decode_error":
        with pytest.raises(media.MediaError, match="无法解码"):
            list(frames)
    elif termination == "missing_pts":
        with pytest.raises(media.MediaError, match="时间戳"):
            list(frames)
    else:
        next(frames)
        frames.close()
    assert opened and all(container.closed for container in opened)
