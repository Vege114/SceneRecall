"""Continuous exact retrieval extraction keeps the established picture contract."""
from fractions import Fraction
import shutil
import subprocess

from PIL import Image
import pytest

from scenerecall import keyframes, media
from test_keyframe_backends import make_video
from test_keyframes import video


@pytest.fixture(autouse=True)
def require_media_tools():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for continuous extraction fixtures")


def many_targets(tmp_path, count=257):
    colors = [Image.new("RGB", (128, 72), color) for color in ("red", "blue")]
    path = video(tmp_path / "many-targets.mp4", [colors[i % 2] for i in range(count) for _ in range(2)],
                 rate="30000/1001", offset=5)
    metadata = media.probe(path)
    plan = keyframes.scan_keyframes(path, 0, metadata["duration_ms"], metadata=metadata, decoder="pyav")
    assert len(plan["frames"]) == count
    return path, metadata, plan["frames"]


def test_more_than_one_batch_uses_one_decoder_and_keeps_identical_jpeg_bytes(tmp_path, monkeypatch):
    path, metadata, frames = many_targets(tmp_path)
    references = [tmp_path / "reference" / f"{i}.jpg" for i in range(len(frames))]
    for start in range(0, len(frames), 128):
        keyframes.extract_keyframes(path, frames[start:start + 128], references[start:start + 128],
                                    metadata=metadata)
    commands = []
    real_popen = keyframes.subprocess.Popen

    def tracked(command, *args, **kwargs):
        if "-i" in command:
            commands.append(command)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(keyframes.subprocess, "Popen", tracked)
    outputs = [tmp_path / "continuous" / f"{i}.jpg" for i in range(len(frames))]
    metrics = {}
    extracted = keyframes.extract_keyframes_stream(path, frames, outputs, metadata=metadata, metrics=metrics)
    assert len(commands) == metrics["extraction_processes"] == metrics["decode_passes"] == 1
    assert metrics["frames"] == len(frames)
    assert any(option in commands[0] for option in ("-/filter:v", "-filter_script:v"))
    assert len(subprocess.list2cmdline(commands[0])) < 4000
    for target, actual, output, reference in zip(frames, extracted, outputs, references, strict=True):
        assert actual["source_pts"] == target["source_pts"]
        assert actual["source_time_base"] == target["source_time_base"]
        assert actual["at_ms"] == target["at_ms"]
        assert output.read_bytes() == reference.read_bytes()


def test_continuous_extraction_rejects_missing_pts_before_publishing(tmp_path):
    path, metadata, frames = many_targets(tmp_path)
    target = dict(frames[len(frames) // 2])
    target["source_pts"] += 1
    relative = target["source_pts"] * Fraction(*target["source_time_base"]) - keyframes._origin(metadata)
    target.update(pts=relative.numerator, time_base=[1, relative.denominator], at_ms=round(relative * 1000))
    frames[len(frames) // 2] = target
    outputs = [tmp_path / "missing" / f"{i}.jpg" for i in range(len(frames))]
    with pytest.raises(media.MediaError, match="精确画面"):
        keyframes.extract_keyframes_stream(path, frames, outputs, metadata=metadata)
    assert not any(output.exists() for output in outputs)


def test_continuous_extraction_cancels_and_reaps_its_decoder(tmp_path, monkeypatch):
    path, metadata, frames = many_targets(tmp_path, count=16)
    real_popen = keyframes.subprocess.Popen
    decoders = []

    def tracked(command, *args, **kwargs):
        process = real_popen(command, *args, **kwargs)
        if "-i" in command:
            decoders.append(process)
        return process

    monkeypatch.setattr(keyframes.subprocess, "Popen", tracked)
    checks = 0

    class Cancelled(Exception):
        pass

    def checkpoint():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise Cancelled

    outputs = [tmp_path / "cancelled" / f"{i}.jpg" for i in range(len(frames))]
    with pytest.raises(Cancelled):
        keyframes.extract_keyframes_stream(path, frames, outputs, metadata=metadata, checkpoint=checkpoint)
    assert decoders and all(process.poll() is not None for process in decoders)
    assert not any(output.exists() for output in outputs)


def test_continuous_extraction_rechecks_source_before_publishing(tmp_path, monkeypatch):
    path, metadata, frames = many_targets(tmp_path, count=8)
    original = keyframes._metadata_for_source
    checks = 0

    def changed(source, snapshot):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise media.MediaError("视频文件已变化")
        return original(source, snapshot)

    monkeypatch.setattr(keyframes, "_metadata_for_source", changed)
    outputs = [tmp_path / "changed" / f"{i}.jpg" for i in range(len(frames))]
    with pytest.raises(media.MediaError, match="已变化"):
        keyframes.extract_keyframes_stream(path, frames, outputs, metadata=metadata)
    assert not any(output.exists() for output in outputs)


@pytest.mark.parametrize("rotation", [0, 90])
def test_continuous_extraction_preserves_vfr_offset_rotation_and_partial_range(tmp_path, rotation):
    source = tmp_path / "vfr.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=128x72:r=10:d=1",
        "-f", "lavfi", "-i", "testsrc2=s=128x72:r=5:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[out]", "-map", "[out]",
        "-fps_mode", "vfr", "-c:v", "libx264", "-output_ts_offset", "5", str(source),
    ], check=True, capture_output=True)
    if rotation:
        source = make_video(tmp_path / "rotation-source.mp4", offset=5)
        path = tmp_path / "rotated.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-display_rotation:v:0", str(rotation),
                        "-i", str(source), "-c", "copy", str(path)],
                       check=True, capture_output=True)
    else:
        path = source
    metadata = media.probe(path)
    if rotation:
        assert (metadata["width"], metadata["height"]) == (72, 128)
    plan = keyframes.scan_keyframes(path, 300, 1900, metadata=metadata, decoder="pyav")
    frames = plan["frames"]
    outputs = [tmp_path / "stream" / f"{i}.jpg" for i in range(len(frames))]
    references = [tmp_path / "reference" / f"{i}.jpg" for i in range(len(frames))]
    keyframes.extract_keyframes(path, frames, references, metadata=metadata)
    result = keyframes.extract_keyframes_stream(path, frames, outputs, metadata=metadata)
    assert [frame["source_pts"] for frame in result] == [frame["source_pts"] for frame in frames]
    assert [output.read_bytes() for output in outputs] == [reference.read_bytes() for reference in references]
