"""Representative-frame regressions use only procedurally generated videos."""
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytest

from scenerecall import keyframes, media


@pytest.fixture(autouse=True)
def require_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for synthetic keyframe tests")


def picture(position=10, color=(230, 120, 50)):
    image = Image.new("RGB", (128, 72), (20, 30, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((position, 12, position + 35, 62), fill=color)
    draw.line((position, 12, position + 35, 62), fill="white", width=3)
    return image


def video(path: Path, pictures, rate="24", offset=None):
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "128x72", "-framerate", rate,
               "-i", "pipe:0", "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast"]
    if offset is not None:
        command.extend(["-output_ts_offset", str(offset)])
    command.append(str(path))
    subprocess.run(command, input=b"".join(image.tobytes() for image in pictures), capture_output=True, check=True)
    return path


def scan(path, **kwargs):
    return media.scan_keyframes(path, 0, media.probe(path)["duration_ms"], **kwargs)


def assert_plan(plan):
    assert plan["algorithm"] == "content-keyframes-v1"
    assert plan["algorithm_parameters"] == keyframes.ALGORITHM_PARAMETERS
    assert plan["algorithm_parameters"] is not keyframes.ALGORITHM_PARAMETERS
    frames = {frame["id"]: frame for frame in plan["frames"]}
    assert len(frames) == len(plan["frames"])
    assert plan["batches"]
    for batch in plan["batches"]:
        assert 1 <= len(batch["frame_ids"]) <= 8
        for frame_id in batch["frame_ids"]:
            frame = frames[frame_id]
            assert frame["shot_id"] == batch["shot_id"]
            assert batch["start_ms"] <= frame["at_ms"] < batch["end_ms"]
            assert round(frame["pts"] * Fraction(*frame["time_base"]) * 1000) == frame["at_ms"]


def test_held_pose_does_not_add_periodic_frames(tmp_path):
    short = scan(video(tmp_path / "short.mp4", [picture()] * 24))
    long = scan(video(tmp_path / "long.mp4", [picture()] * 240))
    assert len(short["frames"]) == len(long["frames"]) == 1
    assert long["frames"][0]["reason"] == "stable_pose"
    assert_plan(long)


def test_stable_poses_dominate_intervening_blur_and_return_pose_is_retained(tmp_path):
    first, second = picture(7), picture(77)
    transition = [Image.blend(first, second, alpha).filter(ImageFilter.GaussianBlur(3))
                  for alpha in (.2, .4, .6, .8)]
    path = video(tmp_path / "poses.mp4", [first] * 12 + transition + [second] * 12 + transition[::-1] + [first] * 12)
    plan = scan(path)
    times = [frame["at_ms"] for frame in plan["frames"]]
    assert any(at < 500 for at in times)
    assert any(667 <= at < 1167 for at in times)
    assert any(at >= 1333 for at in times)
    assert sum(500 <= at < 667 or 1167 <= at < 1333 for at in times) < len(transition)
    assert_plan(plan)


def test_continuous_motion_selects_content_changes_and_limits_model_batches(tmp_path):
    pictures = [picture(at, color=(230, 120, 50)) for at in list(range(0, 91, 3)) * 3]
    plan = scan(video(tmp_path / "motion.mp4", pictures))
    assert 2 < len(plan["frames"]) < len(pictures)
    assert any(frame["reason"] == "content_change" for frame in plan["frames"])
    assert_plan(plan)


def test_short_shots_and_pure_color_are_not_discarded(tmp_path):
    red, blue, green = [Image.new("RGB", (128, 72), color) for color in ("red", "blue", "green")]
    plan = scan(video(tmp_path / "cuts.mp4", [red] * 2 + [blue] * 2 + [green] * 2))
    assert len(plan["shots"]) == len(plan["frames"]) == 3
    assert [shot["start_ms"] for shot in plan["shots"]] == [0, 83, 167]
    assert_plan(plan)


@pytest.mark.parametrize("offset", [None, 5])
def test_precise_pts_survives_fractional_millisecond_rounding_and_offset(tmp_path, offset):
    colors = [Image.new("RGB", (128, 72), color) for color in ("red", "blue", "green", "yellow")]
    path = video(tmp_path / "fractional.mp4", [colors[0]] * 2 + [colors[1]] * 2 + [colors[2]] * 2 + [colors[3]] * 2,
                 rate="30000/1001", offset=offset)
    plan = scan(path)
    assert any(frame["pts"] * Fraction(*frame["time_base"]) * 1000 != frame["at_ms"] for frame in plan["frames"])
    targets = [tmp_path / f"{frame['id']}.jpg" for frame in plan["frames"]]
    extracted = media.extract_keyframes(path, plan["frames"], targets)
    assert [frame["at_ms"] for frame in extracted] == [frame["at_ms"] for frame in plan["frames"]]
    for frame in extracted:
        with Image.open(frame["path"]) as output:
            actual = np.array(output).astype(float).mean(axis=(0, 1))
        target = np.array(colors[min(3, round(frame["at_ms"] / (1001 / 30)) // 2)]).mean(axis=(0, 1))
        assert np.abs(actual - target).max() < 8
    assert_plan(plan)


def test_scan_reports_progress_and_cancellation_reaps_decoder(tmp_path, monkeypatch):
    path = video(tmp_path / "cancel.mp4", [picture(at % 80) for at in range(1000)])
    processes = []
    original = keyframes.subprocess.Popen
    def tracked(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(keyframes.subprocess, "Popen", tracked)
    updates, calls = [], 0
    class Paused(Exception):
        pass
    def checkpoint():
        nonlocal calls
        calls += 1
        if calls > 2:
            raise Paused
    with pytest.raises(Paused):
        scan(path, checkpoint=checkpoint, progress=lambda done, total: updates.append((done, total)))
    assert updates[0][0] == 0
    assert processes and all(process.poll() is not None for process in processes)


def test_completed_scan_reports_final_progress_and_validates_ranges(tmp_path):
    path = video(tmp_path / "progress.mp4", [picture()] * 24)
    updates = []
    plan = scan(path, progress=lambda done, total: updates.append((done, total)))
    assert updates[-1] == (1000, 1000)
    assert_plan(plan)
    with pytest.raises(media.MediaError):
        media.scan_keyframes(path, 0, 2000)
    with pytest.raises(media.MediaError):
        media.extract_keyframes(path, [{**plan["frames"][0], "at_ms": 999}], [tmp_path / "invalid.jpg"])


@pytest.mark.parametrize("offset", [0, 5])
def test_variable_frame_rate_and_partial_ranges_use_real_pts(tmp_path, offset):
    path = tmp_path / "vfr.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=red:s=128x72:r=10:d=1",
        "-f", "lavfi", "-i", "color=blue:s=128x72:r=5:d=1",
        "-f", "lavfi", "-i", "color=green:s=128x72:r=10:d=1",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[out]", "-map", "[out]",
        "-fps_mode", "vfr", "-c:v", "libx264", "-output_ts_offset", str(offset), str(path),
    ], capture_output=True, check=True)
    plan = scan(path)
    assert [frame["at_ms"] for frame in plan["frames"]] == [0, 1000, 2000]
    partial = media.scan_keyframes(path, 1100, 2450)
    assert [frame["at_ms"] for frame in partial["frames"]] == [1200, 2000]
    paths = [tmp_path / f"partial-{i}.jpg" for i in range(len(partial["frames"]))]
    assert [frame["at_ms"] for frame in media.extract_keyframes(path, partial["frames"], paths)] == [1200, 2000]
    assert_plan(partial)


def test_one_frame_shot_and_last_held_frame_keep_original_pts(tmp_path):
    one = video(tmp_path / "one.mp4", [picture()], rate="1")
    assert len(scan(one)["frames"]) == 1
    path = video(tmp_path / "held.mp4", [picture(5), picture(80)], rate="1")
    plan = media.scan_keyframes(path, 900, 2000)
    assert [frame["at_ms"] for frame in plan["frames"]] == [1000]
    result = media.extract_keyframes(path, plan["frames"], [tmp_path / "held-frame.jpg"])
    assert result[0]["at_ms"] == 1000
    with pytest.raises(media.MediaError, match="扩大分析范围"):
        media.scan_keyframes(path, 1500, 1900)


def test_extraction_cancellation_never_publishes_unchecked_image(tmp_path):
    path = video(tmp_path / "source.mp4", [picture()] * 24)
    plan = scan(path)
    output, checks = tmp_path / "cancelled.jpg", 0
    class Cancelled(Exception):
        pass
    def checkpoint():
        nonlocal checks
        checks += 1
        if checks >= 2:
            raise Cancelled
    with pytest.raises(Cancelled):
        media.extract_keyframes(path, plan["frames"], [output], checkpoint=checkpoint)
    assert not output.exists()


def test_legacy_asset_with_rounded_origin_keeps_precise_new_plan_origin(tmp_path):
    from scenerecall.library import Library, atomic_json, read_json
    from scenerecall.models import AssetInput

    path = tmp_path / "fractional-origin.ts"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=red:s=128x72:r=30000/1001:d=0.4", "-c:v", "libx264",
                    "-output_ts_offset", "0.066733", str(path)], capture_output=True, check=True)
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(path), subtitle_mode="embedded"))
    manifest_path = library.asset_dir(asset["id"]) / "manifest.json"
    manifest = read_json(manifest_path)
    exact_origin = Fraction(manifest.pop("start_time_seconds"))
    assert exact_origin * 1000 != manifest["start_time_ms"]
    atomic_json(manifest_path, manifest)
    plan = scan(path)
    assert Fraction(plan["source_start_time_seconds"]) == exact_origin
    outputs = [library.asset_dir(asset["id"]) / "frames" / f"{frame['id']}.jpg" for frame in plan["frames"]]
    for frame in media.extract_keyframes(path, plan["frames"], outputs):
        library.register_frame(asset["id"], frame)
    plan.update(id="plan_legacy_origin", plan_id="plan_legacy_origin", asset_id=asset["id"],
                fingerprint=asset["fingerprint"], start_ms=0, end_ms=asset["duration_ms"], status="completed")
    assert library.validate_keyframe_plan(asset["id"], plan)["frames"] == plan["frames"]
