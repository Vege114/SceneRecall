"""Local media operations. All video times are presentation times, never frame / fps.

FFmpeg does decoding; PySceneDetect scores decoded frames. Feeding frame timestamps
from FFmpeg's showinfo filter avoids the constant-frame-rate assumption in OpenCV
VideoCapture seeking, including for variable-frame-rate input.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from bisect import bisect_left
from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Any


class MediaError(ValueError):
    """Invalid local media or a failed media-tool operation."""


def _tool(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise MediaError(f"缺少 {name}，请安装 FFmpeg 后重试。")
    return executable


def _source(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MediaError("视频文件不存在或不是普通文件。")
    return path


def _run(args: list[str], timeout: int = 120, allow_failure: bool = False) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("媒体处理超时，请缩短处理范围后重试。") from exc
    if result.returncode and not allow_failure:
        # Do not echo a command, which could contain unsafe filenames or URL secrets.
        raise MediaError("媒体文件无法解码，或 FFmpeg 处理失败。")
    return result


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def probe(path: Path) -> dict:
    path = _source(path)
    result = _run([
        _tool("ffprobe"), "-v", "error", "-show_format", "-show_streams",
        "-print_format", "json", str(path),
    ])
    try:
        data = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise MediaError("无法解析视频元数据。") from exc
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if not video:
        raise MediaError("文件中没有可播放的视频流。")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    format_info = data.get("format", {})
    origin = _number(format_info.get("start_time"))
    matroska = "matroska" in format_info.get("format_name", "") or "webm" in format_info.get("format_name", "")
    if matroska and "start_time" not in format_info:
        # Very short B-frame streams may omit start_time from format metadata.
        # Inspect only the initial packets to recover the actual presentation origin.
        first = _run([
            _tool("ffprobe"), "-v", "error", "-select_streams", "V:0", "-read_intervals", "%+#32",
            "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
        ])
        initial_frames = json.loads(first.stdout).get("frames", [])
        initial_times = [_number(frame["best_effort_timestamp_time"]) for frame in initial_frames
                         if "best_effort_timestamp_time" in frame]
        if initial_times:
            origin = min(initial_times)
    duration = _number(video.get("duration"))
    if not duration:
        duration = _number(format_info.get("duration"))
    # Matroska's reported duration can include the initial timestamp offset.
    # FFmpeg playback/seek positions use time relative to that initial offset.
    if matroska:
        duration -= origin
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    rotation = next((s.get("rotation", 0) for s in video.get("side_data_list", [])
                     if "rotation" in s), video.get("tags", {}).get("rotate", 0))
    if abs(_number(rotation)) % 180 == 90:
        width, height = height, width
    if duration <= 0 or not width or not height:
        raise MediaError("视频时长或尺寸无效。")
    return {
        "duration_ms": round(duration * 1000), "width": width, "height": height,
        "video_codec": video.get("codec_name", "unknown"),
        "audio_codec": audio.get("codec_name"),
        "format_name": data.get("format", {}).get("format_name", "unknown"),
        "streams": streams, "video_stream_index": video["index"],
        "start_time_ms": round(origin * 1000),
    }


_TEXT_SUBTITLE_CODECS = {"ass", "ssa", "subrip", "srt", "webvtt", "mov_text", "text"}
_BITMAP_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}
_MAX_SUBTITLE_BYTES = 50 * 1024 * 1024


def subtitle_tracks(metadata: dict) -> list[dict]:
    """List subtitle streams by their global FFmpeg index, including unsupported tracks."""
    tracks = []
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") != "subtitle":
            continue
        index = stream.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue
        codec = str(stream.get("codec_name") or "unknown").lower()
        tags, disposition = stream.get("tags", {}), stream.get("disposition", {})
        supported = codec in _TEXT_SUBTITLE_CODECS
        track = {
            "index": index, "codec": codec, "language": str(tags.get("language") or "und"),
            "title": str(tags.get("title") or ""), "default": bool(disposition.get("default")),
            "forced": bool(disposition.get("forced")), "supported": supported,
        }
        if not supported:
            track["reason"] = ("图像字幕轨暂不支持文本提取，请提供外挂字幕；画面 OCR 仅适用于已烧录在画面内的文字。"
                               if codec in _BITMAP_SUBTITLE_CODECS
                               else "暂不支持该字幕编码的文本提取，请提供外挂字幕。")
        tracks.append(track)
    return tracks


def _subtitle_timestamp(value: str, ass: bool) -> str:
    """Normalize FFmpeg's component-signed negative timestamps without losing time.

    ASS/SRT muxers can write e.g. 0:00:00.-30 / 00:00:00,-300 when
    a cue starts before playback zero. Use one leading sign for our parser;
    do not let the muxer shift every cue forward to make this cue positive.
    """
    match = re.fullmatch(r"(-?\d+):(-?\d+):(-?\d+)[.,](-?\d+)", value.strip())
    if not match:
        raise MediaError("提取的字幕时间戳无效。")
    parts = match.groups()
    hour, minute, second, fraction = map(int, parts)
    if abs(minute) >= 60 or abs(second) >= 60 or abs(fraction) >= (100 if ass else 1000):
        raise MediaError("提取的字幕时间戳无效。")
    unit = 10 if ass else 1
    if parts[0].startswith("-") and all(number >= 0 for number in (minute, second, fraction)):
        total = -(((abs(hour) * 60 + minute) * 60 + second) * 1000 + fraction * unit)
    else:
        total = ((hour * 60 + minute) * 60 + second) * 1000 + fraction * unit
    hour, remainder = divmod(abs(total), 3600000)
    minute, remainder = divmod(remainder, 60000)
    second, ms = divmod(remainder, 1000)
    sign = "-" if total < 0 else ""
    if ass:
        return f"{sign}{hour}:{minute:02}:{second:02}.{ms // 10:02}"
    return f"{sign}{hour:02}:{minute:02}:{second:02},{ms:03}"


def _normalize_subtitle_timestamps(path: Path) -> None:
    # Only timestamp fields are rewritten: the ASS header, style definitions,
    # dialogue overrides and text stay intact for provenance/review.
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".ass":
        def event(match: re.Match) -> str:
            return (match[1] + _subtitle_timestamp(match[2], True) + ","
                    + _subtitle_timestamp(match[3], True) + ",")
        in_events, lines = False, []
        for line in text.splitlines(keepends=True):
            if line.strip().startswith("["):
                in_events = line.strip() == "[Events]"
            if in_events:
                line = re.sub(r"^((?:Dialogue|Comment):[^,\r\n]*,)([^,\r\n]*),([^,\r\n]*),", event, line)
            lines.append(line)
        text = "".join(lines)
    else:
        def timing(match: re.Match) -> str:
            return _subtitle_timestamp(match[1], False) + " --> " + _subtitle_timestamp(match[2], False)
        stamp = r"-?\d+:-?\d+:-?\d+[.,]-?\d+"
        text = re.sub(rf"(?m)^({stamp}) --> ({stamp})$", timing, text)
    path.write_text(text, encoding="utf-8")


def extract_subtitle_track(source: Path, stream_index: int, output: Path,
                           metadata: dict | None = None) -> Path:
    """Extract one local text track, with times relative to the media presentation origin.

    ASS/SSA keeps its original style/header information; other supported text
    codecs are converted to SRT. The returned path reflects that actual suffix.
    Re-probe before extraction so stale client metadata cannot select a different
    stream or apply an old file's origin. No OCR/model call is involved.
    """
    if isinstance(stream_index, bool) or not isinstance(stream_index, int) or stream_index < 0:
        raise MediaError("字幕轨编号必须是非负整数。")
    source = _source(source)
    metadata = probe(source)
    track = next((track for track in subtitle_tracks(metadata) if track["index"] == stream_index), None)
    if track is None:
        raise MediaError("所选字幕轨不存在或不是字幕流。")
    if not track["supported"]:
        raise MediaError(track["reason"])
    ass = track["codec"] in {"ass", "ssa"}
    output = Path(output).expanduser().with_suffix(".ass" if ass else ".srt").resolve()
    if source == output:
        raise MediaError("字幕输出不能覆盖原视频。")
    temporary = _temp_output(output)
    try:
        _run([
            _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-copyts", "-itsoffset", f"{-metadata['start_time_ms'] / 1000:.3f}", "-i", str(source),
            "-map", f"0:{stream_index}", "-c:s", "copy" if ass else "srt",
            "-avoid_negative_ts", "disabled", "-fs", str(_MAX_SUBTITLE_BYTES + 1), str(temporary),
        ], timeout=300)
        size = temporary.stat().st_size
        if not size or size >= _MAX_SUBTITLE_BYTES:
            raise MediaError("提取的字幕为空或超过 50 MB。")
        _normalize_subtitle_timestamps(temporary)
        if temporary.stat().st_size >= _MAX_SUBTITLE_BYTES:
            raise MediaError("提取的字幕超过 50 MB。")
        os.replace(temporary, output)
        return output
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with _source(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounds(start_ms: int, end_ms: int) -> None:
    if isinstance(start_ms, bool) or isinstance(end_ms, bool) or not isinstance(start_ms, int) \
            or not isinstance(end_ms, int) or start_ms < 0 or end_ms <= start_ms:
        raise MediaError("时间范围必须是非负整数毫秒，且结束时间晚于开始时间。")


def sample_times(start_ms: int, end_ms: int, count: int = 4) -> list[int]:
    """Uniform bin midpoints, strictly inside the half-open time range."""
    _bounds(start_ms, end_ms)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 256:
        raise MediaError("每个窗口的抽帧数必须介于 1 和 256。")
    duration = end_ms - start_ms
    return sorted({min(end_ms - 1, start_ms + (2 * i + 1) * duration // (2 * count))
                   for i in range(count)})


def make_windows(shots: list[dict], max_ms: int = 6000) -> list[dict]:
    if isinstance(max_ms, bool) or not isinstance(max_ms, int) or max_ms <= 0:
        raise MediaError("分析窗口时长必须是正整数毫秒。")
    windows = []
    previous_end = -1
    for shot in shots:
        start, end = shot["start_ms"], shot["end_ms"]
        _bounds(start, end)
        if start < previous_end:
            raise MediaError("镜头必须按时间排序且不能重叠。")
        for at in range(start, end, max_ms):
            stop = min(at + max_ms, end)
            windows.append({"id": f"window_{at:012d}_{stop:012d}", "shot_id": shot["id"],
                            "start_ms": at, "end_ms": stop})
        previous_end = end
    return windows


_SHOWINFO = re.compile(rb"\bn:\s*(\d+).*?\bpts_time:\s*([-+\deE.]+)")


def detect_shots(path: Path, start_ms: int, end_ms: int) -> list[dict]:
    """Detect cuts in a requested interval using actual decoder presentation times."""
    _bounds(start_ms, end_ms)
    path = _source(path)
    metadata = probe(path)
    if end_ms > metadata["duration_ms"]:
        raise MediaError("分析结束时间超出视频时长。")
    import numpy as np
    from scenedetect import FrameTimecode
    from scenedetect.detectors import ContentDetector

    detector = ContentDetector(threshold=27.0, min_scene_len=2)
    # A fixed small letterboxed image bounds CPU and eliminates odd-size/rotation surprises.
    width, height = 320, 180
    command = [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info",
        "-seek_timestamp", "1", "-ss", f"{(metadata['start_time_ms'] + start_ms) / 1000:.3f}", "-i", str(path),
        "-t", f"{(end_ms - start_ms) / 1000:.3f}",
        "-map", f"0:{metadata['video_stream_index']}", "-an", "-sn", "-dn",
        "-vf", (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                 f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,showinfo"),
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timestamps: dict[int, int] = {}
    errors: deque[bytes] = deque(maxlen=10)

    def read_timestamps() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            match = _SHOWINFO.search(line)
            if match:
                timestamps[int(match[1])] = start_ms + round(float(match[2]) * 1000)
            else:
                errors.append(line)

    stderr_reader = threading.Thread(target=read_timestamps, daemon=True)
    stderr_reader.start()
    cuts, frame_num = [], 0
    size = width * height * 3
    try:
        assert process.stdout is not None
        while True:
            data = process.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                raise MediaError("视频解码返回了不完整的画面。")
            frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
            # This counter is only used to identify the decoded image. All public
            # times below are mapped back to showinfo PTS, not this nominal rate.
            cuts.extend(detector.process_frame(FrameTimecode(frame_num, fps=30.0), frame))
            frame_num += 1
        cuts.extend(detector.post_process(FrameTimecode(max(0, frame_num - 1), fps=30.0)))
        process.wait(timeout=30)
        stderr_reader.join(timeout=10)
        if process.returncode or not frame_num:
            raise MediaError("镜头检测失败，指定范围内没有可解码的画面。")
        if len(timestamps) < frame_num:
            raise MediaError("无法读取完整的媒体时间戳。")
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
        stderr_reader.join(timeout=5)
    boundaries = sorted({start_ms, end_ms, *(
        timestamps[int(cut)] for cut in cuts
        if int(cut) in timestamps and start_ms < timestamps[int(cut)] < end_ms
    )})
    return [{"id": f"shot_{start:012d}_{end:012d}", "start_ms": start, "end_ms": end}
            for start, end in pairwise(boundaries)]


def _crop_filter(crop: Any) -> str | None:
    if crop is None:
        return None
    if isinstance(crop, dict):
        values = [crop.get(k) for k in ("x", "y", "width", "height")]
    else:
        values = list(crop)
    if len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, (float, int))
                               or not math.isfinite(v) for v in values):
        raise MediaError("字幕区域需要四个有限数值。")
    x, y, width, height = values
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
        raise MediaError("字幕区域必须在画面范围内。")
    return f"crop=iw*{width:.8f}:ih*{height:.8f}:iw*{x:.8f}:ih*{y:.8f}"


def _temp_output(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".media-", suffix=out_path.suffix, dir=out_path.parent)
    os.close(handle)
    return Path(name)


def _decode_frame(path: Path, at_ms: int, output: Path, crop_filter: str | None,
                  origin_ms: int = 0) -> subprocess.CompletedProcess:
    return _run([
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
        "-seek_timestamp", "1", "-ss", f"{(origin_ms + at_ms) / 1000:.3f}", "-i", str(path), "-map", "0:V:0",
        "-frames:v", "1", "-an", "-sn", "-vf", ",".join(f for f in (crop_filter, "showinfo") if f),
        "-fps_mode", "passthrough", "-update", "1", str(output),
    ], allow_failure=True)


def _last_visible_frame(path: Path, output: Path, crop_filter: str | None, metadata: dict) -> int:
    """Decode the final presented image when it is held until stream end.

    A near-end ffprobe seek includes the preceding keyframe, unlike accurate
    FFmpeg seeking which can discard a final frame whose PTS precedes the target.
    No nominal fps or invented timestamp is used. Both subprocesses are bounded.
    """
    origin = metadata["start_time_ms"] / 1000
    seek = max(origin, origin + metadata["duration_ms"] / 1000 - 1)
    result = _run([
        _tool("ffprobe"), "-v", "error", "-select_streams", "V:0", "-read_intervals", f"{seek:.6f}%",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
    ], timeout=60)
    try:
        raw = json.loads(result.stdout)
        times = [float(frame["best_effort_timestamp_time"]) - origin for frame in raw.get("frames", [])
                 if frame.get("best_effort_timestamp_time") is not None]
        times = [time for time in times if math.isfinite(time) and 0 <= time * 1000 < metadata["duration_ms"]]
    except (ValueError, TypeError, KeyError) as exc:
        raise MediaError("无法读取视频末帧时间戳。") from exc
    if not times:
        raise MediaError("视频末尾没有可解码的画面。")
    # Floor the seek so a fractional source PTS is never rounded beyond the frame.
    at = max(0, math.floor(max(times) * 1000 + 1e-6))
    decoded = _decode_frame(path, at, output, crop_filter, metadata["start_time_ms"])
    matches = list(_SHOWINFO.finditer(decoded.stderr))
    if decoded.returncode or not output.stat().st_size or not matches:
        raise MediaError("视频末尾画面解码失败。")
    return at + round(float(matches[0][2]) * 1000)


def extract_frame(path: Path, at_ms: int, out_path: Path, crop: Any = None) -> dict:
    if isinstance(at_ms, bool) or not isinstance(at_ms, int) or at_ms < 0:
        raise MediaError("抽帧时间必须是非负整数毫秒。")
    path, out_path = _source(path), Path(out_path).resolve()
    if path == out_path:
        raise MediaError("抽帧输出不能覆盖原视频。")
    metadata = probe(path)
    if at_ms >= metadata["duration_ms"]:
        raise MediaError("抽帧时间超出视频时长。")
    crop_filter = _crop_filter(crop)
    temporary = _temp_output(out_path)
    try:
        result = _decode_frame(path, at_ms, temporary, crop_filter, metadata["start_time_ms"])
        if not temporary.stat().st_size:
            actual_ms = _last_visible_frame(path, temporary, crop_filter, metadata)
            if actual_ms > at_ms:
                raise MediaError("未能解码请求时间附近的画面。")
        else:
            if result.returncode:
                raise MediaError("媒体文件无法解码，或 FFmpeg 处理失败。")
            matches = list(_SHOWINFO.finditer(result.stderr))
            if not matches:
                raise MediaError("无法读取画面的媒体时间戳。")
            actual_ms = at_ms + round(float(matches[0][2]) * 1000)
        os.replace(temporary, out_path)
        return {"id": out_path.stem, "path": str(out_path), "at_ms": max(0, actual_ms),
                "requested_at_ms": at_ms}
    finally:
        temporary.unlink(missing_ok=True)


def extract_frames(path: Path, times_ms: list[int], out_paths: list[Path], crop: Any = None) -> list[dict]:
    """Decode up to 256 requested frames in one seek, preserving each source PTS.

    Use bounded batches so the worker can check pause/cancel between batches. If
    several requested times resolve to the same VFR frame, each keeps its own ID.
    """
    if not times_ms or len(times_ms) != len(out_paths) or len(times_ms) > 256:
        raise MediaError("批量抽帧需要 1 到 256 个时间戳及相同数量的输出路径。")
    if any(isinstance(at, bool) or not isinstance(at, int) or at < 0 for at in times_ms):
        raise MediaError("抽帧时间必须是非负整数毫秒。")
    if times_ms != sorted(times_ms) or times_ms[-1] - times_ms[0] > 300_000:
        raise MediaError("批量抽帧需要按时间排序，且单批跨度不能超过 5 分钟。")
    source = _source(path)
    metadata = probe(source)
    if times_ms[-1] >= metadata["duration_ms"]:
        raise MediaError("抽帧时间超出视频时长。")
    outputs = [Path(output).resolve() for output in out_paths]
    if source in outputs or len(set(outputs)) != len(outputs):
        raise MediaError("抽帧输出路径必须唯一，且不能覆盖原视频。")
    crop_filter = _crop_filter(crop)
    first = times_ms[0]
    relative = sorted({(at - first) / 1000 for at in times_ms})
    expression = "+".join(
        f"gte(t,{time:.3f})*" + ("isnan(prev_selected_t)" if index == 0
                                 else f"lt(prev_selected_t,{time:.3f})")
        for index, time in enumerate(relative)
    )
    filters = [f for f in (f"select='{expression}'", crop_filter, "showinfo") if f]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scenerecall-frames-") as directory:
        target = Path(directory)
        result = _run([
            _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info", "-y",
            "-seek_timestamp", "1", "-ss", f"{(metadata['start_time_ms'] + first) / 1000:.3f}",
            "-i", str(source), "-map", "0:V:0",
            "-t", f"{(times_ms[-1] - first) / 1000 + 5:.3f}",
            "-frames:v", str(len(relative)), "-an", "-sn", "-vf", ",".join(filters),
            "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0", str(target / "%08d.jpg"),
        ], timeout=600, allow_failure=True)
        files = sorted(target.glob("*.jpg"))
        if result.returncode and files:
            raise MediaError("批量画面解码失败。")
        matches = list(_SHOWINFO.finditer(result.stderr))[:len(files)]
        decoded = [first + round(float(match[2]) * 1000) for match in matches]
        if len(decoded) != len(files):
            raise MediaError("无法读取完整的媒体时间戳。")
        held_last = not decoded or decoded[-1] < times_ms[-1]
        if held_last:
            final_path = target / "last.jpg"
            final_ms = _last_visible_frame(source, final_path, crop_filter, metadata)
            if final_ms > times_ms[-1]:
                raise MediaError("批量抽帧未能覆盖请求时间，不能用未来画面替代。")
            if decoded and final_ms < decoded[-1]:
                raise MediaError("视频末帧时间戳与解码结果不一致。")
            files.append(final_path)
            decoded.append(final_ms)
        records = []
        for at, output in zip(times_ms, outputs, strict=True):
            index = bisect_left(decoded, at)
            if index == len(decoded) and held_last:
                index -= 1
            temporary = _temp_output(output)
            try:
                if output.suffix.lower() in {".jpg", ".jpeg"}:
                    shutil.copyfile(files[index], temporary)
                else:
                    from PIL import Image
                    with Image.open(files[index]) as image:
                        image.save(temporary)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            records.append({"id": output.stem, "path": str(output), "at_ms": decoded[index],
                            "requested_at_ms": at})
        return records


def create_proxy(path: Path, out_path: Path) -> None:
    path, out_path = _source(path), Path(out_path).resolve()
    if path == out_path:
        raise MediaError("播放代理不能覆盖原视频。")
    metadata = probe(path)
    temporary = _temp_output(out_path)
    try:
        # Preserve the original relative timeline (including VFR) instead of forcing fps.
        _run([
            _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-seek_timestamp", "1", "-ss", f"{metadata['start_time_ms'] / 1000:.3f}",
            "-i", str(path), "-map", "0:V:0", "-map", "0:a:0?", "-sn", "-dn",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(temporary),
        ], timeout=60 * 60 * 12)
        if not temporary.stat().st_size:
            raise MediaError("播放代理生成失败。")
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
