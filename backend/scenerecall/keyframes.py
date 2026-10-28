"""Content-driven representative pictures, not codec I-frames or recovered key drawings.

Every decoded picture is inspected locally. Holding a pose longer never causes
periodic samples. Only a small look-ahead and the selected metadata stay in memory.
The precise source PTS is retained separately from rounded UI milliseconds.
"""
from __future__ import annotations

import hashlib
import math
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .media import MediaError, _bounds, _metadata_for_source, _source, _temp_output, _tool
from .file_io import atomic_replace

ALGORITHM = "content-keyframes-v1"
# Persist these with each plan so its selection thresholds remain inspectable
# after a later algorithm revision. Values are scalar for a detached JSON snapshot.
ALGORITHM_PARAMETERS = {
    "analysis_width": 320,
    "analysis_height": 180,
    "descriptor_width": 64,
    "descriptor_height": 36,
    "gaussian_kernel_size": 3,
    "pixel_value_max": 255,
    "local_pose_window": 5,
    "held_pose_min_ms": 80,
    "duplicate_gray_mean_max": 2,
    "duplicate_gray_pixel_delta": 12,
    "duplicate_changed_area_max": .02,
    "duplicate_anchor_difference_max": .012,
    "pose_novelty_min": .035,
    "content_change_novelty_min": .12,
    "sharpness_ratio_min": .35,
    "shot_sharpness_percentile": 75,
    "scene_threshold": 27.0,
    "scene_min_frames": 2,
    "scene_hue_weight": 1.0,
    "scene_saturation_weight": 1.0,
    "scene_luminance_weight": 1.0,
    "scene_edges_weight": 0.0,
    "scene_frame_index_fps": 30.0,
    "max_batch_frames": 8,
}
WIDTH, HEIGHT = ALGORITHM_PARAMETERS["analysis_width"], ALGORITHM_PARAMETERS["analysis_height"]
MAX_BATCH_FRAMES = ALGORITHM_PARAMETERS["max_batch_frames"]
_TIME_BASE = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")
_PTS = re.compile(rb"\bn:\s*(\d+)\s+pts:\s*(-?\d+)")


def _origin(metadata):
    return Fraction(metadata.get("start_time_seconds", str(metadata.get("start_time_ms", 0) / 1000)))


def _seconds(value: Fraction) -> str:
    # Floor rather than round: seeking just after a selected frame would lose it.
    ticks = value.numerator * 1_000_000 // value.denominator
    return f"{'-' if ticks < 0 else ''}{abs(ticks) // 1_000_000}.{abs(ticks) % 1_000_000:06d}"


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _metadata_reader(stream, records, condition, errors):
    base = None
    try:
        for line in stream:
            match = _TIME_BASE.search(line)
            if match:
                base = (int(match[1]), int(match[2]))
            match = _PTS.search(line)
            if match and base is not None:
                with condition:
                    records[int(match[1])] = (int(match[2]), base)
                    condition.notify_all()
    except (OSError, ValueError) as exc:
        errors.append(exc)


def _decode(path, start_ms, end_ms, metadata, checkpoint):
    """Yield low-resolution pixels and exact source timestamps; cancellation is bounded."""
    origin = _origin(metadata)
    container_origin = Fraction(metadata.get("_container_start_time_seconds", str(origin)))
    stop = origin + Fraction(end_ms, 1000)
    if stop <= container_origin:
        return
    # Start one second earlier to preserve held pictures at a partial-range start.
    seek = max(Fraction(0), origin + Fraction(start_ms, 1000) - container_origin - 1)
    command = [
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info", "-copyts",
        "-ss", _seconds(seek), "-t", _seconds(stop - container_origin - seek), "-i", str(path),
        "-map", f"0:{metadata['video_stream_index']}", "-an", "-sn", "-dn",
        "-vf", (f"trim=end={_seconds(stop)},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                 f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,showinfo=checksum=0"),
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    if checkpoint:
        checkpoint()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pictures, records, condition, errors = queue.Queue(maxsize=3), {}, threading.Condition(), []
    stopped = threading.Event()

    def read_pictures():
        try:
            while not stopped.is_set():
                data = process.stdout.read(WIDTH * HEIGHT * 3)
                while not stopped.is_set():
                    try:
                        pictures.put(data, timeout=.1)
                        break
                    except queue.Full:
                        continue
                if not data:
                    break
        except (OSError, ValueError) as exc:
            errors.append(exc)

    readers = [threading.Thread(target=read_pictures, daemon=True),
               threading.Thread(target=_metadata_reader,
                                args=(process.stderr, records, condition, errors), daemon=True)]
    for reader in readers:
        reader.start()
    index, checked, last_output = 0, 0.0, time.monotonic()
    try:
        while True:
            current = time.monotonic()
            if current - checked >= .2:
                if checkpoint:
                    checkpoint()
                checked = current
            try:
                pixels = pictures.get(timeout=.1)
            except queue.Empty:
                if errors or (process.poll() is not None and not readers[0].is_alive()):
                    raise MediaError("关键帧扫描中断，未能读取完整画面。")
                if current - last_output > 30:
                    raise MediaError("关键帧扫描解码超时。")
                continue
            if not pixels:
                break
            last_output = current
            if len(pixels) != WIDTH * HEIGHT * 3:
                raise MediaError("关键帧扫描返回了不完整的画面。")
            while index not in records:
                if checkpoint:
                    checkpoint()
                with condition:
                    condition.wait(timeout=.1)
                if not readers[1].is_alive() or time.monotonic() - last_output > 30:
                    raise MediaError("关键帧扫描缺少实际媒体时间戳。")
            with condition:
                pts, base = records.pop(index)
            index += 1
            source_time = pts * Fraction(*base)
            if source_time >= stop:
                continue
            yield np.frombuffer(pixels, np.uint8).reshape(HEIGHT, WIDTH, 3), pts, base, source_time - origin
        process.wait(timeout=5)
        if process.returncode or errors:
            raise MediaError("关键帧扫描无法解码视频。")
    finally:
        stopped.set()
        _stop(process)
        for stream in (process.stdout, process.stderr):
            if stream:
                stream.close()
        for reader in readers:
            reader.join(timeout=2)


def _rotation_filters(metadata):
    """Mirror FFmpeg autorotation, including the display matrix's flip signs."""
    stream = next((item for item in metadata.get("streams", [])
                   if item.get("index") == metadata["video_stream_index"]), {})
    side = next((item for item in stream.get("side_data_list", []) if "rotation" in item), {})
    try:
        angle = -float(side.get("rotation", stream.get("tags", {}).get("rotate", 0))) % 360
    except (TypeError, ValueError) as exc:
        raise MediaError("PyAV 解码后端无法读取视频旋转信息。") from exc
    if not math.isfinite(angle):
        raise MediaError("PyAV 解码后端无法读取视频旋转信息。")
    matrix = [int(value) for row in re.findall(
        r"^\s*[0-9a-fA-F]+:\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)",
        side.get("displaymatrix", ""), re.MULTILINE) for value in row]
    if matrix and len(matrix) != 9:
        raise MediaError("PyAV 解码后端无法读取视频显示矩阵。")
    if abs(angle - 90) < 1:
        return [("transpose", "cclock_flip" if matrix and matrix[3] > 0 else "clock")]
    if abs(angle - 180) < 1:
        return [(name, None) for name, negative in (
            ("hflip", not matrix or matrix[0] < 0), ("vflip", not matrix or matrix[4] < 0)) if negative]
    if abs(angle - 270) < 1:
        return [("transpose", "clock_flip" if matrix and matrix[3] < 0 else "cclock")]
    if abs(angle) > 1:
        return [("rotate", f"{angle:.6f}*PI/180")]
    return [("vflip", None)] if matrix and matrix[4] < 0 else []


def _pyav_filter_graph(av, frame, metadata):
    # Use libavfilter instead of a separate OpenCV resize implementation. Different
    # bundled FFmpeg versions can still produce different pixels: this is an
    # explicitly selected decoder, never an automatic fallback to another backend.
    graph = av.filter.Graph()
    nodes = [graph.add_buffer(template=frame)]
    for name, args in _rotation_filters(metadata):
        nodes.append(graph.add(name, args))
    nodes.extend([
        graph.add("scale", f"{WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease"),
        graph.add("pad", f"{WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2"),
        graph.add("format", "pix_fmts=bgr24"),
        graph.add("buffersink"),
    ])
    graph.link_nodes(*nodes)
    graph.configure()
    return graph


def _decode_pyav(path, start_ms, end_ms, metadata, checkpoint):
    """Decode every picture with native PTS and a bounded cancellation checkpoint."""
    try:
        import av
    except ImportError as exc:
        raise MediaError("PyAV 解码后端未安装，请修复应用依赖后重试。") from exc
    if checkpoint:
        checkpoint()
    origin = _origin(metadata)
    seek = max(Fraction(0), Fraction(start_ms, 1000) - 1)
    stop = origin + Fraction(end_ms, 1000)
    container = None
    try:
        container = av.open(str(path), timeout=(30.0, 30.0))
        # metadata stores the global stream index, not its ordinal among videos.
        stream = next((item for item in container.streams
                       if item.index == metadata["video_stream_index"] and item.type == "video"), None)
        if stream is None:
            raise MediaError("PyAV 解码后端找不到计划指定的视频流。")
        stream.thread_type = "AUTO"
        if seek:
            ticks = (origin + seek) / Fraction(stream.time_base)
            container.seek(ticks.numerator // ticks.denominator, stream=stream,
                           backward=True, any_frame=False)
        graph, layout, checked = None, None, 0.0
        for frame in container.decode(stream):
            current = time.monotonic()
            if current - checked >= .2:
                if checkpoint:
                    checkpoint()
                checked = current
            if frame.pts is None or frame.time_base is None:
                raise MediaError("关键帧扫描缺少实际媒体时间戳。")
            base = Fraction(frame.time_base)
            pts = frame.pts
            source_time = pts * base
            if source_time >= stop:
                break
            if source_time < origin + seek:
                continue
            current_layout = (frame.width, frame.height, frame.format.name, base)
            if current_layout != layout:
                graph = _pyav_filter_graph(av, frame, metadata)
                layout = current_layout
            graph.push(frame)
            filtered = graph.pull()
            if filtered.pts is None or filtered.pts * Fraction(filtered.time_base) != source_time:
                raise MediaError("PyAV 解码后端改变了画面时间戳。")
            yield filtered.to_ndarray(), pts, (base.numerator, base.denominator), source_time - origin
    except av.error.FFmpegError as exc:
        raise MediaError("PyAV 解码后端无法解码视频。") from exc
    finally:
        if container is not None:
            container.close()


def _difference(left, right):
    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean()
                 / ALGORITHM_PARAMETERS["pixel_value_max"])


def _feature(pixels, pts, base, timestamp):
    kernel = (ALGORITHM_PARAMETERS["gaussian_kernel_size"],) * 2
    size = (ALGORITHM_PARAMETERS["descriptor_width"], ALGORITHM_PARAMETERS["descriptor_height"])
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, kernel, 0)
    return {"descriptor": cv2.resize(cv2.GaussianBlur(pixels, kernel, 0), size),
            "gray": cv2.resize(blurred, size),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "time": timestamp, "source_pts": pts, "source_time_base": list(base),
            "pts": timestamp.numerator, "time_base": [1, timestamp.denominator],
            "at_ms": round(timestamp * 1000)}


class _Selector:
    """Collapse held pictures, then inspect content changes with five-pose look-ahead."""

    def __init__(self):
        self.run = None
        self.recent = deque()
        self.previous_pose = None
        self.last_selected = None
        self.selected = []
        self.best = None
        self.started = False

    def add(self, frame):
        if self.best is None or frame["sharpness"] > self.best["sharpness"]:
            self.best = frame
        if self.run is not None:
            delta = np.abs(frame["gray"].astype(np.int16) - self.run["last"]["gray"].astype(np.int16))
            # Compare against the run anchor as well: slow motion must not merge forever.
            duplicate = (float(delta.mean()) <= ALGORITHM_PARAMETERS["duplicate_gray_mean_max"]
                         and float((delta > ALGORITHM_PARAMETERS["duplicate_gray_pixel_delta"]).mean())
                         <= ALGORITHM_PARAMETERS["duplicate_changed_area_max"]
                         and _difference(frame["descriptor"], self.run["first"]["descriptor"])
                         <= ALGORITHM_PARAMETERS["duplicate_anchor_difference_max"])
            if duplicate:
                self.run["last"] = frame
                if frame["sharpness"] > self.run["best"]["sharpness"]:
                    self.run["best"] = frame
                return
            self._end_run(frame["time"])
        self.run = {"first": frame, "last": frame, "best": frame}

    def _end_run(self, stop):
        frame = {**self.run["best"], "held": stop - self.run["first"]["time"]
                 >= Fraction(ALGORITHM_PARAMETERS["held_pose_min_ms"], 1000)}
        frame["movement"] = (_difference(frame["descriptor"], self.previous_pose["descriptor"])
                             if self.previous_pose is not None else 0)
        self.previous_pose = frame
        self.recent.append(frame)
        if len(self.recent) >= ALGORITHM_PARAMETERS["local_pose_window"]:
            center = ALGORITHM_PARAMETERS["local_pose_window"] // 2
            if not self.started:
                for position in range(center):
                    self._consider(self.recent[position], list(self.recent), edge=position == 0)
                self.started = True
            self._consider(list(self.recent)[center], list(self.recent))
            self.recent.popleft()
        self.run = None

    def _consider(self, frame, neighbors, edge=False):
        if self.last_selected is not None and frame["time"] <= self.last_selected["time"]:
            return
        sharpest = max(item["sharpness"] for item in neighbors)
        if sharpest > 0 and frame["sharpness"] < ALGORITHM_PARAMETERS["sharpness_ratio_min"] * sharpest:
            return
        novelty = (_difference(frame["descriptor"], self.last_selected["descriptor"])
                   if self.last_selected is not None else 1)
        local_min = frame["movement"] <= min(item["movement"] for item in neighbors)
        if frame["held"] and novelty >= ALGORITHM_PARAMETERS["pose_novelty_min"]:
            reason = "stable_pose"
        elif (edge or local_min) and novelty >= ALGORITHM_PARAMETERS["pose_novelty_min"]:
            reason = "representative_pose"
        elif novelty >= ALGORITHM_PARAMETERS["content_change_novelty_min"]:
            reason = "content_change"
        else:
            return
        self.last_selected = frame
        self.selected.append({**{key: value for key, value in frame.items() if key not in {"descriptor", "gray"}},
                              "reason": reason})

    def finish(self, stop):
        if self.run is not None:
            self._end_run(stop)
        # A held opening pose must not disappear simply because there are later poses.
        for frame in self.recent:
            self._consider(frame, list(self.recent), edge=True)
        if not self.selected and self.best is not None:
            self.selected = [{**self.best, "reason": "shot_representative"}]
        if not self.selected:
            return []
        upper = float(np.percentile([f["sharpness"] for f in self.selected],
                                    ALGORITHM_PARAMETERS["shot_sharpness_percentile"]))
        retained = [f for f in self.selected
                    if not upper or f["sharpness"] >= upper * ALGORITHM_PARAMETERS["sharpness_ratio_min"]]
        return [{key: value for key, value in frame.items()
                 if key not in {"descriptor", "gray", "sharpness", "time", "movement", "held"}}
                for frame in retained]


def scan_keyframes(path: Path, start_ms: int, end_ms: int, checkpoint=None, progress=None,
                   *, sample_collector=None, metadata: dict | None = None, decoder: str = "ffmpeg") -> dict:
    from scenedetect import FrameTimecode
    from scenedetect.detectors import ContentDetector

    _bounds(start_ms, end_ms)
    source = _source(path)
    if decoder not in {"ffmpeg", "pyav"}:
        raise MediaError("未知关键帧解码后端。")
    metadata = _metadata_for_source(source, metadata)
    if end_ms > metadata["duration_ms"]:
        raise MediaError("所选分析片段的结束位置超过视频可播放范围，请检查片段时间范围。")
    if progress:
        progress(0, end_ms - start_ms)
    detector = ContentDetector(
        threshold=ALGORITHM_PARAMETERS["scene_threshold"], min_scene_len=ALGORITHM_PARAMETERS["scene_min_frames"],
        weights=ContentDetector.Components(ALGORITHM_PARAMETERS["scene_hue_weight"],
                                           ALGORITHM_PARAMETERS["scene_saturation_weight"],
                                           ALGORITHM_PARAMETERS["scene_luminance_weight"],
                                           ALGORITHM_PARAMETERS["scene_edges_weight"]),
    )
    pending, cuts = deque(), set()
    selector, shots, selected = _Selector(), [], []
    shot_start, last_progress, index = start_ms, 0.0, 0

    def finish_shot(stop):
        nonlocal selector, shot_start
        if stop <= shot_start:
            return
        if sample_collector is not None:
            sample_collector.finish_shot(shot_start, stop)
        frames = selector.finish(Fraction(stop, 1000))
        if frames:
            shot_id = f"shot_{shot_start:012d}_{stop:012d}"
            shots.append({"id": shot_id, "start_ms": shot_start, "end_ms": stop})
            for frame in frames:
                signature = f"{frame['source_pts']}:{frame['source_time_base']}"
                frame.update(id="keyframe_" + hashlib.sha256(signature.encode()).hexdigest()[:24], shot_id=shot_id)
            selected.extend(frames)
        selector, shot_start = _Selector(), stop

    def consume(item):
        number, frame = item
        if number in cuts:
            finish_shot(max(start_ms, min(end_ms, frame["at_ms"])))
        if sample_collector is not None:
            sample_collector.observe(frame)
        selector.add(frame)

    decode = _decode if decoder == "ffmpeg" else _decode_pyav
    decoded = decode(source, start_ms, end_ms, metadata, checkpoint)
    try:
        for pixels, pts, base, timestamp in decoded:
            # Earlier decoding exists only for accurate seeks. Public evidence stays
            # within the requested half-open range, including its rounded PTS.
            if timestamp < Fraction(start_ms, 1000) or not start_ms <= round(timestamp * 1000) < end_ms:
                continue
            frame = _feature(pixels, pts, base, timestamp)
            cuts.update(int(cut) for cut in detector.process_frame(
                FrameTimecode(index, fps=ALGORITHM_PARAMETERS["scene_frame_index_fps"]), pixels))
            pending.append((index, frame))
            if len(pending) > detector.event_buffer_length + 1:
                consume(pending.popleft())
            index += 1
            if progress and time.monotonic() - last_progress >= .25:
                progress(min(end_ms - start_ms, frame["at_ms"] - start_ms), end_ms - start_ms)
                last_progress = time.monotonic()
        if not index:
            raise MediaError("所选范围内没有实际起始时间符合要求的视频帧，请扩大分析范围。")
        cuts.update(int(cut) for cut in detector.post_process(
            FrameTimecode(index - 1, fps=ALGORITHM_PARAMETERS["scene_frame_index_fps"])))
        while pending:
            consume(pending.popleft())
        finish_shot(end_ms)
    finally:
        decoded.close()
    batches = []
    by_shot = {}
    for frame in selected:
        by_shot.setdefault(frame["shot_id"], []).append(frame)
    for shot in shots:
        frames = by_shot[shot["id"]]
        chunks = [frames[i:i + MAX_BATCH_FRAMES] for i in range(0, len(frames), MAX_BATCH_FRAMES)]
        boundaries = [shot["start_ms"]]
        for left, right in zip(chunks, chunks[1:]):
            boundaries.append((left[-1]["at_ms"] + right[0]["at_ms"] + 1) // 2)
        boundaries.append(shot["end_ms"])
        for position, chunk in enumerate(chunks):
            start, stop = boundaries[position:position + 2]
            batches.append({"id": f"keyframes_{start:012d}_{stop:012d}", "shot_id": shot["id"],
                            "start_ms": start, "end_ms": stop, "frame_ids": [frame["id"] for frame in chunk]})
    if checkpoint:
        checkpoint()
    if progress:
        progress(end_ms - start_ms, end_ms - start_ms)
    return {"algorithm": ALGORITHM, "algorithm_parameters": dict(ALGORITHM_PARAMETERS),
            "source_start_time_seconds": str(_origin(metadata)),
            "shots": shots, "frames": selected, "batches": batches}


def _select_expression(timestamps: list[int]) -> str:
    """Keep FFmpeg's expression parser depth logarithmic for large batches.

    A flat eq(pts,...)+eq(pts,...)+... sum forms a left-deep parse tree and
    exceeds FFmpeg's recursion limit before our 128/256-picture batch limits.
    Parenthesize a balanced tree without rounding or widening the PTS matches.
    """
    if len(timestamps) == 1:
        return f"eq(pts,{timestamps[0]})"
    middle = len(timestamps) // 2
    return f"({_select_expression(timestamps[:middle])}+{_select_expression(timestamps[middle:])})"


def extract_keyframes(path: Path, frames: list[dict], out_paths: list[Path], checkpoint=None,
                      *, metadata: dict | None = None) -> list[dict]:
    """Extract a bounded public API batch of exact source pictures."""
    return _extract_keyframes(path, frames, out_paths, checkpoint, metadata=metadata, max_frames=256)


def extract_keyframes_stream(path: Path, frames: list[dict], out_paths: list[Path], checkpoint=None,
                             *, metadata: dict | None = None, metrics: dict | None = None) -> list[dict]:
    """Extract an entire retrieval plan with one continuous FFmpeg decode.

    Pictures stream straight to temporary JPEG files, not an in-memory image
    list. All source PTS and the source identity are checked before any output
    is published. The caller can retain dense evidence without restarting the
    decoder for each 128-picture batch. Scanning and extraction remain two
    separate decode passes so the existing JPEG/color contract stays intact.
    """
    return _extract_keyframes(path, frames, out_paths, checkpoint, metadata=metadata,
                              max_frames=None, metrics=metrics)


def _filter_arguments(expression: str, directory: Path) -> list[str]:
    # Windows CreateProcess limits the complete command line to 32767 UTF-16
    # characters. Dense/long retrieval plans can exceed it even though the
    # balanced select expression itself is valid. Keep the filter in a file.
    script = directory / "selected-frames.filter"
    script.write_text(expression, encoding="utf-8")
    executable = _tool("ffmpeg")
    try:
        version = subprocess.run([executable, "-version"], capture_output=True,
                                 timeout=15, check=True).stdout
        major = int(re.search(rb"ffmpeg version (\d+)", version)[1])
    except (subprocess.SubprocessError, ValueError, TypeError):
        # Unknown/custom builds retain the traditional filter-file spelling.
        major = 0
    # FFmpeg 7 added generic option-file loading. FFmpeg 9 removed the former
    # filter_script option, while earlier supported builds still need it.
    return ["-/filter:v" if major >= 7 else "-filter_script:v", str(script)]


class _ExtractionWatchdog:
    """A continuous long-video extraction may exceed ten minutes while progressing."""

    def __init__(self, continuous):
        self.continuous = continuous
        self.started = self.last_progress = time.monotonic()
        self.frames = 0

    def expired(self, frame_count):
        current = time.monotonic()
        if frame_count > self.frames:
            self.frames, self.last_progress = frame_count, current
        return current - (self.last_progress if self.continuous else self.started) > 600


def _extract_keyframes(path, frames, out_paths, checkpoint, *, metadata, max_frames, metrics=None):
    started = time.perf_counter()
    source = _source(path)
    if not frames or (max_frames is not None and len(frames) > max_frames) or len(frames) != len(out_paths):
        limit = "1–256 个" if max_frames is not None else "至少一个"
        raise MediaError(f"关键帧提取需要 {limit}计划画面及相同数量的输出路径。")
    outputs = [Path(output).resolve() for output in out_paths]
    if source in outputs or len(set(outputs)) != len(outputs):
        raise MediaError("抽帧输出路径必须唯一，且不能覆盖原视频。")
    metadata = _metadata_for_source(source, metadata)
    origin = _origin(metadata)
    container_origin = Fraction(metadata.get("_container_start_time_seconds", str(origin)))
    targets = []
    for frame in frames:
        try:
            pts, base = frame["source_pts"], frame["source_time_base"]
            if isinstance(pts, bool) or not isinstance(pts, int) or len(base) != 2 \
                    or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in base):
                raise ValueError
            timestamp = pts * Fraction(*base)
            relative = timestamp - origin
            if (relative < 0 or relative >= Fraction(metadata["duration_ms"], 1000)
                    or round(relative * 1000) != frame["at_ms"]
                    or frame["pts"] * Fraction(*frame["time_base"]) != relative):
                raise ValueError
            targets.append(timestamp)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise MediaError("关键帧计划的实际时间戳无效或视频来源已变化。") from exc
    if targets != sorted(set(targets)):
        raise MediaError("关键帧必须按真实时间戳排序且不能重复。")
    if len({tuple(frame["source_time_base"]) for frame in frames}) != 1:
        raise MediaError("单批关键帧的源时间基必须相同。")
    if checkpoint:
        checkpoint()
    with tempfile.TemporaryDirectory(prefix="scenerecall-keyframes-") as directory:
        temporary_dir = Path(directory)
        select = _select_expression([frame["source_pts"] for frame in frames])
        expression = f"select='{select}',showinfo=checksum=0"
        filters = (["-vf", expression] if max_frames is not None
                   else _filter_arguments(expression, temporary_dir))
        command = [
            _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info", "-y", "-copyts",
            "-ss", _seconds(max(Fraction(0), targets[0] - container_origin - 1)), "-i", str(source),
            "-map", f"0:{metadata['video_stream_index']}", "-an", "-sn", "-dn",
            *filters, "-frames:v", str(len(frames)),
            "-fps_mode", "passthrough", "-q:v", "2", "-start_number", "0", str(temporary_dir / "%08d.jpg"),
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        records, condition, errors = {}, threading.Condition(), []
        reader = threading.Thread(target=_metadata_reader, args=(process.stderr, records, condition, errors), daemon=True)
        reader.start()
        watchdog = _ExtractionWatchdog(continuous=max_frames is None)
        try:
            while process.poll() is None:
                if checkpoint:
                    checkpoint()
                with condition:
                    decoded_count = len(records)
                if watchdog.expired(decoded_count):
                    raise MediaError("关键帧提取超时。")
                time.sleep(.1)
            reader.join(timeout=2)
            images = sorted(temporary_dir.glob("*.jpg"))
            actual = [pts * Fraction(*base) for _, (pts, base) in sorted(records.items())]
            if process.returncode or errors or len(images) != len(frames) or actual != targets:
                raise MediaError("无法提取计划指定的精确画面；未用相邻帧替代。")
            _metadata_for_source(source, metadata)
            decode_finished = time.perf_counter()
            result = []
            for frame, image, output in zip(frames, images, outputs, strict=True):
                if checkpoint:
                    checkpoint()
                temporary = _temp_output(output)
                try:
                    if output.suffix.lower() in {".jpg", ".jpeg"}:
                        shutil.copyfile(image, temporary)
                    else:
                        from PIL import Image
                        with Image.open(image) as picture:
                            picture.save(temporary)
                    atomic_replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
                result.append({**frame, "id": output.stem, "path": str(output), "requested_at_ms": frame["at_ms"]})
            if metrics is not None:
                metrics.update(decode_passes=1, extraction_processes=1, frames=len(result),
                               decode_ms=round((decode_finished - started) * 1000),
                               output_write_ms=round((time.perf_counter() - decode_finished) * 1000),
                               elapsed_ms=round((time.perf_counter() - started) * 1000))
            return result
        finally:
            _stop(process)
            if process.stderr:
                process.stderr.close()
            reader.join(timeout=2)
