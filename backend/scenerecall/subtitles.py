"""Subtitle ingestion and conservative OCR merging without translated/inferred text."""
from __future__ import annotations

import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class SubtitleError(ValueError):
    """A required subtitle file or OCR record is invalid."""


class _PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "br":
            self.parts.append("\n")


def _plain(text: str) -> str:
    parser = _PlainText()
    # WebVTT inline karaoke timestamps are metadata, not displayed dialogue.
    parser.feed(re.sub(r"<(?:\d+:)?\d{2}:\d{2}\.\d{3}>", "", text))
    return "\n".join(line.strip() for line in "".join(parser.parts).splitlines()).strip()


def language_of(text: str) -> str:
    """Best effort script detection. Latin alone cannot reliably identify a language."""
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    return "und"


def _groups(text: str) -> list[tuple[str, str]]:
    groups: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        language = language_of(line)
        if groups and groups[-1][1] == language:
            previous, _ = groups.pop()
            groups.append((previous + "\n" + line, language))
        else:
            groups.append((line, language))
    return groups


def _id(*parts: Any) -> str:
    return "subtitle_" + hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:24]


def _read(path: Path) -> str:
    raw = path.read_bytes()
    if not raw or len(raw) > 50 * 1024 * 1024:
        raise SubtitleError("字幕文件为空或超过 50 MB。")
    encodings = ["utf-16"] if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else ["utf-8-sig", "gb18030"]
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            if "\x00" in text:
                raise SubtitleError("字幕文件含有无效字符，请保存为 UTF-8 文本。")
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    raise SubtitleError("字幕文件编码无法识别，请保存为 UTF-8 文本。")


_TIME = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[,.](\d{3})$")


def _time(text: str, allow_negative: bool = False) -> int:
    sign = -1 if allow_negative and text.startswith("-") else 1
    if sign == -1:
        text = text[1:]
    match = _TIME.fullmatch(text)
    if not match:
        raise SubtitleError("字幕中存在无法解析的时间戳。")
    hour, minute, second, ms = match.groups()
    if int(minute) >= 60 or int(second) >= 60:
        raise SubtitleError("字幕时间戳的分或秒必须小于 60。")
    return sign * (((int(hour or 0) * 60 + int(minute)) * 60 + int(second)) * 1000 + int(ms))


def _text_cues(text: str, suffix: str, container: bool = False) -> list[tuple[int, int, str]]:
    if suffix == ".vtt":
        lines = text.lstrip("\ufeff").splitlines()
        if not lines or not re.fullmatch(r"WEBVTT(?:[ \t].*)?", lines[0]):
            raise SubtitleError("VTT 字幕缺少 WEBVTT 文件头。")
        text = "\n".join(lines[1:])
        # Metadata preceding the first blank line belongs to the VTT header.
        if text and not text.startswith("\n") and "\n\n" in text:
            text = text.split("\n\n", 1)[1]
    cues = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if suffix == ".vtt" and (lines[0] in ("STYLE", "REGION") or re.match(r"NOTE(?:\s|$)", lines[0])):
            continue
        index = 0 if "-->" in lines[0] else 1
        if index >= len(lines) or "-->" not in lines[index]:
            raise SubtitleError("字幕包含损坏的段落或缺少时间戳。")
        if suffix == ".srt" and index and not lines[0].strip().isdigit():
            raise SubtitleError("SRT 字幕序号无效。")
        timing = re.fullmatch(r"\s*(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[index])
        if not timing:
            raise SubtitleError("字幕时间范围格式无效。")
        start, end = _time(timing[1], container), _time(timing[2], container)
        content = _plain("\n".join(lines[index + 1:]))
        if not content and not container:
            raise SubtitleError("字幕段落没有可显示的文字。")
        cues.append((start, end, content))
    return cues


def parse_subtitles(path: Path, duration_ms: int, offset_ms: int = 0) -> list[dict]:
    """Read external subtitles, strictly rejecting cues outside the asset timeline."""
    return _parse_subtitles(path, duration_ms, offset_ms, container=False)


def parse_container_subtitles(path: Path, duration_ms: int, offset_ms: int = 0) -> list[dict]:
    """Read an extracted track on the playback timeline.

    Empty/clear-screen and drawing-only events are not dialogue. Visible cues
    crossing an asset boundary are clipped and flagged for review, preserving
    their pre-clip times. Completely out-of-range or invalid visible cues fail
    explicitly, so a wrong track/origin cannot silently produce partial success.
    """
    return _parse_subtitles(path, duration_ms, offset_ms, container=True)


def _ass_plain(cue: Any, styles: dict) -> str:
    from pysubs2 import SSAStyle
    from pysubs2.formats.substation import parse_tags

    # A single ASS event may alternate between vector drawings and dialogue.
    # is_drawing discards that entire event; keep only the displayed text spans.
    fragments = parse_tags(cue.text, styles.get(cue.style, SSAStyle.DEFAULT_STYLE), styles)
    text = "".join(fragment for fragment, style in fragments if not style.drawing)
    return _plain(text.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " "))


def _parse_subtitles(path: Path, duration_ms: int, offset_ms: int, container: bool) -> list[dict]:
    if path is None or not str(path).strip():
        raise SubtitleError("外挂字幕模式必须提供字幕文件。")
    path = Path(path).expanduser()
    if not path.is_file():
        raise SubtitleError("字幕文件不存在或不是普通文件。")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms <= 0:
        raise SubtitleError("视频时长必须是正整数毫秒。")
    if isinstance(offset_ms, bool) or not isinstance(offset_ms, int):
        raise SubtitleError("字幕偏移必须是整数毫秒。")
    suffix = path.suffix.lower()
    if suffix not in {".srt", ".vtt", ".ass"}:
        raise SubtitleError("只支持 SRT、VTT 和 ASS 外挂字幕。")
    text = _read(path)
    if suffix == ".ass":
        import pysubs2
        if "[Events]" not in text or not re.search(r"^Format:.*Start.*End.*Text", text, re.MULTILINE):
            raise SubtitleError("ASS 字幕缺少有效的 Events/Format 段。")
        try:
            parsed = pysubs2.SSAFile.from_string(text, format_="ass")
            cues = [(int(cue.start), int(cue.end), _ass_plain(cue, parsed.styles))
                    for cue in parsed if not cue.is_comment and (container or not cue.is_drawing)]
        except Exception as exc:
            raise SubtitleError("ASS 字幕无法解析。") from exc
        dialogue_count = len(re.findall(r"^Dialogue:", text, re.MULTILINE))
        if len(parsed) < dialogue_count:
            raise SubtitleError("ASS 字幕中有损坏的对话段落。")
    else:
        cues = _text_cues(text, suffix, container)
    records = []
    for index, (start, end, content) in enumerate(cues):
        start, end = start + offset_ms, end + offset_ms
        if container and not content.strip():
            # mov_text clear-screen packets and ASS drawing/empty events can
            # legitimately have zero duration and should not invalidate a track.
            continue
        original_start, original_end = start, end
        if container and end > start and end > 0 and start < duration_ms:
            start, end = max(0, start), min(duration_ms, end)
        if start < 0 or end <= start or end > duration_ms:
            raise SubtitleError(f"第 {index + 1} 段字幕超出视频范围或结束时间无效，请检查字幕偏移。")
        if not content.strip():
            continue
        for group, (value, language) in enumerate(_groups(content)):
            clipped = start != original_start or end != original_end
            record = {
                "id": _id(index, group, start, end, value), "start_ms": start, "end_ms": end,
                "text": value, "language": language, "source": "container" if container else "external",
                "review_status": "needs_review" if clipped else "unreviewed", "source_cue_index": index,
                "source_file": path.name, "offset_ms": offset_ms, "evidence_frame_ids": [],
            }
            if clipped:
                record.update(timing_clipped=True, source_start_ms=original_start, source_end_ms=original_end)
            records.append(record)
    if not records:
        raise SubtitleError("字幕文件不包含可用的字幕文字。")
    return sorted(records, key=lambda cue: (cue["start_ms"], cue["end_ms"], cue["source_cue_index"]))


def merge_ocr_frames(frames: list[dict], sample_interval_ms: int = 500) -> list[dict]:
    """Merge only adjacent identical observations, retaining all contributing frames.

    Empty frames, absent lines, and gaps in sampling all break runs. Each language
    remains a separate track. Timing is an observed interval, not inferred speech.
    Optional frame end_ms bounds final output at the asset/run boundary.
    """
    if isinstance(sample_interval_ms, bool) or not isinstance(sample_interval_ms, int) or sample_interval_ms <= 0:
        raise SubtitleError("字幕抽帧间隔必须是正整数毫秒。")
    active: dict[tuple[str, str, bool], dict] = {}
    records, previous_time = [], None
    seen_ids: set[str] = set()
    for frame in sorted(frames, key=lambda f: f["at_ms"]):
        at, frame_id = frame.get("at_ms"), frame.get("frame_id", frame.get("id"))
        if isinstance(at, bool) or not isinstance(at, int) or at < 0 or not isinstance(frame_id, str) or not frame_id:
            raise SubtitleError("字幕识别结果必须关联有效帧和整数时间戳。")
        if frame_id in seen_ids:
            raise SubtitleError("字幕识别结果包含重复的帧 ID。")
        seen_ids.add(frame_id)
        lines = frame.get("lines")
        if not isinstance(lines, list):
            raise SubtitleError("每个字幕识别帧必须包含 lines 列表，空画面使用空列表。")
        gap = previous_time is not None and at - previous_time > sample_interval_ms * 1.5
        if gap:
            records.extend(active.values())
            active = {}
        keys: set[tuple[str, str, bool]] = set()
        for line in lines:
            text, language = line.get("text"), line.get("language", "und")
            if not isinstance(text, str) or not isinstance(language, str):
                raise SubtitleError("字幕识别文字或语言类型无效。")
            text = text.strip()
            uncertain = line.get("uncertain", False)
            if not isinstance(uncertain, bool):
                raise SubtitleError("字幕 uncertain 字段必须是布尔值。")
            if not text and not uncertain:
                continue
            key = (text, language or "und", uncertain)
            if key in keys:
                continue
            keys.add(key)
            end = at + sample_interval_ms
            if isinstance(frame.get("end_ms"), int):
                end = min(end, frame["end_ms"])
            if end <= at:
                raise SubtitleError("字幕帧结束时间必须晚于开始时间。")
            if key not in active:
                active[key] = {
                    "id": _id(frame_id, *key), "start_ms": at, "end_ms": end,
                    "text": text, "language": key[1], "source": "embedded",
                    "review_status": "needs_review" if uncertain else "unreviewed",
                    "evidence_frame_ids": [], "timing_precision_ms": sample_interval_ms,
                }
            record = active[key]
            record["end_ms"] = end
            record["evidence_frame_ids"].append(frame_id)
        for key in list(active):
            if key not in keys:
                record = active.pop(key)
                record["end_ms"] = min(record["end_ms"], at)
                records.append(record)
        previous_time = at
    records.extend(active.values())
    return sorted(records, key=lambda record: (record["start_ms"], record["language"], record["id"]))


def candidate_frames(frames: list[dict]) -> list[dict]:
    """Deduplicate *consecutive* identical crops, with reversible timing/evidence map.

    Exact decoded-pixel hashes avoid treating near-identical subtitle glyphs as the
    same text. Callers must expand each response over source_frames before merging.
    A missing image raises, rather than silently discarding subtitle evidence.
    """
    from PIL import Image

    candidates: list[dict] = []
    previous_hash: str | None = None
    previous_at: int | None = None
    for frame in frames:
        path, at = Path(frame["path"]), frame["at_ms"]
        if not isinstance(at, int) or isinstance(at, bool) or at < 0:
            raise SubtitleError("候选字幕帧时间戳无效。")
        if previous_at is not None and at < previous_at:
            raise SubtitleError("候选字幕帧必须按时间排序。")
        with Image.open(path) as opened:
            pixels = opened.convert("RGB")
            digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
        source = {"frame_id": frame.get("id", frame.get("frame_id")), "at_ms": at}
        if "end_ms" in frame:
            source["end_ms"] = frame["end_ms"]
        if not source["frame_id"]:
            raise SubtitleError("候选字幕帧缺少 ID。")
        if candidates and digest == previous_hash:
            candidates[-1]["source_frames"].append(source)
        else:
            candidates.append({**frame, "source_frames": [source], "pixel_hash": digest})
        previous_hash, previous_at = digest, at
    return candidates
