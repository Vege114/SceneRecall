"""Local subtitle discovery with conservative filename and episode matching.

Discovery only reads media metadata and directory entries. It never invokes a
model, extracts tracks, or parses every subtitle in a collection.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

SUBTITLE_SUFFIXES = {".srt", ".vtt", ".ass"}
_MAX_SCAN_ENTRIES = 50_000
_MAX_CANDIDATES = 2_000
_QUALITY_NUMBERS = {480, 576, 720, 1080, 1440, 2160, 4320}
_LANGUAGES = {
    "chs": "zh-Hans", "sc": "zh-Hans", "zh-hans": "zh-Hans", "简": "zh-Hans",
    "简体": "zh-Hans", "简中": "zh-Hans", "gb": "zh-Hans",
    "cht": "zh-Hant", "tc": "zh-Hant", "zh-hant": "zh-Hant", "繁": "zh-Hant",
    "繁体": "zh-Hant", "繁中": "zh-Hant", "big5": "zh-Hant",
    "zh": "zh", "zho": "zh", "chi": "zh", "cn": "zh", "中文": "zh",
    "中日": "zh", "中英": "zh", "双语": "zh",
    "ja": "ja", "jp": "ja", "jpn": "ja", "日文": "ja",
    "en": "en", "eng": "en", "英文": "en",
}
_EXTRA_SUFFIXES = {"utf8", "utf-8", "sdh", "cc", "forced", "signs", "dialogue", "full"}
_SUFFIX_PART = "|".join(re.escape(value) for value in sorted(set(_LANGUAGES) | _EXTRA_SUFFIXES, key=len, reverse=True))
_LANGUAGE_SUFFIX = re.compile(
    rf"(?:[._\s-]+|\[|\()((?:{_SUFFIX_PART})(?:[&+._\s-]+(?:{_SUFFIX_PART}))*)[\])]?$", re.I)
_TECHNICAL = re.compile(
    r"(?:^|[\s._-])(?:\d{3,4}p|ma?\d+p?|x26[45]|h26[45]|hevc|avc|flac|aac|bdrip|blu-?ray|web-?dl)"
    r"(?=$|[\s._-])", re.I)


def _key(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", value).casefold() if char.isalnum())


def _language_stem(stem: str) -> tuple[str, str, bool]:
    stem = unicodedata.normalize("NFKC", stem)
    languages = []
    forced = False
    while match := _LANGUAGE_SUFFIX.search(stem):
        label = match[1].casefold()
        # Try complete labels first, including zh-hans / zh-hant.
        tags = re.findall(_SUFFIX_PART, label, re.I)
        languages.extend(_LANGUAGES[tag] for tag in tags if tag in _LANGUAGES)
        forced = forced or any(tag in {"forced", "signs"} for tag in tags)
        stem = stem[:match.start()].rstrip(" ._-[(")
    language = next((value for value in languages if value.startswith("zh")),
                    languages[0] if languages else "und")
    return stem, language, forced


def _chinese_number(value: str) -> int | None:
    if value.isascii() and value.isdigit():
        return int(value)
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    total, current = 0, 0
    for char in value:
        if char in digits:
            current = digits[char]
        elif char in "十百千":
            total += (current or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            current = 0
        else:
            return None
    return total + current


def filename_identity(stem: str) -> tuple[str, int | None, int | None]:
    """Ignore release groups / codec tags without confusing resolutions for episodes."""
    stem, _, _ = _language_stem(stem)
    episode = season = None
    seasonal = re.search(r"(?:^|[\s._\-\[\](])S(\d{1,3})[\s._-]*E(\d{1,4})(?:v\d+)?(?=$|[^\d])", stem, re.I)
    explicit = re.search(r"(?:^|[\s._\-\[\](])(?:EP?|Episode)[\s._-]*(\d{1,4})(?:v\d+)?(?=$|[^\d])", stem, re.I)
    chinese = re.search(r"第\s*([\d零〇一二两三四五六七八九十百千]+)\s*[集话話]", stem)
    marker = None
    if seasonal:
        marker, season, episode = seasonal, int(seasonal[1]), int(seasonal[2])
    elif explicit:
        marker, episode = explicit, int(explicit[1])
    elif chinese:
        marker, episode = chinese, _chinese_number(chinese[1])
    else:
        bracketed = list(re.finditer(r"[\[(]\s*(\d{1,4})(?:v\d+)?\s*[\])]", stem, re.I))
        metadata_free = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", stem).strip()
        trailing = re.search(r"(?:^|[\s._-])(\d{1,4})(?:v\d+)?\s*$", metadata_free, re.I)
        marker = next((value for value in bracketed if int(value[1]) not in _QUALITY_NUMBERS
                       and not 1900 <= int(value[1]) <= 2099), None)
        if marker is None and trailing and int(trailing[1]) not in _QUALITY_NUMBERS and not 1900 <= int(trailing[1]) <= 2099:
            # Find the equivalent position in the original filename before tags.
            marker = re.search(rf"(?:^|[\s._-])0*{int(trailing[1])}(?:v\d+)?(?=\s*(?:\[|\(|$))", stem, re.I)
        if marker:
            episode = int(marker[1]) if marker.lastindex else int(trailing[1])
    if marker:
        stem = stem[:marker.start()] + " " + stem[marker.end():]
    chinese_season = re.search(r"第\s*([\d零〇一二两三四五六七八九十百千]+)\s*季", stem)
    if chinese_season:
        season = season if season is not None else _chinese_number(chinese_season[1])
        stem = stem[:chinese_season.start()] + " " + stem[chinese_season.end():]
    # A release year can distinguish same-named films; it is not disposable
    # codec metadata and must not create a cross-remake filename match.
    stem = re.sub(r"\[[^\]]*\]", lambda value: value[0][1:-1]
                  if re.fullmatch(r"\[\s*(?:19|20)\d{2}\s*\]", value[0]) else " ", stem)
    stem = re.sub(r"\((?:[^)]*(?:\d{3,4}p|x26[45]|HEVC|AVC|FLAC|AAC|Blu-?ray|BDRip)[^)]*)\)", " ", stem, flags=re.I)
    stem = _TECHNICAL.sub(" ", stem)
    return _key(stem), season, episode


def _candidate(video: Path, subtitle: Path, root: Path) -> dict:
    stem, language, forced = _language_stem(subtitle.stem)
    video_stem = unicodedata.normalize("NFKC", video.stem)
    video_title, video_season, video_episode = filename_identity(video.stem)
    title, season, episode = filename_identity(subtitle.stem)
    ancestors = subtitle.relative_to(root).parts[:-1]
    folder_seasons = []
    for part in ancestors:
        folder_season = re.search(r"(?:^|[\s._-])(?:Season[\s._-]*|S)(\d{1,3})(?=$|[\s._-])", part, re.I)
        chinese_season = re.search(r"第\s*([\d零〇一二两三四五六七八九十百千]+)\s*季", part)
        if folder_season:
            folder_seasons.append(int(folder_season[1]))
        elif chinese_season:
            folder_seasons.append(_chinese_number(chinese_season[1]))
    score, match = 0, "unmatched"
    # Full release filenames provide stronger evidence than an episode number.
    if _key(stem) == _key(video_stem):
        score, match = 100, "exact"
    elif (video_title and title == video_title and episode == video_episode
          and (season is None or video_season is None or season == video_season)):
        score, match = 90, "filename"
    elif (not title and episode is not None and episode == video_episode
          and (season is None or video_season is None or season == video_season)
          and (video_season is None or all(value == video_season for value in folder_seasons))
          and any(filename_identity(part)[0] == video_title and video_title
                  for part in ancestors)):
        score, match = 75, "episode"
    return {"path": str(subtitle), "name": subtitle.name,
            "relative_path": str(subtitle.relative_to(root)), "match": match,
            "language": language, "score": score, "forced": forced}


def _scan(video: Path) -> tuple[list[dict], bool, list[str]]:
    root = video.parent
    candidates, warnings = [], []
    complete = True
    visited = 0

    def inaccessible(_error):
        nonlocal complete
        complete = False

    for directory, subdirectories, filenames in os.walk(root, onerror=inaccessible, followlinks=False):
        # Never follow directory aliases into another collection or a loop.
        subdirectories[:] = sorted(name for name in subdirectories if not (Path(directory) / name).is_symlink())
        visited += len(subdirectories) + len(filenames)
        if visited > _MAX_SCAN_ENTRIES:
            complete = False
            warnings.append("目录项目过多，自动扫描已达到上限；可用文件选择器选择尚未扫描的字幕。")
            break
        for name in sorted(filenames):
            subtitle = Path(directory) / name
            if subtitle.suffix.lower() not in SUBTITLE_SUFFIXES or subtitle.is_symlink():
                continue
            try:
                if not subtitle.is_file():
                    continue
                subtitle = subtitle.resolve(strict=True)
                if not subtitle.is_relative_to(root):
                    continue
            except OSError:
                complete = False
                continue
            candidates.append(_candidate(video, subtitle, root))
            if len(candidates) >= _MAX_CANDIDATES:
                complete = False
                warnings.append("字幕候选过多，自动扫描已达到上限；可用文件选择器继续选择。")
                break
        if len(candidates) >= _MAX_CANDIDATES:
            break
    if not complete and not warnings:
        warnings.append("部分子文件夹无法读取，请用文件选择器手动选择所需字幕。")
    languages = {"zh-Hans": 0, "zh": 1, "zh-Hant": 2, "ja": 3, "en": 4, "und": 5}
    candidates.sort(key=lambda row: (-row["score"], row["forced"], languages.get(row["language"], 5),
                                     len(Path(row["relative_path"]).parts), row["relative_path"].casefold()))
    return candidates, complete, warnings


def _recommend_external(candidates: list[dict], complete: bool) -> dict | None:
    # An incomplete directory listing cannot establish a unique external match.
    if not complete or not candidates or candidates[0]["score"] <= 0:
        return None
    top = candidates[0]
    tied = [row for row in candidates if row["score"] == top["score"] and row["forced"] == top["forced"]
            and row["language"] == top["language"]]
    # Multiple same-language editions require the user's choice; path order
    # alone is not evidence that one subtitle belongs to the video.
    return top if len(tied) == 1 else None


def detect_subtitle_sources(path: Path, *, metadata: dict | None = None) -> dict:
    from .library import recommended_subtitle_track
    from .media import probe, subtitle_tracks

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("视频路径必须指向文件")
    metadata = metadata if metadata is not None else probe(source)
    tracks = subtitle_tracks(metadata)
    track = recommended_subtitle_track(tracks)
    candidates, complete, warnings = _scan(source)
    external = _recommend_external(candidates, complete)
    if track:
        mode, reason = "container", "检测到可提取文字的字幕轨，已自动选择字幕轨；仍可切换来源或轨道。"
    elif external:
        mode, reason = "external", "已在视频所在文件夹及子文件夹中匹配外挂字幕；仍可更换文件或字幕来源。"
    elif any(row["score"] for row in candidates):
        mode, reason = "auto", "存在多个匹配版本或扫描未完成，请选择这集使用的外挂字幕。"
    elif tracks:
        mode, reason = "auto", "仅检测到位图或不支持的字幕轨，未找到匹配的外挂文字字幕，请选择字幕文件。"
    else:
        mode, reason = "auto", "未检测到可提取字幕轨或匹配的外挂字幕，请选择字幕文件；画面有烧录字幕时可选择 OCR。"
    return {"tracks": tracks, "recommended_stream_index": track["index"] if track else None,
            "duration_ms": metadata["duration_ms"], "external_candidates": candidates,
            "recommended_mode": mode,
            "recommended_subtitle_path": external["path"] if external else None,
            "reason": reason, "scan_complete": complete, "warnings": warnings}
