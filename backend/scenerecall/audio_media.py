"""Local PCM extraction on SceneRecall's existing presentation timeline."""
from __future__ import annotations

import math
import subprocess
import wave
from array import array
from pathlib import Path

from . import media

PREPROCESS_VERSION = "pcm16-mono-16000-presentation-v1"
SAMPLE_RATE = 16000


def audio_tracks(metadata: dict) -> list[dict]:
    origin = metadata.get("start_time_ms", 0)
    result = []
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        tags = stream.get("tags", {})
        result.append({"index": stream["index"], "codec": stream.get("codec_name", "unknown"),
                       "language": tags.get("language", "und"), "title": tags.get("title", ""),
                       "channels": stream.get("channels"), "sample_rate": stream.get("sample_rate"),
                       "default": bool(stream.get("disposition", {}).get("default")),
                       "start_ms": round(float(stream.get("start_time", 0)) * 1000) - origin})
    return result


def extract_audio(source: Path, output: Path, stream_index: int, start_ms: int, end_ms: int,
                  metadata: dict | None = None, *, checkpoint=None) -> dict:
    """Pad timestamp gaps, trim after PTS normalization, and produce exact sample count.

    Do not reset the audio stream to its first packet: that would remove a real
    delay relative to video. All positions use the same origin as media.probe.
    """
    metadata = metadata or media.probe(source)
    if stream_index not in {track["index"] for track in audio_tracks(metadata)}:
        raise ValueError("所选音轨不存在")
    if start_ms < 0 or end_ms <= start_ms or end_ms > metadata["duration_ms"]:
        raise ValueError("音频时间范围无效")
    if checkpoint:
        checkpoint()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.wav")
    origin = float(metadata.get("start_time_seconds", metadata.get("start_time_ms", 0) / 1000))
    duration = (end_ms - start_ms) / 1000
    filters = (f"asetpts=PTS-({origin:.9f})/TB,aresample={SAMPLE_RATE}:async=1:first_pts=0,"
               f"atrim=start={start_ms / 1000:.6f}:end={end_ms / 1000:.6f},"
               f"asetpts=PTS-STARTPTS,apad=whole_dur={duration:.6f}")
    process = subprocess.Popen([media._tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-threads", "1",
                                "-copyts", "-i", str(source), "-map", f"0:{stream_index}", "-vn",
                                "-af", filters, "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                                "-t", f"{duration:.6f}", str(temporary)], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        while True:
            try:
                code = process.wait(timeout=.2)
                break
            except subprocess.TimeoutExpired:
                if checkpoint:
                    checkpoint()
        if code:
            raise media.MediaError("音轨提取失败")
        if checkpoint:
            checkpoint()
        with wave.open(str(temporary)) as wav:
            if wav.getnframes() != (end_ms - start_ms) * SAMPLE_RATE // 1000:
                raise media.MediaError("提取音频长度与原片时间轴不一致")
        temporary.replace(output)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)
    return {"start_ms": start_ms, "end_ms": end_ms, "stream_index": stream_index,
            "sample_rate": SAMPLE_RATE, "channels": 1, "origin_ms": metadata.get("start_time_ms", 0),
            "preprocessing": PREPROCESS_VERSION}


def crop_pcm(source: Path, output: Path, start_ms: int, end_ms: int) -> None:
    with wave.open(str(source), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("声音特征仅接受统一的 16kHz 单声道 PCM16")
        start, end = round(start_ms * SAMPLE_RATE / 1000), round(end_ms * SAMPLE_RATE / 1000)
        if start < 0 or end <= start or end > wav.getnframes():
            raise ValueError("声音片段越过提取范围")
        wav.setpos(start)
        data = wav.readframes(end - start)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.wav")
    with wave.open(str(temporary), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(data)
    temporary.replace(output)


def pcm_samples(path: Path) -> array:
    import sys
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError("声音样本必须为 16kHz 单声道 PCM16")
        values = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        values.byteswap()
    return values


def quality(path: Path) -> dict:
    samples = pcm_samples(path)
    count = len(samples)
    rms = math.sqrt(sum(float(x) ** 2 for x in samples) / max(1, count)) / 32768
    clipping = sum(abs(x) >= 32760 for x in samples) / max(1, count)
    reasons = []
    if count < SAMPLE_RATE:
        reasons.append("too_short")
    if rms < .001:
        reasons.append("silence_or_too_quiet")
    if clipping > .05:
        reasons.append("clipped")
    return {"duration_ms": round(count / SAMPLE_RATE * 1000), "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
            "clipping_fraction": clipping, "usable": not reasons, "reasons": reasons}
