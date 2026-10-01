"""Synthetic container tracks only: no user media or licensed subtitle fixtures."""
import hashlib
import shutil
import subprocess

import pytest

from scenerecall import media
from scenerecall.subtitles import SubtitleError, parse_container_subtitles, parse_subtitles


ASS_HEADER = """[Script Info]
Title: Synthetic style preservation
Comment: Preserve header comments, including commas, and following style fields.
ScriptType: v4.00+
PlayResX: 640
PlayResY: 360

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Synthetic,Arial,30,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def dialogue(start, end, text):
    return f"Dialogue: 0,{start},{end},Synthetic,,0,0,0,,{text}\n"


def ffmpeg(*args):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for synthetic container tests")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)],
                   check=True, capture_output=True)


def make_dual_track(tmp_path, offset=0):
    ass = tmp_path / "styled.ass"
    ass.write_text(ASS_HEADER + dialogue("0:00:00.75", "0:00:01.25", r"{\i1}合成中文{\i0}\Nテスト")
                   + dialogue("0:00:01.50", "0:00:01.90", "Ending"), encoding="utf-8")
    srt = tmp_path / "english.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:01,300\nSynthetic English\n", encoding="utf-8")
    movie = tmp_path / "dual ; $(literal).mkv"
    ffmpeg("-f", "lavfi", "-i", "color=blue:s=32x32:r=10:d=2",
           "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono:d=2", "-i", ass, "-i", srt,
           "-map", "0", "-map", "1", "-map", "2", "-map", "3", "-c:v", "ffv1",
           "-c:a", "pcm_s16le", "-c:s", "copy", "-metadata:s:s:0", "language=chi",
           "-metadata:s:s:0", "title=Simplified", "-metadata:s:s:1", "language=eng",
           "-metadata:s:s:1", "title=English", "-disposition:s:0", "default",
           "-disposition:s:1", "forced", "-output_ts_offset", offset,
           "-avoid_negative_ts", "disabled", movie)
    return movie


@pytest.mark.parametrize("offset", [0, 5, -0.5])
def test_mkv_two_tracks_global_indices_styles_and_positive_negative_origin(tmp_path, offset):
    movie = make_dual_track(tmp_path, offset)
    before = hashlib.sha256(movie.read_bytes()).hexdigest()
    metadata = media.probe(movie)
    assert metadata["start_time_ms"] == round(offset * 1000)
    assert metadata["duration_ms"] == 2000
    tracks = media.subtitle_tracks(metadata)
    assert [(t["index"], t["codec"], t["language"], t["title"]) for t in tracks] == [
        (2, "ass", "chi", "Simplified"), (3, "subrip", "eng", "English")]
    assert tracks[0]["default"] and not tracks[0]["forced"]
    assert tracks[1]["forced"] and not tracks[1]["default"]
    assert all(t["supported"] for t in tracks)
    output = media.extract_subtitle_track(movie, 2, tmp_path / "selected.srt", metadata)
    assert output.suffix == ".ass"
    assert "Title: Synthetic style preservation" in output.read_text()
    assert "Style: Synthetic,Arial,30" in output.read_text()
    assert r"{\i1}合成中文{\i0}\Nテスト" in output.read_text()
    cues = parse_container_subtitles(output, metadata["duration_ms"])
    assert [(c["start_ms"], c["end_ms"], c["text"]) for c in cues] == [
        (750, 1250, "合成中文"), (750, 1250, "テスト"), (1500, 1900, "Ending")]
    assert all(c["source"] == "container" and c["review_status"] == "unreviewed" for c in cues)
    output = media.extract_subtitle_track(movie, 3, tmp_path / "english.ass", metadata)
    assert output.suffix == ".srt"
    english = parse_container_subtitles(output, 2000)
    assert [(c["start_ms"], c["end_ms"], c["text"]) for c in english] == [(500, 1300, "Synthetic English")]
    assert hashlib.sha256(movie.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob(".media-*"))


def test_mp4_mov_text_is_converted_and_clear_screen_packets_are_not_dialogue(tmp_path):
    srt = tmp_path / "input.srt"
    srt.write_text("1\n00:00:00,250 --> 00:00:00,750\nFirst\n\n"
                   "2\n00:00:01,100 --> 00:00:01,600\nLast\n", encoding="utf-8")
    movie = tmp_path / "mov-text.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=red:s=32x32:r=10:d=2", "-i", srt,
           "-c:v", "libx264", "-c:s", "mov_text", movie)
    info = media.probe(movie)
    assert media.subtitle_tracks(info)[0]["codec"] == "mov_text"
    path = media.extract_subtitle_track(movie, 1, tmp_path / "text.ass", info)
    assert path.suffix == ".srt"
    cues = parse_container_subtitles(path, info["duration_ms"])
    assert [(c["start_ms"], c["end_ms"], c["text"]) for c in cues] == [
        (250, 750, "First"), (1100, 1600, "Last")]


def test_no_subtitles_non_subtitle_index_and_stale_metadata_fail(tmp_path):
    movie = tmp_path / "no-subs.mkv"
    ffmpeg("-f", "lavfi", "-i", "color=red:s=32x32:r=10:d=1", "-c:v", "ffv1", movie)
    assert media.subtitle_tracks(media.probe(movie)) == []
    stale = {"start_time_ms": 5000, "streams": [{"index": 1, "codec_type": "subtitle", "codec_name": "ass"}]}
    for index in (0, 1, -1, True, "0"):
        with pytest.raises(media.MediaError):
            media.extract_subtitle_track(movie, index, tmp_path / "out.ass", stale)
    assert not (tmp_path / "out.ass").exists()


def test_track_detection_keeps_bitmap_and_unknown_visible_but_rejects_extraction(tmp_path, monkeypatch):
    streams = [{"index": i, "codec_type": "subtitle", "codec_name": codec}
               for i, codec in enumerate(["ass", "ssa", "srt", "subrip", "webvtt", "mov_text", "text",
                                          "hdmv_pgs_subtitle", "dvd_subtitle", "unknown"])]
    info = {"streams": streams, "start_time_ms": 0}
    tracks = media.subtitle_tracks(info)
    assert all(track["supported"] for track in tracks[:7])
    assert all(not track["supported"] and track["reason"] for track in tracks[7:])
    assert all(track["language"] == "und" and track["title"] == "" for track in tracks)
    source = tmp_path / "mock-local.mkv"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(media, "probe", lambda _: info)
    for index in (7, 8, 9):
        with pytest.raises(media.MediaError, match="外挂字幕"):
            media.extract_subtitle_track(source, index, tmp_path / "out.srt")


def test_bad_media_timeout_and_size_limit_do_not_publish_partial_output(tmp_path, monkeypatch):
    movie = tmp_path / "bad.mkv"
    movie.write_bytes(b"not a container")
    with pytest.raises(media.MediaError):
        media.extract_subtitle_track(movie, 0, tmp_path / "out.srt")
    info = {"start_time_ms": 0, "streams": [{"index": 0, "codec_type": "subtitle", "codec_name": "ass"}]}
    monkeypatch.setattr(media, "probe", lambda _: info)
    output = tmp_path / "previous.ass"
    output.write_text("previous version", encoding="utf-8")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", timeout)
        with pytest.raises(media.MediaError, match="超时"):
            media.extract_subtitle_track(movie, 0, output)
    assert output.read_text() == "previous version"

    def oversized(args, **kwargs):
        from pathlib import Path
        Path(args[-1]).write_bytes(b"x" * 128)

    monkeypatch.setattr(media, "_run", oversized)
    monkeypatch.setattr(media, "_MAX_SUBTITLE_BYTES", 128)
    with pytest.raises(media.MediaError, match="50 MB"):
        media.extract_subtitle_track(movie, 0, output)
    assert output.read_text() == "previous version"
    assert not list(tmp_path.glob(".media-*"))


def test_container_cue_boundaries_empty_and_mixed_drawing_are_explicit(tmp_path):
    path = tmp_path / "edges.ass"
    path.write_text(ASS_HEADER + dialogue("-0:00:00.50", "0:00:00.20", "Before zero")
                    + dialogue("0:00:01.80", "0:00:02.50", "Past end")
                    + dialogue("0:00:00.00", "0:00:00.00", "")
                    + dialogue("0:00:00.00", "0:00:00.00", r"{\p1}m 0 0 l 10 10")
                    + dialogue("0:00:00.50", "0:00:01.00", r"{\p1}m 0 0 l 10 10{\p0}Visible"),
                    encoding="utf-8")
    cues = parse_container_subtitles(path, 2000)
    assert [cue["text"] for cue in cues] == ["Before zero", "Visible", "Past end"]
    assert [(cue["start_ms"], cue["end_ms"]) for cue in cues] == [(0, 200), (500, 1000), (1800, 2000)]
    assert cues[0]["source_start_ms"] == -500 and cues[-1]["source_end_ms"] == 2500
    assert cues[0]["timing_clipped"] and cues[0]["review_status"] == "needs_review"
    with pytest.raises(SubtitleError):
        parse_subtitles(path, 2000)


@pytest.mark.parametrize("start,end", [("0:00:03.00", "0:00:04.00"),
                                        ("-0:00:02.00", "-0:00:01.00"),
                                        ("0:00:01.00", "0:00:01.00"),
                                        ("0:00:02.00", "0:00:01.00")])
def test_completely_outside_or_bad_visible_cues_fail_instead_of_partial_success(tmp_path, start, end):
    path = tmp_path / "bad.ass"
    path.write_text(ASS_HEADER + dialogue("0:00:00.50", "0:00:00.80", "Valid")
                    + dialogue(start, end, "Invalid"), encoding="utf-8")
    with pytest.raises(SubtitleError):
        parse_container_subtitles(path, 2000)


@pytest.mark.parametrize("suffix,negative,positive", [
    ("ass", "0:00:00.-30", "0:00:00.40"), ("srt", "00:00:00,-300", "00:00:00,400")])
def test_ffmpeg_negative_component_timestamps_are_normalized_without_global_shift(tmp_path, suffix, negative, positive):
    path = tmp_path / f"negative.{suffix}"
    text = (ASS_HEADER + dialogue(negative, positive, "Visible") if suffix == "ass"
            else f"1\n{negative} --> {positive}\nVisible\n")
    path.write_text(text, encoding="utf-8")
    media._normalize_subtitle_timestamps(path)
    cue = parse_container_subtitles(path, 2000)[0]
    assert (cue["start_ms"], cue["end_ms"], cue["source_start_ms"]) == (0, 400, -300)
    assert cue["timing_clipped"]


def test_container_empty_text_and_invalid_timestamp_fail_clearly(tmp_path):
    path = tmp_path / "empty.ass"
    path.write_text(ASS_HEADER + dialogue("0:00:00.00", "0:00:00.00", ""), encoding="utf-8")
    with pytest.raises(SubtitleError, match="可用"):
        parse_container_subtitles(path, 2000)
    path.write_text(ASS_HEADER + dialogue("broken", "0:00:01.00", "Visible"), encoding="utf-8")
    with pytest.raises(SubtitleError):
        parse_container_subtitles(path, 2000)


def test_external_ass_still_skips_non_dialogue_drawing_events_before_bounds_check(tmp_path):
    path = tmp_path / "external.ass"
    path.write_text(ASS_HEADER + dialogue("0:00:03.00", "0:00:03.00", r"{\p1}m 0 0 l 10 10")
                    + dialogue("0:00:00.50", "0:00:01.00", "Visible"), encoding="utf-8")
    cues = parse_subtitles(path, 2000)
    assert [cue["text"] for cue in cues] == ["Visible"]
    assert cues[0]["source"] == "external"


def test_srt_timestamp_normalization_does_not_rewrite_dialogue_arrow(tmp_path):
    path = tmp_path / "arrow.srt"
    path.write_text("1\n00:00:00,500 --> 00:00:01,000\nleft --> right\n", encoding="utf-8")
    media._normalize_subtitle_timestamps(path)
    assert parse_container_subtitles(path, 2000)[0]["text"] == "left --> right"
