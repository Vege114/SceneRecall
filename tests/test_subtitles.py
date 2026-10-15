
import pytest
from PIL import Image
from scenerecall.subtitles import SubtitleError, candidate_frames, merge_ocr_frames, parse_container_subtitles, parse_subtitles


def subtitle_file(tmp_path, text, suffix="srt"):
    path = tmp_path / f"sample.{suffix}"
    path.write_text(text, encoding="utf-8")
    return path


def test_external_required_missing_empty_bad_content_and_time_bounds(tmp_path):
    for path in (None, tmp_path / "absent.srt", subtitle_file(tmp_path, "")):
        with pytest.raises(SubtitleError):
            parse_subtitles(path, 5000)
    for text in ("not a subtitle", "1\n00:00:06,000 --> 00:00:07,000\nhello",
                 "1\n00:00:02,000 --> 00:00:01,000\nhello",
                 "1\n00:70:00,000 --> 00:71:00,000\nhello"):
        with pytest.raises(SubtitleError):
            parse_subtitles(subtitle_file(tmp_path, text), 5000)


def test_srt_bilingual_keeps_each_original_language_and_applies_offset(tmp_path):
    path = subtitle_file(tmp_path, "1\n00:00:01,000 --> 00:00:02,000\n<b>等一下！</b>\n待って！\n\n"
                                   "2\n00:00:03,000 --> 00:00:04,000\nHello &amp; goodbye\n")
    cues = parse_subtitles(path, 5000, offset_ms=250)
    assert [c["text"] for c in cues] == ["等一下！", "待って！", "Hello & goodbye"]
    assert [c["language"] for c in cues] == ["zh", "ja", "und"]
    assert cues[0]["start_ms"] == 1250
    assert cues[-1]["end_ms"] == 4250
    assert len({c["id"] for c in cues}) == 3
    assert all(c["source"] == "external" and not c["evidence_frame_ids"] for c in cues)
    assert path.read_text(encoding="utf-8").startswith("1\n")  # Original file is never rewritten.


def test_vtt_ids_settings_notes_and_wrapped_text(tmp_path):
    path = subtitle_file(tmp_path,
        "WEBVTT\n\nNOTE reviewer note\nnot a cue\n\nfirst\n00:01.000 --> 00:02.500 align:start\n"
        "<v Speaker>Hello\nworld &amp; friends</v>\n\n00:03.000 --> 00:04.000\n回来\n", "vtt")
    cues = parse_subtitles(path, 5000)
    assert len(cues) == 2
    assert cues[0]["text"] == "Hello\nworld & friends"
    assert cues[0]["start_ms"] == 1000


def test_ass_formatting_comments_and_bilingual_are_preserved(tmp_path):
    path = subtitle_file(tmp_path,
        "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\i1}回来{\\i0}\\N戻って！\n"
        "Comment: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,not visible\n", "ass")
    cues = parse_subtitles(path, 5000)
    assert [c["text"] for c in cues] == ["回来", "戻って！"]
    assert all(c["end_ms"] == 2000 for c in cues)


def ass_file(tmp_path, events):
    return subtitle_file(tmp_path, "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n" + events, "ass")


@pytest.mark.parametrize("parser", [parse_subtitles, parse_container_subtitles])
def test_ass_nonpositive_events_are_audited_without_inventing_dialogue_or_rewriting_source(tmp_path, parser):
    path = ass_file(tmp_path,
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,回来\\N戻って！\n"
        "Dialogue: 0,0:00:03.00,0:00:02.00,OP_fx,,0,0,0,,{\\k20}反向特效\n"
        "Dialogue: 0,0:00:04.00,0:00:04.00,OP_fx,,0,0,0,,瞬间特效\n"
        "Dialogue: 0,0:00:03.00,0:00:04.00,Default,,0,0,0,,有效台词\n")
    original = path.read_bytes()
    diagnostics = {}
    cues = parser(path, 5000, 100, diagnostics=diagnostics)
    assert [(c["start_ms"], c["end_ms"], c["text"]) for c in cues] == [
        (1100, 2100, "回来"), (1100, 2100, "戻って！"), (3100, 4100, "有效台词")]
    assert [c["source_cue_index"] for c in cues] == [0, 0, 3]
    assert diagnostics["ignored_ass_events"] == [
        {"source_cue_index": 1, "start_ms": 3000, "end_ms": 2000, "style": "OP_fx", "reason": "non_positive_duration"},
        {"source_cue_index": 2, "start_ms": 4000, "end_ms": 4000, "style": "OP_fx", "reason": "non_positive_duration"}]
    assert path.read_bytes() == original
    assert parser(path, 5000, 100) == cues


@pytest.mark.parametrize("parser", [parse_subtitles, parse_container_subtitles])
def test_ass_drawing_and_empty_events_ignore_timing_but_mixed_text_is_preserved(tmp_path, parser):
    path = ass_file(tmp_path,
        "Dialogue: 0,0:00:08.00,0:00:09.00,Default,,0,0,0,,{\\p1}m 0 0 l 10 10\n"
        "Dialogue: 0,0:00:08.00,0:00:09.00,Default,,0,0,0,,\n"
        "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\p1}m 0 0 l 10 10{\\p0}真实台词\n")
    cues = parser(path, 5000)
    assert [c["text"] for c in cues] == ["真实台词"]


@pytest.mark.parametrize("parser", [parse_subtitles, parse_container_subtitles])
def test_ass_without_any_playable_text_still_fails(tmp_path, parser):
    path = ass_file(tmp_path, "Dialogue: 0,0:00:02.00,0:00:01.00,OP_fx,,0,0,0,,反向特效\n")
    with pytest.raises(SubtitleError, match="不包含可用"):
        parser(path, 5000)


@pytest.mark.parametrize("timing,offset", [("0:00:04.00,0:00:06.00", 0), ("0:00:00.00,0:00:01.00", -100)])
def test_ass_visible_out_of_range_or_wrong_offset_still_fails(tmp_path, timing, offset):
    path = ass_file(tmp_path, f"Dialogue: 0,{timing},Default,,0,0,0,,有效台词但时间错误\n")
    with pytest.raises(SubtitleError, match="超出视频范围"):
        parse_subtitles(path, 5000, offset)


def frame(at, *lines, end=None):
    result = {"frame_id": f"f{at}", "at_ms": at, "lines": [
        {"text": line, "language": "zh", "uncertain": False} if isinstance(line, str) else line
        for line in lines]}
    if end is not None:
        result["end_ms"] = end
    return result


def test_ocr_empty_frames_and_sampling_gaps_break_repeated_lines():
    cues = merge_ocr_frames([frame(0, "你好"), frame(500, "你好"), frame(1000),
                             frame(1500, "你好"), frame(3000, "你好", end=3200)])
    assert [(c["start_ms"], c["end_ms"]) for c in cues] == [(0, 1000), (1500, 2000), (3000, 3200)]
    assert cues[0]["evidence_frame_ids"] == ["f0", "f500"]
    assert len({c["id"] for c in cues}) == 3


def test_ocr_bilingual_and_uncertain_empty_text_are_separate():
    ja = {"text": "こんにちは", "language": "ja", "uncertain": False}
    unknown = {"text": "", "language": "und", "uncertain": True}
    cues = merge_ocr_frames([frame(0, "你好", ja), frame(500, ja), frame(1000, unknown)])
    assert {(c["text"], c["start_ms"], c["end_ms"]) for c in cues} == {
        ("你好", 0, 500), ("こんにちは", 0, 1000), ("", 1000, 1500)}
    assert next(c for c in cues if not c["text"])["review_status"] == "needs_review"


def test_ocr_same_text_across_shot_boundary_remains_one_observed_run():
    # Shot boundaries don't prove that a continuously visible subtitle changed.
    cues = merge_ocr_frames([{**frame(0, "再见"), "shot_id": "a"},
                             {**frame(500, "再见"), "shot_id": "b"}])
    assert len(cues) == 1 and cues[0]["end_ms"] == 1000


def test_ocr_invalid_records_fail_instead_of_inventing_no_subtitles():
    for frames in ([{"frame_id": "f", "at_ms": 0}], [frame(0), frame(0)],
                   [frame(0, {"text": "hi", "uncertain": "false"})]):
        with pytest.raises(SubtitleError):
            merge_ocr_frames(frames)


def test_visual_candidate_dedup_keeps_reversible_evidence_and_repeat_after_gap(tmp_path):
    items = []
    for i, color in enumerate(("black", "black", "white", "black")):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (24, 12), color=color).save(path)
        items.append({"id": f"f{i}", "path": str(path), "at_ms": i * 500})
    candidates = candidate_frames(items)
    assert len(candidates) == 3
    assert candidates[0]["source_frames"] == [{"frame_id": "f0", "at_ms": 0},
                                               {"frame_id": "f1", "at_ms": 500}]
    assert candidates[-1]["source_frames"] == [{"frame_id": "f3", "at_ms": 1500}]
