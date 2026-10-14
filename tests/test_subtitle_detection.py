"""Local discovery tests use tiny files and injected synthetic media metadata."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scenerecall.library import Library
from scenerecall.main import create_app
from scenerecall.models import AssetInput
from scenerecall.subtitle_detection import detect_subtitle_sources, filename_identity


def metadata(*codecs):
    return {"duration_ms": 3000, "width": 16, "height": 16,
            "streams": [{"index": index + 1, "codec_type": "subtitle", "codec_name": codec,
                         "disposition": {"default": index == 0}}
                        for index, codec in enumerate(codecs)]}


def movie(directory, name="Show S01E02.mkv"):
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / name
    video.write_bytes(b"synthetic video source for filename matching")
    return video


def subtitle(directory, name, text="synthetic external words"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n", encoding="utf-8")
    return path


def test_text_container_wins_but_also_lists_recursive_sidecars(tmp_path):
    video = movie(tmp_path)
    external = subtitle(tmp_path / "subtitles" / "Chinese", "Show S01E02.chs.srt")
    result = detect_subtitle_sources(video, metadata=metadata("ass", "hdmv_pgs_subtitle"))
    assert result["recommended_mode"] == "container"
    assert result["recommended_stream_index"] == 1
    assert result["recommended_subtitle_path"] == str(external)
    assert result["scan_complete"]
    assert result["external_candidates"][0]["relative_path"] == "subtitles/Chinese/Show S01E02.chs.srt"
    assert result["external_candidates"][0]["language"] == "zh-Hans"


def test_external_matches_release_metadata_and_correct_episode_only(tmp_path):
    video = movie(tmp_path, "[VideoGroup] Example [02][Ma10p_1080p][x265_flac].mkv")
    correct = subtitle(tmp_path / "Subs", "[SubtitleGroup] Example - 02.chs.ASS")
    wrong_episode = subtitle(tmp_path / "Subs", "[SubtitleGroup] Example - 01.chs.ass")
    other_work = subtitle(tmp_path / "Subs", "Other Show - 02.chs.srt")
    result = detect_subtitle_sources(video, metadata=metadata())
    assert result["recommended_mode"] == "external"
    assert result["recommended_subtitle_path"] == str(correct)
    candidates = {Path(row["path"]): row for row in result["external_candidates"]}
    assert candidates[correct]["score"] == 90
    assert candidates[wrong_episode]["score"] == candidates[other_work]["score"] == 0


def test_bitmap_tracks_can_fall_back_to_external_text(tmp_path):
    video = movie(tmp_path)
    external = subtitle(tmp_path, "Show S01E02.srt")
    result = detect_subtitle_sources(video, metadata=metadata("hdmv_pgs_subtitle"))
    assert result["recommended_mode"] == "external"
    assert result["recommended_stream_index"] is None
    assert result["recommended_subtitle_path"] == str(external)
    assert not result["tracks"][0]["supported"]


def test_chinese_language_preference_and_full_dialogue_before_forced(tmp_path):
    video = movie(tmp_path)
    full = subtitle(tmp_path / "Subs", "Show S01E02.chs.srt")
    subtitle(tmp_path, "Show S01E02.en.srt")
    subtitle(tmp_path, "Show S01E02.chs.forced.srt")
    result = detect_subtitle_sources(video, metadata=metadata())
    assert result["recommended_subtitle_path"] == str(full)


def test_ambiguous_editions_and_unrelated_single_file_require_user_choice(tmp_path):
    video = movie(tmp_path)
    subtitle(tmp_path / "SubsA", "Show S01E02.chs.srt")
    subtitle(tmp_path / "SubsB", "Show S01E02.chs.srt")
    result = detect_subtitle_sources(video, metadata=metadata())
    assert result["recommended_mode"] == "auto"
    assert result["recommended_subtitle_path"] is None
    assert "多个匹配" in result["reason"]
    other = movie(tmp_path / "unrelated", "Movie.mkv")
    subtitle(other.parent, "Another movie.srt")
    assert detect_subtitle_sources(other, metadata=metadata())["recommended_mode"] == "auto"


def test_bare_episode_requires_matching_work_folder_and_season_must_agree(tmp_path):
    video = movie(tmp_path, "Show S01E02.mkv")
    bare = subtitle(tmp_path / "Show" / "Subs", "02.chs.srt")
    subtitle(tmp_path / "Subs", "02.en.srt")
    subtitle(tmp_path / "Subs", "Show S02E02.chs.srt")
    subtitle(tmp_path / "Show" / "Season 2", "02.chs.srt")
    result = detect_subtitle_sources(video, metadata=metadata())
    assert result["recommended_subtitle_path"] == str(bare)
    assert result["external_candidates"][0]["match"] == "episode"
    assert all(row["score"] == 0 for row in result["external_candidates"][1:])


@pytest.mark.parametrize(("stem", "expected"), [
    ("[Release] Show [02][1080p][x265_flac]", ("show", None, 2)),
    ("Show S01E02.chs&jpn", ("show", 1, 2)),
    ("Show Episode 2.en", ("show", None, 2)),
    ("Show 第十二集", ("show", None, 12)),
    ("Show 第三季 第十二话", ("show", 3, 12)),
    ("Movie 1080p", ("movie", None, None)),
    ("Movie [2020]", ("movie2020", None, None)),
])
def test_filename_identity(stem, expected):
    assert filename_identity(stem) == expected


def test_does_not_follow_symlink_files_or_folders_outside_video_directory(tmp_path):
    video = movie(tmp_path / "video")
    outside = subtitle(tmp_path / "outside", "Show S01E02.srt")
    (video.parent / "linked-subs").symlink_to(outside.parent, target_is_directory=True)
    (video.parent / "Show S01E02.srt").symlink_to(outside)
    assert detect_subtitle_sources(video, metadata=metadata())["external_candidates"] == []


def test_same_name_film_with_different_release_year_does_not_match(tmp_path):
    video = movie(tmp_path, "Movie [2020].mkv")
    subtitle(tmp_path / "Subs", "Movie [2021].chs.srt")
    result = detect_subtitle_sources(video, metadata=metadata())
    assert result["recommended_mode"] == "auto"
    assert result["external_candidates"][0]["score"] == 0


def test_incomplete_scans_report_limit_and_do_not_auto_select_external(tmp_path, monkeypatch):
    video = movie(tmp_path)
    subtitle(tmp_path / "Subs", "Show S01E02.srt")
    monkeypatch.setattr("scenerecall.subtitle_detection._MAX_SCAN_ENTRIES", 1)
    result = detect_subtitle_sources(video, metadata=metadata())
    assert not result["scan_complete"]
    assert result["recommended_mode"] == "auto"
    assert result["recommended_subtitle_path"] is None
    assert "上限" in result["warnings"][0]


def test_auto_import_recursively_uses_sidecar_and_respects_manual_override(tmp_path, monkeypatch):
    video = movie(tmp_path / "media")
    automatic = subtitle(video.parent / "Subs", "Show S01E02.chs.srt", "automatic words")
    manual = subtitle(video.parent / "different", "manual.srt", "manual words")
    monkeypatch.setattr("scenerecall.media.probe", lambda _path: metadata())
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(video)))
    assert asset["subtitle_mode"] == "external" and asset["subtitle_import_mode"] == "auto"
    assert library.subtitles(asset["id"])[0]["text"] == "automatic words"
    assert library.subtitles(asset["id"])[0]["source_file"] == automatic.name
    manual_library = Library(tmp_path / "manual-library")
    asset = manual_library.register(AssetInput(video_path=str(video), subtitle_mode="external", subtitle_path=str(manual)))
    assert manual_library.subtitles(asset["id"])[0]["text"] == "manual words"
    ocr_library = Library(tmp_path / "ocr-library")
    asset = ocr_library.register(AssetInput(video_path=str(video), subtitle_mode="embedded"))
    assert asset["subtitle_mode"] == "embedded" and ocr_library.subtitles(asset["id"]) == []


def test_auto_import_bitmap_sidecar_and_explicit_container_failure(tmp_path, monkeypatch):
    video = movie(tmp_path / "media")
    subtitle(video.parent / "Subs", "Show S01E02.srt")
    monkeypatch.setattr("scenerecall.media.probe", lambda _path: metadata("dvd_subtitle"))
    library = Library(tmp_path / "library")
    assert library.register(AssetInput(video_path=str(video)))["subtitle_mode"] == "external"
    with pytest.raises(ValueError, match="位图"):
        Library(tmp_path / "explicit").register(AssetInput(video_path=str(video), subtitle_mode="container"))


def test_auto_import_without_safe_match_does_not_run_ocr(tmp_path, monkeypatch):
    video = movie(tmp_path / "media")
    subtitle(video.parent, "Other Show S01E02.srt")
    monkeypatch.setattr("scenerecall.media.probe", lambda _path: metadata())
    library = Library(tmp_path / "library")
    asset = library.register(AssetInput(video_path=str(video)))
    assert asset["subtitle_mode"] == "embedded"
    assert "未检测到" in asset["subtitle_fallback_reason"]
    assert library.subtitles(asset["id"]) == []


def test_detection_endpoint_authentication_contract_and_local_discovery(tmp_path, monkeypatch):
    video = movie(tmp_path / "media")
    external = subtitle(video.parent / "Subs", "Show S01E02.srt")
    monkeypatch.setattr("scenerecall.media.probe", lambda _path: metadata())
    app = create_app(tmp_path / "library", start_worker=False, token="synthetic-token")
    with TestClient(app) as client:
        payload = {"video_path": str(video)}
        assert client.post("/api/media/subtitle-detection", json=payload).status_code == 401
        client.headers["Authorization"] = "Bearer synthetic-token"
        response = client.post("/api/media/subtitle-detection", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["recommended_subtitle_path"] == str(external)
        assert client.post("/api/media/subtitle-detection", json={**payload, "unknown": True}).status_code == 422
        assert client.post("/api/media/subtitle-detection", json={"video_path": str(video.parent / "missing")}).status_code == 404
