import hashlib
import shutil

from fastapi.testclient import TestClient

from scenerecall.library import read_json
from scenerecall.main import create_app
from scenerecall.media import extract_frame
from scenerecall.models import AssetInput
from scenerecall.record_previews import preserve_frames
from test_application import synthetic  # noqa: F401


def test_timeline_preview_and_favorite_survive_source_removal(tmp_path, synthetic):  # noqa: F811
    video, subtitle = synthetic
    source = tmp_path / "original.mp4"
    shutil.copyfile(video, source)
    app = create_app(tmp_path / "library", token="timeline-test", start_worker=False)
    library = app.state.library
    asset = library.register(AssetInput(video_path=str(source), subtitle_mode="external", subtitle_path=str(subtitle)))
    aid = asset["id"]
    client = TestClient(app, headers={"Authorization": "Bearer timeline-test"})
    preview = f"/api/assets/{aid}/preview?at_ms=0"
    picture = client.get(preview)
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    assert picture.content.startswith(b"\xff\xd8")
    assert client.get(f"/api/assets/{aid}/preview?at_ms={asset['duration_ms']}").status_code == 400
    pinned = library.asset_dir(aid) / "preserved-frames"
    assert not pinned.exists()
    library.annotate(aid, {"record_id": library.subtitles(aid)[0]["id"], "favorite": True})
    source.unlink()
    row = client.get("/api/collections").json()[0]
    assert row["thumbnail_url"] == preview
    assert client.get(preview).content == picture.content
    assert client.get(f"/api/assets/{aid}/preview?at_ms=1500").status_code == 404


def test_human_still_keeps_original_bytes_in_separate_directory(tmp_path, synthetic):  # noqa: F811
    source, _ = synthetic
    app = create_app(tmp_path / "library", token="still-test", start_worker=False)
    library = app.state.library
    aid = library.register(AssetInput(video_path=str(source), subtitle_mode="embedded"))["id"]
    directory = library.asset_dir(aid)
    frame = extract_frame(source, 400, directory / "frames/still.jpg")
    library.register_frame(aid, frame)
    original = (directory / "frames/still.jpg").read_bytes()
    preserve_frames(library, aid, ["still"])
    entry = read_json(directory / "frames/registry.json")["still"]
    assert entry["pinned_sha256"] == hashlib.sha256(original).hexdigest()
    (directory / "frames/still.jpg").unlink()
    assert library.frame_bytes(aid, "still") == original


def test_bookmark_only_pins_the_displayed_picture(tmp_path, synthetic):  # noqa: F811
    source, _ = synthetic
    app = create_app(tmp_path / "library", token="cover-test", start_worker=False)
    library = app.state.library
    aid = library.register(AssetInput(video_path=str(source), subtitle_mode="embedded"))["id"]
    directory = library.asset_dir(aid)
    frames = [extract_frame(source, ms, directory / f"frames/{name}.jpg") for ms, name in [(300, "first"), (900, "hit")]]
    library.register_frames(aid, frames)
    library.save_observation(aid, {"id": "record", "asset_id": aid, "run_id": "test", "window_id": "window",
                                   "shot_id": "shot", "start_ms": 0, "end_ms": 1000, "summary": "镜头",
                                   "evidence_frame_ids": ["first", "hit"]})
    saved = library.annotate(aid, {"record_id": "record", "favorite": True, "preview_frame_id": "hit"})
    assert saved["preview_frame_id"] == "hit" and saved["preview_at_ms"] == 900
    edited = library.annotate(aid, {"record_id": "record", "favorite": True, "note": "更新笔记"})
    assert edited["preview_frame_id"] == "hit" and edited["preview_at_ms"] == 900
    assert [file.name for file in (directory / "preserved-frames").iterdir()] == ["hit.jpg"]
