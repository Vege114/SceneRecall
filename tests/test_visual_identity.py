"""Identity invariants with synthetic images/backends, not accuracy benchmarks."""
from copy import deepcopy
import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import pytest

from scenerecall.characters import CharacterStore
from scenerecall.library import Library, atomic_json, read_json
from scenerecall.visual_identity import (IdentityUnavailable, ReferenceGallery, VisualIdentityPipeline,
                                        match_candidates, validate_visual_artifacts)
from scenerecall.visual_identity_api import register_visual_identity_routes


def picture(red=90, size=(128, 192)):
    image = Image.new("RGB", size, (red, 40, 80))
    draw = ImageDraw.Draw(image)
    # Deterministic line detail gives the synthetic image nonzero sharpness.
    for index in range(8, min(size), 9):
        draw.line((index, 0, index, size[1]), fill=(red, 220, 200), width=2)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def visual(tmp_path):
    library = Library(tmp_path / "library")
    asset = {"schema_version": "1.0", "id": "asset_one", "work_id": "work_one", "title": "Synthetic Anime",
             "series": "Synthetic Series", "kind": "animation", "season": 1, "episode": 1,
             "created_at": "2026-10-15T00:00:00Z", "duration_ms": 10000,
             "version": "test", "fingerprint": "synthetic-video"}
    atomic_json(library.asset_dir(asset["id"]) / "manifest.json", asset)
    characters = CharacterStore(library)
    return library, characters, ReferenceGallery(library)


def frame(library, identifier="frame_one", red=90, size=(128, 192), at_ms=200):
    path = library.asset_dir("asset_one") / "frames" / f"{identifier}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(picture(red, size))
    result = {"id": identifier, "path": str(path), "at_ms": at_ms}
    library.register_frame("asset_one", result)
    return result


class FakeCCIP:
    """Known deterministic feature/metric double, no neural inference claims."""
    model_version = "synthetic-ccip-v1"

    def __init__(self, boxes=None):
        self.boxes = boxes
        self.feature_calls = 0
        self.detection_calls = 0
        self.distance_calls = 0

    def detect(self, path):
        self.detection_calls += 1
        with Image.open(path) as image:
            boxes = (self.boxes if self.boxes is not None and "frames" in Path(path).parts
                     else [[0, 0, image.width, image.height]])
        return [{"box": box, "label": "person", "detection_score": .9} for box in boxes]

    def feature(self, path):
        self.feature_calls += 1
        with Image.open(path) as image:
            return [image.getpixel((0, 0))[0] / 255]

    def distances(self, query, features):
        self.distance_calls += 1
        return [abs(query[0] - feature[0]) for feature in features]


def add_role(characters, name="Aoi"):
    return characters.create("asset_one", {"name": name})["id"]


def add_reference(gallery, identity, red=90):
    return gallery.add("asset_one", identity, picture(red), confirmed=True, single_character=True)


def test_first_appearance_independent_reference_and_no_fabricated_presence(visual):
    library, characters, gallery = visual
    identity = add_role(characters)
    external = gallery.add("asset_one", identity, picture(), source_kind="external_cached",
                           source_url="https://example.org/licensed-image.png", confirmed=True, single_character=True)
    catalog = characters.catalog("asset_one")
    assert catalog["profiles"][0]["appearance_count"] == 0
    assert catalog["profiles"][0]["representative"] is None
    assert characters.context("asset_one")["profiles"][0]["id"] == identity
    assert external["source"]["url"] and external["reference_only"]
    backend = FakeCCIP()
    result = VisualIdentityPipeline(library, backend).run("asset_one", [frame(library)], {"enabled": True})
    assert result["status"] == "completed"
    match = result["regions"][0]["match"]
    assert match["status"] == "candidate" and match["character_id"] is None
    assert match["candidate_character_id"] == identity
    assert "match_confidence" not in match
    assert result["references"][0]["reference_only"]
    assert characters.catalog("asset_one")["profiles"][0]["appearance_count"] == 0


def test_first_frame_without_reference_still_produces_unknown_regions(visual):
    library, _characters, _gallery = visual
    result = VisualIdentityPipeline(library, FakeCCIP()).run("asset_one", [frame(library)], {"enabled": True})
    assert result["regions"][0]["match"]["status"] == "unknown"
    assert result["references"] == [] and result["degraded_reason"]


def test_complete_roster_searched_before_top_k_more_than_sixteen(visual):
    library, characters, gallery = visual
    first = None
    for index in range(22):
        identity = add_role(characters, f"Role {index}")
        add_reference(gallery, identity, 20 + index * 10)
        if first is None:
            first = identity
    # The oldest role must survive every newer update and a tiny Top-K prompt.
    result = VisualIdentityPipeline(library, FakeCCIP()).run("asset_one", [frame(library, red=20)],
                                                            {"enabled": True, "top_k": 1})
    assert result["regions"][0]["match"]["candidates"][0]["character_id"] == first
    assert [profile["id"] for profile in result["selected_profiles"]] == [first]
    assert result["metrics"]["feature_extractions"] == 22  # Same exact frame/first reference shares a feature.


def test_multiple_people_are_cropped_with_frame_local_ids_never_cross_shot_merged(visual):
    library, characters, gallery = visual
    identity = add_role(characters)
    add_reference(gallery, identity)
    frames = [frame(library, "frame_one", size=(256, 192)), frame(library, "frame_two", size=(256, 192), at_ms=900)]
    backend = FakeCCIP(boxes=[[0, 0, 128, 192], [128, 0, 256, 192]])
    result = VisualIdentityPipeline(library, backend).run("asset_one", frames, {"enabled": True})
    regions = result["regions"]
    assert [row["local_id"] for row in regions] == ["P1", "P2", "P1", "P2"]
    assert len({row["id"] for row in regions}) == 4
    assert all(row["quality"]["person_count"] == 1 for row in regions)
    assert all(Image.open(row["crop_path"]).size == (128, 192) for row in regions)


def test_overlapping_detection_crop_does_not_enter_ccip(visual):
    library, characters, gallery = visual
    add_reference(gallery, add_role(characters))
    backend = FakeCCIP(boxes=[[0, 0, 256, 192], [80, 0, 176, 192]])
    result = VisualIdentityPipeline(library, backend).run("asset_one", [frame(library, size=(256, 192))], {"enabled": True})
    assert all(row["match"]["status"] == "unknown" for row in result["regions"])
    assert all("multiple_or_no_people" in row["quality"]["reasons"] for row in result["regions"])
    assert backend.feature_calls == 1  # reference only


def test_multireference_side_view_outfit_and_uncalibrated_distance_remains_candidate():
    references = [{"id": "front", "character_id": "A"}, {"id": "side", "character_id": "A"},
                  {"id": "outfit", "character_id": "A"}, {"id": "other", "character_id": "B"}]
    result = match_candidates([.4, .05, .08, .2], references, model_version="ccip-v1")
    assert result["candidate_character_id"] == "A"
    assert result["distance"] == pytest.approx(.065)
    assert result["margin"] == pytest.approx(.135)
    assert result["candidates"][0]["reference_ids"] == ["side", "outfit"]
    assert result["status"] == "candidate" and result["character_id"] is None
    calibration = {"model_version": "ccip-v1", "dataset_id": "held-out", "evaluation_id": "eval_1",
                   "aggregation": "nearest-two-mean-v1", "max_distance": .1, "min_margin": .05}
    assert match_candidates([.4, .05, .08, .2], references, model_version="ccip-v1",
                            calibration=calibration)["status"] == "confirmed"
    assert match_candidates([.4, .3, .4, .2], references, model_version="ccip-v1",
                            calibration=calibration)["status"] == "unknown"
    assert match_candidates([.4, .05, .08, .2], references, model_version="different-model",
                            calibration=calibration)["status"] == "candidate"


def test_unknown_quality_and_unconfirmed_samples_never_pollute_trusted_gallery(visual):
    library, characters, gallery = visual
    identity = add_role(characters)
    candidate = gallery.add("asset_one", identity, picture(), single_character=True)
    assert candidate["status"] == "candidate" and gallery.trusted("asset_one") == []
    with pytest.raises(ValueError, match="质量|清晰度"):
        gallery.add("asset_one", identity, picture(size=(32, 32)), confirmed=True, single_character=True)
    with pytest.raises(ValueError, match="单角色"):
        gallery.add("asset_one", identity, picture(), confirmed=True)
    result = VisualIdentityPipeline(library, FakeCCIP()).run("asset_one", [frame(library)], {"enabled": True})
    assert result["regions"][0]["match"]["status"] == "unknown"
    assert len(gallery.list("asset_one")["references"]) == 1


def test_reference_revocation_has_audit_and_invalidates_match_not_features(visual):
    library, characters, gallery = visual
    identity = add_role(characters)
    reference = add_reference(gallery, identity)
    backend = FakeCCIP()
    pipeline = VisualIdentityPipeline(library, backend)
    frames = [frame(library)]
    first = pipeline.run("asset_one", frames, {"enabled": True})
    confirmed_record = [{"id": "old_observation", "summary": "Preserved raw sentence", "entities": [{
        "id": "P1", "identity_region_id": first["regions"][0]["id"], "character_id": identity,
        "name": "Aoi", "description": "Red jacket"}]}]
    features = backend.feature_calls
    gallery.update("asset_one", identity, reference["id"], status="revoked", reason="Wrong character sample")
    assert pipeline.project_result("asset_one", first)["regions"][0]["match"]["status"] == "unknown"
    projection = characters.decorate_observations("asset_one", confirmed_record)
    assert projection[0]["entities"][0]["character_id"] is None
    assert projection[0]["entities"][0]["name"] is None
    assert projection[0]["summary"] == confirmed_record[0]["summary"]
    assert confirmed_record[0]["entities"][0]["character_id"] == identity
    second = pipeline.run("asset_one", frames, {"enabled": True})
    assert first["regions"][0]["match"]["status"] == "candidate"
    assert second["regions"][0]["match"]["status"] == "unknown"
    assert backend.feature_calls == features
    saved = gallery._read(characters.context("asset_one")["scope_id"])
    assert saved["references"][0]["version"] == 2
    assert saved["audit"][-1]["before"]["status"] == "confirmed"
    assert saved["audit"][-1]["after"]["status"] == "revoked"


def test_rename_manual_review_and_resume_do_not_recompute_features_or_rewrite_raw_prose(visual):
    library, characters, gallery = visual
    identity = add_role(characters)
    add_reference(gallery, identity)
    backend = FakeCCIP()
    pipeline = VisualIdentityPipeline(library, backend)
    frames = [frame(library)]
    first = pipeline.run("asset_one", frames, {"enabled": True})
    counts = backend.feature_calls, backend.detection_calls, backend.distance_calls
    region = first["regions"][0]
    raw_path = pipeline._directory("asset_one") / "regions" / f"{region['id']}.json"
    before = raw_path.read_bytes()
    pipeline.review("asset_one", region["id"], status="confirmed", character_id=identity, reason="Verified manually")
    characters.update("asset_one", identity, {"name": "New name"})
    second = pipeline.run("asset_one", frames, {"enabled": True})
    assert (backend.feature_calls, backend.detection_calls, backend.distance_calls) == counts
    assert second["regions"][0]["match"]["character_name"] == "New name"
    assert raw_path.read_bytes() == before
    records = [{"summary": "An unknown person holds a cup", "entities": [
        {"id": "P1", "identity_region_id": region["id"], "description": "Red outfit"}],
        "events": [{"actor_ids": ["P1"], "target_ids": []}]}]
    original = deepcopy(records)
    decorated = characters.decorate_observations("asset_one", records)
    assert decorated[0]["entities"][0]["character_id"] == identity
    assert decorated[0]["summary"] == original[0]["summary"] and records == original
    assert decorated[0]["events"][0]["actor_ids"] == original[0]["events"][0]["actor_ids"]
    assert decorated[0]["events"][0]["actor_names"] == ["New name"]
    assert decorated[0]["events"][0]["actor_character_ids"] == [identity]
    pipeline.review("asset_one", region["id"], status="unknown", reason="Undo incorrect confirmation")
    assert characters.decorate_observations("asset_one", records)[0]["entities"][0]["character_id"] is None


def test_candidate_dossier_id_cannot_become_confirmed_through_provider_copy(visual):
    library, characters, _gallery = visual
    identity = add_role(characters)
    frame(library)
    record = {"id": "observation_one", "asset_id": "asset_one", "run_id": "run_one", "window_id": "window_one",
              "shot_id": "shot_one", "start_ms": 0, "end_ms": 1000, "summary": "A person stands",
              "evidence_frame_ids": ["frame_one"], "entities": [{"id": "P1", "kind": "person", "name": "Aoi",
              "description": "Aoi stands", "character_id": identity, "identity_status": "candidate",
              "match_confidence": .99, "match_reason": "Candidate Aoi", "evidence_frame_ids": ["frame_one"]}]}
    result = characters.apply_observation("asset_one", record)
    assert result["entities"][0]["character_id"] is None
    assert result["entities"][0]["candidate_character_id"] == identity


def test_cancelled_run_resumes_cached_completed_frames(visual):
    library, characters, gallery = visual
    add_reference(gallery, add_role(characters))
    frames = [frame(library), frame(library, "frame_two", at_ms=300)]
    backend = FakeCCIP()
    pipeline = VisualIdentityPipeline(library, backend)
    def checkpoint():
        # Called before each frame; cancellation after a completed first frame
        # leaves immutable detections, feature blobs and that region available.
        if len(list((pipeline._directory("asset_one") / "regions").glob("*.json"))) == 1:
            raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        pipeline.run("asset_one", frames, {"enabled": True}, checkpoint)
    assert len(pipeline.reviews("asset_one")["regions"]) == 1
    calls = backend.feature_calls
    restored = pipeline.run("asset_one", frames, {"enabled": True})
    assert len(restored["regions"]) == 2
    assert restored["metrics"]["detection_cache_hits"] >= 2
    assert backend.feature_calls == calls  # identical crops use the immutable feature cache


def test_disabled_missing_models_and_pts_disagreement_are_explicit(visual):
    library, _characters, _gallery = visual
    frames = [frame(library)]
    class Unavailable:
        @property
        def model_version(self):
            raise IdentityUnavailable("missing offline weights")
    pipeline = VisualIdentityPipeline(library, Unavailable())
    assert pipeline.run("asset_one", frames)["status"] == "disabled"
    unavailable = pipeline.run("asset_one", frames, {"enabled": True})
    assert unavailable["status"] == "unavailable" and unavailable["degraded_reason"]
    with pytest.raises(ValueError, match="时间轴"):
        VisualIdentityPipeline(library, FakeCCIP()).run("asset_one", [{**frames[0], "at_ms": 0}], {"enabled": True})


def test_local_reference_routes_do_not_fetch_urls_and_keep_original_frame_provenance(visual):
    library, characters, _gallery = visual
    identity = add_role(characters)
    frame(library)
    app = FastAPI()
    register_visual_identity_routes(app, library, None)
    client = TestClient(app)
    base = f"/api/assets/asset_one/characters/{identity}/references"
    response = client.post(base, files={"file": ("reference.png", picture(), "image/png")},
                           data={"source_kind": "external_cached", "source_url": "https://example.invalid/image",
                                 "confirmed": "true", "single_character": "true"})
    assert response.status_code == 200
    reference = response.json()
    assert "path" not in reference
    assert client.get(reference["image_url"]).status_code == 200
    crop = client.post(base + "/crop", json={"frame_id": "frame_one", "box": [0, 0, 1, 1],
                                            "confirmed": True, "single_character": True}).json()
    assert crop["source"]["asset_id"] == "asset_one" and crop["source"]["frame_id"] == "frame_one"
    assert crop["reference_only"] is True
    assert len(client.get(base).json()["references"]) == 2
    assert client.patch(base + "/" + reference["id"], json={"status": "revoked", "reason": "bad reference"}).json()["status"] == "revoked"
    assert client.get("/api/assets/asset_one/identity/settings").json()["enabled"] is False
    assert client.put("/api/assets/asset_one/identity/settings", json={"enabled": True}).json()["enabled"] is True


def test_reference_library_backup_retains_confirmations_and_audit(visual, tmp_path):
    library, characters, gallery = visual
    identity = add_role(characters)
    reference = add_reference(gallery, identity)
    backup = library.backup()
    restored = Library(tmp_path / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    new_gallery = ReferenceGallery(restored)
    assert new_gallery.trusted("asset_one")[0]["id"] == reference["id"]
    assert Path(new_gallery.trusted("asset_one")[0]["path"]).read_bytes() == Path(reference["path"]).read_bytes()
    assert read_json(new_gallery._directory(characters.context("asset_one")["scope_id"]) / "gallery.json")["audit"]


def test_display_rename_preserves_semantic_revision_and_other_edits_advance_it(visual):
    _library, characters, _gallery = visual
    identity = add_role(characters)
    scope = characters.context("asset_one")["scope_id"]
    original = characters._read(scope)
    characters.update("asset_one", identity, {"name": "Changed", "aliases": ["Alias"]})
    renamed = characters._read(scope)
    assert renamed["revision"] == original["revision"] + 1
    assert renamed["semantic_revision"] == original["semantic_revision"]
    characters.update("asset_one", identity, {"description": "Corrected biography"})
    assert characters._read(scope)["semantic_revision"] == renamed["semantic_revision"] + 1


def test_reference_revocation_updates_existing_scene_search_and_only_dependent_inferences(visual):
    from scenerecall.scenes import SceneStore
    from scenerecall.search import SearchEngine
    library, characters, gallery = visual
    identity = add_role(characters)
    reference = add_reference(gallery, identity, red=20)
    add_reference(gallery, add_role(characters, "Other"), red=220)
    current = frame(library, red=20)
    frame(library, "frame_two", at_ms=1200)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    result = pipeline.run("asset_one", [current], {"enabled": True, "calibration": {
        "model_version": "synthetic-ccip-v1", "dataset_id": "synthetic", "evaluation_id": "calibration_fixture",
        "aggregation": "nearest-two-mean-v1", "max_distance": .1, "min_margin": .1}})
    region = result["regions"][0]
    assert region["match"]["character_id"] == identity
    store = SceneStore(library)
    scenes = store.publish("asset_one", "run_fixture", [
        {"start_ms": 0, "end_ms": 1000, "summary": "A person lifts a cup", "frame_ids": ["frame_one"],
         "evidence_frame_ids": ["frame_one"], "inferences": [{"text": "May be preparing to drink", "frame_ids": ["frame_one"]}],
         "entities": [{"id": "P1", "kind": "person", "name": "Aoi", "description": "Red jacket",
                       "character_id": identity, "identity_region_id": region["id"], "evidence_frame_ids": ["frame_one"]}]},
        {"start_ms": 1000, "end_ms": 2000, "summary": "A room", "frame_ids": ["frame_two"],
         "evidence_frame_ids": ["frame_two"], "entities": []}], 0, 2000)
    paths = [library.asset_dir("asset_one") / store._relative(row) for row in scenes]
    originals = {path: path.read_bytes() for path in paths}
    search = SearchEngine(library.root / "indexes", None)
    search.upsert(library.records("asset_one"))
    with search._db() as db:
        assert "aoi" in db.execute("SELECT characters FROM records WHERE id=?", (scenes[0]["id"],)).fetchone()[0]
    app = FastAPI()
    register_visual_identity_routes(app, library, search)
    response = TestClient(app).patch(f"/api/assets/asset_one/characters/{identity}/references/{reference['id']}",
                                    json={"status": "revoked", "reason": "Wrong reference character"})
    assert response.status_code == 200
    projected = library.records("asset_one")
    scene = next(row for row in projected if row["id"] == scenes[0]["id"])
    assert scene["entities"][0]["character_id"] is None
    assert next(row for row in characters.catalog("asset_one")["profiles"] if row["id"] == identity)["appearance_count"] == 0
    assert store.list("asset_one")[0]["stale"] is True
    assert store.list("asset_one")[1]["stale"] is False
    with search._db() as db:
        assert "aoi" not in db.execute("SELECT characters FROM records WHERE id=?", (scenes[0]["id"],)).fetchone()[0]
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_restored_absolute_crop_path_is_never_authority_and_validation_checks_owned_pixels(visual, tmp_path):
    library, characters, gallery = visual
    add_reference(gallery, add_role(characters))
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    result = pipeline.run("asset_one", [frame(library)], {"enabled": True})
    original = result["regions"][0]["crop_path"]
    external = tmp_path / "outside.png"
    external.write_bytes(picture(220))
    result["regions"][0]["crop_path"] = str(external)
    assert pipeline.project_result("asset_one", result)["regions"][0]["crop_path"] == original
    validate_visual_artifacts(library)
    Path(original).unlink()
    with pytest.raises(FileNotFoundError):
        validate_visual_artifacts(library)
