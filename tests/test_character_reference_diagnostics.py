"""Reference detector diagnostics are read-only, model-specific, and never identity reviews."""
import hashlib
from pathlib import Path

from PIL import Image
import pytest
from fastapi import FastAPI
import httpx

from scenerecall.character_confirmation import CharacterConfirmation
from scenerecall.library import atomic_json, digest, read_json
from scenerecall.visual_identity import (DEFAULT_MODEL, PREPROCESS_VERSION, IdentityUnavailable,
                                        VisualIdentityPipeline)
from scenerecall.visual_identity_api import register_visual_identity_routes
from test_visual_identity import FakeCCIP, add_reference, add_role, frame, picture, visual as visual


def bangumi_candidate(visual, name="Mina", red=20):
    library, characters, gallery = visual
    identity = add_role(characters, name)
    scope = characters.roster("asset_one")["scope_id"]
    registry = characters._read(scope)
    url = f"https://lain.bgm.tv/pic/crt/l/{identity}.png"
    registry["profiles"][identity]["external_source"] = {
        "source": "bangumi", "subject_id": 1, "character_id": 1, "image_url": url,
        "url": "https://bgm.tv/character/1", "fetched_at": "2026-11-11T00:00:00Z"}
    atomic_json(characters._path(scope), registry)
    reference = gallery.add("asset_one", identity, picture(red), source_kind="external_cached", source_url=url)
    assert reference["status"] == "candidate"
    return identity, reference


class ReferenceDetector(FakeCCIP):
    def __init__(self, people=0):
        super().__init__()
        self.people = people

    def detect(self, path):
        with Image.open(path) as image:
            if "reference-images" in Path(path).parts and image.getpixel((0, 0))[0] == 20:
                self.detection_calls += 1
                return [{"box": [index * 40, 0, index * 40 + 35, 180], "label": "person", "detection_score": .9}
                        for index in range(self.people)]
        return super().detect(path)


def role(queue, identity):
    return next(row for row in queue["roles"] if row["character_id"] == identity)


def forbid_models(service, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reading confirmation diagnostics must not run or load a model")
    monkeypatch.setattr("scenerecall.visual_identity._backend", forbidden)
    if service.pipeline.backend is not None:
        monkeypatch.setattr(service.pipeline.backend, "detect", forbidden)
        monkeypatch.setattr(service.pipeline.backend, "feature", forbidden)


@pytest.mark.parametrize("people,status", [(0, "no_person"), (2, "multiple_people")])
def test_cached_reference_failure_is_separate_from_manual_pool_eligibility_and_preserves_trust(visual, monkeypatch, people, status):
    library, characters, gallery = visual
    manager, reference = bangumi_candidate(visual)
    other = add_role(characters, "Other")
    add_reference(gallery, other, red=90)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(people))
    pipeline.run("asset_one", [frame(library)], {"enabled": True})
    service = CharacterConfirmation(library, pipeline)
    before_gallery = gallery._read(characters.roster("asset_one")["scope_id"])
    before_confirmation = service._read(characters.roster("asset_one")["scope_id"])
    forbid_models(service, monkeypatch)
    before_files = {path.relative_to(library.root): path.read_bytes() for path in library.root.rglob("*") if path.is_file()}
    queue = service.queue("asset_one")
    assert {path.relative_to(library.root): path.read_bytes() for path in library.root.rglob("*") if path.is_file()} == before_files
    current = role(queue, manager)
    assert current["matching_reference_count"] == 1
    assert current["usable_matching_reference_count"] == 0
    assert current["reference_detection_status"] == "failed"
    assert current["reference_issues"][0]["reference_id"] == reference["id"]
    assert current["reference_issues"][0]["status"] == status
    assert current["reference_issues"][0]["person_count"] == people
    assert current["reference_issues"][0]["reason"] and current["reference_issues"][0]["action"]
    assert current["coverage"]["status"] == "reference_invalid"
    assert current["available_count"] == current["coverage"]["candidate_count"] == 0
    assert current["coverage"]["review_pool_count"] == 1  # Review eligibility is not identity evidence.
    assert manager in queue["review_pool"][0]["eligible_character_ids"]
    assert role(queue, other)["usable_matching_reference_count"] == 1
    assert gallery._read(queue["scope_id"]) == before_gallery
    assert service._read(queue["scope_id"]) == before_confirmation
    assert gallery.list("asset_one", manager)["references"][0]["status"] == "candidate"


def test_missing_detection_cache_is_not_checked_without_model_loading_or_failure(visual, monkeypatch):
    library, characters, gallery = visual
    identity = add_role(characters)
    add_reference(gallery, identity)
    service = CharacterConfirmation(library)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["matching_reference_count"] == 1 and current["usable_matching_reference_count"] == 0
    assert current["reference_detection_status"] == "not_checked"
    assert current["reference_issues"][0]["status"] == "not_checked"
    assert current["reference_issues"][0]["person_count"] is None
    assert current["coverage"]["status"] == "reference_unchecked"
    assert current["reference_model_version"] is None


def test_explicit_model_unavailability_is_distinct_from_reference_detection_failure(visual, monkeypatch):
    library, characters, gallery = visual
    identity = add_role(characters)
    add_reference(gallery, identity)
    class Unavailable:
        @property
        def model_version(self):
            raise IdentityUnavailable("synthetic unavailable model")
    pipeline = VisualIdentityPipeline(library, Unavailable())
    result = pipeline.run("asset_one", [frame(library)], {"enabled": True})
    assert result["status"] == "unavailable" and result["model"] == DEFAULT_MODEL
    service = CharacterConfirmation(library, pipeline)
    monkeypatch.setattr("scenerecall.visual_identity._backend", lambda *args: pytest.fail("queue loaded backend"))
    current = role(service.queue("asset_one"), identity)
    assert current["reference_detection_status"] == "model_unavailable"
    assert current["reference_issues"][0]["status"] == "model_unavailable"
    assert current["reference_issues"][0]["person_count"] is None
    assert current["coverage"]["status"] == "model_unavailable"
    assert gallery.list("asset_one")["references"][0]["status"] == "confirmed"


def test_mixed_references_count_only_usable_detection_and_still_show_other_issue(visual, monkeypatch):
    library, characters, gallery = visual
    identity, _bad = bangumi_candidate(visual)
    good = add_reference(gallery, identity, red=90)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(0))
    result = pipeline.run("asset_one", [frame(library)], {"enabled": True})
    service = CharacterConfirmation(library, pipeline)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["matching_reference_count"] == 2 and current["usable_matching_reference_count"] == 1
    assert current["reference_detection_status"] == "usable"
    assert [issue["status"] for issue in current["reference_issues"]] == ["no_person"]
    assert current["coverage"]["status"] == "pending_review"
    key = digest([good["sha256"], result["model_version"], PREPROCESS_VERSION])
    receipt = read_json(library.root / "cache" / "identity-reference-detections" / f"{key}.json")
    assert receipt["model"] == DEFAULT_MODEL and receipt["model_version"] == result["model_version"]
    assert receipt["reference_sha256"] == good["sha256"] and receipt["preprocessing"] == PREPROCESS_VERSION


def test_model_change_invalidates_prior_reference_detection_without_claiming_failure(visual, monkeypatch):
    library, characters, gallery = visual
    identity, _ = bangumi_candidate(visual)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(0))
    pipeline.run("asset_one", [frame(library)], {"enabled": True})
    pipeline.configure("asset_one", {"model": "ccip-caformer-5_fp32"})
    service = CharacterConfirmation(library, pipeline)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["reference_detection_status"] == "not_checked"
    assert current["reference_issues"][0]["status"] == "not_checked"
    assert current["reference_model_version"] is None


def test_changed_reference_pixels_do_not_reuse_previous_no_person_result(visual, monkeypatch):
    library, characters, gallery = visual
    identity, reference = bangumi_candidate(visual)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(0))
    pipeline.run("asset_one", [frame(library)], {"enabled": True})
    scope = characters.roster("asset_one")["scope_id"]
    saved = gallery._read(scope)
    changed = picture(90)
    gallery._directory(scope).joinpath(reference["id"] + ".png").write_bytes(changed)
    saved["references"][0]["sha256"] = hashlib.sha256(changed).hexdigest()
    saved["references"][0]["version"] += 1
    atomic_json(gallery._directory(scope) / "gallery.json", saved)
    service = CharacterConfirmation(library, pipeline)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["reference_detection_status"] == "not_checked"
    assert current["reference_issues"][0]["status"] == "not_checked"


def test_legacy_real_model_receipt_identifies_zero_person_reference_without_inference(visual, monkeypatch):
    library, _characters, _gallery = visual
    identity, reference = bangumi_candidate(visual)
    pipeline = VisualIdentityPipeline(library)
    version = f"imgutils-fixture:{DEFAULT_MODEL}:deepghs/ccip_onnx@synthetic-commit"
    atomic_json(pipeline._directory("asset_one") / "latest.json", {"status": "completed", "model_version": version})
    key = digest([reference["sha256"], version, PREPROCESS_VERSION])
    atomic_json(library.root / "cache" / "identity-reference-detections" / f"{key}.json", {"detections": []})
    service = CharacterConfirmation(library, pipeline)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["reference_detection_status"] == "failed"
    assert current["reference_issues"][0]["status"] == "no_person"
    assert current["reference_model_version"] == version


def test_reference_diagnostics_dont_modify_existing_human_confirmation_or_skip(visual, monkeypatch):
    library, characters, gallery = visual
    identity = add_role(characters)
    add_reference(gallery, identity)
    pipeline = VisualIdentityPipeline(library, FakeCCIP())
    result = pipeline.run("asset_one", [frame(library)], {"enabled": True})
    service = CharacterConfirmation(library, pipeline)
    service.decide("asset_one", identity, action="confirm", source_asset_id="asset_one", region_id=result["regions"][0]["id"],
                   single_character=True, use_as_reference=True)
    service.decide("asset_one", identity, action="skip")
    scope = characters.roster("asset_one")["scope_id"]
    paths = [service._root(scope) / "state.json", pipeline._directory("asset_one") / "decisions.json", gallery._directory(scope) / "gallery.json"]
    before = {path: path.read_bytes() for path in paths}
    forbid_models(service, monkeypatch)
    queue = service.queue("asset_one")
    assert role(queue, identity)["status"] == "confirmed"
    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize("field", ["reference_sha256", "model", "model_version", "preprocessing"])
def test_mismatched_detection_cache_provenance_is_not_a_reference_failure(visual, monkeypatch, field):
    library, _characters, _gallery = visual
    identity, reference = bangumi_candidate(visual)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(0))
    result = pipeline.run("asset_one", [frame(library)], {"enabled": True})
    key = digest([reference["sha256"], result["model_version"], PREPROCESS_VERSION])
    path = library.root / "cache" / "identity-reference-detections" / f"{key}.json"
    cached = read_json(path)
    cached[field] = "different-input"
    atomic_json(path, cached)
    service = CharacterConfirmation(library, pipeline)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["reference_detection_status"] == "not_checked"
    assert current["reference_issues"][0]["person_count"] is None


@pytest.mark.asyncio
async def test_confirmation_api_returns_diagnostic_without_loading_backend_or_altering_reference(visual, monkeypatch):
    library, _characters, gallery = visual
    identity, reference = bangumi_candidate(visual)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(0))
    pipeline.run("asset_one", [frame(library)], {"enabled": True})
    app = FastAPI()
    register_visual_identity_routes(app, library, None)
    service = app.state.character_confirmation
    forbid_models(service, monkeypatch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/assets/asset_one/identity/confirmation")
    assert response.status_code == 200
    current = role(response.json(), identity)
    assert current["matching_reference_count"] == 1 and current["usable_matching_reference_count"] == 0
    assert current["reference_issues"][0]["status"] == "no_person"
    assert current["reference_issues"][0]["reference_id"] == reference["id"]
    assert current["coverage"]["status"] == "reference_invalid"
    assert gallery.list("asset_one", identity)["references"][0]["status"] == "candidate"


@pytest.mark.parametrize("record", ["latest", "detection"])
@pytest.mark.parametrize("damage", ["malformed_json", "unsupported_schema", "wrong_type", "unreadable"])
def test_unreadable_optional_receipt_degrades_to_not_checked_without_hiding_core_errors(visual, monkeypatch, record, damage):
    from scenerecall import character_confirmation
    library, characters, gallery = visual
    identity, reference = bangumi_candidate(visual)
    pipeline = VisualIdentityPipeline(library, ReferenceDetector(0))
    result = pipeline.run("asset_one", [frame(library)], {"enabled": True})
    if record == "latest":
        path = pipeline._directory("asset_one") / "latest.json"
    else:
        key = digest([reference["sha256"], result["model_version"], PREPROCESS_VERSION])
        path = library.root / "cache" / "identity-reference-detections" / f"{key}.json"
    if damage == "malformed_json":
        path.write_text("{incomplete", encoding="utf-8")
    elif damage == "unsupported_schema":
        path.write_text('{"schema_version":"unsupported"}', encoding="utf-8")
    elif damage == "wrong_type":
        path.write_text("[]", encoding="utf-8")
    else:
        original_read = character_confirmation.read_json
        def unreadable(requested, *args, **kwargs):
            if requested == path:
                raise PermissionError("synthetic unreadable optional receipt")
            return original_read(requested, *args, **kwargs)
        monkeypatch.setattr(character_confirmation, "read_json", unreadable)
    service = CharacterConfirmation(library, pipeline)
    forbid_models(service, monkeypatch)
    current = role(service.queue("asset_one"), identity)
    assert current["reference_detection_status"] == "not_checked"
    assert current["reference_issues"][0]["status"] == "not_checked"
    assert gallery.list("asset_one", identity)["references"][0]["status"] == "candidate"
    # The recovery applies only to rebuildable diagnostics, never core human state.
    scope = characters.roster("asset_one")["scope_id"]
    characters._path(scope).write_text("{invalid core roster", encoding="utf-8")
    with pytest.raises(ValueError):
        service.queue("asset_one")
