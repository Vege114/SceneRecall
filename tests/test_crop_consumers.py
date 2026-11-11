"""Crop consumers continue to work when ordinary recognition PNGs are absent."""
import base64
import hashlib
import io
import json
from pathlib import Path

from fastapi import FastAPI
import httpx
from PIL import Image
import pytest

from scenerecall.character_confirmation import CharacterConfirmation
from scenerecall.identity_scene import identity_context
from scenerecall.library import read_json
from scenerecall.providers import ProviderError, ProviderManager
from scenerecall.visual_identity import VisualIdentityPipeline
from scenerecall.visual_identity_api import register_visual_identity_routes
from test_visual_identity import (FakeCCIP, add_reference, add_role, frame, picture, visual as visual)


def prepare(visual, *, discard=True, boxes=None, size=(128, 192)):
    library, characters, gallery = visual
    character_id = add_role(characters)
    add_reference(gallery, character_id)
    backend = FakeCCIP(boxes=boxes)
    pipeline = VisualIdentityPipeline(library, backend)
    source = frame(library, size=size)
    result = pipeline.run("asset_one", [source], {
        "enabled": True, "crop_retention": "discard" if discard else "retain"})
    assert result["status"] == "completed"
    return CharacterConfirmation(library, pipeline), character_id, source, result


def crop_files(service):
    return list((service.pipeline._directory("asset_one") / "crops").glob("*.png"))


@pytest.mark.asyncio
async def test_public_identity_apis_hide_features_but_keep_durable_restart_data(visual):
    service, _character_id, source, result = prepare(visual)
    region_id = result["regions"][0]["id"]
    raw_path = service.pipeline._directory("asset_one") / "regions" / f"{region_id}.json"
    feature = read_json(raw_path)["feature"]
    assert feature
    service.pipeline.configure("asset_one", {"enabled": True, "crop_retention": "discard"})
    app = FastAPI()
    register_visual_identity_routes(app, service.library, None)
    app.state.visual_identity.backend = service.pipeline.backend
    base = "/api/assets/asset_one/identity"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        reviews = await client.get(base + "/reviews")
        run = await client.post(base + "/run", json={"frame_ids": [source["id"]]})
        review = await client.post(base + f"/reviews/{region_id}", json={"status": "unknown", "reason": "human review"})
    for response in (reviews, run, review):
        assert response.status_code == 200
        rows = response.json().get("regions", [response.json()])
        assert rows
        assert all("feature" not in row and "crop_path" not in row for row in rows)
    assert read_json(raw_path)["feature"] == feature
    assert service.pipeline.reviews("asset_one")["regions"][0]["feature"] == feature


@pytest.mark.asyncio
@pytest.mark.parametrize("discard", [True, False])
async def test_repeated_crop_preview_reconstructs_without_writing_png(visual, monkeypatch, discard):
    service, _character_id, _source, result = prepare(visual, discard=discard)
    row = result["regions"][0]
    expected = service.pipeline.crop_bytes("asset_one", row["id"])
    for path in crop_files(service):
        path.unlink()
    monkeypatch.setattr(service.pipeline.backend, "detect", lambda *args: pytest.fail("preview ran detection"))
    monkeypatch.setattr(service.pipeline.backend, "feature", lambda *args: pytest.fail("preview ran CCIP"))
    app = FastAPI()
    register_visual_identity_routes(app, service.library, None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for _ in range(3):
            response = await client.get(row["crop_url"])
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"
            assert response.headers["cache-control"] == "no-store"
            assert response.content == expected
    assert crop_files(service) == []


def test_confirmation_without_crop_keeps_queue_and_promotes_independent_reference(visual, monkeypatch):
    service, character_id, _source, result = prepare(visual)
    assert crop_files(service) == []
    monkeypatch.setattr(service.pipeline.backend, "detect", lambda *args: pytest.fail("queue ran detection"))
    monkeypatch.setattr(service.pipeline.backend, "feature", lambda *args: pytest.fail("queue ran CCIP"))
    queue = service.queue("asset_one")
    assert queue["roles"][0]["candidates"] == []
    candidate = queue["review_pool"][0]
    assert candidate["confidence_state"] == "uncalibrated"
    assert candidate["region_id"] == result["regions"][0]["id"]
    service.decide("asset_one", character_id, action="confirm", source_asset_id="asset_one",
                   region_id=candidate["region_id"], single_character=True, use_as_reference=True)
    reference = next(row for row in service.gallery.trusted("asset_one")
                     if row["source"].get("region_id") == candidate["region_id"])
    assert Path(reference["path"]).is_file()
    assert Path(reference["path"]).read_bytes() == service.pipeline.crop_bytes("asset_one", candidate["region_id"])
    fresh = CharacterConfirmation(service.library, VisualIdentityPipeline(service.library, FakeCCIP()))
    assert fresh.queue("asset_one")["summary"]["confirmed"] == 1
    assert crop_files(fresh) == []


def test_confirmation_without_crop_rejects_changed_original_pixels(visual):
    service, character_id, source, result = prepare(visual)
    row = result["regions"][0]
    raw = read_json(service.pipeline._directory("asset_one") / "regions" / f"{row['id']}.json")
    assert raw["crop_sha256"]
    before = len(service.gallery.trusted("asset_one"))
    Path(source["path"]).write_bytes(picture(190))
    with pytest.raises(ValueError):
        service.decide("asset_one", character_id, action="confirm", source_asset_id="asset_one",
                       region_id=row["id"], single_character=True, use_as_reference=True)
    assert len(service.gallery.trusted("asset_one")) == before
    assert crop_files(service) == []


@pytest.mark.parametrize("discard", [True, False])
@pytest.mark.parametrize("x0,x1,end_pixel", [(11, 75.5, 76), (42, 106.5, 106)])
def test_provider_region_image_uses_exact_original_pixels_without_crop_file(visual, discard, x0, x1, end_pixel):
    service, _character_id, source, result = prepare(
        visual, discard=discard, boxes=[[x0, 0, x1, 192]], size=(127, 192))
    row = result["regions"][0]
    # Independently crop the original detector bounds. These .5 cases expose
    # changes caused by deriving the rounded pixels again from normalized boxes.
    with Image.open(source["path"]) as original:
        expected_image = original.convert("RGB").crop((x0, 0, end_pixel, 192))
    expected_file = io.BytesIO()
    expected_image.save(expected_file, format="PNG")
    expected = expected_file.getvalue()
    if not discard:
        assert Path(row["crop_path"]).read_bytes() == expected
        Path(row["crop_path"]).unlink()
    assert crop_files(service) == []
    assert hashlib.sha256(expected).hexdigest() == row["crop_sha256"]
    before = {path.relative_to(service.library.root): path.read_bytes()
              for path in service.library.root.rglob("*") if path.is_file()}
    context = identity_context({}, result, [source])
    region_image = context["region_images"][0]
    assert region_image["path"] == source["path"]
    assert region_image["person_crop"] == {key: row[key] for key in (
        "box", "crop_box_pixels", "source_frame_sha256", "crop_sha256")}
    for _ in range(3):
        content = ProviderManager._images([{**region_image, "id": region_image["frame_id"]}])
        assert json.loads(content[0]["text"]) == {"frame_id": source["id"], "at_ms": source["at_ms"]}
        mime, encoded = content[1]["image_url"]["url"].split(",", 1)
        assert mime == "data:image/png;base64"
        actual = base64.b64decode(encoded)
        assert actual == expected
        with Image.open(io.BytesIO(actual)) as image:
            assert image.size == (end_pixel - x0, 192)
            assert image.mode == expected_image.mode and image.tobytes() == expected_image.tobytes()
    assert {path.relative_to(service.library.root): path.read_bytes()
            for path in service.library.root.rglob("*") if path.is_file()} == before
    assert crop_files(service) == []


@pytest.mark.parametrize("tamper,reason", [
    ("source", "source frame pixels changed"),
    ("digest", "Reconstructed person crop pixels changed"),
    ("bounds", "source bounds changed"),
])
def test_provider_region_image_rejects_changed_provenance_without_writing(visual, tamper, reason):
    service, _character_id, source, result = prepare(visual)
    region_image = identity_context({}, result, [source])["region_images"][0]
    if tamper == "source":
        Path(source["path"]).write_bytes(picture(190))
    elif tamper == "digest":
        region_image["person_crop"]["crop_sha256"] = "0" * 64
    else:
        region_image["person_crop"]["crop_box_pixels"][0] += 2
    before = {path.relative_to(service.library.root): path.read_bytes()
              for path in service.library.root.rglob("*") if path.is_file()}
    with pytest.raises(ProviderError, match="人物裁剪来源或像素校验失败") as failure:
        ProviderManager._images([{**region_image, "id": region_image["frame_id"]}])
    assert isinstance(failure.value.__cause__, ValueError)
    assert reason in str(failure.value.__cause__)
    assert {path.relative_to(service.library.root): path.read_bytes()
            for path in service.library.root.rglob("*") if path.is_file()} == before
    assert crop_files(service) == []
