"""Confirmation metadata must not rebuild the whole work's appearance history."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenerecall.characters import CharacterStore
from scenerecall.library import atomic_json
from scenerecall import character_confirmation, visual_identity
from scenerecall.retrieval_store import RetrievalStore
from scenerecall.scenes import SceneStore
from test_character_confirmation import confirm, prepared
from test_visual_identity import visual as visual
from test_organization import bangumi_archive


def test_confirmation_and_queue_use_metadata_roster_without_full_catalog(visual, monkeypatch):
    service, identity = prepared(visual)
    candidate = service.queue("asset_one")["roles"][0]["candidates"][0]
    expected = CharacterStore(service.library).catalog("asset_one")["profiles"][0]

    def no_catalog(*args, **kwargs):
        pytest.fail("one-region confirmation must not rebuild all episode appearances")

    monkeypatch.setattr(CharacterStore, "catalog", no_catalog)
    monkeypatch.setattr(RetrievalStore, "records", no_catalog)
    monkeypatch.setattr(SceneStore, "list", no_catalog)
    monkeypatch.setattr(service.library, "observations", no_catalog)
    confirm(service, identity, candidate)
    result = service.queue("asset_one")
    profile = result["roles"][0]["profile"]
    assert result["summary"]["confirmed"] == 1
    assert profile == {key: value for key, value in expected.items()
                       if key not in {"appearances", "appearance_count", "representative"}}


def test_roster_snapshot_does_not_survive_human_edits_or_reference_revocation(visual):
    service, identity = prepared(visual)
    first = service.queue("asset_one")
    catalog = CharacterStore(service.library)
    saved = deepcopy(catalog._read(first["scope_id"]))
    saved["profiles"][identity]["name"] = "New human name"
    saved["revision"] += 1
    atomic_json(catalog._path(first["scope_id"]), saved)
    second = service.queue("asset_one")
    assert second["roles"][0]["profile"]["name"] == "New human name"
    reference = service.gallery.list("asset_one")["references"][0]
    service.gallery.update("asset_one", identity, reference["id"], status="revoked", reason="incorrect reference")
    assert service.queue("asset_one")["roles"][0]["available_count"] == 0
    assert first["roles"][0]["available_count"] > 0


def test_source_asset_movement_invalidates_reference_with_operation_snapshot(visual):
    service, identity = prepared(visual)
    candidate = service.queue("asset_one")["roles"][0]["candidates"][0]
    confirm(service, identity, candidate)
    catalog = CharacterStore(service.library).roster("asset_one")
    original = service.gallery.list("asset_one", trusted_only=True, _catalog=catalog)["references"]
    assert sum(row["source"]["kind"] == "frame_crop" for row in original) == 1
    # A fresh snapshot lacking that source cannot keep its frame exemplar
    # active, even if the historical gallery and decision remain on disk.
    without_source = {**catalog, "assets": []}
    current = service.gallery.list("asset_one", trusted_only=True, _catalog=without_source)["references"]
    assert all(row["source"]["kind"] != "frame_crop" for row in current)


def test_queue_resolves_only_shown_crops_and_counts_missing_files_exactly(visual, monkeypatch):
    service, _identity = prepared(visual, times=tuple(index * 6000 for index in range(20)))
    raw = service.pipeline.reviews("asset_one")["regions"]
    Path(raw[-1]["crop_path"]).unlink()
    original = character_confirmation.confined_path
    resolved = []

    def counted(base, relative):
        if relative.startswith("crops/") or relative == "crops":
            resolved.append(relative)
        return original(base, relative)

    monkeypatch.setattr(character_confirmation, "confined_path", counted)
    monkeypatch.setattr(visual_identity, "confined_path", counted)
    role = service.queue("asset_one")["roles"][0]
    assert role["available_count"] == 19 and len(role["candidates"]) == 3
    assert len(resolved) == 4  # One confined directory and the three shown files.


def test_crop_inventory_keeps_full_confinement_for_symlinks(visual, monkeypatch):
    service, _identity = prepared(visual, times=(200,))
    chosen = service.queue("asset_one")["roles"][0]["candidates"][0]
    name = f"{chosen['region_id']}.png"

    class LinkedEntry:
        def __init__(self):
            self.name = name
        def stat(self, **kwargs):
            return SimpleNamespace(st_mode=0, st_file_attributes=0)
        def is_symlink(self):
            return True

    class Scan:
        def __enter__(self):
            return iter([LinkedEntry()])
        def __exit__(self, *args):
            return False

    original = character_confirmation.confined_path
    def reject_escape(base, relative):
        if relative == f"crops/{name}":
            raise ValueError("资料包含越界文件引用")
        return original(base, relative)
    monkeypatch.setattr(character_confirmation.os, "scandir", lambda _path: Scan())
    monkeypatch.setattr(character_confirmation, "confined_path", reject_escape)
    with pytest.raises(ValueError, match="越界"):
        service._crop_inventory("asset_one")(chosen["region_id"])


def test_reference_trust_reads_same_source_asset_once_per_operation(visual, monkeypatch):
    service, identity = prepared(visual)
    confirm(service, identity, service.queue("asset_one")["roles"][0]["candidates"][0])
    scope = CharacterStore(service.library).roster("asset_one")["scope_id"]
    saved = service.gallery._read(scope)
    exemplar = next(row for row in saved["references"] if row["source"]["kind"] == "frame_crop")
    saved["references"] = [{**exemplar, "id": f"reference_{index}"} for index in range(50)]
    monkeypatch.setattr(service.gallery, "_read", lambda _scope: saved)
    get_asset = service.library.get_asset
    calls = []
    def counted(asset_id, **kwargs):
        calls.append(asset_id)
        return get_asset(asset_id, **kwargs)
    monkeypatch.setattr(service.library, "get_asset", counted)
    assert len(service.gallery.active_reference_ids("asset_one")) == 50
    assert calls == ["asset_one"]


def test_same_scope_source_roster_still_excludes_disallowed_character(visual):
    service, identity = prepared(visual, times=(200,))
    library = service.library
    source = {**library.get_asset("asset_one"), "id": "asset_two", "episode": 2}
    atomic_json(library.asset_dir("asset_two") / "manifest.json", source)
    first = library.frame_path("asset_one", "frame_0")
    frame_path = library.asset_dir("asset_two") / "frames" / "frame_other.png"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(first.read_bytes())
    frame = library.register_frame("asset_two", {"id": "frame_other", "path": str(frame_path), "at_ms": 200})
    region = service.pipeline.run("asset_two", [frame], {"enabled": True})["regions"][0]
    organization = library.organization._read()
    organization["assets"]["asset_two"]["anime_profile"] = {
        "mode": "bangumi", "dossier": bangumi_archive(200)}
    atomic_json(library.organization.path, organization)
    assert library.get_asset("asset_one")["character_scope_id"] == library.get_asset("asset_two")["character_scope_id"]
    scope = library.get_asset("asset_one")["character_scope_id"]
    before = deepcopy(service.gallery._read(scope))
    assert all(row["asset_id"] == "asset_one" for row in service.queue("asset_one")["roles"][0]["candidates"])
    with pytest.raises(ValueError, match="同一作品名单"):
        service.decide("asset_one", identity, action="confirm", source_asset_id="asset_two", region_id=region["id"],
                       single_character=True, use_as_reference=True)
    assert service.gallery._read(scope) == before
    assert service._read(scope)["decisions"] == {}
    assert not (service.pipeline._directory("asset_two") / "decisions.json").exists()
