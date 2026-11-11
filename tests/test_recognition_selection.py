"""A user's preflight selection controls which cast members recognition can match."""
import pytest
from fastapi.testclient import TestClient

from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.character_confirmation import CharacterConfirmation
from scenerecall.library import Library
from scenerecall.main import create_app
from test_organization import appearance, bangumi_archive, seed
from test_visual_identity import FakeCCIP, add_reference, frame


BASE = "/api/assets/asset_one/identity/confirmation"


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "library", start_worker=False, token="selection-test")
    library = app.state.library
    seed(library, "asset_one")
    seed(library, "asset_two", episode=2)
    archive = bangumi_archive()
    for character, role in zip(archive["characters"], ("主角", "配角"), strict=True):
        character["role"] = role
    AnimeProfileStore(library).apply_asset("asset_one", {"mode": "bangumi", "subject_id": 100}, archive)
    with TestClient(app, headers={"Authorization": "Bearer selection-test"}) as session:
        yield session


def save(client, queue, ids):
    return client.put(BASE + "/selection", json={
        "scope_id": queue["scope_id"], "expected_revision": queue["roster_revision"], "character_ids": ids})


def enabled(queue):
    return {row["character_id"] for row in queue["roles"] if row["recognition_enabled"]}


def test_choose_supporting_role_across_episodes_and_restore_without_annotation(client):
    queue = client.get(BASE).json()
    roles = {row["profile"]["external_source"]["role"]: row["character_id"] for row in queue["roles"]}
    all_ids = {roles["主角"], roles["配角"]}
    assert enabled(queue) == all_ids
    saved = save(client, queue, [roles["配角"]])
    assert saved.status_code == 200, saved.text
    selection = saved.json()
    assert enabled(selection) == {roles["配角"]}
    assert selection["summary"]["pending"] == selection["summary"]["total"] == 1
    assert {row["character_id"] for row in selection["roles"]} == all_ids

    library = client.app.state.library
    backup = library.backup()
    restored = Library(library.root.parent / "restored")
    restored.restore(library.root / "runtime" / "backups" / f"{backup['id']}.zip")
    assert enabled(CharacterConfirmation(restored).queue("asset_two")) == {roles["配角"]}

    empty = save(client, selection, [])
    assert empty.status_code == 200, empty.text
    assert empty.json()["summary"]["total"] == empty.json()["summary"]["pending"] == 0
    assert enabled(empty.json()) == set()
    assert client.app.state.jobs.character_context("asset_two")["profiles"] == []
    again = save(client, empty.json(), list(all_ids))
    assert again.status_code == 200, again.text
    assert enabled(again.json()) == all_ids
    assert {row["id"] for row in client.app.state.jobs.character_context("asset_two")["profiles"]} == all_ids


def test_selection_rejects_stale_and_foreign_updates_atomically(client):
    queue = client.get(BASE).json()
    first, second = [row["character_id"] for row in queue["roles"]]
    saved = save(client, queue, [first])
    assert saved.status_code == 200, saved.text
    revision = saved.json()["roster_revision"]
    assert revision > queue["roster_revision"]
    assert save(client, queue, [second]).status_code == 409
    assert save(client, saved.json(), [second, "character_outside_this_work"]).status_code == 400
    assert save(client, {**saved.json(), "scope_id": "different_work"}, []).status_code in (400, 409)
    current = client.get(BASE).json()
    assert current["roster_revision"] == revision and enabled(current) == {first}
    duplicate = save(client, current, [first])
    assert duplicate.status_code == 200 and duplicate.json()["roster_revision"] == revision


def test_new_model_observation_cannot_bind_deselected_dossier_and_history_survives(client):
    library, characters = client.app.state.library, client.app.state.characters
    queue = client.get(BASE).json()
    first, second = [row["character_id"] for row in queue["roles"]]
    assert appearance(library, characters, character_id=first) == first
    history = library.observations("asset_one")[0]
    saved = save(client, client.get(BASE).json(), [second])
    assert saved.status_code == 200, saved.text
    assert appearance(library, characters, number=2, character_id=first) is None
    assert history in library.observations("asset_one")
    profiles = {row["id"]: row for row in characters.catalog("asset_one")["profiles"]}
    assert profiles[first]["appearance_count"] == 1


def test_selection_changes_matching_of_cached_frames_and_keeps_confirmed_references(client):
    pipeline = client.app.state.visual_identity
    pipeline.backend = FakeCCIP()
    queue = client.get(BASE).json()
    first, second = [row["character_id"] for row in queue["roles"]]
    add_reference(pipeline.gallery, first, red=90)
    add_reference(pipeline.gallery, second, red=190)
    library = client.app.state.library
    confirmed_frame = frame(library, "frame_confirm", red=90, at_ms=100)
    other_frame = frame(library, "frame_rematch", red=95, at_ms=7000)
    config = {"enabled": True, "top_k": 2}
    original = pipeline.run("asset_one", [confirmed_frame, other_frame], config)
    assert original["regions"][1]["match"]["candidate_character_id"] == first
    client.app.state.character_confirmation.decide("asset_one", first, action="confirm", single_character=True,
        use_as_reference=True, source_asset_id="asset_one", region_id=original["regions"][0]["id"])

    saved = save(client, client.get(BASE).json(), [second])
    assert saved.status_code == 200, saved.text
    roles = {row["character_id"]: row for row in saved.json()["roles"]}
    assert not roles[first]["recognition_enabled"] and roles[first]["candidates"] == []
    assert saved.json()["summary"]["confirmed"] == 0
    context = client.app.state.jobs.character_context("asset_one")
    assert [row["id"] for row in context["profiles"]] == [second]
    assert {row["character_id"] for row in context["reference_frames"]} == {second}
    rematched = pipeline.run("asset_one", [other_frame], config)
    match = rematched["regions"][0]["match"]
    assert match["candidate_character_id"] == second
    assert [row["character_id"] for row in match["candidates"]] == [second]
    assert {row["character_id"] for row in rematched["references"]} == {second}

    reenabled = save(client, saved.json(), [first, second])
    assert reenabled.status_code == 200, reenabled.text
    role = next(row for row in reenabled.json()["roles"] if row["character_id"] == first)
    assert role["status"] == "confirmed" and role["trusted_reference_count"] == 1
    rematched = pipeline.run("asset_one", [other_frame], config)
    assert rematched["regions"][0]["match"]["candidate_character_id"] == first
