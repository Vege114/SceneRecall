from copy import deepcopy
from types import SimpleNamespace

from scenerecall.identity_jobs import IdentityJobMixin
from scenerecall.identity_scene import identity_context
from scenerecall.library import atomic_json
from scenerecall.scene_jobs import canonical
from scenerecall.visual_identity import VisualIdentityPipeline
from test_visual_identity import add_reference, add_role, picture, visual  # noqa: F401


def region(identity, frame_id, character_id, reference_id, *, candidate=False):
    return {"id": identity, "entity_id": identity, "local_id": "P1", "frame_id": frame_id,
            "at_ms": 0, "box": [0, 0, .5, 1], "quality": {"accepted": True},
            "crop_path": "/untrusted/foreign-asset.png",
            "match": {"status": "candidate", "character_id": None, "candidate_character_id": character_id,
                      "source": "ccip_unverified_reference" if candidate else "ccip_uncalibrated",
                      "candidates": [{"character_id": character_id, "distance": .1,
                                      "reference_ids": [reference_id]}]}}


def test_current_frames_filtered_before_deepcopy_and_ui_default_still_complete(visual):  # noqa: F811
    library, characters, gallery = visual
    identity = add_role(characters)
    reference = add_reference(gallery, identity)
    pipeline = VisualIdentityPipeline(library)
    result = {"status": "completed", "regions": [region("region_a", "frame_a", identity, reference["id"]),
                                                region("region_b", "frame_b", identity, reference["id"])]}
    original = deepcopy(result)
    full = pipeline.project_result("asset_one", result)
    selected = pipeline.project_result("asset_one", result, frame_ids={"frame_a"})
    frames = [{"id": "frame_a"}]
    context = {"profiles": characters.context("asset_one")["profiles"], "reference_frames": []}
    assert canonical(identity_context(context, selected, frames)) == canonical(identity_context(context, full, frames))
    assert len(full["regions"]) == 2
    assert [row["id"] for row in selected["regions"]] == ["region_a"]
    assert result == original

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise AssertionError("Off-frame regions must never be copied")

    result["regions"][1]["off_frame_payload"] = Uncopyable()
    assert len(pipeline.project_result("asset_one", result, frame_ids=["frame_a"])["regions"]) == 1
    empty = pipeline.project_result("asset_one", result, frame_ids=[])
    assert empty["regions"] == empty["selected_profiles"] == empty["references"] == []


def test_scene_context_passes_only_current_regions_to_projection(monkeypatch):
    rows = [region("region_a", "frame_a", "character_a", "reference_a"),
            region("region_b", "frame_b", "character_b", "reference_b")]
    saved = {"visual": {"status": "completed", "regions": rows}}
    seen = []

    def project(_self, asset_id, result):
        seen.extend(row["id"] for row in result["regions"])
        assert asset_id == "asset_one"
        return result

    monkeypatch.setattr("scenerecall.identity_jobs.read_json", lambda *_: saved)
    monkeypatch.setattr(VisualIdentityPipeline, "project_result", project)
    queue = SimpleNamespace(library=None, get=lambda _: {"asset_id": "asset_one", "config": {}},
                            _identity_path=lambda _: "unused")
    output = IdentityJobMixin.scene_identity_context(queue, "job_one",
        {"profiles": [{"id": "character_a", "name": "Aoi"}], "reference_frames": []}, [{"id": "frame_a"}])
    assert seen == ["region_a"]
    assert [row["id"] for row in output["identity_regions"]] == ["region_a"]
    assert len(saved["visual"]["regions"]) == 2


def test_multiple_candidate_references_are_current_and_revocations_and_human_decisions_apply(visual):  # noqa: F811
    library, characters, gallery = visual
    identity = add_role(characters)
    scope = characters.catalog("asset_one")["scope_id"]
    registry = characters._read(scope)
    sources = [{"source": "bangumi", "subject_id": 1, "character_id": 1,
                "image_url": f"https://lain.bgm.tv/pic/crt/l/{number}.png",
                "url": "https://bgm.tv/character/1", "fetched_at": "2026-10-18T00:00:00Z"}
               for number in range(2)]
    registry["profiles"][identity].update(external_source=sources[0], external_sources=sources[1:])
    atomic_json(characters._path(scope), registry)
    refs = [gallery.add("asset_one", identity, picture(red=90 + index * 20), source_kind="external_cached",
                        source_url=source["image_url"]) for index, source in enumerate(sources)]
    pipeline = VisualIdentityPipeline(library)
    result = {"status": "completed", "regions": [
        region("region_a", "frame_a", identity, refs[0]["id"], candidate=True),
        region("region_b", "frame_b", identity, refs[1]["id"], candidate=True)]}
    current = pipeline.project_result("asset_one", result, frame_ids=["frame_a"])
    assert [row["reference_id"] for row in current["references"]] == [refs[0]["id"]]
    assert len(pipeline.project_result("asset_one", result)["references"]) == 2
    assert current["regions"][0]["match"]["status"] == "candidate"
    assert current["references"][0]["human_confirmed"] is False

    gallery.update("asset_one", identity, refs[0]["id"], status="revoked", reason="Wrong reference")
    revoked = pipeline.project_result("asset_one", result, frame_ids=["frame_a"])
    assert revoked["regions"][0]["match"]["status"] == "unknown"
    assert revoked["selected_profiles"] == revoked["references"] == []
    context = {"profiles": [{"id": identity, "name": "Aoi", "reference_only": True}], "reference_frames": []}
    assert identity_context(context, revoked, [{"id": "frame_a"}])["profiles"] == context["profiles"]

    decisions = {"decisions": {"region_a": {"status": "confirmed", "character_id": identity,
                 "candidate_character_id": None, "source": "human", "reason": "Human reviewed", "version": 1}}}
    atomic_json(pipeline._directory("asset_one") / "decisions.json", decisions)
    human = pipeline.project_result("asset_one", result, frame_ids=["frame_a"])
    assert human["regions"][0]["match"]["status"] == "confirmed"
    assert human["regions"][0]["match"]["source"] == "human"
    decisions["decisions"]["region_a"].update(status="unknown", character_id=None)
    atomic_json(pipeline._directory("asset_one") / "decisions.json", decisions)
    assert pipeline.project_result("asset_one", result, frame_ids=["frame_a"])["references"] == []


def test_all_confirmed_refs_stay_available_and_other_asset_identity_is_rejected(visual):  # noqa: F811
    library, characters, gallery = visual
    identity = add_role(characters)
    refs = [add_reference(gallery, identity, red=value) for value in (90, 120)]
    other = {**library.get_asset("asset_one"), "id": "asset_other", "work_id": "work_other",
             "series": "Other series", "character_scope_id": "work_other"}
    for field in ("project_id", "membership_revision", "anime_profile"):
        other.pop(field, None)
    atomic_json(library.asset_dir("asset_other") / "manifest.json", other)
    foreign = characters.create("asset_other", {"name": "Foreign character"})["id"]
    pipeline = VisualIdentityPipeline(library)
    result = {"status": "completed", "regions": [region("region_a", "frame_a", identity, refs[0]["id"])]}
    current = pipeline.project_result("asset_one", result, frame_ids=["frame_a"])
    assert {row["reference_id"] for row in current["references"]} == {row["id"] for row in refs}
    assert current["regions"][0]["crop_path"] == str(pipeline._directory("asset_one") / "crops" / "region_a.png")
    assert "foreign-asset" not in current["regions"][0]["crop_path"]
    result["regions"][0]["match"].update(character_id=foreign, candidate_character_id=foreign)
    rejected = pipeline.project_result("asset_one", result, frame_ids=["frame_a"])
    assert rejected["regions"][0]["match"]["status"] == "unknown"
    assert rejected["regions"][0]["match"]["character_id"] is None
