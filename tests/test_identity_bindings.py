"""Synthetic decision-contract tests; not model calibration or accuracy claims."""
from copy import deepcopy

import pytest

from scenerecall.identity_bindings import POLICY_VERSION, resolve_observation
from scenerecall.identity_scene import bind_identity_evidence
from scenerecall.library import read_json
from scenerecall.visual_identity import PREPROCESS_VERSION, VisualIdentityPipeline
from test_visual_identity import FakeCCIP, add_reference, add_role, frame, visual  # noqa: F401


@pytest.fixture
def calibrated(visual):  # noqa: F811
    library, characters, gallery = visual
    identity = add_role(characters, "Aoi")
    refs = [add_reference(gallery, identity, red=value) for value in (80, 90)]
    other = add_role(characters, "Other")
    add_reference(gallery, other, red=220)
    add_reference(gallery, other, red=230)
    calibration = {"model_version": FakeCCIP.model_version, "dataset_id": "synthetic-fixture",
        "evaluation_id": "fixture-pair", "aggregation": "nearest-two-mean-v1",
        "max_distance": .1, "min_margin": .1, "binding_policy": POLICY_VERSION,
        "preprocessing": PREPROCESS_VERSION,
        "validation": {"split": "held_out", "auto_precision": .99, "eligible_coverage": .8,
                       "cross_character_error_rate": 0}}
    backend = FakeCCIP()
    pipeline = VisualIdentityPipeline(library, backend)
    settings = pipeline.configure("asset_one", {"enabled": True, "calibration": calibration})
    frames = [frame(library, "frame_a", red=85, at_ms=200), frame(library, "frame_b", red=86, at_ms=600)]
    current = pipeline.run("asset_one", frames, {**settings, "identity_policy": POLICY_VERSION})
    observation = {"id": "obs_one", "asset_id": "asset_one", "run_id": "run_one", "window_id": "win_one",
        "shot_id": "shot_one", "start_ms": 0, "end_ms": 2000, "summary": "人物举手", "identity_policy": POLICY_VERSION,
        "evidence_frame_ids": [row["id"] for row in frames],
        "ordered_frames": [{**row, "shot_id": "shot_one"} for row in frames],
        "entities": [{"id": "P1", "kind": "person", "is_character": True, "description": "举手的人",
            "evidence_frame_ids": [row["id"] for row in frames], "identity_region_id": current["regions"][0]["id"],
            "identity_support_region_ids": [current["regions"][1]["id"]], "observed_traits": ["新外观"],
            "character_id": None, "match_confidence": .999}],
        "events": [{"actor_ids": ["P1"], "target_ids": [], "action": "举手", "start_ms": 200, "end_ms": 600,
                    "evidence_frame_ids": [row["id"] for row in frames]}]}
    return library, characters, gallery, pipeline, backend, identity, other, refs, observation


def test_valid_automatic_match_projects_without_rewriting_observations_or_features(calibrated):
    library, characters, _gallery, pipeline, backend, identity, _other, _refs, observation = calibrated
    original = deepcopy(observation)
    features = backend.feature_calls
    result = resolve_observation(library, "asset_one", observation)
    person = result["entities"][0]
    assert person["binding_status"] == "auto_bound"
    assert person["character_id"] == identity and person["character_name"] == "Aoi"
    assert person["match_confidence"] is None
    assert result["events"][0]["actor_ids"] == ["P1"]
    assert result["events"][0]["actor_names"] == ["Aoi"]
    assert result["summary"] == observation["summary"] and observation == original
    assert backend.feature_calls == features
    # The shared decoration path can promote a previously saved candidate too.
    decorated = characters.decorate_observations("asset_one", [observation])[0]
    assert decorated["entities"][0]["binding_status"] == "auto_bound"
    assert pipeline.settings("asset_one")["calibration"]["dataset_id"] == "synthetic-fixture"


def test_automatic_observation_never_enriches_dossier_or_enrolls_reference(calibrated):
    library, characters, gallery, _pipeline, _backend, identity, _other, _refs, observation = calibrated
    scope = characters.catalog("asset_one")["scope_id"]
    before_registry = read_json(characters._path(scope))
    before_gallery = gallery.list("asset_one")
    saved = characters.apply_observation("asset_one", observation)
    assert saved["entities"][0]["character_id"] == identity
    assert read_json(characters._path(scope)) == before_registry
    assert gallery.list("asset_one") == before_gallery


def test_human_confirmation_and_unknown_override_automatic_results(calibrated):
    library, _characters, _gallery, pipeline, _backend, _identity, other, _refs, observation = calibrated
    region = observation["entities"][0]["identity_region_id"]
    pipeline.review("asset_one", region, status="confirmed", character_id=other, reason="人工核对")
    projected = resolve_observation(library, "asset_one", observation)["entities"][0]
    assert projected["binding_status"] == "human_confirmed" and projected["character_id"] == other
    pipeline.review("asset_one", region, status="unknown", reason="人工撤销")
    projected = resolve_observation(library, "asset_one", observation)["entities"][0]
    assert projected["binding_status"] == "unknown" and projected["character_id"] is None
    assert projected["association_source"] == "human"


def test_revoked_reference_invalidates_automatic_but_not_explicit_human_mapping(calibrated):
    library, _characters, gallery, pipeline, backend, identity, _other, refs, observation = calibrated
    before = backend.feature_calls
    gallery.update("asset_one", identity, refs[0]["id"], status="revoked", reason="错误参考")
    assert resolve_observation(library, "asset_one", observation)["entities"][0]["binding_status"] == "unknown"
    pipeline.review("asset_one", observation["entities"][0]["identity_region_id"],
                    status="confirmed", character_id=identity, reason="独立核实原片")
    assert resolve_observation(library, "asset_one", observation)["entities"][0]["binding_status"] == "human_confirmed"
    assert backend.feature_calls == before


@pytest.mark.parametrize("change", ["calibration", "reference_version", "reference_pixels", "cross_shot", "empty_shot", "one_frame",
                                  "insufficient_precision"])
def test_missing_or_changed_automatic_evidence_remains_candidate(calibrated, change):
    library, _characters, gallery, pipeline, _backend, identity, _other, refs, observation = calibrated
    if change == "calibration":
        settings = pipeline.settings("asset_one")
        settings["calibration"]["evaluation_id"] = "changed-evaluation"
        pipeline.configure("asset_one", {"calibration": settings["calibration"]})
    elif change == "reference_version":
        gallery.update("asset_one", identity, refs[0]["id"], status="confirmed", reason="重新检查")
    elif change == "reference_pixels":
        path = gallery._directory(gallery.list("asset_one")["scope_id"]) / (refs[0]["id"] + ".png")
        path.write_bytes(b"changed pixels")
    elif change == "cross_shot":
        observation["ordered_frames"][1]["shot_id"] = "another_shot"
    elif change == "empty_shot":
        for frame_row in observation["ordered_frames"]:
            frame_row["shot_id"] = ""
    elif change == "one_frame":
        observation["entities"][0]["identity_support_region_ids"] = []
    else:
        settings = pipeline.settings("asset_one")
        settings["calibration"]["validation"]["auto_precision"] = .97
        pipeline.configure("asset_one", {"calibration": settings["calibration"]})
        frames = [{"id": frame["id"], "at_ms": frame["at_ms"]} for frame in observation["ordered_frames"]]
        pipeline.run("asset_one", frames, {**pipeline.settings("asset_one"), "identity_policy": POLICY_VERSION})
    person = resolve_observation(library, "asset_one", observation)["entities"][0]
    assert person["binding_status"] == "candidate" and person["character_id"] is None
    assert person["candidate_character_id"] == identity


def test_independent_single_frame_calibration_can_bind_without_second_current_frame(calibrated):
    library, _characters, _gallery, pipeline, _backend, identity, _other, _refs, observation = calibrated
    settings = pipeline.settings("asset_one")
    settings["calibration"]["single_frame"] = {
        "evaluation_id": "fixture-single", "max_distance": .08, "min_margin": .2,
        "validation": deepcopy(settings["calibration"]["validation"])}
    pipeline.configure("asset_one", {"calibration": settings["calibration"]})
    frames = [{"id": frame["id"], "at_ms": frame["at_ms"]} for frame in observation["ordered_frames"]]
    pipeline.run("asset_one", frames, {**pipeline.settings("asset_one"), "identity_policy": POLICY_VERSION})
    observation["entities"][0]["identity_support_region_ids"] = []
    bound = resolve_observation(library, "asset_one", observation)["entities"][0]
    assert bound["binding_status"] == "auto_bound" and bound["character_id"] == identity
    assert bound["identity_binding"]["support_mode"] == "single_frame"
    assert bound["identity_binding"]["calibration_id"] == "fixture-single"


def test_current_frame_ownership_and_duplicate_entity_claims_cannot_bind(calibrated):
    library, _characters, _gallery, _pipeline, _backend, _identity, _other, _refs, observation = calibrated
    wrong = deepcopy(observation)
    wrong["entities"][0]["evidence_frame_ids"] = ["foreign_frame"]
    assert resolve_observation(library, "asset_one", wrong)["entities"][0]["binding_status"] == "unknown"
    repeated = deepcopy(observation)
    repeated["entities"].append({**deepcopy(repeated["entities"][0]), "id": "P2"})
    assert all(row["binding_status"] == "unknown"
               for row in resolve_observation(library, "asset_one", repeated)["entities"])


def test_old_observations_and_binding_protocol_are_unchanged(calibrated):
    library, _characters, _gallery, pipeline, _backend, identity, _other, _refs, observation = calibrated
    original = deepcopy(observation)
    del original["identity_policy"]
    assert resolve_observation(library, "asset_one", original) == original
    region = pipeline.reviews("asset_one")["regions"][0]
    entity = deepcopy(observation["entities"][0])
    legacy = bind_identity_evidence([entity], {"identity_regions": [region]})[0]
    assert legacy["character_id"] == identity
    entity["identity_support_region_ids"] = []
    modern = bind_identity_evidence([entity], {"identity_policy": POLICY_VERSION, "identity_regions": [region]})[0]
    assert modern["character_id"] is None and modern["association_status"] == "candidate"


def test_rename_projects_current_name_without_recomputing_features(calibrated):
    library, characters, _gallery, _pipeline, backend, identity, _other, _refs, observation = calibrated
    before = backend.feature_calls
    characters.update("asset_one", identity, {"name": "新名称", "aliases": ["旧译名"]})
    result = resolve_observation(library, "asset_one", observation)
    assert result["entities"][0]["character_name"] == "新名称"
    assert result["entities"][0]["aliases"] == ["旧译名"]
    assert result["events"][0]["actor_names"] == ["新名称"]
    assert backend.feature_calls == before


def test_high_model_confidence_without_regions_does_not_create_binding(visual):  # noqa: F811
    library, characters, _gallery = visual
    identity = add_role(characters)
    item = {"identity_policy": POLICY_VERSION, "id": "empty", "entities": [{"id": "P1", "kind": "person",
        "character_id": identity, "match_confidence": 1, "name": "Aoi"}]}
    projected = characters.decorate_observations("asset_one", [item])[0]
    assert projected["entities"][0]["character_id"] is None
    assert projected["entities"][0]["binding_status"] == "unknown"


def test_conflicting_human_candidates_do_not_promote_one_arbitrarily(calibrated):
    library, _characters, _gallery, pipeline, _backend, identity, other, _refs, observation = calibrated
    entity = observation["entities"][0]
    pipeline.review("asset_one", entity["identity_region_id"], status="candidate",
                    character_id=identity, reason="第一帧只确定候选")
    pipeline.review("asset_one", entity["identity_support_region_ids"][0], status="candidate",
                    character_id=other, reason="第二帧候选不同")
    projected = resolve_observation(library, "asset_one", observation)["entities"][0]
    assert projected["binding_status"] == "unknown" and projected["character_id"] is None
    assert projected["association_source"] == "human"


@pytest.mark.parametrize("kind", ["object", "vehicle", "place", "symbol", "other", "animal"])
def test_noncharacter_entities_preserve_visible_features_without_identity_state(calibrated, kind):
    library, _characters, _gallery, _pipeline, _backend, identity, _other, _refs, observation = calibrated
    observation["entities"] = [{"id": "O1", "kind": kind, "name": "可见对象", "description": "银色外壳",
                               "character_id": identity, "candidate_concept_ids": ["concept_one"]}]
    entity = resolve_observation(library, "asset_one", observation)["entities"][0]
    assert entity["name"] == "可见对象" and entity["description"] == "银色外壳"
    assert entity["character_id"] is None and "association_status" not in entity and "binding_status" not in entity
    assert entity["candidate_concept_ids"] == ["concept_one"]
