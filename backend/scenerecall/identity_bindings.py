"""Versioned, read-only identity decisions for retrieval observations.

Reference trust and a current observation's identity are independent. A v2
calibration must name this policy and preprocessing and contain held-out
``validation`` metrics (auto_precision >= .98, eligible_coverage >= .60 and
cross_character_error_rate <= .01). A separately evaluated ``single_frame`` section
can permit one-frame decisions; otherwise two explicitly associated regions
from distinct frames of one shot must independently pass calibrated matching.
No self-reported model confidence, profile enrichment or gallery enrollment
is performed here. Validation fixtures are not animation-quality evidence.
"""
from __future__ import annotations

import copy
import hashlib
import math

from .library import confined_path, digest, read_json, safe_id

POLICY_VERSION = "identity-binding-v2"
CHARACTER_KINDS = {"person", "character", "anthropomorphic_character"}


def uses_policy(observation):
    return observation.get("identity_policy", observation.get("identity_policy_version")) == POLICY_VERSION


def character_entity(entity):
    return entity.get("kind", "person") in CHARACTER_KINDS and entity.get("is_character") is not False


def stamp_match(match, references, config):
    """Attach input provenance to new-policy regions without changing old matches."""
    if config.get("identity_policy") != POLICY_VERSION:
        return match
    from .visual_identity import PREPROCESS_VERSION

    result = copy.deepcopy(match)
    used = {key for candidate in match.get("candidates", []) for key in candidate.get("reference_ids", [])}
    result.update(binding_policy=POLICY_VERSION, preprocessing=PREPROCESS_VERSION,
                  calibration_signature=digest(config.get("calibration")),
                  reference_evidence=[{key: reference[key] for key in ("id", "version", "sha256", "character_id")}
                                      for reference in references if reference["id"] in used])
    return result


def _held_out(value):
    if not isinstance(value, dict) or value.get("split") != "held_out":
        return False
    metrics = [value.get(key) for key in ("auto_precision", "eligible_coverage", "cross_character_error_rate")]
    if any(type(number) not in {int, float} or not math.isfinite(number) or not 0 <= number <= 1
           for number in metrics):
        return False
    return metrics[0] >= .98 and metrics[1] >= .60 and metrics[2] <= .01


class BindingResolver:
    """Reuse one asset's registry/reference snapshot across a projection batch."""

    def __init__(self, library, asset_id):
        from .characters import CharacterStore, scope_id
        from .visual_identity import ReferenceGallery, VisualIdentityPipeline

        self.library, self.asset_id = library, asset_id
        self.asset = library.get_asset(asset_id)
        self.characters = CharacterStore(library)
        self.scope = scope_id(self.asset)
        self.registry = self.characters._read(self.scope)
        policy = self.characters.cast_policy(self.registry, self.asset.get("anime_profile"))
        self.profiles = {key: row for key, row in self.registry["profiles"].items()
                         if self.characters.profile_allowed(row, policy)}
        self.pipeline = VisualIdentityPipeline(library)
        self.directory = self.pipeline._directory(asset_id)
        self.settings = self.pipeline.settings(asset_id)
        self.decisions = read_json(self.directory / "decisions.json", {"decisions": {}})["decisions"]
        gallery = ReferenceGallery(library)
        self.gallery_directory = gallery._directory(self.scope)
        self.references = {row["id"]: row for row in gallery._read(self.scope)["references"]}
        self.trusted = gallery.active_reference_ids(asset_id)
        self.matching = gallery.active_reference_ids(asset_id, include_candidates=True)
        self.frames = read_json(library.asset_dir(asset_id) / "frames" / "registry.json", {})
        self.regions, self.reference_hashes = {}, {}

    def canonical(self, identity):
        if not identity:
            return None
        try:
            value = self.characters._resolve(self.registry, identity)
        except ValueError:
            return None
        return value if value in self.profiles else None

    def region(self, identity, frame_ids):
        if not identity:
            return None
        try:
            identity = safe_id(identity)
        except ValueError:
            return None
        if identity not in self.regions:
            self.regions[identity] = read_json(self.directory / "regions" / f"{identity}.json")
        row = self.regions[identity]
        if (not row or row.get("id") != identity or row.get("frame_id") not in frame_ids
                or row["frame_id"] not in self.frames
                or self.frames[row["frame_id"]]["at_ms"] != row.get("at_ms")):
            return None
        return row

    def reference_valid(self, evidence, identity):
        key = evidence.get("id")
        current = self.references.get(key)
        if (not current or key not in self.trusted or self.canonical(current["character_id"]) != identity
                or any(current.get(field) != evidence.get(field) for field in ("version", "sha256"))):
            return False
        if key not in self.reference_hashes:
            try:
                path = confined_path(self.gallery_directory, f"{safe_id(key)}.png")
                self.reference_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError):
                self.reference_hashes[key] = None
        return self.reference_hashes[key] == evidence["sha256"]

    def calibrated(self, region, identity, *, single=False):
        from .visual_identity import PREPROCESS_VERSION

        match = region.get("match", {})
        calibration = self.settings.get("calibration") or {}
        if (match.get("status") != "confirmed" or match.get("source") != "ccip_calibrated"
                or self.canonical(match.get("character_id")) != identity
                or not region.get("quality", {}).get("accepted")
                or region.get("source_fingerprint") != self.asset["fingerprint"]
                or match.get("binding_policy") != POLICY_VERSION
                or calibration.get("binding_policy") != POLICY_VERSION
                or calibration.get("preprocessing") != PREPROCESS_VERSION
                or match.get("preprocessing") != PREPROCESS_VERSION
                or calibration.get("model_version") != match.get("model_version")
                or match.get("model_version") != region.get("model_version")
                or calibration.get("aggregation") != match.get("aggregation")
                or match.get("calibration_id") != calibration.get("evaluation_id")
                or match.get("calibration_signature") != digest(calibration)
                or not calibration.get("dataset_id") or not calibration.get("evaluation_id")
                or not _held_out(calibration.get("validation"))):
            return False
        thresholds = calibration
        if single:
            thresholds = calibration.get("single_frame") or {}
            if (not thresholds.get("evaluation_id") or thresholds["evaluation_id"] == calibration["evaluation_id"]
                    or not _held_out(thresholds.get("validation"))):
                return False
        distance, margin = match.get("distance"), match.get("margin")
        maximum, minimum = thresholds.get("max_distance"), thresholds.get("min_margin")
        if (any(type(value) not in {int, float} or not math.isfinite(value) or value < 0
                for value in (distance, margin, maximum, minimum))
                or distance > maximum or margin < minimum):
            return False
        best = (match.get("candidates") or [{}])[0]
        used = best.get("reference_ids", [])
        provenance = {row["id"]: row for row in match.get("reference_evidence", [])}
        if (self.canonical(best.get("character_id")) != identity or not used
                or any(key not in provenance or not self.reference_valid(provenance[key], identity) for key in used)):
            return False
        # Different IDs pointing to duplicate pixels do not supply independent views.
        return len({provenance[key]["sha256"] for key in used}) >= 2

    def _decision(self, entity, observation, claims):
        frame_ids = set(entity.get("evidence_frame_ids") or [])
        if not frame_ids or not frame_ids.issubset(observation.get("evidence_frame_ids", [])):
            return {"status": "unknown", "reason": "人物缺少当前观察的独立画面证据"}
        keys = list(dict.fromkeys([entity.get("identity_region_id"),
                                  *(entity.get("identity_support_region_ids") or [])]))
        keys = [key for key in keys if key]
        if any(len(claims.get(key, [])) > 1 for key in keys):
            return {"status": "unknown", "reason": "同一区域被多个局部实体引用，人物归属存在歧义"}
        regions = [self.region(key, frame_ids) for key in keys]
        if not regions or any(row is None for row in regions):
            return {"status": "unknown", "reason": "缺少有效的当前人物区域"}
        main = regions[0]
        manual = [self.decisions[key] for key in keys if key in self.decisions]
        if manual:
            named = {self.canonical(row.get("character_id")) for row in manual if row.get("status") == "confirmed"}
            candidates = {self.canonical(row.get("candidate_character_id")) for row in manual
                          if row.get("status") == "candidate"}
            if (any(row.get("status") == "unknown" for row in manual) or len(named) > 1 or len(candidates) > 1
                    or named and candidates and named != candidates):
                return {"status": "unknown", "source": "human", "reason": "人工保留未知或局部区域的人工决定存在冲突",
                        "manual_versions": [row.get("version") for row in manual]}
            if len(named) == 1 and None not in named:
                return {"status": "human_confirmed", "character_id": next(iter(named)), "source": "human",
                        "reason": "人工核对当前人物区域", "manual_versions": [row.get("version") for row in manual]}
            candidate = self.canonical(manual[0].get("candidate_character_id"))
            return {"status": "candidate" if candidate else "unknown", "candidate_character_id": candidate,
                    "source": "human", "reason": "保留人工候选决定，不由自动匹配覆盖"}
        match = main.get("match", {})
        identity = self.canonical(match.get("character_id") or match.get("candidate_character_id"))
        best = (match.get("candidates") or [{}])[0]
        if not identity and match.get("status") != "unknown":
            identity = self.canonical(best.get("character_id"))
        used = best.get("reference_ids", [])
        if not identity or used and not set(used).issubset(self.matching):
            return {"status": "unknown", "reason": "没有有效候选，或支持参考已撤销/离开作品作用域"}
        current_frames = {frame["id"]: frame for frame in observation.get("ordered_frames", [])}
        shots = {current_frames.get(row["frame_id"], {}).get("shot_id") for row in regions}
        pair = (len({row["frame_id"] for row in regions}) >= 2 and len({row["at_ms"] for row in regions}) >= 2
                and len(shots) == 1 and None not in shots and "" not in shots
                and all(self.calibrated(row, identity) for row in regions))
        single = len(regions) == 1 and self.calibrated(main, identity, single=True)
        if pair or single:
            return {"status": "auto_bound", "character_id": identity, "source": POLICY_VERSION,
                    "reason": "当前同镜头双观察和可信参考通过留出校准" if pair else "当前观察通过独立单帧留出校准",
                    "support_mode": "ordered_pair" if pair else "single_frame",
                    "calibration_id": (self.settings["calibration"]["single_frame"]["evaluation_id"] if single
                                       else match["calibration_id"]),
                    "reference_evidence": copy.deepcopy(match["reference_evidence"]),
                    "region_ids": keys, "model_version": match["model_version"]}
        return {"status": "candidate", "candidate_character_id": identity, "source": match.get("source"),
                "reason": "候选尚未满足本次绑定的校准、参考或有序画面支持条件",
                "region_ids": keys, "distance": match.get("distance"), "margin": match.get("margin")}

    def resolve(self, observation):
        if not uses_policy(observation):
            return copy.deepcopy(observation)
        if observation.get("asset_id", self.asset_id) != self.asset_id:
            raise ValueError("身份观察不属于当前视频")
        result = copy.deepcopy(observation)
        claims = {}
        for entity in result.get("entities", []):
            if not character_entity(entity):
                continue
            for key in set([entity.get("identity_region_id"), *(entity.get("identity_support_region_ids") or [])]):
                if key:
                    claims.setdefault(key, []).append(entity.get("id"))
        for entity in result.get("entities", []):
            if not character_entity(entity):
                entity["character_id"] = None
                for key in ("character_name", "candidate_character_id", "identity_binding", "binding_status",
                            "identity_status", "association_status", "association_source"):
                    entity.pop(key, None)
                continue
            decision = self._decision(entity, result, claims)
            decision.update(policy_version=POLICY_VERSION, observation_id=result.get("id"), entity_id=entity.get("id"),
                            character_scope_id=self.scope)
            decision["id"] = "binding_" + digest(decision)
            identity = decision.get("character_id")
            bound = decision["status"] in {"auto_bound", "human_confirmed"}
            entity.update(identity_binding=decision, binding_status=decision["status"],
                          identity_status="confirmed" if bound else decision["status"],
                          association_status="confirmed" if bound else decision["status"],
                          association_source=decision.get("source", POLICY_VERSION),
                          character_id=identity, candidate_character_id=decision.get("candidate_character_id"),
                          match_confidence=None, name=None, character_name=None, aliases=[],
                          candidate_character_name=None, candidate_character_aliases=[])
            if identity:
                profile = self.profiles[identity]
                entity.update(name=profile["name"], character_name=profile["name"], aliases=list(profile["aliases"]))
            candidate = decision.get("candidate_character_id")
            if candidate:
                profile = self.profiles[candidate]
                entity.update(candidate_character_name=profile["name"],
                              candidate_character_aliases=list(profile["aliases"]))
        result["character_scope_id"] = self.scope
        result["character_revision"] = self.registry["revision"]
        from .identity_scene import project_actions
        return project_actions(result)


def resolve_observation(library, asset_id, observation):
    if not uses_policy(observation):
        return copy.deepcopy(observation)
    return BindingResolver(library, asset_id).resolve(observation)


def resolve_observations(library, asset_id, observations):
    if not any(uses_policy(row) for row in observations):
        return copy.deepcopy(observations)
    resolver = BindingResolver(library, asset_id)
    return [resolver.resolve(row) for row in observations]
