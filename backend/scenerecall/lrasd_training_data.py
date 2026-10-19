"""Auditable LR-ASD training data and metrics; importing this module needs no ML runtime.

Manifest coordinates use 25 fps and half-open frame intervals. Labels are 0/1,
with -1 for uncertain frames. Visual assets contain full-body character crops;
only an explicit saved human review supplies human frame ASD supervision.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import wave


PREPROCESSING = "lrasd-gray112-pcm16-mfcc13-25fps-fullbody-v1"
BUNDLE_SCHEMA = "scenerecall-lrasd-bundle-v1"
ARCHITECTURE = "lr-asd-original-168-v1"
SCORE_DOMAIN = "active_speaker_probability"
TEACHER_LABEL_SOURCE = "ensemble_teacher_weak"
OFFICIAL_LABEL_SOURCE = "cmdam_official_sentence_alignment"
MIXED_LABEL_SOURCE = "mixed_official_and_teacher"
HUMAN_LABEL_SOURCE = "human_frame_asd"
HUMAN_REPLAY_LABEL_SOURCE = "mixed_human_and_official"


def label_semantics(source: str) -> str:
    return {TEACHER_LABEL_SOURCE: "teacher_consensus_agreement",
            OFFICIAL_LABEL_SOURCE: "official_sentence_speaker_alignment_agreement",
            HUMAN_LABEL_SOURCE: "human_reviewed_frame_asd_agreement"}[source]


def validate_human_provenance(row: dict) -> None:
    """A declaration alone cannot turn generated labels into human supervision."""
    annotation = row.get("human_annotation", {})
    labels = row.get("labels")
    if not isinstance(labels, list) or not labels or any(type(label) is not int or label not in {-1, 0, 1} for label in labels):
        raise ValueError("Human frame ASD labels must be explicit integer -1/0/1 values")
    provenance = annotation.get("track_provenance", {}) if isinstance(annotation, dict) else {}
    if (row.get("human_annotated") is not True or row.get("human_frame_asd_ground_truth") is not True
            or row.get("machine_generated") is not False or row.get("synthetic") is not False
            or row.get("annotation_granularity") != "human_reviewed_frame_asd_and_body_boxes"
            or not isinstance(annotation, dict) or annotation.get("reviewed") is not True
            or annotation.get("excluded") is not False or type(annotation.get("revision")) is not int
            or annotation["revision"] < 1 or not annotation.get("reviewed_at")
            or annotation.get("reviewer") != {"kind": "human"}
            or not isinstance(provenance, dict) or provenance.get("boxes_reviewed") is not True
            or provenance.get("labels_source") != "human_input" or provenance.get("model_scores_copied") is not False):
        raise ValueError("Human frame ASD requires a saved, explicit human review and human-input provenance")
    for digest in (row.get("source_fingerprint"), annotation.get("annotation_sha256")):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Human frame ASD requires source and annotation SHA256 bindings")
    if row.get("source_video_sha256") != row["source_fingerprint"]:
        raise ValueError("Human frame ASD source SHA256 must match its fingerprint")


def supervision_summary(rows: list[dict], label_source: str) -> dict:
    """Validate supervision provenance without calling projected labels human ASD truth."""
    modes = {TEACHER_LABEL_SOURCE: {TEACHER_LABEL_SOURCE}, OFFICIAL_LABEL_SOURCE: {OFFICIAL_LABEL_SOURCE},
             MIXED_LABEL_SOURCE: {TEACHER_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE},
             HUMAN_LABEL_SOURCE: {HUMAN_LABEL_SOURCE},
             HUMAN_REPLAY_LABEL_SOURCE: {HUMAN_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE}}
    if label_source not in modes:
        raise ValueError(f"Unsupported label_source: {label_source}")
    observed = {row.get("label_source") for row in rows}
    if observed != modes[label_source]:
        detail = "mixed requires both human and official rows" if label_source == HUMAN_REPLAY_LABEL_SOURCE else "mixed requires both official and teacher rows"
        raise ValueError("Every row label_source must match the requested supervision mode; " + detail)
    counts = {split: {} for split in ("train", "val", "test")}
    for row in rows:
        source = row["label_source"]
        if source == HUMAN_LABEL_SOURCE:
            validate_human_provenance(row)
        elif row.get("human_annotated") is not False or any(row.get(key) is True for key in (
                "human_ground_truth", "human_frame_ground_truth", "human_frame_asd_ground_truth")):
            raise ValueError("These supervision modes require human_annotated=false: neither is manually verified frame-level ASD ground truth")
        if source == TEACHER_LABEL_SOURCE:
            if row.get("machine_generated") is not True:
                raise ValueError("Teacher supervision requires machine_generated=true")
        elif source == OFFICIAL_LABEL_SOURCE and (row.get("official_annotations") is not True
              or row.get("annotation_granularity") != "sentence_speaker_and_character_boxes"):
            raise ValueError("Official CMD-AM supervision requires official_annotations=true and sentence_speaker_and_character_boxes granularity")
        stats = counts[row["split"]].setdefault(source, {
            "windows": 0, "positive_frames": 0, "negative_frames": 0, "masked_frames": 0})
        stats["windows"] += 1
        for label, key in ((1, "positive_frames"), (0, "negative_frames"), (-1, "masked_frames")):
            stats[key] += row["labels"].count(label)
    sources = {}
    for source in sorted(observed):
        sources[source] = {key: sum(split.get(source, {}).get(key, 0) for split in counts.values())
                           for key in ("windows", "positive_frames", "negative_frames", "masked_frames")}
    descriptions = {
        TEACHER_LABEL_SOURCE: ("teacher_consensus_agreement", "machine_teacher_consensus_weak",
                              "Labels are machine teacher consensus, not manually verified frame-level ASD ground truth."),
        OFFICIAL_LABEL_SOURCE: ("official_sentence_speaker_alignment_agreement", "official_sentence_alignment_projected_to_frames",
                               "Official sentence-speaker and character-box annotations are projected to frames; they are not manually verified frame-level ASD ground truth."),
        MIXED_LABEL_SOURCE: ("mixed_official_alignment_and_teacher_agreement", "mixed_official_sentence_alignment_and_teacher_weak",
                             "Training mixes official sentence-speaker/box alignment with machine teacher consensus; neither supplies manually verified frame-level ASD ground truth."),
        HUMAN_LABEL_SOURCE: ("human_reviewed_frame_asd_agreement", "human_reviewed_frame_asd",
                             "Metrics describe this manually reviewed sample and its declared scene split, not cross-title generalization."),
        HUMAN_REPLAY_LABEL_SOURCE: ("mixed_human_asd_and_official_alignment_agreement", "human_reviewed_asd_with_official_alignment_replay",
                                    "Human frame ASD and projected official sentence alignment remain separate supervision sources; pooled scores are not pure human ASD accuracy."),
    }
    semantics, quality, limitation = descriptions[label_source]
    split_semantics = {}
    for split, sources_present in counts.items():
        mode = (HUMAN_REPLAY_LABEL_SOURCE if HUMAN_LABEL_SOURCE in sources_present else MIXED_LABEL_SOURCE) if len(sources_present) > 1 else next(iter(sources_present), None)
        split_semantics[split] = descriptions[mode][0] if mode else "no_supervised_frames"
    summary = {"mode": label_source, "sources": sources, "splits": counts,
            "split_semantics": split_semantics,
            "official_annotations_present": OFFICIAL_LABEL_SOURCE in observed,
            "teacher_labels_present": TEACHER_LABEL_SOURCE in observed,
            "human_frame_ground_truth": observed == {HUMAN_LABEL_SOURCE},
            "semantics": semantics, "label_quality": quality,
            "limitation": limitation + " Held-out agreement does not establish Japanese anime accuracy, calibration, or cross-title generalization."}
    if HUMAN_LABEL_SOURCE in observed:
        summary["human_frame_ground_truth_present"] = True
        summary["limitation"] = limitation + " Same-film scene holdout does not establish cross-title performance or probability calibration; baseline prior film exposure is not ruled out."
    return summary


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_annotation_plan(path: Path | str) -> dict:
    """Freeze scene membership before labels are collected; never random-split windows."""
    plan = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict) or not isinstance(plan.get("items"), list) or not plan["items"]:
        raise ValueError("Annotation plan needs a nonempty items list")
    fingerprint = plan.get("source_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
        raise ValueError("Annotation plan requires the full source SHA256 fingerprint")
    gap = plan.get("minimum_cross_split_gap_ms", 30000)
    if type(gap) is not int or gap < 30000:
        raise ValueError("Scene holdout needs a minimum cross-split gap of at least 30000 ms")
    ids, clips, scenes = set(), set(), {}
    for item in plan["items"]:
        if not isinstance(item, dict):
            raise ValueError("Plan items must be objects")
        for key in ("id", "clip_id", "scene_id"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Plan item requires {key}")
        if item["id"] in ids or item["clip_id"] in clips:
            raise ValueError("Plan item IDs and clip IDs must be unique")
        ids.add(item["id"])
        clips.add(item["clip_id"])
        if item.get("split") not in {"train", "val", "test"}:
            raise ValueError("Plan split must be train, val or test")
        start, end = item.get("start_ms"), item.get("end_ms")
        if type(start) is not int or type(end) is not int or start < 0 or start % 40 or end % 40 or not 640 <= end - start <= 8000:
            raise ValueError("Plan range must align to the 25fps grid and last 0.64–8 seconds")
        if item["scene_id"] in scenes and scenes[item["scene_id"]] != item["split"]:
            raise ValueError("A scene cannot occur in multiple splits")
        scenes[item["scene_id"]] = item["split"]
    _check_temporal_holdout([{**item, "source_fingerprint": fingerprint} for item in plan["items"]], gap)
    return {**plan, "minimum_cross_split_gap_ms": gap, "plan_sha256": canonical_hash(plan)}


def _check_temporal_holdout(intervals: list[dict], gap: int) -> None:
    for index, left in enumerate(intervals):
        for right in intervals[index + 1:]:
            if left["split"] == right["split"]:
                continue
            if left.get("source_fingerprint") != right.get("source_fingerprint"):
                continue
            distance = max(right["start_ms"] - left["end_ms"], left["start_ms"] - right["end_ms"])
            if distance < gap:
                raise ValueError("Cross-split source intervals overlap or are too close for the declared scene holdout")


def validate_scene_holdout_rows(rows: list[dict]) -> None:
    human = [row for row in rows if row.get("label_source") == HUMAN_LABEL_SOURCE]
    if not any(row["split"] != "train" for row in human):
        return
    intervals, scenes, plans = [], {}, set()
    for row in human:
        plan = row.get("annotation_plan", {})
        if (not isinstance(plan, dict) or plan.get("split") != row["split"]
                or not isinstance(plan.get("sha256"), str) or len(plan["sha256"]) != 64
                or not isinstance(plan.get("scene_id"), str) or not plan["scene_id"]
                or type(plan.get("minimum_cross_split_gap_ms")) is not int
                or plan["minimum_cross_split_gap_ms"] < 30000):
            raise ValueError("Human validation/test requires a fixed scene annotation plan")
        plans.add(plan["sha256"])
        scene = (row.get("source_fingerprint"), plan["scene_id"])
        if scene in scenes and scenes[scene] != row["split"]:
            raise ValueError("Human scene leakage across splits")
        scenes[scene] = row["split"]
        if row.get("evaluation_group") != "human-scene-" + canonical_hash(list(scene))[:24]:
            raise ValueError("Human evaluation_group does not match its source scene")
        start, end = row.get("source_start_ms"), row.get("source_end_ms")
        if type(start) is not int or type(end) is not int or start < 0 or end - start != len(row["labels"]) * 40:
            raise ValueError("Human source interval must match its 25fps labels")
        intervals.append({"split": row["split"], "start_ms": start, "end_ms": end,
                          "source_fingerprint": row.get("source_fingerprint")})
    if len(plans) != 1:
        raise ValueError("One human round must use one unchanged annotation plan")
    _check_temporal_holdout(intervals, max(row["annotation_plan"]["minimum_cross_split_gap_ms"] for row in human))


def atomic_json(path: Path | str, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_asset(manifest_path: Path | str, asset: str) -> Path:
    if not isinstance(asset, str) or not asset:
        raise ValueError("Asset path must be a nonempty string")
    path = Path(asset)
    return path if path.is_absolute() else Path(manifest_path).resolve().parent / path


def load_manifest(path: Path | str, *, frames: int = 100, check_assets: bool = True,
                  allow_within_show_group_split: bool = False) -> list[dict]:
    """Validate fixed-length real-data windows and enforce disjoint movie/show groups."""
    path = Path(path)
    rows, ids, groups, movies, clips = [], set(), {}, {}, {}
    required = {"sample_id", "split", "movie", "clip", "track", "start_frame", "end_frame", "fps",
                "visual_path", "audio_path", "audio_start_seconds", "labels"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or required - row.keys():
            raise ValueError(f"Manifest line {line_number} is missing required fields")
        if row.get("schema_version", 1) != 1:
            raise ValueError("Unsupported LR-ASD manifest schema")
        for field in ("sample_id", "movie", "clip", "track"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        if row["sample_id"] in ids:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        ids.add(row["sample_id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        # Explicit episode holdout requires an evaluation_group and still forbids
        # splitting one source clip across train/validation/test.
        group = row.get("evaluation_group", row.get("group", row["movie"]))
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a nonempty movie/show identity")
        if allow_within_show_group_split and not row.get("evaluation_group", row.get("group")):
            raise ValueError("Within-show splitting requires an explicit episode evaluation_group/group")
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Movie/show group leakage across splits: {group}")
        groups[group] = row["split"]
        if not allow_within_show_group_split and row["movie"] in movies and movies[row["movie"]] != row["split"]:
            raise ValueError(f"Movie/show leakage requires explicit within-show holdout opt-in: {row['movie']}")
        movies[row["movie"]] = row["split"]
        clip_identity = (row["movie"], row["clip"])
        if clip_identity in clips and clips[clip_identity] != row["split"]:
            raise ValueError(f"Source clip leakage across splits: {row['clip']}")
        clips[clip_identity] = row["split"]
        for field in ("start_frame", "end_frame"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if row["end_frame"] - row["start_frame"] != frames:
            raise ValueError(f"Every window must contain exactly {frames} frames")
        if not isinstance(row["fps"], (int, float)) or not math.isfinite(row["fps"]) or abs(row["fps"] - 25) > 1e-6:
            raise ValueError("This preprocessing version requires resampling to exactly 25 fps")
        offset = row["audio_start_seconds"]
        if isinstance(offset, bool) or not isinstance(offset, (float, int)) or not math.isfinite(offset) or offset < 0:
            raise ValueError("audio_start_seconds must be finite and nonnegative")
        labels = row["labels"]
        if not isinstance(labels, list) or len(labels) != frames or any(type(x) is not int or x not in {-1, 0, 1} for x in labels):
            raise ValueError("labels must contain one integer -1/0/1 per frame")
        if "mask" in row:
            if not isinstance(row["mask"], list) or len(row["mask"]) != frames or any(type(x) is not bool for x in row["mask"]):
                raise ValueError("mask must contain one boolean per frame")
            row["labels"] = [label if keep else -1 for label, keep in zip(labels, row["mask"])]
        if not any(x >= 0 for x in row["labels"]):
            raise ValueError("A training window must contain at least one unmasked label")
        if check_assets:
            for field in ("visual_path", "audio_path"):
                asset = resolve_asset(path, row[field])
                if not asset.exists():
                    raise ValueError(f"Missing {field}: {asset}")
                if row.get("label_source") == HUMAN_LABEL_SOURCE:
                    expected = row.get("asset_sha256", {}).get("visual" if field == "visual_path" else "audio")
                    if not isinstance(expected, str) or sha256_file(asset) != expected:
                        raise ValueError(f"Human exported asset SHA256 mismatch: {asset}")
        rows.append(row)
    if not rows:
        raise ValueError("Empty training manifest")
    validate_scene_holdout_rows(rows)
    return rows


def split_summary(rows: list[dict]) -> dict:
    result = {}
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split]
        result[split] = {
            "windows": len(selected), "movies": sorted({r["movie"] for r in selected}),
            "groups": sorted({r.get("evaluation_group", r.get("group", r["movie"])) for r in selected}),
            "positive_frames": sum(r["labels"].count(1) for r in selected),
            "negative_frames": sum(r["labels"].count(0) for r in selected),
            "masked_frames": sum(r["labels"].count(-1) for r in selected),
            "track_frames": sum(len(r["labels"]) for r in selected),
            "sha256": canonical_hash(sorted(selected, key=lambda r: r["sample_id"])),
        }
    return result


def assign_group_splits(groups, seed: int = 114) -> dict[str, str]:
    """Deterministic movie/show splits for a preparer; never split adjacent windows."""
    groups = sorted(set(groups))
    if len(groups) < 3:
        raise ValueError("At least three independent movie/show groups are needed")
    random.Random(seed).shuffle(groups)
    holdout = max(1, round(len(groups) * 0.15))
    return {group: "test" if i < holdout else "val" if i < 2 * holdout else "train"
            for i, group in enumerate(groups)}


def load_sample(row: dict, manifest_path: Path | str, *, train: bool = False, seed: int = 114, epoch: int = 0):
    """Load uint8 grayscale crops and compute the original 13-D MFCC front end."""
    import numpy as np
    import python_speech_features

    visual_path = resolve_asset(manifest_path, row["visual_path"])
    if visual_path.suffix.lower() == ".npz":
        with np.load(visual_path, allow_pickle=False) as packed:
            visual = packed["visual"].copy()
    else:
        import cv2

        files = row.get("frame_files")
        if not isinstance(files, list) or len(files) != len(row["labels"]):
            raise ValueError("JPEG visual_path requires an explicit frame_files list")
        images = [cv2.imread(str(visual_path / filename), cv2.IMREAD_GRAYSCALE) for filename in files]
        if any(image is None for image in images):
            raise ValueError(f"Unreadable image in {visual_path}")
        visual = np.stack(images)
    expected = (len(row["labels"]), 112, 112)
    if visual.dtype != np.uint8 or visual.shape != expected:
        raise ValueError(f"visual must be uint8{expected}, received {visual.dtype}{visual.shape}")
    if train:
        # Per-window augmentation is reproducible across resume and loader workers.
        rng = random.Random(canonical_hash([seed, epoch, row["sample_id"]]))
        if rng.random() < 0.5:
            visual = visual[:, :, ::-1].copy()
    with wave.open(str(resolve_asset(manifest_path, row["audio_path"])), "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("audio_path must be uncompressed mono 16 kHz PCM16 WAV")
        first = round(row["audio_start_seconds"] * 16000)
        count = round(len(row["labels"]) / row["fps"] * 16000)
        if first + count > stream.getnframes():
            raise ValueError(f"Audio window exceeds WAV duration: {row['sample_id']}")
        stream.setpos(first)
        audio = np.frombuffer(stream.readframes(count), dtype="<i2")
    mfcc = python_speech_features.mfcc(audio, 16000, numcep=13,
                                      winlen=0.025 * 25 / row["fps"], winstep=0.010 * 25 / row["fps"])
    wanted = 4 * len(row["labels"])
    if len(mfcc) < wanted:
        mfcc = np.pad(mfcc, ((0, wanted - len(mfcc)), (0, 0)), mode="wrap")
    mfcc = np.asarray(mfcc[:wanted], dtype=np.float32)
    if not np.isfinite(mfcc).all():
        raise ValueError("Nonfinite MFCC input")
    return mfcc, visual.astype(np.float32), np.asarray(row["labels"], dtype=np.int64)


def _validate_scores(labels, scores):
    if not labels or len(labels) != len(scores):
        raise ValueError("Metric inputs must have equal nonzero lengths")
    if any(type(label) is not int or label not in {0, 1} for label in labels):
        raise ValueError("Metric labels must be unmasked integer 0/1")
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Scores must be finite probabilities in [0, 1]")


def ranking_metrics(labels: list[int], scores: list[float]) -> dict:
    """Tie-aware AP and ROC AUC, matching standard threshold-group definitions."""
    _validate_scores(labels, scores)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    ordered = sorted(zip(scores, labels), reverse=True)
    tp = fp = 0
    ap = favorable = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        p = sum(label for _, label in ordered[index:end])
        n = end - index - p
        tp += p
        fp += n
        if positives:
            ap += p / positives * tp / (tp + fp)
        # Current positives beat lower-scored negatives and tie this group's negatives.
        favorable += p * (negatives - fp + n / 2)
        index = end
    return {"ap": ap if positives else None, "auc": favorable / (positives * negatives) if positives and negatives else None}


def _counts_metrics(tp, fp, tn, fn) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": tp + fp + tn + fn,
            "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "balanced_accuracy": (recall + specificity) / 2,
            "accuracy": (tp + tn) / (tp + fp + tn + fn)}


def classification_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict:
    _validate_scores(labels, scores)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    tp = sum(label == 1 and score >= threshold for label, score in zip(labels, scores))
    fp = sum(label == 0 and score >= threshold for label, score in zip(labels, scores))
    fn = sum(labels) - tp
    tn = len(labels) - sum(labels) - fp
    return {**ranking_metrics(labels, scores), **_counts_metrics(tp, fp, tn, fn), "threshold": threshold}


def validation_selection_report(rows: list[dict], labels: list[int], scores: list[float], threshold: float,
                                selection: str = "pooled_ap") -> dict:
    """Score full validation once, keeping pooled AP distinct from source-macro AP."""
    if selection not in {"pooled_ap", "macro_source_ap"}:
        raise ValueError("Unknown validation selection metric")
    indices, position = {}, 0
    for row in rows:
        source = row["label_source"]
        count = sum(label >= 0 for label in row["labels"])
        indices.setdefault(source, []).extend(range(position, position + count))
        position += count
    if position != len(labels) or len(labels) != len(scores):
        raise ValueError("Validation provenance does not align with unmasked prediction frames")
    by_source = {}
    for source, selected in sorted(indices.items()):
        targets = [labels[index] for index in selected]
        if selection == "macro_source_ap" and (0 not in targets or 1 not in targets):
            raise ValueError("macro_source_ap requires both validation classes for every label source")
        by_source[source] = classification_metrics(targets, [scores[index] for index in selected], threshold)
    pooled = classification_metrics(labels, scores, threshold)
    value = pooled["ap"] if selection == "pooled_ap" else sum(item["ap"] for item in by_source.values()) / len(by_source)
    return {"metrics": pooled, "by_label_source": by_source, "selection_metric": selection, "selection_value": value}


def select_threshold(labels: list[int], scores: list[float]) -> float:
    """Choose balanced accuracy on validation only, with ties closest to 0.5."""
    _validate_scores(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Threshold selection requires both validation classes")
    best = 0.5
    best_bacc = classification_metrics(labels, scores, best)["balanced_accuracy"]
    tp = fp = 0
    ordered = sorted(zip(scores, labels), reverse=True)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        p = sum(label for _, label in ordered[index:end])
        tp += p
        fp += end - index - p
        threshold = ordered[index][0]
        bacc = (tp / positives + (negatives - fp) / negatives) / 2
        if bacc > best_bacc + 1e-12 or (abs(bacc - best_bacc) < 1e-12 and abs(threshold - 0.5) < abs(best - 0.5)):
            best, best_bacc = threshold, bacc
        index = end
    return best
