#!/usr/bin/env python3
"""Freeze stage-two official CMD-AM replay plus accepted Japanese teacher labels.

Uses only the standard library. Run where the input manifests' absolute asset
paths resolve (the existing Japanese snapshot uses WSL paths). This script never
changes labels, reshuffles splits, trains a model, or activates an app bundle.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import wave
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "backend"))
from scenerecall.lrasd_training_data import (  # noqa: E402
    MIXED_LABEL_SOURCE, OFFICIAL_LABEL_SOURCE, PREPROCESSING, TEACHER_LABEL_SOURCE,
    canonical_hash, load_manifest, resolve_asset, sha256_file, split_summary,
    supervision_summary,
)

SCHEMA = "lrasd-stage2-mixed-assembly-v1"
SPLITS = ("train", "val", "test")
ASSETS = {"visual_path": "visual", "audio_path": "audio", "teacher_visual_path": "teacher_visual"}


def read_json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(data, parse_constant=invalid)


def read_source(path, source, frames):
    """Bind the completed preparation/label summary to an unchanged input snapshot."""
    path = path.resolve()
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("Input manifest must end with a complete newline; do not assemble a growing file")
    raw = [read_json(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
    validated = load_manifest(path, frames=frames, check_assets=True,
                              allow_within_show_group_split=source == TEACHER_LABEL_SOURCE)
    if path.read_bytes() != data:
        raise ValueError("Input manifest changed while it was being read")
    supervision = supervision_summary(validated, source)
    summary_path = path.parent / "label-summary.json"
    summary_data = summary_path.read_bytes()
    summary = read_json(summary_data)
    if (summary.get("training_ready") is not True or summary.get("human_ground_truth") is not False
            or summary.get("accepted_windows", summary.get("accepted_candidates")) != len(raw)):
        raise ValueError("Source summary must declare completed, sufficient, non-human-frame supervision and matching counts")
    provenance = {"manifest_path": str(path), "manifest_sha256": sha256_file(path),
                  "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path),
                  "rows": len(raw), "label_source": source, "supervision": supervision}
    if source == OFFICIAL_LABEL_SOURCE:
        if (summary.get("schema_version") != "cmdam-derived-asd-summary-v1"
                or summary.get("processed_clips") != 40 or summary.get("label_source") != source
                or summary.get("labeled_manifest_sha256") != provenance["manifest_sha256"]):
            raise ValueError("Official preparation must bind the completed 40-clip manifest SHA-256")
        config_path = path.parent / "preparation-config.json"
        config_data = config_path.read_bytes()
        config = read_json(config_data)
        if (canonical_hash(config) != summary.get("config_hash")
                or config.get("frames") != frames or config.get("preprocessing") != PREPROCESSING):
            raise ValueError("Official preparation config hash, frame count or preprocessing mismatch")
        provenance.update(config_path=str(config_path), config_sha256=sha256_file(config_path),
                          preparation_config_hash=summary["config_hash"])
        if config_path.read_bytes() != config_data:
            raise ValueError("Official preparation config changed while being read")
    else:
        config = summary.get("config", {})
        if (summary.get("schema_version") != "lrasd-machine-label-summary-v1"
                or config.get("frames") != frames or config.get("positive_threshold") != .95
                or config.get("negative_threshold") != .05 or config.get("boundary_radius") != 2
                or config.get("min_valid_frames") != math.ceil(.4 * frames)):
            raise ValueError("Japanese snapshot must retain the existing accepted teacher-consensus policy")
        for row in raw:
            weak = row.get("weak_label_metadata", {})
            if (row.get("label_accepted") is not True or weak.get("positive_threshold") != .95
                    or weak.get("negative_threshold") != .05 or weak.get("boundary_radius_frames") != 2
                    or {key: value.get("sha256") for key, value in weak.get("teachers", {}).items()}
                    != {key: value.get("sha256") for key, value in config.get("teachers", {}).items()}):
                raise ValueError("Japanese rows must be accepted under the unchanged, hash-bound teacher policy")
        provenance["teacher_config"] = config
    for row in raw:
        if row.get("synthetic") is not False or row.get("preprocessing") != PREPROCESSING:
            raise ValueError("Every input row must explicitly identify real, compatible prepared assets")
        if row.get("crop_type") not in {"fullbody", "full_body_bbox"}:
            raise ValueError("The student must retain the prepared full-body crops")
    if path.read_bytes() != data or summary_path.read_bytes() != summary_data:
        raise ValueError("Input manifest or completion summary changed during inspection")
    return raw, provenance


def audit_isolation(rows):
    """Check all input rows, including CMD training rows not selected for replay."""
    seen_ids, identities = set(), {}
    for row in rows:
        if row["sample_id"] in seen_ids:
            raise ValueError(f"Duplicate sample_id across inputs: {row['sample_id']}")
        seen_ids.add(row["sample_id"])
        keys = [("group", row.get("evaluation_group", row.get("group", row["movie"]))),
                ("clip", row["movie"], row["clip"])]
        source_hash = row.get("source_video_sha256", row.get("source_sha256"))
        if source_hash:
            keys.append(("source_video_sha256", source_hash))
        for key in keys:
            if key in identities and identities[key] != row["split"]:
                raise ValueError(f"Input group/clip/source leaks across splits: {key[0]}")
            identities[key] = row["split"]


def choose_replay(rows, count, seed):
    train = [row for row in rows if row["split"] == "train"]
    if count < 1 or count > len(train):
        raise ValueError(f"Requested {count} distinct CMD train replay windows; available {len(train)} (no replacement or silent cap)")
    # Hash ranks avoid dependence on input ordering and Python random versions.
    return sorted(train, key=lambda row: (canonical_hash([SCHEMA, seed, row["sample_id"]]), row["sample_id"]))[:count]


def verify_visual(path, frames):
    """Inspect the fixed uint8 NumPy array safely without importing NumPy."""
    with zipfile.ZipFile(path) as archive, archive.open("visual.npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError("Prepared visual asset is not a NumPy array")
        version = stream.read(2)
        if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
            raise ValueError("Unsupported prepared NumPy header version")
        size = 2 if version[0] == 1 else 4
        header_size = struct.unpack("<H" if size == 2 else "<I", stream.read(size))[0]
        if header_size > 65536:
            raise ValueError("Oversized prepared NumPy header")
        header = ast.literal_eval(stream.read(header_size).decode("utf-8" if version[0] == 3 else "latin1").strip())
        if tuple(header.get("shape", ())) != (frames, 112, 112) or header.get("descr") not in ("|u1", "<u1"):
            raise ValueError("Prepared visual must remain uint8[T,112,112]")
        expected_bytes = frames * 112 * 112
        if len(stream.read(expected_bytes + 1)) != expected_bytes:
            raise ValueError("Prepared visual data length does not match its NumPy header")


def freeze_row(row, source_path, provenance, asset_cache, frames):
    result = copy.deepcopy(row)
    expected = row.get("asset_sha256", {})
    for field, key in ASSETS.items():
        if field not in row:
            continue
        asset = resolve_asset(source_path, row[field]).resolve()
        if not asset.is_file() or not isinstance(expected.get(key), str) or len(expected[key]) != 64:
            raise ValueError(f"Missing prepared asset or recorded SHA-256: {row['sample_id']} / {field}")
        actual = asset_cache.get(asset)
        if actual is None:
            actual = sha256_file(asset)
        if actual != expected[key]:
            raise ValueError(f"Prepared asset SHA-256 changed: {row['sample_id']} / {field}")
        if asset not in asset_cache:
            if field != "audio_path":
                verify_visual(asset, frames)
            asset_cache[asset] = actual
        result[field] = str(asset)
    with wave.open(result["audio_path"], "rb") as stream:
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("Prepared audio must remain mono PCM16 at 16 kHz")
        if round(row["audio_start_seconds"] * 16000) + frames * 640 > stream.getnframes():
            raise ValueError("Prepared audio is shorter than its unchanged window")
    historical = row["label_source"] == TEACHER_LABEL_SOURCE and row["split"] == "test"
    if row.get("previously_evaluated_test") is True and not historical:
        raise ValueError("An input has additional historical test use; review the frozen stage-two recipe first")
    result["previously_evaluated_test"] = historical
    result["adaptation_provenance"] = {
        "schema_version": SCHEMA, "input_manifest_sha256": provenance["manifest_sha256"],
        "input_row_sha256": canonical_hash(row), "original_split": row["split"],
        "role": "official_train_replay" if row["label_source"] == OFFICIAL_LABEL_SOURCE and row["split"] == "train"
                else "japanese_teacher_adaptation" if row["split"] == "train" else "heldout_source_reference",
        "previously_evaluated_test": historical, "labels_and_mask_unchanged": True,
        "evaluation_interpretation": "teacher_consensus_agreement" if row["label_source"] == TEACHER_LABEL_SOURCE
                                     else "official_sentence_speaker_alignment_agreement",
    }
    return result


def recipe(seed=114):
    return {"status": "proposed_fixed_recipe_not_training_execution", "stage_name": "cmdam-jp2d-adaptation-v1",
            "parent_bundle": "Required completed official CMD-AM stage-one bundle; fresh optimizer, no --resume",
            "label_source": MIXED_LABEL_SOURCE, "frames": 50, "batch_size": 8,
            "lr": 1e-5, "lr_decay": .95, "epochs": 10, "patience": 4, "seed": seed,
            "class_balance": "inverse-frequency", "allow_within_show_group_split": True,
            "split_strategy": "episode_holdout_same_show",
            "validation_selection": "macro_source_ap",
            "test_policy": "Evaluate both sources only after choosing the best validation checkpoint",
            "previously_evaluated_test_source": [TEACHER_LABEL_SOURCE],
            "human_frame_ground_truth": False, "target_domain_accuracy_validated": False,
            "cross_title_generalization_validated": False,
            "limitations": ["Official sentence-speaker/box labels are projected supervision, not human frame-level ASD truth",
                            "Japanese labels are unchanged AVA/TalkSet face-teacher consensus on full-body student crops",
                            "Japanese heldout episodes were evaluated previously; they are not a pristine test set",
                            "No new human verification, labels, pseudo-labels, voice enrollment, or app activation"]}


def assemble(args):
    if args.output.exists():
        raise FileExistsError("Assembly output already exists; select a fresh directory")
    official, official_provenance = read_source(args.official_manifest, OFFICIAL_LABEL_SOURCE, args.frames)
    japanese, japanese_provenance = read_source(args.jp_manifest, TEACHER_LABEL_SOURCE, args.frames)
    audit_isolation(official + japanese)
    jp_train_count = sum(row["split"] == "train" for row in japanese)
    replay_count = args.cmd_train_count if args.cmd_train_count is not None else args.cmd_replay_ratio * jp_train_count
    replay = choose_replay(official, replay_count, args.seed)
    selected_official = replay + [row for row in official if row["split"] != "train"]
    asset_cache = {}
    rows = [freeze_row(row, path, provenance, asset_cache, args.frames)
            for selected, path, provenance in ((selected_official, args.official_manifest, official_provenance),
                                               (japanese, args.jp_manifest, japanese_provenance)) for row in selected]
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["label_source"], row["sample_id"]))
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()
    # Exercise the exact trainer loader against absolute assets before freezing.
    with tempfile.TemporaryDirectory(prefix="lrasd-stage2-check-") as temporary:
        validation_path = Path(temporary) / "labeled.jsonl"
        validation_path.write_bytes(data)
        validated = load_manifest(validation_path, frames=args.frames, allow_within_show_group_split=True)
    supervision = supervision_summary(validated, MIXED_LABEL_SOURCE)
    for split, minimum in (("train", 20), ("val", 5), ("test", 5)):
        for source in (OFFICIAL_LABEL_SOURCE, TEACHER_LABEL_SOURCE):
            stats = supervision["splits"][split].get(source, {})
            if stats.get("windows", 0) < minimum or not stats.get("positive_frames") or not stats.get("negative_frames"):
                raise ValueError(f"{split}/{source}: require >={minimum} windows and both supervised frame classes")
    for path, provenance in ((args.official_manifest, official_provenance), (args.jp_manifest, japanese_provenance)):
        if sha256_file(path) != provenance["manifest_sha256"] or sha256_file(provenance["summary_path"]) != provenance["summary_sha256"]:
            raise ValueError("A completed source changed during assembly; discard this attempt")
        if provenance.get("config_path") and sha256_file(provenance["config_path"]) != provenance["config_sha256"]:
            raise ValueError("Official preparation config changed during assembly")
    audit = {"schema_version": SCHEMA, "training_ready": True, "frames": args.frames,
             "inputs": {"official": official_provenance, "japanese_teacher": japanese_provenance},
             "selection": {"seed": args.seed, "algorithm": "sha256_rank_without_replacement",
                           "cmd_replay_ratio": args.cmd_replay_ratio if args.cmd_train_count is None else None,
                           "cmd_train_replay_windows": len(replay), "jp_train_windows": jp_train_count,
                           "available_cmd_train_windows": sum(row["split"] == "train" for row in official),
                           "cmd_replay_sample_ids": sorted(row["sample_id"] for row in replay),
                           "heldout_policy": "All official and Japanese validation/test rows retained with original splits"},
             "manifest_sha256": hashlib.sha256(data).hexdigest(), "supervision": supervision,
             "split_summary": split_summary(validated), "verified_unique_assets": len(asset_cache),
             "group_clip_and_source_hash_isolation": "passed_across_all_input_rows",
             "labels_and_masks": "unchanged_from_source_manifests",
             "previously_evaluated_test_sources": [TEACHER_LABEL_SOURCE],
             "human_frame_ground_truth": False, "target_domain_accuracy_validated": False,
             "cross_title_generalization_validated": False,
             "assembler_sha256": sha256_file(__file__), "recipe": recipe(args.seed)}
    audit["assembly_hash"] = canonical_hash(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()  # Exclusive directory creation: never overwrite a prior frozen assembly.
    for filename, content in (("labeled.jsonl", data),
                              ("assembly.json", (json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()),
                              ("recipe.json", (json.dumps(recipe(args.seed), indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())):
        with (args.output / filename).open("xb") as stream:
            stream.write(content)
    return audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest", type=Path, default=PROJECT / "training/lrasd/cmdam/prepared-v1/labeled.jsonl")
    parser.add_argument("--jp-manifest", type=Path, default=PROJECT / "training/lrasd/labels-2s-v1/labeled.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for labeled.jsonl, assembly.json and recipe.json")
    parser.add_argument("--frames", type=int, choices=(50,), default=50)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--cmd-replay-ratio", type=int, default=3, help="Distinct CMD training windows per Japanese training window (default: 3)")
    count.add_argument("--cmd-train-count", type=int, help="Explicit distinct CMD training replay count instead of the ratio")
    parser.add_argument("--seed", type=int, default=114)
    args = parser.parse_args(argv)
    if args.cmd_replay_ratio < 1 or (args.cmd_train_count is not None and args.cmd_train_count < 1):
        parser.error("Replay count/ratio must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    audit = assemble(args)
    print(json.dumps({"event": "stage2_manifest_frozen", "manifest_sha256": audit["manifest_sha256"],
                      "assembly_hash": audit["assembly_hash"], "cmd_replay_windows": audit["selection"]["cmd_train_replay_windows"],
                      "jp_train_windows": audit["selection"]["jp_train_windows"], "training_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
