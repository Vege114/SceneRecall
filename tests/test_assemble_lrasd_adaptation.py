"""Frozen replay/split/provenance checks; no ML runtime or source videos needed."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
import zipfile


def import_assembler():
    path = Path(__file__).resolve().parents[1] / "scripts/assemble_lrasd_adaptation.py"
    spec = importlib.util.spec_from_file_location("lrasd_adaptation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def source(tmp_path, assembler, label_source, counts):
    directory = tmp_path / label_source
    directory.mkdir()
    visual, audio = directory / "body.npz", directory / "audio.wav"
    header = repr({"descr": "|u1", "fortran_order": False, "shape": (50, 112, 112)})
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    with zipfile.ZipFile(visual, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("visual.npy", b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + b"\0" * (50 * 112 * 112))
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * 64000)
    teachers = {"ava": {"sha256": "a" * 64}, "talkset": {"sha256": "b" * 64}}
    rows = []
    official = label_source == assembler.OFFICIAL_LABEL_SOURCE
    for split, count in zip(("train", "val", "test"), counts):
        for index in range(count):
            identity = f"{'cmd' if official else 'jp'}-{split}"
            row = {"schema_version": 1, "sample_id": f"{identity}-{index}", "split": split,
                   "movie": identity if official else "jp-show", "clip": identity, "track": str(index),
                   "evaluation_group": identity, "group": identity, "start_frame": index * 50,
                   "end_frame": (index + 1) * 50, "fps": 25, "visual_path": "body.npz", "audio_path": "audio.wav",
                   "audio_start_seconds": 0, "labels": [1] * 25 + [0] * 25,
                   "mask": [False] + [True] * 49, "preprocessing": assembler.PREPROCESSING,
                   "crop_type": "full_body_bbox", "synthetic": False, "machine_generated": True,
                   "human_annotated": False, "label_source": label_source,
                   "asset_sha256": {"visual": assembler.sha256_file(visual), "audio": assembler.sha256_file(audio)}}
            if official:
                row.update(official_annotations=True, annotation_granularity="sentence_speaker_and_character_boxes")
            else:
                row.update(label_accepted=True, weak_label_metadata={"teachers": teachers, "positive_threshold": .95,
                           "negative_threshold": .05, "boundary_radius_frames": 2})
            rows.append(row)
    manifest = directory / "labeled.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {"training_ready": True, "human_ground_truth": False, "accepted_windows": len(rows)}
    if official:
        config = {"frames": 50, "preprocessing": assembler.PREPROCESSING}
        write_json(directory / "preparation-config.json", config)
        summary.update(schema_version="cmdam-derived-asd-summary-v1", processed_clips=40,
                       label_source=label_source, labeled_manifest_sha256=assembler.sha256_file(manifest),
                       config_hash=assembler.canonical_hash(config))
    else:
        summary.update(schema_version="lrasd-machine-label-summary-v1", config={"frames": 50,
                       "positive_threshold": .95, "negative_threshold": .05, "boundary_radius": 2,
                       "min_valid_frames": 20, "teachers": teachers})
    write_json(directory / "label-summary.json", summary)
    return manifest, rows


def inputs(tmp_path, assembler):
    official, cmd_rows = source(tmp_path, assembler, assembler.OFFICIAL_LABEL_SOURCE, (80, 5, 5))
    japanese, jp_rows = source(tmp_path, assembler, assembler.TEACHER_LABEL_SOURCE, (20, 5, 5))
    args = assembler.parse_args(["--official-manifest", str(official), "--jp-manifest", str(japanese),
                                 "--output", str(tmp_path / "mixed")])
    return args, cmd_rows, jp_rows


class AdaptationAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lrasd-assembly-test-")
        self.addCleanup(self.temporary.cleanup)
        self.assembler = import_assembler()
        self.inputs = inputs(Path(self.temporary.name), self.assembler)

    def test_freeze_preserves_labels_splits_and_historical_test(self):
        assembler = self.assembler
        args, cmd_rows, jp_rows = self.inputs
        audit = assembler.assemble(args)
        rows = [json.loads(line) for line in (args.output / "labeled.jsonl").read_text().splitlines()]
        originals = {row["sample_id"]: row for row in cmd_rows + jp_rows}
        self.assertEqual(audit["selection"]["cmd_train_replay_windows"], 60)
        self.assertEqual(len(rows), 60 + 10 + 30)
        self.assertEqual({r["sample_id"] for r in rows if r["split"] != "train"},
                         {r["sample_id"] for r in originals.values() if r["split"] != "train"})
        for row in rows:
            original = originals[row["sample_id"]]
            for field in ("labels", "mask", "sample_id", "split", "group", "movie", "clip", "evaluation_group", "label_source"):
                self.assertEqual(row[field], original[field])
            self.assertTrue(Path(row["audio_path"]).is_absolute())
            self.assertEqual(row["adaptation_provenance"]["input_row_sha256"], assembler.canonical_hash(original))
            self.assertEqual(row["previously_evaluated_test"],
                             row["label_source"] == assembler.TEACHER_LABEL_SOURCE and row["split"] == "test")
        self.assertEqual(audit["manifest_sha256"], assembler.sha256_file(args.output / "labeled.jsonl"))
        self.assertFalse(audit["supervision"]["human_frame_ground_truth"])
        self.assertEqual(audit["recipe"]["lr"], 1e-5)
        self.assertEqual(audit["recipe"]["epochs"], 10)
        self.assertEqual(audit["recipe"]["validation_selection"], "macro_source_ap")
        with self.assertRaises(FileExistsError):
            assembler.assemble(args)

    def test_replay_is_stable_without_replacement_and_excludes_heldout(self):
        assembler = self.assembler
        _, official, _ = self.inputs
        first = assembler.choose_replay(official, 60, 114)
        second = assembler.choose_replay(list(reversed(official)), 60, 114)
        self.assertEqual([r["sample_id"] for r in first], [r["sample_id"] for r in second])
        self.assertEqual(len({r["sample_id"] for r in first}), 60)
        self.assertTrue(all(r["split"] == "train" for r in first))
        with self.assertRaisesRegex(ValueError, "no replacement or silent cap"):
            assembler.choose_replay(official, 81, 114)

    def test_complete_official_hash_required_before_any_output(self):
        args, _, _ = self.inputs
        path = args.official_manifest.parent / "label-summary.json"
        summary = json.loads(path.read_text())
        summary["labeled_manifest_sha256"] = "0" * 64
        write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "40-clip manifest SHA"):
            self.assembler.assemble(args)
        self.assertFalse(args.output.exists())

    def test_changed_asset_refused_before_any_output(self):
        args, _, _ = self.inputs
        (args.jp_manifest.parent / "body.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "asset SHA-256 changed"):
            self.assembler.assemble(args)
        self.assertFalse(args.output.exists())

    def test_source_hash_leak_rejected_even_in_unselected_cmd_training(self):
        _, official, japanese = self.inputs
        official[0]["source_video_sha256"] = "c" * 64
        japanese[-1]["source_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "leaks across splits"):
            self.assembler.audit_isolation(official + japanese)

    def test_unaccepted_teacher_row_refused(self):
        args, _, _ = self.inputs
        rows = [json.loads(line) for line in args.jp_manifest.read_text().splitlines()]
        rows[0]["label_accepted"] = False
        args.jp_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "must be accepted"):
            self.assembler.assemble(args)
        self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
