"""Human supervision, exact source alignment, and path confinement of the standalone exporter."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np


def load_exporter():
    script = Path(__file__).resolve().parents[1] / "scripts/export_lrasd_annotations.py"
    spec = importlib.util.spec_from_file_location("human_asd_export_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HumanAsdExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="human-asd-export-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.root = self.work / "clips"
        self.directory = self.root / "asd_clip1"
        self.directory.mkdir(parents=True)
        self.output = self.work / "export"
        self.exporter = load_exporter()
        self.clip = self.make_clip(100)
        self.save()
        # Structural/alignment tests do not need the optional OpenCV runtime.
        self.crop = patch.object(self.exporter, "crop_visual", side_effect=lambda frames, boxes:
                                 np.asarray([np.full((112, 112), 0 if box is None else 77, dtype=np.uint8)
                                             for box in boxes]))
        self.crop_mock = self.crop.start()
        self.addCleanup(self.crop.stop)

    def make_clip(self, frames):
        rgb = self.directory / "frames.rgb"
        with rgb.open("wb") as stream:
            stream.truncate(frames * 480 * 270 * 3)
        samples = np.arange(frames * 640, dtype=np.int16)
        with wave.open(str(self.directory / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(samples.astype("<i2").tobytes())
        files = {key: {"relative_path": name, "sha256": self.exporter.sha256_file(self.directory / name),
                       "bytes": (self.directory / name).stat().st_size}
                 for key, name in (("frames_rgb", "frames.rgb"), ("audio_wav", "audio.wav"))}
        return {"schema_version": "1.0", "annotation_version": self.exporter.ANNOTATION_VERSION,
                "asset_id": "asset_film1", "id": "asd_clip1", "revision": 1, "status": "ready",
                "movie": "Synthetic work", "asset_title": "Synthetic work", "source_fingerprint": "a" * 64,
                "source_stat": {"size": 1000000, "mtime_ns": 1234}, "audio_stream_index": 1,
                "start_ms": 4000, "end_ms": 4000 + frames * 40, "frame_count": frames, "fps": 25,
                "server": {"files": files, "source_sha256": "a" * 64,
                           "source_sha256_basis": "library_full_file_sha256", "preprocessing": self.exporter.GRID_VERSION,
                           "preparation": {"kind": "local_decode_detector_proposals", "model_scores_copied": False}},
                "tracks": [{"id": "track_1", "character_id": "char_1", "boxes": [[0.1, 0.1, 0.4, 0.8]] * frames,
                            "labels": [1] * (frames // 2) + [0] * (frames - frames // 2), "reviewed": True,
                            "excluded": False, "note": "synthetic test fixture only", "reviewer": {"kind": "human"},
                            "reviewed_at": "2026-10-17T00:00:00Z", "provenance": {"box_source": "detector_proposal",
                                "boxes_reviewed": True, "labels_source": "human_input", "model_scores_copied": False}}]}

    def save(self):
        (self.directory / "clip.json").write_text(json.dumps(self.clip), encoding="utf-8")

    def export(self, output=None):
        self.save()
        return self.exporter.export_annotations(self.root, output or self.output)

    @staticmethod
    def rows(summary):
        return [json.loads(line) for line in Path(summary["manifest_path"]).read_text(encoding="utf-8").splitlines()]

    def test_exact_two_second_windows_keep_human_provenance_and_pcm_alignment(self):
        self.clip["tracks"][0]["labels"][1] = -1
        self.clip["tracks"][0]["boxes"][1] = None
        summary = self.export()
        rows = self.rows(summary)
        self.assertEqual(summary["accepted_windows"], 2)
        self.assertFalse(summary["training_ready"])
        self.assertEqual(summary["split_counts"], {"train": 2, "val": 0, "test": 0})
        self.assertEqual([(row["source_start_ms"], row["source_end_ms"]) for row in rows], [(4000, 6000), (6000, 8000)])
        self.assertEqual({row["evaluation_group"] for row in rows}, {summary["evaluation_group"]})
        for index, row in enumerate(rows):
            self.assertEqual(row["label_source"], "human_frame_asd")
            self.assertIs(row["human_annotated"], True)
            self.assertIs(row["human_frame_asd_ground_truth"], True)
            self.assertIs(row["machine_generated"], False)
            self.assertEqual(row["human_annotation"]["revision"], 1)
            self.assertEqual(row["human_annotation"]["track_provenance"]["box_source"], "detector_proposal")
            self.assertEqual(row["source_video_sha256"], "a" * 64)
            self.assertEqual(row["human_annotation"]["annotation_sha256"], self.exporter.sha256_file(self.directory / "clip.json"))
            self.assertEqual(row["labels"], self.clip["tracks"][0]["labels"][index * 50:(index + 1) * 50])
            with wave.open(str(self.output / row["audio_path"]), "rb") as stream:
                self.assertEqual((stream.getnframes(), stream.getframerate()), (32000, 16000))
                pcm = np.frombuffer(stream.readframes(32000), dtype="<i2")
            np.testing.assert_array_equal(pcm, np.arange(64000, dtype=np.int16)[index * 32000:(index + 1) * 32000])
            with np.load(self.output / row["visual_path"], allow_pickle=False) as visual:
                self.assertEqual(visual["visual"].shape, (50, 112, 112))
                self.assertEqual(visual["visual"].dtype, np.uint8)
                if index == 0:
                    self.assertFalse(visual["boxes_valid_mask"][1])
                    self.assertEqual(int(visual["visual"][1].max()), 0)
                    self.assertFalse(row["mask"][1])

    def test_unreviewed_excluded_unknown_and_bad_boxes_are_reported_without_fabrication(self):
        track = self.clip["tracks"][0]
        bad = copy.deepcopy(track)
        bad.update(id="track_bad", boxes=[[0.9, 0.1, 0.4, 0.8]] * 100)
        unknown = copy.deepcopy(track)
        unknown.update(id="track_unknown", labels=[-1] * 100)
        unreviewed = copy.deepcopy(track)
        unreviewed.update(id="track_draft", reviewed=False)
        excluded = copy.deepcopy(track)
        excluded.update(id="track_excluded", excluded=True)
        self.clip["tracks"] = [track, bad, unknown, unreviewed, excluded]
        summary = self.export()
        self.assertEqual(summary["accepted_windows"], 2)
        self.assertEqual({row["track"] for row in self.rows(summary)}, {"track_1"})
        self.assertEqual(summary["rejected_tracks"][0]["track"], "track_bad")
        self.assertEqual({row["reason"] for row in summary["skipped"]}, {"all_unknown", "not_reviewed", "excluded"})
        self.assertIn("not_reviewed", {row["reason"] for row in summary["blocked"]})
        self.assertNotIn("all_unknown", {row["reason"] for row in summary["blocked"]})

    def test_all_unknown_track_still_requires_valid_review_and_clip_supervision(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import prepare_lrasd_human_round as preparation

        self.clip["tracks"][0]["labels"] = [-1] * 100
        plan = {"source_fingerprint": "a" * 64, "plan_sha256": "b" * 64, "items": [
            {"id": "item-1", "clip_id": "asd_clip1", "scene_id": "scene-1", "split": "train",
             "start_ms": 4000, "end_ms": 8000}]}
        self.save()
        original = (self.directory / "clip.json").read_bytes()
        readiness = preparation.check_readiness(self.root, plan)
        self.assertFalse(readiness["ready"])
        self.assertEqual([row["reason"] for row in readiness["blocked"]], ["no_usable_reviewed_track"])
        summary = self.exporter.export_annotations(self.root, self.output)
        self.assertEqual(summary["accepted_windows"], 0)
        self.assertEqual([row["reason"] for row in summary["skipped"]], ["all_unknown"])
        self.assertEqual([row["reason"] for row in summary["blocked"]], ["no_usable_reviewed_track"])
        self.assertEqual((self.directory / "clip.json").read_bytes(), original)

        # Unknown labels must not short-circuit provenance or bounding-box
        # validation, and a draft track must still block an otherwise usable clip.
        unknown = copy.deepcopy(self.clip["tracks"][0])
        supervised = copy.deepcopy(unknown)
        supervised.update(id="track_supervised", labels=[1, 0] * 50)
        for change, reason in (({"reviewed": False}, "not_reviewed"),
                               ({"provenance": {"labels_source": "model_guess"}}, "human-input provenance"),
                               ({"boxes": [[0.9, 0.1, 0.4, 0.8]] * 100}, "outside the canvas")):
            with self.subTest(reason=reason):
                self.clip["tracks"] = [{**copy.deepcopy(unknown), **change}, supervised]
                self.save()
                readiness = preparation.check_readiness(self.root, plan)
                self.assertFalse(readiness["ready"])
                self.assertTrue(any(reason in row["reason"] for row in readiness["blocked"]))
                output = self.work / ("invalid-" + str(len(list(self.work.glob("invalid-*")))))
                summary = self.exporter.export_annotations(self.root, output)
                self.assertTrue(any(reason in row["reason"] for row in summary["blocked"]))

    def test_missing_box_for_known_label_and_unsaved_review_are_rejected(self):
        self.clip["tracks"][0]["boxes"][0] = None
        summary = self.export()
        self.assertFalse(summary["export_ready"])
        self.assertIn("known label requires", summary["rejected_tracks"][0]["reason"])
        self.clip["tracks"][0]["boxes"][0] = [0.1, 0.1, 0.4, 0.8]
        self.clip["revision"] = 0
        summary = self.export(self.work / "second")
        self.assertFalse(summary["export_ready"])
        self.assertIn("saved revision", summary["rejected_tracks"][0]["reason"])

    def test_tail_is_discarded_without_padding_or_random_validation_split(self):
        self.clip = self.make_clip(76)
        summary = self.export()
        self.assertEqual(summary["accepted_windows"], 1)
        self.assertEqual(summary["discarded_tails"], [{"clip": "asd_clip1", "frames": 26, "start_frame": 50,
                                                      "reason": "incomplete_2s_window_discarded"}])
        self.assertEqual(self.rows(summary)[0]["split"], "train")

    def test_missing_root_and_bad_json_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.exporter.export_annotations(self.work / "missing", self.output)
        (self.directory / "clip.json").write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid clip.json"):
            self.exporter.export_annotations(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_payload_media_paths_are_never_followed_and_hashes_are_checked(self):
        self.clip["server"]["files"]["frames_rgb"]["relative_path"] = "../../unrelated-file"
        with self.assertRaisesRegex(ValueError, "fixed media"):
            self.export()
        self.clip["server"]["files"]["frames_rgb"]["relative_path"] = "frames.rgb"
        self.clip["server"]["files"]["audio_wav"]["sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_mixed_source_or_clip_id_path_is_rejected(self):
        self.clip["id"] = "../outside"
        with self.assertRaisesRegex(ValueError, "schema/status/ID"):
            self.export()
        self.clip["id"] = "asd_clip1"
        self.save()
        second = self.root / "asd_clip2"
        shutil.copytree(self.directory, second)
        alternate = copy.deepcopy(self.clip)
        alternate.update(id="asd_clip2", source_fingerprint="b" * 64)
        alternate["server"]["source_sha256"] = "b" * 64
        (second / "clip.json").write_text(json.dumps(alternate), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "one unchanged original asset"):
            self.export()

    def test_media_short_audio_and_invalid_frame_grid_fail_before_output(self):
        self.clip["end_ms"] += 40
        with self.assertRaisesRegex(ValueError, "frame grid"):
            self.export()
        self.clip["end_ms"] -= 40
        with wave.open(str(self.directory / "audio.wav"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0" * (100 * 640 * 2 - 2))
        declared = self.clip["server"]["files"]["audio_wav"]
        declared.update(sha256=self.exporter.sha256_file(self.directory / "audio.wav"),
                        bytes=(self.directory / "audio.wav").stat().st_size)
        with self.assertRaisesRegex(ValueError, "exact aligned"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_reproducible_export_hashes_and_no_overwrite(self):
        first = self.export()
        second = self.export(self.work / "second")
        self.assertEqual(first["labeled_manifest_sha256"], second["labeled_manifest_sha256"])
        self.assertEqual(self.rows(first), self.rows(second))
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.export()

    def test_annotation_change_during_export_prevents_published_manifest(self):
        def edit_during_crop(frames, boxes):
            self.clip["revision"] += 1
            self.save()
            return np.zeros((50, 112, 112), dtype=np.uint8)
        self.crop_mock.side_effect = edit_during_crop
        with self.assertRaisesRegex(ValueError, "changed during export"):
            self.export()
        self.assertFalse((self.output / "labeled.jsonl").exists())

    def test_real_crop_matches_rgb_gray112_and_keeps_zero_unknown(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is in the separate visual runtime")
        self.crop.stop()
        # A red left half and blue right half detect RGB/BGR or bbox mistakes.
        frames = np.zeros((2, 270, 480, 3), dtype=np.uint8)
        frames[:, :, :240, 0] = 255
        frames[:, :, 240:, 2] = 255
        visual = self.exporter.crop_visual(frames, [[0, 0, 0.5, 1], None])
        expected = cv2.resize(cv2.cvtColor(frames[0, :, :240], cv2.COLOR_RGB2GRAY), (112, 112),
                              interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(visual[0], expected)
        self.assertEqual(int(visual[0].mean()), 76)
        self.assertEqual(int(visual[1].max()), 0)

    def test_explicit_plan_controls_scene_split_and_preserves_unknown(self):
        plan = {"source_fingerprint": "a" * 64, "minimum_cross_split_gap_ms": 90000,
                "items": [{"id": "item-1", "clip_id": "asd_clip1", "scene_id": "scene-1",
                           "split": "val", "start_ms": 4000, "end_ms": 8000}]}
        plan_path = self.work / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        self.clip["tracks"][0]["labels"][0] = -1
        self.save()
        summary = self.exporter.export_annotations(self.root, self.output, plan_path=plan_path)
        rows = self.rows(summary)
        self.assertEqual(summary["split_counts"], {"train": 0, "val": 2, "test": 0})
        self.assertFalse(summary["training_ready"])
        self.assertEqual(rows[0]["labels"][0], -1)
        self.assertEqual(rows[0]["annotation_plan"]["split"], "val")
        self.assertEqual(rows[0]["annotation_plan"]["scene_id"], "scene-1")
        self.assertTrue(rows[0]["evaluation_group"].startswith("human-scene-"))

    def test_plan_cannot_rebind_source_or_range_and_review_requires_provenance(self):
        plan = {"source_fingerprint": "b" * 64, "items": [
            {"id": "one", "clip_id": "asd_clip1", "scene_id": "scene", "split": "train", "start_ms": 4000, "end_ms": 8000}]}
        path = self.work / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        self.save()
        with self.assertRaisesRegex(ValueError, "differs from its fixed"):
            self.exporter.export_annotations(self.root, self.output, plan_path=path)
        self.clip["tracks"][0]["provenance"]["labels_source"] = "model_guess"
        summary = self.export()
        self.assertFalse(summary["export_ready"])
        self.assertIn("human-input provenance", summary["rejected_tracks"][0]["reason"])

    def test_complete_human_round_exports_frozen_splits_and_commands_without_training(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import prepare_lrasd_human_round as preparation
        from scenerecall.lrasd_training_data import load_manifest, supervision_summary

        self.clip = self.make_clip(200)
        plan = {"source_fingerprint": "a" * 64, "minimum_cross_split_gap_ms": 90000, "items": []}
        for index, (split, start, track_count) in enumerate((("train", 4000, 5), ("val", 200000, 2), ("test", 400000, 2)), 1):
            identity = f"asd_clip{index}"
            directory = self.root / identity
            directory.mkdir(exist_ok=True)
            if index > 1:
                # Sparse zero RGB fixtures avoid allocating movie-sized test buffers.
                with (directory / "frames.rgb").open("wb") as stream:
                    stream.truncate(200 * 480 * 270 * 3)
                shutil.copyfile(self.directory / "audio.wav", directory / "audio.wav")
            clip = copy.deepcopy(self.clip)
            clip.update(id=identity, start_ms=start, end_ms=start + 8000)
            clip["tracks"] = [{**copy.deepcopy(clip["tracks"][0]), "id": f"track_{track}"} for track in range(track_count)]
            clip["tracks"].append({**copy.deepcopy(clip["tracks"][0]), "id": "track_unknown", "labels": [-1] * 200})
            (directory / "clip.json").write_text(json.dumps(clip), encoding="utf-8")
            plan["items"].append({"id": f"item-{index}", "clip_id": identity, "scene_id": f"scene-{index}",
                                  "split": split, "start_ms": start, "end_ms": start + 8000})
        path = self.work / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        original = (self.directory / "clip.json").read_bytes()
        # prepare_round imports its own exporter module; inject the tested crop adapter.
        with patch.object(preparation, "export_annotations", side_effect=lambda root, output, **kw:
                          self.exporter.export_annotations(root, output, **kw)):
            report = preparation.prepare_round(self.root, path, self.output, self.work / "baseline", export=True)
        self.assertTrue(report["ready"])
        self.assertTrue(report["training_ready"])
        self.assertEqual(report["blocked"], [])
        self.assertFalse(report["starts_training"])
        self.assertEqual(report["split_counts"], {"train": 20, "val": 8, "test": 8})
        summary = json.loads((self.output / "dataset/label-summary.json").read_text(encoding="utf-8"))
        self.assertEqual({(row["clip"], row["track"], row["reason"]) for row in summary["skipped"]},
                         {(f"asd_clip{index}", "track_unknown", "all_unknown") for index in range(1, 4)})
        self.assertEqual(set(report["commands"]), {"baseline_validation", "finetune", "final_comparison"})
        self.assertIn("--defer-final-test", report["commands"]["finetune"])
        rows = load_manifest(self.output / "dataset/labeled.jsonl", frames=50, allow_within_show_group_split=True)
        self.assertTrue(supervision_summary(rows, "human_frame_asd")["human_frame_ground_truth"])
        self.assertEqual((self.directory / "clip.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
