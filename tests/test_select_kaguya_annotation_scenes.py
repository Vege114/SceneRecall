"""The scene proposal helper cannot create human ASD labels from subtitles."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/select_kaguya_annotation_scenes.py"
SPEC = importlib.util.spec_from_file_location("select_kaguya_annotation_scenes", SCRIPT)
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


class SceneProposalTests(unittest.TestCase):
    def test_ass_animation_duplicates_and_cleaning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.ass"
            path.write_text("Dialogue: 0,0:03:01.00,0:03:03.00,Default,,0,0,0,,{\\bord2}hello\\Nworld\n"
                            "Dialogue: 0,0:03:01.00,0:03:03.00,Default,,0,0,0,,{\\bord4}hello\\Nworld\n"
                            "Dialogue: 0,0:03:02.00,0:03:05.00,IN1-JP,,0,0,0,,song\n", encoding="utf-8")
            events = SELECT.parse_ass(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]["text"], "hello world")
            proposals, songs = SELECT.propose_scenes(events, count=1, start=180, end=200,
                                                     minimum_separation=6)
            self.assertEqual(songs, [[182.0, 185.0]])
            self.assertEqual(proposals, [])

    def test_temporal_spread_and_no_labels(self):
        events = [{"start": time, "end": time + 2, "style": "Default", "name": "", "text": "dialogue"}
                  for time in range(200, 1100, 20)]
        proposals, _ = SELECT.propose_scenes(events, count=4, start=180, end=1200,
                                           minimum_separation=140)
        self.assertEqual(len(proposals), 4)
        self.assertTrue(all(right["center_seconds"] - left["center_seconds"] >= 140
                            for left, right in zip(proposals, proposals[1:])))
        for proposal in proposals:
            self.assertFalse(proposal["human_active_speaker_ground_truth"])
            self.assertTrue(proposal["requires_visual_review"])
            self.assertNotIn("labels", proposal)
            self.assertNotIn("character_id", proposal)
            self.assertEqual(proposal["start_ms"] % 40, 0)

    def test_song_fragment_union(self):
        self.assertEqual(SELECT.merge_intervals([(10, 12), (11, 13), (14, 16), (30, 31)]),
                         [[10, 16], [30, 31]])

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            SELECT.propose_scenes([], count=0)

    def test_previous_training_audit_filters_exact_source_and_split(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "previous.jsonl"
            source_sha = "a" * 64
            common = {"source_sha256": source_sha, "fps": 25, "labels": [0] * 50,
                      "source_start_seconds": 100, "sample_id": "one", "split": "train"}
            rows = [common, {**common, "sample_id": "two"},
                    {**common, "source_start_seconds": 200, "split": "val"},
                    {**common, "source_start_seconds": 300, "source_sha256": "b" * 64}]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            ranges = SELECT.prior_training_ranges([path], source_sha)
            self.assertEqual(len(ranges), 1)
            self.assertEqual(ranges[0]["end_seconds"], 102)
            self.assertEqual(len(ranges[0]["training_examples"]), 2)
            candidates = [{"start_ms": 210000, "end_ms": 216000},
                          {"start_ms": 230000, "end_ms": 236000}]
            SELECT.audit_holdout_neighborhoods(candidates, ranges, margin=120)
            self.assertFalse(candidates[0]["holdout_eligible_after_temporal_audit"])
            self.assertTrue(candidates[1]["holdout_eligible_after_temporal_audit"])

    def test_previous_training_audit_rejects_missing_manifest_or_invalid_source(self):
        with self.assertRaises(ValueError):
            SELECT.prior_training_ranges([], "")
        with self.assertRaises(FileNotFoundError):
            SELECT.prior_training_ranges([Path("missing-previous-training.jsonl")], "a" * 64)


if __name__ == "__main__":
    unittest.main()
