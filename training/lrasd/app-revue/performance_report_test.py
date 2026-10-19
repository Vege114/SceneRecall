"""Completion-scope regressions with synthetic metadata; never opens a library."""
from copy import deepcopy
from pathlib import Path
import unittest

from performance_collect import target_complete
from performance_report import model


def evidence(first_status):
    episodes = []
    for number, status in ((1, first_status), (2, "cancelled"), (3, "cancelled")):
        job = {"id": f"job_{number}", "asset_id": f"asset_{number}", "run_id": f"run_{number}",
               "status": status, "created_at": "2026-10-18T08:00:00+00:00",
               "updated_at": "2026-10-18T12:00:00+00:00", "request_count": number * 10}
        run = {"job": job, "stage_timings": {}, "receipts": {},
               "lifecycle_envelope": {"start_utc": job["created_at"], "end_utc": job["updated_at"], "seconds": 14400}}
        episodes.append({"episode": number, "asset_id": f"asset_{number}", "analysis_runs": [run], "scan_jobs": []})
    return {"schema_version": "scenerecall-performance-v1", "capture_finished_utc": "2026-10-18T12:10:00+00:00",
            "completion_scope": {"target_episodes": [1], "excluded_episodes": [2, 3]}, "episodes": episodes}


class CompletionScopeTests(unittest.TestCase):
    def report(self, value, annotations=None):
        return model(value, annotations or {}, Path("synthetic.json"), Path("synthetic-annotations.json"))

    def test_episode_one_completed_and_other_two_cancelled_is_final(self):
        data = evidence("completed")
        result = self.report(data)
        self.assertTrue(result["all_complete"])
        self.assertTrue(target_complete(data["episodes"], {1}))
        self.assertEqual((result["complete_count"], result["target_count"]), (1, 1))
        self.assertEqual(result["requests"], 10)
        self.assertEqual(result["excluded_requests"], 50)
        self.assertEqual([ep["episode"] for ep in result["episodes"] if ep["in_scope"]], [1])
        self.assertTrue(all("范围外历史" in ep["scope_label"] for ep in result["episodes"][1:]))

    def test_episode_one_running_stays_partial(self):
        data = evidence("running")
        result = self.report(data)
        self.assertFalse(result["all_complete"])
        self.assertFalse(target_complete(data["episodes"], {1}))
        self.assertEqual(result["complete_count"], 0)

    def test_completed_index_is_not_labelled_as_still_accumulating(self):
        data = evidence("completed")
        data["episodes"][0]["analysis_runs"][0]["stage_timings"]["index"] = {
            "accumulated_elapsed": {"value": 123}}
        result = self.report(data)
        index = next(row for row in result["episodes"][0]["rows"] if row["key"] == "index")
        self.assertEqual(index["state_text"], "已完成任务的累计计时")

    def test_latest_user_scope_overrides_old_three_episode_metadata(self):
        data = evidence("completed")
        data["completion_scope"] = {"target_episodes": [1, 2, 3], "excluded_episodes": []}
        annotations = {"completion_scope": {"target_episodes": [1], "excluded_episodes": [2, 3]}}
        self.assertTrue(self.report(data, annotations)["all_complete"])

    def test_missing_target_run_cannot_be_reported_complete(self):
        data = evidence("completed")
        data["episodes"][0]["analysis_runs"] = []
        self.assertFalse(self.report(data)["all_complete"])
        self.assertFalse(target_complete(data["episodes"], {1}))

    def test_failed_historical_run_does_not_override_completed_latest_target(self):
        data = evidence("completed")
        old = deepcopy(data["episodes"][0]["analysis_runs"][0])
        old["job"].update(status="failed", created_at="2026-10-17T08:00:00+00:00")
        data["episodes"][0]["analysis_runs"].insert(0, old)
        self.assertTrue(self.report(data)["all_complete"])
        self.assertTrue(target_complete(data["episodes"], {1}))


if __name__ == "__main__":
    unittest.main()
