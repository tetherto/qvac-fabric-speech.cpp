#!/usr/bin/env python3
import contextlib
import datetime
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_module("check_delivery", "check-delivery.py")
HOSTS = "box-a,box-b,box-c"
SUNDAY = "2026-11-15"


class CheckDeliveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = pathlib.Path(self.tmp.name) / "results"
        self.deliver("summaries", "box-a", f"{SUNDAY}.json")
        self.deliver("failures", "box-b", f"{SUNDAY}.txt")
        self.deliver("summaries", "box-c", "2026-11-08.json")

    def tearDown(self):
        self.tmp.cleanup()

    def deliver(self, folder, host, name):
        path = self.results / folder / host / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    def run_check(self, event, today="2026-11-16", extra=()):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            check.main(["--results", str(self.results), "--event", event, "--today", today, "--hosts", HOSTS, *extra])
        return dict(line.split("=", 1) for line in out.getvalue().splitlines())

    def test_the_week_is_the_latest_sunday_on_or_before_today(self):
        cases = {"2026-11-15": SUNDAY, "2026-11-16": SUNDAY, "2026-11-21": SUNDAY, "2026-11-22": "2026-11-22"}
        for today, sunday in cases.items():
            self.assertEqual(check.week_sunday(datetime.date.fromisoformat(today)).isoformat(), sunday)

    def test_a_push_waits_until_every_machine_delivered_a_summary_or_a_failure_notice(self):
        self.assertEqual(self.run_check("push"), {"publish": "false", "week": SUNDAY, "missing": "box-c"})
        self.deliver("failures", "box-c", f"{SUNDAY}.txt")
        self.assertEqual(self.run_check("push"), {"publish": "true", "week": SUNDAY, "missing": ""})

    def test_the_schedule_and_a_manual_run_publish_what_has_arrived(self):
        self.assertEqual(self.run_check("schedule")["publish"], "true")
        self.assertEqual(self.run_check("workflow_dispatch"), {"publish": "true", "week": SUNDAY, "missing": "box-c"})

    def test_a_run_from_last_week_does_not_count_for_this_week(self):
        self.assertEqual(self.run_check("push", today="2026-11-09"), {"publish": "false", "week": "2026-11-08", "missing": "box-a,box-b"})

    def test_the_summary_names_every_machine_and_the_decision(self):
        summary_file = pathlib.Path(self.tmp.name) / "summary.md"
        self.run_check("push", extra=["--summary", str(summary_file)])
        text = summary_file.read_text()
        self.assertIn("## Speech bench, week of 2026-11-15", text)
        self.assertIn("| box-a | run summary |", text)
        self.assertIn("| box-b | failure notice |", text)
        self.assertIn("| box-c | nothing yet |", text)
        self.assertIn("Not published: waiting for box-c.", text)

    def test_the_expected_machines_default_to_the_dashboard_list(self):
        dashboards = pathlib.Path(self.tmp.name) / "dashboards.json"
        dashboards.write_text(json.dumps({"landing": "https://hub", "box-a": "https://a", "box-b": "https://b"}))
        self.assertEqual(check.expected_hosts("", dashboards), ["box-a", "box-b"])
        self.assertEqual(check.expected_hosts("x, y,", dashboards), ["x", "y"])
        self.assertEqual(len(check.dashboard_hosts(check.DASHBOARDS_FILE)), 6)


if __name__ == "__main__":
    unittest.main()
