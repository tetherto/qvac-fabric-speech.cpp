#!/usr/bin/env python3
import importlib.util
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


landing = load_module("build_landing", "build-landing.py")


def summary(name, host, started, families, status="ok", problems=()):
    results = [{"backend": "cuda", "family": f, "task": t, "model_label": f, "weights": "Q8_0", "status": "ok", "speed_x": 2.0} for f, t in families.items()]
    return {"schema": 1, "run": {"name": name, "host": host, "started": started, "finished": started, "status": status},
            "machine": {"hostname": host, "os": "Linux", "os_version": "Ubuntu 25.10", "cpu": "Ryzen <9>", "gpu": {"name": "RTX 5090"}},
            "pins": {"engine": {"sha": "abcdef1234567890", "commit_date": "2026-11-01"}, "ggml": {"sha": "fedcba0987654321", "commit_date": "2026-11-01"}},
            "backends": ["cuda"], "families": list(families), "results": results,
            "counts": {"expected": len(results), "ok": len(results), "failed": 0}, "build_failures": [], "problems": list(problems)}


class BuildLandingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        for host in ("qvac-dev-linux-x64", "spark-d9ca"):
            (self.root / "summaries" / host).mkdir(parents=True)
        self.write("qvac-dev-linux-x64", "2026-11-01", summary("2026-11-01", "qvac-dev-linux-x64", "2026-11-01T17:00:00Z", {"parakeet-tdt": "ASR"}))
        self.write("qvac-dev-linux-x64", "2026-11-08", summary("2026-11-08", "qvac-dev-linux-x64", "2026-11-08T17:00:00Z", {"parakeet-tdt": "ASR", "supertonic3": "TTS"},
                                                               status="partial", problems=["cuda: nemotron timed out"]))
        self.write("spark-d9ca", "2026-11-08", summary("2026-11-08", "spark-d9ca", "2026-11-08T17:00:00Z", {"acestep-turbo-q8_0": "Music"}))
        self.summaries = landing.data.load_summaries(self.root / "summaries")
        self.dashboards = {"asr": "https://example.test/asr", "tts": "https://example.test/tts"}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, host, name, data):
        (self.root / "summaries" / host / f"{name}.json").write_text(json.dumps(data))

    def test_task_cards_link_dashboards_and_name_the_machines_with_results(self):
        page = landing.render(self.summaries, self.dashboards, "2026-11-09 08:00 UTC")
        self.assertIn('href="https://example.test/asr"', page)
        self.assertIn("results from RTX 5090", page)
        self.assertIn("results from DGX Spark", page)
        self.assertIn("dashboard link not set", page)

    def test_machine_tiles_show_the_newest_run_and_escape_text(self):
        page = landing.render(self.summaries, self.dashboards, "2026-11-09 08:00 UTC")
        self.assertIn("2026-11-08", page)
        self.assertNotIn("2026-11-01 ·", page)
        self.assertIn('class="tile-status warn">partial', page)
        self.assertIn("Ryzen &lt;9&gt;", page)
        self.assertIn("cuda: nemotron timed out", page)
        self.assertIn("<title>Speech Fabric Weekly Bench</title>", page)

    def test_main_writes_the_page(self):
        out = self.root / "landing.html"
        links = self.root / "dashboards.json"
        links.write_text(json.dumps(self.dashboards))
        landing.main([str(self.root / "summaries"), "-o", str(out), "--dashboards", str(links)])
        self.assertIn("Open the TTS dashboard", out.read_text())


if __name__ == "__main__":
    unittest.main()
