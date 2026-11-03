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


build = load_module("build_dashboard_data", "build-dashboard-data.py")


def row(backend, family, speed=10.0, status="ok", task="ASR"):
    return {"backend": backend, "family": family, "task": task, "model_label": f"Model {family}", "weights": "Q8_0", "status": status,
            "observed_backend": backend.upper(), "wall_ms_median": 1100.0, "wall_ms_min": 1000.0, "wall_ms_max": 1200.0,
            "speed_x": speed if status == "ok" else None, "audio_seconds": 11.0, "peak_rss_mib": 1024.0, "gpu_memory_rise_mib": 512.0,
            "gpu_memory_peak_mib": 600.0, "quality": {"kind": "wer", "value": 0.0125}, "runs": 3, "notes": ""}


def summary(name, started, speeds, backends=("cuda",), host="qvac-dev-linux-x64", tasks=None):
    tasks = tasks or {}
    results = [row(backend, family, speed, task=tasks.get(family, "ASR")) for family, speed in speeds.items() for backend in backends]
    return {"schema": 1, "run": {"name": name, "host": host, "started": started, "finished": started, "status": "ok"},
            "machine": {"hostname": host, "os": "Linux", "cpu": "cpu", "memory_gib": 60, "gpu": {"name": "RTX"}},
            "pins": {"engine": {"sha": "abcdef1234567890", "commit_date": "2026-11-01"}, "ggml": {"sha": "fedcba0987654321", "commit_date": "2026-11-01"}},
            "backends": list(backends), "families": list(speeds), "results": results,
            "counts": {"expected": len(results), "ok": len(results), "failed": 0}, "build_failures": [], "problems": ["one problem"]}


COMPARISONS = [
    {"task": "ASR", "engine": "other.cpp", "model": "Parakeet TDT 0.6B v3", "quantization": "q8_0", "device": "RTX 5090 box", "backend": "cuda",
     "qvac_rtf": "0.0007 / 0.0006", "ratio": "2.09× / 3.18×", "measured": "2026-09-22", "source": "campaign", "harness": "h"},
    {"task": "Music", "engine": "acestep.cpp", "model": "ACE-Step 1.5 Turbo", "quantization": "Q8_0", "device": "RTX 5090 box", "backend": "cuda",
     "qvac_rtf": "0.139 single set", "ratio": "1.91× single set", "measured": "2026-09-14", "source": "campaign", "harness": "h"},
]


class BuildDashboardDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "summaries" / "qvac-dev-linux-x64").mkdir(parents=True)
        self.write("2026-11-01-heavy", summary("2026-11-01-heavy", "2026-11-01T17:00:00Z", {"parakeet-tdt": 10.0, "supertonic3": 100.0, "acestep-turbo-q8_0": 0.5},
                                              tasks={"supertonic3": "TTS", "acestep-turbo-q8_0": "Music"}))
        self.write("2026-11-08", summary("2026-11-08", "2026-11-08T17:00:00Z", {"parakeet-tdt": 12.0, "transcribepsy": 40.0}))
        self.summaries = build.load_summaries(self.root / "summaries")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        (self.root / "summaries" / "qvac-dev-linux-x64" / f"{name}.json").write_text(json.dumps(data))

    def test_weekly_rows_carry_labels_order_and_units(self):
        rows = build.weekly_rows(self.summaries, "asr")
        self.assertEqual(len(rows), 3)
        first = next(r for r in rows if r["run"] == "2026-11-08" and r["family"] == "parakeet-tdt")
        self.assertEqual(first["label"], "Parakeet TDT 0.6B v3 Q8_0")
        self.assertEqual(first["order"], 0)
        self.assertEqual(first["machine"], "RTX 5090")
        self.assertEqual(first["lane"], "RTX 5090 cuda")
        self.assertEqual(first["rss_gib"], 1.0)
        self.assertEqual(first["gpu_mem_rise_gib"], 0.5)
        self.assertEqual(first["quality"], "WER")
        self.assertEqual(first["quality_pct"], 1.25)
        self.assertAlmostEqual(first["rtf"], 1 / 12.0, places=4)
        self.assertEqual(first["engine"], "abcdef1234")

    def test_latest_keeps_newest_cell_and_flags_older_runs(self):
        latest = build.latest_cells(build.weekly_rows(self.summaries, "asr"), self.summaries)
        by_family = {r["family"]: r for r in latest}
        self.assertEqual(set(by_family), {"parakeet-tdt", "transcribepsy"})
        self.assertEqual(by_family["parakeet-tdt"]["run"], "2026-11-08")
        self.assertTrue(by_family["parakeet-tdt"]["current"])
        tts = build.latest_cells(build.weekly_rows(self.summaries, "tts"), self.summaries)
        self.assertEqual(tts[0]["run"], "2026-11-01-heavy")
        self.assertFalse(tts[0]["current"])

    def test_campaign_rows_turn_rtf_and_ratio_into_speeds(self):
        rows = build.campaign_rows(COMPARISONS, "asr")
        self.assertEqual(len(rows), 1)
        asr = rows[0]
        self.assertEqual(asr["label"], "Parakeet TDT 0.6B v3 q8_0")
        self.assertEqual((asr["qvac_rtf_short"], asr["qvac_rtf_long"]), (0.0007, 0.0006))
        self.assertEqual((asr["ratio_short"], asr["ratio_long"]), (2.09, 3.18))
        self.assertAlmostEqual(asr["qvac_speed_long"], 1 / 0.0006, places=2)
        self.assertAlmostEqual(asr["competitor_rtf_long"], 0.0006 * 3.18, places=6)
        self.assertAlmostEqual(asr["competitor_speed_short"], 1 / (0.0007 * 2.09), places=2)
        music = build.campaign_rows(COMPARISONS, "music")[0]
        self.assertEqual((music["qvac_rtf_short"], music["qvac_rtf_long"]), (0.139, None))
        self.assertIsNone(music["competitor_speed_long"])
        self.assertAlmostEqual(music["competitor_speed_short"], 1 / (0.139 * 1.91), places=2)

    def test_build_writes_one_folder_per_task_and_the_runs_file(self):
        out = self.root / "out"
        written = build.build(self.summaries, COMPARISONS, out)
        self.assertEqual(written["asr"], {"weekly": 3, "latest": 2, "campaign": 1})
        self.assertEqual(written["music"], {"weekly": 1, "latest": 1, "campaign": 1})
        self.assertEqual(written["tts"]["campaign"], 0)
        runs = json.loads((out / "runs.json").read_text())
        self.assertEqual([r["run"] for r in runs], ["2026-11-01-heavy", "2026-11-08"])
        self.assertEqual(runs[0]["problems"], "one problem")
        self.assertEqual(runs[0]["backends"], "cuda")
        self.assertTrue((out / "tts" / "latest.json").is_file())

    def test_main_reports_what_it_wrote(self):
        out = self.root / "out"
        build.main([str(self.root / "summaries"), "-o", str(out), "--comparisons", str(self.root / "missing.json")])
        self.assertEqual(json.loads((out / "asr" / "campaign.json").read_text()), [])


if __name__ == "__main__":
    unittest.main()
