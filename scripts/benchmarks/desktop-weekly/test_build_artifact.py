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


build = load_module("build_artifact", "build-artifact.py")


def row(backend, family, speed=10.0, status="ok"):
    return {"backend": backend, "family": family, "task": "ASR", "model_label": f"Model {family}", "weights": "Q8_0", "status": status,
            "observed_backend": backend.upper(), "wall_ms_median": 1100.0, "wall_ms_min": 1000.0, "wall_ms_max": 1200.0,
            "speed_x": speed if status == "ok" else None, "peak_rss_mib": 1024.0, "gpu_memory_rise_mib": 512.0, "gpu_memory_peak_mib": 600.0,
            "quality": {"kind": "wer", "value": 0.0}, "runs": 3, "notes": ""}


def summary(name, started, speeds, os_name="Linux", backends=("cpu", "cuda")):
    results = [row(backend, family, speed) for family, speed in speeds.items() for backend in backends]
    return {"schema": 1, "run": {"name": name, "host": "box", "started": started, "finished": started, "status": "ok"},
            "machine": {"hostname": "box", "os": os_name, "os_version": "Ubuntu", "cpu": "cpu", "memory_gib": 60,
                        "gpu": {"name": "RTX", "driver": "595", "cuda_toolkit": "13.3", "unified_memory": False}, "tools": {"compiler": "gcc"}},
            "pins": {"engine": {"branch": "master", "sha": "abcdef1234567890", "commit_date": "2026-11-01"},
                     "ggml": {"branch": "speech", "sha": "fedcba0987654321", "commit_date": "2026-11-01"},
                     "models": [{"name": "m.gguf", "present": True, "bytes": 1073741824, "sha256": "0123456789abcdef", "source": "registry x"}]},
            "backends": list(backends), "families": list(speeds), "results": results,
            "counts": {"expected": len(results), "ok": len(results), "failed": 0}, "build_failures": [], "problems": []}


class BuildArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "box").mkdir()
        self.write("box", "2026-11-01", summary("2026-11-01", "2026-11-01T17:00:00Z", {"a": 10.0, "b": 100.0}))
        self.write("box", "2026-11-08", summary("2026-11-08", "2026-11-08T17:00:00Z", {"a": 12.0, "b": 90.0}))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, host, name, data):
        (self.root / host / f"{name}.json").write_text(json.dumps(data))

    def test_latest_run_and_history_chips(self):
        summaries = build.load_summaries(self.root)
        hosts = build.runs_by_host(summaries)
        self.assertEqual(build.latest_run(hosts["box"])["run"]["name"], "2026-11-08")
        text = build.render(summaries, 5.0)
        self.assertIn("<title>Speech Fabric Weekly Bench</title>", text)
        self.assertIn('<section id="box">', text)
        self.assertIn('<span class="chip good">+20%</span>', text)
        self.assertIn('<span class="chip warn">-10%</span>', text)
        self.assertIn("Speed on cuda", text)
        self.write("box", "2026-11-15", summary("2026-11-15", "2026-11-15T17:00:00Z", {"a": 12.0}))
        text = build.render(build.load_summaries(self.root), 5.0)
        self.assertIn("Model b", text.split("<h3>History</h3>")[1])
        self.assertIn('id="bench-data"', text)

    def test_results_cells_and_failed_rows(self):
        data = summary("2026-11-15", "2026-11-15T17:00:00Z", {"a": 10.0})
        data["results"][1] = row("cuda", "a", status="timeout")
        text = build.results_table(data)
        self.assertIn('<td class="cell bad">timeout</td>', text)
        self.assertIn('<b class="num">10x</b>', text)
        self.assertIn("1,100 ms · 1.0 GiB", text)

    def test_chart_scales_logarithmically_and_lists_backends(self):
        data = summary("2026-11-15", "2026-11-15T17:00:00Z", {"a": 1.0, "b": 10.0, "c": 100.0})
        scale_max = build.chart_scale_max(data)
        self.assertEqual(scale_max, 100.0)
        self.assertAlmostEqual(build.bar_width(10.0, scale_max), build.CHART_PLOT_WIDTH / 2)
        self.assertAlmostEqual(build.bar_width(100.0, scale_max), build.CHART_PLOT_WIDTH)
        self.assertEqual(build.chart_ticks(scale_max), [1.0, 10.0, 100.0])
        chart = build.speed_chart(data)
        self.assertIn('class="bar cuda"', chart)
        self.assertIn("log scale", chart)

    def test_primary_backend_by_os(self):
        self.assertEqual(build.primary_backend(summary("x", "", {"a": 1.0})), "cuda")
        self.assertEqual(build.primary_backend(summary("x", "", {"a": 1.0}, os_name="Darwin", backends=("cpu", "metal"))), "metal")

    def test_main_writes_page(self):
        out = self.root / "page.html"
        build.main([str(self.root), "-o", str(out)])
        self.assertTrue(out.read_text().startswith("<title>"))


if __name__ == "__main__":
    unittest.main()
