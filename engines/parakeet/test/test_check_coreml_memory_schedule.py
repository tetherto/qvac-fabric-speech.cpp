"""Unit tests for scripts/check-coreml-memory-schedule.py: what each check
compares, that a template stage with no generate or load segment fails, and
that a stage seen more than once in a phase is checked at its largest value."""
import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-coreml-memory-schedule.py"
spec = importlib.util.spec_from_file_location("check_coreml_memory_schedule", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

MIB = 1 << 20
STAGES = ("diarizer-358", "diarizer-745")

# I/O: 1 x 4 x 256 fp32 in (4096 bytes), 1 x 512 fp16 out (1024 bytes).
IO_BYTES = 4096 + 1024


def template(stage, **memory):
    m = {"runtime_weight_bytes": 10 * MIB, "workspace_bytes": 2 * MIB, "compile_peak_bytes": 20 * MIB,
         "ane_compile_reserve_bytes": 300 * MIB,
         "load_peak_bytes": {"cpu_and_ane": 30 * MIB, "all": None, "cpu_and_gpu": None, "cpu_only": None},
         "margin": 1.0}
    m.update(memory)
    return {"stage": stage, "weight_bin_size": 100 * MIB, "generation_peak_bytes": 2 * MIB, "state_bytes": 0,
            "functions": [{"name": "main", "inputs": [{"name": "x", "dtype": "fp32", "shape": [1, 4, 256]}],
                           "outputs": [{"name": "y", "dtype": "fp16", "shape": [1, 512]}]}],
            "memory": m}


# Charged per stage by template(): prepare 2 + 8 + 20 = 30 MiB, load 30 MiB,
# reserve 300 MiB, resident 10 MiB + 2 x I/O + 2 MiB.
RESIDENT = 12 * MIB + 2 * IO_BYTES


def segment(stage, phase, t0, before, peak, **extra):
    s = {"stage": stage, "phase": phase, "t0": t0, "t1": t0 + 1, "before": before, "peak": peak,
         "increment": peak - before, "ane_compiler_peak": 0}
    s.update(extra)
    return s


def run_report():
    segs, t = [], 0.0
    for stage in STAGES:
        segs.append(segment(stage, "generate", t, 100 * MIB, 110 * MIB))
        segs.append(segment(stage, "compile", t + 1, 110 * MIB, 120 * MIB))
        t += 2
    for i, stage in enumerate(STAGES):
        segs.append(segment(stage, "load", t, 100 * MIB, 120 * MIB, ane_compiler_peak=(450 + 50 * i) * MIB,
                            settled_after_ready=(105 + 5 * i) * MIB, resident_increment=5 * MIB))
        t += 1
    return {"mode": "coreml", "baseline_bytes": 100 * MIB, "fit_coreml_bytes": 1000 * MIB,
            "peak_increment_over_baseline": 50 * MIB, "steady_increment_over_baseline": 10 * MIB,
            "ane_compiler_before_loads": 250 * MIB, "coreml_passes": 10, "ggml_passes_after_ready": 0,
            "segments": segs,
            "status": {"stages": [{"stage": s, "compute_units": "cpu_and_ane"} for s in STAGES]}}


class CheckerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.templates = [template(s) for s in STAGES]

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, *reports):
        args = []
        for i, t in enumerate(self.templates):
            d = self.dir / f"t{i}"
            d.mkdir(exist_ok=True)
            (d / "template.json").write_text(json.dumps(t))
            args += ["--template-dir", str(d)]
        for i, r in enumerate(reports):
            p = self.dir / f"run-{i}.json"
            p.write_text(json.dumps(r))
            args.append(str(p))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = checker.main(args)
        return rc, out.getvalue()

    def assertFails(self, report, needle):
        rc, out = self.check(report)
        self.assertEqual(rc, 1, out)
        self.assertIn(needle, out)

    def test_charged_stage_constants(self):
        self.assertEqual(checker.io_bytes(self.templates[0]), IO_BYTES)
        self.assertEqual(checker.charged(self.templates[0], "cpu_and_ane"),
                         (30 * MIB, 30 * MIB, 300 * MIB, RESIDENT))

    def test_null_constants_use_the_runtime_defaults(self):
        t = template("diarizer-358", runtime_weight_bytes=0, workspace_bytes=0, compile_peak_bytes=None,
                     ane_compile_reserve_bytes=None,
                     load_peak_bytes={"cpu_and_ane": None, "all": None, "cpu_and_gpu": None, "cpu_only": None})
        self.assertEqual(checker.charged(t, "cpu_and_ane"),
                         (2 * MIB + 8 * MIB + 50 * MIB, 25 * MIB, int(1.6 * 100 * MIB), 100 * MIB + 2 * IO_BYTES))
        self.assertEqual(checker.charged(t, "all")[1], 100 * MIB)

    def test_a_run_within_every_charge_passes(self):
        rc, out = self.check(run_report(), run_report())
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS", out)

    def test_a_stage_without_a_load_segment_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if not (s["stage"] == "diarizer-745" and s["phase"] == "load")]
        self.assertFails(r, "diarizer-745 has no load segment")

    def test_a_stage_without_a_generate_segment_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if not (s["stage"] == "diarizer-358" and s["phase"] == "generate")]
        self.assertFails(r, "diarizer-358 has no generate segment")

    def test_a_template_stage_absent_from_the_run_fails(self):
        r = run_report()
        r["segments"] = [s for s in r["segments"] if s["stage"] != "diarizer-745"]
        rc, out = self.check(r)
        self.assertEqual(rc, 1, out)
        self.assertIn("diarizer-745 has no generate segment", out)
        self.assertIn("diarizer-745 has no load segment", out)

    def test_a_stage_without_a_template_fails(self):
        r = run_report()
        r["segments"].append(segment("diarizer-1125", "load", 99, 100 * MIB, 101 * MIB, resident_increment=0))
        self.assertFails(r, "no template for diarizer-1125")

    def test_repeated_loads_are_checked_at_the_largest(self):
        r = run_report()
        # A first load over the charge, then a retry within it.
        r["segments"].insert(0, segment("diarizer-358", "load", -5, 100 * MIB, 140 * MIB, resident_increment=0))
        self.assertFails(r, "diarizer-358 load +40.0 MiB over the charged 30.0 MiB")

    def test_repeated_prepares_are_checked_at_the_largest(self):
        r = run_report()
        r["segments"][0:0] = [segment("diarizer-745", "generate", -10, 100 * MIB, 105 * MIB),
                              segment("diarizer-745", "compile", -9, 105 * MIB, 135 * MIB)]
        self.assertFails(r, "diarizer-745 prepare +35.0 MiB over the charged 30.0 MiB")

    def test_a_prepare_spans_its_compile(self):
        r = run_report()
        r["segments"][1]["peak"] = 131 * MIB  # diarizer-358 compile: 131 - 100 (generate before)
        self.assertFails(r, "diarizer-358 prepare +31.0 MiB over the charged 30.0 MiB")

    def test_the_compiler_is_checked_on_its_growth_not_its_size(self):
        # Peak 500 MiB over a reserve of 300 MiB, but it held 250 MiB before
        # the loads: growth 250 MiB passes.
        rc, out = self.check(run_report())
        self.assertEqual(rc, 0, out)
        r = run_report()
        r["ane_compiler_before_loads"] = 150 * MIB
        self.assertFails(r, "ANECompilerService grew 350.0 MiB during the loads, over the reserve 300.0 MiB")

    def test_the_compiler_peak_is_the_largest_over_the_loads(self):
        r = run_report()
        r["segments"][-2]["ane_compiler_peak"] = 600 * MIB  # the first load, not the last
        self.assertFails(r, "ANECompilerService grew 350.0 MiB")

    def test_a_report_without_the_compiler_baseline_fails(self):
        r = run_report()
        del r["ane_compiler_before_loads"]
        self.assertFails(r, "no ane_compiler_before_loads")

    def test_each_row_labels_the_compiler_baseline(self):
        dirty, clean = run_report(), run_report()
        dirty["ane_compiler_baseline"] = "dirty"
        clean["ane_compiler_baseline"] = "clean"
        rc, out = self.check(dirty, clean, run_report())
        self.assertEqual(rc, 0, out)
        rows = [line for line in out.splitlines() if line.startswith("| run-")]
        self.assertEqual(len(rows), 3, out)
        self.assertIn("dirty baseline", rows[0])
        self.assertIn("clean baseline", rows[1])
        self.assertIn("unknown baseline", rows[2])  # a report without the label

    def test_a_resident_step_over_the_charge_fails(self):
        r = run_report()
        r["segments"][-1]["resident_increment"] = RESIDENT + 1
        self.assertFails(r, "diarizer-745 resident step")

    def test_a_load_without_a_resident_step_fails(self):
        r = run_report()
        del r["segments"][-1]["resident_increment"]
        self.assertFails(r, "diarizer-745 load segment without resident_increment")

    def test_steady_growth_over_the_resident_parts_fails(self):
        r = run_report()
        r["steady_increment_over_baseline"] = 2 * RESIDENT + 1
        self.assertFails(r, "steady growth")
        r["steady_increment_over_baseline"] = 2 * RESIDENT
        rc, out = self.check(r)
        self.assertEqual(rc, 0, out)

    def test_a_peak_increment_over_the_fit_charge_fails(self):
        r = run_report()
        r["peak_increment_over_baseline"] = r["fit_coreml_bytes"] + 1
        self.assertFails(r, "over the fit charge")

    def test_a_run_without_a_baseline_fails(self):
        r = run_report()
        r["baseline_bytes"] = 0
        self.assertFails(r, "not a Core ML run with a ggml-only baseline")

    def test_every_run_is_checked(self):
        bad = run_report()
        bad["segments"][-1]["increment"] = 31 * MIB
        rc, out = self.check(run_report(), copy.deepcopy(bad))
        self.assertEqual(rc, 1, out)
        self.assertIn("run-1.json: diarizer-745 load +31.0 MiB", out)


if __name__ == "__main__":
    unittest.main()
