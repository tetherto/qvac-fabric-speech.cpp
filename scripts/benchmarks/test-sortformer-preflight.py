#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("preflight", HERE / "sortformer-preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class RoutingTests(unittest.TestCase):
    def test_all_passes_must_match(self):
        self.assertEqual(PREFLIGHT.validate_routes(
            "diarization-route: pass=1 coreml=1 frames=12\n"
            "diarization-route: pass=2 coreml=1 frames=8", "legacy"),
            {"network_passes": 2, "coreml_passes": 2})
        self.assertEqual(PREFLIGHT.validate_routes(
            "diarization-route: pass=1 coreml=0 frames=12", "metal")["coreml_passes"], 0)

    def test_rejects_availability_fallback_and_incomplete_evidence(self):
        cases = ("Core ML encoder loaded", "diarization-route: pass=1 coreml=0 frames=12",
                 "diarization-route: pass=2 coreml=1 frames=12",
                 "diarization-route: pass=1 coreml=1 frames=0",
                 "diarization-route: pass=1 coreml=1 frames=12\n"
                 "diarization-route: pass=2 coreml=0 frames=12")
        for log in cases:
            with self.subTest(log=log), self.assertRaises(ValueError):
                PREFLIGHT.validate_routes(log, "diarizer")

    def test_environment_does_not_inherit_disabled_routes(self):
        with patch.dict(os.environ, {"PARAKEET_COREML_DISABLE": "1", "PARAKEET_COREML_ONLY": "1"}):
            self.assertNotIn("PARAKEET_COREML_DISABLE", PREFLIGHT.arm_environment("legacy"))
            self.assertNotIn("PARAKEET_COREML_ONLY", PREFLIGHT.arm_environment("metal"))
            self.assertEqual(PREFLIGHT.arm_environment("metal")["PARAKEET_COREML_DISABLE"], "1")

    def test_missing_assets_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing model"):
                PREFLIGHT.stage_arm(root, root / "first", "legacy")
            (root / PREFLIGHT.MODEL_NAME).write_bytes(b"model")
            with self.assertRaisesRegex(ValueError, "incomplete bundle"):
                PREFLIGHT.stage_arm(root, root / "second", "legacy")

    def test_arms_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / PREFLIGHT.MODEL_NAME).write_bytes(b"model")
            bundle = root / (PREFLIGHT.STEM + "-diarizer-410.mlmodelc")
            (bundle / "weights").mkdir(parents=True)
            (bundle / "weights/weight.bin").write_bytes(b"weights")
            (bundle / "model.mil").write_text("model")
            PREFLIGHT.stage_arm(root, root / "metal", "metal")
            PREFLIGHT.stage_arm(root, root / "diarizer", "diarizer")
            self.assertEqual(len(list((root / "metal").iterdir())), 1)
            self.assertEqual(len(list((root / "diarizer").iterdir())), 2)
            self.assertEqual(PREFLIGHT.hash_file(root / "metal" / PREFLIGHT.MODEL_NAME),
                             PREFLIGHT.hash_file(root / "diarizer" / PREFLIGHT.MODEL_NAME))

    def test_summary_groups_runs_and_chunk_sizes(self):
        rows = [{"arm": "metal", "fixture": "abcba", "chunk_ms": 1000, "der": value}
                for value in (0, 0, 1)]
        rows.append({"arm": "metal", "fixture": "abcba", "chunk_ms": 2000, "der": 0.5})
        self.assertEqual([row["der_median"] for row in PREFLIGHT.summarize(rows)], [0, 0.5])

    def test_cli_preserves_staged_path_and_writes_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "abcba.rttm").write_text("SPEAKER sample 1 0 4 <NA> <NA> A <NA> <NA>\n")
            args = SimpleNamespace(binary=root / "binary", samples=root, out=root)
            model = root / "staged" / PREFLIGHT.MODEL_NAME
            def invoke(command, **kwargs):
                self.assertEqual(command[2], str(model))
                self.assertEqual(command[command.index("--backend") + 1], "metal")
                kwargs["stdout"].write('{"speaker":0,"start":0,"end":4}\n')
                kwargs["stderr"].write("diarization-route: pass=1 coreml=1 frames=50\n")
            with patch.object(PREFLIGHT.subprocess, "run", side_effect=invoke):
                result = PREFLIGHT.run_case(args, model, "legacy", "abcba", 1000, 1)
            self.assertEqual(result["der"], 0)
            self.assertEqual(result["coreml_passes"], 1)
            self.assertTrue((root / "legacy-abcba-1000ms-1.score.json").is_file())


class DriverMedianTests(unittest.TestCase):
    def run_driver(self, count, malformed=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build/bin").mkdir(parents=True)
            reference = root / "reference.rttm"
            reference.write_text("SPEAKER sample 1 0 4 <NA> <NA> A <NA> <NA>\n")
            binary = root / "build/bin/probe"
            binary.write_text("#!/usr/bin/env python3\nfrom pathlib import Path\n"
                              f"p=Path({str(root / 'count')!r})\n"
                              "n=int(p.read_text())+1 if p.exists() else 1\np.write_text(str(n))\n"
                              f"if n < {count}: print('{{\"speaker\":0,\"start\":0,\"end\":4}}')\n"
                              + ("else: print('{\"speaker\":0,\"start\":\"invalid\",\"end\":4}')\n" if malformed else ""))
            binary.chmod(0o755)
            spec = {"probe": {"bench_kind": "time-wrapped", "binary": "bin/probe", "cmake_target": "unused",
                              "models": [], "args": [], "audio_duration_seconds": 4,
                              "correctness": {"kind": "der", "reference": str(reference), "collar_ms": 0}}}
            (root / "spec.json").write_text(json.dumps(spec))
            environment = {**os.environ, "BENCH_FAMILIES_JSON": str(root / "spec.json")}
            environment.pop("MODEL_S3_BUCKET", None)
            subprocess.run(["bash", str(HERE / "run-family.sh"), "--family", "probe", "--runs", str(count),
                            "--warmup", "0", "--build-dir", str(root / "build"), "--models-root", str(root / "models"),
                            "--out", str(root / "result.json")], env=environment, check=True, capture_output=True)
            hypothesis = root / f"result.run-{count}.hypothesis.jsonl"
            self.assertTrue(hypothesis.is_file())
            if malformed:
                self.assertIn("invalid", hypothesis.read_text())
            return json.loads((root / "result.json").read_text()), json.loads((root / "result.der-runs.json").read_text())

    def test_odd_median_uses_every_run(self):
        result, report = self.run_driver(3)
        self.assertEqual(result["der_median"], 0)
        self.assertEqual([run["der"] for run in report["runs"]], [0, 0, 1])

    def test_even_median(self):
        result, report = self.run_driver(2)
        self.assertEqual(result["der_median"], 0.5)
        self.assertTrue(report["complete"])

    def test_scoring_failure_is_not_a_partial_median(self):
        result, report = self.run_driver(3, malformed=True)
        self.assertIsNone(result["der_median"])
        self.assertFalse(report["complete"])
        self.assertIsNone(report["runs"][-1]["der"])


if __name__ == "__main__":
    unittest.main()
