"""Admission and intelligibility gates must not report false passes."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid
import wave

spec = importlib.util.spec_from_file_location("moss_e2e", Path(__file__).with_name("moss-vulkan-e2e.py"))
e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e)


class AdmissionTests(unittest.TestCase):
    def test_requires_consecutive_headroom_samples(self):
        memory = [{"uuid": "GPU-a", "total_mib": 20475, "free_mib": free}
                  for free in [17067, 20000, 17000, 20000, 20000]]
        with patch.object(e2e, "gpu_memory", side_effect=memory) as query, \
             patch.object(e2e.time, "monotonic", return_value=0), \
             patch.object(e2e.time, "sleep"):
            e2e.wait_for_gpu(100, io.StringIO(), "GPU-a")
        self.assertEqual(query.call_count, 5)

    def test_busy_gpu_times_out_before_inference(self):
        with patch.object(e2e, "gpu_memory", return_value={
                "uuid": "GPU-a", "total_mib": 20475, "free_mib": 17067}), \
             patch.object(e2e.time, "monotonic", side_effect=[0, 10]), \
             patch.object(e2e.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "inference was not started"):
                e2e.wait_for_gpu(5, io.StringIO(), "GPU-a")

    def test_lock_excludes_other_process_and_releases_after_failure(self):
        gpu = "GPU-" + str(uuid.uuid4())
        path = Path("/tmp") / f"qvac-moss-e2e-{gpu}.lock"
        probe = """import fcntl, sys
with open(sys.argv[1]) as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(7)
"""
        try:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(e2e, "gpu_memory", return_value={"uuid": gpu}), \
                 patch.object(e2e, "wait_for_gpu"):
                with self.assertRaisesRegex(RuntimeError, "synthesis failed"):
                    with e2e.gpu_lease(Path(directory), "test"):
                        self.assertEqual(subprocess.run([sys.executable, "-c", probe, str(path)]).returncode, 7)
                        raise RuntimeError("synthesis failed")
                self.assertEqual(subprocess.run([sys.executable, "-c", probe, str(path)]).returncode, 0)
        finally:
            path.unlink(missing_ok=True)


class IntelligibilityTests(unittest.TestCase):
    def test_existing_quality_issue_requires_exact_upstream_audio(self):
        bad = {"status": "ok", "wer": 1.0, "n_ref_words": 8}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "tts-baseline-commit.txt").write_text(e2e.TTS_BASELINE_SHA)
            def write(name, samples):
                with wave.open(str(output / (name + ".wav")), "wb") as wav:
                    wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    wav.writeframes(samples)
            for name in ("batch", "baseline-batch", "clone", "baseline-clone"):
                write(name, b"\x01\x00" * 20)
            # Known unconditioned defect is exempt; cloning never is.
            self.assertEqual(e2e.classify_tts_scores({"batch": bad, "clone": bad}, output),
                             (["clone"], ["batch"]))
            write("batch", b"\x02\x00" * 20)
            self.assertEqual(e2e.classify_tts_scores({"batch": bad}, output), (["batch"], []))
            (output / "baseline-batch.wav").unlink()
            self.assertEqual(e2e.classify_tts_scores({"batch": bad}, output), (["batch"], []))

    def test_unverified_baseline_or_asr_error_cannot_exempt_a_failure(self):
        bad = {"status": "ok", "wer": 1.0, "n_ref_words": 8}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(e2e, "same_audio", return_value=True):
                self.assertEqual(e2e.classify_tts_scores({"batch": bad}, output), (["batch"], []))
                (output / "tts-baseline-commit.txt").write_text("wrong-commit")
                self.assertEqual(e2e.classify_tts_scores({"batch": bad}, output), (["batch"], []))
                (output / "tts-baseline-commit.txt").write_text(e2e.TTS_BASELINE_SHA)
                error = {**bad, "status": "error"}
                self.assertEqual(e2e.classify_tts_scores({"batch": error}, output), (["batch"], []))

    def test_wrong_or_empty_speech_fails_using_shared_word_error_scorer(self):
        spec = importlib.util.spec_from_file_location("wer", Path(__file__).parents[1] / "benchmarks/compute-wer.py")
        wer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wer)
        for transcript, expected in [(e2e.TTS_TEXT, True), ("No more...", False), ("", False)]:
            with self.subTest(transcript=transcript):
                score = {"status": "ok", **wer.compute_wer(transcript, e2e.TTS_TEXT)}
                self.assertEqual(e2e.intelligibility_passes(score), expected)

    def test_unavailable_missing_or_nonfinite_scores_cannot_pass(self):
        for score in [{}, {"status": "error", "wer": 0, "n_ref_words": 8},
                      {"status": "ok", "wer": None, "n_ref_words": 8},
                      {"status": "ok", "wer": float("nan"), "n_ref_words": 8},
                      {"status": "ok", "wer": 0, "n_ref_words": 0}]:
            with self.subTest(score=score):
                self.assertFalse(e2e.intelligibility_passes(score))


if __name__ == "__main__":
    unittest.main()
