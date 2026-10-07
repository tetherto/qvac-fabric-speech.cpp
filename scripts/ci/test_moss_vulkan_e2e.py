"""GPU admission must not mistake a busy shared runner for a model failure."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

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


if __name__ == "__main__":
    unittest.main()
