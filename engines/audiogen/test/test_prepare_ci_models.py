import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare-ci-models.py"
SPEC = importlib.util.spec_from_file_location("prepare_ci_models", SCRIPT)
staging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(staging)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.write_bytes(b"verified model fixture")
        self.destination = self.root / "cache" / "model.gguf"
        self.checksum = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_download_verifies_and_reuses_cached_file(self):
        staging.download_file(self.source.as_uri(), self.destination, self.checksum)
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())
        with mock.patch.object(staging.urllib.request, "urlopen", side_effect=AssertionError("cache downloaded again")):
            staging.download_file(self.source.as_uri(), self.destination, self.checksum)

    def test_corrupt_cache_is_replaced(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"corrupt")
        staging.download_file(self.source.as_uri(), self.destination, self.checksum)
        self.assertEqual(self.destination.read_bytes(), self.source.read_bytes())

    def test_checksum_failure_preserves_previous_file_and_cleans_partial(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"previous")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            staging.download_file(self.source.as_uri(), self.destination, "invalid")
        self.assertEqual(self.destination.read_bytes(), b"previous")
        self.assertFalse(self.destination.with_suffix(".gguf.partial").exists())

    def test_conversion_key_changes_with_converter_contents(self):
        original = staging.conversion_key()
        real_hash = staging.sha256
        with mock.patch.object(staging, "sha256", side_effect=lambda path:
                               "changed" if path.name == "convert-minimax-music3-to-gguf.py" else real_hash(path)):
            self.assertNotEqual(staging.conversion_key(), original)


if __name__ == "__main__":
    unittest.main()
