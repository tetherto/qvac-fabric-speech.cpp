import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXECUTABLE, MODEL, WAV = sys.argv[1:4]
del sys.argv[1:4]


class NpuCliDispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.missing = Path(self.directory.name) / 'missing'

    def run_cli(self, *arguments, wav=WAV):
        return subprocess.run(
            [EXECUTABLE, '--model', MODEL, '--wav', wav, '--backend', 'cpu',
             '--threads', '1', '--npu-encoder', str(self.missing / 'encoder.onnx'),
             *arguments],
            text=True, capture_output=True, check=False,
        )

    def test_stream_does_not_initialize_encoder_before_reading_audio(self):
        result = self.run_cli('--stream', wav=str(self.missing / 'audio.wav'))
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn('failed to load', result.stderr)
        self.assertNotIn('ONNX encoder', result.stderr)
        self.assertNotIn('NPU encoder', result.stderr)

    def test_diarization_initializes_its_own_engine_first(self):
        result = self.run_cli('--diarization-model', MODEL)
        self.assertEqual(result.returncode, 5, result.stderr)
        self.assertIn('not a diarization GGUF', result.stderr)
        self.assertNotIn('ONNX encoder', result.stderr)
        self.assertNotIn('NPU encoder', result.stderr)

    def test_offline_initializes_requested_encoder(self):
        result = self.run_cli()
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertRegex(result.stderr, 'NPU encoder|ONNX encoder')

    def test_stream_forwards_requested_encoder(self):
        result = self.run_cli('--stream')
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertRegex(result.stderr, 'NPU encoder|ONNX encoder')

    def test_profile_does_not_initialize_unused_encoder(self):
        result = self.run_cli('--profile', '--profile-warmup', '0', '--profile-runs', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('encoder attribution', result.stderr)
        self.assertNotIn('ONNX encoder', result.stderr)
        self.assertNotIn('NPU encoder', result.stderr)


if __name__ == '__main__':
    if not Path(MODEL).is_file() or not Path(WAV).is_file():
        print('SKIP: Parakeet TDT model and audio fixtures are required')
        raise SystemExit(77)
    unittest.main()
