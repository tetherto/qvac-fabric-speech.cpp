import subprocess
import sys
import unittest


EXECUTABLE = sys.argv.pop(1)


class NpuCliTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run([EXECUTABLE, *arguments], text=True, capture_output=True, check=False)

    def test_help(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--npu-encoder', result.stderr + result.stdout)
        self.assertIn('--npu-option', result.stderr + result.stdout)

    def test_option_without_separator(self):
        result = self.run_cli('--npu-option', 'invalid')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--npu-option requires KEY=VALUE', result.stderr)

    def test_option_without_key(self):
        result = self.run_cli('--npu-option', '=value')
        self.assertEqual(result.returncode, 2)
        self.assertIn('--npu-option requires KEY=VALUE', result.stderr)


if __name__ == '__main__':
    unittest.main()
