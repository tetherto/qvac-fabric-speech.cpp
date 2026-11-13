"""Check generated CTest environments without requiring a Windows compiler."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CMAKE, CTEST, MODULE = sys.argv[1:4]
del sys.argv[1:4]
INHERITED_WINDOWS_PATH = "C:/Program Files/Python;D:/tools"
WRAPPERS = ("test-streaming-fixture-validation", "test-voice-features-reference")
PROJECT = """
cmake_minimum_required(VERSION 3.20)
project(test_runtime_environment NONE)
enable_testing()
include("${ENVIRONMENT_MODULE}")
set(WIN32 ${SIMULATE_WINDOWS})
set(ENV{PATH} "C:/Program Files/Python;D:/tools")
foreach(name test-streaming-fixture-validation test-voice-features-reference)
    add_test(NAME ${name} COMMAND "${CMAKE_COMMAND}" -E true)
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    tts_cpp_set_test_runtime_environment(${name})
endforeach()
add_test(NAME native-harness COMMAND "${CMAKE_COMMAND}" -E true)
tts_cpp_set_test_runtime_environment(native-harness)
"""


class RuntimeEnvironmentTests(unittest.TestCase):
    def check_environment(self, windows, generator=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build with spaces"
            (root / "CMakeLists.txt").write_text(PROJECT, encoding="utf-8")
            command = [CMAKE, "-S", str(root), "-B", str(build),
                       f"-DENVIRONMENT_MODULE={MODULE}",
                       f"-DSIMULATE_WINDOWS={'ON' if windows else 'OFF'}",
                       "-DCMAKE_BUILD_TYPE=Release"]
            if generator:
                command.extend(["-G", generator])
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run(
                [CTEST, "--test-dir", str(build), "-C", "Release", "--show-only=json-v1"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            tests = json.loads(result.stdout)["tests"]
            self.assertEqual({test["name"] for test in tests}, {*WRAPPERS, "native-harness"})
            for test in tests:
                properties = {prop["name"]: prop["value"] for prop in test["properties"]}
                environment = dict(value.split("=", 1) for value in properties.get("ENVIRONMENT", []))
                if test["name"] in WRAPPERS:
                    self.assertEqual(environment.pop("PYTHONDONTWRITEBYTECODE"), "1")
                if windows:
                    expected_path = (f"{build.as_posix()}/bin/Release;{build.as_posix()}/bin;"
                                     f"{INHERITED_WINDOWS_PATH}")
                    self.assertEqual(environment, {"PATH": expected_path})
                else:
                    self.assertEqual(environment, {})

    def test_windows_single_config(self):
        self.check_environment(True, "Ninja" if shutil.which("ninja") else None)

    @unittest.skipUnless(shutil.which("ninja"), "Ninja required for multi-config generation")
    def test_windows_multi_config(self):
        self.check_environment(True, "Ninja Multi-Config")

    def test_other_platform_preserves_environment(self):
        self.check_environment(False, "Ninja" if shutil.which("ninja") else None)


if __name__ == "__main__":
    unittest.main()
