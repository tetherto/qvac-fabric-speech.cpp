import importlib.util
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run-fixture-tests.py"
SPEC = importlib.util.spec_from_file_location("fixture_coverage", SCRIPT)
COVERAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COVERAGE)


def registration(name, disabled=False, label="fixture"):
    return {"name": name, "properties": [
        {"name": "LABELS", "value": [label]},
        {"name": "DISABLED", "value": disabled},
    ]}


class FixtureCoverageTest(unittest.TestCase):
    def validate(self, cases, selected=("encoder",), required=()):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.xml"
            report.write_text(f"<testsuite>{cases}</testsuite>", encoding="utf-8")
            COVERAGE.validate_report(report, selected, required)

    def test_rejects_absent_or_disabled_coverage(self):
        for tests in ([], [registration("encoder", disabled=True)], [registration("unit", label="unit")]):
            with self.subTest(tests=tests), self.assertRaises(ValueError):
                COVERAGE.select_tests(tests, "fixture", [])

    def test_selects_only_enabled_tests_in_label(self):
        tests = [registration("encoder"), registration("missing", True), registration("unit", label="unit")]
        self.assertEqual(COVERAGE.select_tests(tests, "fixture", ["encoder"]), ["encoder"])

    def test_required_names_select_only_requested_tests(self):
        tests = [registration("encoder"), registration("decoder")]
        self.assertEqual(COVERAGE.select_tests(tests, "fixture", ["encoder"]), ["encoder"])

    def test_requires_every_requested_family(self):
        tests = [registration("encoder"), registration("decoder", True)]
        for required in (["decoder"], ["unregistered"]):
            with self.subTest(required=required), self.assertRaises(ValueError):
                COVERAGE.select_tests(tests, "fixture", required)

    def test_accepts_executed_required_test(self):
        self.validate('<testcase name="encoder" status="run"/>', required=["encoder"])

    def test_rejects_empty_or_missing_results(self):
        for cases in ("", '<testcase name="other" status="run"/>'):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.validate(cases)

    def test_rejects_only_skipped_results(self):
        for cases in ('<testcase name="encoder"><skipped/></testcase>',
                      '<testcase name="encoder" status="notrun"/>'):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                self.validate(cases)

    def test_optional_skip_does_not_hide_required_skip(self):
        cases = '<testcase name="encoder"/><testcase name="decoder"><skipped/></testcase>'
        self.validate(cases, selected=["encoder", "decoder"], required=["encoder"])
        with self.assertRaises(ValueError):
            self.validate(cases, selected=["encoder", "decoder"], required=["decoder"])

    def test_rejects_failures_and_errors(self):
        for result in ("failure", "error"):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.validate(f'<testcase name="encoder"><{result}/></testcase>')

    def test_reference_suites_report_missing_dependencies(self):
        for name in ("test_dump_nemotron_reference.py", "test_dump_unified_reference.py"):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, "-S", str(Path(__file__).with_name(name))],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 77, result.stdout + result.stderr)


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ctest"), "requires CMake and CTest")
class FixtureRunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        source = Path(cls.temporary.name)
        cls.build = source / "build"
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.21)\n'
            'project(fixture-runner NONE)\n'
            'enable_testing()\n'
            'add_test(NAME encoder.check COMMAND "${CMAKE_COMMAND}" -E true)\n'
            f'add_test(NAME unavailable COMMAND "{Path(sys.executable).as_posix()}" -c "raise SystemExit(3)")\n'
            'set_tests_properties(encoder.check unavailable PROPERTIES LABELS fixture SKIP_RETURN_CODE 3)\n',
            encoding="utf-8",
        )
        subprocess.run(["cmake", "-S", str(source), "-B", str(cls.build)],
                       capture_output=True, text=True, check=True)

    def run_fixture(self, required):
        arguments = types.SimpleNamespace(test_dir=self.build, label="fixture", require=required,
                                         configuration=None, timeout=10)
        COVERAGE.run_tests(arguments)

    def test_reads_real_ctest_results(self):
        self.run_fixture([])
        self.run_fixture(["encoder.check"])

    def test_rejects_real_runtime_skip(self):
        with self.assertRaisesRegex(ValueError, "did not execute successfully"):
            self.run_fixture(["unavailable"])


if __name__ == "__main__":
    unittest.main()
