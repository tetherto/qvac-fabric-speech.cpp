"""Tests for Nemotron 3 Diarization model preparation scripts."""

import tempfile
import unittest
from pathlib import Path

import convert_nemotron_diarization as converter
import download_nemotron_diarization as downloader


class NemotronPreparationTests(unittest.TestCase):
    def test_downloads_pinned_gguf_and_source(self):
        calls = []

        def fake_download(**arguments):
            calls.append(arguments)
            return str(Path(arguments["local_dir"]) / arguments["filename"])

        with tempfile.TemporaryDirectory() as temporary:
            model_directory = Path(temporary) / "models"
            artifacts = downloader.download_model(
                model_directory, include_source=True, download=fake_download)

        self.assertEqual(
            [artifact.name for artifact in artifacts],
            [downloader.GGUF_FILENAME, downloader.NEMO_FILENAME],
        )
        self.assertEqual(calls[0]["repo_id"], downloader.MODEL_REPOSITORY)
        self.assertEqual(calls[0]["revision"], downloader.MODEL_REVISION)
        self.assertEqual(calls[1]["revision"], downloader.MODEL_REVISION)

    def test_conversion_preserves_official_gguf(self):
        calls = []

        def fake_run(command, **arguments):
            calls.append((command, arguments))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / downloader.NEMO_FILENAME
            output = directory / "converted" / converter.OUTPUT_FILENAME
            script = directory / "converter" / "convert_model.py"
            converter.convert_model(source, output, script, run=fake_run)
            resolved_source = source.resolve()
            resolved_output = output.resolve()

        command, arguments = calls[0]
        self.assertEqual(command[1], str(script))
        self.assertEqual(Path(command[2]), resolved_source)
        self.assertEqual(command[3:], ["--outfile", str(resolved_output), "--outtype", "q8_0"])
        self.assertEqual(arguments["cwd"], script.parent)
        self.assertTrue(arguments["check"])
        self.assertNotEqual(output.name, downloader.GGUF_FILENAME)

    def test_uses_existing_converter_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            script = directory / "convert_model.py"
            script.touch()
            self.assertEqual(converter.ensure_converter(directory), script)

    def test_fetches_pinned_converter_source(self):
        calls = []

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "NeMo-Speech.cpp"
            script = directory / "convert_model.py"

            def fake_run(command, **arguments):
                calls.append(command)
                if "checkout" in command:
                    directory.mkdir(parents=True)
                    script.touch()

            self.assertEqual(converter.ensure_converter(directory, run=fake_run), script)

        self.assertEqual(calls[0][1], "clone")
        self.assertEqual(calls[1][-1], converter.CONVERTER_REVISION)
        self.assertEqual(calls[2][-1], converter.CONVERTER_REVISION)


if __name__ == "__main__":
    unittest.main()
