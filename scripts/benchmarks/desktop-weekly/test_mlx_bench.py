#!/usr/bin/env python3
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_module("mlx_bench", "mlx-bench.py")


def args(**overrides):
    base = bench.parse_args(["--engine", "parakeet-mlx", "--repo", "mlx-community/parakeet-tdt-0.6b-v3", "--revision", "ed2b7e8c", "--model-dir", "m",
                             "--wav", "jfk.wav", "--bench-runs", "3", "--bench-warmup", "1", "--bench-json", "out.json"])
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TimingTest(unittest.TestCase):
    def test_warm_ups_run_untimed_and_the_last_timed_text_is_kept(self):
        calls = []
        fn = lambda: calls.append(len(calls)) or f"text {len(calls)}"
        bench.warm_up(fn, 2)
        times, text = bench.timed_runs(fn, 3)
        self.assertEqual((len(calls), len(times), text), (5, 3, "text 5"))
        self.assertTrue(all(ms >= 0 for ms in times))

    def test_bench_json_has_the_native_driver_fields(self):
        with mock.patch.object(bench, "peak_memory_mib", return_value=2159.66):
            payload = bench.bench_json(args(), 787.94, [84.9, 85.1, 84.0], " the text\n", 11.0)
        self.assertEqual((payload["backend"], payload["engine"], payload["model_revision"], payload["load_ms"]), ("Metal", "parakeet-mlx", "ed2b7e8c", 787.9))
        self.assertEqual(payload["inference_ms"], {"median": 84.9, "min": 84.0, "max": 85.1, "all": [84.9, 85.1, 84.0]})
        self.assertAlmostEqual(payload["rtf_median"], 84.9 / 1000 / 11.0)
        self.assertEqual((payload["transcript"], payload["runs"], payload["warmup"], payload["peak_memory_mib"]), ("the text", 3, 1, 2159.7))

    def test_a_clip_at_another_rate_is_refused(self):
        with self.assertRaises(SystemExit):
            bench.check_rate(44100, 16000)
        bench.check_rate(16000, 16000)


class ArgsTest(unittest.TestCase):
    def test_a_tts_engine_needs_once_text_and_out_and_an_asr_engine_a_clip(self):
        parsed = bench.parse_args(["--engine", "mlx-audio-chatterbox-turbo", "--repo", "r", "--revision", "v", "--model-dir", "m", "--once", "--text", "hi", "--out", "o.wav"])
        self.assertEqual((parsed.engine in bench.TTS_ENGINES, parsed.text, parsed.out), (True, "hi", "o.wav"))
        with self.assertRaises(SystemExit):
            bench.parse_args(["--engine", "mlx-audio-chatterbox-turbo", "--repo", "r", "--revision", "v", "--model-dir", "m", "--text", "hi", "--out", "o.wav"])
        with self.assertRaises(SystemExit):
            bench.parse_args(["--engine", "mlx-audio-nemotron", "--repo", "r", "--revision", "v", "--model-dir", "m", "--bench-json", "b.json"])
        with self.assertRaises(SystemExit):
            bench.parse_args(["--engine", "mlx-audio-nemotron", "--repo", "r", "--revision", "v", "--model-dir", "m", "--wav", "a.wav"])

    def test_once_mode_announces_the_backend_for_the_driver(self):
        with mock.patch.object(bench, "backend_name", return_value="Metal"), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            bench.run_once(lambda: " hello ")
        self.assertEqual(out.getvalue(), "backend: Metal\nhello\n")

    def test_synthesized_audio_joins_the_segments(self):
        class Segment:
            def __init__(self, audio):
                self.audio = audio

        class Model:
            def generate(self, text):
                yield Segment([0.1, 0.2])
                yield Segment([0.3])

        self.assertEqual(list(bench.synthesized_audio(Model(), "hi")), [0.1, 0.2, 0.3])


class SnapshotTest(unittest.TestCase):
    def test_a_pinned_snapshot_is_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            self.assertFalse(bench.pinned(folder, "abc"))
            (folder / bench.REVISION_MARKER).write_text("abc\n", encoding="utf-8")
            self.assertTrue(bench.pinned(folder, "abc"))
            self.assertFalse(bench.pinned(folder, "def"))
            bench.fetch_snapshot("repo", "abc", folder)

    def test_write_json_ends_with_a_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "out.json"
            bench.write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 1})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main()
