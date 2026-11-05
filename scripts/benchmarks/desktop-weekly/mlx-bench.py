#!/usr/bin/env python3
"""Native-style bench of MLX speech engines for run-family.sh (macOS only).

Loads the model once from a pinned Hugging Face snapshot, decodes the clip with soundfile outside the timed region,
runs the untimed warm-ups and the timed transcriptions, and writes the bench JSON the driver reads for native families
(inference_ms, rtf_median, backend, transcript). With --once it transcribes the clip a single time and prints the text,
for time-wrapped families.
"""
import argparse
import json
import pathlib
import statistics
import time

PARAKEET = "parakeet-mlx"
WHISPER = "mlx-whisper"
ENGINES = (PARAKEET, WHISPER)
BACKEND = "Metal"
REVISION_MARKER = ".revision"
MS_PER_S = 1000.0
BYTES_PER_MIB = 1024 * 1024
LANGUAGE = "en"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--repo", required=True, help="Hugging Face repo of the MLX conversion")
    parser.add_argument("--revision", required=True, help="commit sha of the conversion")
    parser.add_argument("--model-dir", required=True, help="where the snapshot is kept (downloaded on the first run)")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--bench-runs", type=int, default=5)
    parser.add_argument("--bench-warmup", type=int, default=1)
    parser.add_argument("--bench-json")
    parser.add_argument("--once", action="store_true", help="transcribe once and print the text (time-wrapped mode)")
    return parser.parse_args(argv)


def pinned(model_dir, revision):
    marker = pathlib.Path(model_dir) / REVISION_MARKER
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == revision


def fetch_snapshot(repo, revision, model_dir):
    if pinned(model_dir, revision):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=model_dir)
    (pathlib.Path(model_dir) / REVISION_MARKER).write_text(revision + "\n", encoding="utf-8")


def read_clip(path):
    import soundfile
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate


def check_rate(rate, expected):
    if rate != expected:
        raise SystemExit(f"clip sample rate {rate} differs from the model's {expected}")


def load_parakeet(model_dir):
    from parakeet_mlx import from_pretrained
    return from_pretrained(model_dir)


def parakeet_transcriber(model, samples):
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel
    audio = mx.array(samples)
    return lambda: model.generate(get_logmel(audio, model.preprocessor_config))[0].text


def whisper_transcriber(model_dir, samples):
    import mlx_whisper
    return lambda: mlx_whisper.transcribe(samples, path_or_hf_repo=model_dir, language=LANGUAGE, verbose=None)["text"]


def whisper_rate():
    from mlx_whisper.audio import SAMPLE_RATE
    return SAMPLE_RATE


def build_transcriber(engine, model_dir, samples, rate):
    if engine == PARAKEET:
        model = load_parakeet(model_dir)
        check_rate(rate, model.preprocessor_config.sample_rate)
        return parakeet_transcriber(model, samples)
    check_rate(rate, whisper_rate())
    return whisper_transcriber(model_dir, samples)


def timed_call(fn):
    start = time.perf_counter()
    text = fn()
    return (time.perf_counter() - start) * MS_PER_S, text


def warm_up(fn, count):
    for _ in range(count):
        fn()


def timed_runs(fn, count):
    results = [timed_call(fn) for _ in range(count)]
    return [ms for ms, _ in results], results[-1][1] if results else ""


def peak_memory_mib():
    import mlx.core as mx
    return mx.get_peak_memory() / BYTES_PER_MIB


def bench_json(args, load_ms, times, text, audio_seconds):
    median = statistics.median(times)
    return {"engine": args.engine, "backend": BACKEND, "model_repo": args.repo, "model_revision": args.revision, "load_ms": round(load_ms, 1),
            "inference_ms": {"median": round(median, 3), "min": round(min(times), 3), "max": round(max(times), 3), "all": [round(t, 3) for t in times]},
            "rtf_median": median / MS_PER_S / audio_seconds, "audio_seconds": audio_seconds, "runs": len(times), "warmup": args.bench_warmup,
            "transcript": text.strip(), "peak_memory_mib": round(peak_memory_mib(), 1)}


def write_json(path, payload):
    pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_once(fn):
    print(fn().strip())


def run_bench(args, fn, load_ms, audio_seconds):
    warm_up(fn, args.bench_warmup)
    times, text = timed_runs(fn, args.bench_runs)
    payload = bench_json(args, load_ms, times, text, audio_seconds)
    write_json(args.bench_json, payload)
    print(json.dumps({k: payload[k] for k in ("engine", "inference_ms", "rtf_median", "transcript")}))


def main(argv=None):
    args = parse_args(argv)
    fetch_snapshot(args.repo, args.revision, args.model_dir)
    samples, rate = read_clip(args.wav)
    start = time.perf_counter()
    fn = build_transcriber(args.engine, args.model_dir, samples, rate)
    load_ms = (time.perf_counter() - start) * MS_PER_S
    if args.once:
        return run_once(fn)
    if not args.bench_json:
        raise SystemExit("--bench-json is required without --once")
    run_bench(args, fn, load_ms, len(samples) / rate)


if __name__ == "__main__":
    main()
