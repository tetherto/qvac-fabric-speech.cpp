#!/usr/bin/env python3
"""Full-checkpoint Vulkan smoke tests; model files come from the model registry.

Run from the repository root after building moss-cli and moss-transcribe.
Requires AWS CLI credentials and MODEL_S3_BUCKET. Outputs contain logs, WAVs,
transcripts and checkpoint hashes, never checkpoint weights. This checks native
inference, not addon integration or perceptual audio quality.
"""

import argparse
import array
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import wave


CASES = {
    "tts-f16": "moss-tts-delay-f16",
    "ttsd-f16": "moss-ttsd-f16",
    "sfx-f16": "moss-sfx-v2-f16",
    "sfx-q8_0": "moss-sfx-v2-q8_0",
    "speech-bf16": "moss-speech-bf16",
    "speech-q8_0": "moss-speech-q8_0",
    "transcribe-f16": "moss-transcribe-diarize-f16",
    "transcribe-q5_0": "moss-transcribe-diarize-q5_0",
    "transcribe-q8_0": "moss-transcribe-diarize-q8_0",
}

TTS_TEXT = "Hello, this is a test of speech synthesis."
TTS_LONG_TEXT = (TTS_TEXT + " We are checking that every audio chunk arrives in order, "
                 "including the final part of this sentence.")
# Same immutable reference model as the existing TTS intelligibility benchmark.
ASR_SPEC = {
    "repo": "ggerganov/whisper.cpp",
    "revision": "5359861c739e955e79d9a303bcbc70fb988958b1",
    "name": "ggml-tiny.bin",
    "sha256": "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
}
MAX_TTS_WER = 0.25
TTS_BASELINE_SHA = "8ae24fffce8d25fe4bfcc9b8d81f51374b29e65c"
UNCONDITIONED_OUTPUTS = ("batch", "stream", "cpu-reference")


def intelligibility_passes(score):
    wer = score.get("wer")
    return (score.get("status") == "ok" and isinstance(wer, (float, int)) and
            math.isfinite(wer) and 0 <= wer <= MAX_TTS_WER and score.get("n_ref_words", 0) > 0)


def same_audio(left, right):
    # Compare signal data, not incidental WAV metadata. Exact agreement is
    # deliberately required before exempting an existing wrong-text result.
    try:
        with wave.open(str(left), "rb") as a, wave.open(str(right), "rb") as b:
            return (a.getparams() == b.getparams() and a.getnframes() > 0 and
                    a.readframes(a.getnframes()) == b.readframes(b.getnframes()))
    except (OSError, wave.Error, EOFError):
        return False


def classify_tts_scores(scores, output):
    failed, existing = [], []
    baseline = output / "tts-baseline-commit.txt"
    verified_base = baseline.is_file() and baseline.read_text().strip() == TTS_BASELINE_SHA
    for name, score in scores.items():
        if intelligibility_passes(score):
            continue
        # A failed ASR invocation is never an existing quality limitation.
        wer = score.get("wer")
        valid_score = (score.get("status") == "ok" and isinstance(wer, (float, int)) and
                       math.isfinite(wer) and wer > MAX_TTS_WER and score.get("n_ref_words", 0) > 0)
        if (verified_base and valid_score and name in UNCONDITIONED_OUTPUTS and
                same_audio(output / f"{name}.wav", output / f"baseline-{name}.wav")):
            existing.append(name)
        else:
            failed.append(name)
    return failed, existing


def score_tts_outputs(asr_binary, models, output):
    benchmarks = Path(__file__).resolve().parents[1] / "benchmarks"
    spec = importlib.util.spec_from_file_location("prepare_asr", benchmarks / "prepare-tts-asr.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    model = prepare.prepare(ASR_SPEC, models.resolve())
    expected = {name: "short" for name in ["batch", "stream", "clone", "cpu-reference"]}
    for chunk in (7, 25):
        for suffix in ("batch", "stream", "after-callback-cancel", "after-explicit-cancel"):
            expected[f"tts-c{chunk}-{suffix}"] = "short"
    expected.update({"tts-long-batch": "long", "tts-long-stream": "long"})
    scores = {}
    for name, length in expected.items():
        audio = output / f"{name}.wav"
        if not audio.is_file():
            scores[name] = {"status": "missing", "wer": None}
            continue
        score_path = output / f"{name}-intelligibility.json"
        # The scorer uses CPU, a fresh transcript, and deterministic ASR settings.
        # Expected text is only used for scoring; it is never an ASR prompt.
        completed = subprocess.run([
            sys.executable, str(benchmarks / "compute-tts-intelligibility.py"),
            "--audio", str(audio), "--reference", str(output / f"reference-{length}.txt"),
            "--asr-binary", str(asr_binary), "--asr-model", str(model),
            "--json-out", str(score_path), "--transcript-out", str(output / f"{name}-transcript.txt"),
            "--log-out", str(output / f"{name}-asr.log")], check=False, timeout=360)
        score = json.loads(score_path.read_text()) if score_path.exists() else {"status": "error", "wer": None}
        if completed.returncode != 0:
            score["status"] = "error"
        scores[name] = {key: score.get(key) for key in ("status", "wer", "n_ref_words", "n_hyp_words", "n_edits")}
    failed, existing = classify_tts_scores(scores, output)
    return {"status": "failed" if failed else "passed", "max_wer": MAX_TTS_WER,
            "asr_model": ASR_SPEC, "failed_outputs": failed, "scores": scores,
            "quality_status": "failed" if failed or existing else "passed",
            "pre_existing_quality_issues": existing, "baseline_commit": TTS_BASELINE_SHA}


def gpu_memory():
    row = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=30).strip().split(",")
    uuid, total, free = (value.strip() for value in row)
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise RuntimeError("Cannot identify physical GPU for admission control")
    return {"uuid": uuid, "total_mib": int(total), "free_mib": int(free)}


def wait_for_gpu(deadline, report, expected_uuid):
    # Other repositories do not share our matrix limit or lock. Require near-idle
    # memory twice before loading a checkpoint; never terminate foreign jobs.
    ready = 0
    while time.monotonic() < deadline:
        state = gpu_memory()
        report.write(json.dumps(state) + "\n")
        report.flush()
        if state["uuid"] != expected_uuid:
            raise RuntimeError("GPU identity changed while waiting")
        ready = ready + 1 if state["free_mib"] >= state["total_mib"] - 1024 else 0
        if ready == 2:
            return
        print(f"GPU admission: {state['free_mib']}/{state['total_mib']} MiB free; "
              "waiting for stable headroom", flush=True)
        time.sleep(5)
    raise RuntimeError("GPU admission timed out: shared GPU is busy; inference was not started")


@contextmanager
def gpu_lease(output, name):
    state = gpu_memory()
    path = Path("/tmp") / f"qvac-moss-e2e-{state['uuid']}.lock"
    # Keep the inode: unlinking a lock permits two independent owners. Read-only
    # descriptors allow runner accounts to share flock without granting writes.
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.fchmod(fd, 0o644)
    except FileExistsError:
        fd = os.open(path, os.O_RDONLY)
    deadline = time.monotonic() + 900
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("GPU admission timed out waiting for another OpenMOSS run")
                time.sleep(5)
        with (output / f"{name}-gpu-admission.jsonl").open("w") as report:
            wait_for_gpu(deadline, report, state["uuid"])
        yield
    finally:
        os.close(fd)


def download(name, models, output):
    date = "2026-09-25" if name in {
        "moss-tts-delay-f16", "moss-codec-decoder-f16", "moss-codec-encoder-f16"
    } else "2026-10-01"
    key = f"qvac_models_compiled/ggml/openmoss/{date}/{name}.gguf"
    target = models / f"{name}.gguf"
    subprocess.run([
        "aws", "s3", "cp", "--only-show-errors",
        f"s3://{os.environ['MODEL_S3_BUCKET']}/{key}", str(target)
    ], check=True, timeout=1800)
    digest = hashlib.sha256()
    with target.open("rb") as source:
        if source.read(4) != b"GGUF":
            raise RuntimeError(f"Not a GGUF: {name}")
        source.seek(0)
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    with (output / "models.sha256").open("a") as report:
        report.write(f"{digest.hexdigest()}  {key}\n")
    return str(target)


def audio_stats(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise RuntimeError(f"Expected mono PCM16: {path}")
        rate = audio.getframerate()
        samples = array.array("h", audio.readframes(audio.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    if len(samples) < rate // 4 or not any(samples):
        raise RuntimeError(f"Empty, silent or shorter than 250 ms: {path}")
    return {"seconds": len(samples) / rate, "sample_rate": rate,
            "peak": max(abs(x) for x in samples) / 32768,
            "rms": math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768}


def reference(source, target, offset, seconds=3):
    # Short real-speech excerpts, resampled to the TTS codec's required 24 kHz.
    with wave.open(str(source), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise RuntimeError("Expected mono PCM16 reference")
        rate = audio.getframerate()
        audio.setpos(offset * rate)
        samples = array.array("h", audio.readframes(seconds * rate))
    if sys.byteorder != "little":
        samples.byteswap()
    resampled = array.array("h")
    for i in range(len(samples) * 24000 // rate):
        position = i * rate / 24000
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        resampled.append(round(samples[left] + (samples[right] - samples[left]) * (position - left)))
    if sys.byteorder != "little":
        resampled.byteswap()
    with wave.open(str(target), "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(resampled.tobytes())


def run_case(case, build, models, output, baseline_binary):
    model = download(CASES[case], models, output)
    family = case.split("-")[0]
    sample = Path("engines/parakeet/test/samples/jfk.wav")
    tts = str(build / "engines/tts/moss-cli")
    common = ["--gpu", "--threads", "4", "--seed", "1234"]
    checks = {}

    def invoke(name, args, wav=True, cpu=False):
        log = output / f"{name}.log"
        # Capture memory immediately around inference, not just before the
        # build/download. Include UUIDs to identify runner services sharing a GPU.
        def gpu_snapshot(phase):
            with (output / f"{name}-gpu-{phase}.txt").open("w") as stream:
                for query in [
                    ["--query-gpu=uuid,name,memory.total,memory.used,memory.free", "--format=csv"],
                    ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"],
                ]:
                    subprocess.run(["nvidia-smi", *query], stdout=stream,
                                   stderr=subprocess.STDOUT, check=False, timeout=30)

        try:
            with nullcontext() if cpu else gpu_lease(output, name):
                gpu_snapshot("before")
                with log.open("w") as stream:
                    stream.write(json.dumps(args) + "\n")
                    stream.flush()
                    subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT,
                                   check=True, timeout=1200,
                                   env={**os.environ, "TTS_CPP_GPU_BACKEND": "vulkan"})
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Put the actual engine error in the job log as well as the artifact.
            print(log.read_text(errors="replace")[-12000:], file=sys.stderr, flush=True)
            raise
        finally:
            gpu_snapshot("after")
        backend = "CPU" if cpu else "Vulkan"
        if not re.search(r"\[(?:moss-cli|moss-transcribe|moss-speech-e2e|moss-speech-codec-e2e|moss-tts-stream-e2e)\] backend: " + backend + r"\d*", log.read_text()):
            raise RuntimeError(f"{name} did not select {backend}; see {log}")
        checks[name] = audio_stats(output / f"{name}.wav") if wav else {"backend": backend}
        print(f"PASS {name}: {checks[name]}", flush=True)

    if family in {"tts", "ttsd"}:
        decoder = download("moss-codec-decoder-f16", models, output)
        encoder = download("moss-codec-encoder-f16", models, output)
        ref1, ref2 = output / "reference1.wav", output / "reference2.wav"
        reference(sample, ref1, 0)
        reference(sample, ref2, 3, 5)
        base = [tts, "--backbone", model, "--decoder", decoder, "--language", "en",
                "--max-new-tokens", "192", "--context", "4096", *common]
        if family == "tts":
            base += ["--text", TTS_TEXT]
            for name, extra in [("batch", []), ("stream", ["--stream", "--stream-chunk-frames", "25"]),
                                ("clone", ["--encoder", encoder, "--ref-audio", str(ref1)])]:
                invoke(name, [*base, *extra, "--out", str(output / f"{name}.wav")])
            # Same checkpoint, text and seed: distinguish common prompt/model
            # failures from backend-dependent generation drift. CI only.
            invoke("cpu-reference", [arg for arg in base if arg != "--gpu"] +
                   ["--out", str(output / "cpu-reference.wav")], cpu=True)
            # Same worker, ggml pin, checkpoint, prompt, seed and backend on
            # the upstream base. Preserve evidence of pre-existing defects.
            for name in UNCONDITIONED_OUTPUTS:
                args = [str(baseline_binary), *base[1:]]
                if name == "cpu-reference":
                    args.remove("--gpu")
                if name == "stream":
                    args += ["--stream", "--stream-chunk-frames", "25"]
                invoke(f"baseline-{name}", [*args, "--out", str(output / f"baseline-{name}.wav")],
                       cpu=name == "cpu-reference")
            invoke("stream-agreement", [str(build / "engines/tts/test-moss-tts-stream-e2e"),
                   model, decoder, encoder, str(ref1), str(output),
                   str(output / "reference-short.txt"), str(output / "reference-long.txt")], wav=False)
        else:
            # Two reference slots exercise dialogue conditioning; these are
            # excerpts from the same speaker, not a voice identity quality test.
            invoke("dialogue", [*base, "--encoder", encoder, "--dialogue-ref", str(ref1),
                   "--dialogue-ref", str(ref2), "--text",
                   "[S1] And so, my fellow Americans. [S2] Ask not what your country can do for you. "
                   "[S1] Hello there. [S2] How are you?",
                   "--out", str(output / "dialogue.wav")])
    elif family == "sfx":
        invoke("sound", [tts, "--mode", "sfx", "--model", model, *common,
               "--text", "Rain falling on a tin roof.", "--seconds", "3",
               "--out", str(output / "sound.wav")])
    elif family == "speech":
        codec = download("moss-speech-codec-f16", models, output)
        invoke("codec-agreement", [str(build / "engines/tts/test-moss-speech-codec-e2e"),
               codec, str(output)], wav=False)
        for mode in ("single", "cfg"):
            checks[f"codec-{mode}"] = audio_stats(output / f"codec-vulkan-{mode}.wav")
        invoke("reply", [str(build / "engines/tts/test-moss-speech-e2e"),
               model, codec, str(sample), str(output)])
        checks["reply-repeat"] = audio_stats(output / "reply-repeat.wav")
    else:
        transcript = output / "transcript.json"
        invoke("transcribe", [str(build / "engines/parakeet/moss-transcribe"),
               "--model", model, "--audio", str(sample), "--backend", "vulkan",
               "--threads", "4", "--max-new-tokens", "256", "--out", str(transcript)], wav=False)
        result = json.loads(transcript.read_text())
        segments = result["segments"]
        if not segments or not all(re.fullmatch(r"S\d+", s["speaker"]) and
                                   0 <= s["start"] <= s["end"] and s["text"].strip() for s in segments):
            raise RuntimeError("Missing or invalid speaker-labelled transcription")
        text = " ".join(s["text"] for s in segments).lower()
        if "country" not in text or "ask" not in text:
            raise RuntimeError("JFK transcript missing expected words 'ask' and 'country'")
        checks["transcribe"]["segments"] = len(segments)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--build", type=Path, default=Path("build-moss-e2e"))
    parser.add_argument("--models", type=Path, default=Path("models-moss-e2e"))
    parser.add_argument("--output", type=Path, default=Path("moss-e2e-results"))
    parser.add_argument("--asr-binary", type=Path, default=Path("build-moss-asr/bin/whisper-cli"))
    parser.add_argument("--tts-baseline-binary", type=Path,
                        default=Path("build-moss-baseline/engines/tts/moss-cli"))
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case == "tts-f16":
        (args.output / "reference-short.txt").write_text(TTS_TEXT + "\n")
        (args.output / "reference-long.txt").write_text(TTS_LONG_TEXT + "\n")
    result = {"case": args.case, "status": "failed",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    try:
        result["checks"] = run_case(args.case, args.build.resolve(), args.models, args.output,
                                   args.tts_baseline_binary.resolve())
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
        traceback.print_exc()
    finally:
        if args.case == "tts-f16":
            # Still score existing audio when a later synthesis fails, preserving
            # both the original failure and CPU/Vulkan quality evidence.
            try:
                result["intelligibility"] = score_tts_outputs(args.asr_binary.resolve(), args.models, args.output)
                existing = result["intelligibility"]["pre_existing_quality_issues"]
                if existing:
                    print("Non-blocking upstream quality failures (identical baseline audio): " +
                          ", ".join(existing), flush=True)
                if result["intelligibility"]["status"] != "passed":
                    result["status"] = "failed"
                    print("TTS intelligibility gate failed: " +
                          ", ".join(result["intelligibility"]["failed_outputs"]), file=sys.stderr)
            except Exception as error:
                result["intelligibility"] = {"status": "error", "reason": str(error)}
                result["status"] = "failed"
                traceback.print_exc()
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
