#!/usr/bin/env python3
"""Full-checkpoint Vulkan smoke tests; model files come from the model registry.

Run from the repository root after building moss-cli and moss-transcribe.
Requires AWS CLI credentials and MODEL_S3_BUCKET. Outputs contain logs, WAVs,
transcripts and checkpoint hashes, never checkpoint weights. This checks native
inference, not addon integration or perceptual audio quality.
"""

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
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


def reference(source, target, offset):
    # Short real-speech excerpts, resampled to the TTS codec's required 24 kHz.
    with wave.open(str(source), "rb") as audio:
        if audio.getsampwidth() != 2 or audio.getnchannels() != 1:
            raise RuntimeError("Expected mono PCM16 reference")
        rate = audio.getframerate()
        audio.setpos(offset * rate)
        samples = array.array("h", audio.readframes(3 * rate))
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


def run_case(case, build, models, output):
    model = download(CASES[case], models, output)
    family = case.split("-")[0]
    sample = Path("engines/parakeet/test/samples/jfk.wav")
    tts = str(build / "engines/tts/moss-cli")
    common = ["--gpu", "--threads", "4", "--seed", "1234"]
    checks = {}

    def invoke(name, args, wav=True):
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

        gpu_snapshot("before")
        try:
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
        if not re.search(r"\[(?:moss-cli|moss-transcribe)\] backend: Vulkan\d*", log.read_text()):
            raise RuntimeError(f"{name} did not select Vulkan; see {log}")
        checks[name] = audio_stats(output / f"{name}.wav") if wav else {"backend": "Vulkan"}
        print(f"PASS {name}: {checks[name]}", flush=True)

    if family in {"tts", "ttsd"}:
        decoder = download("moss-codec-decoder-f16", models, output)
        encoder = download("moss-codec-encoder-f16", models, output)
        ref1, ref2 = output / "reference1.wav", output / "reference2.wav"
        reference(sample, ref1, 0)
        reference(sample, ref2, 4)
        base = [tts, "--backbone", model, "--decoder", decoder, "--language", "en",
                "--max-new-tokens", "192", "--context", "4096", *common]
        if family == "tts":
            base += ["--text", "Hello, this is a test of speech synthesis."]
            for name, extra in [("batch", []), ("stream", ["--stream", "--stream-chunk-frames", "25"]),
                                ("clone", ["--encoder", encoder, "--ref-audio", str(ref1)])]:
                invoke(name, [*base, *extra, "--out", str(output / f"{name}.wav")])
        else:
            # Two reference slots exercise dialogue conditioning; these are
            # excerpts from the same speaker, not a voice identity quality test.
            invoke("dialogue", [*base, "--encoder", encoder, "--dialogue-ref", str(ref1),
                   "--dialogue-ref", str(ref2), "--text", "[S1] Hello there. [S2] How are you?",
                   "--out", str(output / "dialogue.wav")])
    elif family == "sfx":
        invoke("sound", [tts, "--mode", "sfx", "--model", model, *common,
               "--text", "Rain falling on a tin roof.", "--seconds", "3",
               "--out", str(output / "sound.wav")])
    elif family == "speech":
        codec = download("moss-speech-codec-f16", models, output)
        base = [tts, "--mode", "s2s", "--model", model, "--codec", codec,
                "--audio", str(sample), "--max-new-tokens", "256", *common]
        invoke("reply", [*base, "--max-reply-seconds", "4", "--out", str(output / "reply.wav")])
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
    args = parser.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"case": args.case, "status": "failed",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    try:
        result["checks"] = run_case(args.case, args.build.resolve(), args.models, args.output)
        result["status"] = "passed"
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
