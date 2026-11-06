#!/usr/bin/env python3
"""Integration check for the full TranscribePsy EU gate and CAFE routes.

Requires a converted full-EU GGUF. A missing optional model skips in CTest.
"""

import argparse
import math
import re
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


def transcribe(binary: Path, model: Path, wav: Path, language: str):
    result = subprocess.run(
        [str(binary), "--model", str(model), "--wav", str(wav),
         "--language", language, "--backend", "cpu", "--threads", "4", "--verbose"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip(), result.stderr


def gate_decision(log: str):
    match = re.search(
        r"TranscribePsy EU gate: p\(non-English\)=([0-9.]+), "
        r"threshold=([0-9.]+), route=(en|other), frames=(\d+)/(\d+)", log)
    assert match, "gate decision missing from log"
    return float(match.group(1)), float(match.group(2)), match.group(3), \
        int(match.group(4)), int(match.group(5))


def write_wav(path: Path, pcm: bytes, sample_rate: int):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file():
        print(f"SKIP: model fixture missing: {args.model}")
        return 77

    samples = Path(__file__).parent / "samples"
    for filename, explicit, expected_route, words in (
        ("jfk.wav", "en", "en", ("fellow", "Americans", "country")),
        ("fr-multipiece-16k.wav", "other", "other", ("analyses", "politiques")),
    ):
        wav = samples / filename
        explicit_text, _ = transcribe(args.binary, args.model, wav, explicit)
        auto_text, log = transcribe(args.binary, args.model, wav, "auto")
        probability, threshold, route, _, _ = gate_decision(log)
        assert route == expected_route, (filename, log)
        assert (probability >= threshold) == (expected_route == "other")
        assert auto_text == explicit_text, (filename, auto_text, explicit_text)
        assert all(word.lower() in auto_text.lower() for word in words), auto_text

    with wave.open(str(samples / "jfk.wav"), "rb") as source:
        rate = source.getframerate()
        pcm = source.readframes(source.getnframes())
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        prefix = (pcm * 2)[:20 * rate * 2]
        loud_tail = b"".join(struct.pack("<h", int(28000 * math.sin(2 * math.pi * 440 * i / rate)))
                             for i in range(10 * rate))
        short_path = directory / "prefix.wav"
        long_path = directory / "long.wav"
        aligned_path = directory / "aligned.wav"
        silent_path = directory / "silent.wav"
        write_wav(short_path, prefix, rate)
        write_wav(long_path, prefix + loud_tail, rate)
        write_wav(aligned_path, pcm[:int(0.8 * rate) * 2], rate)
        write_wav(silent_path, bytes(int(0.8 * rate) * 2), rate)
        _, short_log = transcribe(args.binary, args.model, short_path, "auto")
        _, long_log = transcribe(args.binary, args.model, long_path, "auto")
        assert gate_decision(short_log)[:3] == gate_decision(long_log)[:3], \
            (short_log, long_log)
        _, aligned_log = transcribe(args.binary, args.model, aligned_path, "auto")
        assert gate_decision(aligned_log)[3:] == (10, 11), aligned_log
        _, silent_log = transcribe(args.binary, args.model, silent_path, "auto")
        assert gate_decision(silent_log)[3:] == (10, 11), silent_log
    print("TranscribePsy EU: English/French gate and explicit routes agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
