#!/usr/bin/env python3
"""Derived benchmark inputs that need no third-party package: resample a mono PCM16 WAV to a codec's rate."""
from __future__ import annotations

import argparse
import array
import sys
import wave

PCM16_BYTES = 2
MONO = 1


def read_pcm16_mono(path):
    with wave.open(path, "rb") as audio:
        if audio.getsampwidth() != PCM16_BYTES or audio.getnchannels() != MONO:
            raise RuntimeError(f"{path}: expected mono PCM16")
        samples = array.array("h", audio.readframes(audio.getnframes()))
        rate = audio.getframerate()
    if sys.byteorder != "little":
        samples.byteswap()
    return samples, rate


def interpolate(samples, position):
    left = int(position)
    right = min(left + 1, len(samples) - 1)
    fraction = position - left
    return int(round(samples[left] * (1 - fraction) + samples[right] * fraction))


def resample_samples(samples, rate, target_rate):
    count = len(samples) * target_rate // rate
    return array.array("h", (interpolate(samples, index * rate / target_rate) for index in range(count)))


def write_pcm16_mono(path, samples, rate):
    if sys.byteorder != "little":
        samples = array.array("h", samples)
        samples.byteswap()
    with wave.open(path, "wb") as audio:
        audio.setnchannels(MONO)
        audio.setsampwidth(PCM16_BYTES)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())


def resample_file(source, out, target_rate):
    samples, rate = read_pcm16_mono(source)
    write_pcm16_mono(out, resample_samples(samples, rate, target_rate), target_rate)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resample = commands.add_parser("resample")
    resample.add_argument("--source", required=True)
    resample.add_argument("--out", required=True)
    resample.add_argument("--rate", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    resample_file(args.source, args.out, args.rate)


if __name__ == "__main__":
    main()
