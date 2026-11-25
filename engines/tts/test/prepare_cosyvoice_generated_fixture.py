#!/usr/bin/env python3
"""Write a seeded voice.gguf and parity-tokens.txt for the generated-sidecar gates.

test-cosyvoice-coreml --generated DIR CACHE runs the Engine with pinned speech
tokens (the LM is never loaded) and the stage parity gates on DIR's real flow
and HiFT GGUFs. Its other inputs need no checkpoint: a voice with 60 prompt
tokens, 120 prompt mel frames and a speaker embedding, and 200 speech tokens,
all drawn from a fixed seed, so the flow sees 520 mel frames (within the
front-end's 1024) and HiFT 400 (within its 512).
"""

import argparse
from pathlib import Path

import gguf
import numpy as np

PROMPT_TOKENS = 60
SPEECH_TOKENS = 200
SPEECH_VOCAB = 6561
MEL = 80
SPEAKER = 192


def write_fixture(directory, seed=20261123):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    prompt = rng.integers(0, SPEECH_VOCAB, PROMPT_TOKENS, dtype=np.int32)
    # Log-mel-like values: the range a 24 kHz CosyVoice3 prompt mel takes.
    feat = rng.normal(-6.0, 2.0, (2 * PROMPT_TOKENS, MEL)).astype(np.float32)
    embedding = rng.normal(0.0, 1.0, SPEAKER).astype(np.float32)
    writer = gguf.GGUFWriter(str(directory / "voice.gguf"), "cosyvoice3-voice")
    writer.add_string("voice.name", "seeded")
    writer.add_string("voice.prompt_text", "")
    writer.add_string("voice.language", "en")
    writer.add_uint32("voice.prompt_len", PROMPT_TOKENS)
    writer.add_uint32("voice.mel_len", 2 * PROMPT_TOKENS)
    writer.add_tensor("voice/prompt_stok", prompt)
    writer.add_tensor("voice/prompt_token", prompt)
    writer.add_tensor("voice/prompt_feat", feat)
    writer.add_tensor("voice/embedding", embedding)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    tokens = rng.integers(0, SPEECH_VOCAB, SPEECH_TOKENS)
    (directory / "parity-tokens.txt").write_text(" ".join(str(int(t)) for t in tokens) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    write_fixture(parser.parse_args().out)
