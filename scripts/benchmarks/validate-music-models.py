#!/usr/bin/env python3
"""Check locally provisioned music GGUF stage coverage before invoking generation.

This is a presence/header check, not a claim that tensor payloads are valid.
Native model loading performs full format/architecture validation.
"""
import sys
from pathlib import Path


def validate(directory, family):
    root = Path(directory)
    if not root.is_dir():
        return False
    paths = list(root.glob('*.gguf'))
    if family == 'minimax':
        paths += list((root / 'mm3').glob('*.gguf'))
    names = []
    for p in paths:
        try:
            with p.open('rb') as f:
                if f.read(4) == b'GGUF':
                    names.append(p.name.lower())
        except OSError:
            continue
    if family == 'minimax':
        lm = {n[len('mm3-lm-'):-5] for n in names if n.startswith('mm3-lm-')}
        synth = {n[len('mm3-synth-'):-5] for n in names if n.startswith('mm3-synth-')}
        return bool(lm & synth)
    if family == 'acestep':
        stages = set()
        for n in names:
            for stage, tokens in [('text', ('embedding', 'text-enc', 'textenc')),
                                  ('vae', ('vae',)), ('lm', ('-lm', 'lm-', '_lm', 'ace-lm', '5hz-lm')),
                                  ('dit', ('turbo', 'dit', 'v15', 'sft'))]:
                if any(t in n for t in tokens):
                    stages.add(stage)
                    break
        return stages == {'text', 'vae', 'lm', 'dit'}
    return False


if __name__ == '__main__':
    raise SystemExit(0 if validate(sys.argv[1], sys.argv[2]) else 1)
