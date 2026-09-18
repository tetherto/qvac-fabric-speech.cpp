#!/usr/bin/env python3
"""Hash the generator binary and provisioned GGUF files outside generation timing."""
import hashlib
import json
from pathlib import Path
import sys


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


if __name__ == '__main__':
    root, binary, device = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    paths = sorted(set(root.glob('*.gguf')) | set((root/'mm3').glob('*.gguf')))
    print(json.dumps({'binary': identity(binary), 'model_files': [identity(p) for p in paths],
                      'requested_device': device, 'scope': 'all provisioned GGUFs in the generator model directory; native CLI selects stages by filename'}))
