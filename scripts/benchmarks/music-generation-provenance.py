#!/usr/bin/env python3
"""Hash the generator binary and provisioned GGUF files outside generation timing."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def build_identity(binary):
    """Retain relevant build options without dumping arbitrary cache variables."""
    for directory in binary.resolve().parents:
        cache = directory / 'CMakeCache.txt'
        if not cache.is_file():
            continue
        options = {}
        for line in cache.read_text().splitlines():
            if line.startswith(('#', '//')) or '=' not in line or ':' not in line:
                continue
            key_type, value = line.split('=', 1)
            key, kind = key_type.split(':', 1)
            if key in {'CMAKE_BUILD_TYPE', 'CMAKE_C_COMPILER', 'CMAKE_CXX_COMPILER',
                       'CMAKE_C_COMPILER_ID', 'CMAKE_CXX_COMPILER_ID', 'CMAKE_GENERATOR'} or (
                    kind == 'BOOL' and key.startswith(('GGML_', 'AUDIOGEN_', 'SPEECH_'))):
                options[key] = value
        return {**identity(cache), 'options': options}
    return None


def source_identity():
    source = Path(__file__).resolve().parent.parent.parent
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=source,
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {'commit': commit, 'dirty': dirty,
                'scope': 'benchmark source checkout; binary identity is recorded separately'}
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(root, binary, device):
    paths = sorted(path for directory in (root, root / 'mm3') if directory.is_dir()
                   for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() == '.gguf')
    preparation_file = root / 'provenance.json'
    preparation = None
    if preparation_file.is_file():
        preparation = {**identity(preparation_file),
                       'metadata': json.loads(preparation_file.read_text())}
    return {'binary': identity(binary), 'model_files': [identity(path) for path in paths],
            'requested_device': device, 'preparation': preparation,
            'build_cache': build_identity(binary), 'source_checkout': source_identity(),
            'environment': {name: os.environ.get(name) for name in (
                'MM3_DIT_NO_FLASH', 'MM3_LM_FLASH', 'MM3_LM_NO_FLASH')},
            'scope': 'all provisioned GGUFs in the generator model directory; native CLI selects stages by filename'}


if __name__ == '__main__':
    print(json.dumps(provenance(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]),
                     allow_nan=False))
