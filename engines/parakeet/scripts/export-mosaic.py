#!/usr/bin/env python3
"""Default Parakeet-Mosaic exporter: the frozen 503.7 MB V2 recipe."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('export-mosaic-compact.py')), run_name='__main__')
