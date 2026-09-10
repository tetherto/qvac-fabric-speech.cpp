#!/usr/bin/env python3
"""Exercise the CLI benchmark's TDT Core ML fail-fast/reporting contract."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "bench.json"
        completed = subprocess.run(
            [
                str(args.binary),
                "--model", str(args.model),
                "--wav", str(args.wav),
                "--bench",
                "--bench-warmup", "0",
                "--bench-runs", "1",
                "--bench-json", str(result_path),
                "--require-coreml",
                "--n-gpu-layers", "999",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        if completed.returncode != 0:
            raise AssertionError(
                f"CLI failed with {completed.returncode}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        payload = json.loads(result_path.read_text())
        assert payload["encoder_coreml_all_runs"] is True, payload
        assert payload["encoder_backend"].startswith("coreml-"), payload
        assert payload["backend"].startswith("ggml-"), payload

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
