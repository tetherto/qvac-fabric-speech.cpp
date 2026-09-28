"""Convert NVIDIA's Nemotron 3 Diarization NeMo checkpoint with NeMo-Speech.cpp."""

import argparse
import subprocess
import sys
from pathlib import Path

from download_nemotron_diarization import (
    DEFAULT_MODEL_DIRECTORY,
    NEMO_FILENAME,
    download_artifact,
)


CONVERTER_REPOSITORY = "https://github.com/NVIDIA/NeMo-Speech.cpp.git"
CONVERTER_REVISION = "97a15afa5caa9bce5baaa86c1184103877af4101"
CONVERTER_DIRECTORY = DEFAULT_MODEL_DIRECTORY / "sources" / "NeMo-Speech.cpp"
OUTPUT_FILENAME = "Nemotron-3-Diarization.converted.q8_0.gguf"
OUTPUT_TYPE = "q8_0"


def ensure_converter(converter_directory, run=subprocess.run):
    converter = converter_directory / "convert_model.py"
    if converter.is_file():
        return converter
    converter_directory.parent.mkdir(parents=True, exist_ok=True)
    if not (converter_directory / ".git").is_dir():
        run(["git", "clone", "--filter=blob:none", "--no-checkout",
             CONVERTER_REPOSITORY, str(converter_directory)], check=True)
    run(["git", "-C", str(converter_directory), "fetch", "origin",
         CONVERTER_REVISION], check=True)
    run(["git", "-C", str(converter_directory), "checkout",
         "--detach", CONVERTER_REVISION], check=True)
    if not converter.is_file():
        raise FileNotFoundError(converter)
    return converter


def convert_model(source, output, converter, run=subprocess.run):
    output.parent.mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(converter), str(source.resolve()),
         "--outfile", str(output.resolve()), "--outtype", OUTPUT_TYPE],
        check=True, cwd=converter.parent)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--converter-dir", type=Path, default=CONVERTER_DIRECTORY)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.source is None:
        from huggingface_hub import hf_hub_download

        source = download_artifact(NEMO_FILENAME, args.model_dir, hf_hub_download)
    else:
        source = args.source
    output = args.output or args.model_dir / OUTPUT_FILENAME
    converter = ensure_converter(args.converter_dir)
    if args.prepare_only:
        print(source)
        print(converter.parent / "requirements.txt")
        return
    convert_model(source, output, converter)
    print(output)


if __name__ == "__main__":
    main()
