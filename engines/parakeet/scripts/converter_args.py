import argparse
from pathlib import Path
from typing import Sequence


QUANT_CHOICES = ["f32", "f16", "bf16", "q8_0", "q5_0", "q4_0"]
HEAD_CHOICES = ["auto", "ctc", "rnnt", "tdt", "eou", "sortformer"]
DEFAULT_MODEL_STEM = "parakeet-ctc-0.6b"


def parse_args(
    description: str | None = None,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(f"models/{DEFAULT_MODEL_STEM}.nemo"),
        help="Path to .nemo archive (tarball). Downloads from HF if missing.",
    )
    parser.add_argument(
        "--packed-dir", type=Path,
        help="Directory containing model.pt and architecture/ from a parakeet_affine_packed_v1 export.",
    )
    parser.add_argument(
        "--encoder-coreml-only", action="store_true",
        help="Write predictor/joint, preprocessor, tokenizer, and metadata only; requires a matching Core ML encoder sidecar at load.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output GGUF path. Defaults to "
            f"models/{DEFAULT_MODEL_STEM}.<quant>.gguf."
        ),
    )
    parser.add_argument(
        "--quant",
        choices=QUANT_CHOICES,
        default=None,
        help=(
            "Weight dtype for 2D projection matrices. Biases / norms / BN "
            "stay at f32. q8_0 is the .nemo default (~2x smaller than f16, bit-equal "
            "transcripts on clean speech across CTC/TDT/EOU/Sortformer); "
            "packed checkpoints default to f16. Pass --quant f16 for the floating-point baseline, or "
            "--quant bf16 for bfloat16 projection weights."
        ),
    )
    parser.add_argument(
        "--allow-requantize-packed", action="store_true",
        help="Acknowledge a second quantization of --packed-dir weights (for example q8_0 or q4_0).",
    )
    parser.add_argument(
        "--hf-repo",
        default="nvidia/parakeet-ctc-0.6b",
        help="HF model id to download from if --ckpt is missing.",
    )
    parser.add_argument(
        "--head",
        choices=HEAD_CHOICES,
        default="auto",
        help=(
            "Override the auto-detected head. Use 'rnnt' to export the "
            "Transducer branch of a hybrid RNNT+CTC checkpoint instead of "
            "its default CTC branch."
        ),
    )
    args = parser.parse_args(argv)
    if args.quant is None:
        args.quant = "f16" if args.packed_dir else "q8_0"
    if args.packed_dir and args.quant not in ("f16", "f32") and not args.allow_requantize_packed:
        parser.error(
            "--packed-dir restores the checkpoint's affine weights; "
            f"--quant {args.quant} would quantize them again. "
            "Pass --allow-requantize-packed to opt in."
        )
    if args.encoder_coreml_only and (not args.packed_dir or args.quant != "f16"):
        parser.error("--encoder-coreml-only currently requires --packed-dir and --quant f16")
    if args.out is None:
        if args.packed_dir:
            suffix = ".coreml-only" if args.encoder_coreml_only else ""
            args.out = args.packed_dir / f"{args.packed_dir.name}{suffix}.{args.quant}.gguf"
        else:
            args.out = Path(f"models/{DEFAULT_MODEL_STEM}.{args.quant}.gguf")
    return args
