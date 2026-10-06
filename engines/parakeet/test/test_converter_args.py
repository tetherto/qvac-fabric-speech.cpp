import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from converter_args import QUANT_CHOICES, parse_args  # noqa: E402


class ConverterArgsTest(unittest.TestCase):
    def test_default_output_uses_default_quantization(self):
        args = parse_args(argv=[])

        self.assertEqual(
            args.out,
            Path("models/parakeet-ctc-0.6b.q8_0.gguf"),
        )

    def test_default_output_tracks_explicit_quantization(self):
        args = parse_args(argv=["--quant", "f16"])

        self.assertEqual(
            args.out,
            Path("models/parakeet-ctc-0.6b.f16.gguf"),
        )

    def test_default_output_tracks_every_quant_choice(self):
        for quant in QUANT_CHOICES:
            with self.subTest(quant=quant):
                args = parse_args(argv=["--quant", quant])

                self.assertEqual(
                    args.out,
                    Path(f"models/parakeet-ctc-0.6b.{quant}.gguf"),
                )

    def test_quant_choices_cover_documented_tiers(self):
        self.assertEqual(
            QUANT_CHOICES,
            ["f32", "f16", "bf16", "q8_0", "q5_0", "q4_0"],
        )

    def test_unknown_quantization_is_rejected(self):
        with self.assertRaises(SystemExit) as ctx:
            parse_args(argv=["--quant", "q2_k"])

        self.assertNotEqual(ctx.exception.code, 0)

    def test_explicit_output_is_preserved(self):
        output = Path("models/custom-name.gguf")
        args = parse_args(
            argv=["--quant", "q5_0", "--out", str(output)],
        )

        self.assertEqual(args.out, output)

    def test_default_checkpoint_and_repo_agree(self):
        args = parse_args(argv=[])

        self.assertEqual(args.ckpt, Path("models/parakeet-ctc-0.6b.nemo"))
        self.assertEqual(args.hf_repo, "nvidia/parakeet-ctc-0.6b")

    def test_head_defaults_to_auto(self):
        self.assertEqual(parse_args(argv=[]).head, "auto")

    def test_rnnt_head_override(self):
        self.assertEqual(parse_args(argv=["--head", "rnnt"]).head, "rnnt")

    def test_packed_defaults_preserve_fp16(self):
        args = parse_args(argv=["--packed-dir", "packed"])
        self.assertEqual(args.quant, "f16")
        self.assertEqual(args.out, Path("packed/packed.f16.gguf"))

    def test_packed_requantization_requires_opt_in(self):
        for quant in ("q4_0", "q8_0", "bf16"):
            with self.subTest(quant=quant):
                with self.assertRaises(SystemExit):
                    parse_args(argv=["--packed-dir", "packed", "--quant", quant])
                args = parse_args(argv=["--packed-dir", "packed", "--quant", quant,
                                        "--allow-requantize-packed"])
                self.assertEqual(args.quant, quant)

    def test_encoder_free_requires_packed_fp16(self):
        with self.assertRaises(SystemExit):
            parse_args(argv=["--encoder-coreml-only"])
        with self.assertRaises(SystemExit):
            parse_args(argv=["--packed-dir", "packed", "--quant", "f32",
                             "--encoder-coreml-only"])
        args = parse_args(argv=["--packed-dir", "packed", "--encoder-coreml-only"])
        self.assertEqual(args.out, Path("packed/packed.coreml-only.f16.gguf"))


if __name__ == "__main__":
    unittest.main()
