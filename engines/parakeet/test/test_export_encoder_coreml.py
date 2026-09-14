#!/usr/bin/env python3
"""Model-free checks for the TDT/EOU Core ML exporter graph helpers."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is None:
    print("SKIP: torch is not installed")
    raise SystemExit(0)

if "coremltools" not in sys.modules:
    try:
        import coremltools  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["coremltools"] = types.ModuleType("coremltools")

SCRIPT = Path(__file__).parents[1] / "scripts" / "export-encoder-coreml.py"
SPEC = importlib.util.spec_from_file_location("export_encoder_coreml", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def eou_meta():
    return {
        "parakeet.model.type": "eou",
        "parakeet.encoder.causal_downsampling": True,
        "parakeet.encoder.conv_context_size": "causal",
        "parakeet.encoder.conv_norm_type": "layer_norm",
        "parakeet.encoder.att_context_style": "chunked_limited",
        "parakeet.encoder.att_context_size_left": 2,
        "parakeet.encoder.att_context_size_right": 1,
    }


class ExportContractTests(unittest.TestCase):
    def test_accepts_fixed_eou_and_rejects_flexible(self):
        self.assertEqual(EXPORTER.validate_export_contract(eou_meta()), "eou")
        with self.assertRaisesRegex(ValueError, "fixed shape"):
            EXPORTER.validate_export_contract(eou_meta(), flexible=True)

    def test_rejects_incompatible_eou_metadata(self):
        meta = eou_meta()
        meta["parakeet.encoder.conv_norm_type"] = "batch_norm"
        with self.assertRaisesRegex(ValueError, "conv_norm_type"):
            EXPORTER.validate_export_contract(meta)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_chunked_attention_visibility(self):
        mask = EXPORTER.chunked_attention_mask(6, 2, 1, torch.float32, "cpu")[0, 0]
        self.assertEqual(mask[0, 0].item(), 0.0)
        self.assertEqual(mask[0, 1].item(), 0.0)
        self.assertLess(mask[0, 2].item(), -1e20)
        self.assertEqual(mask[3, 2].item(), 0.0)
        self.assertEqual(mask[3, 3].item(), 0.0)
        self.assertLess(mask[3, 4].item(), -1e20)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_causal_subsampling_dimensions(self):
        weights = {
            "encoder.subsampling.conv0.weight": torch.zeros(2, 1, 3, 3),
            "encoder.subsampling.conv0.bias": torch.zeros(2),
            "encoder.subsampling.conv1_dw.weight": torch.zeros(2, 1, 3, 3),
            "encoder.subsampling.conv1_dw.bias": torch.zeros(2),
            "encoder.subsampling.conv1_pw.weight": torch.zeros(2, 2, 1, 1),
            "encoder.subsampling.conv1_pw.bias": torch.zeros(2),
            "encoder.subsampling.conv2_dw.weight": torch.zeros(2, 1, 3, 3),
            "encoder.subsampling.conv2_dw.bias": torch.zeros(2),
            "encoder.subsampling.conv2_pw.weight": torch.zeros(2, 2, 1, 1),
            "encoder.subsampling.conv2_pw.bias": torch.zeros(2),
            "encoder.subsampling.out.weight": torch.zeros(4, 4),
            "encoder.subsampling.out.bias": torch.zeros(4),
        }
        out = EXPORTER.causal_subsampling(torch.zeros(8, 9), weights)
        self.assertEqual(tuple(out.shape), (1, 2, 4))

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_causal_layer_norm_convolution_shape(self):
        class Ref:
            @staticmethod
            def layer_norm(x, weight, bias):
                return torch.nn.functional.layer_norm(
                    x, (x.size(-1),), weight=weight, bias=bias)

        weights = {
            "conv.pw1.weight": torch.zeros(4, 2, 1),
            "conv.pw1.bias": torch.zeros(4),
            "conv.dw.weight": torch.zeros(2, 1, 3),
            "conv.dw.bias": torch.zeros(2),
            "conv.norm.weight": torch.ones(2),
            "conv.norm.bias": torch.zeros(2),
            "conv.pw2.weight": torch.zeros(2, 2, 1),
            "conv.pw2.bias": torch.zeros(2),
        }
        out = EXPORTER.conformer_conv(
            Ref(), torch.zeros(1, 5, 2), weights, "conv",
            causal=True, layer_norm=True)
        self.assertEqual(tuple(out.shape), (1, 5, 2))


if __name__ == "__main__":
    unittest.main()
