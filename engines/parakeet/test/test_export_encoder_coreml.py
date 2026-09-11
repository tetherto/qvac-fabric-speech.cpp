#!/usr/bin/env python3
"""Model-free checks for the TDT/EOU/Sortformer Core ML exporter helpers."""

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


def sortformer_v2_1_meta():
    return {
        "parakeet.model.type": "sortformer",
        "parakeet.model_variant": "sortformer-streaming-v2.1-aosc",
        "parakeet.encoder.d_model": 4,
        "parakeet.encoder.n_layers": 0,
        "parakeet.encoder.n_heads": 1,
        "parakeet.encoder.pos_emb_max_len": 32,
        "parakeet.encoder.xscaling": True,
        "parakeet.encoder.causal_downsampling": False,
        "parakeet.encoder.conv_context_size": "default",
        "parakeet.encoder.att_context_style": "regular",
        "parakeet.encoder.att_context_size_left": -1,
        "parakeet.encoder.att_context_size_right": -1,
    }


class ExportContractTests(unittest.TestCase):
    def test_accepts_fixed_eou_and_rejects_flexible(self):
        self.assertEqual(EXPORTER.validate_export_contract(eou_meta()), "eou")
        with self.assertRaisesRegex(ValueError, "fixed shape"):
            EXPORTER.validate_export_contract(eou_meta(), flexible=True)

    def test_accepts_sortformer_v2_1_full_and_bypass_exports(self):
        meta = sortformer_v2_1_meta()
        self.assertEqual(EXPORTER.validate_export_contract(meta), "sortformer")
        self.assertEqual(
            EXPORTER.validate_export_contract(meta, bypass_pre_encode=True),
            "sortformer")

    def test_rejects_flexible_sortformer_export(self):
        with self.assertRaisesRegex(ValueError, "fixed shape"):
            EXPORTER.validate_export_contract(sortformer_v2_1_meta(), flexible=True)

    def test_rejects_other_sortformer_variants(self):
        for variant in ("", "sortformer-v1", "sortformer-streaming-v2"):
            with self.subTest(variant=variant):
                meta = sortformer_v2_1_meta()
                meta["parakeet.model_variant"] = variant
                with self.assertRaisesRegex(ValueError, "sortformer-streaming-v2.1-aosc"):
                    EXPORTER.validate_export_contract(meta)

    def test_rejects_non_sortformer_bypass_export(self):
        with self.assertRaisesRegex(ValueError, "only for Sortformer"):
            EXPORTER.validate_export_contract(eou_meta(), bypass_pre_encode=True)

    def test_rejects_incompatible_sortformer_encoder_metadata(self):
        invalid_cases = (
            ("parakeet.encoder.causal_downsampling", True, "non-causal downsampling"),
            ("parakeet.encoder.conv_context_size", "causal", "non-causal convolution"),
            ("parakeet.encoder.att_context_style", "chunked_limited", "full-context attention"),
            ("parakeet.encoder.att_context_size_right", 1, "unbounded attention context"),
        )
        for key, value, message in invalid_cases:
            with self.subTest(key=key, value=value):
                meta = sortformer_v2_1_meta()
                meta[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    EXPORTER.validate_export_contract(meta)

    def test_rejects_incompatible_eou_metadata(self):
        meta = eou_meta()
        meta["parakeet.encoder.conv_norm_type"] = "batch_norm"
        with self.assertRaisesRegex(ValueError, "conv_norm_type"):
            EXPORTER.validate_export_contract(meta)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_bypass_module_preserves_time_and_feature_orientation(self):
        module = EXPORTER.BypassEncoderModule(
            ref=types.SimpleNamespace(), weights={}, meta=sortformer_v2_1_meta()).eval()
        pre_encode = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        output = module(pre_encode, torch.ones(3, dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 4))
        torch.testing.assert_close(output, pre_encode.transpose(0, 1) * 2.0)

        padded = module(
            pre_encode, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32))
        torch.testing.assert_close(padded[:2], output[:2])
        torch.testing.assert_close(padded[2], torch.zeros(4))

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
