#!/usr/bin/env python3
"""Regression checks for MiniMax block attention, cache updates, and export contracts."""

import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from minimax_coreml_models import TransformerBlock, DitBlock, rotary
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

HIDDEN = 16
HEADS = 4
DIM = 4
KV_HEADS = 2
CACHE = 8
EPS = 1e-6


def exporter():
    spec = importlib.util.spec_from_file_location("export_minimax_coreml", SCRIPTS / "export-minimax-coreml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights(stage):
    generator = np.random.default_rng(42)
    sizes = {"attn_norm.weight": (HIDDEN,), "attn_output.weight": (HIDDEN, HIDDEN),
             "ffn_norm.weight": (HIDDEN,)}
    if stage == "dit":
        sizes.update({"attn_norm.bias": (HIDDEN,), "ffn_norm.bias": (HIDDEN,),
                      "attn_qkv.weight": (HIDDEN * 3, HIDDEN), "ffn_in.weight": (HIDDEN * 4, HIDDEN),
                      "ffn_in.bias": (HIDDEN * 4,), "ffn_out.weight": (HIDDEN, HIDDEN * 2), "ffn_out.bias": (HIDDEN,)})
    else:
        kv = KV_HEADS * DIM if stage == "lm" else HIDDEN
        sizes.update({"attn_q.weight": (HIDDEN, HIDDEN), "attn_k.weight": (kv, HIDDEN), "attn_v.weight": (kv, HIDDEN),
                      "ffn_gate.weight": (HIDDEN * 2, HIDDEN), "ffn_up.weight": (HIDDEN * 2, HIDDEN),
                      "ffn_down.weight": (HIDDEN, HIDDEN * 2)})
        if stage == "lm":
            sizes.update({"attn_q_norm.weight": (DIM,), "attn_k_norm.weight": (DIM,)})
    return {name: generator.normal(0, 0.1, size).astype(np.float32) for name, size in sizes.items()}


def make_model(stage):
    config = dict(hidden=HIDDEN, heads=HEADS, kv_heads=KV_HEADS if stage == "lm" else HEADS,
                  head_dim=DIM, eps=EPS, rope_dim=0 if stage == "depth" else DIM, rope_base=10000.0)
    return (DitBlock if stage == "dit" else TransformerBlock)(weights(stage), config, stage).eval()


def mask(positions):
    return torch.where(torch.arange(CACHE).reshape(1, -1) <= positions.reshape(-1, 1), 0.0, -float("inf")).reshape(1, 1, -1, CACHE)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class MiniMaxCoreMLTest(unittest.TestCase):
    def test_rotary_uses_neox_pairs_and_preserves_unrotated_tail(self):
        value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        actual = rotary(value, torch.tensor([3.0]), DIM, 10000.0).numpy().reshape(-1)
        angle = 3 * np.power(10000.0, -np.arange(0, DIM, 2) / DIM)
        expected = np.r_[np.arange(2) * np.cos(angle) - np.arange(2, 4) * np.sin(angle),
                         np.arange(2) * np.sin(angle) + np.arange(2, 4) * np.cos(angle), np.arange(4, 8)]
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def check_cached_decode(self, stage):
        torch.manual_seed(7)
        model = make_model(stage)
        hidden = torch.randn(2, 4, HIDDEN)
        cache_shape = (2, model.config["kv_heads"], CACHE, DIM)
        keys, values = torch.zeros(cache_shape), torch.zeros(cache_shape)
        positions = torch.arange(4, dtype=torch.float32)
        full = model(hidden, positions, keys, values, mask(positions))[0]
        output = []
        for position in range(4):
            pos = torch.tensor([float(position)])
            step, new_k, new_v = model(hidden[:, position:position + 1], pos, keys, values, mask(pos))
            output.append(step)
            keys[:, :, position:position + 1] = new_k.permute(0, 2, 1, 3)
            values[:, :, position:position + 1] = new_v.permute(0, 2, 1, 3)
            if stage == "lm":
                keys, values = keys.half().float(), values.half().float()
        torch.testing.assert_close(torch.cat(output, 1), full, atol=1e-6, rtol=1e-6)

    def test_lm_cached_decode_matches_causal_prefill(self):
        self.check_cached_decode("lm")

    def test_depth_cached_decode_matches_causal_prefill(self):
        self.check_cached_decode("depth")

    def test_mask_excludes_previous_utterance_future_cache(self):
        model = make_model("lm")
        hidden = torch.ones(2, 1, HIDDEN)
        shape = (2, KV_HEADS, CACHE, DIM)
        position = torch.tensor([0.0])
        zero = model(hidden, position, torch.zeros(shape), torch.zeros(shape), mask(position))[0]
        dirty = model(hidden, position, torch.ones(shape) * 100, torch.ones(shape) * 200, mask(position))[0]
        torch.testing.assert_close(zero, dirty, atol=0, rtol=0)

    def test_cfg_branches_remain_independent(self):
        torch.manual_seed(9)
        model = make_model("dit")
        hidden = torch.randn(2, 3, HIDDEN)
        positions = torch.arange(3, dtype=torch.float32)
        together = model(hidden, positions)
        separate = torch.cat([model(row.unsqueeze(0), positions) for row in hidden], 0)
        torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-6)

    def test_dit_ffn_uses_value_then_gate_order(self):
        model = make_model("dit")
        hidden = torch.ones(2, 1, HIDDEN)
        norm = model.norm(hidden, "ffn_norm")
        value, gate = model.linear(norm, "ffn_in").chunk(2, -1)
        expected = hidden + model.linear(value * F.silu(gate), "ffn_out")
        actual = model.finish(hidden, torch.zeros_like(hidden))
        torch.testing.assert_close(actual, expected)

    def test_dit_pointwise_projection_matches_dense_math(self):
        torch.manual_seed(5)
        model = make_model("dit")
        hidden = torch.randn(2, 7, HIDDEN)
        expected = F.linear(hidden, model.weight("ffn_in.weight"), model.weight("ffn_in.bias"))
        torch.testing.assert_close(model.linear(hidden, "ffn_in"), expected, atol=1e-6, rtol=1e-6)

    def test_precision_preserves_attention_and_residual_projections(self):
        from types import SimpleNamespace
        select = exporter().use_half_precision
        def operation(kind, name):
            return SimpleNamespace(op_type=kind, inputs={"weight": SimpleNamespace(name=name)})
        self.assertFalse(select(operation("linear", "weights_attn_q__weight")))
        self.assertFalse(select(operation("linear", "weights_ffn_down__weight")))
        self.assertFalse(select(operation("linear", "weights_attn_qkv__weight")))
        self.assertFalse(select(operation("linear", "weights_ffn_out__weight")))
        self.assertTrue(select(operation("linear", "weights_ffn_in__weight")))
        self.assertTrue(select(operation("linear", "weights_ffn_gate__weight")))
        self.assertTrue(select(operation("linear", "weights_ffn_up__weight")))
        self.assertFalse(select(operation("reduce_mean", "unused")))
        self.assertTrue(select(operation("conv", "weights_ffn_in__weight")))

    def test_export_ranges_match_model_metadata(self):
        module = exporter()
        self.assertEqual(module.stage_limits({"qwen3.context_length": 10240}, "lm"), (10240, 10240))
        self.assertEqual(module.stage_limits({"mm3.depth.max_position": 16}, "depth"), (16, 16))


if __name__ == "__main__":
    unittest.main()
