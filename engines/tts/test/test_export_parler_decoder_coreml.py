#!/usr/bin/env python3
"""Model-free unit tests for scripts/parler/export-decoder-coreml.py.

A tiny Parler decoder is built from random tensors in the GGUF's names and
torch layouts and checked against an independent plain-PyTorch decoder written
from the definitions (pre-norm layers, causal self-attention over the whole
sequence, cross-attention, GELU FFN, final LayerNorm, per-codebook heads). The
exported step, driven like the engine drives the package -- read-only cache
states, the current position scored separately, one row written by the host
after each call -- must reproduce the full-sequence decode at every position,
for per-layer and stacked state groupings. With coremltools, the converted
program must read every state and write none. Requires torch; skips without it.
"""
import importlib.util
import contextlib
import io
import json
import pathlib
import subprocess
import tempfile
import types
import unittest
from unittest import mock

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import coremltools  # noqa: F401
    HAVE_COREMLTOOLS = True
except ImportError:
    HAVE_COREMLTOOLS = False

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "parler" / "export-decoder-coreml.py"

LAYERS = 2
D = 32
HEADS = 4
FF = 48
N_Q = 3
VOCAB = 44
CTX = 16
CROSS = 8
EPS = 1e-5


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_parler_decoder_coreml", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hparams():
    return {"n_layer": LAYERS, "d_model": D, "n_head": HEADS, "d_ff": FF, "n_q": N_Q, "vocab": VOCAB,
            "ln_eps": EPS, "max_position": 64, "max_length": 24, "bos": VOCAB - 3 + 1, "variant": "tiny"}


def tiny_tensors(rng):
    def w(out, inp):
        return (rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)

    def norm(name, t):
        t[name + ".weight"] = (1 + 0.1 * rng.standard_normal(D)).astype(np.float32)
        t[name + ".bias"] = (0.1 * rng.standard_normal(D)).astype(np.float32)

    t = {}
    for l in range(LAYERS):
        p = f"dec.blk.{l}."
        for n in ("attn_q", "attn_k", "attn_v", "attn_o", "cross_q", "cross_k", "cross_v", "cross_o"):
            t[p + n + ".weight"] = w(D, D)
        t[p + "ffn_up.weight"] = w(FF, D)
        t[p + "ffn_down.weight"] = w(D, FF)
        for n in ("attn_norm", "cross_norm", "ffn_norm"):
            norm(p + n, t)
    norm("dec.output_norm", t)
    for k in range(N_Q):
        t[f"dec.lm_heads.{k}.weight"] = w(VOCAB, D)
    return t


def reference_decode(t, x, cross_states):
    """Full-sequence causal decode, [S, D] inputs -> [S, N_Q, VOCAB] logits."""
    x = torch.from_numpy(x)
    c = torch.from_numpy(cross_states)
    s, hd = x.shape[0], D // HEADS
    g = {k: torch.from_numpy(v) for k, v in t.items()}

    def ln(y, name):
        return F.layer_norm(y, (D,), g[name + ".weight"], g[name + ".bias"], EPS)

    def attn(q, k, v, causal):
        q, k, v = (z.reshape(z.shape[0], HEADS, hd).transpose(0, 1) for z in (q, k, v))
        scores = q @ k.transpose(-1, -2) / hd ** 0.5
        if causal:
            scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
        return (torch.softmax(scores, -1) @ v).transpose(0, 1).reshape(-1, D)

    for l in range(LAYERS):
        p = f"dec.blk.{l}."
        y = ln(x, p + "attn_norm")
        x = x + attn(y @ g[p + "attn_q.weight"].T, y @ g[p + "attn_k.weight"].T,
                     y @ g[p + "attn_v.weight"].T, True) @ g[p + "attn_o.weight"].T
        y = ln(x, p + "cross_norm")
        x = x + attn(y @ g[p + "cross_q.weight"].T, c @ g[p + "cross_k.weight"].T,
                     c @ g[p + "cross_v.weight"].T, False) @ g[p + "cross_o.weight"].T
        y = ln(x, p + "ffn_norm")
        x = x + F.gelu(y @ g[p + "ffn_up.weight"].T) @ g[p + "ffn_down.weight"].T
    x = ln(x, "dec.output_norm")
    return torch.stack([x @ g[f"dec.lm_heads.{k}.weight"].T for k in range(N_Q)], 1).numpy()


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class DecoderStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = load_exporter()
        rng = np.random.default_rng(7)
        cls.t = tiny_tensors(rng)
        cls.x = rng.standard_normal((CTX - 2, D)).astype(np.float32)
        cls.cross_states = rng.standard_normal((CROSS - 3, D)).astype(np.float32)
        cls.want = reference_decode(cls.t, cls.x, cls.cross_states)

    def stepped(self, layers_per_state, k_layout="rows"):
        ex = self.ex
        model = ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float32, layers_per_state, k_layout).eval()
        ks, vs, cmask = ex.cross_kv(self.t, hparams(), self.cross_states, CROSS)
        session = ex.TorchSession(model)
        session.set_cross(ks, vs)
        return np.stack([session.step(row, n, cmask) for n, row in enumerate(self.x)])

    def test_step_matches_full_sequence_per_layer_states(self):
        np.testing.assert_allclose(self.stepped(1), self.want, atol=2e-5, rtol=0)

    def test_step_matches_full_sequence_stacked_states(self):
        np.testing.assert_allclose(self.stepped(LAYERS), self.want, atol=2e-5, rtol=0)

    def test_step_matches_full_sequence_transposed_keys(self):
        np.testing.assert_allclose(self.stepped(LAYERS, "cols"), self.want, atol=2e-5, rtol=0)

    def test_new_kv_are_unscaled_projections(self):
        ex = self.ex
        model = ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float32).eval()
        x = torch.from_numpy(self.x[0].reshape(1, D, 1, 1))
        with torch.no_grad():
            _, nk, nv = model(x, torch.from_numpy(ex.self_mask(0, CTX)),
                              torch.from_numpy(ex.self_mask(CROSS, CROSS)))
        y = F.layer_norm(torch.from_numpy(self.x[0]), (D,), torch.from_numpy(self.t["dec.blk.0.attn_norm.weight"]),
                         torch.from_numpy(self.t["dec.blk.0.attn_norm.bias"]), EPS).numpy()
        hd = D // HEADS
        np.testing.assert_allclose(nk[0, :, 0].numpy(), (self.t["dec.blk.0.attn_k.weight"] @ y).reshape(HEADS, hd),
                                   atol=1e-5)
        np.testing.assert_allclose(nv[0, :, 0].numpy(), (self.t["dec.blk.0.attn_v.weight"] @ y).reshape(HEADS, hd),
                                   atol=1e-5)
        self.assertEqual(tuple(nk.shape), (LAYERS, HEADS, 1, hd))

    def test_grouping_must_divide_layers(self):
        with self.assertRaises(ValueError):
            self.ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float32, 3)

    def test_cross_kv_masks_and_refuses_overflow(self):
        ks, vs, mask = self.ex.cross_kv(self.t, hparams(), self.cross_states, CROSS)
        n = self.cross_states.shape[0]
        self.assertEqual(ks[0].shape, (1, HEADS, CROSS, D // HEADS))
        self.assertTrue((mask[..., :n] == 0).all() and (mask[..., n:] == self.ex.MASK_OFF).all())
        self.assertTrue((ks[1][:, :, n:] == 0).all() and (vs[1][:, :, n:] == 0).all())
        with self.assertRaises(ValueError):
            self.ex.cross_kv(self.t, hparams(), np.zeros((CROSS + 1, D), np.float32), CROSS)

    def test_sidecar_name_drops_quant_tag(self):
        self.assertEqual(self.ex.sidecar_stem("m/parler-mini-v1-q8_0.gguf"), "parler-mini-v1-decoder")
        self.assertEqual(self.ex.sidecar_stem("parler-indic-f32.gguf"), "parler-indic-decoder")
        self.assertEqual(self.ex.sidecar_stem("parler-mini-v1.Q8_0.gguf"), "parler-mini-v1-decoder")
        self.assertEqual(self.ex.sidecar_stem("parler-mini-v1-F16.gguf"), "parler-mini-v1-decoder")

    def test_parity_requires_at_least_one_matching_case(self):
        with tempfile.TemporaryDirectory() as ref:
            # An unrelated file must not make a missing reference set pass.
            np.save(pathlib.Path(ref) / "other_step_logits.npy", np.zeros((1, N_Q, VOCAB)))
            with self.assertRaisesRegex(ValueError, "no matching case"):
                self.ex.run_parity(None, {}, {}, ref)

    def test_parity_accepts_one_available_case_and_checks_tolerance(self):
        logits = np.zeros((1, N_Q, VOCAB), np.float32)
        model = types.SimpleNamespace(ctx=1, cross=1)
        with tempfile.TemporaryDirectory() as ref:
            np.save(pathlib.Path(ref) / "case1_step_logits.npy", logits)
            np.save(pathlib.Path(ref) / "case1_cross_states.npy", np.zeros((1, 1, D), np.float32))
            with mock.patch.object(self.ex, "teacher_inputs", return_value=([np.zeros(D)], 0, None)), \
                    mock.patch.object(self.ex, "cross_kv", return_value=([], [], None)), \
                    mock.patch.object(self.ex, "TorchSession") as session, contextlib.redirect_stdout(io.StringIO()):
                session.return_value.step.return_value = logits[0]
                self.ex.run_parity(model, {}, {}, ref)
                session.return_value.step.return_value = logits[0] + 1
                with self.assertRaisesRegex(SystemExit, "rebuild differs"):
                    self.ex.run_parity(model, {}, {}, ref)

    def test_cli_rejects_invalid_combinations_before_reading_weights(self):
        invalid = ((["--layers", "1", "--out", "tiny.mlpackage", "--parity-dir", "ref"], "parity requires"),
                   (["--quality-steps", "1"], "requires --parity-dir"),
                   (["--no-export"], "requires --parity-dir"),
                   (["--layers", "0"], "must be positive"),
                   (["--layers", "-1"], "must be positive"),
                   (["--layers", "1"], "name it with --out"),
                   (["--quality-steps", "-1"], "must be nonnegative"),
                   (["--no-export", "--parity-dir", "ref", "--quality-steps", "1"], "requires export"))
        for argv, message in invalid:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    self.ex.parse_args(["--gguf", "unused.gguf"] + argv)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
        args = self.ex.parse_args(["--gguf", "unused.gguf", "--parity-dir", "ref", "--no-export"])
        self.assertTrue(args.no_export)

    def test_layers_exceeding_model_fails_before_building_decoder(self):
        args = self.ex.parse_args(["--gguf", "unused.gguf", "--layers", "3", "--out", "tiny.mlpackage"])
        with mock.patch.object(self.ex, "parse_args", return_value=args), \
                mock.patch.object(self.ex, "read_gguf", return_value=({}, hparams())), \
                mock.patch.object(self.ex, "DecoderStep") as decoder:
            with self.assertRaisesRegex(SystemExit, "exceeds the decoder's 2 layers"):
                self.ex.main()
            decoder.assert_not_called()

    def test_compile_failure_includes_compiler_stderr(self):
        failure = subprocess.CalledProcessError(1, ["xcrun"], stderr="specific compiler diagnostic")
        with tempfile.TemporaryDirectory() as out, mock.patch.object(self.ex.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "specific compiler diagnostic") as error:
                self.ex.compile_model("tiny.mlpackage", out)
            self.assertIs(error.exception.__cause__, failure)

    def test_head_weight_guard_rejects_constexpr_and_unknown_producers(self):
        for op_type in ("constexpr_affine_dequantize", "constexpr_blockwise_shift_scale",
                        "constexpr_lut_to_dense", "add"):
            weight = types.SimpleNamespace(op=types.SimpleNamespace(op_type=op_type))
            with self.subTest(op_type=op_type), self.assertRaisesRegex(RuntimeError, "--heads-fp16"):
                self.ex.head_weight_constants(weight)
        constant = types.SimpleNamespace(op=types.SimpleNamespace(op_type="const", name="head_weight"))
        cast = types.SimpleNamespace(op=types.SimpleNamespace(op_type="cast", x=constant))
        self.assertEqual(self.ex.head_weight_constants(cast), {"head_weight"})

    def test_heads_fp16_fails_when_no_projection_matches(self):
        fn = mock.Mock()
        fn.find_ops.return_value = []
        ml = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": fn}))
        with self.assertRaisesRegex(RuntimeError, "no matching.*head projection"):
            self.ex.compress(ml, "fp16", True, N_Q * VOCAB)

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_compression_excludes_heads_and_checks_result(self):
        import coremltools.optimize.coreml as cto

        ex = self.ex
        const = types.SimpleNamespace(op=types.SimpleNamespace(op_type="const", name="fused_head_weight"),
                                      shape=(N_Q * VOCAB, D, 1, 1))
        head = types.SimpleNamespace(name="fused_heads", weight=const)
        fn = mock.Mock()
        fn.find_ops.return_value = [head]
        ml = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": fn}))
        for mode, compressor in (("int8", "linear_quantize_weights"), ("int4", "linear_quantize_weights"),
                                 ("pal6", "palettize_weights")):
            with self.subTest(mode=mode), mock.patch.object(cto, compressor, return_value=ml) as compress:
                self.assertIs(ex.compress(ml, mode, True, N_Q * VOCAB), ml)
                config = compress.call_args.args[1]
                self.assertEqual(config.op_name_configs, {"fused_head_weight": None})
        bad_weight = types.SimpleNamespace(shape=const.shape,
                                          op=types.SimpleNamespace(op_type="constexpr_blockwise_shift_scale"))
        bad_fn = mock.Mock()
        bad_fn.find_ops.return_value = [types.SimpleNamespace(name=head.name, weight=bad_weight)]
        bad_ml = types.SimpleNamespace(_mil_program=types.SimpleNamespace(functions={"main": bad_fn}))
        with mock.patch.object(cto, "linear_quantize_weights", return_value=bad_ml):
            with self.assertRaisesRegex(RuntimeError, "constexpr quantized"):
                ex.compress(ml, "int8", True, N_Q * VOCAB)

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_int8_heads_stay_constant_while_other_weights_are_quantized(self):
        ex = self.ex
        # Cross the compressor's default 2048-element threshold for non-head
        # weights as well, so successful exclusion cannot pass vacuously.
        t, hp = dict(self.t), hparams()
        hp["d_ff"] = FF * 2
        for layer in range(LAYERS):
            for name, axis in (("ffn_up", 0), ("ffn_down", 1)):
                key = f"dec.blk.{layer}.{name}.weight"
                t[key] = np.concatenate([t[key], t[key]], axis=axis)
        model = ex.DecoderStep(t, hp, CTX, CROSS, torch.float16, LAYERS).eval()
        ml = ex.convert_coreml(model, "int8", True, ex.contract(model, hp, "int8+fp16-heads", None))
        for head in ex.head_projections(ml, N_Q * VOCAB):
            self.assertTrue(ex.head_weight_constants(head.weight))
            self.assertEqual(head.weight.dtype.__name__, "fp16")
        ops = ml._mil_program.functions["main"].operations
        self.assertTrue(any(op.op_type.startswith("constexpr_") for op in ops))

    @unittest.skipUnless(HAVE_COREMLTOOLS, "coremltools not installed")
    def test_converted_program_reads_every_state_and_writes_none(self):
        ex = self.ex
        for per_state, k_layout in ((1, "rows"), (LAYERS, "cols")):
            model = ex.DecoderStep(self.t, hparams(), CTX, CROSS, torch.float16, per_state, k_layout).eval()
            meta = ex.contract(model, hparams(), "fp16", None)
            ml = ex.convert_coreml(model, "fp16", False, meta)   # raises if a state is dropped or written
            fn = ml._mil_program.functions["main"]
            self.assertEqual(len(fn.find_ops(op_type="read_state")), 4 * LAYERS // per_state)
            got = json.loads(ml.user_defined_metadata["parler.decoder.contract"])
            self.assertEqual(got["states"]["layers_per_state"], per_state)
            k_inner = [CTX, D // HEADS] if k_layout == "rows" else [D // HEADS, CTX]
            self.assertEqual(got["states"]["k_layout"], k_layout)
            self.assertEqual(got["states"]["shapes"]["self_k_00"], [per_state, HEADS] + k_inner)
            self.assertEqual(got["states"]["shapes"]["self_v_00"], [per_state, HEADS, CTX, D // HEADS])
            self.assertEqual(got["outputs"]["new_k"], [LAYERS, HEADS, 1, D // HEADS])


if __name__ == "__main__":
    unittest.main()
