#!/usr/bin/env python3
"""Model-free checks for bucket padding, CFG, rotary attention and tracing."""

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    print("SKIP: numpy and torch required")
    raise SystemExit(77)
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "vector_export", SCRIPTS / "export-supertonic-vector-coreml.py"
)
ve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ve)


def random_tensors(heads):
    rng = np.random.default_rng(3)
    t = {}

    def value(name, shape, scale=0.02):
        t[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def linear(p, ic, oc, matmul=True):
        value(p + ".weight", (ic, oc) if matmul else (oc, ic, 1))
        value(p + ".bias", (oc,))

    def norm(p):
        t[p + ".weight"] = np.ones(512, np.float32)
        value(p + ".bias", (512,))

    def conv(p):
        value(p + ".dwconv.weight", (512, 1, 5), 0.1)
        value(p + ".dwconv.bias", (512,))
        norm(p + ".norm.norm")
        linear(p + ".pwconv1", 512, 64, False)
        linear(p + ".pwconv2", 64, 512, False)
        value(p + ".gamma", (512,), 0.2)

    pre = ve.PREFIX
    value(pre + "proj_in.net.weight", (512, 144, 1))
    value(pre + "proj_out.net.weight", (144, 512, 1))
    linear(pre + "time_encoder.mlp.0.linear", 64, 256, False)
    t[pre + "time_encoder.mlp.0.linear.weight"] = t[
        pre + "time_encoder.mlp.0.linear.weight"
    ][..., 0]
    linear(pre + "time_encoder.mlp.2.linear", 256, 64, False)
    t[pre + "time_encoder.mlp.2.linear.weight"] = t[
        pre + "time_encoder.mlp.2.linear.weight"
    ][..., 0]
    for g in range(4):
        base = pre + "main_blocks."
        i = g * 6
        for j in range(4):
            conv(base + f"{i}.convnext.{j}")
        linear(base + f"{i + 1}.linear.linear", 64, 512)
        conv(base + f"{i + 2}.convnext.0")
        conv(base + f"{i + 4}.convnext.0")
        p = base + f"{i + 3}.attn."
        for name, ic, oc in [
            ("W_query", 512, heads * 64),
            ("W_key", 256, heads * 64),
            ("W_value", 256, heads * 64),
            ("out_fc", heads * 64, 512),
        ]:
            linear(p + name + ".linear", ic, oc)
        norm(base + f"{i + 3}.norm.norm")
        p = base + f"{i + 5}.attention."
        for name, ic, oc in [
            ("W_query", 512, 256),
            ("W_key", 256, 256),
            ("W_value", 256, 256),
            ("out_fc", 256, 512),
        ]:
            linear(p + name + ".linear", ic, oc)
        norm(base + f"{i + 5}.norm.norm")
    for i in range(4):
        conv(pre + f"last_convnext.convnext.{i}")
    value("vector_estimator:/Expand_output_0", (1, 50, 256), 0.5)
    p = "vector_estimator:tts.ttl.uncond_masker."
    value(p + "text_special_token", (1, 256, 1), 0.5)
    for name in ("style_key_special_token", "style_value_special_token"):
        value(p + name, (1, 50, 256), 0.5)
    return t


class VectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        rng = np.random.default_rng(5)
        cls.latent = rng.standard_normal((1, 144, 9)).astype(np.float32)
        cls.text = rng.standard_normal((1, 256, 7)).astype(np.float32)
        cls.style = rng.standard_normal((1, 50, 256)).astype(np.float32)
        cls.theta = np.linspace(1, 1000, 32, dtype=np.float32)

    def test_padded_buckets_equal_unpadded_for_v2_and_v3(self):
        for heads, cs, us in ((4, 1, 0), (8, 4, 3)):
            t = random_tensors(heads)
            model = ve.VectorEstimator(t, heads, cs, us).eval()
            for length, text_length in ((1, 1), (2, 3), (9, 7)):
                with self.subTest(heads=heads, length=length), torch.no_grad():
                    latent = self.latent[..., :length]
                    text = self.text[..., :text_length]
                    exact = ve.inputs_for(
                        t,
                        self.theta,
                        latent,
                        text,
                        self.style,
                        2,
                        5,
                        length,
                        text_length,
                    )
                    padded = ve.inputs_for(
                        t, self.theta, latent, text, self.style, 2, 5, 32, 16
                    )
                    want = model(*exact)
                    got = model(*padded)[..., :length]
                    torch.testing.assert_close(got, want, atol=4e-6, rtol=3e-5)

    def test_cfg_combines_independent_conditional_and_unconditional_fields(self):
        t = random_tensors(8)
        null = "vector_estimator:tts.ttl.uncond_masker."
        uncond = dict(t)
        uncond["vector_estimator:/Expand_output_0"] = t[
            null + "style_key_special_token"
        ]
        cond = ve.VectorEstimator(t, 8, 1, 0).eval()
        uncond_model = ve.VectorEstimator(uncond, 8, 1, 0).eval()
        guided = ve.VectorEstimator(t, 8, 4, 3).eval()
        inp = ve.inputs_for(
            t, self.theta, self.latent, self.text, self.style, 1, 5, 16, 16
        )
        null_text = np.repeat(
            t[null + "text_special_token"].reshape(1, 256, 1),
            self.text.shape[-1],
            axis=-1,
        )
        uncond_inp = ve.inputs_for(
            t,
            self.theta,
            self.latent,
            null_text,
            t[null + "style_value_special_token"],
            1,
            5,
            16,
            16,
        )
        with torch.no_grad():
            want = 4 * cond(*inp) - 3 * uncond_model(*uncond_inp)
            torch.testing.assert_close(guided(*inp), want, atol=4e-6, rtol=3e-5)

    def test_trace_preserves_schedule_style_and_runtime_lengths(self):
        t = random_tensors(4)
        model = ve.VectorEstimator(t, 4, 1, 0).eval()
        field = ve.VelocityField(model).eval()
        inp = ve.inputs_for(
            t, self.theta, self.latent, self.text, self.style, 0, 5, 16, 16
        )
        with torch.no_grad():
            traced = torch.jit.trace(field, inp[:-1])
            changed = ve.inputs_for(
                t,
                self.theta,
                self.latent[..., :2],
                self.text[..., :3],
                self.style * 0.5,
                3,
                8,
                16,
                16,
            )
            torch.testing.assert_close(traced(*changed[:-1]), field(*changed[:-1]))
            self.assertGreater(
                (traced(*changed[:-1])[..., :2] - traced(*inp[:-1])[..., :2])
                .abs()
                .max()
                .item(),
                0.01,
            )

    def test_exported_velocity_and_host_integration_match_step(self):
        for heads, cs, us in ((4, 1, 0), (8, 4, 3)):
            t = random_tensors(heads)
            model = ve.VectorEstimator(t, heads, cs, us).eval()
            inp = ve.inputs_for(
                t, self.theta, self.latent, self.text, self.style, 2, 5, 16, 16
            )
            with torch.no_grad():
                velocity = ve.VelocityField(model)(*inp[:-1])
                if us:
                    velocity = cs * velocity[:1] - us * velocity[1:]
                host = inp[0] + velocity * inp[3] * inp[-1]
                torch.testing.assert_close(host, model(*inp))

    def test_rotary_attention_matches_numpy(self):
        t = random_tensors(4)
        p = ve.PREFIX + "main_blocks.3.attn"
        layer = ve.TextAttention(t, p, 4).eval()
        rng = np.random.default_rng(71)
        x = rng.standard_normal((1, 3, 512)).astype(np.float32)
        text = rng.standard_normal((1, 5, 256)).astype(np.float32)

        def project(x, name):
            return (
                x @ t[p + "." + name + ".linear.weight"]
                + t[p + "." + name + ".linear.bias"]
            )

        def rotate(y, length):
            y = y.reshape(1, length, 4, 64).transpose(0, 2, 1, 3)
            angles = (
                np.arange(length, dtype=np.float32)[:, None]
                / length
                * self.theta[None, :]
            )
            co, si = np.cos(angles)[None, None], np.sin(angles)[None, None]
            a, b = y[..., :32], y[..., 32:]
            return np.concatenate((a * co - b * si, b * co + a * si), -1), co, si

        q, cq, sq = rotate(project(x, "W_query"), 3)
        k, ck, sk = rotate(project(text, "W_key"), 5)
        v = project(text, "W_value").reshape(1, 5, 4, 64).transpose(0, 2, 1, 3)
        scores = q @ k.transpose(0, 1, 3, 2) / 16
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        y = (probs @ v).transpose(0, 2, 1, 3).reshape(1, 3, 256)
        want = project(y, "out_fc")
        with torch.no_grad():
            got = layer(
                torch.from_numpy(x),
                torch.from_numpy(text),
                torch.zeros(1, 1, 1, 5),
                *[torch.from_numpy(a) for a in (cq, sq, ck, sk)],
            ).numpy()
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
