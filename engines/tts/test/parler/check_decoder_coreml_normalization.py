#!/usr/bin/env python3
"""Tiny normalization execution probes; no changes to the decoder exporter.

Compare channel-axis arithmetic, last-axis LayerNorm, fp32 arithmetic and a
conv-reduction spelling that avoids the reduce-mean LayerNorm fusion pattern.
Each model has just one output so intermediate-output taps cannot disable the
optimization being investigated. All inputs and weights are synthetic.
"""
import argparse
from collections import Counter
from pathlib import Path
import traceback

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from check_decoder_coreml_parity import load_exporter


def last_axis_norm(self, x):
    # Batch and spatial dimensions are all one: reshape preserves element order.
    y = F.layer_norm(x.reshape(1, 1, 1, self.channels), (self.channels,),
                     self.weight.reshape(-1), self.bias.reshape(-1), self.eps)
    return y.reshape(1, self.channels, 1, 1)


class Probe(nn.Module):
    def __init__(self, mode, weight, bias, projection, eps):
        super().__init__()
        self.mode, self.eps, self.channels = mode, eps, int(weight.size)
        self.register_buffer("weight", torch.from_numpy(weight.copy()).reshape(1, -1, 1, 1))
        self.register_buffer("bias", torch.from_numpy(bias.copy()).reshape(1, -1, 1, 1))
        self.register_buffer("projection", torch.from_numpy(projection.copy())[:, :, None, None])
        self.register_buffer("average", torch.full((1, weight.size, 1, 1), 1.0 / weight.size))

    def forward(self, x):
        if self.mode == "identity":
            return x + torch.zeros_like(x)
        if self.mode == "last_axis":
            return last_axis_norm(self, x)
        if self.mode == "conv_reductions":
            mean = F.conv2d(x, self.average)
            centred = x - mean
            variance = F.conv2d(centred * centred, self.average)
        else:
            mean = x.mean(1, keepdim=True)
            centred = x - mean
            variance = (centred * centred).mean(1, keepdim=True)
        if self.mode == "mean":
            return mean
        if self.mode == "centred":
            return centred
        if self.mode == "variance":
            return variance
        y = centred * torch.rsqrt(variance + self.eps) * self.weight + self.bias
        return F.conv2d(y, self.projection) if self.mode == "channel_projected" else y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--export-candidate", type=Path,
                        help="export a tiny last-axis decoder candidate instead of primitive probes")
    args = parser.parse_args()
    assert args.gguf.stat().st_size < 2_000_000, "use only the synthetic tiny GGUF"
    exporter = load_exporter()
    tensors, hp = exporter.read_gguf(args.gguf)
    assert hp["d_model"] == 32 and hp["n_layer"] == 2
    if args.export_candidate:
        args.export_candidate.mkdir(parents=True, exist_ok=True)
        # This module instance is private to this diagnostic process. Production
        # ChannelNorm and exporter defaults remain unchanged.
        exporter.ChannelNorm.forward = last_axis_norm
        decoder = exporter.DecoderStep(tensors, hp, 64, 8, torch.float16, hp["n_layer"]).eval()
        for module in decoder.modules():
            if isinstance(module, exporter.ChannelNorm):
                module.channels = hp["d_model"]
        meta = exporter.contract(decoder, hp, "fp16", args.gguf)
        model = exporter.convert_coreml(decoder, "fp16", False, meta)
        model.user_defined_metadata["parler.test.normalization"] = "last-axis-candidate"
        package = args.export_candidate / "parler-tiny-decoder.mlpackage"
        model.save(str(package))
        compiled = exporter.compile_model(package, args.export_candidate)
        exporter.report_placement(compiled)
        exporter.report_placement(compiled, "CPU_AND_NE")
        print(f"Exported diagnostic last-axis candidate {package}", flush=True)
        return 0
    if not args.out:
        parser.error("--out is required for primitive probes")
    args.out.mkdir(parents=True, exist_ok=True)
    weight, bias = tensors["dec.blk.0.attn_norm.weight"], tensors["dec.blk.0.attn_norm.bias"]
    projection = tensors["dec.blk.0.attn_k.weight"]
    rng = np.random.default_rng(813)
    inputs = [(f"random-{i}", rng.standard_normal((1, hp["d_model"], 1, 1)).astype(np.float16)) for i in range(3)]
    inputs += [("constant-zero", np.zeros((1, hp["d_model"], 1, 1), np.float16)),
               ("constant-nonzero", np.full((1, hp["d_model"], 1, 1), 3.5, np.float16)),
               ("wide", np.tile(np.array([-1024, 1024], np.float16), hp["d_model"] // 2).reshape(1, -1, 1, 1))]
    modes = ("identity", "mean", "centred", "variance", "channel", "channel_projected",
             "last_axis", "conv_reductions", "channel_fp32")
    passed = True
    for mode in modes:
        try:
            probe = Probe(mode, weight, bias, projection, hp["ln_eps"]).eval()
            traced = torch.jit.trace(probe, torch.zeros_like(torch.from_numpy(inputs[0][1])).float())
            precision = ct.precision.FLOAT32 if mode == "channel_fp32" else ct.precision.FLOAT16
            model = ct.convert(traced, inputs=[ct.TensorType(name="hidden", shape=inputs[0][1].shape, dtype=np.float16)],
                               outputs=[ct.TensorType(name="value", dtype=np.float16)],
                               minimum_deployment_target=ct.target.macOS15, compute_precision=precision,
                               compute_units=ct.ComputeUnit.CPU_ONLY, convert_to="mlprogram")
            package = args.out / (mode + ".mlpackage")
            model.save(str(package))
            print(f"normalization mode={mode} MIL=" + str(dict(Counter(op.op_type for op in model._mil_program.functions["main"].operations))), flush=True)
            for units in (ct.ComputeUnit.CPU_ONLY, ct.ComputeUnit.ALL):
                try:
                    runtime = ct.models.MLModel(str(package), compute_units=units)
                    previous = None
                    for label, value in inputs:
                        with torch.no_grad():
                            want = probe(torch.from_numpy(value.astype(np.float32))).numpy()
                        got = np.asarray(runtime.predict({"hidden": value})["value"], dtype=np.float32)
                        peak = float(np.max(np.abs(got - want))) if np.isfinite(got).all() else float("inf")
                        beta = bias.reshape(1, -1, 1, 1)
                        if mode == "channel_projected":
                            beta = (projection @ bias).reshape(1, -1, 1, 1)
                        beta_error = float(np.max(np.abs(got - beta))) if got.shape == beta.shape else float("nan")
                        delta = float(np.max(np.abs(got - previous))) if previous is not None else float("nan")
                        print(f"normalization mode={mode} units={units.name} input={label} peak={peak:.6f} "
                              f"beta_error={beta_error:.6f} change_from_previous={delta:.6f}", flush=True)
                        passed &= peak < 0.02
                        previous = got.copy()
                except Exception:
                    print(f"FAIL normalization mode={mode} units={units.name}", flush=True)
                    traceback.print_exc()
                    passed = False
        except Exception:
            print(f"FAIL normalization mode={mode} conversion", flush=True)
            traceback.print_exc()
            passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
