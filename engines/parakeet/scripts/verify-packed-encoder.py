#!/usr/bin/env python3
"""Verify every F16 GGUF encoder tensor and every affine LUT restoration."""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import gguf
import numpy as np

from packed_tdt import load_packed


BLOCK_SUFFIXES = {
    "norm_ff1": "norm_feed_forward1",
    "ff1": "feed_forward1",
    "norm_attn": "norm_self_att",
    "attn.q": "self_attn.linear_q",
    "attn.k": "self_attn.linear_k",
    "attn.v": "self_attn.linear_v",
    "attn.out": "self_attn.linear_out",
    "attn.pos": "self_attn.linear_pos",
    "attn.pos_bias_u": "self_attn.pos_bias_u",
    "attn.pos_bias_v": "self_attn.pos_bias_v",
    "norm_conv": "norm_conv",
    "conv.pw1": "conv.pointwise_conv1",
    "conv.dw": "conv.depthwise_conv",
    "conv.pw2": "conv.pointwise_conv2",
    "norm_ff2": "norm_feed_forward2",
    "ff2": "feed_forward2",
    "norm_out": "norm_out",
}

SUBSAMPLING = {
    "conv0": "conv.0", "conv1_dw": "conv.2", "conv1_pw": "conv.3",
    "conv2_dw": "conv.5", "conv2_pw": "conv.6", "out": "out",
}


def expected_for(name, state):
    def array(key):
        return state[key].numpy().astype(np.float32)

    if name.startswith("preproc."):
        key = ("preprocessor.featurizer.fb" if name.endswith("mel_filterbank")
               else "preprocessor.featurizer.window")
        value = array(key)
        return value[0] if name.endswith("mel_filterbank") else value
    if name.startswith("encoder.subsampling."):
        match = re.fullmatch(r"encoder\.subsampling\.([^.]*)\.(weight|bias)", name)
        part, kind = match.groups()
        value = array("encoder.pre_encode." + SUBSAMPLING[part] + "." + kind)
        if kind == "weight" and value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        return value.astype(np.float16 if kind == "weight" else np.float32)
    match = re.fullmatch(r"encoder\.blk\.(\d+)\.(.*)", name)
    layer, suffix = match.groups()
    prefix = f"encoder.layers.{layer}."
    if suffix.startswith("conv.bn."):
        bn = prefix + "conv.batch_norm."
        gamma, beta = array(bn + "weight"), array(bn + "bias")
        mean, var = array(bn + "running_mean"), array(bn + "running_var")
        scale = gamma / np.sqrt(var + 1e-5)
        return scale if suffix.endswith("scale") else beta - mean * scale
    if suffix == "attn.qkv.weight":
        return np.concatenate([array(prefix + f"self_attn.linear_{x}.weight")
                               for x in "qkv"], axis=0).astype(np.float16)
    for dest, source in sorted(BLOCK_SUFFIXES.items(), key=lambda x: -len(x[0])):
        if suffix == dest or suffix.startswith(dest + "."):
            rest = suffix[len(dest):]
            key = prefix + source + rest
            value = array(key)
            if suffix.endswith(".weight") and value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            return value.astype(np.float16 if suffix.endswith(".weight") and
                                (dest.startswith(("ff", "attn.", "conv."))) else np.float32)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", type=Path, default=Path("frozen_tdt_v1"))
    ap.add_argument("--gguf", type=Path, default=Path("frozen_tdt_v1/frozen-tdt-v1.f16.gguf"))
    args = ap.parse_args()
    _, state, _, _ = load_packed(args.packed_dir)
    reader = gguf.GGUFReader(str(args.gguf), "r")
    spec = importlib.util.spec_from_file_location("parakeet_ref", Path(__file__).with_name("ref-encoder-from-gguf.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    checked = 0
    failures = []
    for tensor in reader.tensors:
        if not tensor.name.startswith(("encoder.", "preproc.")):
            continue
        actual = ref.ggml_to_torch(tensor).numpy()
        expected = expected_for(tensor.name, state)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            failures.append(tensor.name)
        checked += 1
    lut_checked = 0
    lut_failures = []
    for name, record in state.records.items():
        if not name.startswith("encoder.") or record["kind"] != "affine":
            continue
        codes = record["codes"].numpy()
        if record["bits"] == 4:
            unpacked = np.empty(codes.size * 2, dtype=np.uint8)
            unpacked[0::2], unpacked[1::2] = codes & 15, codes >> 4
            codes = unpacked
        codes = codes.reshape(record["shape"])
        minimum, step = record["minimum"].numpy(), record["step"].numpy()
        lut = (minimum[:, None] + step[:, None] *
               np.arange(1 << record["bits"], dtype=np.float32)[None, :]).astype(np.float16)
        rows = np.arange(codes.shape[0]).reshape((-1,) + (1,) * (codes.ndim - 1))
        restored = lut[rows, codes]
        if not np.array_equal(restored, state[name].numpy()):
            lut_failures.append(name)
        lut_checked += 1
    result = {"gguf_encoder_tensors_checked": checked, "gguf_failures": failures,
              "lut_affine_tensors_checked": lut_checked, "lut_failures": lut_failures}
    print(json.dumps(result, indent=2))
    if failures or lut_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
