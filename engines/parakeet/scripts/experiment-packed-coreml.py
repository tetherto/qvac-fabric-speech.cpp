#!/usr/bin/env python3
"""Bounded Core ML preservation probe using slices of the frozen TDT encoder.

Writes small Core ML packages and exact expected FP16 weight files. No k-means
or fresh quantization is performed. Run from the repository root.
"""

import argparse
import json
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


CASES = {
    "int8_linear": "encoder.layers.0.feed_forward1.linear1.weight",
    "int8_conv": "encoder.layers.0.conv.pointwise_conv1.weight",
    "int4_linear": "encoder.layers.6.feed_forward1.linear1.weight",
    "int4_conv": "encoder.layers.6.conv.pointwise_conv1.weight",
}


def codes_for(record):
    codes = record["codes"].numpy()
    if record["bits"] == 4:
        out = np.empty(codes.size * 2, dtype=np.uint8)
        out[0::2] = codes & 15
        out[1::2] = codes >> 4
        codes = out
    return codes.reshape(record["shape"])


def size_of(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def create_case(record, name, mode, out_dir, constant=False):
    shape = (8, 64) if len(record["shape"]) == 2 else (8, 64, 1)
    codes = codes_for(record)[:8, :64].copy()
    minimum = record["minimum"][:8].numpy().copy()
    step = record["step"][:8].numpy().copy()
    if constant:
        # Explicitly exercise a channel with zero step and nonzero minimum.
        codes[0] = 0
        minimum[0] = np.float32(-0.75)
        step[0] = 0
    shape_params = (8,) + (1,) * (len(shape) - 1)
    expected = (minimum.reshape(shape_params) +
                codes.astype(np.float32) * step.reshape(shape_params)).astype(np.float16)
    stem = f"{name}{'_constant' if constant else ''}_{mode}"
    expected_path = out_dir / f"{stem}.expected.f16"
    expected.tofile(expected_path)

    if mode == "affine":
        data = codes.astype(types.np_uint4_dtype if record["bits"] == 4 else np.uint8)
        scale = step.copy()
        offset = np.zeros_like(scale)
        active = scale != 0
        offset[active] = -minimum[active] / scale[active]
        # A constant channel cannot use min/step as scale*(code-offset).
        # Its original codes are immaterial to the reconstructed weights.
        if constant:
            data[0] = 0
            scale[0] = 1
            offset[0] = -minimum[0]
        scale = scale.reshape(shape_params)
        offset = offset.reshape(shape_params)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
                    opset_version=ct.target.macOS15)
        def program(x):
            weight = mb.constexpr_blockwise_shift_scale(
                data=data, scale=scale, offset=offset)
            return mb.add(x=x, y=mb.cast(x=weight, dtype="fp16"))
    else:
        values = np.arange(1 << record["bits"], dtype=np.float32)
        lut = (minimum[:, None] + step[:, None] * values[None, :]).astype(np.float16)
        lut = lut.reshape((8,) + (1,) * (len(shape) - 1) + (len(values), 1))
        indices = codes.astype(types.np_uint4_dtype if record["bits"] == 4 else np.uint8)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)],
                    opset_version=ct.target.macOS15)
        def program(x):
            weight = mb.constexpr_lut_to_dense(indices=indices, lut=lut)
            return mb.add(x=x, y=weight)

    model = ct.convert(program, convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15)
    package = out_dir / f"{stem}.mlpackage"
    model.save(str(package))
    compiled = out_dir / f"{stem}.mlmodelc"
    subprocess.run(["xcrun", "coremlc", "compile", str(package), str(out_dir)], check=True,
                   capture_output=True, text=True)
    ops = [op.type for function in model.get_spec().mlProgram.functions.values()
           for block in function.block_specializations.values()
           for op in block.operations]
    return {
        "name": stem, "source_tensor": name, "mode": mode, "shape": shape,
        "bits": record["bits"], "constant_channel": constant,
        "python_equal": bool(np.array_equal(expected,
            (minimum.reshape(shape_params) + codes.astype(np.float32) *
             step.reshape(shape_params)).astype(np.float16))),
        "package_bytes": size_of(package), "compiled_bytes": size_of(compiled),
        "has_constexpr": any(op.startswith("constexpr_") for op in ops),
        "package": str(package), "compiled": str(compiled),
        "expected": str(expected_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("frozen_tdt_v1/model.pt"))
    ap.add_argument("--out", type=Path, default=Path("/private/tmp/frozen-tdt-coreml-probe"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)["tensors"]
    results = []
    for label, key in CASES.items():
        for mode in ("affine", "lut"):
            results.append(create_case(source[key], label, mode, args.out))
    for mode in ("affine", "lut"):
        results.append(create_case(source[CASES["int4_linear"]], "int4_linear",
                                   mode, args.out, constant=True))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
