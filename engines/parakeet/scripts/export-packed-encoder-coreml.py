#!/usr/bin/env python3
"""Export the frozen TDT encoder with its original INT4/INT8 code indices.

Per-output-channel FP16 lookup tables contain exactly the checkpoint's
reconstructed FP16 values. Diagnostic modes can store some or all projections
densely. FP16 computation changes arithmetic precision; convolution fusions
can also change stored weights unless --fp16-no-fuse is selected. This
performs no k-means or second quantization.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.scope import ScopeInfo

from packed_tdt import load_packed


AFFINE_SUFFIXES = {
    "ff1.linear1.weight": "feed_forward1.linear1.weight",
    "ff1.linear2.weight": "feed_forward1.linear2.weight",
    "ff2.linear1.weight": "feed_forward2.linear1.weight",
    "ff2.linear2.weight": "feed_forward2.linear2.weight",
    "attn.q.weight": "self_attn.linear_q.weight",
    "attn.k.weight": "self_attn.linear_k.weight",
    "attn.v.weight": "self_attn.linear_v.weight",
    "attn.out.weight": "self_attn.linear_out.weight",
    "attn.pos.weight": "self_attn.linear_pos.weight",
    "conv.pw1.weight": "conv.pointwise_conv1.weight",
    "conv.dw.weight": "conv.depthwise_conv.weight",
    "conv.pw2.weight": "conv.pointwise_conv2.weight",
}


def source_name(gguf_name):
    match = re.fullmatch(r"encoder\.blk\.(\d+)\.(.*)", gguf_name)
    if match and match.group(2) in AFFINE_SUFFIXES:
        return f"encoder.layers.{match.group(1)}.{AFFINE_SUFFIXES[match.group(2)]}"
    return None


def use_packed_lookup(source, args):
    if args.dense_diagnostic:
        return False
    if args.dense_conv_diagnostic and ".conv." in source:
        return False
    return True


def expanded_codes(record):
    codes = record["codes"].numpy()
    if record["bits"] == 4:
        unpacked = np.empty(codes.size * 2, dtype=np.uint8)
        unpacked[0::2], unpacked[1::2] = codes & 15, codes >> 4
        codes = unpacked
    shape = tuple(record["shape"])
    if len(shape) == 3 and shape[-1] == 1:
        shape = shape[:-1]
    return codes.reshape(shape)


def lut_and_indices(record):
    indices = expanded_codes(record)
    minimum, step = record["minimum"].numpy(), record["step"].numpy()
    values = np.arange(1 << record["bits"], dtype=np.float32)
    lut = (minimum[:, None] + step[:, None] * values[None, :]).astype(np.float16)
    lut = lut.reshape((indices.shape[0],) + (1,) * (indices.ndim - 1) +
                      (len(values), 1))
    index_dtype = types.np_uint4_dtype if record["bits"] == 4 else np.uint8
    return lut, indices.astype(index_dtype)


def load_exporter():
    path = Path(__file__).with_name("export-encoder-coreml.py")
    spec = importlib.util.spec_from_file_location("parakeet_coreml_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(args):
    exporter = load_exporter()
    ref = exporter.load_reference_encoder(Path(__file__).parent)
    weights, meta = ref.load_gguf(args.gguf)
    if exporter.validate_export_contract(meta) != "tdt":
        raise ValueError("The packed sidecar requires a TDT GGUF")
    _, state, _, _ = load_packed(args.packed_dir)
    module = exporter.EncoderModule(ref, weights, meta).eval()
    example = torch.zeros(128, args.n_mel_frames, dtype=torch.float32)
    with torch.inference_mode():
        expected_shape = tuple(module(example).shape)
        traced = torch.jit.trace(module, example, check_trace=False)
    removed_passes = {
        "common::const_elimination", "common::const_deduplication",
        "common::fuse_conv_scale", "common::fuse_conv_bias",
        "common::fuse_conv_batchnorm", "common::fuse_matmul_weight_bias",
        "common::fuse_linear_bias", "common::fuse_squeeze_expand_dims",
    }
    pipeline = copy.deepcopy(ct.PassPipeline.DEFAULT)
    pipeline.remove_passes(removed_passes)
    program = ct.convert(
        traced, convert_to="milinternal",
        inputs=[ct.TensorType(name="mel", shape=tuple(example.shape), dtype=np.float32)],
        outputs=[ct.TensorType(name="encoder_out", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
        pass_pipeline=pipeline,
    )

    replaced = []
    expected_affine = {source_name(key): key for key in module._keys
                       if source_name(key) and use_packed_lookup(source_name(key), args)}
    with program.functions["main"]:
        for op in list(program.functions["main"].operations):
            match = re.fullmatch(r"w_(\d+)", op.name)
            if op.op_type != "const" or not match:
                continue
            key = module._keys[int(match.group(1))]
            name = source_name(key)
            if name is None or not use_packed_lookup(name, args):
                continue
            record = state.records[name]
            if record["kind"] != "affine":
                raise ValueError(f"Expected packed affine weight: {name}")
            lut, indices = lut_and_indices(record)
            rows = np.arange(indices.shape[0]).reshape(
                (-1,) + (1,) * (indices.ndim - 1))
            restored = lut.reshape(indices.shape[0], -1)[rows, indices]
            dense = op.val.val
            if dense.shape != restored.shape or not np.array_equal(dense, restored.astype(dense.dtype)):
                raise ValueError(f"MIL weight differs from packed checkpoint: {name}")
            with mb.scope(*(ScopeInfo(source=source, data=data)
                            for source, data in op.scopes.items())):
                value = mb.constexpr_lut_to_dense(
                    indices=indices, lut=lut, before_op=op,
                    name=op.name + "_packed_lut")
                if dense.dtype == np.float32:
                    value = mb.cast(x=value, dtype="fp32", before_op=op,
                                    name=op.name + "_packed_fp32")
            op.enclosing_block.replace_uses_of_var_after_op(
                anchor_op=op, old_var=op.outputs[0], new_var=value,
                force_replace=True)
            op.enclosing_block.remove_ops([op])
            replaced.append(name)
    if set(replaced) != set(expected_affine):
        missing = sorted(set(expected_affine) - set(replaced))
        samples = []
        for name in missing[:8]:
            key = expected_affine[name]
            index = module._keys.index(key)
            source_values = state[name].numpy().astype(np.float32).ravel()
            candidates = [(op.name, tuple(op.val.val.shape),
                str(op.val.val.dtype), float(np.max(np.abs(op.val.val.ravel().astype(np.float32)-source_values))),
                bool(op.val.val.ndim == 2 and np.array_equal(op.val.val.T.ravel(), source_values)))
                for op in program.functions["main"].operations if op.op_type == "const" and
                op.val.val is not None and hasattr(op.val.val, "size") and
                op.val.val.size == source_values.size]
            matches = [(op.name, op.op_type, tuple(op.val.val.shape)) for op in
                program.functions["main"].operations if op.op_type == "const" and
                op.val.val is not None and hasattr(op.val.val, "size") and
                op.val.val.size == source_values.size and
                np.array_equal(op.val.val.ravel(), source_values)][:3]
            samples.append((key, index, [(op.name, op.op_type) for op in
                program.functions["main"].operations if re.match(rf"w_{index}(?:$|_)", op.name)][:6],matches,candidates[:6]))
        raise ValueError(f"Missing packed MIL weights ({len(missing)}): {samples}")

    # The first conversion mutates its pass pipeline when FLOAT32 is requested,
    # removing FP16 casting. Build a fresh pipeline for this explicit candidate.
    if args.fp16_no_fuse:
        final_pipeline = copy.deepcopy(ct.PassPipeline.DEFAULT)
        final_pipeline.remove_passes(removed_passes - {
            "common::const_elimination", "common::const_deduplication"})
    else:
        final_pipeline = ct.PassPipeline.DEFAULT if args.optimize_after else pipeline
    final_precision = (ct.precision.FLOAT16 if args.fp16_no_fuse or args.optimize_after
                       else ct.precision.FLOAT32)
    model = ct.convert(program, convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15,
                       pass_pipeline=final_pipeline,
                       compute_precision=final_precision)
    lut_ops = sum(op.type == "constexpr_lut_to_dense"
                  for function in model.get_spec().mlProgram.functions.values()
                  for block in function.block_specializations.values()
                  for op in block.operations)
    if lut_ops != len(replaced):
        raise ValueError(f"Core ML model retained {lut_ops} lookup tensors, expected {len(replaced)}")
    with (args.packed_dir / "model.pt").open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    model.user_defined_metadata["parakeet.encoder.source_sha256"] = source_sha256
    model.save(str(args.out))
    compiled = None
    if args.compile_dir:
        subprocess.run(["xcrun", "coremlc", "compile", str(args.out),
                        str(args.compile_dir)], check=True)
        compiled = args.compile_dir / (args.out.stem + ".mlmodelc")
    representation = ("dense-diagnostic" if args.dense_diagnostic else
                      "dense-convolutions-with-exact-lookups" if args.dense_conv_diagnostic else
                      "exact-lookups")
    report = {"lookup_tensors": lut_ops, "encoder_output_shape": expected_shape,
              "representation": representation,
              "compute_precision": str(final_precision),
              "convolution_weight_fusions_disabled": not args.optimize_after,
              "source_sha256": source_sha256,
              "package": str(args.out), "compiled": str(compiled) if compiled else None,
              "package_bytes": sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()),
              "compiled_bytes": (sum(p.stat().st_size for p in compiled.rglob("*") if p.is_file())
                                 if compiled else None)}
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", type=Path, default=Path("frozen_tdt_v1"))
    ap.add_argument("--gguf", type=Path, default=Path("frozen_tdt_v1/frozen-tdt-v1.f16.gguf"))
    ap.add_argument("--n-mel-frames", type=int, default=1501)
    ap.add_argument("--out", type=Path, default=Path("frozen_tdt_v1/frozen-tdt-v1-encoder.mlpackage"))
    ap.add_argument("--compile-dir", type=Path, default=Path("frozen_tdt_v1"))
    ap.add_argument("--optimize-after", action="store_true",
                    help="Run Core ML's default optimization passes after exact LUT insertion.")
    ap.add_argument("--dense-diagnostic", action="store_true",
                    help="Temporary dense-FP16 comparison export; does not preserve packed storage.")
    ap.add_argument("--dense-conv-diagnostic", action="store_true",
                    help="Temporary hybrid export with dense convolution weights; no requantization.")
    ap.add_argument("--fp16-no-fuse", action="store_true",
                    help="Request FP16 computation while retaining original convolution weights.")
    args = ap.parse_args()
    if args.dense_diagnostic and args.dense_conv_diagnostic:
        ap.error("select only one diagnostic representation")
    if args.fp16_no_fuse and args.optimize_after:
        ap.error("--fp16-no-fuse requires omitting --optimize-after")
    if (args.dense_diagnostic or args.dense_conv_diagnostic) and args.out == Path("frozen_tdt_v1/frozen-tdt-v1-encoder.mlpackage"):
        ap.error("diagnostic exports require a separate explicit --out path")
    export(args)


if __name__ == "__main__":
    main()
