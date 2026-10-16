#!/usr/bin/env python3
"""Export MiniMax-Music3 transformer blocks from GGUF to optional Core ML sidecars."""

import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import torch

from minimax_coreml_models import CFG_BATCH, DitBlock, TransformerBlock, configuration, read_block

HASH_CHUNK_BYTES = 1024 * 1024
TRACE_TOKENS = 2
TRACE_CACHE = 4
DIT_MAX_TOKENS = 4097


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_limits(metadata, stage):
    if stage == "lm":
        return int(metadata["qwen3.context_length"]), int(metadata["qwen3.context_length"])
    if stage == "depth":
        positions = int(metadata["mm3.depth.max_position"])
        return positions, positions
    return DIT_MAX_TOKENS, 0


def sample_inputs(config, stage, tokens=TRACE_TOKENS, cache=TRACE_CACHE):
    inputs = {"hidden": torch.zeros(CFG_BATCH, tokens, config["hidden"]),
              "positions": torch.arange(tokens, dtype=torch.float32)}
    if stage != "dit":
        shape = (CFG_BATCH, config["kv_heads"], cache, config["head_dim"])
        inputs.update(keys=torch.zeros(shape), values=torch.zeros(shape),
                      mask=torch.zeros(1, 1, tokens, cache))
    return inputs


def input_specs(ct, inputs, max_tokens, max_cache):
    tokens = ct.RangeDim(1, max_tokens, default=inputs["hidden"].shape[1])
    cache = ct.RangeDim(1, max_cache, default=inputs["keys"].shape[2]) if max_cache else None
    specs = []
    for name, value in inputs.items():
        shape = list(value.shape)
        if name == "hidden":
            shape[1] = tokens
        elif name == "positions":
            shape[0] = tokens
        elif name in ("keys", "values"):
            shape[2] = cache
        elif name == "mask":
            shape[2], shape[3] = tokens, cache
        specs.append(ct.TensorType(name=name, shape=tuple(shape), dtype=np.float32))
    return specs


def block_count(metadata, stage):
    return int(metadata[("qwen3" if stage == "lm" else "mm3." + stage) + ".block_count"])


def use_half_precision(op):
    if op.op_type == "conv":
        return True
    if op.op_type != "linear":
        return False
    name = op.inputs["weight"].name
    return any(part in name for part in ("ffn_gate__weight", "ffn_up__weight", "ffn_in__weight"))


def convert_block(args, layer, source_hash):
    import coremltools as ct

    tensors, metadata = read_block(args.gguf, args.stage, layer)
    config = configuration(metadata, args.stage)
    model = (DitBlock if args.stage == "dit" else TransformerBlock)(tensors, config, args.stage).eval()
    max_tokens, max_cache = stage_limits(metadata, args.stage)
    trace_cache = min(TRACE_CACHE, max_cache) if max_cache else 0
    trace_tokens = min(TRACE_TOKENS, max_tokens, trace_cache or max_tokens)
    inputs = sample_inputs(config, args.stage, trace_tokens, trace_cache)
    traced = torch.jit.trace(model, tuple(inputs.values()))
    output_names = ["hidden_out"] if args.stage == "dit" else ["hidden_out", "new_keys", "new_values"]
    precision = ct.transform.FP16ComputePrecision(op_selector=use_half_precision)
    converted = ct.convert(traced, convert_to="mlprogram", inputs=input_specs(ct, inputs, max_tokens, max_cache),
                           outputs=[ct.TensorType(name=name, dtype=np.float32) for name in output_names],
                           minimum_deployment_target=ct.target.macOS14, compute_precision=precision,
                           compute_units=ct.ComputeUnit.CPU_AND_NE)
    converted.user_defined_metadata.update({"mm3_schema": "1", "mm3_source_sha256": source_hash,
        "mm3_stage": args.stage, "mm3_layer": str(layer),
        **{"mm3_" + name: str(config[name]) for name in ("hidden", "heads", "kv_heads", "head_dim")}})
    output = Path(args.out) if args.out else Path(str(args.gguf) + "-coreml")
    output.mkdir(parents=True, exist_ok=True)
    package = output / f"{args.stage}-block-{layer}.mlpackage"
    converted.save(str(package))
    if args.compile:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(output)], check=True)
    print(f"Exported {package}", flush=True)


def export(args):
    _, metadata = read_block(args.gguf, args.stage, 0)
    count = block_count(metadata, args.stage)
    layers = range(count) if args.layer is None else [args.layer]
    if any(layer < 0 or layer >= count for layer in layers):
        raise ValueError(f"Layer must be in [0, {count})")
    source_hash = fingerprint(args.gguf)
    for layer in layers:
        convert_block(args, layer, source_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--stage", choices=("lm", "depth", "dit"), required=True)
    parser.add_argument("--layer", type=int, help="Export one block; omitted exports every block in the stage")
    parser.add_argument("--out", type=Path, help="Default: <gguf>-coreml")
    parser.add_argument("--compile", action="store_true")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
