#!/usr/bin/env python3
"""Export the Pocket TTS FlowLM generation step to a Core ML sidecar.

One prediction runs one generated frame (FlowLM advance plus a one-step LSD
flow) with the KV cache in Core ML state; see scripts/pocket_coreml_models.py.
The model is rebuilt in PyTorch from the same GGUF tensors the ggml engine
loads and written next to the GGUF under the engine's naming rule:

  flow-lm.gguf      -> flow-lm-step.mlmodelc
  flow-lm-f16.gguf  -> flow-lm-step.mlmodelc

`--context` fixes how many positions (voice, text and generated frames) the
state holds; the engine keeps a chunk that needs more on ggml. The step
computes in float16 with float32 inputs and outputs, and targets macOS 15 /
iOS 18 (stateful models). Core ML state cannot be placed on the Neural Engine,
so the engine loads it on CPU + GPU by default. The sidecar records the
checkpoint's source hash; the engine refuses a sidecar of another checkpoint.

  python3 scripts/export-pocket-coreml.py --gguf models/pocket/flow-lm.gguf --compile
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from pocket_coreml_models import INPUTS, OUTPUTS, STATES, FlowStep, example_inputs, read_gguf

QUANT_TAGS = ("f32", "f16", "bf16", "q8_0", "q5_0", "q4_0")
SUFFIX = "-step"
DEFAULT_CONTEXT = 1024
SCHEMA = "1"


def model_stem(path):
    return re.sub(r"[.-](?:" + "|".join(QUANT_TAGS) + r")$", "", Path(path).stem, flags=re.IGNORECASE)


def convert(model, inputs):
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(model, tuple(inputs[name] for name in INPUTS), check_trace=False)
    states = [ct.StateType(wrapped_type=ct.TensorType(shape=tuple(getattr(model, name).shape), dtype=np.float16),
                           name=name) for name in STATES]
    return ct.convert(traced, convert_to="mlprogram",
                      inputs=[ct.TensorType(name=name, shape=tuple(inputs[name].shape), dtype=np.float32)
                              for name in INPUTS],
                      outputs=[ct.TensorType(name=name, dtype=np.float32) for name in OUTPUTS],
                      states=states, minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16)


def export(args):
    import coremltools as ct

    tensors, metadata = read_gguf(args.gguf)
    stem = model_stem(args.gguf) + SUFFIX
    package = args.output / (stem + ".mlpackage")
    compiled = args.output / (stem + ".mlmodelc")
    for path in (package, compiled) if args.compile else (package,):
        if path.exists():
            if not args.force:
                raise FileExistsError(f"{path} exists (pass --force to replace it)")
            shutil.rmtree(path)
    model = FlowStep(tensors, metadata, args.context).eval()
    inputs = example_inputs(metadata, args.context)
    converted = convert(model, inputs)
    source = str(metadata["pocket.source_sha256"])
    converted.user_defined_metadata["pocket.coreml.schema"] = SCHEMA
    converted.user_defined_metadata["pocket.coreml.context"] = str(args.context)
    converted.user_defined_metadata["pocket.source_sha256"] = source
    converted.user_defined_metadata["pocket.coreml.source"] = args.gguf.name
    converted.save(str(package))
    manifest = {"source": args.gguf.name, "source_sha256": source, "context": args.context, "schema": SCHEMA,
                "inputs": {k: list(v.shape) for k, v in inputs.items()}, "outputs": list(OUTPUTS),
                "states": {name: list(getattr(model, name).shape) for name in STATES},
                "coremltools": ct.__version__, "torch": torch.__version__}
    (args.output / (stem + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile:
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(args.output)],
                       check=True, capture_output=True, text=True)
        print(compiled)
    print(package)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", required=True, type=Path, help="Pocket FlowLM GGUF (F32 or F16)")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT,
                        help="positions the state holds: voice + text + generated frames (default %(default)s)")
    parser.add_argument("--output", type=Path, help="output directory (default: next to the GGUF)")
    parser.add_argument("--compile", action="store_true", help="also compile to <output>/<stem>-step.mlmodelc")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    args = parser.parse_args()
    args.output = args.output or args.gguf.parent
    if not 2 <= args.context <= 8192:
        parser.error("--context must be 2..8192 (the engine's context limit)")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export(args)


if __name__ == "__main__":
    main()
