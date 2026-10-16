#!/usr/bin/env python3
"""Create malformed-output sidecars for native compatibility regression tests."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

import coremltools as ct
import numpy as np
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from minimax_coreml_models import TransformerBlock, configuration, read_block


class TruncatedOutput(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, positions, keys, values, mask):
        hidden, keys, values = self.block(hidden, positions, keys, values, mask)
        return hidden[..., :hidden.shape[-1] // 2], keys, values


def load_exporter():
    spec = importlib.util.spec_from_file_location("export_minimax_coreml", SCRIPTS / "export-minimax-coreml.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_invalid(directory, truncate):
    exporter = load_exporter()
    source = directory / "mm3-lm-f16.gguf"
    tensors, metadata = read_block(source, "lm", 0)
    config = configuration(metadata, "lm")
    block = TransformerBlock(tensors, config, "lm").eval()
    model = TruncatedOutput(block).eval() if truncate else block
    inputs = exporter.sample_inputs(config, "lm")
    traced = torch.jit.trace(model, tuple(inputs.values()))
    names = ["hidden_out" if truncate else "renamed_hidden", "new_keys", "new_values"]
    converted = ct.convert(traced, inputs=exporter.input_specs(ct, inputs, *exporter.stage_limits(metadata, "lm")),
                           outputs=[ct.TensorType(name=name, dtype=np.float32) for name in names],
                           convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                           minimum_deployment_target=ct.target.macOS14, compute_units=ct.ComputeUnit.CPU_AND_NE)
    valid = ct.models.MLModel(str(source) + "-coreml/lm-block-0.mlpackage", skip_model_load=True)
    converted.user_defined_metadata.update(valid.user_defined_metadata)
    package = directory / ("bad-output-shape.mlpackage" if truncate else "bad-output-name.mlpackage")
    converted.save(str(package))
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(package), str(directory)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    args = parser.parse_args()
    export_invalid(args.fixture_dir, False)
    export_invalid(args.fixture_dir, True)


if __name__ == "__main__":
    main()
