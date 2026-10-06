#!/usr/bin/env python3
"""Verify serialized Core ML representations of the checkpoint's affine tensors.

Supports the original lookup export and the dense-convolution hybrid. This
checks weight bytes and indices, not runtime numerical equivalence or WER.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil.frontend.milproto.load import load_mil_proto

from packed_tdt import load_packed


def load_script(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identical(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return (left.shape == right.shape and left.dtype == right.dtype and
            left.tobytes() == right.tobytes())


def verify(package, packed_dir, gguf_path):
    packed_exporter = load_script('export-packed-encoder-coreml.py', 'packed_exporter')
    exporter = packed_exporter.load_exporter()
    ref = exporter.load_reference_encoder(Path(__file__).parent)
    weights, meta = ref.load_gguf(gguf_path)
    module = exporter.EncoderModule(ref, weights, meta).eval()
    _, state, _, _ = load_packed(packed_dir)
    model = ct.models.MLModel(str(package), skip_model_load=True)
    spec = model.get_spec()
    program = load_mil_proto(spec.mlProgram, spec.specificationVersion, model.weights_dir)
    operations = list(program.functions['main'].operations)
    by_name = {op.name: op for op in operations}
    expected = [(index, packed_exporter.source_name(key))
                for index, key in enumerate(module._keys)
                if packed_exporter.source_name(key)]
    # The fixed TDT graph executes five subsampling convolutions followed by
    # pw1, depthwise, pw2 for each encoder block. Validate the count and every
    # value before using this structural mapping; optimizer names can change.
    conv_names = [name for _, name in expected if '.conv.' in name]
    conv_ops = [op for op in operations if op.op_type == 'conv']
    failures = []
    if len(conv_ops) != len(conv_names) + 5:
        failures.append('Unexpected convolution graph structure')
    conv_by_source = dict(zip(conv_names, conv_ops[5:]))
    counts = {'lookup_int4': 0, 'lookup_int8': 0,
              'dense_convolution_int4': 0, 'dense_convolution_int8': 0}
    for index, name in expected:
        record = state.records[name]
        lookup = by_name.get(f'w_{index}_packed_lut')
        if lookup is not None:
            lut, indices = packed_exporter.lut_and_indices(record)
            if not identical(lookup.lut.val, lut):
                failures.append(f'{name}: LUT bytes differ')
            if (lookup.indices.dtype.get_bitwidth() != record['bits'] or
                    not identical(lookup.indices.val, indices)):
                failures.append(f'{name}: indices or bit width differ')
            counts[f'lookup_int{record["bits"]}'] += 1
        elif name in conv_by_source:
            value = conv_by_source[name].weight.val
            if value is None or not identical(value, state[name].numpy()):
                failures.append(f'{name}: dense FP16 weight bytes differ')
            counts[f'dense_convolution_int{record["bits"]}'] += 1
        else:
            failures.append(f'{name}: missing representation')
    if sum(counts.values()) != len(expected):
        failures.append('Not every affine tensor was verified')
    return {'package': str(package), 'affine_tensors_checked': sum(counts.values()),
            'counts': counts, 'failures': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--packed-dir', type=Path, default=Path('frozen_tdt_v1'))
    parser.add_argument('--gguf', type=Path, default=Path('frozen_tdt_v1/frozen-tdt-v1.f16.gguf'))
    args = parser.parse_args()
    result = verify(args.package, args.packed_dir, args.gguf)
    print(json.dumps(result, indent=2))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
