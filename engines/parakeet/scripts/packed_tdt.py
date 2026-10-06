"""Read the local parakeet_affine_packed_v1 checkpoint for GGUF conversion."""

from collections.abc import Mapping
from pathlib import Path

import torch
import yaml


class PackedTensors(Mapping):
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, name):
        record = self.records[name]
        if record["kind"] == "raw":
            return record["tensor"]
        if record["kind"] != "affine":
            raise ValueError(f"Unsupported packed tensor kind: {record['kind']}")
        bits = record["bits"]
        if bits not in (4, 8):
            raise ValueError(f"Unsupported affine width: {bits}")
        shape = tuple(record["shape"])
        count = 1
        for dim in shape:
            count *= dim
        codes = record["codes"]
        if codes.numel() != (count * bits + 7) // 8:
            raise ValueError(f"Incorrect packed code count for {name}")
        if bits == 4:
            # Consecutive values occupy the low and high nibble, respectively.
            unpacked = torch.empty(count, dtype=torch.uint8)
            unpacked[0::2] = codes & 15
            unpacked[1::2] = codes >> 4
            codes = unpacked
        values = codes.reshape(shape).float()
        rows = shape[0]
        broadcast_shape = (rows,) + (1,) * (len(shape) - 1)
        minimum = record["minimum"].reshape(broadcast_shape)
        step = record["step"].reshape(broadcast_shape)
        # The source stores per-output-channel affine quantization. Restore the
        # quantizer's half-precision values before the GGUF writer sees them.
        return (minimum + values * step).to(getattr(torch, record["dtype"].split(".")[-1]))


def load_packed(directory: Path):
    config_path = directory / "architecture" / "model_config.yaml"
    checkpoint_path = directory / "model.pt"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("format") != "parakeet_affine_packed_v1":
        raise ValueError("Expected a parakeet_affine_packed_v1 checkpoint")
    if checkpoint.get("model_name") != "nvidia/parakeet-tdt-0.6b-v3":
        raise ValueError("This packed import supports Parakeet TDT 0.6B v3 only")
    tokenizer_name = Path(config["tokenizer"]["model_path"].split("nemo:", 1)[-1]).name
    tokenizer = (directory / "architecture" / tokenizer_name).read_bytes()
    return config, PackedTensors(checkpoint["tensors"]), tokenizer, {}
