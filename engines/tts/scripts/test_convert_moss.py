#!/usr/bin/env python3
"""Converter coverage for the MOSS Delay scripts: fabricates a tiny
MossTTSDelay checkpoint and a tiny MOSS audio tokenizer on disk, runs both
converters, and validates the emitted GGUFs (tensor census, mapped names,
metadata) with the gguf reader. Skips cleanly when numpy or gguf are absent.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import gguf
except ImportError as error:
    print(f"SKIP: {error.name} is not installed")
    sys.exit(0)

SCRIPTS = Path(__file__).resolve().parent

N_EMBD = 16
N_FF = 32
N_HEADS = 2
N_KV_HEADS = 1
HEAD_DIM = 8
N_VQ = 2
TEXT_VOCAB = 24
AUDIO_HEAD = 8
D_MODEL = 8
RVQ_DIM = 8
CODE_DIM = 4
CODE_SIZE = 8
OUT_DIM = 4


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, dict] = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        data = np.ascontiguousarray(tensor, dtype=np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    payload = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(payload)))
        f.write(payload)
        for blob in blobs:
            f.write(blob)


def make_backbone_checkpoint(root: Path) -> Path:
    model_dir = root / "backbone"
    model_dir.mkdir()
    tensors: dict[str, np.ndarray] = {
        "language_model.embed_tokens.weight": np.zeros((TEXT_VOCAB, N_EMBD)),
        "language_model.norm.weight": np.zeros(N_EMBD),
    }
    layer = "language_model.layers.0."
    tensors[layer + "input_layernorm.weight"] = np.zeros(N_EMBD)
    tensors[layer + "self_attn.q_proj.weight"] = np.zeros((N_HEADS * HEAD_DIM, N_EMBD))
    tensors[layer + "self_attn.k_proj.weight"] = np.zeros((N_KV_HEADS * HEAD_DIM, N_EMBD))
    tensors[layer + "self_attn.v_proj.weight"] = np.zeros((N_KV_HEADS * HEAD_DIM, N_EMBD))
    tensors[layer + "self_attn.o_proj.weight"] = np.zeros((N_EMBD, N_HEADS * HEAD_DIM))
    tensors[layer + "self_attn.q_norm.weight"] = np.zeros(HEAD_DIM)
    tensors[layer + "self_attn.k_norm.weight"] = np.zeros(HEAD_DIM)
    tensors[layer + "post_attention_layernorm.weight"] = np.zeros(N_EMBD)
    tensors[layer + "mlp.gate_proj.weight"] = np.zeros((N_FF, N_EMBD))
    tensors[layer + "mlp.up_proj.weight"] = np.zeros((N_FF, N_EMBD))
    tensors[layer + "mlp.down_proj.weight"] = np.zeros((N_EMBD, N_FF))
    for i in range(N_VQ):
        tensors[f"emb_ext.{i}.weight"] = np.zeros((AUDIO_HEAD, N_EMBD))
    for k in range(N_VQ + 1):
        rows = TEXT_VOCAB if k == 0 else AUDIO_HEAD
        tensors[f"lm_heads.{k}.weight"] = np.zeros((rows, N_EMBD))
    write_safetensors(model_dir / "model.safetensors", tensors)

    config = {
        "model_type": "moss_tts_delay",
        "n_vq": N_VQ,
        "audio_vocab_size": AUDIO_HEAD - 1,
        "audio_pad_code": AUDIO_HEAD - 1,
        "audio_start_token_id": 3,
        "audio_end_token_id": 4,
        "audio_user_slot_token_id": 5,
        "audio_assistant_gen_slot_token_id": 6,
        "audio_assistant_delay_slot_token_id": 7,
        "sampling_rate": 24000,
        "language_config": {
            "num_hidden_layers": 1,
            "hidden_size": N_EMBD,
            "intermediate_size": N_FF,
            "num_attention_heads": N_HEADS,
            "num_key_value_heads": N_KV_HEADS,
            "head_dim": HEAD_DIM,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    vocab = {f"tok{i}": i for i in range(TEXT_VOCAB)}
    tokenizer = {
        "model": {"vocab": vocab, "merges": ["a b"]},
        "added_tokens": [
            {"id": 0, "content": "<|endoftext|>"},
            {"id": 1, "content": "<|im_start|>"},
            {"id": 2, "content": "<|im_end|>"},
        ],
    }
    (model_dir / "tokenizer.json").write_text(json.dumps(tokenizer))
    return model_dir


def make_codec_checkpoint(root: Path) -> Path:
    model_dir = root / "tokenizer-model"
    model_dir.mkdir()
    tensors: dict[str, np.ndarray] = {
        "quantizer.input_proj.weight": np.zeros((RVQ_DIM, RVQ_DIM, 1)),
        "quantizer.input_proj.bias": np.zeros(RVQ_DIM),
        "quantizer.output_proj.weight": np.zeros((OUT_DIM, RVQ_DIM, 1)),
        "quantizer.output_proj.bias": np.zeros(OUT_DIM),
    }
    for iq in range(N_VQ):
        q = f"quantizer.quantizers.{iq}."
        tensors[q + "codebook.weight"] = np.zeros((CODE_SIZE, CODE_DIM))
        tensors[q + "in_proj.weight"] = np.zeros((CODE_DIM, RVQ_DIM, 1))
        tensors[q + "in_proj.bias"] = np.zeros(CODE_DIM)
        tensors[q + "out_proj.weight"] = np.zeros((RVQ_DIM, CODE_DIM, 1))
        tensors[q + "out_proj.bias"] = np.zeros(RVQ_DIM)
    for prefix, transformer_in in (("encoder.1.", RVQ_DIM), ("decoder.0.", OUT_DIM)):
        layer = prefix + "transformer.layers.0."
        tensors[layer + "self_attn.in_projs.0.weight"] = np.zeros((3 * D_MODEL, D_MODEL, 1))
        tensors[layer + "self_attn.out_projs.0.weight"] = np.zeros((D_MODEL, D_MODEL, 1))
        tensors[layer + "linear1.weight"] = np.zeros((2 * D_MODEL, D_MODEL, 1))
        tensors[layer + "linear2.weight"] = np.zeros((D_MODEL, 2 * D_MODEL, 1))
        tensors[layer + "norm1.weight"] = np.zeros(D_MODEL)
        tensors[layer + "norm1.bias"] = np.zeros(D_MODEL)
        tensors[layer + "norm2.weight"] = np.zeros(D_MODEL)
        tensors[layer + "norm2.bias"] = np.zeros(D_MODEL)
        tensors[prefix + "input_proj.weight"] = np.zeros((D_MODEL, transformer_in, 1))
        tensors[prefix + "output_proj.weight"] = np.zeros((transformer_in, D_MODEL, 1))
    write_safetensors(model_dir / "model.safetensors", tensors)

    def transformer(dimension: int) -> dict:
        return {
            "module_type": "Transformer",
            "input_dimension": dimension,
            "output_dimension": dimension,
            "d_model": D_MODEL,
            "num_heads": 2,
            "num_layers": 1,
            "max_period": 10000.0,
        }

    patch = {"module_type": "PatchedPretransform", "patch_size": OUT_DIM}
    config = {
        "sampling_rate": 24000,
        "downsample_rate": OUT_DIM,
        "quantizer_kwargs": {
            "quantizer_type": "rlfq",
            "input_dim": RVQ_DIM,
            "rvq_dim": RVQ_DIM,
            "output_dim": OUT_DIM,
            "num_quantizers": N_VQ,
            "codebook_size": CODE_SIZE,
            "codebook_dim": CODE_DIM,
        },
        "encoder_kwargs": [patch, transformer(RVQ_DIM)],
        "decoder_kwargs": [transformer(OUT_DIM), patch],
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def run_converter(script: str, args: list[str]) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{script} failed:\n{result.stdout}\n{result.stderr}")


def reader_tensors(path: Path) -> dict[str, tuple[int, ...]]:
    reader = gguf.GGUFReader(str(path))
    return {t.name: tuple(int(d) for d in t.shape) for t in reader.tensors}


def check_backbone(path: Path) -> None:
    names = reader_tensors(path)
    expected = 3 + 2 * N_VQ + 11
    assert len(names) == expected, f"backbone tensor census: {len(names)} != {expected}"
    for required in ("token_embd.weight", "output.weight", "output_norm.weight",
                     "token_embd_audio.0.weight", "output_audio.1.weight",
                     "blk.0.attn_q.weight", "blk.0.ffn_down.weight"):
        assert required in names, f"backbone is missing {required}"
    print("backbone converter: PASS")


def check_codec(encoder: Path, decoder: Path) -> None:
    for path, section in ((encoder, "encoder"), (decoder, "decoder")):
        names = reader_tensors(path)
        for required in ("quantizer.quantizers.0.codebook.weight",
                         "blk.0.layer.0.attn_qkv.weight",
                         "blk.0.layer.0.ffn_up.weight"):
            assert required in names, f"{section} is missing {required}"
    print("codec converter: PASS")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        backbone_dir = make_backbone_checkpoint(root)
        backbone_out = root / "backbone.gguf"
        run_converter("convert-moss-delay-to-gguf.py",
                      [str(backbone_dir), "--outtype", "f32", "--outfile", str(backbone_out)])
        check_backbone(backbone_out)

        codec_dir = make_codec_checkpoint(root)
        encoder_out = root / "encoder.gguf"
        decoder_out = root / "decoder.gguf"
        run_converter("convert-moss-codec-to-gguf.py",
                      [str(codec_dir), "--outtype", "f32",
                       "--encoder-outfile", str(encoder_out),
                       "--decoder-outfile", str(decoder_out)])
        check_codec(encoder_out, decoder_out)
    print("moss converters: OK")


if __name__ == "__main__":
    main()
