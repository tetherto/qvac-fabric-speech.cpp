#!/usr/bin/env python3
"""Converter coverage for the MOSS scripts: fabricates a tiny MossTTSDelay
checkpoint, a tiny MOSS audio tokenizer, a tiny MOSS-SoundEffect pipeline and a
tiny MOSS-Transcribe-Diarize checkpoint on disk, runs the converters, and validates the emitted GGUFs (tensor census,
mapped names, folded weights, metadata) with the gguf reader. Skips cleanly
when numpy or gguf are absent.
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
SFX_WIDTH = 32
SFX_FF = 64
SFX_HEAD_DIM = 16
SFX_LATENT = 4
SFX_DECODER_DIM = 8
SFX_RATES = [2, 3]
SFX_KERNEL = 7
SFX_GAIN = 2.0
TD_MELS = 4
TD_FFT = 16
TD_ENC = 8
TD_ENC_FF = 16
TD_ENC_CTX = 8
TD_MERGE = 4
TD_TEXT = 32
TD_TEXT_FF = 64
TD_HEAD_DIM = 8
TD_ADDED = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|audio_start|>", "<|audio_end|>",
            "<|audio_pad|>", "<think>"]


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


def sfx_text_encoder_tensors() -> dict[str, np.ndarray]:
    kv_dim = SFX_HEAD_DIM
    q_dim = 2 * SFX_HEAD_DIM
    layer = "model.layers.0."
    return {
        "model.embed_tokens.weight": np.ones((TEXT_VOCAB, SFX_WIDTH)),
        "model.norm.weight": np.ones(SFX_WIDTH),
        "lm_head.weight": np.ones((TEXT_VOCAB, SFX_WIDTH)),
        layer + "input_layernorm.weight": np.ones(SFX_WIDTH),
        layer + "self_attn.q_proj.weight": np.ones((q_dim, SFX_WIDTH)),
        layer + "self_attn.k_proj.weight": np.ones((kv_dim, SFX_WIDTH)),
        layer + "self_attn.v_proj.weight": np.ones((kv_dim, SFX_WIDTH)),
        layer + "self_attn.o_proj.weight": np.ones((SFX_WIDTH, q_dim)),
        layer + "self_attn.q_norm.weight": np.ones(SFX_HEAD_DIM),
        layer + "self_attn.k_norm.weight": np.ones(SFX_HEAD_DIM),
        layer + "post_attention_layernorm.weight": np.ones(SFX_WIDTH),
        layer + "mlp.gate_proj.weight": np.ones((SFX_FF, SFX_WIDTH)),
        layer + "mlp.up_proj.weight": np.ones((SFX_FF, SFX_WIDTH)),
        layer + "mlp.down_proj.weight": np.ones((SFX_WIDTH, SFX_FF)),
    }


def sfx_dit_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "patch_embedding.weight": np.ones((SFX_WIDTH, SFX_LATENT, 1)),
        "patch_embedding.bias": np.ones(SFX_WIDTH),
        "scale_shift_table": np.ones((1, 2, SFX_WIDTH)),
        "proj_out.weight": np.ones((SFX_LATENT, SFX_WIDTH)),
        "proj_out.bias": np.ones(SFX_LATENT),
    }
    for linear, (rows, cols) in {
        "condition_embedder.text_embedder.linear_1": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.text_embedder.linear_2": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.time_embedder.linear_1": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.time_embedder.linear_2": (SFX_WIDTH, SFX_WIDTH),
        "condition_embedder.time_proj": (6 * SFX_WIDTH, SFX_WIDTH),
    }.items():
        tensors[linear + ".weight"] = np.ones((rows, cols))
        tensors[linear + ".bias"] = np.ones(rows)
    block = "blocks.0."
    for attn in ("attn1", "attn2"):
        for proj in ("to_q", "to_k", "to_v", "to_out.0"):
            tensors[f"{block}{attn}.{proj}.weight"] = np.ones((SFX_WIDTH, SFX_WIDTH))
            tensors[f"{block}{attn}.{proj}.bias"] = np.ones(SFX_WIDTH)
        tensors[f"{block}{attn}.norm_q.weight"] = np.ones(SFX_WIDTH)
        tensors[f"{block}{attn}.norm_k.weight"] = np.ones(SFX_WIDTH)
    tensors[block + "norm2.weight"] = np.ones(SFX_WIDTH)
    tensors[block + "norm2.bias"] = np.ones(SFX_WIDTH)
    tensors[block + "ffn.net.0.proj.weight"] = np.ones((SFX_FF, SFX_WIDTH))
    tensors[block + "ffn.net.0.proj.bias"] = np.ones(SFX_FF)
    tensors[block + "ffn.net.2.weight"] = np.ones((SFX_WIDTH, SFX_FF))
    tensors[block + "ffn.net.2.bias"] = np.ones(SFX_WIDTH)
    tensors[block + "scale_shift_table"] = np.ones((1, 6, SFX_WIDTH))
    return tensors


def add_weight_norm(tensors: dict[str, np.ndarray], prefix: str, shape: tuple[int, ...]) -> None:
    direction = np.arange(1, int(np.prod(shape)) + 1, dtype=np.float64).reshape(shape)
    norm = np.sqrt((direction ** 2).sum(axis=tuple(range(1, len(shape))), keepdims=True))
    tensors[prefix + ".weight_v"] = direction
    tensors[prefix + ".weight_g"] = SFX_GAIN * norm


def add_residual_units(tensors: dict[str, np.ndarray], prefix: str, channels: int) -> None:
    for unit in range(3):
        unit_prefix = f"{prefix}.{unit + 2}.block"
        tensors[unit_prefix + ".0.alpha"] = np.ones((1, channels, 1))
        add_weight_norm(tensors, unit_prefix + ".1", (channels, channels, SFX_KERNEL))
        tensors[unit_prefix + ".1.bias"] = np.ones(channels)
        tensors[unit_prefix + ".2.alpha"] = np.ones((1, channels, 1))
        add_weight_norm(tensors, unit_prefix + ".3", (channels, channels, 1))
        tensors[unit_prefix + ".3.bias"] = np.ones(channels)


def sfx_vae_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "post_quant_conv.weight": np.ones((SFX_LATENT, SFX_LATENT, 1)),
        "post_quant_conv.bias": np.ones(SFX_LATENT),
    }
    add_weight_norm(tensors, "decoder.model.0", (SFX_DECODER_DIM, SFX_LATENT, SFX_KERNEL))
    tensors["decoder.model.0.bias"] = np.ones(SFX_DECODER_DIM)
    channels = SFX_DECODER_DIM
    for block, rate in enumerate(SFX_RATES):
        prefix = f"decoder.model.{block + 1}.block"
        tensors[prefix + ".0.alpha"] = np.full((1, channels, 1), 0.5)
        add_weight_norm(tensors, prefix + ".1", (channels, channels // 2, 2 * rate))
        tensors[prefix + ".1.bias"] = np.ones(channels // 2)
        add_residual_units(tensors, prefix, channels // 2)
        channels //= 2
    last = len(SFX_RATES) + 1
    tensors[f"decoder.model.{last}.alpha"] = np.ones((1, channels, 1))
    add_weight_norm(tensors, f"decoder.model.{last + 1}", (1, channels, SFX_KERNEL))
    tensors[f"decoder.model.{last + 1}.bias"] = np.ones(1)
    return tensors


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def make_sfx_checkpoint(root: Path) -> Path:
    model_dir = root / "sfx"
    for sub in ("text_encoder", "transformer", "vae", "tokenizer", "scheduler"):
        (model_dir / sub).mkdir(parents=True)
    write_safetensors(model_dir / "text_encoder" / "model.safetensors", sfx_text_encoder_tensors())
    write_safetensors(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", sfx_dit_tensors())
    write_safetensors(model_dir / "vae" / "vae.safetensors", sfx_vae_tensors())
    write_json(model_dir / "text_encoder" / "config.json", {
        "num_hidden_layers": 1, "hidden_size": SFX_WIDTH, "intermediate_size": SFX_FF,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": SFX_HEAD_DIM,
        "rope_theta": 1000000.0, "rms_norm_eps": 1e-6})
    write_json(model_dir / "transformer" / "config.json", {
        "dim": SFX_WIDTH, "in_dim": SFX_LATENT, "out_dim": SFX_LATENT, "ffn_dim": SFX_FF,
        "text_dim": SFX_WIDTH, "freq_dim": SFX_WIDTH, "eps": 1e-6, "patch_size": [1],
        "num_heads": 2, "num_layers": 1, "has_image_input": False, "vae_type": "dac"})
    write_json(model_dir / "vae" / "config.json", {
        "latent_dim": SFX_LATENT, "decoder_dim": SFX_DECODER_DIM, "decoder_rates": SFX_RATES,
        "sample_rate": 48000})
    write_json(model_dir / "model_index.json", {"max_inference_seconds": 30})
    write_json(model_dir / "scheduler" / "scheduler_config.json", {"shift": 5.0, "num_train_timesteps": 1000})
    vocab = {f"tok{i}": i for i in range(TEXT_VOCAB)}
    write_json(model_dir / "tokenizer" / "tokenizer.json", {
        "model": {"vocab": vocab, "merges": ["a b"]},
        "added_tokens": [{"id": 0, "content": "<|endoftext|>"}]})
    write_json(model_dir / "tokenizer" / "tokenizer_config.json", {"pad_token": "<|endoftext|>"})
    return model_dir


def byte_level_vocab() -> dict[str, int]:
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = [byte for byte in range(256) if byte not in printable]
    mapping = {byte: byte for byte in printable} | {byte: 256 + i for i, byte in enumerate(extra)}
    return {chr(mapping[byte]): byte for byte in range(256)}


def transcribe_tokenizer_json() -> dict:
    vocab = byte_level_vocab()
    added = [{"id": len(vocab) + i, "content": token, "special": token != "<think>", "single_word": False,
              "lstrip": False, "rstrip": False, "normalized": False} for i, token in enumerate(TD_ADDED)]
    return {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                              "use_regex": True},
            "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                                                "trim_offsets": True, "use_regex": True},
            "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                      "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
                      "ignore_merges": False, "vocab": vocab, "merges": []}}


def transcribe_encoder_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    prefix = "model.whisper_encoder."
    tensors = {prefix + "conv1.weight": rng.standard_normal((TD_ENC, TD_MELS, 3)),
               prefix + "conv1.bias": rng.standard_normal(TD_ENC),
               prefix + "conv2.weight": rng.standard_normal((TD_ENC, TD_ENC, 3)),
               prefix + "conv2.bias": rng.standard_normal(TD_ENC),
               prefix + "embed_positions.weight": rng.standard_normal((TD_ENC_CTX, TD_ENC)),
               prefix + "layer_norm.weight": np.ones(TD_ENC), prefix + "layer_norm.bias": np.zeros(TD_ENC)}
    layer = prefix + "layers.0."
    for name in ("self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj"):
        tensors[layer + name + ".weight"] = rng.standard_normal((TD_ENC, TD_ENC))
        tensors[layer + name + ".bias"] = rng.standard_normal(TD_ENC)
    tensors[layer + "self_attn.k_proj.weight"] = rng.standard_normal((TD_ENC, TD_ENC))
    for name in ("self_attn_layer_norm", "final_layer_norm"):
        tensors[layer + name + ".weight"] = np.ones(TD_ENC)
        tensors[layer + name + ".bias"] = np.zeros(TD_ENC)
    tensors[layer + "fc1.weight"] = rng.standard_normal((TD_ENC_FF, TD_ENC))
    tensors[layer + "fc1.bias"] = rng.standard_normal(TD_ENC_FF)
    tensors[layer + "fc2.weight"] = rng.standard_normal((TD_ENC, TD_ENC_FF))
    tensors[layer + "fc2.bias"] = rng.standard_normal(TD_ENC)
    return tensors


def transcribe_text_tensors(vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4)
    prefix = "model.language_model."
    layer = prefix + "layers.0."
    q_dim, kv_dim = 2 * TD_HEAD_DIM, TD_HEAD_DIM
    return {prefix + "embed_tokens.weight": rng.standard_normal((vocab_size, TD_TEXT)),
            prefix + "norm.weight": np.ones(TD_TEXT),
            layer + "input_layernorm.weight": np.ones(TD_TEXT),
            layer + "self_attn.q_proj.weight": rng.standard_normal((q_dim, TD_TEXT)),
            layer + "self_attn.k_proj.weight": rng.standard_normal((kv_dim, TD_TEXT)),
            layer + "self_attn.v_proj.weight": rng.standard_normal((kv_dim, TD_TEXT)),
            layer + "self_attn.o_proj.weight": rng.standard_normal((TD_TEXT, q_dim)),
            layer + "self_attn.q_norm.weight": np.ones(TD_HEAD_DIM),
            layer + "self_attn.k_norm.weight": np.ones(TD_HEAD_DIM),
            layer + "post_attention_layernorm.weight": np.ones(TD_TEXT),
            layer + "mlp.gate_proj.weight": rng.standard_normal((TD_TEXT_FF, TD_TEXT)),
            layer + "mlp.up_proj.weight": rng.standard_normal((TD_TEXT_FF, TD_TEXT)),
            layer + "mlp.down_proj.weight": rng.standard_normal((TD_TEXT, TD_TEXT_FF))}


def transcribe_adaptor_tensors() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(5)
    prefix = "model.vq_adaptor.layers."
    return {prefix + "0.weight": rng.standard_normal((TD_TEXT, TD_ENC * TD_MERGE)),
            prefix + "0.bias": rng.standard_normal(TD_TEXT),
            prefix + "2.weight": rng.standard_normal((TD_TEXT, TD_TEXT)),
            prefix + "2.bias": rng.standard_normal(TD_TEXT),
            prefix + "3.weight": np.ones(TD_TEXT), prefix + "3.bias": np.zeros(TD_TEXT)}


def transcribe_tensors(vocab_size: int) -> dict[str, np.ndarray]:
    return {**transcribe_encoder_tensors(), **transcribe_text_tensors(vocab_size), **transcribe_adaptor_tensors()}


def transcribe_config(vocab_size: int) -> dict:
    return {"audio_token_id": 256 + TD_ADDED.index("<|audio_pad|>"), "audio_merge_size": TD_MERGE,
            "adaptor_input_dim": TD_ENC * TD_MERGE,
            "text_config": {"vocab_size": vocab_size, "hidden_size": TD_TEXT, "intermediate_size": TD_TEXT_FF,
                            "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
                            "head_dim": TD_HEAD_DIM, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
                            "rope_theta": 1000000},
            "audio_config": {"num_mel_bins": TD_MELS, "d_model": TD_ENC, "encoder_layers": 1,
                             "encoder_attention_heads": 2, "encoder_ffn_dim": TD_ENC_FF,
                             "max_source_positions": TD_ENC_CTX, "scale_embedding": False}}


def make_transcribe_checkpoint(root: Path) -> Path:
    model_dir = root / "transcribe"
    model_dir.mkdir()
    vocab_size = 256 + len(TD_ADDED)
    write_safetensors(model_dir / "model.safetensors", transcribe_tensors(vocab_size))
    write_json(model_dir / "config.json", transcribe_config(vocab_size))
    write_json(model_dir / "preprocessor_config.json", {
        "sampling_rate": 16000, "n_fft": TD_FFT, "hop_length": 4, "feature_size": TD_MELS,
        "n_samples": 4 * 2 * TD_ENC_CTX, "nb_max_frames": 2 * TD_ENC_CTX})
    write_json(model_dir / "processor_config.json", {"audio_tokens_per_second": 12.5,
                                                     "time_marker_every_seconds": 5, "enable_time_marker": True})
    write_json(model_dir / "generation_config.json", {"max_new_tokens": 64})
    write_json(model_dir / "tokenizer.json", transcribe_tokenizer_json())
    return model_dir


def reader_string(path: Path, key: str) -> str:
    field = gguf.GGUFReader(str(path)).fields[key]
    return bytes(field.parts[field.data[0]]).decode("utf-8")


def check_transcribe(path: Path, model_dir: Path) -> None:
    from tokenizers import Tokenizer

    names = reader_tensors(path)
    expected = 2 + 11 + 7 + 15 + 6 + 1
    assert len(names) == expected, f"transcribe tensor census: {len(names)} != {expected}"
    assert names["enc.conv1.bias"] == (1, TD_ENC), "conv biases become columns"
    assert names["audio.mel_filters"] == (TD_FFT // 2 + 1, TD_MELS), "mel filters are stored bins x mels"
    conv = reader_tensor(path, "enc.conv1.weight")
    assert conv.tensor_type == gguf.GGMLQuantizationType.F16, "conv kernels stay F16 for im2col"
    assert "enc.blk.0.attn_k.bias" not in names, "the key projection has no bias"
    types = [int(value) for value in reader_field(path, "tokenizer.ggml.token_type")]
    first_added = 256
    assert types[0] == int(gguf.TokenType.NORMAL), "byte tokens are normal"
    assert types[first_added] == int(gguf.TokenType.CONTROL), "special added tokens are control"
    assert types[first_added + TD_ADDED.index("<think>")] == int(gguf.TokenType.USER_DEFINED)
    stored = [int(value) for value in reader_field(path, "moss-transcribe.default_prompt_ids")]
    reference = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt = reader_string(path, "moss-transcribe.default_prompt")
    assert stored == reference.encode(prompt, add_special_tokens=False).ids, "prompt ids come from tokenizers"
    assert int(reader_field(path, "moss-transcribe.token.audio_pad")[0]) == first_added + 5
    print("transcribe converter: PASS")


def check_transcribe_quantized(path: Path) -> None:
    tensor = reader_tensor(path, "text.blk.0.attn_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0, "q8_0 quantizes decoder matrices"
    narrow = reader_tensor(path, "enc.blk.0.attn_q.weight")
    assert narrow.tensor_type == gguf.GGMLQuantizationType.F16, "rows shorter than a q8_0 block fall back to f16"
    assert reader_tensor(path, "enc.conv2.weight").tensor_type == gguf.GGMLQuantizationType.F16
    assert reader_tensor(path, "enc.pos_embd").tensor_type == gguf.GGMLQuantizationType.F32
    print("transcribe q8_0 converter: PASS")


def expect_transcribe_failure(model_dir: Path, out: Path, needle: str, label: str) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / "convert-moss-transcribe-to-gguf.py"), str(model_dir),
                             "--outfile", str(out)], capture_output=True, text=True)
    assert result.returncode != 0, f"{label}: converter accepted the checkpoint"
    assert needle in result.stderr, f"{label}: wrong failure: {result.stderr[-400:]}"
    print(f"transcribe converter rejects {label}: PASS")


def run_transcribe_checks(root: Path) -> None:
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("SKIP: tokenizers is not installed; the transcribe converter is untested")
        return
    model_dir = make_transcribe_checkpoint(root)
    out = root / "transcribe.gguf"
    run_converter("convert-moss-transcribe-to-gguf.py", [str(model_dir), "--outtype", "f32", "--outfile", str(out)])
    check_transcribe(out, model_dir)
    quantized = root / "transcribe-q8.gguf"
    run_converter("convert-moss-transcribe-to-gguf.py",
                  [str(model_dir), "--outtype", "q8_0", "--outfile", str(quantized)])
    check_transcribe_quantized(quantized)
    tensors = transcribe_tensors(256 + len(TD_ADDED))
    tensors["model.whisper_encoder.layers.0.unknown.weight"] = np.ones(TD_ENC)
    write_safetensors(model_dir / "model.safetensors", tensors)
    expect_transcribe_failure(model_dir, root / "bad.gguf", "unmapped tensor", "an unmapped tensor")


def reader_tensor(path: Path, name: str):
    reader = gguf.GGUFReader(str(path))
    return next(t for t in reader.tensors if t.name == name)


def reader_field(path: Path, key: str) -> list:
    field = gguf.GGUFReader(str(path)).fields[key]
    return [field.parts[index][0] for index in field.data]


def check_sfx(path: Path) -> None:
    names = reader_tensors(path)
    expected = 2 + 11 + 15 + 27 + 4 + 28 * len(SFX_RATES) + 4
    assert len(names) == expected, f"sfx tensor census: {len(names)} != {expected}"
    assert "text.blk.0.attn_q.weight" in names and "dit.blk.0.cross_norm.bias" in names
    assert not any(name.startswith("lm_head") for name in names), "the unused LM head is dropped"
    assert names["dit.patch_embd.weight"] == (SFX_LATENT, SFX_WIDTH), "patch conv becomes a linear"
    assert names["dit.blk.0.modulation"] == (SFX_WIDTH, 6), "modulation drops its batch axis"
    up_width = SFX_RATES[0] * 2 * (SFX_DECODER_DIM // 2)
    assert names["vae.blk.0.up.weight"] == (SFX_DECODER_DIM, up_width), "transposed conv becomes GEMM columns"
    conv_in = reader_tensor(path, "vae.conv_in.weight")
    assert conv_in.tensor_type == gguf.GGMLQuantizationType.F16, "conv kernels stay F16 for im2col"
    direction = np.arange(1, SFX_DECODER_DIM * SFX_LATENT * SFX_KERNEL + 1, dtype=np.float32)
    assert np.allclose(np.asarray(conv_in.data, dtype=np.float32).reshape(-1), SFX_GAIN * direction,
                       rtol=1e-3), "weight norm folds to g * v / |v|"
    inv = np.asarray(reader_tensor(path, "vae.blk.0.snake.inv").data, dtype=np.float32).reshape(-1)
    assert np.allclose(inv, 2.0, rtol=1e-6), "snake stores 1 / (alpha + eps)"
    assert list(reader_field(path, "moss-sfx.vae.decoder_rates")) == SFX_RATES
    assert int(reader_field(path, "tokenizer.ggml.padding_token_id")[0]) == 0
    print("sound effect converter: PASS")


def check_sfx_half(path: Path) -> None:
    tensor = reader_tensor(path, "dit.blk.0.self_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.F16, "f16 stores linear weights as f16"
    assert reader_tensor(path, "text.output_norm.weight").tensor_type == gguf.GGMLQuantizationType.F32
    print("sound effect f16 converter: PASS")


def move_vae_to_checkpoint(model_dir: Path) -> bool:
    try:
        import torch
    except ImportError:
        print("SKIP: torch is not installed; the .pth VAE path is untested")
        return False
    vae_dir = model_dir / "vae"
    state = {name: torch.from_numpy(np.asarray(tensor, dtype=np.float32)) for name, tensor in sfx_vae_tensors().items()}
    kwargs = {"latent_dim": SFX_LATENT, "decoder_dim": SFX_DECODER_DIM, "decoder_rates": SFX_RATES,
              "sample_rate": 48000, "continuous": True}
    torch.save({"state_dict": state, "metadata": {"kwargs": kwargs}}, vae_dir / "vae.pth")
    (vae_dir / "vae.safetensors").unlink()
    write_json(vae_dir / "config.json", {"latent_dim": SFX_LATENT, "sample_rate": 48000})
    return True


def check_sfx_bfloat(path: Path) -> None:
    tensor = reader_tensor(path, "dit.blk.0.self_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.BF16, "bf16 stores linear weights as bf16"
    print("sound effect bf16 converter: PASS")


def expect_converter_failure(args: list[str], needle: str, label: str) -> None:
    result = subprocess.run([sys.executable, str(SCRIPTS / "convert-moss-sfx-to-gguf.py"), *args],
                            capture_output=True, text=True)
    assert result.returncode != 0, f"{label}: converter accepted the checkpoint"
    assert needle in result.stderr, f"{label}: wrong failure: {result.stderr[-400:]}"
    print(f"sound effect converter rejects {label}: PASS")


def add_unmapped_dit_tensor(model_dir: Path) -> None:
    tensors = sfx_dit_tensors()
    tensors["blocks.0.unknown.weight"] = np.ones(SFX_WIDTH)
    write_safetensors(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", tensors)


def restore_dit_tensors(model_dir: Path) -> None:
    write_safetensors(model_dir / "transformer" / "diffusion_pytorch_model.safetensors", sfx_dit_tensors())


def write_discrete_vae_checkpoint(model_dir: Path) -> None:
    import torch

    state = {name: torch.from_numpy(np.asarray(tensor, dtype=np.float32)) for name, tensor in sfx_vae_tensors().items()}
    torch.save({"state_dict": state, "metadata": {"kwargs": {"decoder_rates": SFX_RATES, "continuous": False}}},
               model_dir / "vae" / "vae.pth")


def check_sfx_quantized(path: Path) -> None:
    tensor = reader_tensor(path, "dit.blk.0.self_q.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0, "q8_0 quantizes linear weights"
    assert reader_tensor(path, "dit.blk.0.self_q.bias").tensor_type == gguf.GGMLQuantizationType.F32
    print("sound effect q8_0 converter: PASS")


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

        sfx_dir = make_sfx_checkpoint(root)
        sfx_out = root / "sfx.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "f32", "--outfile", str(sfx_out)])
        check_sfx(sfx_out)
        sfx_q8 = root / "sfx-q8.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "q8_0", "--outfile", str(sfx_q8)])
        check_sfx_quantized(sfx_q8)
        sfx_f16 = root / "sfx-f16.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "f16", "--outfile", str(sfx_f16)])
        check_sfx_half(sfx_f16)
        sfx_bf16 = root / "sfx-bf16.gguf"
        run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "bf16", "--outfile", str(sfx_bf16)])
        check_sfx_bfloat(sfx_bf16)
        add_unmapped_dit_tensor(sfx_dir)
        expect_converter_failure([str(sfx_dir), "--outfile", str(root / "bad.gguf")], "unmapped DiT tensor",
                                 "an unmapped DiT tensor")
        restore_dit_tensors(sfx_dir)
        if move_vae_to_checkpoint(sfx_dir):
            sfx_pth = root / "sfx-pth.gguf"
            run_converter("convert-moss-sfx-to-gguf.py", [str(sfx_dir), "--outtype", "f32", "--outfile", str(sfx_pth)])
            check_sfx(sfx_pth)
            print("sound effect .pth VAE converter: PASS")
            write_discrete_vae_checkpoint(sfx_dir)
            expect_converter_failure([str(sfx_dir), "--outfile", str(root / "bad.gguf")], "continuous DAC",
                                     "a quantized DAC checkpoint")
        run_transcribe_checks(root)
    print("moss converters: OK")


if __name__ == "__main__":
    main()
