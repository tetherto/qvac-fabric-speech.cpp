"""MiniMax transformer blocks using the same weights and layouts as the native GGUF graphs."""

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CFG_BATCH = 2


def read_block(path, stage, layer):
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(path))
    prefix = f"{'blk' if stage == 'lm' else stage + '.blk'}.{layer}."
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name.startswith(prefix):
            shape = tuple(int(n) for n in reversed(tensor.shape))
            array = dequantize(tensor.data, tensor.tensor_type).reshape(shape)
            tensors[tensor.name[len(prefix):]] = np.array(array, dtype=np.float32, copy=True)
    if not tensors:
        raise ValueError(f"No {stage} block {layer} in {path}")
    metadata = {name: field.contents() for name, field in reader.fields.items()}
    return tensors, metadata


def configuration(metadata, stage):
    prefix = "qwen3" if stage == "lm" else "mm3." + stage
    hidden = int(metadata[prefix + ".embedding_length"])
    heads = int(metadata[prefix + (".attention.head_count" if stage == "lm" else ".head_count")])
    config = {"hidden": hidden, "heads": heads,
              "kv_heads": int(metadata["qwen3.attention.head_count_kv"]) if stage == "lm" else heads,
              "head_dim": int(metadata["qwen3.attention.key_length"]) if stage == "lm" else int(metadata[prefix + ".head_dim"])}
    if stage == "dit":
        config.update(eps=float(metadata[prefix + ".layer_norm_eps"]),
                      rope_dim=int(metadata[prefix + ".rope_dim"]), rope_base=float(metadata[prefix + ".rope_theta"]))
    elif stage == "lm":
        config.update(eps=float(metadata["qwen3.attention.layer_norm_rms_epsilon"]),
                      rope_dim=config["head_dim"], rope_base=float(metadata["qwen3.rope.freq_base"]))
    else:
        config.update(eps=float(metadata[prefix + ".rms_eps"]), rope_dim=0, rope_base=1.0)
    return config


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight


def rotary(x, positions, dim, base):
    frequency = torch.pow(torch.tensor(base), -torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    angles = positions.reshape(1, 1, -1, 1) * frequency
    first, second = x[..., :dim // 2], x[..., dim // 2:dim]
    rotated = torch.cat((first * angles.cos() - second * angles.sin(),
                         first * angles.sin() + second * angles.cos()), dim=-1)
    return torch.cat((rotated, x[..., dim:]), dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, tensors, config, stage):
        super().__init__()
        self.weights = nn.ParameterDict({k.replace(".", "__"): nn.Parameter(torch.tensor(v), requires_grad=False)
                                        for k, v in tensors.items()})
        self.config = config
        self.stage = stage

    def weight(self, name):
        return self.weights[name.replace(".", "__")]

    def linear(self, x, name):
        bias = self.weights.get((name + ".bias").replace(".", "__"))
        if self.stage == "dit" and name == "ffn_in":
            channels = x.transpose(1, 2).unsqueeze(2)
            kernel = self.weight(name + ".weight").unsqueeze(-1).unsqueeze(-1)
            return F.conv2d(channels, kernel, bias).squeeze(2).transpose(1, 2)
        return F.linear(x, self.weight(name + ".weight"), bias)

    def norm(self, x, name):
        if self.stage != "dit":
            return rms_norm(x, self.weight(name + ".weight"), self.config["eps"])
        return F.layer_norm(x, (self.config["hidden"],), self.weight(name + ".weight"),
                            self.weight(name + ".bias"), self.config["eps"])

    def heads(self, x, count):
        return x.reshape(x.shape[0], x.shape[1], count, self.config["head_dim"]).permute(0, 2, 1, 3)

    def projections(self, x, positions):
        if self.stage == "dit":
            q, k, v = self.linear(x, "attn_qkv").chunk(3, dim=-1)
        else:
            q, k, v = (self.linear(x, "attn_" + name) for name in ("q", "k", "v"))
        q, k, v = self.heads(q, self.config["heads"]), self.heads(k, self.config["kv_heads"]), self.heads(v, self.config["kv_heads"])
        if self.stage == "lm":
            q = rms_norm(q, self.weight("attn_q_norm.weight"), self.config["eps"])
            k = rms_norm(k, self.weight("attn_k_norm.weight"), self.config["eps"])
        if self.config["rope_dim"]:
            q = rotary(q, positions, self.config["rope_dim"], self.config["rope_base"])
            k = rotary(k, positions, self.config["rope_dim"], self.config["rope_base"])
        return q, k, v

    def attention(self, q, k, v, mask=None):
        repeat = self.config["heads"] // self.config["kv_heads"]
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.config["head_dim"])
        if mask is not None:
            scores = scores + mask
        value = scores.softmax(-1) @ v
        return value.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], self.config["hidden"])

    def finish(self, hidden, attention):
        hidden = hidden + self.linear(attention, "attn_output")
        norm = self.norm(hidden, "ffn_norm")
        if self.stage == "dit":
            value, gate = self.linear(norm, "ffn_in").chunk(2, dim=-1)
            return hidden + self.linear(value * F.silu(gate), "ffn_out")
        return hidden + self.linear(F.silu(self.linear(norm, "ffn_gate")) * self.linear(norm, "ffn_up"), "ffn_down")

    def forward(self, hidden, positions, keys, values, mask):
        q, k, v = self.projections(self.norm(hidden, "attn_norm"), positions)
        selector = (positions.reshape(-1, 1) == torch.arange(keys.shape[2], dtype=torch.float32)).to(torch.float32)
        keep = (1.0 - selector.sum(0)).reshape(1, 1, -1, 1)
        cache_k, cache_v = k, v
        if self.stage == "lm":
            cache_k = k.to(torch.float16).to(torch.float32)
            cache_v = v.to(torch.float16).to(torch.float32)
        keys = keys * keep + (cache_k.transpose(-2, -1) @ selector).transpose(-2, -1)
        values = values * keep + (cache_v.transpose(-2, -1) @ selector).transpose(-2, -1)
        output = self.finish(hidden, self.attention(q, keys, values, mask))
        return output, k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)


class DitBlock(TransformerBlock):
    def forward(self, hidden, positions):
        q, k, v = self.projections(self.norm(hidden, "attn_norm"), positions)
        return self.finish(hidden, self.attention(q, k, v))
