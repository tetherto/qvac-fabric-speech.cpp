# parakeet.cpp

Parakeet is a pure C++/ggml implementation of NVIDIA FastConformer ASR,
end-of-turn detection, and Sortformer speaker diarization. Inference requires no
Python, PyTorch, NeMo, or ONNX Runtime. A single `parakeet::Engine` loads CTC,
RNN-T, TDT, EOU, Nemotron, or Sortformer GGUFs and selects the implementation
from GGUF metadata.

## Supported checkpoints

| HF repository | Decoder/task | Mel | `d_model × layers` | Vocab | Parameters | GGUF size | Recorded RTF | Languages and notes |
|---|---|---:|---|---:|---:|---|---|---|
| `nvidia/parakeet-ctc-0.6b` | CTC | 80 | 1024 × 24 | 1024 | 600 M | 697 MiB q8_0 / 1.3 GiB f16 | 0.014–0.046 Metal | English |
| `nvidia/parakeet-ctc-1.1b` | CTC | 80 | 1024 × 42 | 1024 | 1.1 B | 1217 MiB q8_0 | 0.026–0.074 Metal | English |
| `ai4bharat/indic-conformer-600m-multilingual` | CTC-only hybrid export | 80 | 1024 × 24 | 5632 + blank | 600 M | ~701 MiB q8_0 / ~373 MiB q4_0 / 1.3 GiB f16 | 0.008 q8_0 Metal / 0.0019 q8_0 Vulkan | 22 Indic languages; requires `--language` or `EngineOptions::language` |
| `nvidia/parakeet-unified-en-0.6b` | RNN-T | 128 | 1024 x 24 | 1024 | 600 M | 707 MiB q8_0 | 0.004 q8_0 Vulkan / 0.028 q8_0 Metal | English; offline full-context encoder with an optional Core ML sidecar; cache-aware streaming at 80/160/560/1040 ms chunks with 0-1040 ms right context |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT | 128 | 1024 × 24 | 8192 | 600 M | 715 MiB q8_0 / 1.34 GiB f16 | 0.006 q8_0 Metal | About 25 languages, with punctuation and capitalization |
| `nvidia/parakeet-tdt-1.1b` | TDT | 80 | 1024 × 42 | 1024 | 1.1 B | 1225 MiB q8_0 | 0.027–0.079 Metal | English only; no punctuation or capitalization |
| `nvidia/parakeet_realtime_eou_120m-v1` | RNN-T + `<EOU>` | 128 | 512 × 17 | 1027 | 120 M | 246 MiB f16 / 132 MiB q8_0 | 0.0052 Vulkan | English ASR and native end-of-turn token |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | Prompt-conditioned RNN-T | 128 | 1024 × 24 | 13087 | 600 M | ~1.3 GiB f16 | 0.108 CPU | Locale-conditioned ASR; empty language selects `auto`; cache-aware streaming at 80/160/320/560/1120 ms |
| `nvidia/diar_sortformer_4spk-v1` | Sortformer | 80 | 512 × 18 | n/a | 123 M | 263 MiB f16 / 141 MiB q8_0 / 75 MiB q4_0 | 0.0020 Vulkan | Up to four speakers; offline and sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Sortformer | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Streaming-trained; sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Sortformer + AOSC | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Audio-Online Speaker Cache preserves slots across long gaps; Core ML exact-shape batch/AOSC encoder |

TDT 0.6B-v3 and TDT 1.1B are distinct model contracts: only 0.6B-v3 is
multilingual and punctuation/capitalization-aware. Encoder topology, including
causal subsampling, convolution normalization, and chunked-limited attention,
comes from GGUF metadata.

Unified RNN-T uses standard greedy transducer decoding. Offline inference runs
the encoder in full-context mode, and with `PARAKEET_COREML=ON` eligible batch
windows route through the fixed-capacity Core ML sidecar. Mode 2 and
`StreamSession` use the native cache-aware encoder, which builds its own ggml
graph and never uses the fixed-shape sidecar: `StreamingOptions::chunk_ms`
selects the chunk and `right_lookahead_ms` the right context, both snapped down
to the nearest trained value (chunks 80, 160, 560, 1040 ms; right context 0, 80,
160, 240, 320, 560, 1040 ms; the default 1000 ms chunk runs as 560 ms); the
encoder keeps a 5.6 s attention cache plus a 4-frame convolution cache per layer
and only encodes `chunk + right context` new frames per step. The
`left_context_ms` knob is ignored. GGUFs converted before the
`parakeet.unified.*` metadata existed fall back to the published checkpoint
contexts.

Nemotron offline inference uses the GGUF's default 320 ms operating point
(`att_context_size=[56,3]`). `EngineOptions::language` accepts the locale aliases
stored in the GGUF, and an empty value resolves to `auto`. The selected locale is
broadcast as a 128-wide one-hot prompt, concatenated to every encoder frame, and
projected before RNN-T decoding. The cache-aware streaming path incrementally
converts arbitrary PCM bursts to mel frames, maintains bounded 56-frame
attention and 8-frame convolution caches, and matches NeMo at all five supported
operating points. Set `StreamingOptions::chunk_ms` to `80`, `160`, `320`, `560`,
or `1120` to select the corresponding trained right-context configuration.
Both callback streaming and live `StreamSession` input use the native caches;
the sliding-window `left_context_ms` and `right_lookahead_ms` knobs are ignored
for Nemotron.

## Core ML encoder sidecars

Unified RNN-T uses the same fixed-capacity sidecar contract as TDT. Shorter
inputs are zero-padded to the exported capacity, and oversized inputs use the
existing overlapping long-form window plan:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-unified-en-0.6b.q8_0.gguf \
  --n-mel-frames 1501 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/parakeet-unified-en-0.6b-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

Place the compiled directory beside the GGUF as
`parakeet-unified-en-0.6b-encoder.mlmodelc`. Missing or incompatible sidecars
and prediction failures fall back to ggml.

### Nemotron

Nemotron supports a fixed-shape sidecar for offline transcription. Export the
encoder at the intended mel length using the standard name:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

Place the compiled directory beside any quantization of the GGUF using the
corresponding `.mlmodelc` name. Exact-shape offline encoder invocations use
Core ML. Longer offline inputs are tiled into overlapping exact-shape windows
using the trained 56-frame left and 3-frame right attention context; native
cache-aware streaming remains on ggml. Mel subsampling, locale prompt
projection, and RNN-T decoding also remain on ggml.

### Sortformer v2.1

On Apple platforms, `PARAKEET_COREML=ON` supports two optional fixed-shape
sidecars for the tagged `sortformer-streaming-v2.1-aosc` variant. The batch
sidecar accepts mel features and the AOSC sidecar accepts post-subsampling
embeddings plus a validity mask. Mel preprocessing and the Sortformer
transformer/speaker head remain on ggml.

Export and compile both sidecars from the F16 GGUF:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1-encoder.mlpackage \
  --compile-dir engines/parakeet/models

python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1.f16.gguf \
  --bypass-pre-encode --n-encoder-frames 410 \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlpackage \
  --compile-dir engines/parakeet/models
```

Place the compiled directories beside any quantization of the same GGUF as
`diar_streaming_sortformer_4spk-v2.1-encoder.mlmodelc` and
`diar_streaming_sortformer_4spk-v2.1-encoder-bypass-pre-encode.mlmodelc`.
Batch routing requires the exact exported mel-frame count because that graph
has no padding-validity mask; shorter and mismatched inputs use ggml. The AOSC
sidecar accepts up to its masked encoder-frame capacity (410 for the default
cache/FIFO/chunk geometry), while larger or custom geometries use ggml. Missing
or incompatible sidecars and prediction failures fall back to ggml. A bypass
sidecar that fails prediction is quarantined for the lifetime of the engine so
subsequent chunks go directly to ggml. See
[docs/backends.md](docs/backends.md#core-ml-encoder-sidecar) for Unified RNN-T/TDT/EOU/Nemotron
details and runtime controls.

## Performance

`RTF = inference_time / audio_duration`; lower is faster. The latest recorded
Linux x86-64 CI run used q8_0 registry models, one warmup, and five timed runs
on an NVIDIA RTX 4000 SFF Ada:

| Model | CPU RTF | CPU wall | Vulkan RTF | Vulkan wall |
|---|---:|---:|---:|---:|
| CTC | 0.112 | 2256 ms | 0.0022 | 43 ms |
| TDT | 0.130 | 2607 ms | 0.0044 | 88 ms |
| EOU | 0.051 | 1034 ms | 0.0034 | 68 ms |
| Sortformer | 0.046 | 922 ms | 0.0019 | 38 ms |
| Sortformer streaming | 0.032 | 646 ms | 0.0034 | 69 ms |

The same run also covers the self-hosted Apple M4 Mac mini (Metal, q8_0):
CTC 0.0113, TDT 0.0150, EOU 0.0097, Sortformer 0.0061, Sortformer
streaming 0.0070.

Source: [workflow run 31603189415](https://github.com/tetherto/qvac/actions/runs/31603189415),
12 August 2026, runner `qvac-ubuntu2204-x64-gpu`, benchmarking the published
`@qvac/asr-ggml@0.1.1` addon (released 2026-08-03, pinning `parakeet-cpp`
2026-08-03).

### Multi-machine benchmark (2026-09)

Maintainer-run measurement of Parakeet TDT 0.6b v3 across four machines and
seven device-backend lanes, timed from outside the process — the engine's own
timer is not quoted. Compute per transcription is an external two-point
slope, so model load, process start-up, and wav decode cancel out of the
number. Clips are byte-identical on all machines: `jfk.wav` 11.00 s,
`ls90.wav` 98.49 s. Build: engine `46afe7d9`, ggml `speech@157b299f`,
`parakeet-tdt-0.6b-v3.q8_0.gguf` (715 MiB, 100 % Q8_0 body).

| Device | Backend | short 11.0 s | long 98.49 s |
|---|---|--:|--:|
| MacBook Air M5 | Metal | 60.53 ms (RTF 0.0055) | 755.80 ms (RTF 0.0077) |
| MacBook Air M5 | CPU | 445.44 ms (RTF 0.0405) | withheld |
| RTX 3080 desktop | CUDA | 13.49 ms (RTF 0.0012) | 115.51 ms (RTF 0.0012) |
| RTX 3080 desktop | Vulkan | 15.41 ms (RTF 0.0014) | 143.98 ms (RTF 0.0015) |
| Strix Halo | Vulkan | 30.21 ms (RTF 0.0027) | 228.27 ms (RTF 0.0023) |
| RTX 5090 box | CUDA | 8.00 ms (RTF 0.0007) | 58.46 ms (RTF 0.0006) |
| RTX 5090 box | Vulkan | 11.56 ms (RTF 0.0011) | 69.51 ms (RTF 0.0007) |

Peak GPU memory on the long clip (`nvidia-smi` per-process sampling at 5 Hz,
warm run only; RADV and Metal expose no equivalent counter): 2008 MiB on the
RTX 3080 under CUDA and 1748 MiB under Vulkan; 2386 MiB on the RTX 5090
under CUDA and 1935 MiB under Vulkan.

Accuracy: jfk 0.00 % and ls90 0.80 % WER (`compute-wer.py`, `english`
normaliser) on every lane; the statement of record stays the 500-utterance
LibriSpeech run at 2.15–2.22 % across backends and quantisation tiers.

Method, caveats, CI snapshots, and the CUDA/Metal decode
optimization history: [docs/performance.md](docs/performance.md).

## Documentation

| Topic | Where |
|---|---|
| Build modes, CMake options, repository layout | [docs/build.md](docs/build.md) |
| Backends, runtime selection, Core ML encoder sidecar | [docs/backends.md](docs/backends.md) |
| Model download and GGUF conversion | [docs/models.md](docs/models.md) |
| Public C++ API and Sortformer AOSC | [docs/api.md](docs/api.md) |
| CLI, microphone examples, memory-fit preflight | [docs/cli.md](docs/cli.md) |
| Tests and NeMo parity | [docs/testing.md](docs/testing.md) |
| Benchmark method, CI snapshots, decode optimization history | [docs/performance.md](docs/performance.md) |

## License

Code is Apache-2.0. CTC, RNN-T, TDT, and Sortformer weights are CC-BY-4.0 unless
their model card says otherwise. `parakeet_realtime_eou_120m-v1` uses the
NVIDIA Open Model License. Nemotron 3.5 ASR Streaming 0.6B uses OpenMDW-1.1.
No weights are shipped by this repository.
