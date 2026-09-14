# Performance across modalities and hosts

Part of the [qvac-fabric-speech.cpp documentation](../README.md).

### ASR, end-of-utterance, diarization

CI numbers from the published `@qvac/asr-ggml@0.1.1` addon ([run 31603189415](https://github.com/tetherto/qvac/actions/runs/31603189415), 2026-08-12), `q8_0` GGUFs, 1 warmup plus 5 timed runs, host `qvac-ubuntu2204-x64-gpu` (CPU: Intel Core i5-13500, GPU: NVIDIA RTX 4000 SFF Ada, Vulkan). Full table: [engines/parakeet/README.md](../engines/parakeet/README.md#performance).

| Model | CPU RTF | CPU wall | Vulkan RTF | Vulkan wall |
|---|--:|--:|--:|--:|
| Parakeet CTC | 0.112 | 2256 ms | 0.0022 | 43 ms |
| Parakeet TDT | 0.130 | 2607 ms | 0.0044 | 88 ms |
| Parakeet EOU | 0.051 | 1034 ms | 0.0034 | 68 ms |
| Sortformer | 0.046 | 922 ms | 0.0019 | 38 ms |
| Sortformer streaming | 0.032 | 646 ms | 0.0034 | 69 ms |
| Whisper base | 0.035 | 699 ms | 0.0057 | 117 ms |
| Whisper small | 0.122 | 2453 ms | 0.0098 | 200 ms |

#### speech-cpp CI (2026-09-07, CPU + macOS)

CPU only on Linux; macOS whisper rows run on Metal (`MTL0`).

| Model | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Parakeet CTC 0.6b q8_0 | linux | ggml-cpu | 2111 | 0.192 | 858 |
| Parakeet CTC 0.6b q8_0 | macos | ggml-cpu | 184 | 0.0170 | 914 |
| Whisper tiny | linux | (CPU) | 1035 | 0.0940 | 182 |
| Whisper tiny | macos | Metal | 565 | 0.0510 | 240 |
| Whisper base | linux | (CPU) | 1906 | 0.173 | 298 |
| Whisper base | macos | Metal | 584 | 0.0530 | 360 |
| Whisper small | linux | (CPU) | 6122 | 0.557 | 797 |
| Whisper small | macos | Metal | 564 | 0.0510 | 878 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Text-to-speech

CI numbers from the published `@qvac/tts-ggml@0.6.2` addon ([run 31603192731](https://github.com/tetherto/qvac/actions/runs/31603192731), 2026-08-12), `q4_0` GGUFs, same host. Full table: [engines/tts/README.md](../engines/tts/README.md#performance).

| Model | CPU RTF | Vulkan RTF | Vulkan wall | Vulkan tok/s |
|---|--:|--:|--:|--:|
| Chatterbox Turbo | 1.54 | 0.099 | 410 ms | 173 |
| Chatterbox Multilingual | 5.81 | 0.182 | 1036 ms | 77 |
| Supertonic | 0.113 | 0.018 | 78 ms | 952 |
| Supertonic Multilingual | 0.101 | 0.013 | 84 ms | 1087 |
| Supertonic 3 | 0.225 | 0.029 | 118 ms | 631 |

#### speech-cpp CI (2026-09-07, CPU + macOS)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Chatterbox: chatterbox-t3-turbo-q8_0 | linux | CPU | 8078 | — | 1835 |
| Chatterbox: chatterbox-t3-turbo-q8_0 | macos | CPU | 2626 | — | 1746 |
| Supertonic: supertonic3-q8_0 | linux | (CPU) | 708 | 0.484 | 876 |
| Supertonic: supertonic3-q8_0 | macos | (CPU) | 49 | 0.0340 | 584 |
| Parler: parler-mini-v1-q8_0 | linux | CPU | 249362 | 18.8 | 1964 |
| Parler: parler-mini-v1-q8_0 | macos | MTL0 (Metal) | 12204 | 0.582 | 1307 |
| Cosyvoice: cosyvoice3-llm-q8_0 | linux | CPU | 67695 | 18.0 | 1416 |
| Cosyvoice: cosyvoice3-llm-q8_0 | macos | CPU | 46825 | 12.3 | 1467 |

`—` RTF for chatterbox because it is text-driven variable output.

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Music generation & other engines (speech-cpp CI, 2026-09-07)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | linux | (CPU) | 36817 | 9.20 | 1702 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | macos | (CPU) | 6698 | 1.67 | 2222 |
| Lavasr: lavasr-denoiser-f16 | linux | (CPU) | 4102 | 0.684 | 155 |
| Lavasr: lavasr-denoiser-f16 | macos | (CPU) | 2090 | 0.348 | 209 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Apple silicon

| Model | Host | Backend | Quantization | RTF | vs real-time |
|---|---|---|---|--:|--:|
| Parakeet TDT 0.6b v3 | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.015 | 67x |
| Parakeet CTC | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.011 | 88x |
| Whisper small | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.027 | 37x |
| Chatterbox Turbo | Mac Studio M3 Ultra | Metal | `q4_0` | 0.16 | 6.4x |
| Chatterbox Turbo | Mac Studio M3 Ultra | CPU (NEON) | `q4_0` | 1.05 | 0.96x |
| Chatterbox Multilingual (`--cfm-steps 7`) | Mac Studio M3 Ultra | Metal | `q4_0` | 0.30 | 3.3x |
| Chatterbox Multilingual | Apple M4 | Metal | `q4_0` | 1.37 | 0.73x |

### Streaming latency

Chatterbox on Apple M4 Metal, 317 speech tokens (12.7 s of audio), `--stream-first-chunk-tokens 10 --stream-chunk-tokens 25 --stream-cfm-steps 1`. Full table: [engines/tts/README.md](../engines/tts/docs/performance.md#streaming-mode--low-latency-playback).

| Metric | Value |
|---|--:|
| first audio out | 279 ms |
| steady-state chunk RTF | 0.30 to 0.63 |
| overall RTF | 0.90 |

On-device Android and iOS performance is tracked by the benchmark lanes in [QVAC](https://github.com/tetherto/qvac).
