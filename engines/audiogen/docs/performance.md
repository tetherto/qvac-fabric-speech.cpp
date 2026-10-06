# audiogen engine: performance

Part of the [audiogen documentation](../README.md).

These are dated measurements. Compare timings only with the matching model,
workload, backend, and timing boundary.

## ACE-Step 1.5 multi-machine benchmark (2026-09)

Measured with the repo's own benchmark harness
([`benchmarks/comparison`](../benchmarks/comparison/README.md),
`node run-comparison.js --backend ...`): 10 fixed prompts, 1 warm-up + 3
timed runs each, 5 s cooldown, `failOnGpuFallback: true`. Build: engine
`46afe7d9`, ggml `speech@157b299f`; GGUFs `Qwen3-Embedding-0.6B-Q8_0`,
`acestep-5Hz-lm-0.6B-Q8_0`, `acestep-v15-turbo-Q8_0`, `vae-BF16`, identical
SHA-256 on every host. `gen` is generation compute, `e2e` adds the lazy
stage loads inside `generate()`.

| Device | Backend | gen | e2e | RTF |
|---|---|--:|--:|--:|
| MacBook Air M5 | Metal | 9,016 ms | 9,682 ms | 0.957 |
| RTX 3080 desktop | CUDA | 1,796 ms | 2,146 ms | 0.198 |
| RTX 3080 desktop | Vulkan | 1,580 ms | 1,912 ms | 0.164 |
| Strix Halo | Vulkan | 2,330 ms | 2,570 ms | 0.253 |
| RTX 5090 box | CUDA | 1,338 ms | 1,772 ms | 0.139 |
| RTX 5090 box | Vulkan | 1,435 ms | 1,864 ms | 0.157 |

Vulkan beats CUDA on the 3080 (1,580 vs 1,796 ms) — that is the ACE-Step LM
running on the GPU for every Vulkan device except Mali (PR #234). Output QC
from the harness: no failures and no silent renders in 180 rounds, 10/10
unique WAV hashes per lane, and duration error at most 1.2 s against the
requested song length.

## ACE-Step 1.5 on Snapdragon 8 Elite (2026-10)

Galaxy S25 (Adreno 830, Hexagon v79), `music-cli --dur 30` with
`ACESTEP_KEEP_STAGES=1`, `q8_0` DiT, median of two runs. `generate` is the
`[acestep-timing]` total.

| Placement | DiT | VAE | generate | Speedup vs OpenCL |
|---|--:|--:|--:|--:|
| CPU | 55,213 ms | 64,092 ms | 145,988 ms | 0.61x |
| OpenCL | 19,229 ms | 51,827 ms | 89,657 ms | 1.00x |
| `--backend hexagon` | 12,471 ms | 15,352 ms | 52,740 ms | 1.70x |
| `--backend hexagon --lm-backend opencl` | 11,584 ms | 14,969 ms | 44,155 ms | 2.03x |

Per-stage parity and setup are in [docs/backends.md](backends.md#hexagon-npu).

## MiniMax-Music3 f16 on RTX 5090 (2026-09)

`mm3-replay --mode full` on the RTX 5090 box with the `f16` pair: one fixed
caption, seed 7, warm driver shader cache, in-process pipeline time (the
`[MM3-Pipe]` total, model load excluded). Before is engine `075045a2` on ggml
`speech@8b980299`; after adds the CFG batching, LM flash attention, stacked
projections and LM cache layout described in [docs/pipeline.md](pipeline.md)
and runs on the same ggml plus two CUDA changes: `GGML_PREC_F32` served on
half-precision weights by the f32 matrix-vector kernel
([qvac-ext-ggml#104](https://github.com/tetherto/qvac-ext-ggml/pull/104)) and a
row-bias add folded into the half-precision cuBLAS GEMM's f32 conversion
([qvac-ext-ggml#105](https://github.com/tetherto/qvac-ext-ggml/pull/105)). CUDA 13.3 (CUDA graphs on), Vulkan SDK 1.4.341.1, driver
595.91.07.

| Song | Backend | Before | After | RTF after | Speedup |
|---|---|--:|--:|--:|--:|
| 20 s (500 frames) | CUDA | 15,120 ms | 11,660 ms | 0.58 | 1.30x |
| 20 s (500 frames) | Vulkan | 15,682 ms | 13,247 ms | 0.66 | 1.18x |
| 2 min (3000 frames) | CUDA | 102,289 ms | 73,949 ms | 0.62 | 1.38x |
| 2 min (3000 frames) | Vulkan | 105,393 ms | 84,222 ms | 0.70 | 1.25x |

On the 2-minute song the LM decode step drops from 15.8 to 9.6 ms (CUDA) and
15.7 to 11.3 ms (Vulkan), the depth decoder from 7.3 to 6.1 ms and 8.6 to
7.4 ms per frame, and the flow stage from 29.3 to 23.1 s and 28.5 to 24.2 s.
The LM and depth steps now stream their f16 weights at about 1.55 TB/s on
CUDA, close to what the card sustains, so the AR loop is bandwidth-bound.

The quantized pairs speed up as well (same box and method, 20 s track):

| Pair | Backend | Before | After | Speedup |
|---|---|--:|--:|--:|
| `q8_0` | CUDA | 11,229 ms | 9,070 ms | 1.24x |
| `q8_0` | Vulkan | 13,055 ms | 10,175 ms | 1.28x |
| `q4_k_m` | CUDA | 9,705 ms | 7,505 ms | 1.29x |
| `q4_k_m` | Vulkan | 11,388 ms | 8,504 ms | 1.34x |

## speech-cpp CI (2026-09-07, CPU + macOS)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | linux | (CPU) | 36817 | 9.20 | 1702 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | macos | (CPU) | 6698 | 1.67 | 2222 |


Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).
