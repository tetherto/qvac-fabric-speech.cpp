# parakeet engine: performance deep dive

Part of the [parakeet engine documentation](../README.md).

## Multi-machine benchmark: method and caveats

Method and caveats:

- **Two-point slope**: each cell runs the clip `R_lo=1` and `R_hi` times in
  one process; `slope = (median_wall(R_hi) - median_wall(R_lo)) / (R_hi -
  R_lo)`. `R_hi` = 11, or 51–101 for the short clip on the MacBook. 3 timed
  reps per point (5 on the MacBook) plus an untimed warm-up, repeats via the
  CLI's native `--bench-runs`, one job at a time, `CUDA_VISIBLE_DEVICES=0`,
  `GGML_VK_VISIBLE_DEVICES=0`, thread count fixed per host (16 / 10 / 8).
- **Backend proven per cell** by `.backend` in the bench JSON.
- **Gate**: worst rep spread at `R_hi` per row — at most 10 % clean; rows
  under proven host contention are withheld. The M5 CPU long row is withheld
  because the measurement session drove the 16 GiB machine into swap, so
  that row would measure the SSD; M5 rows in general carry fanless-throttling
  spread of 7–24 % (three Metal sessions put the short clip at 51 / 61 /
  81 ms), so treat them as directional.
- **Linux CPU lanes are out of scope**: the shipping x86/Strix CPU path has
  a known graph-structure inefficiency, fixed in engine PR #227 / registry
  PR #360, neither of which is in the `46afe7d9` tree measured here. The M5
  CPU short row is kept.
- One build note: ggml needed `#include <cuda/iterator>` in
  `src/ggml-cuda/{top-k,argsort}.cu` to compile under CUDA 13.3 (build fix
  only).

### speech-cpp CI (2026-09-07)

Fresh CPU-baseline snapshot from `speech-benchmark-desktop.yml` on the
hosted-Linux and self-hosted macOS runners (5 timed runs + 1 warmup, `jfk.wav`
fixture, ~11 s).

| Model | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|---:|---:|---:|
| Parakeet CTC 0.6b q8_0 | linux | ggml-cpu | 2111 | 0.192 | 858 |
| Parakeet CTC 0.6b q8_0 | macos | ggml-cpu | 184 | 0.0170 | 914 |
| Whisper base | linux | (CPU) | 1906 | 0.173 | 298 |
| Whisper base | macos | Metal | 584 | 0.0530 | 360 |
| Whisper small | linux | (CPU) | 6122 | 0.557 | 797 |
| Whisper small | macos | Metal | 564 | 0.0510 | 878 |
| Whisper tiny | linux | (CPU) | 1035 | 0.0940 | 182 |
| Whisper tiny | macos | Metal | 565 | 0.0510 | 240 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### TDT decode on CUDA and Metal

The fused decoder ops, the unrolled greedy loop and the per-backend lowerings
were measured against the engine as published before them, on the same
machine in the same session: parakeet-tdt-0.6b-v3, `n_gpu_layers 1`,
`--bench --bench-warmup 10 --bench-runs 20`, the two builds alternated, the
median run time of each build kept. Cells read `before -> after ms (speedup)`.
Ten warmup runs matter on Metal: the first two or three timed runs of a
process are up to 30 % faster than the rest, so a short warmup reports a
transient, not the sustained rate.

| Backend | Quant | 30 s clip | 90 s clip | 306 s clip |
|---|---|---:|---:|---:|
| CUDA, RTX 3080, CUDA graphs on | q8_0 | 80.0 -> 32.0 (2.50x) | 308 -> 119 (2.59x) | 1292 -> 495 (2.61x) |
| CUDA, RTX 3080, CUDA graphs on | q4_0 | 79.4 -> 30.8 (2.58x) | 305 -> 116 (2.63x) | 1275 -> 484 (2.64x) |
| CUDA, RTX 3080, CUDA graphs off | q8_0 | 80.0 -> 34.9 (2.29x) | 308 -> 127 (2.42x) | not measured |
| CUDA, RTX 3080, CUDA graphs off | q4_0 | 79.4 -> 33.5 (2.37x) | 305 -> 125 (2.45x) | not measured |
| Metal, Apple M5 | q8_0 | 518 -> 281 (1.84x) | 2561 -> 1095 (2.34x) | 18130 -> 5531 (3.28x) |
| Metal, Apple M5 | q4_0 | 486 -> 254 (1.91x) | 2416 -> 1020 (2.37x) | 17559 -> 5216 (3.37x) |

The 306 s cells come from an earlier run with two warmups and seven timed
runs, medians over six rounds; that clip's run-to-run drift is under 2 %, so
the shorter warmup does not move it. CUDA graphs are a ggml build option
(`GGML_CUDA_GRAPHS`) that is off by default; the 11 s clip lands between
2.2x and 2.3x on CUDA and around 2x on Metal. The Metal host carried a
1-minute load between 2 and 3 during these runs.

Word error rate on the same 500-utterance LibriSpeech subset, Whisper text
normalisation: CPU 2.22 %, CUDA q8_0 2.21 % and q4_0 2.28 %, Metal q8_0
2.15 % and q4_0 2.24 %. Sortformer, streaming and ASR-plus-diarization
outputs are byte-equal to the previous build on both backends.

### Nemotron 3 Diarization on GPU

Nemotron 3 Diarization runs its whole graph on CUDA, Vulkan, and Metal as one
scheduler split with no CPU fallback. Medians of warm runs of the official
q8_0 GGUF with a prewarmed engine on `diarization-sample-16k.wav` (27.3 s)
and the same clip repeated ten times (273 s). Streaming uses the default live
geometry (1040 ms chunks, 80 ms right context, 264 speaker-cache and 80 FIFO
rows) and feeds 100 ms blocks as fast as the engine accepts them; the time
covers the whole session including `finalize()`.

| Backend | Offline 27.3 s | Offline 273 s | Streaming 27.3 s | Streaming 273 s | Slowest chunk |
|---|---:|---:|---:|---:|---:|
| CPU, Ryzen 9 9950X3D, 32 threads | 118-246 ms | 1169 ms | 2.1 s | 36.3 s | 459 ms |
| CUDA, RTX 5090 | 5.9 ms | 40.4 ms | 91 ms | 1.0 s | 4.1 ms |
| Vulkan, RTX 5090 | 5.9 ms | 59.4 ms | 106 ms | 1.09 s | 8.9 ms |
| Metal, Apple M3 Ultra | 16.2 ms | 199 ms | 333 ms | 3.87 s | 15.7 ms |

The CPU host was shared with other jobs, so its row is indicative only.
Offline probabilities against the NeMo-Speech.cpp reference on the 27.3 s
clip, as maximum absolute error and relative L2 error: CPU 0.023 and 0.0016,
CUDA 0.037 and 0.0021, Vulkan 0.031 and 0.0016, Metal 0.033 and 0.0015. The
test tolerance is 0.04 and 0.003. On the 273 s input, the GPU probabilities
differ from CPU by 0.0005 to 0.0006 on average, and 0.02 % to 0.03 % of
frame decisions at the 0.5 threshold flip. The per-block error stays flat
across the input, so long contexts do not drift. Setting `GGML_PREC_F32` on
the matmuls lowers the CUDA maximum error to 0.033 at a 10 % cost and doubles
the Vulkan time, so the graph keeps the default precision.

The first call compiles GPU kernels. With the NVIDIA shader disk cache
disabled, Vulkan spends 2.3 s on the first offline call and 0.25 s on the
first streaming chunk; CUDA spends 67 ms. `EngineOptions::prewarm` runs one
offline pass of `prewarm_audio_seconds` and one chunk at the default live
geometry during construction, after which the first offline call takes 8 ms
on both backends and no streaming chunk exceeds 5 ms.

Backends without fused flash attention for this graph, such as ggml-opencl
on Adreno, use an unfused attention on the GPU. Forced on the RTX 5090, it
costs 15 % per streaming chunk (CUDA 3.80 -> 4.35 ms, Vulkan 3.90 -> 4.48 ms)
and about 3x on the 273 s offline graph (CUDA 16.6 -> 50.8 ms, Vulkan 28.3 ->
61.1 ms), because its score matrix is quadratic in the input length. Sending
the fused attention to the CPU instead, which is what the scheduler would do
without the fallback, makes a streaming chunk 10x slower (CUDA 38.7 ms,
Vulkan 40.0 ms). The unfused path passes the same NeMo-Speech.cpp reference
check on CPU, CUDA, Vulkan, Metal, and Adreno OpenCL.

On a Samsung Galaxy S25 (Snapdragon 8 Elite, Adreno 830), the same 27.3 s
clip and harness, with the F32 matmul precision Adreno receives:

| Backend | Offline 27.3 s | Streaming 27.3 s | Slowest chunk | Max abs | Rel L2 |
|---|---:|---:|---:|---:|---:|
| CPU, Oryon | 331 ms | 5.7 s | 478 ms | 0.027 | 0.0017 |
| OpenCL, Adreno 830 | 176 ms | 2.3 s | 154 ms | 0.033 | 0.0015 |
| Vulkan, Adreno 830 | 726 ms | 12.4 s | 712 ms | 0.035 | 0.0016 |

Without the F32 request, OpenCL measured 0.037 and 0.0030 and Vulkan 0.042
and 0.0040, at 2-3 % less time. OpenCL runs the unfused attention because
ggml-opencl rejects flash attention on Adreno. With
`GGML_OPENCL_FA_ADRENO=1` it takes 108 ms offline and 86 ms per chunk, but it
fails the test's negative-peak check and changes the streaming segments, so
leave it off.

### Nemotron 3 Diarization on long offline inputs

A single offline graph loses speaker identity on inputs longer than about
two minutes, so offline inputs past the 90 s long-form window run through
the speaker-cache path in 30 s chunks with 1040 ms of left and right
context. Frame-level DER (no collar, optimal speaker mapping, 200 ms
minimum segment) against the fixture RTTMs, RTX 5090 CUDA:

| Input | Single graph | Long-form, 1.04 s chunks | 10 s chunks | 30 s chunks | 60 s chunks |
|---|---:|---:|---:|---:|---:|
| `abcba.wav`, 160.6 s, 3 speakers | 28.3 % | 12.4 % | 26.9 % | 3.7 % | 3.9 % |
| `abcdba.wav`, 191.2 s, 4 speakers | 47.5 % | 23.4 % | 14.5 % | 4.3 % | 4.4 % |

On prefixes of the same files the single graph matches the 30 s long-form
path to within 0.2 points up to 105 s (3.2-4.1 %), slips at 120 s (5.2-5.4 %
against 4.4-4.5 %), and collapses at 140 s (34-36 % against 4.1-5.1 %); the
single graph also fails the re-entry check of `test-sortformer-aosc-speakers`
on both full files. The 90 s auto window keeps single-graph results for short
inputs and leaves a margin below the measured degradation. The long-form
path reaches 3.70 % and 4.36 % on CUDA, Vulkan, Metal, Adreno OpenCL and
Vulkan, and the Snapdragon CPU; the x86 CPU reaches 3.71 % on `abcba` but
8.0 % on `abcdba`, where one near-tie flips a speaker, which is why the tests
gate at 15 %. On the RTX 5090 a
273 s input takes 82 ms long-form against 57 ms for the single graph, and a
one-hour input takes 1.0 s. The engine feeds the long-form session in 30 s
blocks, so its sample ring stays bounded instead of holding a copy of the
whole input.
