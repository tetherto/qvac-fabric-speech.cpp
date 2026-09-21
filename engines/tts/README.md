# tts-cpp

Native C++17/ggml speech synthesis, voice cloning, and post-synthesis
enhancement for the QVAC speech stack. There is no Python, PyTorch, or ONNX
Runtime dependency after models have been converted to GGUF.

This package exposes six public synthesis engine APIs covering seven synthesis
families and nine model lines: Chatterbox Turbo, Chatterbox Multilingual,
Supertonic 1, 2, and 3, Parler-TTS, CosyVoice3, Audio8, and Pocket TTS. LavaSR is a
speech-enhancement pipeline, not a TTS synthesizer. The API count follows the
installed public headers under `include/tts-cpp/`; the two Chatterbox families
share one engine API, while the Supertonic generations share another.

Pocket TTS has a native CPU engine, streaming addon/public SDK support, and
metadata-only memory preflight. Package rollout and broader platform validation
remain in progress. See the
[implementation status and validation guide](docs/pocket-tts.md).

This directory is the in-tree `engines/tts` package in
[`qvac-fabric-speech.cpp`](../../README.md). It consumes the system
`ggml-speech` package built from
[`qvac-ext-ggml@speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech).
The checkout intentionally starts without a local `ggml/` or `patches/`
overlay; a bundled build stages the speech fork under `engines/tts/ggml` via
`scripts/setup-ggml.sh` (pinned ref, idempotent on re-run). The standalone
[`chatterbox.cpp`](https://github.com/gianni-cor/chatterbox.cpp) repository is
the separate bundled-ggml development path.

## Capabilities

### Supported models and backends

Backends below are the paths validated or explicitly implemented by each
engine, not every backend ggml can compile.

| Model line | Languages | Voice source | Native rate | CPU | Metal | Vulkan | OpenCL | CUDA |
|---|---|---|---:|:---:|:---:|:---:|:---:|:---:|
| Chatterbox Turbo | English | built-in or zero-shot reference WAV/profile | 24 kHz | yes | yes | yes | yes | yes |
| Chatterbox Multilingual | 23 | built-in or zero-shot reference WAV/profile | 24 kHz | yes | yes | yes | yes | yes |
| Supertonic 1 | English | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ Core ML vocoder sidecar: `Engine::vocoder_on_coreml()` reports load status, `SynthesisResult::vocoder_synthesis_backend` the per-call path, ggml fallback otherwise) | yes | yes | yes |
| Supertonic 2 | `en`, `ko`, `es`, `pt`, `fr` | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ Core ML vocoder sidecar: `Engine::vocoder_on_coreml()` reports load status, `SynthesisResult::vocoder_synthesis_backend` the per-call path, ggml fallback otherwise) | yes | yes | yes |
| Supertonic 3 | 31 languages plus `na` | preset or external style tensors/JSON | 44.1 kHz | yes | yes (+ Core ML vocoder sidecar: `Engine::vocoder_on_coreml()` reports load status, `SynthesisResult::vocoder_synthesis_backend` the per-call path, ggml fallback otherwise) | yes | yes | yes |
| Parler-TTS mini/large/Indic | English or 21 Indic languages | natural-language description | 44.1 kHz | yes | yes | yes | yes | yes |
| Fun-CosyVoice3-0.5B | model-advertised multilingual text | baked voice or zero-shot/cross-lingual reference WAV; instruct controls | 24 kHz | yes | yes | yes | yes | yes |
| Audio8-TTS-Preview-0.6B | multilingual checkpoint vocabulary | model voice or zero-shot reference WAV + transcript | 44.1 kHz | yes | yes (+ Core ML codec sidecar: `Engine::codec_on_coreml()` reports load status, `SynthesisResult::codec_synthesis_backend` the per-call path, ggml fallback otherwise) | yes | yes | yes |
| Pocket TTS | English | prepared voice; cloning requires encoder-enabled weights | 24 kHz | yes | no | no | no | no |
| LavaSR denoiser | language agnostic | input PCM | rate preserving | yes | yes | yes | yes | yes |
| LavaSR enhancer | language agnostic | input PCM | 48 kHz | yes | yes | yes | yes | yes |

Chatterbox on CUDA depends on two ggml-cuda fixes carried by
[`qvac-ext-ggml@speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech).
Without the first (`im2col` / `pad` striding the grid Y and Z axes, which are
capped at 65535) a chunk past roughly ten seconds aborts the launch; Chatterbox
splits on sentences but readily emits a thirty-second chunk from ordinary
prose, so long-form input reaches it. Without the second (`conv_transpose_1d`
bounded to the kernel taps that reach each output) the HiFT vocoder is slower
on CUDA than on CPU and dominates the pipeline. With both, Chatterbox Turbo
runs at RTF 0.05 against 0.72 on CPU, and Multilingual at 0.08 against 3.23 --
CPU cannot keep up with real time on Multilingual and CUDA is comfortably
ahead of it.

What that row is based on: `test-s3gen` compares every S3Gen and HiFT stage to
the Python reference dump under an explicit per-stage tolerance and returns
non-zero when one is exceeded. It is registered per backend (`test-s3gen-cuda`,
`test-s3gen-vulkan`), each arm pinning its backend through
`TTS_CPP_GPU_BACKEND` and failing rather than falling back to the CPU.

T3 carries a weaker claim, and the shape of it matters. Its `t3-mtl-ref`
fixture cannot be regenerated -- the dump script pins no conditioning, so a
fresh dump disagrees with the C++ by about 4e-2 on CPU and CUDA alike -- so
there is no reference parity for T3 on any backend. Nor do CPU and GPU agree
token for token: T3 decodes autoregressively, so a single differing argmax
re-rolls the remainder and the two emit different counts for the same input
(55 against 56, 64 against 68, 63 against 69 on the three phrases the arm
uses). `test-chatterbox-parity-mtl-{cuda,vulkan}` therefore asserts what does
hold -- both backends produce audio, and within 1.35x of each other's duration,
which truncation, runaway generation and silent failure all break. T3 on CUDA
is gated at that strength and no more.

Parler on CUDA is covered by the same per-stage reference harnesses as the
other GPU backends, registered per backend as `test-parler-dac-{cuda,vulkan}`
and run with the full Parler suite pinned to each: 15/15 on CUDA and on
Vulkan against the mini-v1 fixtures. The Indic checkpoint remains hub-gated,
so its arm carries the mini/large result, as it always has. The DAC
range-equivalence check now states its real contract: bit-identity on the
CPU, and a measured tolerance on GPU backends, where a ranged decode near a
sequence edge builds a shorter graph than the full decode and a GEMM over a
different shape legitimately reduces in a different order (worst observed:
5.12e-4 on CUDA, under 1e-5 on Vulkan, against a 2e-3 bar; the window
arithmetic errors the check exists to catch are hop-scale).

Audio8 on CUDA required one ggml-cuda fix and a rethink of what its
harnesses measure. The fix: the transpose fast path of the CUDA copy kernel
ignores destination strides, so the V-cache append -- a transposed source
into a strided cache view -- corrupted attention from the first prefill step;
with the guarded kernel the LM's per-step numerics match the Vulkan-vs-CPU
baseline node for node. The rethink: Audio8's networks are expansive enough
that sub-ulp cross-backend rounding grows into large per-element differences,
and its rollouts are free-running, so one differing argmax re-rolls the rest.
The harnesses now gate on observables that are stable across backends -- a
teacher-forced LM trace with per-step numeric bounds and a token-agreement
rate (CPU exact, CUDA 96.9 percent, Vulkan 100), codec codes agreement with
energy-level latent bounds, and rollout audio sanity with correlation
reported for information -- with every bar measured on all three backends.
The same redesign is what fixed test-audio8-codec and test-audio8-lm-vulkan,
which had been failing on unmodified master since the speech ggml moved past
their calibration.

CosyVoice3 supports `f32` weights, `q8_0`/`q4_0` LM and flow weights,
`f16`/`bf16` flow weights, and `f16` HiFT weights. The recommended desktop
GPU combination is a `q8_0` LM, `q8_0` flow, and `f16` HiFT — except on
Metal, where the `f16` flow measured slightly ahead of `q8_0` (the GEMMs
there are compute-bound, not weight-bandwidth-bound), so Apple targets
prefer `q8_0` LM + `f16` flow + `f16` HiFT; on CPU use a
`bf16` flow on AVX512-BF16 hosts and `f16` elsewhere (measured on a 16-core
Zen 5, 16 threads, same pinned 14.8 s utterance: flow+vocoder wall 16.7 s
with f32 weights on a ggml built without tinyBLAS falls to 8.6 s with the
f16 tier and 7.7 s with the bf16 tier on a `GGML_LLAMAFILE=ON` build, the
bundled-ggml default since this change). The optional
`cosyvoice-cli --flow-cut-prompt` flag reduces flow work: only the first DiT
block attends to the voice-prompt frames, and subsequent blocks process the
generated region. It changes reference output and is off by default. See
[CosyVoice3 conversion and usage](docs/cosyvoice3.md) for details and the
unsupported LM `f16` caveat.

CosyVoice3 on Metal takes graph paths the profiler singled out on Apple
GPUs, all gated by the cross-backend and per-backend harnesses: the LM's
single-token decode runs one flash-attention node per layer instead of the
masked matmul/softmax chain (with the all-zeros causal mask elided — the
greedy trajectory stays bit-identical to the CPU's, measured over 1680
consecutive steps), newer LM GGUFs feed it one fused `qkv_proj` matvec, the
DiT's flash attention takes f16 K/V operands, its grouped `conv_pos_embed`
collapses from 64 dispatches per Euler step to one batched im2col + matmul,
and the vocoder's snake activations run as single fused `GGML_OP_SNAKE`
kernels on every backend. Measured against the previous revision on an M3
Ultra with `--greedy`, so both legs decode the identical 550-token trajectory
(22.0 s audio) and every stage is directly comparable, f16 flow + f16 HiFT
throughout: end-to-end RTF 0.214 -> 0.179 (4710 -> 3934 ms, 1.20x, 5.6x real
time), of which LM decode 4.89 -> 3.73 ms/token (1.31x, the flash-attention
step plus the fused `qkv_proj` GGUF), DiT 1463 -> 1359 ms (1.08x) and HiFT
decode 190 -> 163 ms (1.17x). On an M4 the same change moves end-to-end RTF
0.598 -> 0.544 and HiFT decode 838 -> 686 ms (sampled legs, so the
trajectories differ slightly). The remaining DiT time is machine-rate GEMM
and flash attention (13+ TFLOPS measured per op), which is why the f16 tier,
not `q8_0`, is the Metal recommendation: measured on the same machine with
the HiFT tier held at f16, flow f32 -> f16 moves the DiT 1426 -> 1351 ms
while f16 -> `q8_0` moves it back to 1403 ms. The f16 HiFT tier is worth
1.13-1.14x on the vocoder decode there — less than the fused snake above it,
because Metal's `mul_mm` already stages f32 operands as half, so a narrower
weight dtype saves bandwidth rather than arithmetic. The vocoder's host-side
SineGen2 source excitation is threaded over harmonics and the mix arithmetic
on the engine's `n_threads` (the RNG pass stays sequential, so the output is
byte-identical at any thread count): 59.7 -> 34.7 ms on the M3 Ultra and
49.0 -> 34.7 ms on the M4, same pinned trajectories.

CosyVoice3 on CUDA is covered by the same per-stage reference harnesses as
its other GPU backends, each registered per backend --
`test-cosyvoice-{flow,llm,hift,conv1d,frontend,clone}-{cuda,vulkan}` and
`test-s3tokenizer-v3-{cuda,vulkan}[-q8_0]` -- so both desktop backends stay
pinned rather than whichever selection prefers: the full CosyVoice and
s3tokenizer suite passes 27/27 pinned to CUDA and to Vulkan. Greedy long-form
synthesis (67 s of audio, well past the launch-abort threshold the
grid-striding fix removed) produces the identical sample count (1617600) on
CUDA, Vulkan and the CPU -- each measured -- at roughly 12 s wall on either
GPU against 526 s on the CPU. The CUDA column here is engine-level validation; the consumer path --
the published `speech-cpp[cuda]` port and the addon behind it -- ships with
the registry bump that closes this model series, which is when that port and
the qvac workflows exercise it end to end.

Supertonic on CUDA rests on a backend-capability distinction the F16-weight
auto policy now probes: the conv-via-im2col lowerings put the weight in
mul_mat's second operand, and ggml-cuda (and ggml-cpu) accept an F16 weight as
the first operand while rejecting it as the second, where ggml-vulkan accepts
both. Materialising F16 weights on such a backend left the scheduler with a
node no backend claims and aborted the first synthesis. The auto policy
requires both orientations, so CUDA and any future backend in its position
keep F32 weights -- slower but correct, the same trade the policy already
makes on CPU and OpenCL -- while Vulkan and Metal keep the F16 roster. With
that in place all three Supertonic generations synthesize on CUDA at every
shipped tier (f32, f16, q8_0, q4_0), the direct and scheduler paths are
bit-identical per backend (`test-supertonic-sched-equivalence-{cuda,vulkan}`),
and CUDA matches the CPU's output length exactly on the reference sentences.

Where a build carries both CUDA and Vulkan, selection prefers CUDA on NVIDIA:
measured on one binary on an RTX 3090, Chatterbox Turbo is 8.6x faster on CUDA
than on that card's Vulkan adapter. Set `TTS_CPP_GPU_BACKEND` to `cuda`,
`vulkan`, `metal` or `opencl` to pin one for a test arm or a comparison; an
unrecognised value is rejected rather than silently dropping to the CPU.

Chatterbox Multilingual's native tokenizer covers `en, es, fr, de, it, pt, nl,
pl, tr, sv, da, fi, no, el, ms, sw, ar, ko`; `ja`, `he`, `ru`, `zh`, and `hi`
use external preprocessing. Japanese requires MeCab/IPAdic and Chinese requires
a Cangjie5 TSV.

## Performance

End-to-end speed; `RTF = generation_time / audio_duration` (lower is faster).
CI carries the authoritative Linux x86-64 numbers; the multi-machine tables
cover Supertonic 3 and Audio8. Apple-silicon deep dives, the speech-cpp CI
snapshot, streaming latency, and reproduction steps live in
[docs/performance.md](docs/performance.md).

### CI benchmarks (latest `ggml-speech`, Linux x86-64)

End-to-end RTF measured in CI on the `tetherto/qvac` self-hosted runners, using
the q4 GGUFs from the QVAC model registry, 1 warmup + 5 timed runs.
`RTF = generation_time / audio_duration` (lower is faster; RTF is the
backend-comparable metric, wall time is workload-specific).

| Engine                  | CPU RTF | Vulkan RTF | Vulkan wall | Vulkan tok/s |
|-------------------------|--------:|-----------:|------------:|-------------:|
| Chatterbox (Turbo)      |    1.54 |      0.099 |      410 ms |          173 |
| Chatterbox Multilingual |    5.81 |      0.182 |     1036 ms |           77 |
| Supertonic              |   0.113 |      0.018 |       78 ms |          952 |
| Supertonic Multilingual |   0.101 |      0.013 |       84 ms |         1087 |
| Supertonic 3            |   0.225 |      0.029 |      118 ms |          631 |

_Source: workflow run [#31603192731](https://github.com/tetherto/qvac/actions/runs/31603192731)
(2026-08-12), runner `qvac-ubuntu2204-x64-gpu`, GPU **NVIDIA RTX 4000 SFF Ada
Generation** (`backend=vulkan`), benchmarking the published
`@qvac/tts-ggml@0.6.2` addon (released 2026-08-03, pinning `tts-cpp`
2026-08-03#1). This run adds the previously missing Supertonic GPU lanes, so
the Vulkan columns are now recorded for all engines._

### Supertonic 3 multi-machine benchmark (2026-09)

Maintainer-run measurement on four machines. `supertonic-cli` performs one
synthesis per process (it has no repeat flag), so the number a user feels is
the **end-to-end process wall**, model load included, timed from outside the
process. Two text lengths, ~9.6 s and ~26.5 s of speech; f16 weights
(`supertonic3-f16.gguf`, 197 MiB); engine `46afe7d9`, ggml
`speech@157b299f`. Load is the two-point intercept at zero audio,
`wall_short - slope * audio_short`.

| Device | Backend | wall short / long | RTF (long) | load |
|---|---|--:|--:|--:|
| MacBook Air M5 | Metal | 0.66 / 0.65 s | 0.025 | 0.66 s |
| MacBook Air M5 | CPU | 1.72 / 2.66 s | 0.101 | 1.18 s |
| RTX 3080 desktop | CUDA | 0.71 / 0.74 s | 0.028 | 0.69 s |
| RTX 3080 desktop | Vulkan | 0.61 / 0.62 s | 0.024 | 0.61 s |
| Strix Halo | Vulkan | 0.61 / 0.65 s | 0.025 | 0.59 s |
| RTX 5090 box | CUDA | 0.79 / 0.82 s | 0.031 | 0.77 s |
| RTX 5090 box | Vulkan | 0.72 / 0.74 s | 0.028 | 0.70 s |

On the GPU lanes synthesis is cheap enough that 17 s of extra speech costs
less than the run-to-run noise on a 0.6 s process, so the compute-only slope
over text length — `(wall_long - wall_short) / (audio_long - audio_short)` —
is above the noise floor only on the M5 CPU (0.0561 s per audio-second), the
RTX 3080 CUDA (0.0022), Strix Halo Vulkan (0.0022), and the RTX 5090 (CUDA
0.0020, Vulkan 0.0014); start-up dominates the wall everywhere else. One
caveat: `supertonic-cli` caps one batch synthesis at ~28.5 s of audio, which
sets the long text point (3x the short text, not 8x).

### Audio8 multi-machine benchmark (2026-09)

Maintainer-run measurement on three machines. `audio8-cli` performs one
synthesis per process, so the number is the **end-to-end process wall**, model
load included, timed from outside the process. Two text lengths, ~9.6 s and
~24 s of speech; q8_0 weights (`audio8-lm-q8_0.gguf` 800 MiB +
`audio8-codec-decoder-q8_0.gguf` 201 MiB); engine `0f9fc817`, ggml
`speech@157b299f`, seed 42, 16 threads on the Linux boxes and 10 on the Mac.
Load is the two-point intercept at zero audio,
`wall_short - slope * audio_short`.

| Device | Backend | wall short / long | RTF (long) | load |
|---|---|--:|--:|--:|
| RTX 5090 box | CUDA | 2.12 / 4.82 s | 0.203 | 0.48 s |
| RTX 5090 box | Vulkan | 2.22 / 5.12 s | 0.215 | 0.37 s |
| RTX 5090 box | CPU | 8.53 / 19.96 s | 0.840 | 0.46 s |
| Strix Halo | Vulkan | 3.87 / 9.58 s | 0.403 | 0.24 s |
| Strix Halo | CPU | 7.93 / 18.40 s | 0.774 | 0.53 s |
| Mac mini M4 | Metal | 6.76 / 15.43 s | 0.649 | 0.92 s |
| Mac mini M4 | CPU | 10.82 / 26.97 s | 1.134 | ~0 s |

Audio8 is autoregressive, so unlike Supertonic its compute-only slope over
text length is well above the noise floor on every lane
(`(wall_long - wall_short) / (audio_long - audio_short)`): 0.183 s per
audio-second on the RTX 5090 CUDA, 0.200 on its Vulkan, 0.393 on Strix Halo
Vulkan, 0.610 on Mac Metal, and 0.75–1.14 on the CPU lanes. Every GPU lane
runs faster than real time; the Mac mini M4 CPU lane is the only one that does
not (RTF 1.13). Method: 1 untimed warm-up plus 5 timed runs per cell, engines
alternated, 3 s cooldown, worst rep-to-rep spread in any cell 3.2 % (10 %
gate), outputs deterministic per cell and non-silent (peak amplitude
0.47–0.89), device confirmed in every run log (`on CUDA0` / `on Vulkan0` /
`on MTL0`). One build note: the ggml Vulkan build needs the Khronos
SPIRV-Headers include path on hosts without a system copy.

The table is the campaign as measured at engine `0f9fc817`. The RTX 5090 GPU
lanes were re-measured after the decode-loop work and the fast-AR graph cache
(same harness, same protocol, same box): **CUDA 1.31 / 2.67 s, RTF 0.116**,
slope 0.096 s per audio-second, and **Vulkan 1.65 / 3.73 s, RTF 0.157**, slope
0.144. The intercept of the wall-versus-audio fit — what a caller pays before
the first second of speech — is 0.45 s on CUDA and 0.32 s on Vulkan. The CPU
lane and the other machines are unchanged and are not re-stated here.

## Documentation

| Topic | Where |
|---|---|
| APIs, pipelines, per-surface defaults | [docs/api.md](docs/api.md) |
| Voice conditioning (emotion, pace) | [docs/voice-conditioning.md](docs/voice-conditioning.md) |
| Chatterbox | [docs/chatterbox.md](docs/chatterbox.md) |
| Parler-TTS | [docs/parler.md](docs/parler.md) |
| Supertonic | [docs/supertonic.md](docs/supertonic.md) |
| CosyVoice3 | [docs/cosyvoice3.md](docs/cosyvoice3.md) |
| Audio8 | [docs/audio8.md](docs/audio8.md) |
| Pocket TTS | [docs/pocket-tts.md](docs/pocket-tts.md) |
| LavaSR enhancement | [docs/lavasr.md](docs/lavasr.md) |
| Build paths and repository layout | [docs/build.md](docs/build.md) |
| CLIs, weight conversion, end-to-end runs | [docs/cli.md](docs/cli.md) |
| Performance deep dive (Apple silicon, CI, streaming) | [docs/performance.md](docs/performance.md) |
| Fixtures and static validation | [docs/testing.md](docs/testing.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) |
| Voice-clone backward passes | [docs/voiceclone-backward-gap-matrix.md](docs/voiceclone-backward-gap-matrix.md) |

## License

The package code is released under the [MIT License](LICENSE). Models and
conversion-time dependencies retain their own terms; see [NOTICE](NOTICE) for
canonical upstream sources and license identities. This in-tree package does
not bundle ggml.
