# audiogen-cpp

Native [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) and MiniMax-Music3 text-to-music in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): caption and lyrics in, stereo audio out, no Python or PyTorch at inference time.

| Property | Value |
|---|---|
| CMake project | `audiogen-cpp` v0.1.0 |
| Public API | `tts_cpp::acestep::Engine`, `tts_cpp::minimax::Engine` |
| Output | interleaved stereo PCM, model-defined sample rate, `pcm[t * 2 + ch]` |
| Backends | CPU, Vulkan (including Android Mali iGPUs), Metal, OpenCL (validated on Adreno 700+), CUDA; optional Core ML VAE-decoder sidecar on Apple (`AUDIOGEN_COREML`) |
| ggml | requires the `ggml-speech` port for the custom `ggml_snake` and `ggml_col2im_1d` ops |
| Consumed by | the `@qvac/audiogen-ggml` addon in [QVAC](https://github.com/tetherto/qvac) |

## Supported models

| Model | Task | Output | Quantization | Backends |
|---|---|---|---|---|
| ACE-Step v15 (base and turbo DiT variants) | text-to-music, multi-track (lego) stems, editing, LRC timestamps | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0` | CPU, Vulkan, Metal, OpenCL (Adreno 700+), CUDA; optional Core ML VAE-decoder sidecar |
| MiniMax-Music3 | text-to-music | 44.1 kHz stereo | `f16`, `q8_0`; LM, DiT and depth decoder also `q4_k_m` | desktop CPU + GPU (CUDA, Vulkan, Metal) |

On Apple, `AUDIOGEN_COREML=ON` (default `OFF`) adds an optional Core ML
sidecar for the ACE-Step Oobleck VAE decoder, loaded from
`<vae>-decoder.mlmodelc` next to the VAE GGUF (`vae-BF16.gguf` resolves to
`vae-decoder.mlmodelc`). Every DiT variant shares that VAE, so one sidecar
serves them all. It decodes in 64-latent-frame overlapped windows; latents
shorter than one window and any sidecar failure fall back to ggml, and
`ACESTEP_COREML_DISABLE=1` forces ggml. Measured on an M5, it decodes 1.24x
(30 s) to 1.32x (60 s) faster than ggml Metal. MiniMax-Music3 has no Core ML
sidecar. See [docs/backends.md](docs/backends.md#core-ml-vae-decoder-sidecar).

## Performance

The [desktop benchmark workflow](../../.github/workflows/speech-benchmark-desktop.yml)
supports ACE-Step (`acestep`) and MiniMax-Music3 (`minimax`). Set its opt-in
`music_alignment` input to `true` (default `false`) to retain generated WAVs and
score caption adherence with CLAP. The per-family `bench-<family>-<runner>`
artifacts retain generation/scorer logs, binary/model hashes, scorer provenance,
per-run scores, and scored/expected coverage alongside performance results for
14 days. Scores are non-gating diagnostics with no quality pass threshold;
missing models or scoring failures produce null scores and explicit coverage,
without discarding successful generation performance. MiniMax requires
provisioned generator models; unavailable models remain an unavailable result.
See the [music alignment guide](../../scripts/benchmarks/music-alignment.md)
for setup, artifacts, and pilot controls. The diagnostic workload is longer than
the default performance workload, so their timings use different baselines;
CLAP scoring is excluded from generation timing.

For MiniMax, the benchmark preparer downloads pinned Hugging Face sources and
builds a verified GGUF pair independently of S3. Install the separate
`scripts/benchmarks/requirements-minimax.txt` environment and build
`acestep-quantize` alongside `mm3-replay`. Run `scripts/benchmarks/prepare-minimax.py`
with `--quant q4_k_m`, the quantizer, and its ggml libraries; retain its JSON report.
Cold conversion requires 64 GiB available RAM on Linux or physical RAM on macOS,
plus 50 GiB workspace and missing source downloads. Smaller inference hosts can
verify a transferred bundle with `--prepared-dir` instead.

From the repository root, invoke the prepared bundle with:

```sh
MUSIC_ALIGNMENT=1 MINIMAX_PREPARATION_REPORT="$PWD/minimax-preparation.json" \
  scripts/benchmarks/run-family.sh --family minimax --runs 1 --warmup 0 \
  --build-dir build --models-root "$PWD/bench-models" --out artifacts/minimax.json
```

See the [complete preparation commands](../../scripts/benchmarks/music-alignment.md#minimax-model-preparation)
for library discovery, dependency installation, and the desktop workflow's
`minimax_model_dir`, `publish_minimax_models`, and `minimax_artifact_run_id` inputs.

`music-cli` always enables
verbose engine output and prints per-stage wall clock to stderr (`[music-cli]`,
`[acestep-timing]`). Direct library use prints `[acestep-timing]` only when
`EngineOptions::verbose` is enabled; it defaults to `false`.

For a reproducible engine-to-engine measurement against upstream
`acestep.cpp` (`ace-lm` + `ace-synth`, no addon), see
[`benchmarks/comparison/README.md`](benchmarks/comparison/README.md).

### ACE-Step 1.5 multi-machine benchmark (2026-09)

Measured with the repo's own benchmark harness
([`benchmarks/comparison`](benchmarks/comparison/README.md),
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

### speech-cpp CI (2026-09-07, CPU + macOS)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Audio8: audio8-lm-q8_0 | linux | (CPU) | 4906 | — | 1193 |
| Audio8: audio8-lm-q8_0 | macos | (CPU) | 1602 | — | 1208 |
| Lavasr: lavasr-denoiser-f16 | linux | (CPU) | 4102 | 0.684 | 155 |
| Lavasr: lavasr-denoiser-f16 | macos | (CPU) | 2090 | 0.348 | 209 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | linux | (CPU) | 36817 | 9.20 | 1702 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | macos | (CPU) | 6698 | 1.67 | 2222 |

`—` RTF for audio8 (text-driven variable output).

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

## Documentation

| Topic | Where |
|---|---|
| Pipeline, model stages, model setup | [docs/pipeline.md](docs/pipeline.md) |
| Backends, environment overrides, Core ML VAE sidecar | [docs/backends.md](docs/backends.md) |
| Build | [docs/build.md](docs/build.md) |
| Audio editing (repaint, FlowEdit) | [docs/editing.md](docs/editing.md) |
| Command-line tools | [docs/cli.md](docs/cli.md) |
| Tests and parity | [docs/testing.md](docs/testing.md) |
| Engine comparison harness | [benchmarks/comparison/README.md](benchmarks/comparison/README.md) |

## License

`audiogen-cpp` is MIT licensed; see [LICENSE](LICENSE). Model weights and upstream model code carry their own terms, listed in [NOTICE](NOTICE).
