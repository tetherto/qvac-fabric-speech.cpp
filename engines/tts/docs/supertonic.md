# tts engine: Supertonic

Part of the [tts engine documentation](../README.md).

## Supertonic GGUF

The tree also contains a Supertonic path.  It is
model-specific: the official Supertone ONNX files and assets are converted
into one GGUF, then a ggml C++ runtime runs the known Supertonic stages on
CPU or, with `--n-gpu-layers > 0`, on Metal (Apple), Vulkan, OpenCL (Adreno),
or CUDA when that backend is compiled in.  Metal is the fastest backend
measured so far: on an Apple M5 (10-core GPU) Supertonic 3 `q8_0` renders a
3.1 s / 9.6 s / 16.3 s utterance in `32.5 / 52.8 / 77.0 ms` (`RTF 0.010 /
0.005 / 0.005`, medians of 40 warm runs), 2.2x / 2.0x / 2.0x faster than the
Metal path before the one-graph stages and kernel fusions below, measured back
to back against it; the `f16` GGUF renders the same texts in `31.6 / 51.7 /
75.9 ms`.  Supertonic 2 `q8_0` measured `91.4 ms` total on an M2 (see
[`PROGRESS_SUPERTONIC.md`](../PROGRESS_SUPERTONIC.md)).

There are three related upstream bundles:

- `Supertone/supertonic` is the stable English bundle.  It should be used for
  English and does **not** wrap text in language tags.
- `Supertone/supertonic-2` is the multilingual bundle.  It should use the
  open/close language-tag path (`<lang>...</lang>`).  The older prefix-only
  form (`<lang>... `) can make English prompts stutter.
- `Supertone/supertonic-3` advertises 31 languages plus `na` for unknown source
  language and uses the open/close language-tag path.

Current status:

- `scripts/dump-supertonic-reference.py` dumps ONNX Runtime reference tensors;
  see [Fixtures and static validation](testing.md#fixtures-and-static-validation) for
  the exact v2/v3 artifact layout.
- `scripts/setup-supertonic2.sh` downloads the official Hugging Face bundle
  through `huggingface_hub` and writes the local GGUF.
- `scripts/convert-supertonic2-to-gguf.py` writes `models/supertonic.gguf`
  (English) or `models/supertonic2.gguf` (multilingual), depending on flags.
  When `--onnx-dir` is omitted, it downloads the selected repo just like the
  Chatterbox converters.
- `build/tts-cli` autodetects Supertonic GGUFs from `supertonic.arch` and can
  synthesize a 44.1 kHz wav on CPU or GPU (`--n-gpu-layers`, with
  `--vulkan-device` to pick an adapter), but rejects Supertonic streaming
  flags. `build/supertonic-cli` is the full batch/streaming surface.
- All five stages pass numerical parity against the ONNX reference
  (preprocess, duration, text encoder, vector estimator, vocoder), and the
  full pipeline (`test-supertonic-pipeline`) reproduces the ONNX reference
  waveform when fed the same initial noise tensor.
- The production path is GGML-backed for duration, text encoder, vector
  estimator, and vocoder.  Text relative-position self-attention and FFN blocks
  are expressed with stock GGML ops, and the speech-prompted text attention /
  vector attention blocks use `ggml_flash_attn_ext` where the math allows it.
- Vector attention uses strided Q/K/V GGML views where the time-channel layout
  permits it.  The vector runtime also keeps persistent graph/allocr caches for
  attention, ConvNeXt group, and tail islands, plus fused ConvNeXt boundary /
  tail update graphs and portable custom CPU kernels for pointwise Conv1D,
  depthwise Conv1D, row-wise layer norm, dense time matmul, and fused
  bias/GELU/residual elementwise work.
- The vocoder keeps a persistent GGML graph cache and uses portable
  BLAS/Accelerate-backed causal Conv1D custom ops for the hot projection paths.
  BLAS worker threads are capped by default to avoid nested oversubscription
  under GGML task-level threading.
- Off the CPU backend the duration encoder, the text encoder, the whole CFM
  loop and the vocoder each run as one graph compute, and the
  relative-position bias is a strided view over a padded tensor instead of a
  mask chain.  Where the fused Supertonic kernels exist (Metal) the CFM step
  runs in `[C, T]` layout with classifier-free guidance batched along time;
  other GPU backends keep the per-pass step inside the one graph.  ggml-metal fuses
  the Supertonic depthwise convolution into the following channel layer norm
  and the mat-mat epilogues (bias, bias + residual, bias + GELU, gamma +
  residual) into its mat-mat kernels.  `SUPERTONIC_DISABLE_CT_STEP=1`,
  `SUPERTONIC_DISABLE_TEXT_ONE_GRAPH=1`, `SUPERTONIC_DISABLE_DURATION_ONE_GRAPH=1`,
  `GGML_METAL_FUSION_MM_EPILOGUE_DISABLE=1` and `GGML_METAL_FUSION_DW_LN_DISABLE=1`
  restore the previous paths for A/B runs.
- `SUPERTONIC_VECTOR_PROFILE=1` and `SUPERTONIC_TEXT_PROFILE=1` print
  per-island timings for tuning graph boundaries.  Current text profiling shows
  stock-op relpos is ~0.7-0.8 ms/layer on the quick prompt, so a fused relpos
  op is deferred until backend profiling proves it necessary.
- CPU thread count is controlled by `--threads`.  The default follows the graph
  path the build takes.  Where an Accelerate / CBLAS pointwise path is compiled
  (Apple), the CPU backend runs the per-island path, which regresses when
  oversubscribed, so the default stays at 4 threads.  Where it is not (a Linux
  or Windows build without a pointwise BLAS), the CPU backend runs the same
  fused one-graph path the GPU backends take and the default leaves an eighth of
  the logical CPUs unsubscribed.  Full subscription regresses on both boxes
  measured (Ryzen 9 7950X3D, Ryzen AI MAX+ 395) and the regression survives an
  OpenMP barrier, so it is oversubscription rather than ggml's spin barrier.
  GPU backends keep the 4-thread default: they run only a handful of host ops.
- Current CPU benchmark artifacts live in
  `artifacts/supertonic-thread-matrix/`.  The final matched matrix on this
  machine uses F1, 5 denoise steps, speed `1.05`, `runs=3`, `warmup=1`, and
  ONNX Runtime `CPUExecutionProvider` only.  GGML wins 10 of 12 matched
  thread/prompt comparisons.  The only end-to-end losses are quick English at
  4 threads (`157.7 ms` vs `148.8 ms`) and long English at 4 threads
  (`361.2 ms` vs `351.5 ms`).
- The current quick prompt 4-thread medians are `13.5 ms` text encoder,
  `96.3 ms` vector estimator, `43.6 ms` vocoder, and `157.7 ms` total
  (`RTF 0.050`).  Portuguese 4-thread now wins end to end (`234.3 ms` GGML vs
  `250.8 ms` ONNX), with GGML vocoder at `68.9 ms` vs ONNX `95.6 ms`.

Latest matched CPU matrix, median total milliseconds:

| Prompt | GGML 1t | GGML 2t | GGML 3t | GGML 4t | ONNX 1t | ONNX 2t | ONNX 3t | ONNX 4t |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| quick English | 298.0 | 189.4 | 157.7 | 157.7 | 373.8 | 218.5 | 168.3 | 148.8 |
| longer English | 757.5 | 491.2 | 390.3 | 361.2 | 1103.0 | 580.6 | 555.7 | 351.5 |
| Portuguese smoke | 457.2 | 292.9 | 251.0 | 234.3 | 610.6 | 344.6 | 268.3 | 250.8 |

Example:

```bash
# Stable English bundle: no language wrapping.
bash scripts/setup-supertonic2.sh --arch supertonic

cmake --build build --target tts-cli
./build/tts-cli \
  --model models/supertonic.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --out /tmp/supertonic.wav

# Multilingual bundle: uses the <lang>...</lang> wrapping path.
bash scripts/setup-supertonic2.sh

cmake --build build --target tts-cli
./build/tts-cli \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice M1 --language en --steps 5 --speed 1.05 \
  --out /tmp/supertonic.wav

# Bit-exact reproduction of the ONNX reference run (pass the same noise tensor)
./build/tts-cli --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice M1 --language en --steps 5 --speed 1.05 \
  --noise-npy artifacts/supertonic-ref-quick/noise.npy \
  --out /tmp/supertonic.wav

# Matched GGML benchmark with machine-readable metrics.
./build/supertonic-bench \
  --model models/supertonic2.gguf \
  --text "The quick brown fox jumps over the lazy dog." \
  --voice F1 --language en --steps 5 --speed 1.05 \
  --threads 4 --runs 5 --warmup 1 \
  --json-out artifacts/supertonic-bench.json

# Matched ONNX Runtime benchmark.  Use open_close wrapping for Supertonic 2.
python scripts/bench-supertonic-onnx.py \
  --onnx-dir /path/to/supertonic-pytorch/onnx_models/onnx \
  --assets-dir /path/to/supertonic-pytorch/assets \
  --voice-style /path/to/supertonic-pytorch/assets/voice_styles/F1.json \
  --text "The quick brown fox jumps over the lazy dog." \
  --lang en --language-wrap-mode open_close \
  --steps 5 --speed 1.05 --threads 1 --runs 5 --warmup 1 \
  --json-out artifacts/supertonic-onnx-bench.json
```

## Core ML vocoder sidecar

`TTS_CPP_COREML=ON` is Apple-only. It enables an optional Core ML sidecar for
the **vocoder** -- the latent unpack and denormalize, the embed convolution,
the ten causal ConvNeXt blocks, the pre-baked BatchNorm affine, and the head.
The vocoder is the second-largest stage of a synthesis (about a fifth of a
Metal run, more than a quarter of a CPU one) and the one Supertonic stage with
a fixed convolutional topology and a single variable dimension; the duration
predictor, text encoder, and vector estimator stay on ggml. The earlier
ONNX-Runtime CoreML-EP measurement (see PROGRESS_SUPERTONIC.md) ran the whole
pipeline through shape-generic graphs and lost to ggml everywhere; the sidecar
is a fixed-shape, float16 `jit.trace` export of the vocoder alone, which is
the form Core ML places on the Neural Engine. Export it from the model GGUF:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install -r engines/parakeet/scripts/requirements-coreml.txt
python engines/tts/scripts/export-supertonic-coreml.py \
  --gguf models/supertonic2.gguf --compile-dir models
```

The compiled sidecar must sit next to the model GGUF as
`<basename-minus-quant>-vocoder.mlmodelc`: `supertonic2.gguf`,
`supertonic2-f16.gguf` and `supertonic2-q8_0.gguf` all resolve to
`supertonic2-vocoder.mlmodelc`, so one sidecar serves every tier. Export from
the f32 or f16 tier: a block-quantized source bakes its dequantized vocoder
weights under the same shared name (the exporter warns when that happens). The
exporter rebuilds the vocoder in PyTorch from the GGUF tensors -- replicate
(edge) left padding on every causal convolution, exactly like
`causal_replicate_pad_1d` in the ggml graph -- so it needs neither the ONNX
bundle nor its runtime; `--parity-dir <ref>` checks that rebuild against
`final_latent.npy` / `wav_full.npy` from `dump-supertonic-reference.py`
first. With the sidecar present the engine reports it on
`Engine::vocoder_on_coreml()` at load and per call on
`SynthesisResult::last_vocoder_backend`; `supertonic-bench` prints and emits the
same label as `vocoder_backend`.

The export is fixed-shape, `--window` latent frames (default 64, 4.46 s,
196608 samples) in and the matching samples out. The engine walks an
utterance in windows of exactly that width: every convolution in the vocoder
is causal, so a window's output is exact from the first frame whose receptive
field lies inside it, and the engine drops each window's leading causal
context (20 latent frames for this checkpoint, computed by
`supertonic_coreml_vocoder_context_frames` from the same kernel shapes the
ggml graph uses) just as a mid-utterance ggml frame ignores everything beyond
its receptive field. An utterance shorter than one window is zero-padded on
the right, which changes nothing before the padding because nothing in the
stack looks forward.

Runtime controls, mirroring the Audio8 sidecar: `SUPERTONIC_COREML_DISABLE=1`
skips the sidecar at load; `SUPERTONIC_COREML_COMPUTE_UNITS` in `cpu_only` /
`cpu_and_gpu` / `cpu_and_ane` narrows the requested placement (default: all);
`SUPERTONIC_COREML_STRICT=1` (test-only) fails instead of falling back to the
ggml graph. Any sidecar failure -- missing, malformed, wrong shape, or a
prediction error -- falls back to the ggml vocoder for that call.

`test-supertonic-coreml-path` locks the naming rule without a model;
`test-supertonic-coreml-parity` (staged via `SUPERTONIC_COREML_TEST_MODELS_DIR`)
gates the sidecar against the ggml vocoder at cosine 0.999 on a padded, an
exact-window, and a stitched multi-window latent, checks the
`SUPERTONIC_COREML_DISABLE` fallback, the strict refusal without a sidecar,
the invalid-sidecar fallback, that a CPU-backed engine's pre-warm reaches the
sidecar, and runs one public `Engine` synthesis end to
end; `test-supertonic-coreml-exporter` unit-tests the PyTorch rebuild
(causality, unpack layout, squeezed pointwise re-expansion, receptive field,
stem rule) without a checkpoint.

Measured with `supertonic-bench` (`supertonic2.gguf` f32, voice `M1`, 5 steps,
4 threads, `--n-gpu-layers 999`, 5 runs after 2 warmups, ggml Metal as the
reference for every stage but the vocoder; parity cosine >= 0.99995 in every
cell), vocoder stage median and end-to-end total:

| Machine | Utterance | ggml Metal vocoder | Core ML vocoder (`coreml-all`) | total | real-time |
|---|---|---:|---:|---:|---:|
| Apple M4 (mini) | 3.2 s | 6.6 ms | **2.3 ms (2.9x)** | 30.0 -> 26.0 ms | 107x -> 123x |
| Apple M4 (mini) | 17.9 s | 31.4 ms | **13.2 ms (2.4x)** | 88.3 -> 70.1 ms | 203x -> 255x |
| Apple M3 Ultra | 3.2 s | 3.0 ms | 4.5 ms (0.7x) | 29.3 -> 32.2 ms | 109x -> 100x |
| Apple M3 Ultra | 17.9 s | 8.9 ms | 14.0 ms (0.6x) | 41.4 -> 47.0 ms | 433x -> 381x |

The compute-plan places 164 of the 165 exported ops on the Neural Engine, so
the sidecar's speed is the ANE's: it beats Metal on consumer-class GPUs (the
M4's 10 cores) and loses to workstation-class ones (the M3 Ultra), while
freeing the GPU for the vector estimator either way. Ship the sidecar only
where it wins -- it is presence-driven, so the decision is per-deployment, not
per-build.

Window sizing: sidecar compute scales with total padded frames, so the cost
of a window width is its padding overhead -- the 20-frame causal context
repaid per interior window plus the zero-padded tail of the last window. At
the 257-frame (17.9 s) benchmark length windows 64, 128, and 192 all pad to
the same 384 frames and measure within noise, which says nothing general: at
64 the steady-state overhead is 20/44 (+45%) per interior window, at 128 it
is 20/108 (+19%), at 192 it is 20/172 (+12%). 64 stays the default because
typical utterances and every streamed chunk fit one or two windows, where the
last-window tail dominates and a small window wastes the least; re-tune
`--window` (and re-measure) for workloads dominated by long batch utterances.
On the streaming path each chunk vocodes independently, so a short first
chunk still pays one full window (~2.5 ms on M4) -- a second, smaller
exported window is the natural follow-up if first-chunk latency matters.
