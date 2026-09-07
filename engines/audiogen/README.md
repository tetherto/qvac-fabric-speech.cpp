# audiogen-cpp

Native [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) and MiniMax-Music3 text-to-music in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): caption and lyrics in, stereo audio out, no Python or PyTorch at inference time.

| Property | Value |
|---|---|
| CMake project | `audiogen-cpp` v0.1.0 |
| Public API | `tts_cpp::acestep::Engine`, `tts_cpp::minimax::Engine` |
| Output | interleaved stereo PCM, model-defined sample rate, `pcm[t * 2 + ch]` |
| Backends | CPU, Vulkan (including Android Mali iGPUs), Metal, OpenCL (validated on Adreno 700+), CUDA |
| ggml | requires the `ggml-speech` port for the custom `ggml_snake` and `ggml_col2im_1d` ops |
| Consumed by | the `@qvac/audiogen-ggml` addon in [QVAC](https://github.com/tetherto/qvac) |

## Pipeline

```
caption + lyrics (+ bpm, key, time signature, language)
   |
   v
LM (acestep-5Hz-lm, Qwen3 causal) -> metadata + acoustic codes
FSQ detokenizer                    -> DiT context latents
text encoder (Qwen3-Embedding)     -> prompt embeddings
condition encoder                  -> cross-attention states
DiT (flow matching, Euler)         -> 64-channel acoustic latent
VAE (AutoencoderOobleck)           -> 48 kHz stereo PCM
```

The VAE upsamples by 1920, so `T_audio = T_latent * 1920`. Latents are time-major, `latent[t * 64 + c]`.

### MiniMax-Music3

```
caption + lyrics -> byte-exact Qwen prompt -> Qwen3 LM
                 -> RVQ depth decoder -> condition encoder
                 -> windowed flow DiT -> DAC vocoder -> 44.1 kHz stereo PCM
```

MiniMax uses two GGUF files: `mm3-lm-<quant>.gguf` for the Qwen3 global LM and
`mm3-synth-<quant>.gguf` for the RVQ depth decoder, condition encoder, flow DiT,
and vocoder. Set `EngineOptions::model_dir`, or provide `lm_model_path` and
`synth_model_path` explicitly. Directory discovery matches quantized pairs
case-insensitively, prefers `q8_0`, then `f16`, then `bf16`, then the k-quant
variants `acestep-quantize` can emit in descending fidelity (`q6_k`, `q5_k_m`,
`q5_k_s`, `q4_k_m`, `q4_k_s`, `q3_k_l`, `q3_k_m`, `q3_k_s`, `q2_k`), then
`f32` (a dequantized diagnostic pair, only picked when nothing else matches),
and rejects duplicate candidates. The engine is desktop-only.
It runs on CPU by default; `EngineOptions::device` (or, when that is empty,
the `MM3_DEVICE` environment variable) accepts `cpu`, `gpu`, or `auto`. `gpu`
requires a usable GPU backend and fails engine creation without one; `auto`
takes a GPU when available and otherwise falls back to the CPU. On a GPU the
weights and graphs live on the first usable backend the ggml registry offers
(CUDA, Vulkan, Metal, ...) and a CPU backend backs any unsupported op; the
full model pair must fit in device memory (~22 GB for the f16 pair, 12.7 GB
for `q8_0`, 7.6 GB for `q4_k_m`). The LM's KV cache and compute buffers are
released after the AR stage and rebuilt on the next generation, and the
vocoder decodes in overlapped tiles whose interiors are bit-identical to a
single-shot decode, so the `q4_k_m` pair completes full generations on a
10 GiB GPU (RTX 3080, peak 9.4 GiB alongside a desktop) and on a 16 GB
Apple-silicon Mac over Metal (peak RSS 8.4 GiB), both of which the `q8_0`
pair cannot fit. The flow DiT runs with flash attention by default on a GPU
backend (off on CPU); set `MM3_DIT_NO_FLASH=1` to force it off. The LM keeps
non-flash attention by default because flash attention drifts its sampled
logits; set `MM3_LM_FLASH=1` to opt it in on a GPU, or `MM3_LM_NO_FLASH=1` to
force it off (this wins if both are set). One MiniMax engine instance may be
active at a time because its compute graphs are shared. The weight-free
`test-minimax-metal-ops` regression compares the 4096-channel condition
projection and every DAC transposed-convolution stride against CPU on Metal.

Vulkan and CUDA are the measured GPU backends at `q8_0`/`f16`, both on an RTX
5090. On each, `mm3-replay --mode replay` forcing the recorded official prompt
tokens, codes, and noise reproduces the CPU rendering (time-domain audio
correlation 0.9998 on Vulkan, 0.9993 on CUDA), `--mode condcheck` confirms
byte-identical DiT velocities across repeated computes, and
`test-minimax-quality` lands the final flow latents on the learned manifold.
Vulkan is measured with both the `q8_0` and the `f16` pair; the 22 GB `f16`
pair far exceeds ggml's 1 GiB Vulkan suballocation block and loads through the
chunked device buffers ggml creates automatically. The full
`test-backend-ops` suite passes on the same device, including the
`b_absmax=1e5` LM-shaped `mul_mat` stress cases. The `q4_k_m` pair is
measured on CPU, Metal (M5), Vulkan (Strix Halo RADV and RTX 3080), and CUDA
(RTX 3080): teacher-forced replays of the same inputs agree with the
F32-dequantized reference at 0.97 waveform correlation on every backend
(the `q8_0` pair's own figure on the same probe is 0.9995), condcheck stays
byte-identical, and `q4_k_m` generates faster than `q8_0` (Strix Vulkan
94.7 s vs 105.5 s, CPU 397 s vs 471 s for the same 12 s clip). OpenCL
remains unmeasured for MiniMax and takes the same all-on-GPU placement, so
measure it the same way before shipping it.

The frame rate, maximum frame count, flow defaults, and output sample rate come
from GGUF metadata. Current converted files specify 25 frames per second, at
most 9000 frames, 30 flow steps, CFG 1.7, and 44100 Hz output.

Download the Comfy-Org single-file safetensors checkpoint (the converter's
preferred source; requires the Hugging Face CLI,
`pip install -U huggingface_hub`) and convert it:

```sh
scripts/download-minimax-music3.sh --dir checkpoints/MiniMax-Music3
python3 scripts/convert-minimax-music3-to-gguf.py \
  --src checkpoints/MiniMax-Music3 --out models/minimax --quant f16
```

An already-downloaded checkpoint converts the same way: point `--src` at the
local directory. The converter emits the two-file contract and writes the
MiniMax-Music3 Community License identifier into both files. Ship the upstream
model license with converted weights.

For a smaller footprint, quantize the `f16` pair to `Q4_K_M` (or `Q4_K_S`) with
`acestep-quantize` (built with `AUDIOGEN_BUILD_EXECUTABLES`, shared with the
ACE-Step engine):

```sh
./build/engines/audiogen/acestep-quantize models/minimax/mm3-lm-f16.gguf \
                                          models/minimax/mm3-lm-q4_k_m.gguf Q4_K_M
./build/engines/audiogen/acestep-quantize models/minimax/mm3-synth-f16.gguf \
                                          models/minimax/mm3-synth-q4_k_m.gguf Q4_K_M
```

This quantizes the LM and, within the synth file, the flow DiT to the chosen
k-quant while the RVQ depth decoder is held at `q8_0` (its per-frame matvec
graphs need the integer fast path; F16 weights are several times slower on
scalar-fp16 Vulkan devices), except `depth.pos_embd.weight`, which the depth
graph views raw and which stays F32. The condition encoder and vocoder keep their
converted (F16/F32) precision. Quantize from the `f16` pair, not `q8_0` —
`acestep-quantize` only requantizes BF16/F16/F32 source tensors, so an
already-`q8_0` tensor passes through untouched.

`mm3-replay` (built with `AUDIOGEN_BUILD_EXECUTABLES`) is the MiniMax CLI and
parity harness:

```sh
./build/engines/audiogen/mm3-replay --models models/minimax --out mm3-out --mode full \
                                    --caption "warm lo-fi beat with vinyl crackle" \
                                    --lyrics "[Instrumental]" --max-frames 300
```

`--mode full` runs the pipeline from a caption/lyrics pair and records the
prompt token ids it used (`tokens.i32`) next to the emitted semantic/acoustic
codes, so a full run's output directory doubles as a replay input set.
`--mode replay` forces recorded prompt tokens, semantic/acoustic codes, and
per-window initial noise through the native pipeline and dumps the per-window
latents, frame hiddens, and stitched audio for 1:1 comparison against the
official implementation, and `--mode condcheck` verifies the DiT emits
byte-identical velocities across repeated computes. `--dump-iters N` (optionally with
`--dump-dir <dir>`) additionally writes the first N AR iterations' semantic
logits, CFG-guided logits, LM hidden states, feedback embeddings, and depth
hiddens as raw f32 files; because the LM and depth decoder still run under
forced tokens, replay-mode dumps are teacher-forced and directly comparable
across quantization levels and backends. `test-minimax-quality` (built with
`AUDIOGEN_BUILD_TESTS`, skipped unless `AUDIOGEN_TEST_MINIMAX_MODELS_DIR` is
set) is the model-backed regression: it asserts DiT determinism and that a
short generation's final flow latents land on the learned data manifold
instead of stalling near the Gaussian noise they started from.
`test-minimax-quality-q4` runs the same checks against the pair named by
`AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR`.

### ACE-Step audio editing

When `GenerateParams::edit_plan` is non-empty, the engine takes the editing
path instead: it VAE-encodes `source_audio`, skips the LM and FSQ
detokenizer, executes each `RepaintParams` or `FlowEditParams` operation in
order, and VAE-decodes the final latent.

### ACE-Step LRC generation

With `GenerateParams::generate_lrc` set, the engine aligns the request lyrics
with the generated audio and returns karaoke-style LRC timestamps in
`GenerateResult::metadata.lrc`, plus an alignment confidence score in
`metadata.lyrics_score` (`[0, 1]`). After sampling, one extra DiT forward at
the final timestep runs with explicit softmax on the validated lyric
cross-attention heads (the graph stops at the deepest captured layer), the
captured matrices are converted from the ggml column-major layout, and DTW
aligns each lyric line with the audio timeline. Requires lyrics — with Simple
Mode the LM-written lyrics are aligned — and the official 24-layer/16-head
DiT; the capture path is fully separate from the sampling graph, so normal
inference is untouched when disabled.

### ACE-Step Simple Mode

With `GenerateParams::simple_mode` set, `caption` is a short natural-language
query ("a romantic modern salsa with male lead vocals for a wedding") and the
LM inspire pass composes the full request before synthesis: a detailed
caption, complete lyrics, and every metadata field left unset — BPM,
key/scale, time signature, vocal language, and duration when `duration <= 0`.
Fields the caller sets are forced through the metadata FSM and kept. `lyrics`
must be empty (the LM writes them) or `[Instrumental]`, which forwards the
instrumental hint to the LM. Simple Mode requires the plain `text2music` task
with no pre-supplied `audio_codes`; the composed request is reported back in
`GenerateResult::metadata`. The inspire pass emits `lm` progress ticks and
honors cancellation like every other stage.

### ACE-Step Query Rewriting

With `GenerateParams::rewrite_query` set, the LM FORMAT pass reworks a full
request before synthesis: the caption is rewritten into a detailed musical
description and the lyrics are regenerated preserving their content, with any
unset metadata filled through the same FSM the inspire pass uses. Unlike
Simple Mode — which expands a bare query and writes lyrics from scratch —
Query Rewriting takes caption AND lyrics as input, so both are required, and
the lyrics must be real lyric text: an `"[Instrumental]"` request belongs to
Simple Mode, which forwards the instrumental hint. The two modes are mutually
exclusive. The rewritten request is reported back in
`GenerateResult::metadata`. Faithful rewriting needs the 1.7B LM
(`acestep-5Hz-lm-1.7B`): the 0.6B drifts genre, voice, and language.

### ACE-Step quality scoring

With `GenerateParams::compute_quality_score` set, the generated audio codes
are teacher-forced back through the LM "understand" prompt and the request is
scored: caption and lyrics earn a normalized PMI (their mean log-prob given
the codes against the same text under a no-input prompt) and each set
metadata field earns a rank-weighted top-k recall of its YAML line. The
weight-normalized global score (caption 0.5, lyrics 0.3, metadata 0.2) lands
in `GenerateResult::metadata.quality_score` in `[0, 1]` with the per-condition
breakdown in `quality_report` — made for ranking a batch of takes (stem tasks
on the base DiT vary strongly by seed) and keeping the best. Scoring runs
extra LM forwards after code generation, reports `score` progress ticks, and
honors cancellation; it requires the LM code path, so cover / lego tasks and
the audio edit path reject it.

### ACE-Step audio understanding

`Engine::understand` runs the reverse pipeline: 48 kHz stereo audio is
VAE-encoded, the FSQ tokenizer (an attention pooler in the DiT GGUF under
`tokenizer.*`) turns the latents into 5 Hz semantic codes, and the LM
"listener" describes them — a descriptive caption plus BPM, key/scale, time
signature, duration estimate, and vocal language, decoded through the same
FSM-constrained metadata block generation uses. `UnderstandResult` also
returns the recovered codes, directly reusable as
`GenerateParams::audio_codes`. Lyrics are intentionally NOT reported: the
LM's transcription hallucinates on real songs, so the field is unsupported.
An optional `vocal_language` hint is forced through the FSM instead of the
LM's guess. Stages report as `source`, `tok`, and `understand` (unknown
total), all cancellable.

## Model stages

Four GGUF files provide six runtime weight sets. The DiT GGUF contains the
condition encoder and FSQ detokenizer weights as well as the DiT weights; the
other files provide the text encoder, LM, and VAE. Point `--models <dir>` at a
directory and the engine classifies files by filename stem, or pass explicit
per-file paths, which always win over the scan.

| Stage | Upstream model | Filename stems matched | Flag |
|---|---|---|---|
| Text encoder | Qwen3-Embedding | `embedding`, `text-enc`, `textenc` | `--text` |
| LM | ACE-Step 5 Hz LM (Qwen3 causal) | `-lm`, `lm-`, `_lm`, `ace-lm`, `5hz-lm` | `--lm` |
| DiT + condition encoder + FSQ detokenizer | ACE-Step v1.5 diffusion transformer | `turbo`, `dit`, `v15`, `sft` | `--dit` |
| VAE | AutoencoderOobleck | `vae` | `--vae` |

The most specific stems (`embedding`, `vae`) are tested first so no short token such as `lm` can claim an unrelated file.

### DiT variants

Detected from the `acestep.is_turbo` GGUF key (absent means base or sft) and
`general.name` (an "sft" substring marks the sft fine-tune).

| Variant | Default steps | Default shift | Default guidance | Notes |
|---|--:|--:|--:|---|
| turbo | 8 | 3.0 | 1.0 | fastest; guidance-distilled, CFG overrides are clamped to 1.0 |
| base | 50 | 1.0 | 7.0 | CFG via APG; the only variant supporting the lego stem task |
| sft | 50 | 1.0 | 7.0 | CFG via APG; no stem tasks |

### Multi-Track (lego)

`task_type: "lego"` generates a new instrument layer that follows
`source_audio` and returns only that stem, trimmed to the source length for
sample-for-sample mixing. It requires a base DiT (turbo and sft are rejected)
and a `track` name — one of vocals, backing_vocals, drums, bass, guitar,
keyboard, percussion, strings, synth, fx, brass, woodwinds:

```bash
music-cli --models <dir> --task lego --track guitar \
  --caption "clean electric guitar with syncopated fills" \
  --lyrics "[Instrumental]" --src-audio song.wav --out stem.wav
```

`--guidance F` overrides the DiT guidance scale on base/sft (0 = auto).

Weights load quantized. `f32`, `f16`, and `bf16` are handled for norms and biases, and the detokenizer's `special_tokens` may be `q8_0`. The stage GGUFs are built in tree by `scripts/convert-acestep-to-gguf.py` and the `acestep-quantize` binary (see [Model setup](#model-setup)), both adapted from [acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp), the upstream C++/ggml implementation this port follows, which also publishes pre-quantized GGUFs at [Serveurperso/ACE-Step-1.5-GGUF](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF).

### Model setup

Three four-file combinations are validated. All use the same fixed text
encoder, LM, and VAE; select one DiT:

| Role | Filename | Quantization | Approximate size | Variants |
|---|---|---|---:|---|
| Text encoder | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `q8_0` | 748 MB | all |
| LM | `acestep-5Hz-lm-0.6B-Q8_0.gguf` | `q8_0` | 677 MB | all |
| VAE | `vae-BF16.gguf` | `bf16` | 322 MB | all |
| DiT | `acestep-v15-turbo-Q4_K_M.gguf` | `q4_k_m` | 1.35 GB | `turbo-q4` |
| DiT | `acestep-v15-turbo-Q8_0.gguf` | `q8_0` | 2.37 GB | `turbo-q8` |
| DiT | `acestep-v15-sft-Q8_0.gguf` | `q8_0` | 2.37 GB | `sft` |

Build these files locally in three steps: download the upstream safetensors
checkpoints from Hugging Face, convert each stage to a self-contained BF16
GGUF, and quantize the text encoder, LM, and DiT (the VAE always stays BF16).
The download script needs the Hugging Face CLI
(`pip install -U huggingface_hub`), the converter needs
`pip install numpy gguf`, and `acestep-quantize` builds with
`AUDIOGEN_BUILD_EXECUTABLES`. When built from the repository root, binaries
live under `./build/engines/audiogen/`; a standalone `engines/audiogen` build
places them under `./build/audiogen/` instead.

```sh
scripts/download-acestep-checkpoints.sh --dir checkpoints
python3 scripts/convert-acestep-to-gguf.py --checkpoints checkpoints --out models/bf16

mkdir -p models/acestep
./build/engines/audiogen/acestep-quantize models/bf16/Qwen3-Embedding-0.6B-BF16.gguf \
                                        models/acestep/Qwen3-Embedding-0.6B-Q8_0.gguf Q8_0
./build/engines/audiogen/acestep-quantize models/bf16/acestep-5Hz-lm-0.6B-BF16.gguf \
                                        models/acestep/acestep-5Hz-lm-0.6B-Q8_0.gguf Q8_0
./build/engines/audiogen/acestep-quantize models/bf16/acestep-v15-turbo-BF16.gguf \
                                        models/acestep/acestep-v15-turbo-Q4_K_M.gguf Q4_K_M
cp models/bf16/vae-BF16.gguf models/acestep/
```

Quantize the turbo DiT to `Q8_0` instead for the `turbo-q8` combination; pass
`--sft` to the download script and quantize `acestep-v15-sft-BF16.gguf` to
`Q8_0` for the `sft` one. Keep one DiT per model directory rather than relying
on `--models` alone: directory iteration does not define which matching DiT is
selected, so with several present pass the intended file explicitly with
`--dit` / `EngineOptions::dit_model_path`. The external
`Serveurperso/ACE-Step-1.5-GGUF` files above have separate provenance and are
not the combinations validated by QVAC.

## Backends

`n_gpu_layers > 0` (`--gpu` on the CLI) selects a GPU backend through the ggml
registry. Selection tries Adreno 700+ OpenCL first; validated
Vulkan/Metal/CUDA discrete devices, then validated integrated devices; and
finally other discrete, then integrated GPU backends. Integrated-device
support is required because Vulkan reports Android UMA adapters such as
Pixel's Mali-G715 as `IGPU`.

| Stage | Placement when a GPU is selected |
|---|---|
| DiT, VAE | GPU |
| Text encoder, condition encoder | GPU, unless `ACESTEP_ENCODERS_CPU` is present |
| LM | GPU on Metal, OpenCL, CUDA, and Vulkan Mesa RADV devices; CPU on other Vulkan devices and every unmeasured backend |
| FSQ detokenizer | GPU on Vulkan, Metal, OpenCL, and CUDA; CPU on every unmeasured backend |

The LM and detokenizer use an allowlist rather than a denylist: a backend keeps the CPU placement until the stage has been measured on it, so adding one cannot silently regress generated audio. Metal and OpenCL are validated for both stages; the recorded OpenCL validation used an Adreno 740. Vulkan is validated for the detokenizer everywhere, and for the autoregressive LM per device (`vulkan_device_lm_validated`): Mesa RADV devices run the LM on the GPU, while every other Vulkan device keeps the CPU placement, because Mali-G715 testing showed code collapse and early termination on Vulkan. CUDA needs the `snake` / `col2im_1d` VAE kernels from the `ggml-speech` fork's CUDA backend and its `GGML_PREC_F32` MUL_MAT support (the LM-shaped q4_0/q4_K strided-B `b_absmax=1e5` stress cases produced NaN before the fork routed explicit-precision matmuls to the f32 cuBLAS path); with that ggml, `test-backend-ops` on an RTX 5090 passes the full suite, the Q8_0 LM matches the F32-dequantized reference at 0.9999 logit cosine with an identical greedy trajectory where the CPU Q8_0 path sits at 0.994, and `--quantized-batch-cfg-regression` passes, so the LM runs on the GPU. The HIP/MUSA builds of the same backend register as `ROCm`/`MUSA` and stay off the allowlist until measured.

Measurement is against an F32-dequantized reference (`scripts/dequant_gguf.py`), not against CPU. CPU is not automatically ground truth for a quantized model — ggml's CPU matmul quantizes activations to Q8_1 internally. On Metal the LM reproduces the F32 argmax trajectory exactly where CPU Q8_0 diverges at the first token. The RADV validation (AMD Strix Halo, Radeon 8060S) followed the same protocol: on 300-token prefills the Vulkan Q8_0 LM matches the F32 reference argmax on 3/3 probes with logit cosine >= 0.99999995, where CPU Q8_0 matches on 1/3 at cosine ~0.9998, and the GPU LM stage runs ~2x faster than the CPU path on that device. On Mali Vulkan, however, the LM produces repeated semantic codes and can terminate at roughly half the requested duration, while the same device produces a diverse full-length sequence with the LM on CPU.

On a GPU that supports F32 flash attention, Phase 2 also decodes the conditional and unconditional CFG paths in one batched graph. This matches the reference `acestep.cpp` LM path: the same prompt, model, sampler settings, and seed produce the same semantic-code sequence. Unsupported backends keep the separate F32 manual-attention path.

`lm-smoke --gpu --quantized-batch-cfg-regression --model <Q4-or-Q8-LM.gguf>` compares the compact batched head against two full-vocabulary decode streams. It requires quantized tied embeddings and fails if either stream changes argmax or drops below `0.99999` logit cosine.

The policy itself lives in [`src/acestep/stage_placement.h`](src/acestep/stage_placement.h), separate from the engine, so it is unit tested without a GPU.

### Environment overrides

Applied after the allowlist. The LM, detokenizer, and encoder overrides use
presence semantics: even a value of `0` activates them, and CPU wins when both
CPU and GPU are present for one stage. `ACESTEP_VAE_GPU` checks its first
character: `1` selects GPU and every other present value selects CPU.

| Variable | Effect |
|---|---|
| `ACESTEP_LM_GPU` / `ACESTEP_LM_CPU` | presence forces the LM onto the GPU or CPU |
| `ACESTEP_DETOK_GPU` / `ACESTEP_DETOK_CPU` | presence forces the detokenizer onto the GPU or CPU |
| `ACESTEP_ENCODERS_CPU` | presence moves the encoders to the CPU to trim wired memory |
| `ACESTEP_VAE_GPU` | a value beginning with `1` forces GPU; any other present value forces CPU |
| `ACESTEP_KEEP_STAGES` | values beginning with `1`, `t`, `T`, `y`, or `Y` eagerly load and keep every stage resident |
| `ACESTEP_LM_DUMP_LAYERS` | write the LM's per-layer prefill hidden states to this file path |
| `ACESTEP_LM_DUMP_TOKENS` | write the Phase-2 prompt token IDs as CSV |
| `ACESTEP_LM_DUMP_LOGITS` | write the conditional Phase-2 prefill logits as raw F32 |
| `ACESTEP_VAE_PROFILE` | print the VAE per-op-type time inventory |
| `ACESTEP_VAE_WIN_CORE` | diagnostic-only positive integer that pins the decode window core; not a supported tuning API |

Use `ACESTEP_LM_GPU` or `ACESTEP_DETOK_GPU` to take the measurement that would widen the allowlist for a new backend, without a rebuild.

### Parity debug hooks

`ACESTEP_PARITY_DEBUG` is a compile-time macro, not an environment variable, and there is no CMake option for it: configure with `-DCMAKE_CXX_FLAGS=-DACESTEP_PARITY_DEBUG`. The hooks below are `#ifdef`-ed out of `generate()` without it, so setting them on a default build does nothing.

| Variable | Effect |
|---|---|
| `ACESTEP_DUMP_DIR` | log the tokenizer shapes and write `our_dit_output.bin`, `our_noise.bin`, `our_context.bin`, `our_enc_hidden.bin` here for tensor-level comparison against acestep.cpp |
| `ACESTEP_INJECT_NOISE`, `ACESTEP_INJECT_ENC`, `ACESTEP_INJECT_CONTEXT` | replace the noise, encoder hidden states, or DiT context with an acestep.cpp `--dump` tensor to isolate a diverging stage |

Dumps use a flat `[ndim, d0, d1]` int32 header followed by the `f32` payload, the same format the injection hooks read.

### Memory

By default no stage stays resident after `create()`: `generate()` loads each stage immediately before its step and frees it right after, so only one stage is resident for the LM, detokenizer, DiT, and VAE steps rather than all six at once, which is what keeps a non-entitled iOS app inside its memory budget. The one overlap is the condition encoder: it supplies the silence frame that pads the DiT context before the text encoder runs and its own forward comes after text encoding, so it is loaded first and the two encoders are co-resident until the text encoder is freed. That pair is the only two-stage overlap, not necessarily the largest memory footprint. A truthy `ACESTEP_KEEP_STAGES` opts out, eagerly loads every stage in `Engine::create()`, and keeps them resident. With stages kept resident, the LM and DiT also retain their reusable forward graphs and activation pools between generations (they are freed with the model in the default per-stage mode), trading additional steady-state memory for repeat-generation speed.

Long VAE decodes are split into overlapping windows. The engine probes the
active backend's maximum allocation size against the real decode graph and
shrinks the core window when needed; short inputs remain a single graph.
`ACESTEP_VAE_WIN_CORE` can pin the core only for diagnostics. VAE encode is not
windowed and still allocates one full graph.

## Build

Standalone from the repository root, against an installed `ggml-speech`:

```sh
cmake -S engines/audiogen -B build/audiogen -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build/audiogen -j
./build/audiogen/music-cli --help
```

The umbrella build uses `SPEECH_BUILD_AUDIOGEN=ON` (the default). In-tree CLIs
and tests link an object library, so they work when `BUILD_SHARED_LIBS=ON`
(whisper's umbrella default) and `audiogen-cpp` is a hidden shared library.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install -DSPEECH_BUILD_AUDIOGEN=ON
cmake --build build -j
./build/engines/audiogen/music-cli --help
```

Visual Studio, Xcode, and other multi-config generators add a configuration
directory such as `Release/` beneath the executable directory. See the
[top-level README](../../README.md) for the shared ggml build.

| Option | Default | Effect |
|---|---|---|
| `AUDIOGEN_BUILD_LIBRARY` | `ON` | build the library; linkage follows `BUILD_SHARED_LIBS` |
| `AUDIOGEN_BUILD_EXECUTABLES` | `ON` standalone, `OFF` as a subdirectory | CLIs and per-stage smoke harnesses |
| `AUDIOGEN_BUILD_TESTS` | `ON` standalone, `OFF` as a subdirectory | unit, integration, and backend parity tests |
| `AUDIOGEN_BUILD_MINIMAX` | `ON` on desktop, unavailable on Android and iOS | MiniMax-Music3 engine; CPU by default, GPU through `EngineOptions::device` |
| `AUDIOGEN_INSTALL` | `ON` | generate install rules |
| `AUDIOGEN_USE_SYSTEM_GGML` | `ON` | `find_package(ggml)`; required, there is no supported vendored ggml in this tree |
| `AUDIOGEN_CCACHE` | `ON` | use ccache when available |

Consume the installed package with:

```cmake
find_package(audiogen-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE audiogen-cpp::audiogen-cpp)
```

Minimal C++ usage:

```cpp
#include <audiogen-cpp/acestep/engine.h>

#include <exception>
#include <iostream>

int main() {
    try {
        tts_cpp::acestep::EngineOptions options;
        options.models_dir = "models/acestep";
        options.n_gpu_layers = 99;

        auto engine = tts_cpp::acestep::Engine::create(options);
        tts_cpp::acestep::GenerateParams params;
        params.caption = "Driving synth pop with bright analog leads";
        params.duration = 8.0f;

        bool cancel_requested = false;
        auto result = engine->generate(params, [&cancel_requested](const std::string &, int, int) {
            return !cancel_requested; // return false to cancel cooperatively
        });
        std::cout << result.pcm.size() / 2 << " stereo frames at "
                  << result.sample_rate << " Hz\n";
        // result.pcm is interleaved: pcm[frame * 2 + channel].
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

## Audio editing

Repaint and FlowEdit are separate operations. Repaint regenerates a time
range while preserving the surrounding audio; FlowEdit morphs source
caption/lyrics conditioning toward target caption/lyrics conditioning. Both
can run independently, can be repeated, and can be composed in any order.
Because each operation consumes the previous operation's result,
`FlowEdit -> Repaint` is intentionally different from
`Repaint -> FlowEdit`.

Editing requires `GenerateParams::source_audio` as normalized, interleaved,
stereo PCM at 48 kHz. The engine derives duration from this buffer and bypasses
the LM and FSQ detokenizer. Repaint ranges must remain inside the source;
outpainting is not supported.

```cpp
tts_cpp::acestep::GenerateParams params;
params.source_audio = source_pcm_48khz_stereo;
params.seed = 22883;

tts_cpp::acestep::FlowEditParams flow;
flow.source_caption = "bright late-1990s pop";
flow.source_lyrics = original_lyrics;
flow.target_caption = "dark analog synthwave";
flow.target_lyrics = "[Instrumental]";
flow.n_min = 0.0f;
flow.n_max = 1.0f;
flow.n_avg = 1;
params.edit_plan.emplace_back(std::move(flow));

tts_cpp::acestep::RepaintParams repaint;
repaint.start_seconds = 10.0f;
repaint.end_seconds = 20.0f;
repaint.mode = tts_cpp::acestep::RepaintMode::Balanced;
repaint.strength = 0.5f;
repaint.caption = "expressive analog synthesizer solo";
repaint.lyrics = "[Instrumental]";
params.edit_plan.emplace_back(std::move(repaint));

auto result = engine->generate(params);
```

`GenerateParams::seed` seeds the first operation. Operation at index `i` uses
`seed + i`, so changing plan order also changes its deterministic noise.

### Repaint options

| Field | Default | Meaning |
|---|---|---|
| `start_seconds` | `0` | inclusive start of the repaint region |
| `end_seconds` | `-1` | end of the region; `-1` means source end |
| `mode` | `Balanced` | `Conservative`, `Balanced`, or `Aggressive` source preservation |
| `strength` | `0.5` | balanced-mode regeneration strength in `[0, 1]`; `0` preserves more and `1` regenerates more |
| `caption` | `GenerateParams::caption` | target description for this operation |
| `lyrics` | `GenerateParams::lyrics` | target lyrics for this operation |

`Conservative` maximizes source injection, boundary blending, and
post-decode waveform preservation. `Aggressive` disables those preservation
steps. `strength` controls the interpolation only in `Balanced` mode.

### FlowEdit options

| Field | Default | Meaning |
|---|---|---|
| `source_caption` | required | description of the current audio |
| `source_lyrics` | `[Instrumental]` | current lyrics |
| `target_caption` | required | desired description |
| `target_lyrics` | `[Instrumental]` | desired lyrics |
| `n_min` | `0` | beginning of the active diffusion window, in `[0, 1]` |
| `n_max` | `1` | end of the active diffusion window, in `[0, 1]` |
| `n_avg` | `1` | forward-noise samples averaged per active step; must be at least `1` |

FlowEdit v1 is the validated Turbo, Euler, no-CFG path. It requires
`diffusion_guidance_scale == 1`, with DCW, ADG, and Heun disabled; unsupported
combinations are rejected rather than silently ignored.

## Command line tools

| Binary | Purpose |
|---|---|
| `music-cli` | end-to-end text-to-music |
| `acestep-cli` | VAE decode and reconstruction roundtrip harness |
| `textenc-smoke`, `lm-smoke`, `lmgen-smoke`, `bpe-smoke`, `detok-smoke`, `cond-smoke`, `dit-smoke` | per-stage smoke harnesses |

### music-cli

```sh
# all four GGUFs in one directory
./build/audiogen/music-cli --models models/acestep --out song.wav --dur 8 --seed 42

# explicit per-stage paths, GPU, custom prompt
./build/audiogen/music-cli --dit dit.gguf --lm lm.gguf --text emb.gguf --vae vae.gguf \
                  --caption "driving synth pop, bright analog leads" \
                  --lyrics "[Instrumental]" --bpm 128 --key "C major" --tsig 4/4 \
                  --lang en --steps 8 --shift 3.0 --gpu --out song.wav

# Simple Mode: one short query becomes a complete song (caption, lyrics and
# metadata composed by the LM; --dur 0 lets the LM pick the duration too)
./build/audiogen/music-cli --models models/acestep --simple \
                  --caption "a romantic modern salsa with male lead vocals for a wedding" \
                  --dur 0 --gpu --out salsa.wav

# condition generation with a 48 kHz PCM16 WAV timbre reference
./build/music-cli --models models/acestep --ref-audio reference-48k.wav \
                  --caption "warm Latin pop with a male lead vocal" \
                  --lyrics "[Verse]\nA brand new lyric" --gpu --out referenced.wav

# repaint 10s..20s while preserving the rest of the source
./build/audiogen/music-cli --models models/acestep --src-audio source-48k.wav \
                  --caption "expressive analog synthesizer solo" \
                  --lyrics "[Instrumental]" --repaint-start 10 --repaint-end 20 \
                  --repaint-mode balanced --repaint-strength 0.5 \
                  --seed 22883 --gpu --out repainted.wav

# FlowEdit uses --caption/--lyrics as the target
./build/audiogen/music-cli --models models/acestep --src-audio source-48k.wav \
                  --flow-source-caption "bright late-1990s pop" \
                  --flow-source-lyrics "[Instrumental]" \
                  --caption "dark analog synthwave" --lyrics "[Instrumental]" \
                  --flow-n-min 0 --flow-n-max 1 --flow-n-avg 1 \
                  --seed 22883 --gpu --out flow-edited.wav

# ordered, repeated, composable operations
./build/audiogen/music-cli --models models/acestep --src-audio source-48k.wav \
                  --edit-plan edit-plan.json --seed 22883 --gpu --out edited.wav
```

| Flag | Default | Meaning |
|---|---|---|
| `--models DIR` | | directory holding the four stage GGUFs |
| `--dit`, `--lm`, `--text`, `--vae` | | explicit per-stage GGUF paths |
| `--caption TEXT` | built-in pop-rock prompt | text prompt |
| `--lyrics TEXT` | `[Instrumental]` | lyrics |
| `--dur SECONDS` | `8` | target length, drives the LM code count |
| `--seed N` | `42` | negative means random |
| `--bpm N`, `--key STR`, `--tsig STR`, `--lang CODE` | inferred | optional metadata hints |
| `--steps N`, `--shift F` | per variant | sampler overrides |
| `--no-dcw` | DCW enabled | disable the official Haar low/high correction applied after each DiT step |
| `--no-loudness` | normalization on | skip the percentile loudness normalization (99.999th percentile to 1.0, tail hard-clipped) applied to generation output; edits and lego stems are never normalized |
| `--lrc out.lrc` | off | write synchronized lyric timestamps (LRC) next to the WAV; requires lyrics |
| `--temp F`, `--topp F`, `--topk N`, `--cfg F` | `0.85`, `0.9`, off, `2.0` | LM sampling for the audio codes |
| `--no-phase1` | off | skip the LM metadata auto-fill pass |
| `--simple` | off | Simple Mode: expand `--caption` into a full request (lyrics regenerate unless `--lyrics "[Instrumental]"` is passed) |
| `--rewrite` | off | Query Rewriting: the LM formats `--caption` and `--lyrics` into a detailed request, preserving the lyric content (needs the 1.7B LM) |
| `--score` | off | teacher-forced LM quality score of the generated codes, printed with its per-condition breakdown |
| `--understand FILE` | | reverse pipeline: describe a 48 kHz PCM16 WAV (metadata + caption + recovered codes) instead of generating |
| `--req FILE` | | request JSON; pre-supplied `audio_codes` skip the LM stage |
| `--ref-audio FILE` | | 48 kHz PCM16 WAV used by ACE-Step's timbre-conditioning path |
| `--src-audio FILE` | | 48 kHz PCM16 source WAV; required for editing |
| `--repaint-start SEC`, `--repaint-end SEC` | `0`, source end | standalone repaint range |
| `--repaint-mode MODE` | `balanced` | `conservative`, `balanced`, or `aggressive` |
| `--repaint-strength F` | `0.5` | balanced-mode strength in `[0, 1]` |
| `--flow-source-caption TEXT` | | enables standalone FlowEdit and describes the current source |
| `--flow-source-lyrics TEXT` | `[Instrumental]` | current source lyrics |
| `--flow-n-min F`, `--flow-n-max F` | `0`, `1` | active FlowEdit diffusion window |
| `--flow-n-avg N` | `1` | forward-noise samples averaged per active step |
| `--edit-plan FILE` | | ordered JSON edit plan; cannot be combined with standalone edit flags |
| `--normalize` | off for edit plans | peak-normalize edited CLI output; avoid for partial repaint when outside-region PCM must remain exact |
| `--gpu`, `--threads N` | CPU, hardware concurrency | compute placement |
| `--backends-dir DIR` | | directory containing staged dynamic ggml backend modules; required by Android and Linux arm64 dynamic-backend builds |
| `--dump-stages DIR` | | write one `.bin` per stage into an existing directory |
| `--out PATH` | `music_out.wav` | output WAV path |

`--req` accepts a flat JSON object. Request values override overlapping CLI
values after CLI parsing:

| Field | Accepted value |
|---|---|
| `caption`, `lyrics`, `keyscale`, `timesignature`, `vocal_language` | string |
| `bpm`, `inference_steps`, `seed` | number |
| `duration`, `shift`, `dcw_scaler`, `dcw_high_scaler` | number |
| `dcw_enabled` | boolean, or `0` / `1` |
| `audio_codes` | quoted comma-separated string, for example `"12,34,56"` |

`audio_codes` is currently not parsed as a JSON numeric array. A non-empty
value bypasses the LM and feeds the codes directly to the FSQ detokenizer.

`--edit-plan` accepts an object with an `operations` array. Operations execute
in array order and may repeat:

```json
{
  "operations": [
    {
      "type": "flow-edit",
      "source_caption": "bright late-1990s pop",
      "source_lyrics": "[Instrumental]",
      "target_caption": "dark analog synthwave",
      "target_lyrics": "[Instrumental]",
      "n_min": 0,
      "n_max": 1,
      "n_avg": 1
    },
    {
      "type": "repaint",
      "start": 10,
      "end": 20,
      "mode": "balanced",
      "strength": 0.5,
      "caption": "expressive analog synthesizer solo",
      "lyrics": "[Instrumental]"
    }
  ]
}
```

The sampler enables ACE-Step's single-level Haar DCW `double` mode by default.
At timestep `t`, the low band uses `t * 0.05` and the high band uses
`(1 - t) * 0.02`, matching the official Python defaults. The correction runs
on the host latent between DiT steps, so it is backend-independent and adds
negligible work compared with a transformer forward.

Reference WAV input is decoded at the `music-cli` boundary and must be PCM16
stereo or mono at 48 kHz. The engine receives normalized interleaved stereo PCM
and encodes it in overlapping VAE windows for bounded memory on long inputs.

### acestep-cli

```sh
./build/audiogen/acestep-cli --model vae.gguf --t-latent 32 --out out.wav
./build/audiogen/acestep-cli --model vae.gguf --roundtrip --in in.wav --seconds 2.56 --out out.wav
```

The first form is the default mode: it decodes a synthetic latent, which checks that real weights load and that the decode graph (`ggml_col2im_1d` + `ggml_snake`) runs on the selected backend. `--roundtrip` encodes a real WAV and prints the per-channel reconstruction correlation, the audible end-to-end VAE check. Both forms take `--gpu`.

### acestep-fit-params

```sh
./build/audiogen/acestep-fit-params --models-dir /path/to/acestep --duration 60 --n-gpu-layers 99
```

Memory-fit preflight: projects whether the four stage GGUFs plus a generation
workload fit the memory available right now, reading only GGUF metadata (no
weights load, nothing runs). It resolves the same backends and stage placement
`Engine::create` would, drives each stage loader in metadata-only mode, and
prices the real compute graphs at the workload's shapes through ggml's
size-only allocator APIs, so the projection tracks the runtime by construction.
The default projection follows the per-stage low-memory residency above (the
peak phase per memory pool); `--keep-stages 1` (or `ACESTEP_KEEP_STAGES`)
projects the everything-resident mode. Exit code follows the `@qvac/model-fit`
contract: 0 = fits, 1 = does not fit (a valid answer), 2 = error. `--json`
emits the projection for the SDK; the library entry point is
`tts_cpp::acestep::fit_params` (`include/audiogen-cpp/acestep/fit.h`), and
hosts can link the CLI as `acestep_fit_cli_main`. Workload knobs: `--duration`,
`--text-tokens`, `--lyric-tokens`, `--lm-cfg`, `--guidance`,
`--with-source-audio`, `--margin-mib`.

## Tests and parity

```sh
cmake -S engines/audiogen -B build/audiogen -DAUDIOGEN_BUILD_TESTS=ON \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build/audiogen -j
ctest --test-dir build/audiogen
```

`test-acestep-units`, `test-acestep-converter`, `test-minimax-units`,
`test-minimax-converter`, `test-minimax-quant-audit`, and
`test-minimax-dump-compare` cover weight-free CPU logic and need no GGUFs.
ACE-Step coverage includes the BPE tokenizer (UTF-8 decode, merges, byte
fallback, decode round-trip) on a hand-built vocabulary. MiniMax coverage
includes metadata compatibility, model-pair selection, Unicode token classes,
frame validation, prompt assembly and the request-utils caption/lyrics
cleanup helpers, unconditional masking, deterministic noise, flow scheduling,
condition length, window stitching, sampler edge cases, and converter output
transactions. Set `AUDIOGEN_TEST_MINIMAX_MODELS_DIR` to a directory containing
the MiniMax GGUF pair to run `test-minimax-integration`, which covers model
loading, generation output, progress, and cancellation.
On a Metal build, `ctest --test-dir build/audiogen -R
test-minimax-metal-ops` runs the model-free CPU/Metal condition and vocoder
parity regression. It skips with return code 77 either when Metal is
unavailable or when the Metal device cannot run `MUL_MAT` — both parity graphs
are matmul-based, and ggml gates `GGML_OP_MUL_MAT` on simdgroup reduction
(`MTLGPUFamilyApple7`+), which virtualized GPUs such as the ones on hosted macOS
CI runners do not report. Meaningful parity coverage therefore needs a
non-virtualized Apple GPU, which is why the audiogen CI macOS lane runs on the
self-hosted `qvac-macos26-arm64-gpu` runner and fails rather than passes if the
test skips there.
`test-acestep-integration` exercises the ACE-Step public API when model paths
are supplied and otherwise reports a skipped test.
`test-audiogen-comparison-lib` runs the engine-comparison harness's own
`node:test` suites (`benchmarks/comparison/tests/`) on hosts with node,
so the harness's adapters, aggregation, and report logic stay verified.

Dumps are the tool for localising a backend divergence. Run the same prompt twice with `--dump-stages` (ACE-Step stages) or `--dump-iters` (MiniMax AR iterations), then compare:

| Script | Purpose |
|---|---|
| `scripts/stage_cos.py` | per-stage cosine and rel_l2 between two dump directories; fails under a worst-stage cosine of 0.999, and also on a missing, truncated, or mismatched counterpart |
| `scripts/dequant_gguf.py` | rewrite a GGUF with every quantized tensor dequantized to F32, giving a ground-truth trajectory to compare a quantized run against |
| `scripts/compare_ar_dumps.py` | finite-subset cosine, argmax agreement, and top-8 overlap between two `--dump-iters` directories; gates on `--min-cosine` / `--min-argmax-agree` and fails a pair it could not actually compare |
| `scripts/audit_quant_types.py` | audit a quantized GGUF against the deny-list of tensors that must never be quantized, plus a per-type byte summary |

`stage_cos.py` and `dequant_gguf.py` need `numpy`; `dequant_gguf.py` and
`audit_quant_types.py` also need `gguf`. `compare_ar_dumps.py` uses `numpy`
when it is installed and falls back to an equivalent stdlib path when it is not.

An informal local comparison against upstream `acestep.cpp` measured 0.98 to
0.99 end-to-end correlation on identical codes, and LM greedy decoding matched
upstream argmax. This is not reproducible benchmark evidence: the repository
does not pin the model set, upstream revision, command, correlation definition,
or result artifact. Use stage dumps and the scripts above for a recorded
comparison.

## Performance

There is no CI benchmark lane for this engine yet. `music-cli` always enables
verbose engine output and prints per-stage wall clock to stderr (`[music-cli]`,
`[acestep-timing]`). Direct library use prints `[acestep-timing]` only when
`EngineOptions::verbose` is enabled; it defaults to `false`.

For a reproducible engine-to-engine measurement against upstream
`acestep.cpp` (`ace-lm` + `ace-synth`, no addon), see
[`benchmarks/comparison/README.md`](benchmarks/comparison/README.md).

## License

`audiogen-cpp` is MIT licensed; see [LICENSE](LICENSE). Model weights and upstream model code carry their own terms, listed in [NOTICE](NOTICE).
