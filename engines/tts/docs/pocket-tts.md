# tts engine: Pocket TTS

Part of the [tts documentation](../README.md).

Pocket provides English text-to-speech through a persistent CPU, GPU or Hexagon
FlowLM/Mimi engine and incremental 24 kHz mono PCM callbacks. Its public API is
`<tts-cpp/pocket/engine.h>`; `pocket-cli` uses that API. F32 arithmetic is used
throughout. F16 is a storage option whose weights expand to F32 at load.

Prepared voices are checkpoint-specific. The tested released checkpoint has
a disabled reference encoder, so voice cloning requires separate
encoder-enabled weights; random-weight encoder parity does not establish
cloning quality. Each instance admits one synthesis at a time. Callbacks run
on the calling thread; returning false or calling `cancel()` ends streaming.
Reentrant/overlapping synthesis throws. Each request resets voice, codec and
RNG; native seeds are not PyTorch-equivalent.

## Conversion

Run these commands from the repository root. Prepare the conversion environment:

```sh
python3.12 -m venv /tmp/pocket-venv
/tmp/pocket-venv/bin/pip install -r engines/tts/scripts/pocket-flow-requirements.txt
```

Python 3.11+ and PyTorch are conversion/reference dependencies only. Use an
installed speech-branch ggml as described in the [build guide](../../../docs/BUILD.md).

For a complete bundle, supply explicit local upstream assets. The output
directory must not already exist; it is published after all stages succeed.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/convert-pocket-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --tokenizer /path/to/tokenizer.model --voice /path/to/alba.safetensors \
  --output /path/to/pocket-bundle
GGML_PREFIX=/path/to/ggml-install
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_EXECUTABLES=ON
cmake --build build/pocket --target pocket-cli
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --output /tmp/pocket.wav
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --gpu \
  --text "Hello from Pocket TTS on Metal." --output /tmp/pocket-metal.wav
```

The bundle contains `flow-lm.gguf`, `mimi.gguf`, `frontend.json`, `voice.gguf`
and a SHA-256 manifest. Use a prepared voice from the same upstream checkpoint
revision as the weights: legacy voice files do not contain a checkpoint hash,
so the converter binds the caller-selected assets, rather than proving the
provenance of arbitrary voice files. Native loading rejects mixed bundle
hashes and invalid cache lengths, types or payloads.

The identity SentencePiece unigram frontend includes byte fallback, Unicode
case/whitespace/digit metadata, punctuation repair and decimal-aware sentence
splitting. It matches 322 upstream corpus cases. Generate the fixture using
`dump-pocket-frontend-reference.py --tokenizer /path/to/tokenizer.model
--output /path/to/pocket-bundle/frontend-corpus.json`, then configure
`TTS_CPP_POCKET_MODEL_DIR` to enable frontend and public-engine CTest cases.

The persistent engine restores the chosen voice for each text chunk and runs
FlowLM/Mimi on separate workers with at most 16 queued latent frames.
CPU remains the default. Set `EngineOptions::use_gpu = true` or pass `--gpu`
to select Metal, CUDA, Vulkan or OpenCL for both the language model and the codec,
including their persistent weights and caches. A supported GPU backend must be available;
the request fails explicitly otherwise. `backend_name()` and CLI JSON reports
identify the selected backend. `backends_dir` (CLI: `--backends-dir`) locates
dynamically loaded ggml backends using the library's existing process-wide
loader. For backend-specific validation, set `TTS_CPP_GPU_BACKEND` to
`metal`, `cuda`, `vulkan`, `opencl` or `webgpu`; the selected backend must match the request.
The Fabric seed uses a specified Box–Muller transform. An encoder-enabled
checkpoint can use `--reference-audio` instead of the prepared voice; the
tested public checkpoint rejects that path.

## OpenCL execution

`EngineOptions.backend = "opencl"` (CLI: `--backend opencl`) explicitly
selects OpenCL; `"webgpu"` explicitly selects the WebGPU (Dawn) backend;
`"cpu"` explicitly selects CPU. Empty/`"auto"` follows `use_gpu`.
Explicit OpenCL or WebGPU requests fail if that backend is unavailable. Automatic GPU selection
also supports Metal, CUDA and Vulkan, retaining the registry's Adreno tier
policy. Hexagon is selected explicitly as described below. Both FlowLM and Mimi
use the same selected device, retaining
F32 weights and persistent caches on that device. Prepared voices and small
conditioning/output values still cross the host boundary.

Use an installed fabric ggml build with `GGML_OPENCL=ON` and its OpenCL backend
available to the existing registry loader. Dynamic builds must ship the
backend libraries beside the application, or set the public backend directory
before constructing the first engine (see `<tts-cpp/backend.h>`).

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --backend opencl \
  --text "Hello from Pocket TTS." --output /tmp/pocket-opencl.wav \
  --runs 3 --report /tmp/pocket-opencl.json
```

The CLI prints both stage backends and the device description; its JSON records
`flow_lm_backend`, `mimi_backend`, and `device`. Native callers can inspect
`flow_lm_backend_name()`, `mimi_backend_name()`, and `device_name()`.
Every graph checks backend operation support before allocation or execution;
an incompatible ggml build reports the unsupported operation instead of
executing a partial graph or falling back to CPU.

FlowLM supplies an explicit causal mask to OpenCL softmax. This avoids the
reviewed ggml-speech `DIAG_MASK_INF` vector kernel, whose lane writes corrupt
neighboring vectors at attention widths divisible by eight. The existing
64-position full-context/chunked-prefill regression exposed this on Adreno 830;
CPU retains its original graph. Preflight prices both the device mask and its
host upload staging.

The ggml OpenCL backend shares kernel arguments, scratch and lifecycle state
between handles on a device. Pocket serializes its OpenCL submissions,
initialization and backend release across its workers and engine instances.
Each worker still owns its weights, persistent state and graph arena, so
their memory can overlap. CPU workers remain concurrent. This Pocket gate
does not cover another Fabric engine using the same ggml OpenCL context;
applications must serialize such mixed-engine use until ggml provides that
protection itself.

## Hexagon execution

`EngineOptions.backend = "hexagon"` (CLI: `--backend hexagon`) explicitly
selects Qualcomm Hexagon `HTP0`; `"HTP0"` is an equivalent alias. Both stages use
the selected session's RPC buffers and execute their complete graphs on the
DSP. CPU remains the default, and `--gpu` uses the GPU selection described above.
An unavailable HTP runtime or unsupported graph operation is an error with no CPU
fallback. The stage backend names must both report `HTP0`.

Use an Android arm64 build of the speech ggml dependency with
`GGML_HEXAGON=ON`, built using matching Hexagon SDK, Hexagon Tools and Android
NDK versions. The reference toolchain is SDK 6.6.0.0, Tools 19.0.07 and NDK r28b.
Ship `libggml-hexagon.so` with the ggml libraries and the architecture-matched
`libggml-htp-v*.so` DSP skeletons from that same build. The loader extends
`ADSP_LIBRARY_PATH` from the configured backend directory. Android FastRPC
access and compatible vendor `libcdsprpc.so`/drivers are required; detecting a
Snapdragon GPU does not establish DSP availability.

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --backend hexagon \
  --text "Hello from Pocket TTS." --output /tmp/pocket-hexagon.wav \
  --runs 3 --report /tmp/pocket-hexagon.json
```

The baseline retains expanded F32 weights, F32 caches and `GGML_PREC_F32`
matmul requests, which select HVX computation. HMX/FP16 acceleration and
quantization are separate tuning work. FlowLM uses an explicit causal mask
because Hexagon has no `DIAG_MASK_INF` operation, and lowers `MEAN` into F32
`SUM_ROWS` followed by division by the row width. Mimi retains its existing
tap-wise F32 convolution graphs and explicit sliding-window mask.

The HTP backend shares one queue, graph cache and mapping state per registered
device. Pocket serializes HTP graph construction through buffer destruction,
weight loading, tensor transfers, reset and backend release across all Pocket
workers and instances. A recursive gate permits FlowLM's nested projection and
attention graphs. CPU workers remain concurrent. Other Fabric engines using
the same HTP session need application-level serialization until ggml supplies
backend-wide protection. Different processes have separate sessions.

Pocket rejects a buffer type whose packing limit exceeds the runtime's
effective RPC payload cap, before model allocation or memory preflight. This
also covers packed weights/state and graph arenas containing several valid
tensors. With the current 2 GiB mapping limit and 4 KiB guard,
`GGML_HEXAGON_MBUF=2048` or larger is incompatible; leave the 1024 MiB default
or use at most 2047 MiB. Set this before backend initialization: the retained
session and buffer type settings, rather than the current environment, are
checked.

Enable `TTS_CPP_POCKET_HEXAGON_TESTS=ON` with the three Pocket fixture directories
to register explicit Hexagon FlowLM F32/F16, Mimi, fit and engine tests. These
reuse the existing CPU tolerances and cover cached/chunked prefill, Mimi's
sliding-window boundary, streaming, cancellation, reuse and parallel instances.
Missing fixture files remain visibly disabled; an enabled hardware test fails
if Hexagon is unavailable. Hardware validation and timing results must be
reported separately from successful host builds.

`frames_after_eos` (CLI: `--frames-after-eos`) accepts an explicit 0–100-frame
tail, or -1 for the checkpoint/text default. Synthesis and memory preflight
reserve this tail in addition to the text-duration estimate. All chunks must
fit the context before any PCM is emitted; detecting EOS near the estimate's
end still permits the complete tail. If EOS never arrives within the original
text estimate, generation fails rather than spending the reserved tail.

`scripts/convert-pocket-flow-lm-to-gguf.py` takes a local Kyutai checkpoint and
its YAML configuration. It selects `flow_lm.*` tensors from the safetensors
bundle, leaving Mimi out. BF16 source weights expand to F32. Unknown/missing
FlowLM weights, unsupported flow types, and invalid dimensions fail explicitly.
The output is replaced only after a successful complete export.

GGUF architecture: `pocket-tts-flow-lm`; `pocket.schema_version = 1`.
Keys under `pocket.*` record dimensions, RoPE base, BOS policy, flow type,
source SHA-256, and the input configuration. Tensor names retain upstream
names with the `flow_lm.` prefix removed. PyTorch tensor dimensions are reversed
by GGUF without transposing data. `bos_before_voice` is flattened to a vector.
This format is specific to Fabric's FlowLM stage and is not interchangeable
with llama.cpp's full Pocket TTS model/mmproj files.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/convert-pocket-flow-lm-to-gguf.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /path/to/flow-lm-f32.gguf
```

The C++ core does not load Python, safetensors, ONNX Runtime, or a second ggml copy.

## Core ML sidecars

On Apple platforms a `TTS_CPP_COREML=ON` build can run the two per-frame
stages on Core ML, whatever the ggml backend. Both require macOS 15 / iOS 18.

| Sidecar | GGUF → sidecar | Runs | Default placement |
|---|---|---|---|
| FlowLM step | `flow-lm.gguf` → `flow-lm-step.mlmodelc` | each generated frame: `input_linear`, the six transformer layers, `out_norm` and the one-step LSD flow, with the KV cache in Core ML state | `cpu_and_gpu`: Core ML cannot place model state on the Neural Engine |
| Mimi decoder | `mimi.gguf` → `mimi-decoder.mlmodelc` | one Mimi decode frame (1920 samples) per prediction: upsample, transformer, SEANet decoder. The conv histories and the 250-position attention window are explicit inputs and outputs that the engine keeps | `cpu_and_ane` |

Text prefill, voice preparation and the EOS head stay on ggml. The step's
voice KV rows are copied into its state once per engine, and each chunk's text
KV rows before its first frame. The two sidecars work independently, so a
bundle can carry either one.

Export from the bundle's GGUFs (FlowLM F32 or F16), using the Core ML
toolchain from `engines/parakeet/scripts/requirements-coreml.txt`:

```sh
python3 -m venv /tmp/coreml-venv
/tmp/coreml-venv/bin/pip install -r engines/parakeet/scripts/requirements-coreml.txt
/tmp/coreml-venv/bin/python engines/tts/scripts/export-pocket-coreml.py \
  --gguf /path/to/pocket-bundle/flow-lm.gguf /path/to/pocket-bundle/mimi.gguf --compile
```

Each GGUF's architecture selects its sidecar, which is written next to it.
Like every TTS sidecar, it is named by dropping a quant tag and adding a
suffix, so `flow-lm-f16.gguf` also maps to `flow-lm-step.mlmodelc`.

The engine loads a sidecar only after these checks pass:
- every input, output and state shape matches;
- the schema is supported;
- the checkpoint hash in the sidecar matches its GGUF's `pocket.source_sha256`;
- one warm-up prediction succeeds.

Bundles share file names across checkpoints, so a sidecar from another
checkpoint is rejected rather than run.

### Generated sidecars and background loading

Both sidecars load through the shared Core ML runtime (`engines/coreml`), in
the background. A prebuilt sidecar next to its GGUF is used as it is. Without
one, and with a cache directory (`EngineOptions::coreml.cache_dir`, or
`QVAC_COREML_CACHE_DIR`), a `TTS_CPP_COREML=ON` build on macOS 15 or later
generates the sidecar from the GGUF itself into that cache: no Python, no
download. The library embeds two weight-free templates made from the
unchanged exporter:

| Template | From | Default units | Notes |
|---|---|---|---|
| `pocket-step-c1024` | `flow-lm.gguf` (F32 or F16) | `cpu_and_gpu` | context 1024; one model per Engine (its KV cache is Core ML state) |
| `pocket-mimi` | `mimi.gguf` (F32 or F16) | `cpu_and_ane` | shared by every Engine on the same weights |

Generation writes `weight.bin` from the GGUF's tensors (fp16) and the
checkpoint hash into the spec, then compiles the package on the device; the
step's folded time embedding is computed in C++
(`engines/tts/coreml-provided/pocket.cpp`). Generated bytes match the
exporter's: byte for byte except one folded bias value one fp16 ulp away on
the English 2026-04 checkpoint. A generated artifact is shared by every
process using the same cache directory and weights; later engines adopt it.
Generation reads the file the ggml stage loaded: FlowLM and Mimi each read
their GGUF through one descriptor, and a GGUF replaced or removed after that
load fails the stage with `source_changed` (it stays on ggml) instead of
generating from other weights.

Until a sidecar is ready, its stage runs on ggml. Each chunk takes its
backends once, at its start (the step on the producer thread, the decoder on
the calling thread), so a chunk never switches backend; a sidecar that
becomes ready mid-call serves the next chunk.
`SynthesisResult::coreml_pending` reports a call in which a stage ran on ggml
only because its sidecar was not ready yet. A step model new to an Engine
receives the voice rows before its first chunk.

`EngineOptions::coreml` (`tts-cpp/coreml_options.h`):

| Field | Effect |
|---|---|
| `cache_dir` | Where generated sidecars and warm-load hints live. Unset: no generation (prebuilt sidecars still load). |
| `wait` | `none` (default on macOS): never block, apart from a short bounded wait for a load this host finished quickly before. `construct`: the constructor waits until both sidecars are ready or failed. `call`: each chunk waits for the sidecars it needs. |
| `prewarm` | `none`: nothing loads before the first chunk requests it. Otherwise both sidecars (Pocket's minimal set) are requested at construction. |
| `use` | `require`: like `wait = call`, but a sidecar that cannot serve a chunk is an error, and a chunk too long for the step is refused before any PCM. `off`: ggml only. |
| `generate` | `never`: no generation; prebuilt sidecars still load. |
| `compute_units` | Both stages; `coreml_step_units` / `coreml_decoder_units` override it per stage. An explicit `cpu_and_ane` for the step throws (Core ML state cannot be placed on the Neural Engine, so the load would fail after seconds); from `coreml_compute_units` or `POCKET_COREML_COMPUTE_UNITS` the step keeps `cpu_and_gpu` instead. |
| `memory_budget_bytes` | The budget fit was assessed against. When only the ggml projection fits it, the Engine resolves `use = off` (memory-degrade). |

Unset fields fall back to `coreml_compute_units` / `coreml_strict`, then to
the `POCKET_COREML_*` environment. `Engine::coreml_status()` returns both
stages' state, route (`prebuilt`, `cached`, `generated` or `none`), units,
label, artifact and timings as JSON; `Engine::coreml_wait(timeout_ms)` waits
for both (`Engine::cancel()` ends a blocking wait). `Engine::coreml_construction()`
returns the construction group's result as JSON (`success` when every requested
sidecar reached ready; `nothing_to_prepare` for an empty group, which
succeeds; each slot's final state; the stages with no route under
`skipped_route_none`). `Engine::coreml_prewarm(scope, wait)` requests both
sidecars after construction (scope `none` requests nothing), for example with
`prewarm = none` to prepare them before the first synthesis, and returns that
group's result the same way. `pocket-cli` takes `--coreml-cache-dir`, `--coreml-wait`,
`--coreml-prewarm`, `--coreml-step-units`, `--coreml-decoder-units` and
`--coreml-status-json`. Unlike the library (whose default is no cache, so no
generation), `pocket-cli` generates by default: without `--coreml-cache-dir`
or `QVAC_COREML_CACHE_DIR` it uses `~/Library/Application
Support/qvac-speech/coreml` (not `Library/Caches`, which macOS may purge,
costing a cold compile). `--coreml-status-json` refuses a path that names the
output WAV, the report or a model input (also through a symlink or another
spelling of the same file) before synthesizing. It also refuses any path inside
a compiled Core ML model (`.mlmodelc` / `.mlpackage`), inside the Core ML cache
directory, or naming (through a symlink or a hard link) a file of a sidecar
next to the GGUFs, of the cache or of a stage's resolved sidecar, so the status
never truncates compiled weights.

A generated route needs the stage's units to be validated for memory (see
fit below); otherwise the stage reports `unsupported(unvalidated_units)` and
runs on ggml. The templates are validated for their default units.

Step `--context` (default 1024) fixes how many positions the state holds:
voice, text and generated frames. A chunk that needs more positions than this
runs FlowLM on ggml. A chunk needs voice frames, plus tokens, plus the frame
estimate, plus the EOS tail. For a full default chunk (`max_tokens = 50`) with
the 126-frame English voice that is 126 + 50 + 234 + a 3–5 frame tail, about
415 positions. `--context 512` makes each step about 15% faster on an M2 when
every chunk fits. A step sidecar whose context the voice already fills is not
loaded.

Policy, resolved once per `Engine`:

| Setting | Effect |
|---|---|
| `EngineOptions::coreml_compute_units` / `POCKET_COREML_COMPUTE_UNITS` | Unset keeps each sidecar's default. `all`, `cpu_and_gpu`, `cpu_only` or `cpu_and_ane` applies to both sidecars; with `cpu_and_ane` the step keeps `cpu_and_gpu`. An explicit invalid value throws; an invalid inherited value warns and keeps the defaults. |
| `EngineOptions::coreml_strict` / `POCKET_COREML_STRICT` | A sidecar that cannot load, or fails a prediction, becomes an error instead of falling back to ggml. A chunk too long for the step is rejected before any PCM; without strict mode that chunk runs FlowLM on ggml. Presence of the environment variable means true. |
| `POCKET_COREML_DISABLE` | Keeps FlowLM and Mimi on ggml. |

Without strict mode, a failing prediction retires that sidecar for every
Engine using it. ggml then catches up by replaying the current chunk:
- For the step, the frames it generated are prefilled into the ggml cache in
  one batch.
- For the decoder, the chunk's earlier latents are decoded again on ggml,
  with the output discarded, to rebuild Mimi's streaming state.

In strict mode the sidecar is retired the same way and the call fails.

`Engine::flow_lm_on_coreml()` and `Engine::mimi_on_coreml()` report whether
each sidecar is loaded and not retired. `SynthesisResult::flow_lm_backend` and
`SynthesisResult::mimi_backend` report where a call's work ran:
- `"ggml"`;
- the sidecar label, such as `"coreml-gpu"` or `"coreml-ane"`, when all of it
  ran there;
- `"mixed"` when some ran on ggml. For the step that also includes
  `steps > 1`, whose extra flow steps run on ggml from the sidecar's hidden
  output.

`pocket-cli` reports both labels for each run. With a prebuilt sidecar,
`fit_params` returns `Error` with `coreml-memory-estimate-unavailable`. With
generated sidecars, fit adds their Core ML envelope (`FitResult::coreml`; see
the fit section).

Measured on an Apple M2 (macOS 15) with the English 2026-04 bundle, against
ggml F32:

| Check | Result |
|---|---|
| Teacher-forced FlowLM, 48 frames (`test-pocket-coreml --engine-fixture`) | hidden cosine ≥ 0.999997; EOS logit error ≤ 0.012; next latent error ≤ 0.0032 |
| Mimi decoder, same latents streamed | SNR 46–49 dB vs ggml Mimi |
| One step, context 1024 / 512 | `cpu_and_gpu` 5.5 / 4.6 ms; `cpu_only` 8.1 / 5.7 ms; ggml advance + flow: Metal ~8–13 ms, CPU ~23 ms |
| One Mimi frame | `cpu_and_ane` 1.5 ms; `cpu_only` 2.2 ms; `cpu_and_gpu` 5.1 ms; ggml: Metal 6.1 ms (1.3 ms/frame batched by 16), CPU 22–28 ms |
| 11.04 s utterance, warm runs, default CPU backend | RTF **0.100** with both sidecars, 0.304 with the step only, 0.311 for ggml; with `--threads 4`, 0.083 vs 0.17–0.19 |
| Same utterance, `--gpu` (Metal) | RTF 0.065–0.073 with both sidecars, 0.073–0.077 with the step only, 0.097 for ggml; first audio 16–23 ms vs 20–26 ms |
| Whisper base.en transcripts, CPU and Metal | identical to ggml, word for word |

With both sidecars, generation is limited by the step, at about 5.5 ms per
80 ms frame. On the default CPU backend, the remaining cost is ggml's text
prefill for each chunk; more `--threads` reduce it.

Tests:

- `test/test_export_pocket_coreml.py` checks the PyTorch rebuilds on random
  weights.
  - Step: decoding frame by frame through the stateful step must match a full
    causal forward, and the RoPE tables must match ggml mode 0.
  - Decoder: the phase-form upsample and transposed convolutions must match
    PyTorch's transposed convolutions, and decoding in calls of any frame
    count, across the attention window, must give the same stream.
  - Both: names and constants are checked against the runtime, and both are
    exported to Core ML.
- `test-pocket-coreml` covers the policy, naming, detection, the fit refusal,
  option validation (including the refused `cpu_and_ane` step), the envelope
  arithmetic and memory-degrade.
- `test-pocket-coreml --generated BUNDLE --cache-dir DIR` (a bundle with no
  prebuilt sidecar) generates both sidecars: routes and fit; with a fresh
  cache, a first call on ggml while they are pending (matching Core ML off)
  and ready afterwards; readiness at construction; the generated step and
  decoder against ggml (the gates below); adoption by a second Engine; fit
  unchanged by the populated cache; strict mode; memory-degrade.
- `test-pocket-coreml-shapes` (`POCKET_COREML_TEST_SHAPES`, written by
  `test/prepare_pocket_coreml_fixture.py`) runs two references:
  - a tiny real step against its FP32 reference, after copying prefix rows
    into the state, and again after a NaN row past the prefix (which
    rewriting the prefix does not clear) is reset away;
  - a random full-size decoder against its eager stream, in calls of several
    sizes, and again after a reset that follows a failed frame.

  It also checks that sidecars declaring any other shape, schema or checkpoint
  hash, or the wrong statefulness, are rejected before prediction. That is 42
  defective sidecars.
- `test-pocket-coreml-engine` (`POCKET_COREML_TEST_BUNDLE`, a bundle directory
  with both exported sidecars) compares both sidecars against ggml and checks
  the batched ggml replay of a retired step's frames. It also checks the
  `Engine` labels, stream versus batch, multi-chunk calls,
  `POCKET_COREML_DISABLE`, `steps > 1`, the strict refusal and the fit refusal.


## Reproduce numerical validation


The fixture generator calls Kyutai's actual neural modules at commit
`0c2db3bdea7c991c568989cc11b503f14483fabc`, pinned by the requirements file.
It verifies that revision before generating data. Its default small-weight
fixture needs no downloaded model or tokenizer. Its PyTorch oracle recomputes
full prefixes while the native implementation uses cached decoding.

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/dump-pocket-flow-reference.py \
  --output /tmp/pocket-reference
/tmp/pocket-venv/bin/python engines/tts/test/pocket/test_converter.py /tmp/pocket-reference

# GGML_PREFIX must contain an installed fabric ggml build compatible
# with the rest of this checkout, including the CPU registry graph-planner export.
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_EXECUTABLES=OFF \
  -DTTS_CPP_BUILD_TESTS=ON -DTTS_CPP_POCKET_REFERENCE_DIR=/tmp/pocket-reference
cmake --build build/pocket --target test-pocket-flow-lm test-pocket-load
ctest --test-dir build/pocket -R test-pocket- --output-on-failure
```

The native tests check embedding lookup, prefill, four cached audio steps,
EOS, 1/2/4/16 flow steps, denormalization, chunked prefill, reset, invalid
inputs, and a trajectory that feeds native sampled latents back into the LM.
The loader test needs no model files. Numerical tests are visibly disabled
by CMake when fixtures are absent; no reference downloads occur during build.

On a configured OpenCL host, set `-DTTS_CPP_POCKET_OPENCL_TESTS=ON` and all three
fixture directories to add explicit OpenCL runs of the existing FlowLM F32/F16,
Mimi, public engine, and metadata-only fit tests. These retain the CPU numerical
tolerances and cover cache/context boundaries, streaming, reset, cancellation,
repeated synthesis and error recovery. A missing OpenCL backend fails an enabled
OpenCL test; it cannot pass through CPU fallback. The fixture-free backend test
also checks the CPU default, allowlist and unavailable-OpenCL behavior.

For example, on the target host after installing its compatible ggml build:

```sh
cmake -S engines/tts -B build/pocket -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$GGML_PREFIX" -DTTS_CPP_BUILD_TESTS=ON \
  -DTTS_CPP_POCKET_OPENCL_TESTS=ON \
  -DTTS_CPP_POCKET_REFERENCE_DIR=/path/to/pocket-reference \
  -DTTS_CPP_POCKET_MIMI_REFERENCE_DIR=/path/to/pocket-mimi-reference \
  -DTTS_CPP_POCKET_MODEL_DIR=/path/to/pocket-bundle
cmake --build build/pocket --parallel
ctest --test-dir build/pocket -R 'test-pocket-.*opencl' --output-on-failure
```

For a real checkpoint, use the same generator with `--weights` and `--config`:

```sh
/tmp/pocket-venv/bin/python engines/tts/scripts/dump-pocket-flow-reference.py \
  --weights /path/to/model.safetensors --config /path/to/config.yaml \
  --output /tmp/pocket-real-reference
build/pocket/test-pocket-flow-lm /tmp/pocket-real-reference/flow-lm-f32.gguf \
  /tmp/pocket-real-reference 0.0001
```


For real-checkpoint and complete bundle tests, supply the fixtures expected
by `TTS_CPP_POCKET_REFERENCE_DIR`, `TTS_CPP_POCKET_MIMI_REFERENCE_DIR` and
`TTS_CPP_POCKET_MODEL_DIR`, then reconfigure. Inspect `ctest -N` and distinguish
disabled tests from executed tests. Exact test counts are revision-dependent.

## Native memory preflight


`tts-cpp/pocket/fit.h` exposes `pocket::fit_params(FitOptions)`. It tokenizes the
requested text, validates voice/context capacity, and prices the actual FlowLM
and Mimi graphs with ggml's size-only allocator on the selected CPU or GPU
backend. CPU projections additionally include the CPU work-buffer planner.
That planner is resolved through the CPU backend's registry, so preflight
works when CPU variants are dynamically loaded (Linux/Android).
CPU projection requires the `ggml_graph_plan` registry export from
[qvac-ext-ggml#92](https://github.com/tetherto/qvac-ext-ggml/pull/92);
older backends report an explicit preflight error instead of omitting scratch memory.
GGUF tensor payloads are never read and model/graph buffers are never allocated
or executed during preflight. Backend initialization can allocate its own scratch.
The estimate includes expanded F32 weights, state, overlapping worker compute,
host containers, thread stacks, load staging, and native batch PCM buffers.
It excludes application and SDK/JS copies. A prebuilt
[Core ML sidecar](#core-ml-sidecars) makes preflight return `Error`
with `coreml-memory-estimate-unavailable`. A generated one (with
`engine.coreml.cache_dir`, decided from the options, OS, build and template
match, never from the cache, so the result is the same before and after the
first generation) adds the templates' Core ML envelope to the projection: the
two resident parts (weights, I/O and the step's KV state) plus the larger of
the stages' prepare and load transients (one Core ML job runs at a time).
`FitResult::coreml` reports the route, the envelope, whether it was measured
for the resolved units, and the ggml-only verdict. A budget, either
`FitOptions::memory_budget_bytes` or `engine.coreml.memory_budget_bytes`
(the smaller when both are set), caps the available memory for every
verdict. Memory-degrade needs `engine.coreml.memory_budget_bytes`, the budget
the Engine decides it with: when only the ggml projection fits
min(free, that budget), fit reports `disabled_reason = "memory"` and the ggml
verdict, and the Engine then runs ggml only. A budget from
`FitOptions::memory_budget_bytes` alone (or one smaller than the Core ML
budget) never degrades, because that Engine would keep Core ML: fit charges
the Core ML envelope and reports `does-not-fit` when only the ggml projection
fits. Pass the budget as `engine.coreml.memory_budget_bytes` (as `pocket-cli`
does) to get the degrade.

With `coreml.memory_budget_bytes` set, the Engine's constructor repeats the
fit to decide memory-degrade. It uses `EngineOptions::fit_text` and
`fit_margin_bytes` (defaults `"Hello."` and 256 MiB, the `FitOptions`
margin): set them to the text and headroom of the `fit_params` call that
assessed the engine, so the engine reaches the same decision. `pocket-cli`
does this for its `--text`, `--fit-budget-mib` and `--fit-margin-mib`, in
`--fit` and in synthesis alike.

The envelope's constants are measured by `test-pocket-coreml --generated
BUNDLE --memory-schedule DIR`: ggml FlowLM and Mimi work while both sidecars
are generated, compiled, loaded and warmed in the background. Each stage's
prepare and load and warm-up, the steady state with both ready and
ANECompilerService's absolute size while Mimi loads need 1.25 times their
measurement within the constant that charges them, and the whole run its
measurement within the envelope; a run whose probes failed (no baseline, a
failed footprint sample, no compiler reading while Mimi loads) or that never
sampled a phase fails. Each constant is 1.25 times the largest value over
every run on every validated host (`template.json` `memory`), rounded up to
1 MiB, so a run that sets a new maximum fails until the constants are
recomputed. Validated on an Apple M2 (macOS 27.2.1, 27 runs) and the CI
runners' Apple M4 Max (macOS 26.7.2, 15 runs over five jobs on two Macs),
MiB:

| Phase | M2 | M4 Max | Charged |
|---|---|---|---|
| step prepare (generate + compile) | 57.8–80.5 | 57.3–82.5 | 104 |
| step load + warm-up (`cpu_and_gpu`) | 30.3–95.7 | 207.0–448.5 | 561 |
| Mimi prepare | 21.1–62.5 | 8.1–58.9 | 79 |
| Mimi load + warm-up (`cpu_and_ane`) | 0.8–49.9 | 8.1–50.4 | 64 |
| steady, both running | 48.0–103.3 | 250.5–458.5 | 574 (both resident parts, with the step's `workspace_bytes` 361.1) |
| whole run | 139.8–152.2 | 308.1–341.0 | envelope 1135.0 |
| ANECompilerService while Mimi loads (absolute) | 83.3–400.2 | 312.0–696.2 | reserve 871 |

On the M4 Max the step on `cpu_and_gpu` holds GPU working memory the M2 does
not, so every device is charged that larger footprint (the envelope is the
largest placement's, not each chip's). That working memory is bimodal on the
CI runners: some jobs measure the step's load and warm-up at 207–293 MiB and
the steady state at 250–275 MiB, others at 368–449 and 414–459 MiB, on both
Macs and for every run of a job (Parler's DAC on `all` shows the same two
modes in the same jobs), so the constants cover the larger. The compiler's
absolute size includes whatever else compiled for the Neural Engine on the
same Mac (no run found the service clean; on the CI runners the size before
Mimi's load ranged from 309.6 to 637.0 MiB depending on the jobs before it,
and its growth while Mimi loaded was 0.4–232.1 on the M4 Max and 19.8–41.6 on
the M2), so the reserve is conservative; it only gates when a cold Neural
Engine load may start, and is not part of fit. The CI macOS runner runs the
schedule three times on every pull request (`tts-ci.yml`).

Preflight is a memory projection, not a
validation of tensor contents, checkpoint encoder capability, or speech quality.

OpenCL fit uses the same selection rules as synthesis. Adreno OpenCL is treated
as shared system memory even though ggml advertises it as a GPU. Other devices
use ggml's shared-memory classification; discrete devices must fit both device
and host memory independently. Host availability is sampled before backend
initialization to charge that allocation once. OpenCL's reported free memory
is a backend estimate/cap, currently global memory minus 1 GiB in the reviewed
speech backend; on Adreno the result is also limited by OS available RAM.
Missing/invalid memory telemetry produces a preflight error.

The OpenCL state total includes one conservative shared-backend scratch bound,
reported separately in the text report. Its constants and F32 dispatch audit
come from qvac-ext-ggml speech `06f1b668`: the Adreno initialization pools total
395,149,312 bytes before device allocation caps. The estimate also bounds
independently retained weight/activation transpose and strided GEMM input
pools, including old allocations retained by queued work during growth.
Builds without those Adreno pools can therefore be overestimated. Re-audit
these constants and F32 dispatch paths when updating ggml; driver/kernel
compiler overhead is implementation-dependent and covered by the requested
margin rather than an exact measurement. This is not a live allocation trace.

Hexagon preflight uses the versioned ggml registry procedure
`ggml_backend_hexagon_get_runtime_info_v1`. Older builds without this procedure
fail explicitly for Hexagon, while CPU and OpenCL remain usable. The query
reads retained session configuration, so changing environment variables after
backend initialization cannot silently underprice existing queues or tracing
buffers. A configuration that disables HTP queueing or compute is rejected.

HTP RPC buffers share OS RAM; its zero-valued device telemetry is replaced by
OS available RAM sampled before backend loading. Backend-reported bounds cover
all initialized sessions' RPC queues and retained/prospective graph metadata.
Model and state estimates include backend tensor alignment, a conservative RPC
allocation count, guard pages and host page rounding. Each graph arena reserves
16 chunks' guard/page overhead, matching the reviewed allocator, and both
workers' arenas are counted concurrently. Individual tensors must fit the
selected session's buffer limit; the conservative sum of live model buffers and
arenas must fit its batch mapping limit. VTCM is reported separately and graph
capability checks enforce the backend's available VTCM constraints. VTCM is not
charged as ordinary OS RAM. Opaque driver/DSP heaps and app copies still require
the requested safety margin. Re-audit graph chunking when updating ggml.

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --fit --fit-margin-mib 256 \
  --fit-budget-mib 1024 --report /tmp/pocket-fit.json
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle --gpu \
  --text "Hello from Pocket TTS." --fit --report /tmp/pocket-metal-fit.json
```

The optional budget further limits available OS memory. On iOS, the projection
also respects `os_proc_available_memory()` when supported; a zero app allowance
returns does-not-fit. Linux uses MemAvailable; container/cgroup limits are not
automatically detected, so hosts must supply their application ceiling. The CLI
returns 0 for fit, 1 for no-fit, and 2 for a preflight error. Native callers receive
`FitStatus` and a reason/report. Fit remains a point-in-time estimate; another
process can consume memory between preflight and loading.

For an OpenCL preflight add `--backend opencl` to that command. For device
benchmarks, run the CPU and OpenCL commands with identical assets, text,
threads, steps and seed. Record the model hashes, phone/GPU, OS and driver,
build options, load time, first-run and warm first-audio latency/RTF, and peak
host/device memory. `--runs 1` records the first synthesis; `--runs 3` performs
one warmup and records three warm runs. Validate output with the existing
`scripts/validate-pocket-audio.py` workflow and listen for defects. Hardware
correctness, audio quality and speedup require measured results on the target
device; host CPU checks do not establish them.


Reference input is bounded before model allocation. Accepted reference input is uncompressed
PCM8/16/24/32 or float32/64 WAV, 1–64 channels, 8–192 kHz, at most 30 seconds and
64 MiB on disk.

## Performance and integration evidence

The [OnePlus 13s OpenCL report](history/pocket-opencl-adreno.md) records
CPU/OpenCL correctness, speech checks, first-use and warm timings, and memory
preflight on Adreno 830. Both stages passed device validation. It retains the
initial timings and an expanded 60-request comparison, which observed a
1.08–1.26x median OpenCL throughput advantage with substantial run variability.
The report includes the standalone Android build and driver setup used for
reproduction.

The [archived integration report](history/pocket-integration.md#final-registry-build-and-release-review)
records the September 2026 pinned-registry Apple M2 comparison: generation
5–16% above upstream, native RTF about 0.18–0.19, and first audio 74–116 ms.
Those warm-cache measurements supersede the earlier manual-build slowdown;
SDK startup and transport latency are additional.

The archive also records native, addon, Node SDK socket transport and iOS
Simulator/worklet validation at specific source pins. Physical-device,
Android and broader voice/listening coverage remain limitations of that
record; it does not establish current package publication or platform parity.
For public SDK usage and current package versions, see
[QVAC](https://github.com/tetherto/qvac).
