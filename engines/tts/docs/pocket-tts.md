# tts engine: Pocket TTS

Part of the [tts documentation](../README.md).

Pocket provides English text-to-speech through a persistent CPU FlowLM/Mimi
engine and incremental 24 kHz mono PCM callbacks. Its public API is
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
FlowLM/Mimi on separate CPU workers with at most 16 queued latent frames.
The Fabric seed uses a specified Box–Muller transform. An encoder-enabled
checkpoint can use `--reference-audio` instead of the prepared voice; the
tested public checkpoint rejects that path.

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

# GGML_PREFIX must contain an installed qvac-ext-ggml speech build compatible
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
and Mimi graphs with ggml's size-only allocator and CPU work-buffer planner.
The planner is resolved through the selected CPU backend's registry, so
preflight works when CPU variants are dynamically loaded (Linux/Android).
This requires the `ggml_graph_plan` registry export from
[qvac-ext-ggml#92](https://github.com/tetherto/qvac-ext-ggml/pull/92);
older backends report an explicit preflight error instead of omitting scratch memory.
GGUF tensor payloads are never read, allocated, or executed during preflight.
The estimate includes expanded F32 weights, state, overlapping worker compute,
host containers, thread stacks, load staging, and native batch PCM buffers.
It excludes application and SDK/JS copies. This is a memory projection, not a
validation of tensor contents, checkpoint encoder capability, or speech quality.

```sh
build/pocket/pocket-cli --model-dir /path/to/pocket-bundle \
  --text "Hello from Pocket TTS." --fit --fit-margin-mib 256 \
  --fit-budget-mib 1024 --report /tmp/pocket-fit.json
```

The optional budget further limits available OS memory. On iOS, the projection
also respects `os_proc_available_memory()` when supported; a zero app allowance
returns does-not-fit. Linux uses MemAvailable; container/cgroup limits are not
automatically detected, so hosts must supply their application ceiling. The CLI
returns 0 for fit, 1 for no-fit, and 2 for a preflight error. Native callers receive
`FitStatus` and a reason/report. Fit remains a point-in-time estimate; another
process can consume memory between preflight and loading.


Reference input is bounded before model allocation. Accepted reference input is uncompressed
PCM8/16/24/32 or float32/64 WAV, 1–64 channels, 8–192 kHz, at most 30 seconds and
64 MiB on disk.

## Performance and integration evidence

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
