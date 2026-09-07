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
| `nvidia/parakeet-unified-en-0.6b` | RNN-T | 128 | 1024 × 24 | 1024 | 600 M | 707 MiB q8_0 | 0.004 q8_0 Vulkan / 0.028 q8_0 Metal | English; offline full-context encoder |
| `nvidia/parakeet-tdt-0.6b-v3` | TDT | 128 | 1024 × 24 | 8192 | 600 M | 715 MiB q8_0 / 1.34 GiB f16 | 0.006 q8_0 Metal | About 25 languages, with punctuation and capitalization |
| `nvidia/parakeet-tdt-1.1b` | TDT | 80 | 1024 × 42 | 1024 | 1.1 B | 1225 MiB q8_0 | 0.027–0.079 Metal | English only; no punctuation or capitalization |
| `nvidia/parakeet_realtime_eou_120m-v1` | RNN-T + `<EOU>` | 128 | 512 × 17 | 1027 | 120 M | 246 MiB f16 / 132 MiB q8_0 | 0.0052 Vulkan | English ASR and native end-of-turn token |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | Prompt-conditioned RNN-T | 128 | 1024 × 24 | 13087 | 600 M | ~1.3 GiB f16 | 0.108 CPU | Locale-conditioned ASR; empty language selects `auto`; cache-aware streaming at 80/160/320/560/1120 ms |
| `nvidia/diar_sortformer_4spk-v1` | Sortformer | 80 | 512 × 18 | n/a | 123 M | 263 MiB f16 / 141 MiB q8_0 / 75 MiB q4_0 | 0.0020 Vulkan | Up to four speakers; offline and sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2` | Sortformer | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Streaming-trained; sliding-history streaming |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | Sortformer + AOSC | 128 | 512 × 17 | n/a | 117 M | 251 MiB f16 / 134 MiB q8_0 / 72 MiB q4_0 | similar to v1 offline | Audio-Online Speaker Cache preserves slots across long gaps |

TDT 0.6B-v3 and TDT 1.1B are distinct model contracts: only 0.6B-v3 is
multilingual and punctuation/capitalization-aware. Encoder topology, including
causal subsampling, convolution normalization, and chunked-limited attention,
comes from GGUF metadata.

Unified RNN-T uses standard greedy transducer decoding. Its encoder was trained
with dynamic chunked convolution and attention, but this implementation runs it
offline in full-context mode. Mode 2 and `StreamSession` use buffered window
re-encoding; native NeMo cache-aware encoder state is not implemented.

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

## Build modes

### Standalone engine build

The standalone build owns its ggml dependency. From the repository root, clone
the pinned `qvac-ext-ggml@speech` branch into `engines/parakeet/ggml`, then
configure the engine directly:

```bash
engines/parakeet/scripts/setup-ggml.sh
cmake -S engines/parakeet -B build-parakeet -DCMAKE_BUILD_TYPE=Release
cmake --build build-parakeet -j
```

Standalone defaults enable the library, CLI, tests, and examples. Outputs are
under the selected build directory, for example:

```text
build-parakeet/parakeet
build-parakeet/live-mic
build-parakeet/live-mic-attributed
build-parakeet/test-*
```

`setup-ggml.sh` does not apply local patches. Backend filename-prefix loading,
non-Adreno OpenCL support, and the OpenCL program-binary cache now live as
commits on [`qvac-ext-ggml@speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech).

### Umbrella speech-stack build

The repository-level build requires an installed `ggml-speech` package and
forces `PARAKEET_USE_SYSTEM_GGML=ON`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-speech/install
cmake --build build -j
```

The Parakeet CLI is then `build/engines/parakeet/parakeet`. The umbrella
defaults follow `SPEECH_BUILD_EXECUTABLES` and `SPEECH_BUILD_TESTS`, and force
`PARAKEET_BUILD_EXAMPLES=OFF`. Enable examples only in a direct engine build.

### CMake options

| Option | Standalone default | As an umbrella subdirectory | Effect |
|---|---:|---:|---|
| `PARAKEET_BUILD_LIBRARY` | `ON` | `ON` | Build `qvac-parakeet`; linkage follows `BUILD_SHARED_LIBS`, otherwise static |
| `PARAKEET_BUILD_EXECUTABLES` | `ON` | inherited from `SPEECH_BUILD_EXECUTABLES` | Build target `parakeet-cli`, output name `parakeet` |
| `PARAKEET_BUILD_TESTS` | `ON` | inherited from `SPEECH_BUILD_TESTS` | Build and register test harnesses |
| `PARAKEET_BUILD_EXAMPLES` | `ON` | `OFF` | Build `live-mic` and `live-mic-attributed` |
| `PARAKEET_INSTALL` | `ON` | `ON` | Generate library, headers, CMake package, and pkg-config install rules |
| `PARAKEET_USE_SYSTEM_GGML` | `OFF` | `ON` | Use `find_package(ggml)` instead of `engines/parakeet/ggml` |
| `PARAKEET_GGML_LIB_PREFIX` | `ON` | no effect with system ggml | Name bundled libraries `speech-ggml-*` |
| `PARAKEET_COREML` | `OFF` | `OFF` | Apple-only offline TDT encoder sidecar |
| `PARAKEET_OPENMP` | `ON` | `ON` | Link OpenMP when available; auto-disabled on Windows non-MinGW unless explicitly overridden |
| `PARAKEET_FLASH_ATTN` | Metal, CUDA and Vulkan `ON`, otherwise `OFF` | same | Fused encoder attention with the rel-pos bias folded into the mask; selected per backend at load, the CPU path always keeps the unfused graph |
| `PARAKEET_CCACHE` | `ON` | `ON` | Use ccache for Parakeet targets when available |

### Installed package

With `-DCMAKE_INSTALL_PREFIX=<prefix>`, `cmake --install <build-dir>` installs:

```text
<prefix>/include/parakeet/*.h
<prefix>/lib/libqvac-parakeet.*
<prefix>/lib/cmake/qvac-parakeet/
<prefix>/lib/pkgconfig/qvac-parakeet.pc
```

The existing downstream fixture in `test/consumer/CMakeLists.txt` uses:

```cmake
find_package(qvac-parakeet CONFIG REQUIRED)
target_link_libraries(consumer PRIVATE qvac::parakeet)
```

Configure consumers with both the Parakeet and ggml prefixes when they are
separate:

```bash
cmake -S engines/parakeet/test/consumer -B consumer-build \
  -DCMAKE_PREFIX_PATH="/path/to/parakeet;/path/to/ggml"
cmake --build consumer-build
```

For pkg-config, use `--static` when linking a static package so private OpenMP
or Apple framework dependencies are included:

```bash
export PKG_CONFIG_PATH=/path/to/parakeet/lib/pkgconfig:/path/to/ggml/lib/pkgconfig
pkg-config --cflags qvac-parakeet
pkg-config --static --libs qvac-parakeet
```

The complete install-tree check is:

```bash
engines/parakeet/scripts/test-package-consumption.sh --help
engines/parakeet/scripts/test-package-consumption.sh \
  --ggml-prefix /path/to/ggml-install
```

Options are `--ggml-prefix <dir>` (required), `--work-dir <dir>`, `--coreml`
(Apple only), and `--keep`. The script verifies artifact names, the CMake
consumer, and static pkg-config consumption. Set
`PARAKEET_PKGTEST_CMAKE_ARGS` for additional CMake definitions such as an
explicit OpenMP installation.

## Backends and runtime selection

Any combination of Metal, CUDA, Vulkan, and OpenCL may be built or dynamically
loaded. Parakeet selects one primary GPU through the ggml backend registry when
`n_gpu_layers > 0`; this is not a fixed compile-time CUDA/Metal/Vulkan/OpenCL
cascade. `n_gpu_layers` is currently a boolean offload request: positive means
the whole encoder, not a partial layer count.

Runtime tiering is:

1. Prefer OpenCL for Adreno 700+, where it is validated ahead of Vulkan.
2. Otherwise choose the first registered non-OpenCL GPU, such as Vulkan,
   Metal, CUDA, or Mali Vulkan.
3. Use non-Adreno OpenCL only when no non-OpenCL GPU can initialize.
4. Fall back to CPU.

Adreno 6xx OpenCL is skipped because it produces incorrect output. Set
`PARAKEET_ALLOW_ADRENO_6XX=1` to opt in explicitly. Mali uses Vulkan for the
encoder and CTC/TDT/EOU computation, but the Sortformer diarization head is
routed to CPU because that head is incorrect on Mali Vulkan.

TDT and EOU predictor/joint decoding uses ggml graphs on Metal, Vulkan, and
CUDA. CPU and OpenCL use the scalar decoder path; OpenCL lacks the graph
operation support required by this decoder path. The EOU encoder can still run
on OpenCL while its decoder runs scalar.

The graph decoder adapts to what the active backend reports through
`ggml_backend_supports_op`, probed once at load. Where the backend runs the
fused LSTM cell (`GGML_OP_LSTM_CELL`) and the transducer step control
(`GGML_OP_TDT_STEP`), which ggml-speech implements for CPU, CUDA and Metal, the
TDT decoder runs up to eight greedy steps per graph: the joint, argmax, LSTM
update, duration bookkeeping and state selection stay on the device and the
host reads the emitted tokens back once per graph. A backend without those
ops keeps one graph per step. Either path produces the same token sequence as
the sequential loop; the `test-tdt-unroll-parity` and `test-tdt-lstm-parity`
harnesses guard that. The encoder applies the same rule to the conformer's
depthwise convolution (`GGML_OP_CONV_2D_DW` in place where the backend
reports it, `im2col` and matmul elsewhere; the subsampler switches only on a
GPU that passed the probe, CPU keeps its previous lowering) and to the gated
GLU (`a * sigmoid(b)` as one op where the backend reports it). Fused attention
is a build option (`PARAKEET_FLASH_ATTN`) that CUDA, Metal and Vulkan take, and
only after the backend accepts the exact node the encoder builds; CPU and
OpenCL keep the unfused graph in every build. The mel front-end runs on up
to eight host threads with output byte-equal to the single-thread result.

The CUDA path was validated on an RTX 3080 (TDT q8_0 and q4_0 transcripts,
Sortformer and streaming output byte-equal to the pre-change build, LibriSpeech
WER within noise of the CPU reference) but hardware decoder parity is not yet
covered by CI.

`GGML_BACKEND_DL=ON` builds load backend modules at runtime. Android defaults
this on and builds Vulkan, OpenCL, and CPU variants. Pass a module directory via
CLI `--backends-dir` or `EngineOptions::backends_dir`. Backend registration is
process-global: the first `Engine` construction loads from that directory and
later engines reuse the populated registry. Leave it empty for ggml's default
search path. In static `GGML_BACKEND_DL=OFF` builds the setting is a no-op.

Use `Engine::backend_device()`, `backend_name()`, and `encoder_backend()` to
observe the post-fallback result rather than inferring it from build flags.

Example backend configurations:

```bash
cmake -S engines/parakeet -B build-metal -DGGML_METAL=ON
cmake -S engines/parakeet -B build-cuda -DGGML_CUDA=ON
cmake -S engines/parakeet -B build-vulkan -DGGML_VULKAN=ON
cmake -S engines/parakeet -B build-opencl -DGGML_OPENCL=ON
```

## Core ML encoder sidecar

`PARAKEET_COREML=ON` is Apple-only. It enables an optional offline TDT
FastConformer encoder sidecar; CTC and EOU do not use it. Mel preprocessing and
TDT decoding remain in the normal pipeline. The compiled sidecar must sit next
to the GGUF and use this name:

```text
<model-basename-with-quant-stripped>-encoder.mlmodelc
```

For example, both `parakeet-tdt-0.6b-v3.f16.gguf` and
`parakeet-tdt-0.6b-v3.q8_0.gguf` resolve to
`parakeet-tdt-0.6b-v3-encoder.mlmodelc`.

Export and compile a fixed-shape sidecar:

```bash
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-tdt-0.6b-v3.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --out engines/parakeet/models/parakeet-tdt-0.6b-v3-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

Benchmark fixed lengths and inspect ANE/GPU/CPU placement:

```bash
python engines/parakeet/scripts/bench-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet-tdt-0.6b-v3.f16.gguf \
  --mel-frames 138 826 2201
```

The default export is fixed-shape and accelerates only the exported mel length;
other lengths fall back to ggml. `--flexible` exports a RangeDim model, but it
is a correctness/experimentation path: measured flexible graphs place no
operations on ANE and can be substantially slower than ggml Metal.

A missing sidecar, load failure, incompatible shape, or runtime prediction
failure falls back to the ggml encoder. Set `PARAKEET_COREML_DISABLE=1` to
force ggml, including for parity or benchmarking.

## Models and conversion

The engine runs GGUF, not `.nemo`. `scripts/download-all-models.sh` downloads
NeMo archives only; every downloaded checkpoint must still be converted.
Python is required only for conversion, Core ML export, and NeMo parity tooling;
runtime inference remains pure C++.

Create an isolated environment using a Python version supported by the selected
NeMo release:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch gguf numpy pyyaml soundfile librosa sentencepiece \
  "nemo_toolkit[asr]" huggingface_hub
```

Convert the downloaded checkpoint:

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet-ctc-0.6b.nemo \
  --out engines/parakeet/models/parakeet-ctc-0.6b.q8_0.gguf \
  --quant q8_0
```

The converter defaults to CTC 0.6B and `q8_0`. When `--out` is omitted, it
derives `models/parakeet-ctc-0.6b.<quant>.gguf` from `--quant`. For every other
checkpoint, pass an explicit `--ckpt`, `--hf-repo`, and `--out`; otherwise a
missing local checkpoint can cause the default CTC repository to be downloaded.
Keep the quantization in explicit filenames:

```text
<model>.q8_0.gguf
<model>.f16.gguf
```

Use f16 for numerical parity against NeMo references and q8_0 for normal runtime
fixtures. Small tensors or dimensions unsuitable for block quantization remain
f16. Hybrid IndicConformer exports are CTC-only and include per-language token
ranges.

Recorded CTC 0.6B quantization results on an M4 Air CPU:

| Quantization | Size | 20-second encoder | 11-second encoder | Transcript |
|---|---:|---:|---:|---|
| f32 | 2.4 GiB | n/a | n/a | exact |
| f16 | 1.3 GiB | 1221 ms | ~680 ms | bit-equal |
| q8_0 | 697 MiB | 839 ms | 460 ms | bit-equal |
| q5_0 | 453 MiB | 1475 ms | ~650 ms | bit-equal |
| q4_0 | 372 MiB | 1080 ms | 595 ms | bit-equal |

Every supported checkpoint is covered by this local pipeline:
`scripts/download-all-models.sh` fetches the `.nemo` archive (including the
AI4Bharat IndicConformer hybrid) and `scripts/convert-nemo-to-gguf.py` turns it
into the runnable GGUF. The IndicConformer conversion emits the per-language
token ranges the run-time `--language` selection needs:

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
  --out engines/parakeet/models/indic-conformer-600m-multilingual.q8_0.gguf \
  --quant q8_0
```

## Public C++ API

Include `<parakeet/parakeet.h>` for the complete surface or individual headers.

| API | Purpose |
|---|---|
| `Engine::transcribe` | One-shot WAV transcription for CTC, RNN-T, TDT, EOU, or Nemotron |
| `Engine::transcribe_samples` | One-shot float PCM transcription |
| `Engine::transcribe_stream` / `transcribe_samples_stream` | Chunked callbacks; one offline encode for legacy families, native cache-aware streaming for Nemotron |
| `Engine::stream_start` | Live session; rolling-window re-encoding for legacy families, native cache-aware streaming for Nemotron |
| `StreamSession::feed_pcm_f32` / `feed_pcm_i16` | Push arbitrary-sized PCM blocks |
| `Engine::diarize` / `diarize_samples` | Offline Sortformer diarization |
| `Engine::diarize_start` | Live Sortformer session |
| `transcribe_with_speakers` / `transcribe_samples_with_speakers` | CTC, RNN-T, TDT, or EOU transcription attributed with Sortformer |
| `EngineOptions::prewarm` / `prewarm_audio_seconds` | Run a configurable encoder-only synthetic forward during construction |
| `parakeet_log_set` | Install the host logging sink |
| `parakeet_cli_main` | Embed the same CLI entry point exported by the library |
| `fit_params` | Memory-fit preflight: project whether a GGUF + workload fits the device without loading weights (`<parakeet/fit.h>`) |
| `parakeet_fit_cli_main` | Embed the `parakeet-fit-params` tool entry point |

Minimal one-shot transcription:

```cpp
#include <parakeet/parakeet.h>

#include <cstdio>

int main() {
    parakeet::EngineOptions options;
    options.model_gguf_path = "models/parakeet-ctc-0.6b.q8_0.gguf";

    parakeet::Engine engine(options);
    const auto result = engine.transcribe("audio.wav");
    std::puts(result.text.c_str());
}
```

Mode 3 is duplex rolling-context/sliding-window re-encoding for CTC, RNN-T, TDT,
and EOU. It re-runs the encoder for each window using `left_context_ms` and
`right_lookahead_ms`; it does not maintain an encoder KV cache or convolution
cache. Nemotron Mode 3 uses the cache-aware encoder instead and ignores those
window knobs.

Always call `finalize()` to drain the partial tail. Destroying a
`StreamSession` or `SortformerStreamSession` cancels it and does not finalize
it. `cancel()` stops future work; `Engine::cancel()` may be called from another
thread, but inference calls on one Engine are otherwise not concurrent-safe.
Cancel, join the worker, and only then destroy an Engine; destruction does not
wait for in-flight calls.

`StreamingCallback` and `StreamEventCallback` run synchronously from the API
call processing the audio. `StreamingSegment::is_eou_boundary` and
`StreamEventType::EndOfTurn` mean the EOU model emitted `<EOU>`. They are not
VAD state changes. Sortformer emits `VadStateChanged` from speaker
probabilities. CTC/RNN-T/TDT/Nemotron can opt into RMS energy VAD with
`enable_energy_vad`; configure it with `energy_vad_threshold_db`,
`energy_vad_window_ms`, and `energy_vad_hangover_ms`.

Long-form one-shot transcription uses bounded encoder windows when
`long_form_window_frames` is exceeded and trims
`long_form_context_frames` at seams. Short inputs retain the single-pass path.

## Sortformer AOSC

AOSC is primarily activated by the explicit GGUF metadata
`parakeet.model_variant=sortformer-streaming-v2.1-aosc`. Encoder shape
`n_layers=17` and `n_mels=128` is only a legacy fallback for GGUFs created
before that metadata key existed. Convert current v2.1 checkpoints again to
avoid relying on the heuristic.

`SortformerStreamingOptions` exposes `spkcache_enable`, `spkcache_len`,
`fifo_len`, `chunk_left_context_ms`, `chunk_right_context_ms`, and
`spkcache_update_period`. After `diarize_start()`,
`SortformerStreamSession::aosc_active()` reports whether the session actually
took the AOSC path.

Every non-cancelled `finalize()` drains any trailing partial chunk and then
emits exactly one final synthetic terminator (`speaker_id=-1`,
`is_final=true`, and `start_s == end_s`). Real speaker segments always remain
non-final. Repeated `finalize()` calls are idempotent, while cancellation
suppresses the terminator.

## CLI and microphone examples

The file CLI accepts:

```text
parakeet --model <model.gguf> (--wav <16-kHz-mono.wav> |
         --pcm-in <raw> --pcm-format s16le|f32le --pcm-rate 16000) [options]
```

Useful groups include `--threads`, `--n-gpu-layers`, `--backends-dir`,
`--language`, `--stream`, `--stream-duplex`, context/chunk options,
`--diarization-model`, OpenCL environment controls, `--bench`, `--profile`,
and `--dump-mel`. Run `parakeet --help` for the complete list.

```bash
build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --n-gpu-layers 1

build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --stream --stream-duplex --emit jsonl

build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diarization-model engines/parakeet/models/diar_sortformer_4spk-v1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# standalone Sortformer diarization: speaker segments only, no ASR model
build-parakeet/parakeet \
  --model engines/parakeet/models/diar_sortformer_4spk-v1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# IndicConformer: --language is required and selects the token range
build-parakeet/parakeet \
  --model engines/parakeet/models/indic-conformer-600m-multilingual.q8_0.gguf \
  --wav engines/parakeet/test/samples/hi-16k.wav --language hi

# Nemotron: locale alias or auto; empty also selects auto
build-parakeet/parakeet \
  --model engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --language en-US
```

`live-mic` runs CTC/RNN-T/TDT/EOU/Nemotron transcription or Sortformer diarization.
`live-mic-attributed` combines a CTC/RNN-T/TDT/EOU ASR model with Sortformer.
Both capture 16 kHz mono through miniaudio, accept independent streaming and
backend options, and finalize tail audio on Ctrl-C.

## Memory-fit preflight (`parakeet-fit-params`)

`parakeet-fit-params` projects whether a model fits the device memory
available right now, **without loading any weights**: it reads only GGUF
metadata, drives the same loader/graph builders as a real run through ggml's
size-only allocator APIs, and compares the projection against the device's
free memory. Library API in `<parakeet/fit.h>` (`parakeet::fit_params`);
status semantics follow the SDK's `@qvac/model-fit` contract
(0 = fits, 1 = does not fit, 2 = error — also the exit code).

```bash
# CPU projection for the default 300 s workload
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf

# GPU projection, machine-readable
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --n-gpu-layers 1 --json
```

The workload (`--audio-seconds`, default 300) sets the longest single
transcribe input to project for. Device-side memory is bounded by the
long-form encoder window, so the device projection saturates once the audio
exceeds one window; host-side buffers (full-input mel, stitched encoder
output) keep growing with it. The projection prices the encoder graph
*cache*, not one graph: a windowed pass keeps up to three window graphs
resident (`run_encoder`'s 3-slot LRU), and all of them are counted.
Sortformer diarization does not window — its device projection grows with
the full input (`O(T^2)` head attention).

The projection is exact where it can be: `test-fit-params` asserts the
projected weight, encoder-compute, and Sortformer-head bytes equal what a
real load/encode/diarize allocates, byte for byte. The projection covers the
offline paths; streaming sessions build smaller per-chunk graphs but rotate
through the same graph cache with session-dependent keys, so treat the
offline projection as a guide, not a proven bound, for streaming.

## Tests and NeMo parity

Set up fixtures in this order:

1. Download the required `.nemo` archive.
2. Convert it to the exact q8_0 runtime or f16 parity GGUF filename expected by
   `CMakeLists.txt`.
3. Generate NeMo `.npy` references with the matching `dump-*-reference.py`
   script when the test requires references.
4. Configure after fixtures exist so CTest registrations detect them.
5. Build, inspect `ctest -N`, then run CTest.

```bash
engines/parakeet/scripts/download-all-models.sh

python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet-ctc-0.6b.nemo \
  --out engines/parakeet/models/parakeet-ctc-0.6b.f16.gguf --quant f16

python engines/parakeet/scripts/dump-ctc-reference.py \
  --wav engines/parakeet/test/samples/jfk.wav

cmake -S engines/parakeet -B build-parakeet -DCMAKE_BUILD_TYPE=Release
cmake --build build-parakeet -j
ctest --test-dir build-parakeet -N
ctest --test-dir build-parakeet --output-on-failure
```

`download-all-models.sh` produces `.nemo` files, not runnable GGUFs. Missing
model, audio, or reference fixtures cause individual tests to be registered as
`DISABLED`, not failed. Configure again after adding a fixture. Each test
carries exactly one label out of `unit`, `fixture`, `cpu`, `gpu`, and `perf`.
The fixture-free suite (the one CI runs) is selected by excluding the
GPU-bound and timing-bound labels:

```bash
ctest --test-dir build-parakeet -LE 'gpu|perf' --output-on-failure
```

Model-free logic tests (`-L unit`) cover the CTC language mask, mel FFT
parity and per-feature CMVN, RNN-T graph construction, long-form window
planning, Sortformer finalization and probability thresholding, the
streaming energy VAD, the SentencePiece detokenizer, and the NeMo
converter (Python).

Fixture roots are configurable with `PARAKEET_TEST_MODEL_DIR`,
`PARAKEET_TEST_AUDIO_DIR`, and `PARAKEET_TEST_REF_DIR`. `test-gpu-vs-cpu`
(formerly `test-vk-vs-cpu`) is available when Vulkan or Metal is configured.
`verify-gguf-roundtrip.py`,
`ref-encoder-from-gguf.py`, and `streaming-reference.py` support converter and
parity investigation.

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

## Repository layout

| Path | Role |
|---|---|
| `CMakeLists.txt` | Library, CLI, tests, examples, and install package |
| `cmake/` | CMake package and pkg-config templates |
| `include/parakeet/` | Public API |
| `src/` | Engine, decoders, backend selection, preprocessing, and CLI |
| `examples/` | Microphone examples |
| `test/consumer/` | Installed-package consumer fixture |
| `scripts/` | ggml setup, conversion, Core ML, references, and package validation |
| `ggml/` | Standalone bundled checkout of `qvac-ext-ggml@speech` |
| `models/`, `artifacts/`, `test/samples/` | Local/runtime test fixtures |
| `PROGRESS.md` | Detailed implementation and parity history |

## License

Code is Apache-2.0. CTC, RNN-T, TDT, and Sortformer weights are CC-BY-4.0 unless
their model card says otherwise. `parakeet_realtime_eou_120m-v1` uses the
NVIDIA Open Model License. Nemotron 3.5 ASR Streaming 0.6B uses OpenMDW-1.1.
No weights are shipped by this repository.
