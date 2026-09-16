# tts engine: APIs, pipelines, and surfaces

Part of the [tts engine documentation](../README.md).

### API, streaming, and CLI matrix

| Family | Installed header | Streaming API | CLI surface |
|---|---|---|---|
| Chatterbox | `<tts-cpp/chatterbox/engine.h>` | true incremental S3Gen/HiFT chunks | `tts-cli` batch and streaming |
| Supertonic 1/2/3 | `<tts-cpp/supertonic/engine.h>` | text-chunk pipeline streaming | `supertonic-cli` batch/streaming; `tts-cli` batch only and rejects streaming flags |
| Parler | `<tts-cpp/parler/engine.h>` | incremental callback via `stream_chunk_frames` | `parler-cli` and `tts-cli` batch only; neither exposes streaming |
| CosyVoice3 | `<tts-cpp/cosyvoice/engine.h>` | callback is post-hoc chunking after full generation | `cosyvoice-cli` |
| Audio8 | `<tts-cpp/audio8/engine.h>` | no | `audio8-cli` |
| Pocket TTS | `<tts-cpp/pocket/engine.h>` | incremental FlowLM/Mimi PCM callbacks | `pocket-cli` batch WAV and memory preflight |
| LavaSR | `<tts-cpp/lavasr/{denoiser,enhancer}.h>` | block-oriented enhancement APIs | `lavasr-bench` |

`tts-cli` is intentionally a limited metadata dispatcher: it handles
Chatterbox, batch Supertonic, and a reduced Parler surface. Use the dedicated
CLIs for the complete Supertonic, Parler, CosyVoice3, and Audio8 options.

### Defaults that differ by surface

| Setting | `tts-cli` | Library engines and dedicated CLIs |
|---|---|---|
| Seed | `0` unless `--seed` is supplied | generally `42` |
| Threads | capped at 4 by default | Chatterbox, Supertonic, Parler, and Audio8 cap automatic selection at 4; CosyVoice stage behavior is not generalized here |
| Parler greedy | repaired to sampling with a warning because argmax does not terminate | same; use `seed` for reproducibility |
| Chatterbox streaming CFM | Turbo accepts 1 or 2 steps | Multilingual standard CFM requests below the model timestep count are floored to 10 |

## Public APIs

The installed CMake target is `tts-cpp::tts-cpp`. Persistent engine classes
load their model once and reuse it across synthesis calls:

| Namespace | Primary surface | Result |
|---|---|---|
| `tts_cpp::chatterbox` | `Engine::synthesize` | 24 kHz PCM by default; Turbo/Multilingual selected by GGUF metadata |
| `tts_cpp::supertonic` | `Engine::synthesize` | model-rate PCM, usually 44.1 kHz |
| `tts_cpp::parler` | `Engine::synthesize` | 44.1 kHz PCM conditioned by a description |
| `tts_cpp::cosyvoice` | `Engine::synthesize` | 24 kHz PCM |
| `tts_cpp::audio8` | `Engine::synthesize` | 44.1 kHz PCM, optionally cloned from `VoicePrompt`; `Engine::codec_on_coreml()` reports whether the Core ML codec sidecar loaded and `SynthesisResult::codec_synthesis_backend` where each call's codec synthesis ran (`"ggml"` or a `coreml-*` label), see the [Audio8 guide](audio8.md#core-ml-codec-sidecar) |
| `tts_cpp::lavasr` | `Denoiser` / `Enhancer` | enhanced PCM |

The public surface also includes Chatterbox's lower-level
`s3gen_synthesize_to_wav`, `s3gen_preload`, and `s3gen_unload`, plus
`tts_cpp_cli_main`. Symbols without `TTS_CPP_API`, including `detail`
namespaces used by tests, are private and hidden from shared-library consumers.

### Consumer integration

Downstream projects in the QVAC speech stack consume `tts-cpp` via
the matching `tts-cpp` vcpkg port (this in-tree subtree).  ggml comes
from the [`ggml-speech`](https://github.com/tetherto/qvac-registry-vcpkg)
sister port, which vendors the
[`qvac-ext-ggml/speech`](https://github.com/tetherto/qvac-ext-ggml/tree/speech)
branch with all Metal / OpenCL / Vulkan patches pre-applied.  Once
both ports are installed, integration on the consumer side is one
`find_package` call:

```cmake
find_package(tts-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tts-cpp::tts-cpp)
```

```cpp
#include <tts-cpp/chatterbox/engine.h>

tts_cpp::chatterbox::EngineOptions opts;
opts.t3_gguf_path    = "models/chatterbox-t3-turbo.gguf";
opts.s3gen_gguf_path = "models/chatterbox-s3gen.gguf";
opts.n_gpu_layers    = 99;
tts_cpp::chatterbox::Engine engine(opts);
auto result = engine.synthesize("Hello, world.");
// result.pcm is 24 kHz mono float32 PCM ready to be written as wav.
```

The supertonic test/bench harnesses link against `tts-cpp` directly
and use detail-namespaced symbols outside the `TTS_CPP_API` public
surface, so the integrated port keeps the default
`TTS_CPP_BUILD_SHARED=OFF` and `TTS_CPP_BUILD_TESTS=OFF`.  See
[**Useful CMake options**](build.md#useful-cmake-options) below for the full
flag table. The streaming text chunker (`split_for_streaming`:
sentence/clause/whitespace boundary priority, first-chunk latency knob,
tiny-tail merge, CJK sentence ends) is pinned model-free by
`test-supertonic-chunker`.

## Pipelines

```
      text                                                 24 kHz wav
       │                                                        ▲
       ▼                                                        │
  ┌────────────────────────────────────────────────────────────────┐
  │                       tts-cli (libtts-cpp)                     │
  │                                                                │
  │      T3      ──►   S3Gen encoder   ──►        CFM              │
  │  text → toks       toks → h                   h → mel          │
  │                                                                │
  │                         HiFT vocoder  ──►  24 kHz wav          │
  └────────────────────────────────────────────────────────────────┘
       ▲                                              ▲
   text tokenizer                              reference voice
   (embedded in T3 GGUF metadata)              (embedded in S3Gen GGUF)
```

`tts-cli` handles both Chatterbox variants via `chatterbox.variant` metadata.
There is no separate `chatterbox` executable target. Supertonic and Parler
GGUFs are autodetected from their architecture metadata.

| Stage         | Turbo                                      | Multilingual                                        |
|---------------|--------------------------------------------|-----------------------------------------------------|
| Tokenizer     | GPT-2 byte-level BPE (English)             | HuggingFace `tokenizers.json` (23 langs, NFKD pre)  |
| T3 backbone   | GPT-2 Medium, 24 layers, single forward    | Llama-520M, 30 layers, CFG cond+uncond per token    |
| CFM solver    | Meanflow, 2 Euler steps                    | Standard, 10 Euler steps with `cfg_rate=0.7`        |
| HiFT vocoder  | shared (same checkpoint format)            | shared (same checkpoint format)                     |

The other synthesis pipelines are:

| Family | Pipeline |
|---|---|
| Supertonic | text preprocessing → duration → text encoder → vector estimator → vocoder |
| Parler | Flan-T5 description encoder → delay-pattern decoder → DAC |
| CosyVoice3 | Qwen2.5 LM → DiT flow → CausalHiFT |
| Audio8 | DualAR semantic/fast LM → 10-codebook codec decoder |
| LavaSR | optional UL-UNAS denoiser → Vocos bandwidth-extension enhancer |
