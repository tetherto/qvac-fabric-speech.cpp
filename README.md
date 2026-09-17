# qvac-fabric-speech.cpp

On-device speech and audio AI in pure C++ on [ggml](https://github.com/tetherto/qvac-ext-ggml): speech-to-text, speaker diarization, end-of-utterance detection, text-to-speech, voice cloning, speech enhancement, and music generation.

| Property | Value |
|---|---|
| CMake project | `qvac-speech` (feature-gated superbuild over `third_party/` + `engines/`) |
| Runtime dependencies | ggml only. No Python, PyTorch, or ONNX Runtime at inference time |
| Engines | `third_party/whisper.cpp`, `engines/parakeet`, `engines/tts`, `engines/audiogen` |
| Models | every model loads from GGUF (see [Supported models](#supported-models)) |
| Desktop | Linux, macOS, Windows |
| Mobile | Android (arm64-v8a), iOS (arm64) |
| Backends | CPU, Metal, Vulkan, OpenCL (Adreno), CUDA, Apple Core ML (encoder, codec, and VAE sidecars) |
| Quantization | `f32`, `f16`, `bf16`, `q8_0`, `q6_k`, `q5_0`, `q5_1`, `q4_0`, `q4_k_m` (per model, see tables) |
| Shared ggml | one `ggml-speech` vcpkg port, built from [qvac-ext-ggml@speech](https://github.com/tetherto/qvac-ext-ggml/tree/speech) |
| Language | C++17 |

## Supported models

One row per model. `Backends` lists available engine paths; row notes and the
engine-specific guides qualify model-level validation.

### Speech-to-text and translation

| Model | Engine | Languages | Params | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| `whisper-tiny` / `tiny.en` | whisper | 99 + translation | 39 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-base` / `base.en` | whisper | 99 + translation | 74 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-small` / `small.en` | whisper | 99 + translation | 244 M | `f16`, `q5_1`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-small.en-tdrz` | whisper | English | 244 M | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | tinydiarize speaker turns |
| `whisper-medium` / `medium.en` | whisper | 99 + translation | 769 M | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v1` | whisper | 99 + translation | 1.55 B | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v2` | whisper | 99 + translation | 1.55 B | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v3` | whisper | 99 + translation | 1.55 B | `f16`, `q5_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | |
| `whisper-large-v3-turbo` | whisper | 99 + translation | 809 M | `f16`, `q5_0`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA, Core ML | fastest large-class decode |
| `silero-v5.1.2` | whisper | language agnostic | 2 M | `f16` | CPU, Metal, Vulkan, CUDA | voice activity detection; GPU is opt-in via `use_gpu`, default CPU |
| `silero-v6.2.0` | whisper | language agnostic | 2 M | `f16` | CPU, Metal, Vulkan, CUDA | voice activity detection; GPU is opt-in via `use_gpu`, default CPU |
| `nvidia/parakeet-ctc-0.6b` | parakeet | English | 600 M | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | offline + streaming + long-form |
| `nvidia/parakeet-ctc-1.1b` | parakeet | English | 1.1 B | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | offline + streaming + long-form |
| `ai4bharat/indic-conformer-600m-multilingual` | parakeet | 22 Indic | 600 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan | CTC export by default; `--head rnnt` exports the Transducer branch; CTC requires `--language` / `EngineOptions::language` |
| `nvidia/parakeet-unified-en-0.6b` | parakeet | English | 600 M | `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML offline encoder | RNN-T decode; offline + buffered streaming + long-form |
| `nvidia/parakeet-tdt-0.6b-v3` | parakeet | ~25 + punctuation and capitalization | 600 M | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML offline encoder | graph decoder on Metal/Vulkan/CUDA; scalar on CPU/OpenCL |
| `nvidia/parakeet-tdt-1.1b` | parakeet | English | 1.1 B | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML offline encoder | no punctuation; graph decoder on Metal/Vulkan/CUDA |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | parakeet | locale-conditioned multilingual | 600 M | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | cache-aware streaming at 80/160/320/560/1120 ms; empty language selects `auto` |

### End-of-utterance and diarization

| Model | Engine | Task | Params | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| `nvidia/parakeet_realtime_eou_120m-v1` | parakeet | low-latency ASR + end-of-turn | 120 M | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML exact-shape encoder | decoder graphs on Metal/Vulkan/CUDA, scalar on CPU/OpenCL; `is_eou_boundary` |
| `nvidia/diar_sortformer_4spk-v1` | parakeet | diarization, up to 4 speakers | 123 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | offline + sliding-history live |
| `nvidia/diar_streaming_sortformer_4spk-v2` | parakeet | diarization, up to 4 speakers | 117 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | streaming-trained encoder |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | parakeet | diarization, up to 4 speakers | 117 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML exact-shape batch/AOSC encoder | Audio-Online Speaker Cache, stable slots across gaps; speaker head stays on ggml |

Parakeet's CUDA path was validated on an RTX 3080 (TDT q8_0 and q4_0
transcripts, Sortformer and streaming output byte-equal to the previous build,
LibriSpeech WER within noise of the CPU reference) but is not yet covered by
hardware decoder parity CI. CUDA in these rows denotes hardware-validated
availability, not CI coverage.

Pair any CTC, RNN-T, TDT, or EOU GGUF with a Sortformer GGUF via `--diarization-model` for an attributed "who said what" transcript. See the [Parakeet backend, Core ML, streaming, conversion, and package guide](engines/parakeet/README.md).

Sortformer v2.1 can offload either an exact-shape batch encoder or the masked
AOSC block stack to Apple Core ML; the speaker head remains on ggml. Export the
batch sidecar with
`python engines/parakeet/scripts/export-encoder-coreml.py --gguf <v2.1.f16.gguf> --wav <fixed-shape.wav> --out <v2.1-encoder.mlpackage> --compile-dir <model-dir>`.
For AOSC, replace `--wav` with
`--bypass-pre-encode --n-encoder-frames 410` and use an
`-encoder-bypass-pre-encode.mlpackage` output. The compiled directories must
end in `-encoder.mlmodelc` and `-encoder-bypass-pre-encode.mlmodelc`,
respectively. Batch inputs must match the exported mel-frame count; AOSC inputs
may be shorter than its masked capacity. Missing sidecars, incompatible shapes,
and prediction failures fall back to ggml.

### Text-to-speech and voice cloning

| Model | Engine | Languages | Sample rate | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| Chatterbox Turbo | tts | English | 24 kHz | `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | zero-shot voice cloning, 2-step meanflow CFM, streaming |
| Chatterbox Multilingual | tts | 23 | 24 kHz | `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | zero-shot voice cloning, CFG, `--cfm-steps` knob, streaming |
| Supertonic v1 | tts | English | 44.1 kHz | `f32`, `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | preset voices, streaming |
| Supertonic v2 | tts | 5 (`en`, `ko`, `es`, `pt`, `fr`) | 44.1 kHz | `f32`, `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | preset voices, streaming |
| Supertonic v3 | tts | 31 + `na` | 44.1 kHz | `f32`, `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | preset voices, streaming, `na` for unknown source language |
| Parler-TTS mini-v1 | tts | English | 44.1 kHz | `f32`, `f16`, `q8_0`, `q6_k` | CPU, Metal, Vulkan, OpenCL, CUDA | description-conditioned voice, no cloning |
| Parler-TTS large-v1 | tts | English | 44.1 kHz | `f32`, `f16`, `q8_0`, `q6_k` | CPU, Metal, Vulkan, OpenCL, CUDA | description-conditioned voice |
| Indic Parler-TTS | tts | 21 Indic | 44.1 kHz | `f32`, `f16`, `q8_0`, `q6_k` | CPU, Metal, Vulkan, OpenCL, CUDA | Indic prompt BPE tokenizer |
| Fun-CosyVoice3-0.5B | tts | model-advertised multilingual text | 24 kHz | `f32`; LM and flow also `q8_0`, `q4_0`; flow and HiFT also `f16`; flow also `bf16` | CPU, Metal, Vulkan, OpenCL, CUDA | Qwen2.5 LM + DiT flow + CausalHiFT; zero-shot/cross-lingual cloning from a reference WAV (native speech_tokenizer_v3 + CAM++); Metal, desktop Vulkan, desktop CUDA, and OpenCL are the validated GPU paths |
| Audio8-TTS-Preview-0.6B | tts | multilingual | 44.1 kHz | `f32`, `f16`, `q8_0`; LM also `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA; optional Core ML codec-synthesis sidecar (`TTS_CPP_COREML`, Apple) | DualAR + DAC codec, zero-shot cloning from reference audio and transcript |
| Pocket TTS | tts | English | 24 kHz | `f32`; `f16` as storage | CPU | FlowLM + Mimi, prepared voice, streaming; cloning requires encoder-enabled weights |

When a TTS build carries both CUDA and Vulkan, backend selection prefers CUDA
on NVIDIA hardware; `TTS_CPP_GPU_BACKEND=cuda|vulkan|metal|opencl` pins one
backend for a test arm or comparison and rejects a value that selects no usable
device. The per-model validation each backend column rests on is documented in
the [TTS capability table](engines/tts/README.md#capabilities).

### Speech enhancement

| Model | Engine | Task | Rate | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| LavaSR denoiser (UL-UNAS) | tts | speech denoising | rate preserving, 16 kHz internal STFT | `f32`, `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | applied after synthesis or on captured audio |
| LavaSR enhancer (Vocos BWE) | tts | bandwidth extension | native in, 48 kHz out | `f32`, `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | ConvNeXt + ISTFT head |

### Music generation

| Model | Engine | Task | Rate | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| ACE-Step v15 turbo | audiogen | text-to-music | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0`, `q4_k_m` | CPU, Vulkan, Metal, OpenCL (Adreno 700+), CUDA; optional Core ML VAE-decoder sidecar (`AUDIOGEN_COREML`, Apple) | 8 diffusion steps by default |
| ACE-Step v15 sft | audiogen | text-to-music | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0` | CPU, Vulkan, Metal, OpenCL (Adreno 700+), CUDA; optional Core ML VAE-decoder sidecar (`AUDIOGEN_COREML`, Apple) | 50 diffusion steps by default |
| ACE-Step v15 base | audiogen | text-to-music, multi-track (lego) stems | 48 kHz stereo | `f32`, `f16`, `bf16`, `q8_0` | CPU, Vulkan, Metal, OpenCL (Adreno 700+), CUDA; optional Core ML VAE-decoder sidecar (`AUDIOGEN_COREML`, Apple) | 50 diffusion steps by default, `--task lego --track <layer>` |
| MiniMax-Music3 | audiogen | text-to-music | 44.1 kHz stereo | `f16`, `q8_0`; LM+DiT also `q4_k_m` | desktop CPU + GPU (CUDA, Vulkan, Metal via `EngineOptions::device`) | 25 fps, 30 flow steps, two GGUF files; `test-minimax-metal-ops` checks Metal condition/vocoder parity on an Apple7+ GPU |

The Audio8 Core ML sidecar takes the codec's synthesis stack (the stage that
dominates a CPU synthesis) off ggml; it is exported from the decoder GGUF by
`engines/tts/scripts/export-audio8-codec-coreml.py`, hosts read
`Engine::codec_on_coreml()` (sidecar loaded) and
`SynthesisResult::codec_synthesis_backend` (where a call's codec synthesis ran;
the language model always stays on `backend_name()`), and any sidecar failure
falls back to ggml -- see the
[Audio8 guide](engines/tts/docs/audio8.md#core-ml-codec-sidecar). The
ACE-Step Core ML sidecar is exported by
`engines/audiogen/scripts/export-vae-coreml.py` at its 64-latent-frame Neural
Engine operating point, optionally weight-palettized (`--palettize 8` halves
the sidecar at unchanged speed and quality gate); see the
[audiogen backends guide](engines/audiogen/docs/backends.md#core-ml-vae-decoder-sidecar) for
the measured constraints and benchmark tooling.

## Performance

`RTF = inference_time / audio_duration`, lower is better. Each engine
README carries its models-to-speed tables; the per-modality CI tables,
Apple-silicon numbers, and streaming latency live in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md). AudioGen additionally has a
reproducible
[engine comparison harness](engines/audiogen/benchmarks/comparison/README.md)
for CPU, Metal, Vulkan, and CUDA.

The [desktop benchmark workflow](.github/workflows/speech-benchmark-desktop.yml)
accepts `music_alignment=true` (default `false`) to score caption adherence with
CLAP for AudioGen's ACE-Step (`acestep`) and MiniMax-Music3 (`minimax`) families.
Its per-family artifacts retain generated WAVs, generation/scorer logs,
binary/model hashes, scorer provenance, scores, and scored/expected coverage.
These are non-gating diagnostics with no quality pass threshold; unavailable
scores remain null without discarding generation performance. See the
[music alignment guide](scripts/benchmarks/music-alignment.md) for setup,
artifact layout, and the distinct diagnostic timing baseline.

### Multi-machine benchmarks (2026-09)

Maintainer-run measurements across up to four machines — a MacBook Air M5 and
Mac mini M4 (Metal, CPU), an RTX 3080 desktop (CUDA, Vulkan), a Strix Halo box
(Vulkan, CPU), and an RTX 5090 box (CUDA, Vulkan, CPU) — every lane timed from
outside the process. The engine READMEs carry the full tables, method, and
build pins.

| Task | Model | Highlights |
|---|---|---|
| ASR | Parakeet TDT 0.6b v3 | RTF 0.0006–0.0055 on the GPU lanes; 0.00 % / 0.80 % WER on the jfk / ls90 clips — [full table](engines/parakeet/README.md#multi-machine-benchmark-2026-09) |
| TTS | Supertonic 3 | end-to-end wall 0.61–0.82 s on every GPU lane (RTF 0.024–0.031) — [full table](engines/tts/README.md#supertonic-3-multi-machine-benchmark-2026-09) |
| TTS | Audio8 0.6b | GPU RTF 0.12–0.65, faster than real time on every GPU lane — [full table](engines/tts/README.md#audio8-multi-machine-benchmark-2026-09) |
| Music | ACE-Step 1.5 | generation 1,338–2,330 ms on the GPU lanes (RTF 0.14–0.25) — [full table](engines/audiogen/README.md#ace-step-15-multi-machine-benchmark-2026-09) |

### Brain-computer interface

CI numbers from the published `@qvac/bci-whispercpp@0.6.0` addon ([run 31602627344](https://github.com/tetherto/qvac/actions/runs/31602627344), 2026-08-12), `ggml-bci-windowed` model. Throughput in tokens/s, higher is better.

| Host | CPU tok/s | Vulkan tok/s | Vulkan wall |
|---|--:|--:|--:|
| Linux x86-64 (i5-13500 / RTX 4000 SFF Ada) | 27.0 | 355.6 | 42 ms |
| Windows x64 (`qvac-win25-x64-gpu`) | 20.1 | 36.0 | 349 ms |
| Linux arm64 (`ubuntu-24.04-arm`, CPU-only lane) | 16.8 | n/a | n/a |

The macOS arm64 lane runs on the GitHub-hosted `macos-26` runner, whose virtualised Metal device is not representative (6.6 tok/s vs 398 tok/s on the previously used self-hosted M-series box), so it is omitted here.

## Use in QVAC

These engines ship inside [QVAC](https://github.com/tetherto/qvac) as SDK addons, which consume the `speech-cpp` vcpkg port built from this repo. The CLIs here are development and validation entry points: for anything beyond them, such as the JavaScript and TypeScript APIs on the Bare runtime and desktop plus mobile app integration, see QVAC.

| QVAC addon | Wraps | `speech-cpp` features consumed |
|---|---|---|
| `@qvac/asr-ggml` | speech-to-text, diarization, end-of-utterance | `whisper`, `parakeet` |
| `@qvac/tts-ggml` | text-to-speech, voice cloning, speech enhancement | `tts` |
| `@qvac/audiogen-ggml` | music generation | `audiogen` |
| `@qvac/bci-whispercpp` | brain-computer interface transcription | `whisper` |

## Licenses

| Component | Code license | Model weights |
|---|---|---|
| `third_party/whisper.cpp` | MIT | MIT (OpenAI Whisper), Silero VAD models under their own terms |
| `engines/parakeet` | Apache-2.0 | CC-BY-4.0, except `parakeet_realtime_eou_120m-v1` under the NVIDIA Open Model License |
| `engines/tts` | MIT | Chatterbox MIT; Parler, CosyVoice3, Audio8, and LavaSR Apache-2.0; Supertonic OpenRAIL-M |
| `engines/audiogen` | MIT | ACE-Step 1.5 MIT, Qwen3-Embedding Apache-2.0, MiniMax-Music3 Community License |

Per-engine `NOTICE` files list every third-party dependency and its license.

## Documentation

| Topic | Where |
|---|---|
| Product using these engines | [QVAC](https://github.com/tetherto/qvac) |
| Speech-to-text engine | [third_party/whisper.cpp/README.md](third_party/whisper.cpp/README.md) |
| Whisper subtree deltas | [third_party/whisper.cpp/PATCHES.md](third_party/whisper.cpp/PATCHES.md) |
| Architecture, pipelines, repo layout | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Building the stack | [docs/BUILD.md](docs/BUILD.md) |
| Command-line tools and examples | [docs/CLI.md](docs/CLI.md) |
| Performance across modalities and hosts | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Whisper subtree sync process | [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) |
| ASR, diarization, end-of-utterance | [engines/parakeet/README.md](engines/parakeet/README.md) |
| Text-to-speech and enhancement | [engines/tts/README.md](engines/tts/README.md) |
| Music generation | [engines/audiogen/README.md](engines/audiogen/README.md) |
| Engine deep dives (build, backends, APIs, CLIs, models, tests) | [engines/parakeet/docs/](engines/parakeet/docs), [engines/tts/docs/](engines/tts/docs), [engines/audiogen/docs/](engines/audiogen/docs) |
| TTS memory behaviour | [engines/tts/MEMORY.md](engines/tts/MEMORY.md) |
| Development journals | [engines/parakeet/PROGRESS.md](engines/parakeet/PROGRESS.md), [engines/tts/PROGRESS.md](engines/tts/PROGRESS.md) |
