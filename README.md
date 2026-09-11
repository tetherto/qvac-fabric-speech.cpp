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
| Backends | CPU, Metal, Vulkan, OpenCL (Adreno), CUDA, Apple Core ML (encoder sidecar) |
| Quantization | `f32`, `f16`, `bf16`, `q8_0`, `q6_k`, `q5_0`, `q5_1`, `q4_0`, `q4_k_m` (per model, see tables) |
| Shared ggml | one `ggml-speech` vcpkg port, built from [qvac-ext-ggml@speech](https://github.com/tetherto/qvac-ext-ggml/tree/speech) |
| Language | C++17 |

## Architecture

```
+-----------------------------+  +-----------------------------+
| third_party/whisper.cpp     |  | engines/parakeet            |
| speech-to-text              |  | ASR + diarization + EOU     |
+-----------------------------+  +-----------------------------+
+-----------------------------+  +-----------------------------+
| engines/tts                 |  | engines/audiogen            |
| TTS + cloning + enhancement |  | text-to-music               |
+-----------------------------+  +-----------------------------+
                    |                     :
                    v                     : optional encoder sidecar
   ggml-speech (qvac-ext-ggml@speech)     v
                    |                Apple Core ML
   +--------+-------+-------+---------+
   v        v       v       v         v
  CPU     Metal  Vulkan  OpenCL     CUDA
                        (Adreno)
```

Every component consumes one system ggml, so the whole stack shares a single ggml pin and file set. The `ggml/` tree vendored inside the whisper subtree is never compiled.

## Pipelines

```
whisper   wav  -> log-mel -> encoder -> decoder -> text            (+ Silero VAD, + Core ML encoder)
parakeet  wav  -> log-mel -> FastConformer encoder -> CTC | RNN-T | TDT | EOU | Nemotron | Sortformer
                                                   -> text | speaker segments | turn boundary
tts       text -> LM (T3 / Llama / Qwen2.5) -> acoustic tokens -> CFM or flow -> vocoder -> wav
                                                   (+ LavaSR denoise -> bandwidth extension)
audiogen  caption + lyrics -> ACE-Step LM -> FSQ detokenizer -> text encoder
                           -> condition encoder -> DiT flow matching
                           -> Oobleck VAE -> 48 kHz stereo
          short query -> LM inspire (Simple Mode) -> caption + lyrics + metadata
          caption + lyrics -> LM format (Query Rewriting) -> detailed request
                           -> same ACE-Step pipeline
          lyrics + generated audio -> DiT cross-attention probe -> DTW
                           -> synchronized LRC timestamps
          generated codes + request -> teacher-forced LM -> quality score
          audio -> VAE encode -> FSQ tokenize -> LM listener
                           -> metadata + caption + recovered codes
          caption + lyrics -> MiniMax Qwen3 LM -> RVQ depth decoder
                           -> condition encoder -> flow DiT -> vocoder -> stereo
```

## Repo layout

```
CMakeLists.txt              feature-gated umbrella superbuild
third_party/whisper.cpp/    upstream whisper.cpp, vendored as a git subtree,
                            pinned @ v1.9.1 (f049fff9); every QVAC delta is
                            declared in PATCHES.md and enforced by CI
engines/
  parakeet/                 ASR + diarization + end-of-utterance (NVIDIA Parakeet family)
  tts/                      text-to-speech, voice cloning, speech enhancement
  audiogen/                 music generation (ACE-Step, MiniMax-Music3)
docs/UPSTREAM-SYNC.md       how to sync the whisper subtree
```

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
| `nvidia/parakeet-tdt-0.6b-v3` | parakeet | ~25 + punctuation and capitalization | 600 M | `f32`, `f16`, `q8_0`, `q5_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML offline encoder | graph decoder on Metal/Vulkan/CUDA; scalar on CPU/OpenCL |
| `nvidia/parakeet-tdt-1.1b` | parakeet | English | 1.1 B | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA; Core ML offline encoder | no punctuation; graph decoder on Metal/Vulkan/CUDA |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | parakeet | locale-conditioned multilingual | 600 M | `f16` | CPU, Metal, Vulkan, OpenCL, CUDA | cache-aware streaming at 80/160/320/560/1120 ms; empty language selects `auto` |

### End-of-utterance and diarization

| Model | Engine | Task | Params | Quantization | Backends | Notes |
|---|---|---|---|---|---|---|
| `nvidia/parakeet_realtime_eou_120m-v1` | parakeet | low-latency ASR + end-of-turn | 120 M | `f16`, `q8_0` | CPU, Metal, Vulkan, OpenCL, CUDA | decoder graphs on Metal/Vulkan/CUDA, scalar on CPU/OpenCL; `is_eou_boundary` |
| `nvidia/diar_sortformer_4spk-v1` | parakeet | diarization, up to 4 speakers | 123 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | offline + sliding-history live |
| `nvidia/diar_streaming_sortformer_4spk-v2` | parakeet | diarization, up to 4 speakers | 117 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | streaming-trained encoder |
| `nvidia/diar_streaming_sortformer_4spk-v2.1` | parakeet | diarization, up to 4 speakers | 117 M | `f16`, `q8_0`, `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | Audio-Online Speaker Cache, stable slots across gaps |

Parakeet's CUDA path was validated on an RTX 3080 (TDT q8_0 and q4_0
transcripts, Sortformer and streaming output byte-equal to the previous build,
LibriSpeech WER within noise of the CPU reference) but is not yet covered by
hardware decoder parity CI. CUDA in these rows denotes hardware-validated
availability, not CI coverage.

Pair any CTC, RNN-T, TDT, or EOU GGUF with a Sortformer GGUF via `--diarization-model` for an attributed "who said what" transcript. See the [Parakeet backend, Core ML, streaming, conversion, and package guide](engines/parakeet/README.md).

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
| Fun-CosyVoice3-0.5B | tts | model-advertised multilingual text | 24 kHz | `f32` | CPU, Metal, Vulkan, OpenCL, CUDA | Qwen2.5 LM + DiT flow + CausalHiFT; zero-shot/cross-lingual cloning from a reference WAV (native speech_tokenizer_v3 + CAM++); Metal, desktop Vulkan, desktop CUDA, and OpenCL are the validated GPU paths |
| Audio8-TTS-Preview-0.6B | tts | multilingual | 44.1 kHz | `f32`, `f16`, `q8_0`; LM also `q4_0` | CPU, Metal, Vulkan, OpenCL, CUDA | DualAR + DAC codec, zero-shot cloning from reference audio and transcript |

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

## Build

Prerequisites: CMake >= 3.20, a C++17 compiler, git.

```sh
# 1) system ggml (the branch the ggml-speech vcpkg port is cut from; the port
#    pins one commit, so check its portfile REF to match a port build exactly)
git clone --depth 1 --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/ggml-install
cmake --build ggml-src/build -j && cmake --install ggml-src/build

# 2) the speech stack (whisper + parakeet + tts + audiogen, one shared ggml)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/ggml-install
cmake --build build -j
```

Stage 1 installs a **shared** ggml. Stage 2 follows `BUILD_SHARED_LIBS` (whisper
defaults it `ON` for the umbrella). `audiogen-cpp` can be shared; its CLIs and
tests link an object library so they still see hidden internals. Consumers keep
`audiogen-cpp::audiogen-cpp`.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `SPEECH_BUILD_WHISPER` | `ON` | build `third_party/whisper.cpp` |
| `SPEECH_BUILD_PARAKEET` | `ON` | build `engines/parakeet` |
| `SPEECH_BUILD_TTS` | `ON` | build `engines/tts` |
| `SPEECH_BUILD_AUDIOGEN` | `ON` | build `engines/audiogen` |
| `AUDIOGEN_BUILD_MINIMAX` | desktop `ON`, mobile `OFF` | build the desktop MiniMax-Music3 engine (CPU by default, GPU via `EngineOptions::device`) |
| `SPEECH_BUILD_EXECUTABLES` | `ON` | build the CLIs; set `OFF` for library-only builds |
| `SPEECH_BUILD_TESTS` | `OFF` | build the engine test harnesses |
| `SPEECH_BUILD_WHISPER_TESTS` | `OFF` | also build whisper's tests (committed weightless stubs cover tiny..large pipeline smokes; only `test-vad-full` needs a downloaded model) |

GPU backends come from the ggml build: `-DGGML_VULKAN=ON`, `-DGGML_OPENCL=ON`, `-DGGML_CUDA=ON`; Metal is on by default on Apple. Core ML is gated per engine and defaults to off on both, so add `-DWHISPER_COREML=ON -DPARAKEET_COREML=ON` on Apple for the Whisper encoder and Parakeet offline TDT encoder sidecars. For tests, configure with `-DSPEECH_BUILD_TESTS=ON`, then run the non-GPU suite with `ctest --test-dir build -LE 'gpu|perf'`. A Metal build also exposes `test-minimax-metal-ops`, the model-free AudioGen CPU/Metal parity regression; it skips unless the Metal device supports `MUL_MAT` (simdgroup reduction, `MTLGPUFamilyApple7`+), which rules out the virtualized GPUs on hosted macOS runners.

Each engine also configures standalone (`cmake -S engines/parakeet`, and so on), which is what the CI lanes use.

### Consumable packages

One vcpkg port, [`speech-cpp`](https://github.com/tetherto/qvac-registry-vcpkg/tree/main/ports/speech-cpp), builds this repo through the umbrella `CMakeLists.txt` above: engine features select what gets built, and every enabled engine links the single `ggml-speech` ggml. Consumers depend on the engines they need, for example `speech-cpp[whisper,parakeet,vulkan]`, and the backend features (`metal`, `vulkan`, `opencl`) fan out to the matching `ggml-speech` features so the whole stack resolves one ggml.

| Feature | `find_package` | Imported target |
|---|---|---|
| (always) | `ggml` | `ggml::ggml` |
| `speech-cpp[whisper]` | `whisper` | `whisper::whisper` |
| `speech-cpp[parakeet]` | `qvac-parakeet` | `qvac::parakeet` |
| `speech-cpp[tts]` | `tts-cpp` | `tts-cpp::tts-cpp` |
| `speech-cpp[audiogen]` | `audiogen-cpp` | `audiogen-cpp::audiogen-cpp` |

The per-engine `whisper-cpp`, `parakeet-cpp`, `tts-cpp` and `audiogen-cpp` ports that predate `speech-cpp` are superseded: they pinned this repo at four different commits, and `speech-cpp` replaces them with one pin for the whole stack.

## Command line tools

| Binary | Engine | Purpose |
|---|---|---|
| `whisper-cli` | whisper | transcribe and translate, with optional Silero VAD |
| `parakeet` | parakeet | transcribe, diarize, detect end-of-utterance, benchmark |
| `tts-cli` | tts | Chatterbox, Supertonic, and Parler synthesis, autodetected from GGUF metadata |
| `parler-cli` | tts | full Parler-TTS flag surface |
| `supertonic-cli` | tts | standalone Supertonic synthesis |
| `cosyvoice-cli` | tts | CosyVoice3 synthesis |
| `audio8-cli` | tts | Audio8 synthesis and zero-shot voice cloning |
| `music-cli` | audiogen | end-to-end text-to-music |
| `acestep-cli` | audiogen | Oobleck VAE decode and roundtrip harness |
| `acestep-quantize` | audiogen | requantize converted ACE-Step or MiniMax-Music3 stage GGUFs |
| `mm3-replay` | audiogen | MiniMax-Music3 generation and parity harness |
| `lavasr-bench` | tts | denoiser and enhancer benchmark |
| `mel2wav` | tts | HiFT mel to wav |

### Whisper

```sh
./third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
                        -f third_party/whisper.cpp/samples/jfk.wav
```

### Parakeet

Models are converted from NeMo checkpoints with `download-all-models.sh` and
`convert-nemo-to-gguf.py`. The downloader covers every supported checkpoint,
including the AI4Bharat IndicConformer hybrid; see
[engines/parakeet/README.md](engines/parakeet/README.md).

Hybrid RNNT+CTC checkpoints export CTC by default. Pass `--head rnnt` to export
their Transducer branch; conversion validates that the joint output is exactly
vocabulary plus blank. `dump-rnnt-reference.py` selects the same NeMo branch
for token-level parity testing.

```sh
# transcribe (the GGUF metadata selects CTC / RNN-T / TDT / EOU / Nemotron)
./build/engines/parakeet/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav

# transcribe with speaker attribution
./build/engines/parakeet/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
                                  --diarization-model models/diar_sortformer_4spk-v1.f16.gguf \
                                  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# streaming end-of-utterance, JSONL events
./build/engines/parakeet/parakeet --model models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav \
                                  --stream --stream-chunk-ms 1500 --emit jsonl
```

### Text-to-speech

GGUF conversion steps and the umbrella/direct/vcpkg build-path matrix are in
[engines/tts/README.md](engines/tts/README.md). The umbrella build enables this
package with `SPEECH_BUILD_TTS=ON`.

```sh
# Chatterbox Turbo, with voice cloning from a reference wav
./build/engines/tts/tts-cli --model      models/chatterbox-t3-turbo.gguf \
                            --s3gen-gguf models/chatterbox-s3gen.gguf \
                            --reference-audio me.wav \
                            --text "Hello from native C plus plus." --out out.wav

# Chatterbox Multilingual
./build/engines/tts/tts-cli --model      models/chatterbox-t3-mtl-q4_0.gguf \
                            --s3gen-gguf models/chatterbox-s3gen-mtl-q4_0.gguf \
                            --text "Hola, esto es una demostracion multilingue." \
                            --language es --cfm-steps 7 --out out.wav

# Supertonic, preset voice
./build/engines/tts/tts-cli --model models/supertonic2.gguf --voice M1 --language en \
                            --text "The quick brown fox jumps over the lazy dog." --out out.wav

# Parler-TTS, description-conditioned
./build/engines/tts/parler-cli --model models/parler-mini-v1-q8_0.gguf \
                               --description "A female speaker with a calm, clear voice, close up." \
                               --text "Hey, how are you doing today?" --out out.wav

# CosyVoice3
./build/engines/tts/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
                                  --text "Hello from a fully on-device pipeline." --out out.wav

# Audio8; drop --n-gpu-layers to stay on the CPU
./build/engines/tts/audio8-cli --lm models/audio8-lm-q8_0.gguf \
                               --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
                               --text "Hello from Audio8." \
                               --n-gpu-layers 99 --out out.wav
```

`--emotion` and `--pace` work the same way on every engine that supports them;
each CLI lists its own supported values via `--list-emotions` / `--list-paces`.
See [Voice conditioning](engines/tts/README.md#voice-conditioning-cross-engine).

```sh
./build/engines/tts/parler-cli --model models/parler-indic-q8_0.gguf \
                               --emotion happy --pace moderate \
                               --text "आज मौसम बहुत अच्छा है।" --out out.wav

./build/engines/tts/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
                                  --emotion happy \
                                  --text "Hello from a fully on-device pipeline." --out out.wav
```

### Music generation

AudioGen uses four GGUF files for six runtime weight sets. The DiT file also
contains the FSQ detokenizer and condition encoder; see the
[AudioGen model setup](engines/audiogen/README.md#model-setup) for the
validated file combinations and the download, conversion, and quantization
steps that produce them.

```sh
./build/engines/audiogen/music-cli --models models/acestep \
                                   --caption "driving synth pop, bright analog leads, 120 bpm" \
                                   --lyrics "[Instrumental]" --dur 8 --gpu --out song.wav
```

## Performance

`RTF = inference_time / audio_duration`, lower is better. The parakeet and tts
READMEs carry their full tables, methodology, and reproduction steps. AudioGen
has a reproducible
[engine comparison harness](engines/audiogen/benchmarks/comparison/README.md)
for CPU, Metal, Vulkan, and CUDA; `music-cli` also reports per-stage wall clock
on stderr.

### ASR, end-of-utterance, diarization

CI numbers from the published `@qvac/asr-ggml@0.1.1` addon ([run 31603189415](https://github.com/tetherto/qvac/actions/runs/31603189415), 2026-08-12), `q8_0` GGUFs, 1 warmup plus 5 timed runs, host `qvac-ubuntu2204-x64-gpu` (CPU: Intel Core i5-13500, GPU: NVIDIA RTX 4000 SFF Ada, Vulkan). Full table: [engines/parakeet/README.md](engines/parakeet/README.md#performance).

| Model | CPU RTF | CPU wall | Vulkan RTF | Vulkan wall |
|---|--:|--:|--:|--:|
| Parakeet CTC | 0.112 | 2256 ms | 0.0022 | 43 ms |
| Parakeet TDT | 0.130 | 2607 ms | 0.0044 | 88 ms |
| Parakeet EOU | 0.051 | 1034 ms | 0.0034 | 68 ms |
| Sortformer | 0.046 | 922 ms | 0.0019 | 38 ms |
| Sortformer streaming | 0.032 | 646 ms | 0.0034 | 69 ms |
| Whisper base | 0.035 | 699 ms | 0.0057 | 117 ms |
| Whisper small | 0.122 | 2453 ms | 0.0098 | 200 ms |

#### speech-cpp CI (2026-09-07, CPU + macOS)

CPU only on Linux; macOS whisper rows run on Metal (`MTL0`).

| Model | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Parakeet CTC 0.6b q8_0 | linux | ggml-cpu | 2111 | 0.192 | 858 |
| Parakeet CTC 0.6b q8_0 | macos | ggml-cpu | 184 | 0.0170 | 914 |
| Whisper tiny | linux | (CPU) | 1035 | 0.0940 | 182 |
| Whisper tiny | macos | Metal | 565 | 0.0510 | 240 |
| Whisper base | linux | (CPU) | 1906 | 0.173 | 298 |
| Whisper base | macos | Metal | 584 | 0.0530 | 360 |
| Whisper small | linux | (CPU) | 6122 | 0.557 | 797 |
| Whisper small | macos | Metal | 564 | 0.0510 | 878 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Text-to-speech

CI numbers from the published `@qvac/tts-ggml@0.6.2` addon ([run 31603192731](https://github.com/tetherto/qvac/actions/runs/31603192731), 2026-08-12), `q4_0` GGUFs, same host. Full table: [engines/tts/README.md](engines/tts/README.md#performance).

| Model | CPU RTF | Vulkan RTF | Vulkan wall | Vulkan tok/s |
|---|--:|--:|--:|--:|
| Chatterbox Turbo | 1.54 | 0.099 | 410 ms | 173 |
| Chatterbox Multilingual | 5.81 | 0.182 | 1036 ms | 77 |
| Supertonic | 0.113 | 0.018 | 78 ms | 952 |
| Supertonic Multilingual | 0.101 | 0.013 | 84 ms | 1087 |
| Supertonic 3 | 0.225 | 0.029 | 118 ms | 631 |

#### speech-cpp CI (2026-09-07, CPU + macOS)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Chatterbox: chatterbox-t3-turbo-q8_0 | linux | CPU | 8078 | — | 1835 |
| Chatterbox: chatterbox-t3-turbo-q8_0 | macos | CPU | 2626 | — | 1746 |
| Supertonic: supertonic3-q8_0 | linux | (CPU) | 708 | 0.484 | 876 |
| Supertonic: supertonic3-q8_0 | macos | (CPU) | 49 | 0.0340 | 584 |
| Parler: parler-mini-v1-q8_0 | linux | CPU | 249362 | 18.8 | 1964 |
| Parler: parler-mini-v1-q8_0 | macos | MTL0 (Metal) | 12204 | 0.582 | 1307 |
| Cosyvoice: cosyvoice3-llm-q8_0 | linux | CPU | 67695 | 18.0 | 1416 |
| Cosyvoice: cosyvoice3-llm-q8_0 | macos | CPU | 46825 | 12.3 | 1467 |

`—` RTF for chatterbox because it is text-driven variable output.

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Music generation & other engines (speech-cpp CI, 2026-09-07)

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | linux | (CPU) | 36817 | 9.20 | 1702 |
| Acestep: Qwen3-Embedding-0.6B-Q8_0 | macos | (CPU) | 6698 | 1.67 | 2222 |
| Lavasr: lavasr-denoiser-f16 | linux | (CPU) | 4102 | 0.684 | 155 |
| Lavasr: lavasr-denoiser-f16 | macos | (CPU) | 2090 | 0.348 | 209 |

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Brain-computer interface

CI numbers from the published `@qvac/bci-whispercpp@0.6.0` addon ([run 31602627344](https://github.com/tetherto/qvac/actions/runs/31602627344), 2026-08-12), `ggml-bci-windowed` model. Throughput in tokens/s, higher is better.

| Host | CPU tok/s | Vulkan tok/s | Vulkan wall |
|---|--:|--:|--:|
| Linux x86-64 (i5-13500 / RTX 4000 SFF Ada) | 27.0 | 355.6 | 42 ms |
| Windows x64 (`qvac-win25-x64-gpu`) | 20.1 | 36.0 | 349 ms |
| Linux arm64 (`ubuntu-24.04-arm`, CPU-only lane) | 16.8 | n/a | n/a |

The macOS arm64 lane runs on the GitHub-hosted `macos-26` runner, whose virtualised Metal device is not representative (6.6 tok/s vs 398 tok/s on the previously used self-hosted M-series box), so it is omitted here.

### Apple silicon

| Model | Host | Backend | Quantization | RTF | vs real-time |
|---|---|---|---|--:|--:|
| Parakeet TDT 0.6b v3 | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.015 | 67x |
| Parakeet CTC | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.011 | 88x |
| Whisper small | Mac mini M4 (CI, `mac-mini-m4-gpu`) | Metal | `q8_0` | 0.027 | 37x |
| Chatterbox Turbo | Mac Studio M3 Ultra | Metal | `q4_0` | 0.16 | 6.4x |
| Chatterbox Turbo | Mac Studio M3 Ultra | CPU (NEON) | `q4_0` | 1.05 | 0.96x |
| Chatterbox Multilingual (`--cfm-steps 7`) | Mac Studio M3 Ultra | Metal | `q4_0` | 0.30 | 3.3x |
| Chatterbox Multilingual | Apple M4 | Metal | `q4_0` | 1.37 | 0.73x |

### Streaming latency

Chatterbox on Apple M4 Metal, 317 speech tokens (12.7 s of audio), `--stream-first-chunk-tokens 10 --stream-chunk-tokens 25 --stream-cfm-steps 1`. Full table: [engines/tts/README.md](engines/tts/README.md#streaming-mode--low-latency-playback).

| Metric | Value |
|---|--:|
| first audio out | 279 ms |
| steady-state chunk RTF | 0.30 to 0.63 |
| overall RTF | 0.90 |

On-device Android and iOS performance is tracked by the benchmark lanes in [QVAC](https://github.com/tetherto/qvac).

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
| Whisper subtree sync process | [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) |
| ASR, diarization, end-of-utterance | [engines/parakeet/README.md](engines/parakeet/README.md) |
| Text-to-speech and enhancement | [engines/tts/README.md](engines/tts/README.md) |
| Music generation | [engines/audiogen/README.md](engines/audiogen/README.md) |
| TTS memory behaviour | [engines/tts/MEMORY.md](engines/tts/MEMORY.md) |
| Development journals | [engines/parakeet/PROGRESS.md](engines/parakeet/PROGRESS.md), [engines/tts/PROGRESS.md](engines/tts/PROGRESS.md) |
