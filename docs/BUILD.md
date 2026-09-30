# Building the speech stack

Part of the [qvac-fabric-speech.cpp documentation](../README.md).

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

GPU backends come from the ggml build: `-DGGML_VULKAN=ON`, `-DGGML_OPENCL=ON`, `-DGGML_CUDA=ON`; Metal is on by default on Apple. Core ML is gated per engine and defaults to off everywhere, so add `-DWHISPER_COREML=ON -DPARAKEET_COREML=ON -DTTS_CPP_COREML=ON -DAUDIOGEN_COREML=ON` on Apple for the Whisper encoder, Parakeet encoder (CTC/IndicConformer, Unified RNN-T, TDT, EOU, Nemotron, Sortformer v2.1), Supertonic vocoder, Audio8 codec synthesis, and ACE-Step VAE decoder sidecars. EOU and Nemotron sidecars accelerate exact compiled mel shapes only; the per-model matrix is in the [root README](../README.md#apple-core-ml-sidecars). For tests, configure with `-DSPEECH_BUILD_TESTS=ON`, then run the non-GPU suite with `ctest --test-dir build -LE 'gpu|perf'`. A Metal build also exposes `test-minimax-metal-ops`, the model-free AudioGen CPU/Metal parity regression; it skips unless the Metal device supports `MUL_MAT` (simdgroup reduction, `MTLGPUFamilyApple7`+), which rules out the virtualized GPUs on hosted macOS runners.

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
