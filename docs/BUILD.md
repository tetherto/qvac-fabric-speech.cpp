# Building the speech stack

Part of the [speech-stack documentation](../README.md).

## Build

Prerequisites: CMake 3.20 or newer, a C++17 compiler, and Git. Model conversion
and optional Core ML export have separate Python dependencies in the engine
guides. Backend toolchains, such as CUDA or the Vulkan SDK, are required only
when building those ggml backends.

From the repository root, build and install a shared speech-branch ggml, then
configure the umbrella against its install prefix. These development commands
use the checked-out branch revision; record `git -C ggml-src rev-parse HEAD`
with your build. To reproduce a registry build, fetch and check out the exact
source `REF` in the [`ggml-speech` portfile](https://github.com/tetherto/qvac-registry-vcpkg/blob/main/ports/ggml-speech/portfile.cmake)
before configuring, rather than using the moving branch tip.

```sh
git clone --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
cmake -S ggml-src -B ggml-src/build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_LLAMAFILE=ON \
  -DCMAKE_INSTALL_PREFIX="$PWD/ggml-install"
cmake --build ggml-src/build --parallel
cmake --install ggml-src/build

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/ggml-install"
cmake --build build --parallel
```

Whisper, Parakeet and AudioGen follow `BUILD_SHARED_LIBS`; Whisper defaults it
to `ON` in the umbrella. TTS independently uses `TTS_CPP_BUILD_SHARED`, default
`OFF`. See [TTS linkage and test restrictions](../engines/tts/docs/build.md#useful-cmake-options)
before enabling it. All umbrella engines use the same installed ggml.

### Windows and multi-config generators

Run from the repository root in Developer PowerShell. This example uses the
Visual Studio generator and a Release configuration:

```powershell
git clone --branch speech https://github.com/tetherto/qvac-ext-ggml ggml-src
$speechGgmlInstall = Join-Path (Get-Location) "ggml-install"
cmake -S ggml-src -B ggml-src/build `
  -DBUILD_SHARED_LIBS=ON -DGGML_LLAMAFILE=ON `
  "-DCMAKE_INSTALL_PREFIX=$speechGgmlInstall"
cmake --build ggml-src/build --config Release --parallel
cmake --install ggml-src/build --config Release
cmake -S . -B build "-DCMAKE_PREFIX_PATH=$speechGgmlInstall"
cmake --build build --config Release --parallel
$env:PATH = "$speechGgmlInstall\bin;$speechGgmlInstall\lib;$env:PATH"
build\bin\Release\whisper-cli.exe --help
build\engines\parakeet\Release\parakeet.exe --help
```

Visual Studio and Xcode place executables under the selected configuration
directory. Use `--config Release` for build/install and `-C Release` for CTest.
Ninja and Make use the single-config paths shown in the CLI guides. For a
registry reproduction, select the pinned ggml source before the configure step
here as well. Dynamic backend builds also need their backend modules staged;
see the relevant engine backend guide.

### CMake options

| Option | Default | Effect |
|---|---|---|
| `SPEECH_BUILD_WHISPER` | `ON` | Build `third_party/whisper.cpp` |
| `SPEECH_BUILD_PARAKEET` | `ON` | Build `engines/parakeet` |
| `SPEECH_BUILD_TTS` | `ON` | Build `engines/tts` |
| `SPEECH_BUILD_AUDIOGEN` | `ON` | Build `engines/audiogen` |
| `SPEECH_BUILD_EXECUTABLES` | `ON` | Build development CLIs |
| `SPEECH_BUILD_TESTS` | `OFF` | Build engine test harnesses and enable root CTest |
| `SPEECH_BUILD_WHISPER_TESTS` | `OFF` | Build Whisper tests; use with `SPEECH_BUILD_TESTS=ON` |
| `AUDIOGEN_BUILD_MINIMAX` | Desktop `ON`; mobile forced `OFF` | Build MiniMax-Music3 on desktop |
| `TTS_CPP_BUILD_SHARED` | `OFF` | Select TTS linkage independently of `BUILD_SHARED_LIBS` |

GPU backends come from the ggml build: `GGML_VULKAN`, `GGML_OPENCL`,
`GGML_CUDA`, or `GGML_HEXAGON`; Metal defaults on for Apple builds.
Core ML options default off and are Apple-only: `WHISPER_COREML`,
`PARAKEET_COREML`, `TTS_CPP_COREML`, `AUDIOGEN_COREML`.
The [sidecar overview](../README.md#apple-core-ml-sidecars) links to each
engine's export, input-routing and fallback contracts.

### Tests

For the non-GPU engine suite:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/ggml-install" -DSPEECH_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build -LE 'gpu|perf' --output-on-failure
```

Fixture-backed engine tests are disabled when required files are absent at
configure time. Reconfigure after provisioning models/references. If enabling
Whisper tests too, its committed weightless fixtures cover pipeline smokes;
`test-vad-full` requires a downloaded model. The model-download-free CI filter is:

```sh
ctest --test-dir build -LE 'gpu|perf|medium|large' \
  -E '^test-vad-full$' --output-on-failure
```

On multi-config generators add `-C Release`. Disabled or skipped tests are
not executed coverage. Model/backend parity setup is documented in the engine
testing guides.

### Standalone builds

Each engine also configures directly. Defaults and ggml acquisition differ:
[Parakeet](../engines/parakeet/docs/build.md), [TTS](../engines/tts/docs/build.md),
[AudioGen](../engines/audiogen/docs/build.md).

## Consumable packages

The [`speech-cpp`](https://github.com/tetherto/qvac-registry-vcpkg/tree/main/ports/speech-cpp)
vcpkg port builds the umbrella. Select engine features such as
`speech-cpp[whisper,parakeet,vulkan]`; backend features select the corresponding
shared `ggml-speech` dependency features. CMake package names remain engine-specific.

| Feature | `find_package(... CONFIG REQUIRED)` | Imported target |
|---|---|---|
| Shared dependency | `ggml` | `ggml::ggml` |
| `speech-cpp[whisper]` | `whisper` | `whisper::whisper` |
| `speech-cpp[parakeet]` | `qvac-parakeet` | `qvac::parakeet` |
| `speech-cpp[tts]` | `tts-cpp` | `tts-cpp::tts-cpp` |
| `speech-cpp[audiogen]` | `audiogen-cpp` | `audiogen-cpp::audiogen-cpp` |

For example, after installing `speech-cpp[tts]` through the registry/toolchain:

```cmake
find_package(tts-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE tts-cpp::tts-cpp)
```
