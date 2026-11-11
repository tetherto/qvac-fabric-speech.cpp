# Physical iPhone Core ML validation

This small UIKit host runs the ordinary Parakeet CLI on a worker thread, emits
process peak RSS, physical footprint and thermal state, and exits with the CLI status. It also runs
the full-EU API/streaming regression test (`--api-check`) and queries the actual
device's Core ML compute plans (`--compute-plan`). It is a development harness;
models, samples, build products and signing credentials stay outside the source tree.

For a desktop CLI using an explicit current GGML checkout, this small parent
project matches the validation build and avoids ignored stale local headers:

```bash
export SPEECH_SOURCE="$PWD/engines/parakeet"
export GGML_SOURCE="/path/to/current/qvac-ext-ggml"
DRIVER_DIR=/tmp/eu-desktop-driver
mkdir -p "$DRIVER_DIR"
cat > "$DRIVER_DIR/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(eu-desktop LANGUAGES C CXX OBJC OBJCXX)
enable_testing()
set(PARAKEET_COREML ON CACHE BOOL "" FORCE)
set(PARAKEET_INSTALL OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_EXECUTABLES ON CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_TESTS ON CACHE BOOL "" FORCE)
add_subdirectory("$ENV{GGML_SOURCE}" ggml)
add_subdirectory("$ENV{SPEECH_SOURCE}" parakeet)
target_include_directories(qvac-parakeet BEFORE PRIVATE "$ENV{GGML_SOURCE}/include")
target_include_directories(parakeet.coreml BEFORE PRIVATE "$ENV{GGML_SOURCE}/include")
CMAKE
cmake -S "$DRIVER_DIR" -B /tmp/eu-desktop -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_METAL=ON \
  -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" \
  -DPARAKEET_TEST_MODEL_DIR="$ASSETS/models"
cmake --build /tmp/eu-desktop --target parakeet -j4
PARAKEET_CLI=/tmp/eu-desktop/parakeet/parakeet
```

Use Xcode with a paired device, Developer Mode, and a development provisioning
profile for that device. The profile must permit the bundle ID you provide.

```bash
cmake -S engines/parakeet/examples/ios-coreml-bench -B /tmp/eu-phone -G Ninja \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 \
  -DCMAKE_BUILD_TYPE=Release -DQVAC_GGML_SOURCE_DIR="$GGML_SOURCE"
cmake --build /tmp/eu-phone --target qvac-eu-bench -j4
python3 engines/parakeet/examples/ios-coreml-bench/package.py \
  --binary /tmp/eu-phone/qvac-eu-bench.app/qvac-eu-bench \
  --profile "$PROVISIONING_PROFILE" --identity "$SIGN_IDENTITY" \
  --bundle-id "$BUNDLE_ID" --model-dir "models=$MODEL_DIR" \
  --samples "$ASSETS/inputs" --output /tmp/TranscribePsyBench.app
xcrun devicectl device install app --device "$DEVICE_ID" /tmp/TranscribePsyBench.app
xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
  --model models/transcribepsy-250m-eu.f16.gguf --wav jfk.wav \
  --language auto --backend cpu --threads 4 \
  --bench --bench-runs 5 --bench-warmup 2 --require-coreml --bench-json bench.json
xcrun devicectl device copy from --device "$DEVICE_ID" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --source Documents/bench.json --destination /tmp/bench.json
```

`GGML_SOURCE` is a current `qvac-ext-ggml` checkout on its `speech` branch.
`MODEL_DIR` contains the GGUF and exported `.mlmodelc` sidecars from the
[full-EU guide](../../docs/transcribepsy-eu-coreml.md). Packaging copies only
GGUFs and compiled models, excluding checkpoints and `.mlpackage` intermediates.
Pass another `--model-dir models6=PATH` to include a compressed comparison set.
Use a fresh output bundle when changing model sets so removed resources cannot
remain in a reused bundle.

For API and physical-device placement checks (append `--unwindowed` to the API
command when the flexible sidecars are packaged; append `--context-parity`
instead to compare precise callback/push tokens with native and test encoder
padding boundaries):

```bash
xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
  --api-check models/transcribepsy-250m-eu.f16.gguf jfk.wav fr-multipiece-16k.wav
xcrun devicectl device process launch --device "$DEVICE_ID" --console -- "$BUNDLE_ID" \
  --compute-plan models
```

Set runtime overrides through `devicectl --environment-variables`, for example
`'{"PARAKEET_COREML_DISABLE":"1"}'` for CPU/Metal baselines or
`'{"PARAKEET_COREML_ONLY":"1"}'` to measure encoder-owned memory. Set
`--backend metal` for a Metal baseline. For callbacks add `--stream`; for push PCM
add `--stream --stream-duplex`, adjusting context/feed options as needed. JSON output
filenames are redirected into the app's Documents directory. Keep the phone
connected and available for development while running the matrix.


To reproduce the complete serial measurement matrix, export the FP16 fixed and
FP32 flexible and static GPU sidecars into `$ASSETS/models`, and the six-bit fixed sidecars into
`$ASSETS/models6`. Place the same F16 GGUF and decoder sidecar in both directories;
place the Q8 GGUF in `models` as well. CPU-only and unwindowed deployments need
the FP32 sidecars beside whichever GGUF they use. Build the desktop CLI with
`PARAKEET_COREML=ON` and a current GGML dependency. Keep other conversion/build
and inference jobs stopped during measurements.

```bash
python3 engines/parakeet/examples/ios-coreml-bench/prepare_inputs.py \
  --samples engines/parakeet/test/samples --output "$ASSETS/inputs"
# Package and install the prepared inputs and both model directories as above.
python3 engines/parakeet/examples/ios-coreml-bench/benchmark.py \
  --assets "$ASSETS" --output "$ASSETS/results" --binary "$PARAKEET_CLI" \
  --device "$DEVICE_ID" --bundle "$BUNDLE_ID"
python3 engines/parakeet/examples/ios-coreml-bench/stream_benchmark.py \
  --assets "$ASSETS" --output "$ASSETS/stream-results" --binary "$PARAKEET_CLI" \
  --device "$DEVICE_ID" --bundle "$BUNDLE_ID"
$PYTHON engines/parakeet/examples/ios-coreml-bench/compiled_precision.py \
  --assets "$ASSETS" --binary "$PARAKEET_CLI" --output "$ASSETS/compiled-precision.json"
$PYTHON engines/parakeet/examples/ios-coreml-bench/precise_precision.py \
  --assets "$ASSETS" --binary "$PARAKEET_CLI" --output "$ASSETS/precise-precision.json"
```

`benchmark.py` runs Mac first, then iPhone, and resumes completed case names from
`results.json`. Use a new output directory after changing a model, binary or
configuration. Each case preserves its command/settings, per-run timings,
transcript, reference WER/CER, gate decision, actual Core ML dispatch, and process
peak RSS. Streaming measurements use three independent processes per case and
exclude load from the reported inference time; they do not claim an in-process
warm benchmark. API regression logs also show the first callback's input time
and processing time. Pass `--devices mac` or `--devices iphone` for one platform.

The precise sweep checks actual compiled CPU/GPU predictions against the float32
reference, including 1–8 mel frames, aligned/nonaligned frontend padding boundaries,
60-second English and 51-second adapted French. Streaming `--resume` reuses
completed rows, and `--backends coreml` reruns that backend separately when its
implementation changes. Keep the old logs and retain only unchanged baselines.
Phone cases wait for nominal thermal state before each process. Initial static
FP32 specialization is separate from cached inference; preserve its load logs.

For automatic/two-second and explicit one-second push, compare identical native/Core ML
contexts with `default_stream_parity.py`, using the same arguments as
`stream_benchmark.py`. It records exact native transcript equality and reference
WER/CER for English/French, long speech and explicit adapted routing. This check
records the model-specific default and preserves explicit one-second coverage.

Explicit encoder context and decoder/compute policy regressions use
`context_policy_benchmark.py` with the same arguments as `benchmark.py`.
`auto_quality_benchmark.py` checks automatic encoding below/above decoder
crossover, native continuous decode, and missing optional decoder/ownership.
For its encoder-only deployment, prepare a `models-nodecoder` directory with the
same GGUF/encoder/gate/precise sidecars and omit the decoder sidecar; package it
with `--model-dir models-nodecoder=PATH`. This deployment still uses Core ML
encoders and records native decoder dispatch truthfully.

For the focused first/warm French streaming experiment, append `--warm-stream`
to the `--api-check` command. It runs one first session plus three reused-Engine
sessions per callback/push mode, grouping modes to retain the precise capacity,
and compares native Metal with Core ML. The first session includes lazy precise
program loading; repeated sessions retain identical text. Use `--explicit-context`
instead to exercise API 750-frame windows/25-frame context with auto/force0/force1
decoder policies and actual prediction counters.
