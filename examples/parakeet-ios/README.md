# Parakeet Lab for iOS

For the non-technical build, device-installation, and recording walkthrough,
start with [`../README.md`](../README.md). The one-command preparation path is:

```bash
./examples/parakeet-ios/scripts/prepare-demo.sh
```

Native SwiftUI demo for Parakeet TDT 0.6B v3 Q8_0. It compares the same bundled
GGUF in two on-device modes:

- **Core ML** — compiled Core ML encoder with the TDT decoder on QVAC Metal.
- **Metal** — encoder and decoder on QVAC Metal through ggml.

The app accepts the bundled sample, Files imports, and direct HTTPS audio URLs.
Audio is converted locally to mono 16 kHz PCM WAV. Each benchmark transcribes
the audio once, streams timestamped segments into the UI, and reports inference
seconds, real-time speed, and model-load seconds.

## Requirements

- Apple Silicon Mac with Xcode and CMake.
- Physical arm64 iPhone or iPad running iOS 16.4 or later.
- `parakeet-tdt-0.6b-v3.q8_0.gguf`.
- The matching compiled `parakeet-tdt-0.6b-v3-encoder.mlmodelc` directory.

Models are deliberately not committed. Their paths are passed while generating
the Xcode project, and every Xcode build copies them into `ParakeetLab.app` using
the exact filenames expected by the native engine.

## Generate and run

From the repository root:

```bash
chmod +x examples/parakeet-ios/scripts/generate-xcode-project.sh

examples/parakeet-ios/scripts/generate-xcode-project.sh \
  --gguf /absolute/path/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --coreml /absolute/path/parakeet-tdt-0.6b-v3-encoder.mlmodelc

open examples/parakeet-ios/build-ios/ParakeetLab.xcodeproj
```

In Xcode, select the **ParakeetLab** target, choose your signing team and a
connected iOS device, then Run. This project is device-only because the demo is
intended to measure the device's Metal GPU and Neural Engine.

To bundle another sample audio file:

```bash
examples/parakeet-ios/scripts/generate-xcode-project.sh \
  --gguf /absolute/path/model.q8_0.gguf \
  --coreml /absolute/path/model-encoder.mlmodelc \
  --sample /absolute/path/sample.wav
```

The GGUF plus Core ML sidecar make the app bundle large. Keep them outside Git
and provide their paths only at project-generation time.
