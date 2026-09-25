#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
model="$root/engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf"
audio="$root/test.wav"
build="$root/examples/parakeet-ios/build-integration"
fixture="$build/11-minute-mono.wav"

[[ -f "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }
[[ -d "${model%.q8_0.gguf}-encoder.mlmodelc" ]] || {
  echo "Missing Core ML sidecar: ${model%.q8_0.gguf}-encoder.mlmodelc" >&2
  exit 1
}
[[ -f "$audio" ]] || { echo "Missing test audio: $audio" >&2; exit 1; }

cmake -S "$root/examples/parakeet-ios/Tests" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" --target parakeet-ios-long-audio-integration -j 8

ffmpeg -hide_banner -loglevel error -y -i "$audio" -t 660 \
  -ac 1 -ar 16000 -c:a pcm_s16le "$fixture"

exec "$build/parakeet-ios-long-audio-integration" "$model" "$fixture"
