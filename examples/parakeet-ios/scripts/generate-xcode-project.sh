#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  generate-xcode-project.sh --gguf PATH --coreml PATH [--sample PATH] [--output DIR]

Required model inputs are copied into ParakeetLab.app by the Xcode build.
The Core ML input must be an already compiled .mlmodelc directory.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGUF=""
COREML=""
SAMPLE=""
OUTPUT="${APP_ROOT}/build-ios"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf) GGUF="${2:-}"; shift 2 ;;
    --coreml) COREML="${2:-}"; shift 2 ;;
    --sample) SAMPLE="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$GGUF" ]] || { echo "GGUF not found: $GGUF" >&2; exit 1; }
[[ -d "$COREML" ]] || { echo "Compiled Core ML model not found: $COREML" >&2; exit 1; }

GGUF="$(cd "$(dirname "$GGUF")" && pwd)/$(basename "$GGUF")"
COREML="$(cd "$COREML" && pwd)"

args=(
  -S "$APP_ROOT"
  -B "$OUTPUT"
  -G Xcode
  -DCMAKE_SYSTEM_NAME=iOS
  -DCMAKE_OSX_SYSROOT=iphoneos
  -DCMAKE_OSX_ARCHITECTURES=arm64
  -DCMAKE_OSX_DEPLOYMENT_TARGET=16.4
  -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=YES
  -DPARAKEET_IOS_GGUF="$GGUF"
  -DPARAKEET_IOS_COREML="$COREML"
)
if [[ -n "$SAMPLE" ]]; then
  [[ -f "$SAMPLE" ]] || { echo "Sample audio not found: $SAMPLE" >&2; exit 1; }
  SAMPLE="$(cd "$(dirname "$SAMPLE")" && pwd)/$(basename "$SAMPLE")"
  args+=( -DPARAKEET_IOS_SAMPLE="$SAMPLE" )
fi

cmake "${args[@]}"
printf '\nGenerated: %s/ParakeetLab.xcodeproj\n' "$OUTPUT"
printf 'Open the project, select your signing team and an iOS device, then Run.\n'
