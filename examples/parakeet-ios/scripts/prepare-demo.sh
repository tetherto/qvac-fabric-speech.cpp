#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_ROOT}/../.." && pwd)"

REVISION="541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
NEMO_SHA256="3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d"
NEMO_URL="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/${REVISION}/parakeet-tdt-0.6b-v3.nemo"

MODEL_DIR="${PARAKEET_IOS_MODEL_DIR:-${REPO_ROOT}/engines/parakeet/models}"
VENV_DIR="${PARAKEET_IOS_VENV:-${REPO_ROOT}/.venv-coreml}"
BUILD_DIR="${PARAKEET_IOS_BUILD_DIR:-${APP_ROOT}/build-ios}"
SAMPLE="${PARAKEET_IOS_SAMPLE:-${REPO_ROOT}/test.wav}"
if [[ ! -f "$SAMPLE" ]]; then
  SAMPLE="${REPO_ROOT}/engines/parakeet/test/samples/jfk.wav"
fi

NEMO="${MODEL_DIR}/parakeet-tdt-0.6b-v3.nemo"
Q8="${MODEL_DIR}/parakeet-tdt-0.6b-v3.q8_0.gguf"
F16="${MODEL_DIR}/parakeet-tdt-0.6b-v3.f16.gguf"
PACKAGE="${MODEL_DIR}/parakeet-tdt-0.6b-v3-encoder.mlpackage"
COMPILED="${MODEL_DIR}/parakeet-tdt-0.6b-v3-encoder.mlmodelc"

say() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
done_message() { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing '$1'. See examples/README.md."
}

gguf_is_qvac() {
  local path="$1"
  "$VENV_DIR/bin/python" - "$path" <<'PY'
import sys
import gguf

reader = gguf.GGUFReader(sys.argv[1])
raise SystemExit(0 if "parakeet.encoder.d_model" in reader.fields else 1)
PY
}

convert_model() {
  local quant="$1"
  local destination="$2"
  local temporary="${destination}.generating"

  if [[ -f "$destination" ]] && gguf_is_qvac "$destination"; then
    done_message "Using $(basename "$destination")"
    return
  fi
  if [[ -f "$destination" ]]; then
    local preserved="${destination%.gguf}.incompatible-$(date +%Y%m%d-%H%M%S).gguf"
    mv "$destination" "$preserved"
    printf 'Preserved incompatible GGUF as %s\n' "$preserved"
  fi

  say "Creating the QVAC ${quant} GGUF"
  "$VENV_DIR/bin/python" \
    "$REPO_ROOT/engines/parakeet/scripts/convert-nemo-to-gguf.py" \
    --ckpt "$NEMO" \
    --hf-repo nvidia/parakeet-tdt-0.6b-v3 \
    --quant "$quant" \
    --out "$temporary"
  mv "$temporary" "$destination"
  gguf_is_qvac "$destination" || die "Generated GGUF failed metadata validation."
  done_message "Created $(basename "$destination")"
}

require_command curl
require_command cmake
require_command shasum
require_command xcrun
require_command python3.11
xcrun --find coremlcompiler >/dev/null 2>&1 \
  || die "Core ML compiler not found. Install and open the full Xcode application."

mkdir -p "$MODEL_DIR"

say "Preparing the isolated Core ML conversion environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3.11 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  -r "$REPO_ROOT/engines/parakeet/scripts/requirements-coreml.txt" \
  "sentencepiece>=0.2,<0.3"
done_message "Python conversion environment is ready"

say "Checking the pinned NVIDIA checkpoint"
if [[ -f "$NEMO" ]] && [[ "$(shasum -a 256 "$NEMO" | awk '{print $1}')" == "$NEMO_SHA256" ]]; then
  done_message "Using the verified checkpoint"
else
  if [[ -f "$NEMO" ]]; then
    die "Existing checkpoint has the wrong checksum: $NEMO"
  fi
  if [[ -f "${NEMO}.download" ]]; then
    curl --fail --location --retry 3 --continue-at - --progress-bar \
      --output "${NEMO}.download" "$NEMO_URL"
  else
    curl --fail --location --retry 3 --progress-bar \
      --output "${NEMO}.download" "$NEMO_URL"
  fi
  actual="$(shasum -a 256 "${NEMO}.download" | awk '{print $1}')"
  [[ "$actual" == "$NEMO_SHA256" ]] \
    || die "Checkpoint checksum mismatch. Expected $NEMO_SHA256, got $actual."
  mv "${NEMO}.download" "$NEMO"
  done_message "Downloaded and verified the checkpoint"
fi

convert_model q8_0 "$Q8"
convert_model f16 "$F16"

if [[ -d "$COMPILED" ]]; then
  done_message "Using the compiled Core ML encoder"
else
  say "Exporting and compiling the Core ML encoder"
  if [[ -e "$PACKAGE" ]]; then
    preserved_package="${PACKAGE%.mlpackage}.incomplete-$(date +%Y%m%d-%H%M%S).mlpackage"
    mv "$PACKAGE" "$preserved_package"
    printf 'Preserved the previous uncompiled package as %s\n' "$preserved_package"
  fi
  "$VENV_DIR/bin/python" \
    "$REPO_ROOT/engines/parakeet/scripts/export-encoder-coreml.py" \
    --gguf "$F16" \
    --n-mel-frames 1501 \
    --palettize-bits 6 \
    --palettize-group-size 16 \
    --out "$PACKAGE" \
    --compile-dir "$MODEL_DIR"
  [[ -d "$COMPILED" ]] || die "Core ML compilation did not create $COMPILED"
  done_message "Created the compiled Core ML encoder"
fi

say "Generating the device-only Xcode project"
"$SCRIPT_DIR/generate-xcode-project.sh" \
  --gguf "$Q8" \
  --coreml "$COMPILED" \
  --sample "$SAMPLE" \
  --output "$BUILD_DIR"

printf '\n\033[1;32mDemo ready.\033[0m\n'
printf 'Open it with:\n  open %q\n' "$BUILD_DIR/ParakeetLab.xcodeproj"
