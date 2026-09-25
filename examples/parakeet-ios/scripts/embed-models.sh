#!/usr/bin/env bash

set -euo pipefail

gguf="$1"
coreml="$2"
sample="$3"
bundle="${TARGET_BUILD_DIR:?}/${FULL_PRODUCT_NAME:?}"

cp -f "$gguf" "$bundle/parakeet-tdt-0.6b-v3.q8_0.gguf"
rm -rf "$bundle/parakeet-tdt-0.6b-v3-encoder.mlmodelc"
cp -R "$coreml" "$bundle/parakeet-tdt-0.6b-v3-encoder.mlmodelc"
sample_output="$bundle/parakeet-demo-sample.wav"
sample_temporary="$bundle/parakeet-demo-sample.tmp.wav"
rm -f "$sample_temporary"
afconvert "$sample" "$sample_temporary" -f WAVE -d LEI16@16000 -c 1
mv -f "$sample_temporary" "$sample_output"
