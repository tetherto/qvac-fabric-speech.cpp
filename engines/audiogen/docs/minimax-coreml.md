# MiniMax-Music3 Core ML transformer blocks

Part of the [audiogen engine documentation](../README.md).

The experimental Apple Core ML path replaces individual MiniMax-Music3 LM,
acoustic depth-decoder, and DiT transformer blocks. Tokenization, embeddings,
output heads, sampling, conditioning, flow integration, window overlap, and
the DAC vocoder retain their native ggml implementations. Android and iOS
availability is unchanged.

## Build and export

Build the library against the speech fork of ggml with `AUDIOGEN_COREML=ON`:

```sh
cmake -S engines/audiogen -B build-audiogen-coreml \
  -DCMAKE_BUILD_TYPE=Release -DAUDIOGEN_COREML=ON \
  -DCMAKE_PREFIX_PATH=/path/to/ggml-install
cmake --build build-audiogen-coreml -j 8

python3.11 -m venv .venv-minimax-coreml
.venv-minimax-coreml/bin/python -m pip install \
  -r engines/parakeet/scripts/requirements-coreml.txt

.venv-minimax-coreml/bin/python engines/audiogen/scripts/export-minimax-coreml.py \
  --gguf /path/to/mm3-lm-f16.gguf --stage lm --compile
.venv-minimax-coreml/bin/python engines/audiogen/scripts/export-minimax-coreml.py \
  --gguf /path/to/mm3-synth-f16.gguf --stage depth --compile
.venv-minimax-coreml/bin/python engines/audiogen/scripts/export-minimax-coreml.py \
  --gguf /path/to/mm3-synth-f16.gguf --stage dit --compile
```

The exporter reads GGUF tensors, including supported quantized types, and
exports every block in the selected stage. `--layer N` exports one block.
The production checkpoint has 36 LM, four depth, and 36 DiT blocks. Packages
require macOS 14 or later and are compiled with Apple's `coremlcompiler`.

Compiled blocks live beside their source GGUF:

```text
mm3-lm-f16.gguf
mm3-lm-f16.gguf-coreml/lm-block-0.mlmodelc
mm3-lm-f16.gguf-coreml/lm-block-1.mlmodelc
...
mm3-synth-f16.gguf
mm3-synth-f16.gguf-coreml/depth-block-0.mlmodelc
mm3-synth-f16.gguf-coreml/dit-block-0.mlmodelc
...
```

Export before creating the engine. `Engine::create` and `mm3-replay` discover
compiled blocks and keep the accepted models resident until unload. The
compute policy is CPU and Neural Engine; the ggml `device` selection still
controls the remaining native operations. A compute policy alone does not
prove that a particular operation ran on the ANE. Use Apple's compute plan
or Instruments to inspect placement.

## Compatibility and state

Each block records the SHA256 of its complete source GGUF, its stage, layer,
and attention dimensions. A stale hash or incompatible input shape selects
the native block. Output names and types are checked, and a cached, stateless
probe validates dynamically inferred output shapes before graph selection.
Partial exports are supported: a request can use Core ML
and ggml blocks together. Building without Core ML preserves the ggml path.

The native KV cache remains authoritative. Core ML returns only the current
step's keys and values, which the native graph writes into the same cache
used by the native blocks. LM F16 cache storage and depth F32 cache storage
are preserved. Prefill and cached decoding share the same exported model;
input shape ranges follow the checkpoint's context and depth metadata.
Larger or unsupported artifact shapes use ggml without truncating a request.

Normalization and attention score products retain F32. Selected feed-forward
projections use F16 for ANE eligibility. The DiT feed-forward input projection
is expressed as an equivalent pointwise convolution so its F16 compute can
use ANE while its attention and output projections retain F32. Prediction
outputs must have
the expected shapes and contain finite values; a prediction failure is
reported to the caller before consuming generated outputs. Failed callbacks
write finite placeholders to isolate the failed request from later cache reuse. A new generation
clears previous prediction errors and follows the existing cache reset and
cancellation paths. Cancellation is observed between graph operations; an
individual synchronous Core ML prediction is not interrupted.

Both native weights and accepted Core ML models remain resident. This uses
more memory than the native path. The existing memory-fit preflight projects
ggml allocations; it does not currently measure Core ML residency. Do not
interpret its projection as a complete Core ML memory estimate.

## Verification

Portable unit tests check missing-sidecar fallback. Apple unit tests also
check source hashing. Python tests cover causal cached attention, stale cache
masking, NeoX rotary layout, CFG isolation, the DiT value/gate order, export
ranges, and precision selection.

Create a small complete GGUF pair and export all three stages to exercise the
native bridge without downloading checkpoint weights:

```sh
build-audiogen-coreml/test-minimax-coreml --write-fixtures /tmp/minimax-coreml-fixture

.venv-minimax-coreml/bin/python engines/audiogen/scripts/export-minimax-coreml.py \
  --gguf /tmp/minimax-coreml-fixture/mm3-lm-f16.gguf --stage lm --compile
.venv-minimax-coreml/bin/python engines/audiogen/scripts/export-minimax-coreml.py \
  --gguf /tmp/minimax-coreml-fixture/mm3-synth-f16.gguf --stage depth --compile
.venv-minimax-coreml/bin/python engines/audiogen/scripts/export-minimax-coreml.py \
  --gguf /tmp/minimax-coreml-fixture/mm3-synth-f16.gguf --stage dit --compile

.venv-minimax-coreml/bin/python engines/audiogen/test/prepare_minimax_coreml_fixtures.py \
  --fixture-dir /tmp/minimax-coreml-fixture

AUDIOGEN_TEST_MINIMAX_COREML_DIR=/tmp/minimax-coreml-fixture \
  build-audiogen-coreml/test-minimax-coreml
```

This compares prefill plus three cached LM steps, two acoustic frames, and
batched DiT CFG against the native graphs. It checks repeated-generation
cache reset, incompatible shapes, stale source hashes, and prediction-error
propagation. The fixture gate is a maximum absolute error of 0.003; repeated
calls must agree within 0.000001. Missing fixtures produce CTest skip code 77.
The Apple audiogen workflow creates the fixtures and runs this test explicitly.

For checkpoint diagnostics, `test-minimax-coreml --real` uses the same
environment variable with a full GGUF pair. It compares the first real block
of each stage on Metal and Core ML, including native output heads, using
relative RMS error at most 0.002 and cosine similarity at least 0.9999.
It also reports maximum absolute error. This is block-level verification;
it does not replace listening to complete generated music or measuring
matched end-to-end workloads.
