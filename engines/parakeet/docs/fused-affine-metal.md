# Experimental exact affine Metal execution

This opt-in path executes `parakeet-affine-v1` GGUF carriers without restoring
whole dense weight matrices. Model files and effective recipe values are
unchanged. It requires the matching `tetherto/qvac-ext-ggml` dependency branch
`feat/parakeet-fused-affine`, initially pinned at
`5ad06152d1041fb1aedfa0a2210511ca2eebf130`.

Build a separate checkout; do not replace dependencies of an existing benchmark.
Use the ordinary Metal build with that GGML checkout at `engines/parakeet/ggml`.
`PARAKEET_AFFINE_FUSED=1` enables the new encoder graph path. Core ML should be
disabled for native comparisons (`PARAKEET_COREML_DISABLE=1`).
`PARAKEET_DEBUG_AFFINE=1` reports fused matmul counts when an encoder graph is
built. Leave this diagnostic off in timed runs.

The new GGML API uses unit-scale Q4_0/Q8_0 blocks as lossless integer carriers,
plus separate per-output-channel FP32 minimum/step. It decodes weight tiles,
performs separate FP32 multiply/add, rounds weights to FP16, and applies SIMD
matrix multiplication with FP32 accumulation. Only contiguous 2D FP32
activations are accepted by the backend; the engine adds an activation copy
when needed. CPU execution is explicitly rejected. The ordinary reconstruction
path remains the default. An explicit fused request with old GGML headers or a
non-Metal engine backend fails rather than silently selecting another path.

`scripts/export-affine-metal-cases.py` exports real FFN carrier cases from a
packed GGUF and its exact dense control. Activations are synthetic,
FP16-representable values stored in FP32. Output tails and constant channels
are separate synthetic cases. `scripts/bench-affine-metal.mm` compares native
GGML dense, native reconstruction, native fused, and a standalone Metal
dispatch, rotating 20 samples after five warmups. It verifies bitwise weight
restoration and relative L2 below 0.001 against dense and CPU references.
These primitive timings do not establish transcription speed or WER. Validate
actual audio and per-record differences before making deployment claims.

The earlier standalone-only prototype is preserved at speechcpp commit
`a61e549`. Its real FFN cases were 0.35–0.88 ms fused versus 2.3–3.0 ms with
native reconstruction; dense FP16 remained generally faster. Those are
resident, synchronized primitive timings, not encoder or full-call latency.
