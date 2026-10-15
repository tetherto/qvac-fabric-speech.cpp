# Experimental exact affine Metal execution

This opt-in path executes `parakeet-affine-v1` GGUF carriers without restoring
whole dense weight matrices. Model files and effective recipe values are
unchanged. It requires the matching `tetherto/qvac-ext-ggml` dependency branch
`feat/parakeet-fused-affine-compact`, pinned at
`033f5c56d6d7476b08bfc83993138adddec6f5c6`. This includes the original fused
operation and the tested 8 KiB staging-memory refinement. The earlier
`feat/parakeet-fused-affine` / `5ad06152` branch is the first fused baseline.

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

## Measured native implementation on M4

The integrated engine was tested at speechcpp commit `994ae970` with GGML
`5ad06152`. The exact retained mixed-precision GGUF is 610,573,408 bytes; no
model file changes or additional sidecars are needed. On 99 complete short
recordings per clean/noisy condition, three fresh-process repetitions gave
median whole-transcription times of 6.048/6.140 seconds, versus
26.122/26.076 for reconstruction, 7.093/7.163 for Redux Photon MPS, and
4.783/4.871 for original FP16 native Metal. Eight warmups and loading/compilation
are excluded. All 594 fused transcripts match reconstruction exactly, with
27/189 word errors out of 940 reference words per condition. These are subset
measurements, not confidence intervals or full-suite qualification.

A compatible shared-memory-reuse variant is pinned at GGML commit
`033f5c56d6d7476b08bfc83993138adddec6f5c6`, branch
`feat/parakeet-fused-affine-compact`. It uses one 8 KiB allocation for operand
staging and the subsequent output-tail phase, instead of separate 6+8 KiB
arrays. Arithmetic and model weights are unchanged. In a separate paired
three-repetition comparison, it achieved 5.861/5.977 seconds versus
6.004/6.119 for the original fused binary and 4.770/4.875 for a fresh original
FP16 control. All 594 variant transcripts match V1 exactly. Redux was not
rerun in this second comparison; its preceding results are context only.
The improvement over V1 is about 2.3–2.4%; original FP16 remains faster.

Both versions passed 28 real-weight/synthetic-edge primitive cases, five
regression test groups, and the CPU/invalid-input rejection checks in
`scripts/test-affine-backend-contract.cpp`. Full prediction, file hash and
paired-ratio audits passed. The implementation remains opt-in and experimental.


## Integration and PR handoff

Two repositories carry this implementation:

- `tetherto/qvac-fabric-speech.cpp`, branch
  `feat/parakeet-fused-affine-metal`: exact affine GGUF exporter/reader,
  engine graph integration, fixed Core ML padding/valid-length handling,
  primitive probes and regression tests.
- `tetherto/qvac-ext-ggml`, branch
  `feat/parakeet-fused-affine-compact`, commit
  `033f5c56d6d7476b08bfc83993138adddec6f5c6`: the affine operation API,
  Metal support/dispatch/kernel, CPU rejection and compact staging storage.

The speechcpp branch is based on its `master`; the GGML changes are based on
its `speech` integration branch. Review/land the dependency there before
switching ordinary consumers to it. The default `scripts/setup-ggml.sh` still
selects `speech`, which does not yet expose this experimental operation. Do
not assume a normal dependency setup enables fusion: an explicit fused request
with an older GGML build fails by design. For feature review in a **new** checkout,
clone the compact dependency explicitly before configuring:

```sh
git clone --branch feat/parakeet-fused-affine-compact \
  https://github.com/tetherto/qvac-ext-ggml.git engines/parakeet/ggml
git -C engines/parakeet/ggml checkout 033f5c56d6d7476b08bfc83993138adddec6f5c6
cmake -S engines/parakeet -B build/parakeet-affine \
  -DCMAKE_BUILD_TYPE=Release -DPARAKEET_COREML=ON \
  -DPARAKEET_INSTALL=OFF -DPARAKEET_BUILD_EXAMPLES=OFF
cmake --build build/parakeet-affine --target qvac-parakeet \
  test-coreml-paths test-coreml-shapes test-packed-affine -j4
ctest --test-dir build/parakeet-affine --output-on-failure \
  -R 'test-coreml-paths|test-coreml-shapes|test-packed-affine|^test_coreml_padding$'
```

Use an appropriate Python environment at configuration time for the Python
tests. Core ML conversion was validated with PyTorch 2.7 / coremltools 9.
The experimental standalone backend-contract probe is supplied under
`scripts/test-affine-backend-contract.cpp`; it was compiled and run against the
matching Metal build separately, and is not registered as a CTest target.
Install packaging and non-Apple/device qualification are separate from the
validated native Metal path. No default backend or global dependency pin is
changed by this feature branch.

For native execution set `PARAKEET_AFFINE_FUSED=1` and
`PARAKEET_COREML_DISABLE=1`. For sidecar execution use a model-derived masked
encoder and unset `PARAKEET_COREML_DISABLE`. Core ML replaces the encoder;
the fused encoder operation is not used in that lane. The existing speechcpp
frontend/decoder remain in use. Default reconstruction remains available.

## Larger held-out native validation

The unchanged runtime at speechcpp `994ae970` and compact GGML `033f5c56`
completed three fresh-process repetitions per model on all LibriSpeech
test-clean (2,620 clips), test-other (2,939 clips), and a deterministic paired
0 dB white-noise copy of test-clean. Native speechcpp uses Metal, four CPU
threads and no Core ML. The Redux control uses Photon MPS on the same device
and common complete audio. No resampling, duration truncation or external
chunking was introduced. Eight warmups and loading/compilation are excluded
from full file-to-text wall time.

| Condition | Original FP16 WER | Packed affine WER | Redux WER | Median paired latency reduction vs Redux |
|---|---:|---:|---:|---:|
| Clean | 2.250% | 2.329% | 2.649% | 22.19% |
| Difficult | 3.885% | 4.174% | 5.083% | 20.09% |
| White noise, 0 dB | 9.213% | 10.563% | 12.220% | 21.25% |

All 27 completion records and 73,611 predictions passed the frozen input,
source, environment, route and output audit. Raw transcripts repeated exactly
across repetitions. Packed affine retains one difficult-speech blank and two
noisy-speech blanks per repetition; controls have none. A paired reader-cluster
bootstrap favors packed affine over Redux on all three sets, but does not
establish results on other languages/domains. The clean difference from original
has an interval crossing zero, which is not proof of equivalence. Original FP16
remains faster and more accurate; Redux remains smaller. Three timing repetitions
are not confidence intervals.

The actual packed GGUF remains 610,573,408 bytes versus 1,256,547,552 bytes for
the matched original FP16 layout. A separate matched short-clip Core ML comparison is tracked independently;
these native results do not establish its speed or quality.
The full native runtime/source files are unchanged by this documentation update.
