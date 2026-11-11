# Full TranscribePsy EU on Apple Core ML

The full 25-language `qvac/TranscribePsy-250M-EU` uses its 24-layer English
FastConformer trunk, six CAFE adapters for the `other` route, and a language gate
at layer 11. CPU and Metal already execute both routes. Core ML exports use
channel-first fixed windows and explicit padding masks so the English trunk,
adapted trunk, and 12-layer gate prefix run through the same validated encoder
contract on macOS and iOS. The small attentive-pooling language classifier stays
on CPU. `auto` uses that classifier's original threshold and cropped input.

## Export and deployment

Convert the checkpoint, adapters and gate to one GGUF as described in
[models.md](models.md#transcribepsy). Use an isolated environment with
`scripts/requirements-coreml.txt` and `safetensors`, then export each route:

```bash
MODEL_DIR=engines/parakeet/models
PYTHON=.venv-coreml/bin/python
$PYTHON engines/parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --eu-route en --nbits 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder.mlpackage" --compile-dir "$MODEL_DIR"
$PYTHON engines/parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --eu-route other --nbits 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder-other.mlpackage" --compile-dir "$MODEL_DIR"
$PYTHON engines/parakeet/scripts/export-encoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
  --eu-route gate --nbits 0 --check \
  --out "$MODEL_DIR/transcribepsy-250m-eu-encoder-gate.mlpackage" --compile-dir "$MODEL_DIR"
$PYTHON engines/parakeet/scripts/export-decoder-coreml-ane.py \
  --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" --nbits 0 \
  --out "$MODEL_DIR/transcribepsy-250m-eu-decoder.mlpackage" --compile-dir "$MODEL_DIR"
```

Run these commands from the repository root. Main encoders default to 1,500 mel frames
(15 s). The gate export uses at least 2,001 frames, covering the model's first
20 seconds including preprocessing padding. `--nbits 0` retains FP16 weights;
`--nbits 6` uses grouped-channel palettization. `--palettize-workers` can speed up
k-means at the cost of RAM. Export from F16 when evaluating compression separately
from GGUF quantization. The generic exporter accepts full EU only with an explicit route and `--flexible`.

True unwindowed inference and compute policies excluding ANE additionally require
three flexible FP32 sidecars:

```bash
for ROUTE in en other gate; do
  SUFFIX=""; [ "$ROUTE" = en ] || SUFFIX="-$ROUTE"
  $PYTHON engines/parakeet/scripts/export-encoder-coreml.py \
    --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
    --flexible --eu-route "$ROUTE" --precision fp32 \
    --n-mel-frames 1500 --min-frames 1 --max-frames 40000 \
    --out "$MODEL_DIR/transcribepsy-250m-eu-encoder$SUFFIX-flexible.mlpackage" \
    --compile-dir "$MODEL_DIR"
done
```

Flexible full-EU exports default to FP32 and runtime execution uses Core ML CPU
under every accepted compute policy. On the evaluated Apple runtimes, flexible
FP16 predictions contain non-finite features even on CPU; GPU execution of the
FP32 relative-position graph aborts inside MPS gather/reshape kernels. FP16
flexible full-EU export and incompatible runtime sidecars are rejected. Fixed
FP16/6-bit sidecars retain the fast Neural Engine path. The flexible range above
covers about 400 seconds; choose a deployment maximum appropriate to device
memory. Attention memory still grows quadratically for true unwindowed inputs.

Stateful callback/push streaming uses FP32 encoders to preserve words and decoder
continuity. On macOS, export two static capacities per route for GPU execution:

```bash
for FRAMES in 1500 6008; do
  for ROUTE in en other; do
    SUFFIX=""; [ "$ROUTE" = en ] || SUFFIX="-$ROUTE"
    $PYTHON engines/parakeet/scripts/export-encoder-coreml-ane.py \
      --gguf "$MODEL_DIR/transcribepsy-250m-eu.f16.gguf" \
      --eu-route "$ROUTE" --precision fp32 --nbits 0 \
      --window-mel-frames "$FRAMES" \
      --out "$MODEL_DIR/transcribepsy-250m-eu-encoder$SUFFIX-fp32-$FRAMES.mlpackage" \
      --compile-dir "$MODEL_DIR"
  done
done
```

On macOS the smallest fitting static FP32 program runs with CPU/GPU permission.
Larger contexts, policies excluding GPU, and all iOS precise inference use the
flexible FP32 CPU program. iPhone deployments can omit the four static FP32
programs. Only one
precise main encoder (route and capacity combined) remains resident per Engine;
changing the route or required capacity evicts the previous one. The gate has
its own smaller CPU program. Optional static programs improve performance;
matching flexible programs cover every supported context without changing it.

On iOS 18.6.2, static FP32 GPU tensors initially match the native encoder, but
reusing an Engine across routes or running the CPU gate before the GPU encoder
can abort inside MPSGraph (`shape.count = 0 != strides.count = 4`). Allowing Core ML
to allocate its own output did not resolve the assertion. iOS therefore uses the
reliable masked FP32 **Core ML CPU** programs for stateful, unwindowed and precise
compute-policy execution. Every mode still executes Core ML; this path does not
claim Neural Engine or GPU acceleration. Offline fixed FP16/6-bit encoding keeps
its validated Neural Engine execution. macOS retains the validated static GPU
path. The table below reports these stages separately.

Precise programs preserve the native frontend's frame grid: for example, 400 valid
mel frames plus one preprocessing padding frame produce 51 encoder rows. The
extra row contains the subsampling projection bias and participates in native
attention. Masking only convolution inputs, while retaining that row, avoids
changing stateful decoder emissions. Flexible sidecars therefore require both
`mel` and `mel_valid_mask`, with native-padding metadata. Older single-input
full-EU flexible exports must be regenerated.

Build with `PARAKEET_COREML=ON`. Ship the compiled `.mlmodelc` directories
beside the GGUF, preserving the names above; the runtime strips trailing GGUF
quantization tags such as `.f16` or `.q8_0` when deriving names. English and adapted
exports must have matching capacities. Model variant, route and layer-count
metadata prevent EU-sub, exchanged-route, and wrong-depth sidecars from loading.
Missing/incompatible sidecars use native CPU/Metal fallback with a full GGUF.
`PARAKEET_COREML_ONLY=1` skips native encoder allocation only when all three
compatible fixed sidecars are available. Once ownership succeeds, a failed or
missing required flexible encoder returns an error instead of reading unallocated
native weights. Leave `PARAKEET_COREML_ONLY` unset when CPU/Metal recovery is
required. Benchmark JSON exposes `encoder_owned_by_coreml` separately from
`encoder_coreml_all_runs`; availability or dispatch alone does not prove native
weights were omitted.

Accepted `PARAKEET_COREML_COMPUTE_UNITS` values are `cpu_and_ane`, `all`,
`cpu_and_gpu`, and `cpu_only`; the same setting selects decoder units. For full EU,
`all` and `cpu_and_ane` pin the fixed encoders to CPU and Neural Engine. Policies
excluding ANE select precise FP32 programs, because FP16 fixed graphs produce
non-finite predictions on the evaluated CPU/GPU kernels. `cpu_only` uses CPU;
`cpu_and_gpu` uses static GPU programs on macOS when they fit and CPU otherwise. Streaming
always uses precise programs: `cpu_and_ane` uses their reliable CPU execution.
Actual labels report `coreml-ane`, `coreml-cpu-gpu`, or `coreml-cpu`; allowing a processor
does not require dispatching work to it.
Set compute-policy environment variables before constructing the model/Engine;
changing environment variables is not a live reconfiguration API. Additive
CPU-only/CPU-GPU policies retain the English fixed handle for capacity/seam
metadata and availability, and skip unused fixed adapted/gate handles. Precise
programs load lazily as needed. Core ML-only ownership still validates all three
fixed programs before omitting native encoder weights.
`PARAKEET_COREML_DISABLE=1` disables both sidecar families. Core ML may perform
unsupported operations on CPU, so inspect the exporter's compute-plan report
before claiming ANE placement.

## API, CLI, and streaming behavior

`EngineOptions::language` and CLI `--language` accept `en`, `other`, and `auto`
(also empty). Core ML supports `transcribe`, `transcribe_samples`,
`transcribe_mel`, full-input callback streaming, and `StreamSession` push PCM.
Default long recordings and large streaming contexts use bounded, overlapping
encoder windows. Explicit `EngineOptions::long_form_window_frames = -1` or CLI
`--window-sec 0` preserves true unwindowed attention using the flexible sidecar
when the static FP32 sidecar is too short. It does not silently tile the input.
Offline masked partial windows use fixed FP16/6-bit sidecars. Stateful calls use
FP32 programs and retain their native context and encoder frame counts.

Streaming defaults to two-second chunks for full EU, improving measured long
French push quality without changing explicit requests. API `chunk_ms = 0` means
auto; positive values, including 1000, remain explicit. CLI omission selects the
model default; `--stream-chunk-ms 1000` preserves the former one-second behavior.
Other ASR families retain their previous default. The effective value is exposed
by `StreamSession::options()` and CLI diagnostics.

Auto push streams buffer their initial 20 seconds, evaluate the same gate input
as offline transcription, and retain the chosen route for the session. Shorter
streams resolve the gate on `finalize()`. This delay follows the model's gate
contract; select known `en`/`other` to emit at the usual chunk-plus-lookahead
latency. Feed block size does not affect gate selection. New sessions reset the
route, including when reusing one Engine for different languages. The existing
push API had not applied full-EU language selection before this implementation.

The optional batched TDT decoder serves offline calls with automatic window and
context settings (`long_form_window_frames = 0`, `long_form_context_frames = 0`).
Language `auto` is independent of those two settings. Explicit window/context
requests use the ordinary overlapping encoder plan instead of pause-cut segments.
Full-EU automatic decoder
dispatch uses the measured platform crossover: at least four independent
segments on macOS with a Core ML CPU decoder, or eight on iOS with an ANE decoder.
Short inputs retain the faster native sequential decoder. Other decoder placements
require an explicit batch override, because no speed advantage was established.
The default full-EU decoder uses Core ML CPU on macOS and CPU/ANE on iOS.
`PARAKEET_COREML_DECODER_COMPUTE_UNITS` overrides just that stage; otherwise an
explicit global compute-unit setting takes precedence over the platform default. `PARAKEET_ANE_BATCH_DECODE=1`
forces Core ML decode when a compatible decoder is available: automatic window/context can use parallel independent
segments; explicit window/context and true unwindowed input use one continuous
Core ML lane over the context-preserving encoder output. A missing/incompatible optional decoder emits a warning under force1 and uses
native decode, with `decoder_used_coreml=false`; that case is not Core ML decoder
execution. `=0` forces native continuous decode, and
`PARAKEET_ANE_DECODE_HOST=1` compares sequential decode over exactly the same
independent segments when window/context are automatic. Independent windows reset decoder state, so long-form
wording can differ from the sequential path; use `=0` to retain that decoder
continuity. Callback and push streaming retain native stateful decoder
state while accelerating the encoder. This keeps callback timing and continuity.

`EngineResult::encoder_used_coreml` includes the gate prefix and every encoder
window. `decoder_used_coreml` reports actual batched predictions, rather than a
loaded decoder. `StreamingSegment::encoder_used_coreml` reports actual per-segment
encoder/gate dispatch. CLI benchmark logs/JSON report observed dispatch and
`--require-coreml` rejects per-invocation encoder or gate fallback.

## Validation and performance

Measurements use the full EU F16 checkpoint, four GGML threads (Core ML manages
its own scheduling), and matched audio/decoding
settings on an Apple M4 MacBook Air (16 GB, macOS 27.0.1) and iPhone 16e
(A18, iOS 18.6.2), built with Xcode 26.5 and GGML `6fe02a1`. Both builds use
`GGML_NATIVE=OFF`. Timed offline runs exclude model load and warmup; cold and
lazy specialization are reported separately. CPU/Metal baselines explicitly
disable Core ML. Final phone timing rows require nominal thermal state.

The device compute plans assign all fixed English/adapted/gate compute operations
(1537 / 1561 / 781 for FP16, 1500/1500/2001 mel frames) to the Neural Engine.
The masked FP32 CPU programs assign 1816 / 1864 / 940 compute operations to CPU.
Static FP32 macOS plans assign 1537 / 1561 encoder compute operations to GPU.
`None` entries are non-compute program operations. Runtime dispatch flags and
decoder prediction counters separately prove that inference executed Core ML.
These are compute-plan placement reports, not an Instruments hardware trace or
an ANE-only claim for preprocessing, classification and decoding.

### Exercised support matrix

| Mode or configuration | macOS | Physical iPhone | Actual encoder / decoder |
|---|---|---|---|
| Short offline English/French, `auto`/`en`/`other` | API + CLI | API + CLI | Fixed Core ML / native for short clips |
| Long windowed English/French | CLI, 300 s / 51 s | CLI, 300 s / 51 s | Fixed Core ML windows / measured decoder policy |
| True unwindowed input beyond ANE capacity | Static FP32 GPU, 60 s / 51 s | Flexible FP32 CPU, 60 s / 51 s | No tiling; native encoder frame grid |
| Precomputed mel and full-input callbacks | All routes; precise callbacks GPU/CPU | All routes; precise callbacks CPU | Precomputed/batch tokens match; callbacks match same-context native |
| Push PCM, 127- and 4,096-sample feeds | FP32 GPU/CPU, all routes | FP32 CPU, all routes | Core ML encoder / native stateful decoder |
| Large streaming context and API unwindowed context | 20 s context | 20 s context | Precise Core ML retains caller context and padding |
| Engine reuse, route reset, silence prefix, cancel/finalize | Passed | Passed | No stale route or duplicate final callback |
| Four compute policies, forced decoder stages | CLI | CLI | Observed per-stage labels; CPU fallback inside Core ML where documented |
| F16/Q8 GGUF and six-bit fixed sidecars | CLI; six-bit short + long | CLI; six-bit short + long | Same route metadata and decoder contract |
| Core ML-only native-weight omission | Direct ownership + full API/stream/unwindowed suite | Same full suite + direct ownership | Native encoder blocks absent; actual Core ML gate/encoder |
| Missing/exchanged gate or adapted sidecar; missing decoder | Negative CLI tests | Shared runtime, positive deployment tests | Native fallback or `--require-coreml` error; optional decoder remains native |
| Owned model with missing flexible sidecar | Explicit no-window rejection | Explicit no-window rejection | Fails safely with native weights absent |
| One through eight mel frames | Compiled tensors + native/CoreML CLI parity | Native/CoreML CLI parity | FP32 Core ML CPU, all three routes; empty PCM retains existing API error |

This matrix records exercised combinations, not every Cartesian combination of
options. The file-based Engine methods share the same PCM implementations.
The optional fixed sidecars accelerate default/windowed deployment; continuous
force0 with automatic window/context, true
unwindowed operation and compute policies excluding ANE require the additional
FP32 sidecars. CLI `--window-sec` controls offline operation; the CLI warns that
stream/profile modes ignore that offline option. Streaming Engine callers can
set `long_form_window_frames = -1` explicitly.

### Numerical and quality checks

The float32 PyTorch export graph agrees with the float32 reference at roughly
`1e-6` relative L2. This is a graph-equivalence check, not compiled FP16 accuracy.
The final masked/native-padding programs were checked on 110 actual compiled
predictions against the float32 reference: maximum relative L2 `1.12e-6` for CPU
and `1.23e-5` for static GPU, including 1–8 mel frames, aligned/nonaligned padding
boundaries, English 60 s and adapted French 51 s. Native runtime comparisons also
include its half-precision attention activation casts; measured boundary error
is below 0.004 with identical encoder row counts and stateful transcripts.

Actual ANE FP16 and six-bit predictions were compared against the fixed-mask float32 export reference
on a full 15-second input, an 800-millisecond prefix, and French speech. Across
nine route/input combinations each, maximum relative L2 was 0.0897 (FP16) and
0.1043 (six-bit). The largest FP16 difference occurred in the **intermediate
12-layer gate features** on English with a silence tail; the final English and
adapted encoder outputs had smaller errors. The six-bit maximum occurred in the
adapted 800-millisecond prefix. These are accumulated low-precision/compiler
errors, not a claim of bitwise equivalence. The FP32 compiled checks separate
these effects from an incorrect export graph; downstream transcription quality
is evaluated separately below.

Gate probabilities from the compiled ANE prefix differed by at most 0.00256
(FP16) / 0.00889 (six-bit). All six measured compiled gate decisions agreed with
the float32 reference: English probabilities 0.047–0.212 and French 0.938–0.940,
versus threshold 0.8582. No measured case was near that threshold. A recording
near the threshold could change route under quantization, so validate the gate
on the intended deployment corpus. The first-20-second gate contract also means
an unrepresentative or silent prefix can select a poor route; a known `en` or
`other` option avoids that model-level ambiguity.

The references here cover English/French routes and representative silence,
short prefixes, Hindi stability and repeated long speech. They establish backend
parity and route coverage, not accuracy across a 25-language evaluation corpus.
Forcing a wrong language route remains a model-level quality risk. In an explicit
one-second push configuration, forcing `other` for English JFK omits its final
phrase in both native and precise Core ML; `en`/`auto` preserve it. Exact native
comparisons cover that case, automatic/two-second and explicit one-second chunks, short/long French,
and reused Engines. Core ML does not change the caller's chunk size to conceal it.

The retained results include normalized WER/CER, reference words, transcript,
actual stage dispatch, repeat timings, and process memory. See the
[reproduction harness](../examples/ios-coreml-bench/README.md) for commands.

Explicit one-second push on the 51-second French fixture produces 42.86% WER in
both native and precise Core ML with identical text. Automatic/two-second chunks
reduce that to 7.14%, consisting of French homophone spelling substitutions,
with all twelve utterances present. The implementation changes the model's
automatic default and preserves caller-requested one-second context semantics.


### Measured offline latency

Median processing milliseconds, excluding model load and warmup. Short cases use
five timed runs after two warmups; long cases use three after one. All use
language `auto`. Long primary rows request **60 s windows and 25% overlap** on
every backend: native resolves 750/94 encoder frames; fixed Core ML is bounded
to 187/46 (95-frame advance), so the attention contexts differ as documented.
The decoder stays native for explicit window/context requests.

| Device / audio | CPU ms | Metal ms | Core ML ms | Metal / Core ML | Core ML WER |
|---|---:|---:|---:|---:|---:|
| Mac / English 11 s | 450.5 | 92.9 | 34.8 | 2.67x | 0.00% |
| Mac / French 3.26 s | 136.6 | 44.4 | 30.1 | 1.47x | 0.00% |
| Mac / English 300 s, explicit 60 s/.25 | 17003.9 | 2337.6 | 730.4 | 3.20x | 0.00% |
| Mac / French 51.10 s, explicit 60 s/.25 | 2331.4 | 341.2 | 154.3 | 2.21x | 9.52% |
| iPhone / English 11 s | 1421.8 | 182.7 | 48.7 | 3.75x | 0.00% |
| iPhone / French 3.26 s | 457.2 | 81.8 | 38.9 | 2.11x | 0.00% |
| iPhone / English 300 s, explicit 60 s/.25 | 66339.8 | 6937.7 | 996.9 | 6.96x | 0.00% |
| iPhone / French 51.10 s, explicit 60 s/.25 | 8943.1 | 1018.2 | 232.0 | 4.39x | 9.52% |

English and short French have 0 WER on all three backends. Long French native
WER is 13.10%; explicit Core ML is 9.52%. The strict French differences here are
`Ses`/`Ces` homophone spellings, not missing utterances. WER/CER and full case
metadata are retained in [the compact results](transcribepsy-eu-coreml-results.json).

### Automatic windows and decoder selection

Automatic **language** and automatic **window/context** are separate settings.
These rows leave both window/context at 0 and exercise the default independent
encoder policy. They are reported without a speedup ratio against unmatched
native explicit-window baselines. The converted GGUF has no optional long-form
hints and a generic description: native auto uses a 3750-frame ceiling with 256
context (the 300 s example balances to 2388-frame windows), rather than 750/94.

| Device / audio | Core ML ms | RTF | Actual decoder | WER |
|---|---:|---:|---|---:|
| Mac / English 300 s | 386.4 | 0.0013 | coreml-cpu | 0.00% |
| Mac / French 51.10 s | 115.5 | 0.0023 | coreml-cpu | 0.00% |
| iPhone / English 300 s | 557.7 | 0.0019 | coreml-ane | 0.00% |
| iPhone / French 51.10 s | 207.0 | 0.0041 | CPU | 0.00% |

Below the measured decoder crossover, or without the optional decoder, the same independent encoder segments use
the native host decoder. Encoder segmentation does not change at the crossover
or when the optional decoder is absent. Default French 51 s then has 0 WER on
both devices, including encoder-only/owned deployment. Explicit force0 retains
continuous decoding and uses precise Core ML over the normal native attention
context for oversized auto-window input; its French 13.10% WER matches native
homophone spelling and retains all twelve utterances.

### True unwindowed processing

| Device / audio | Core ML ms | Actual encoder | WER |
|---|---:|---|---:|
| Mac / English 60 s | 361.1 | coreml-cpu-gpu | 0.00% |
| Mac / French 51.10 s | 365.5 | coreml-cpu-gpu | 13.10% |
| iPhone / English 60 s | 1410.9 | coreml-cpu | 0.00% |
| iPhone / French 51.10 s | 1199.2 | coreml-cpu | 13.10% |

### First and warm streaming sessions

The 48-case streaming matrix uses separate processes: it excludes initial Engine
construction but includes lazy precise-program loading. It is **not** a
steady-state measurement. The focused 51.10 s French experiment below reuses one
Engine per backend, grouping callbacks then push sessions; first is followed by
three timed warm sessions with identical text. Phone starts/ends nominal.
All streaming decoders are native; Core ML encoders are FP32 GPU on Mac and
FP32 CPU on iPhone. Callback/push WER is 13.10%/7.14%, matching native.

| Device / mode | Metal first ms | Metal warm ms | Core ML first ms | Core ML warm ms | Core ML warm RTF |
|---|---:|---:|---:|---:|---:|
| Mac / callback | 443.7 | 376.4 | 5003.9 | 446.6 | 0.0087 |
| Mac / push | 1686.5 | 1703.5 | 3558.3 | 1714.8 | 0.0336 |
| iPhone / callback | 828.0 | 714.7 | 1724.7 | 1269.9 | 0.0249 |
| iPhone / push | 3160.4 | 3159.1 | 4525.5 | 4759.5 | 0.0931 |

Mac warm push is similar to Metal; callback is modestly slower. iPhone precise
Core ML CPU is slower than Metal in both modes but remains comfortably realtime.
The fixed FP16 ANE graph is not used for stateful decoding. First-use/lazy load
can dominate short sessions; initialization and cache state must be included in
a deployment latency budget. The first Mac callback above included a new precise
GPU program specialization, unlike the cached separate-process matrix.

### Startup, memory, and validation receipts

Fresh iPhone installs needed 26–43 s to initialize/specialize the three fixed
encoder programs, loaded eagerly under the default ANE policy. Shipping compiled
`.mlmodelc` avoids package conversion, but does not eliminate target/OS-specific
specialization. Repeat-process fixed initialization is usually about 0.5–0.7 s.
The final masked FP32 CPU programs first loaded in about 3.7–4.0 s per main route
and 1.6 s for the gate; cached loads were about 0.13–0.16 s. These are observed
artifact/runtime costs, not a guaranteed startup bound. Caches were warmed
separately from timed offline inference; app updates/model changes can invalidate
that specialization.

Owned short ANE deployment measured about 144–150 MiB peak RSS on Mac/iPhone,
versus roughly 640–666 MiB with the unused native encoder copy retained. Physical
iPhone owned mixed API/reset/no-window suites peaked around 673–724 MiB in kernel
physical footprint. The focused mixed Metal/Core ML streaming process peaked
at 2646 MiB RSS/1132MiB physical footprint; these are different OS measures and
include both sequentially exercised backends, not one encoder allocation.
The lazy runtime keeps only one precise main route/capacity handle per Engine.

Validation includes14 distinct relevant CTests (12 initial plus 4 changed-area
passes with 2 overlap), 132 offline/config cases, 48 first-session stream cases,
98 tiny-input positives and2 owned missing-flexible rejections,16 default/explicit
chunk-quality cases,18 explicit-context/load-policy CLI cases,22 automatic/missing
decoder quality cases,32 focused first/warm streaming sessions,110 compiled FP32
predictions and18 compiled FP16/six-bit predictions. Fair thermal receipts are
retained separately: final stream samples are nominal; one unchanged native CPU
French-push row uses its two stable nominal samples (0.2%spread), all other
stream rows use three. No model/build/cache data belongs in the git change.


### Quantization coverage and refreshed six-bit results

Fixed encoder/gate exports support grouped-channel palettization. Full-EU
validation covers FP16 (`--nbits 0`) and six-bit eligible weights (`--nbits 6`),
with FP16 computation and some FP16 constants retained when grouping is not
applicable. Exportable widths 2, 3, 4, and 8 have not been evaluated for this
model on these devices. Precise FP32 programs deliberately reject palettization;
stateful and continuous precise paths therefore use unquantized FP32 programs.

Compiled program sizes, measured with `du -sk`:

| Program | FP16 MiB | Six-bit MiB |
|---|---:|---:|
| English encoder | 449.56 | 178.76 |
| Adapted encoder | 450.52 | 179.18 |
| Gate prefix | 229.27 | 92.24 |

Together the three encoder programs shrink about 60.1%. The tested `models6`
**TDT decoder is entirely FP16**, including its joint output projection, and is
identical to the 17.87 MiB FP16 decoder in `models`. Decoder palettization is
exportable (`--nbits`, plus separate `--nbits-out`) but was not validated in this
full-EU evaluation. A Q8_0 GGUF paired with FP16 Core ML sidecars is a hybrid:
it does not make the Core ML encoder Q8.

The four long six-bit rows were refreshed on 2026-11-11 against the final explicit
window/context dispatch guard, with three timed runs after one warmup. Both
phone cases finished at nominal thermal state. They use explicit 60 s/.25
windows, actual Core ML encoder/gate predictions, and **native decoding**.
Earlier long six-bit receipts that reported independent Core ML decoding under
those explicit settings are historical and superseded.

| Device / audio | Six-bit processing ms | WER | CER |
|---|---:|---:|---:|
| Mac / English 11 s | 35.724 | 0% | 0% |
| Mac / French 3.26 s | 30.781 | 0% | 0% |
| iPhone / English 11 s | 54.138 | 0% | 0% |
| iPhone / French 3.26 s | 40.308 | 0% | 0% |
| Mac / English 300 s | 752.684 | 1.136% | 1.286% |
| Mac / French 51.10 s | 156.110 | 9.524% | 1.094% |
| iPhone / English 300 s | 1058.619 | 1.136% | 1.286% |
| iPhone / French 51.10 s | 238.100 | 9.524% | 1.094% |

Short measurements retain the previously validated five-run/two-warmup receipts.
Both long English outputs omit one repeated five-word “and so my fellow
Americans” prefix (5/440 reference words); the matched FP16 explicit-window
results have 0 WER. French differences are eight `Ses`/`Ces` spelling substitutions
among 84 words, matching the FP16 explicit-window WER. The refreshed timings show no inference-speed advantage over FP16. Six-bit compression
has a small observed long-English quality cost; it is not transcript-identical
to FP16 on every case. Current raw receipts are in
`quant6-refresh-20261111/` under the retained artifact directory, and the four
records replace their superseded entries in the compact and durable final summaries.
