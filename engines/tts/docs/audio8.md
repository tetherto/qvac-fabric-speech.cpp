# tts engine: Audio8

Part of the [tts engine documentation](../README.md).

## Audio8

[Audio8](https://github.com/Audio8-AI/Audio8_TTS) is a DualAR zero-shot TTS
model: a 24-layer Qwen2.5-0.6B-shaped **slow AR** emits one semantic token per
codec frame, a 4-layer **fast AR** expands each of those into the frame's
remaining 9 codebooks, and a DAC-style residual-vector-quantised **codec**
turns the 10-codebook frames into 44.1 kHz audio at 2048 samples (21.5 Hz) per
frame.  Cloning needs no speaker encoder: the codec *encoder* turns a reference
wav into codes, and those codes plus the reference transcript are prepended to
the prompt.

**Status — CPU, Metal, OpenCL, and desktop Vulkan, validated against the
reference.**  Text-to-speech and voice cloning both run in-process on macOS and
iOS Metal, on Android/Adreno OpenCL, and on Linux and Windows Vulkan.  Pass
`--n-gpu-layers 99` to offload every stage; omit it for CPU.  On Metal that is
**4.2x** CPU at q4_0 and **3.2x** at q8_0 on an iPhone 17, measured on device
with both arms in one launch.  At F32 the GPU reproduces the CPU code trajectory
exactly, frame for frame, which is the strongest available statement that the
graphs compute the same function; quantised tiers fork their trajectories under
either backend and are judged by WER instead.

On OpenCL the two halves of the model behave differently and are reported
separately, because a single overall number is not reproducible on that device.
**Codec synthesis is 3.5-4.7x faster than CPU** and is stable run to run.  The
autoregressive loop is not GPU-bound at all: it is bound by the host issuing
roughly one dispatch every 24 us, so its wall time follows which CPU core the
issuing thread is scheduled onto.  Pinning that thread to the big cluster took a
q4_0 utterance from 30952/22011 ms across two runs to 17197/17201 ms — a 41%
spread collapsing to 0.02% — while codec synthesis did not move.

| backend | device | tier | codec synth | overall, same cores | overall, unpinned |
|---|---|---|---:|---:|---:|
| Metal | iPhone 17 | q8_0 | — | 3.2x | — |
| Metal | iPhone 17 | q4_0 | — | 4.2x | — |
| OpenCL | Adreno 740 | q8_0 | 4.7x | 1.9x | 1.1x |
| OpenCL | Adreno 740 | q4_0 | 5.1x | 2.3x | 1.3x |

Read the two overall columns as a range, not as one number with a caveat.  "Same
cores" pins both arms to the big cluster, which makes the GPU arm reproducible
(2.9% spread) but costs the CPU arm, whose five threads then contend for four
cores.  "Unpinned" is the median of every interleaved, thermally gated run, and
is what an application that does not set affinity should expect; its GPU arm
ranged 126-245 ms/frame at q8_0 for the same binary.  Both were measured on a
cold device with a 50 °C gate between arms.  Real-time factors are quoted
elsewhere in this file as wall-clock seconds per second of audio, so **lower is
faster** and a value above 1 means slower than real time.

### Convert

Three GGUFs, because the halves have different lifetimes: the LM and the codec
decoder are needed for every synthesis, the codec encoder only to enrol a voice
from a wav.  Both codec halves carry the codebooks, so each file stands alone.
`scripts/download-audio8-checkpoint.sh` fetches the upstream checkpoint from
[Audio8/Audio8-TTS-Preview-0.6b](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b)
(Hugging Face CLI: `pip install -U huggingface_hub`).

```bash
scripts/download-audio8-checkpoint.sh --dir models/Audio8-TTS-Preview-0.6b

python3 scripts/convert-audio8-lm-to-gguf.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --outfile models/audio8-lm-q8_0.gguf --dtype q8_0

for part in encoder decoder; do
  python3 scripts/convert-audio8-codec-to-gguf.py \
      --model-dir models/Audio8-TTS-Preview-0.6b --part $part \
      --outfile models/audio8-codec-$part-q8_0.gguf --dtype q8_0
done
```

| `--dtype` | LM | codec decoder | codec encoder |
|---|---:|---:|---:|
| `f32` | 2.3 GB | 499 MB | 793 MB |
| `f16` | 1.2 GB | 251 MB | 398 MB |
| `q8_0` | 801 MB | 201 MB | 252 MB |
| `q4_0` | 602 MB | — | — |

**The f16 LM is free.**  The LM checkpoint ships in bfloat16, whose 8 mantissa
bits fit inside f16's 10, so half precision stores it without loss — f32 and
f16 builds agree to 3e-8, which is below the f32 rounding of the conversion
itself.  Prefer f16 over f32 unless you are debugging.  The codec ships in f32
and its f16 build is genuinely lossy (worst tensor deviation 1.6e-3).

`--dtype` sets a ceiling, not a blanket cast.  Each tensor is classified by
what it is used for, in `role_of()` in either converter: body matmuls take the
block format, bulk weights that must stay element-addressable (embedding
tables, convolution kernels) stop at f16, and tensors that decide a greedy
choice stay f32 in every build — the RoPE tables, the sampling heads, the
codebooks (the encoder picks codes by nearest neighbour, so rounding them moves
the argmin) and every 1-D parameter, the snake alphas among them.  A q4_0 codec
is not offered: over half the decoder is convolution kernels that block formats
cannot address, so the tier would buy little and cost audible quality.

Two things the converters do that are worth knowing before reading them:

- **RoPE tables are baked, not recomputed.**  The reference builds them in
  bfloat16 even under float32 inference, and that ~2e-3 rounding is not
  cosmetic — recomputing in float32 changes the greedy fast-AR codebook choices
  on the very first frame and the trajectories separate from there.
  `--rope-precision f32` is available to reproduce that measurement.  The
  tables are emitted as separate `_cos` and `_sin` planes, the shape
  `apply_rope_in_graph` consumes.
- **Q and K rows are reordered.**  The reference rotates each head's adjacent
  pair `(2j, 2j+1)`; ggml and the rest of this tree rotate `(j, head_dim/2+j)`.
  Reordering the projection rows at conversion time avoids a strided gather in
  every attention block, and attention cannot tell the difference because the
  same permutation lands on Q and K and their dot product does not depend on
  the order of the terms.  `verify-audio8-conversion.py` checks this by
  comparing attention scores, not tensors.
- **The text head is replaced by a semantic head.**  The head is tied to the
  155776-row input embedding, but the sampler masks every logit outside
  `[semantic_begin, semantic_end]` plus EOS, so the converter emits just those
  4097 rows as `lm/sem_head`.  Masked-out logits contribute nothing to the
  top-k/top-p renormalisation, so this is exact, and it turns a 155776-row
  projection per step into a 4097-row one.

### Verify

`verify-audio8-conversion.py` reloads the checkpoint through its own remote
code and compares every converted tensor against it, at a tolerance derived
from what the tensor's storage can actually represent — exact for f32, one
reconstruction step for the block formats.  It recomputes the RoPE tables
independently rather than trusting the converter's copy.

Each GGUF is also checked for completeness on its own, against the half of the
checkpoint the converter's own split says it carries, so a tensor written to
the wrong half fails here rather than at load time.  It takes any subset of the
three files, but at least one.

```bash
python3 scripts/verify-audio8-conversion.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --lm-gguf models/audio8-lm-q8_0.gguf \
    --codec-encoder-gguf models/audio8-codec-encoder-q8_0.gguf \
    --codec-decoder-gguf models/audio8-codec-decoder-q8_0.gguf
```

### Reference fixtures

Three dumpers write the `.npy` fixtures the C++ stages will be checked against.
They capture stage boundaries, not just endpoints, so a failing stage can be
located without bisecting the network.

```bash
python3 scripts/dump-audio8-tokenizer-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b --out-dir artifacts/audio8-ref

# reference-free generation: prompt, embeddings, per-step hidden states,
# semantic logits before and after filtering, fast-AR logits, emitted codes.
# --dump-wav also decodes those codes, which test-audio8-engine compares against
python3 scripts/dump-audio8-lm-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --out-dir artifacts/audio8-ref --max-new-tokens 32 --dump-wav

# codec both ways: decode the codes above, encode a reference wav.  This is the
# recording the cloning fixtures below are enrolled from, so the tests expect it
python3 scripts/dump-audio8-codec-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --codes artifacts/audio8-ref/codes.npy \
    --audio test/reference-audio/audio8-reference-en.wav \
    --out-dir artifacts/audio8-ref

# the cloning path end to end: reference codes + transcript -> prompt -> codes
python3 scripts/dump-audio8-lm-reference.py \
    --model-dir models/Audio8-TTS-Preview-0.6b \
    --text "Tear in eye, your dress you'll tear." \
    --reference-codes artifacts/audio8-ref/codec_enc_codes.npy \
    --reference-text "What these pictures? An elephant with uh so many legs." \
    --out-dir artifacts/audio8-ref-clone --max-new-tokens 24 --dump-wav
```

The dumps are taken with `do_sample=False`.  The shipped sampler draws Gumbel
noise and applies repetition-aware resampling, neither of which reproduces
across runtimes; greedy decoding makes the trajectory comparable, and the
filtered score vectors are dumped alongside the tokens so everything feeding
the draw can be checked even where the draw itself cannot.  `--dump-wav` runs
the generated codes back through the codec so the engine test has an
end-to-end waveform to compare against, not just codes.

CTest looks for the results in fixed places, and a fixture it cannot find turns
its test into a "Not Run (Disabled)" rather than a failure, so the recipe above
writes where the tests read: the dumps into `artifacts/audio8-ref` and
`artifacts/audio8-ref-clone`, the GGUFs into `models/`.  `cmake` prints a
`disabled (missing fixture(s): ...)` line at configure time for whatever is
still absent, which is the quickest way to see that a run is thinner than it
looks.

### Run

```bash
# text only: the LM and the codec's synthesis half
audio8-cli --lm models/audio8-lm-q8_0.gguf \
           --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
           --text "Hello from a fully on-device C++ pipeline." \
           --out out.wav --threads 8 --n-gpu-layers 99

# on the GPU: same command, everything on the device
audio8-cli --lm models/audio8-lm-q8_0.gguf \
           --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
           --text "Hello from a fully on-device C++ pipeline." \
           --out out.wav --n-gpu-layers 99

# cloning: add the analysis half, a reference wav and what it says
audio8-cli --lm models/audio8-lm-q8_0.gguf \
           --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
           --codec-encoder models/audio8-codec-encoder-q8_0.gguf \
           --ref-audio voice.wav --ref-text "What the recording says." \
           --text "Now say this instead." --out out.wav --threads 8
```

`--greedy` reproduces the fixtures; otherwise `--seed`, `--temperature`,
`--top-k` and `--top-p` drive the sampler.  `--max-frames` caps the output at
2048 samples each (~46 ms), `--output-sample-rate` resamples away from the
codec's native 44.1 kHz, and `--backends-dir` points a `GGML_BACKEND_DL` build
at its backend libraries.  The reference wav is resampled and downmixed on the
way in, so any format `dr_wav` reads will do.

`--n-gpu-layers` is all-or-nothing by design: it selects the backend the whole
model runs on rather than splitting layers across two, so any count above zero
puts everything on the device and zero keeps it on CPU.  A GPU run copies the
weights into device memory rather than mapping them, so plan for the GGUF size
again on top of the file itself.  `--verbose` prints the per-stage times and the
block width the codec settled on, and `--dump-codes` writes the discrete code
trajectory as one comma-separated line per frame, which two runs can be diffed
on to see whether a backend or a quantisation tier changed what was generated.

Codec synthesis runs in blocks whose width is chosen from a memory budget rather
than fixed: the widest block whose scratch fits a quarter of what the backend
reports free, capped at 384 MiB.  A backend that reports no total at all is
saying it cannot tell rather than that it has nothing, and gets the cap.  Blocks
exist only to bound that scratch — each one re-runs the causal context of the one
before it — so a wider block is strictly less work, and the width never changes
the samples.  Set `codec_model::synthesis_block_frames` to pin a width or
`synthesis_scratch_budget` to pin the budget.

The same thing through the library, from `<tts-cpp/audio8/engine.h>`:

```cpp
tts_cpp::audio8::EngineOptions opts;
opts.lm_gguf_path            = "models/audio8-lm-q8_0.gguf";
opts.codec_decoder_gguf_path = "models/audio8-codec-decoder-q8_0.gguf";
opts.codec_encoder_gguf_path = "models/audio8-codec-encoder-q8_0.gguf";  // cloning only

tts_cpp::audio8::Engine engine(opts);
auto plain = engine.synthesize("Hello.");

auto voice = tts_cpp::audio8::load_voice_prompt("voice.wav",
                                                "What the recording says.");
auto cloned = engine.synthesize("Now say this instead.", voice);
```

`VoicePrompt` is mono float32 plus a transcript, so a caller that already has
samples can fill it directly; `load_voice_prompt` is there for the common case
of a file on disk.

The engine holds the models resident across calls and caches the codes for the
most recent voice prompt, so re-using a reference skips the codec encoder.
`cancel()` stops an in-flight `synthesize()` at the next language model step or
codec block, whichever comes first; the call throws rather than returning a
partial waveform.

The three GGUFs have to come from one checkpoint, and the engine checks that
they do — codebook counts and widths, the codec's two halves against each other
and both against the language model — from their headers, before it reads a
byte of weights.

### Core ML codec sidecar

`TTS_CPP_COREML=ON` is Apple-only. It enables an optional Core ML sidecar for
the codec's **synthesis stack** -- the two upsampling stages and the DAC
decoder, everything from the windowed post transformer's output to the
waveform. That stack is the single largest stage of a CPU synthesis (5.4 s of
a 14 s run for 4.5 s of audio on an M2) and about a quarter of a Metal one,
and it is the part of Audio8 with a fixed, convolutional shape; the
autoregressive language model, the quantizer banks and the post transformer
stay on ggml. Export the sidecar from the decoder GGUF:

```bash
python3.11 -m venv .venv-coreml
. .venv-coreml/bin/activate
python -m pip install -r engines/parakeet/scripts/requirements-coreml.txt
python engines/tts/scripts/export-audio8-codec-coreml.py \
  --gguf models/audio8-codec-decoder-q8_0.gguf --compile-dir models
```

The compiled sidecar must sit next to the decoder GGUF as
`<basename-minus-quant>.mlmodelc`: `audio8-codec-decoder-q8_0.gguf`,
`-f16.gguf` and `-f32.gguf` all resolve to `audio8-codec-decoder.mlmodelc`,
because every tier stores the synthesis stack's kernels at f16 or better and
the export is the same whichever file it was taken from. The exporter
rebuilds the stack in PyTorch from the GGUF tensors, so it needs neither the
checkpoint nor its remote code; `--parity-dir artifacts/audio8-ref` checks
that rebuild against the reference dumps first (measured: waveform cosine
1.0000000, max error 1e-6). With the sidecar present `audio8-cli --verbose`
reports `codec synthesis on the Core ML sidecar` at load and the compute
label (`coreml-all`) in the timing breakdown.

The export is fixed-shape, `--window` post frames (default 64, 2.97 s,
131072 samples) in and the matching samples out. The engine walks an
utterance in windows of exactly that width: every convolution in the stack is
causal, so a window's output is exact from the first frame whose receptive
field lies inside it, and the engine drops each window's leading causal
context (10 frames for this checkpoint, computed by the same walk the ggml
block path uses) just as the ggml blocks drop theirs. An utterance shorter
than one window is zero-padded on the right, which changes nothing before the
padding because nothing in the stack looks forward. `test-audio8-coreml-windows`
locks that plan without a model; `test-audio8-codec-coreml-parity` gates the
sidecar against the ggml synthesis at cosine 0.999 on a short (padded) and a
long (stitched) utterance, and checks that a cancel between windows stops the
pass; both run on the TTS CI macOS lane, which converts the decoder from the
upstream checkpoint and exports the sidecar itself.

Two things the exporter does that are worth knowing before reading it. Every
transposed convolution is emitted in its exact phase form -- a causal `Conv1d`
over the input followed by a depth-to-space shuffle, which is also how
`codec_ops.cpp` computes it -- rather than as `ConvTranspose1d`, whose native
Neural Engine kernel miscomputes at stride 4 (the ACE-Step VAE export hit
this first). And the Snake activation squares its sine as a product,
`sin * sin`, never `sin ** 2`: the `pow` the latter lowers to is miscomputed
on the Neural Engine when its result feeds a convolution (measured on an M2,
macOS 15.7: the first DAC stage came out at cosine 0.39 against the CPU, with
the Snake alone and the convolution alone both exact), while the product is
exact on every compute unit.

Measured on an Apple M2 (macOS 15.7, f32 decoder GGUF, ggml Metal as the
reference, `bench-audio8-codec-coreml`, median of 3), synthesis stage only,
parity cosine 0.99999 in every cell:

| window | placement | 10 s (216 frames) | 24 s (517 frames) |
|---:|---|---:|---:|
| 64 | `coreml-all` (default) | 1518 -> 1198 ms, **1.27x** | 3683 -> 3023 ms, **1.22x** |
| 64 | `cpu_and_gpu` | 1466 -> 976 ms, **1.50x** | 3628 -> 2500 ms, **1.45x** |
| 32 | `coreml-all` | 1.27x | 1.22x |
| 32 | `cpu_and_gpu` | 1.25x | 1.28x |
| 128 | `coreml-all` | 0.51x | 0.49x |
| 128 | `cpu_and_gpu` | 1.53x | 1.51x |

Two placement facts sit behind that table. On this OS the Neural Engine has
no `sin` kernel, so every Snake activation leaves it: the compute plan puts
the first twelve sines on the CPU and, from the second DAC stage on, hands
the whole tail of the graph (17 sines, 18 convolutions) to the GPU, and the
device handoffs eat most of what the Neural Engine saves on the convolutions
it does keep. `AUDIO8_COREML_COMPUTE_UNITS=cpu_and_gpu` is therefore the
faster choice on a macOS 15 host; the default stays all-units because it is
correct everywhere, because the ACE-Step sidecar measured all-units best on an
M5 running macOS 26, and because the TTS CI macOS lane (macOS 26) reports the
default's speedup in its job summary for the current OS. And the 128-frame
window, which only buys a few percent on the GPU, collapses to half of Metal's
speed under mixed placement, so 64 is the default: the widest window that is
robust under either placement, at a causal-context overhead of 10 in 64 frames.
The very first load of a fresh export pays a one-time on-device compilation
(tens of seconds), which the OS caches for later loads.

Those are steady-state numbers from a resident engine. A one-shot
`audio8-cli` run also pays the sidecar's first-prediction warm-up and, for a
short utterance, the fixed window: a 4.3 s synthesis (92 frames, two 64-frame
windows for what ggml does in one block) came out at 747 ms on the sidecar
against 654 ms on Metal under all-units placement, and at 560 ms with
`cpu_and_gpu`; at 23.8 s (512 frames) the same one-shot run measured 3444 ms
on Metal, 3241 ms on the sidecar with all units and 2408 ms with
`cpu_and_gpu`. The engine keeps the sidecar resident across `synthesize()`
calls, so a host that speaks more than once pays the warm-up once.

Set `AUDIO8_COREML_DISABLE=1` to force the ggml synthesis, including for
parity or benchmarking. `AUDIO8_COREML_STRICT=1` turns the silent ggml
fallback into a synthesis failure, so a test cannot measure ggml and attribute
it to Core ML -- the parity test and benchmark set it for their Core ML legs.
`AUDIO8_COREML_COMPUTE_UNITS=cpu_only|cpu_and_gpu|cpu_and_ane` overrides the
default all-units placement for comparisons. `bench-audio8-codec-coreml`
times the ggml GPU synthesis against the sidecar on deterministic 10 s and
24 s code sequences (median of 3 after a warm-up that absorbs the one-time
on-device compilation of a fresh export) and prints a markdown table; it
fails below the parity gate, so its numbers are correctness-checked, and the
TTS CI macOS lane appends its table to the job summary.

Two reports tell a host where the codec ran. `Engine::codec_on_coreml()` is
the load status: true when a sidecar next to the decoder GGUF initialised (the
language model and the post transformer still run on `backend_name()`).
`SynthesisResult::codec_synthesis_backend` is per call: `"ggml"`, or the
sidecar's compute label (`coreml-all`, `coreml-gpu`, ...) when the synthesis
stack actually ran there. The two differ whenever a loaded sidecar cannot
serve a call -- a window that cannot carry the causal context, or a Core ML
prediction failure -- and the engine falls back to the ggml blocks. Because
that fallback stays possible, the memory-fit projection (`audio8-fit-params`)
prices the ggml synthesis arena whether or not a sidecar is present; with a
working sidecar it over-reports by that arena rather than under-reporting the
fallback. `test-audio8-codec-coreml-parity` covers all of it: an absent
sidecar under `AUDIO8_COREML_STRICT`, a sidecar directory that is not a model
(load status false, ggml synthesis), and a sidecar exported at a window that
cannot carry the context (`--window 8`; loaded, falls back bit-exactly to the
ggml decode, fails under `AUDIO8_COREML_STRICT`), each also through the public
`Engine` with a synthetic language model.

### Engine notes

Roughly a second of audio per second of CPU on eight cores of a desktop x86-64,
q8_0 weights: 16 s of speech in 19 s wall, 5.5 s of cloned speech in 10 s
including enrolment. For measured multi-machine numbers, see
[the Audio8 benchmark in Performance](../README.md#audio8-multi-machine-benchmark-2026-09).

**The codec is chunked, in both directions.**  Its convolution stacks, not the
LM, are what made memory grow with utterance length — a 24 s decode used to
need a 1.5 GB arena.  Each direction is now split into a whole-sequence graph
and a block graph that walks the utterance, and each block is re-fed the exact
receptive field its stack needs (`span_through_*` in `codec_ops.cpp` walks the
strides and dilations to compute it) and drops the samples that context
produced.  The result is bit-identical to running in one pass whatever the
block size, which `test-audio8-codec` asserts by re-running both directions at
a deliberately tiny block and comparing.  Decode now holds ~130 MB flat, encode
~140 MB for a 10 s reference.  End to end that is 1.25 GB resident for
text-only and 1.67 GB for cloning, of which 1.25 GB is the q8_0 weights
themselves.

The one part that still grows with input length is the encoder's analysis
transformer: `ggml_soft_max_ext` materialises the full score matrix, so a
reference much beyond 30 s gets expensive.  References of a few seconds are
what the model expects anyway.

**Quantised weights take a different route per backend.**  `GGML_PREC_F32`
asks a matmul to accumulate in f32, which a block-quantised weight has already
spent: on CUDA the default route is the integer dot product every CPU build of
this tier also takes, so the marker there buys nothing and costs a
dequantise-to-f32 round trip through cuBLAS: 48% of GPU kernel time, and a
fifth of the decode's wall, the loop being host-bound rather than GPU-bound.
The graphs therefore drop it on CUDA-resident quantised weights and keep it
everywhere else, because ggml-vulkan reduces quantised matmuls in f16 under
`PREC_DEFAULT`.  It reaches the codec too: its post-quantiser transformer is
56 of the q8_0 decoder's tensors.

Measured against the PyTorch fixtures at q8_0, both halves land at the CPU
q8_0 build's own accuracy, which is what this tier is judged at — the previous
CUDA route was *more* accurate than CPU, at twice the cost, because it never
quantised the activations.

| RMS deviation from the f32 reference | CPU q8_0 | CUDA before | CUDA after |
|---|--:|--:|--:|
| LM semantic logits | 0.106 | 0.077 | 0.100 |
| LM fast logits | 0.182 | 0.146 | 0.173 |
| codec latent | 2.20e-2 | 1.19e-2 | 2.18e-2 |
| codec waveform | 7.04e-4 | 3.16e-4 | 5.52e-4 |

f32 and f16 weights keep the marker on every backend, so the exact CPU/GPU
trajectory match at F32 is unaffected.  `test-audio8-quantised-precision` pins
the scoping per backend, over the LM's built graphs and the codec decoder's
resident weights.

**The fast head's graphs are built once and replayed.**  Every frame walks the
same positions with the same shapes, and each position's causal mask is all
zeros -- a fast position attends to the whole frame prefix -- so the mask
depends only on the position and is written when the graph is built.  A cached
graph must own its allocator (a shared arena moves under the other graphs the
first time a bigger one reserves) and must never reach the scheduler fallback,
which resets one shared arena per graph; a build that lands there drops what
was built and the per-call path serves the rest of the model's life.
`test-audio8-fast-cache-sched` pins both transitions.

**Sampling follows the reference's order, which is unusual.**  top-k and top-p
run on the raw logits and the temperature is applied to what survives, so
temperature does not affect which candidates are in the running.  Semantic
tokens additionally go through repetition-aware resampling: drawing a token
that already appears in a short trailing window triggers one re-draw at a
narrower nucleus and a higher temperature.  The window holds tokens the model
actually drew and nothing else, so an utterance opens with no history at all;
the opening token stays out of it, as it does in the reference.  `--greedy`
bypasses all of it.

### Test

The C++ stages are checked against the fixtures above, one CTest target per
stage boundary:

```bash
ctest -R audio8 --output-on-failure
```

| target | checks |
|---|---|
| `test-audio8-tokenizer` | BPE ids and the ChatML prompt, both cloning and not |
| `test-audio8-lm` | prompt embeddings, per-step hidden states, semantic and fast-AR logits, emitted codes |
| `test-audio8-codec` | encode and decode at every stage boundary, block-size independence, and that a cancel stops the block loop |
| `test-audio8-sampler` | filtered score vectors, which is the part of the draw that is reproducible |
| `test-audio8-ras` | the repetition-aware window: which draws enter it, eligibility, eviction, and the retry's nucleus |
| `test-audio8-engine` | both public paths end to end against the decoded waveforms, and the refusal of GGUFs that disagree |
| `test-audio8-timing` | that the per-stage times are disjoint and bounded by the total they are reported against |
| `test-audio8-cli` | the CLI's flags, `-ngl` and `--n-gpu-layers` among them, and the `--dump-codes` file format |
| `test-audio8-cli-verbose` | the same flags through the binary: `--verbose` reaching stderr and codes surviving a synthesis |
| `test-audio8-sampling-filter` | the top-k / top-p / temperature candidate filter on synthetic score vectors: k and p cutoffs, temperature ordering, degenerate cases |

`test-audio8-ras`, `test-audio8-cli`, and `test-audio8-sampling-filter` are the
three that run on a checkout with no models; every other target needs the
dumps.

On a build with a GPU backend compiled in, the `lm`, `codec` and `engine`
suites are registered a second time per backend as `test-audio8-<suite>-metal`,
`-vulkan` and `-opencl`. Each arm names its backend in `AUDIO8_TEST_GPU` and
asserts it before reading a number, so an arm that fell back to CPU, or that was
handed the other arm's GPU, fails rather than passing on someone else's result.

The engine test hands the cloning path a wav rather than pre-computed codes, so
it exercises the codec encoder the way a caller would.
