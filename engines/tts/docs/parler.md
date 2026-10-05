# tts engine: Parler-TTS

Part of the [tts engine documentation](../README.md).

## Parler-TTS

Description-conditioned TTS: the transcript (`--text`) is spoken in a voice
controlled by a natural-language description (`--description`).  Supports
`parler-tts/parler-tts-mini-v1`, `parler-tts/parler-tts-large-v1` and
`ai4bharat/indic-parler-tts` (21 languages incl. Hindi/Gujarati);
model differences are pure GGUF metadata (`parler.*` keys), no code
branching.  Pipeline: Flan-T5 encoder (description → cross-attention K/V,
precomputed once and cached per description) → delay-pattern decoder LM
(9 DAC codebooks, MusicGen-style stagger, HF-faithful EOS gating) → DAC
codec decode → 44.1 kHz mono PCM.  Validated backends are CPU, Metal,
Vulkan, and OpenCL (Adreno): the GPU path (F16 flash attention, DAC
upsampling as phase matmuls) is gated on that allowlist in
`src/parler/gguf.cpp`, and any other GPU backend is released at load and
replaced by CPU rather than run unvalidated.  The Hexagon NPU runs only on
explicit request (see [Snapdragon Hexagon NPU](#snapdragon-hexagon-npu)).  GPU loads and mmap-backed CPU
loads fuse the per-layer QKV projections and the nine LM heads into single
row-concatenated matmuls (byte-exact; the CPU allocate-and-stream fallback
stays unfused so the projections are never resident twice, and
`PARLER_NO_FUSED` disables fusion everywhere).  Every backend samples each
step over the top-k candidate set with reused scratch buffers, and host
backends read the step logits in place from the graph buffer instead of
downloading a copy.  The graph dispatch uses the shared `sched_dispatch`
dual path like the other engines.

On the CPU backend the decoder's self- and cross-attention also run through
ggml's flash-attention kernel, over an F32 KV cache, so they multiply and
accumulate in F32 like the manual softmax(QK^T)V path while no longer copying
the whole cache every step (`PARLER_NO_FA` restores the manual path), and the
FFN's exact-erf GELU is viewed as rows so every thread shares it.  The CPU DAC
runs each snake activation as one fused node (over reciprocals precomputed at
load from the F32 snake alphas every converted GGUF carries); on Apple
builds its convolutions and transposed convolutions run on Accelerate's sgemm
with the snakes on vForce (`PARLER_DAC_NO_ACCEL` keeps the ggml kernels), which
also shrinks the DAC compute buffer from 268 MiB to 99 MiB.  Accelerate tiles
by shape, so a windowed or streamed DAC decode matches a whole-sequence one
within float tolerance there rather than bit for bit, as on a tinyBLAS build.
See [Performance](performance.md#parler-tts-on-cpu-mac-studio-m3-ultra) for
the Mac Studio M3 Ultra numbers.

Indic-class checkpoints ship a second, SentencePiece-BPE **prompt**
tokenizer (90k vocab covering the Indic scripts, byte fallback) alongside
the Flan-T5 description tokenizer; the converter embeds both
(`parler.prompt_tokenizer.*` keys) and the engine routes prompts through
the BPE tokenizer when present.  mini/large GGUFs are unaffected (one
shared tokenizer, byte-identical output).  The language is auto-detected
from the prompt script; descriptions stay English (name a recommended
voice — e.g. Rohit for Hindi, Yash for Gujarati — for stable speakers).

```sh
# convert (single GGUF: T5 + decoder + DAC + tokenizer; ~3.4 GB f32)
python3 scripts/parler/convert-to-gguf.py \
    --model-id parler-tts/parler-tts-mini-v1 --dtype f32 \
    --out models/parler-mini-v1-f32.gguf

# synthesize (also autodetected by tts-cli via parler.arch + --description)
./build/parler-cli --model models/parler-mini-v1-f32.gguf \
    --text "Hey, how are you doing today?" \
    --description "A female speaker with a calm, clear voice, close up." \
    --out out.wav
```

The indic checkpoint is gated on HF (`gated: auto` — any account gets
instant access); pass a downloaded snapshot directory as `--model-id`
with `--reference-repo ai4bharat/indic-parler-tts` for provenance.

Instead of writing a full `--description`, the voice can be configured
through template flags (`build_description()` in
`<tts-cpp/parler/description.h>` — the same renderer the downstream addon
uses).  The rendered text follows the models' training-caption phrasing,
and every flag has a working default — with no flags at all the engine
uses the models' recommended fallback caption ("The speaker speaks
naturally. The recording is very high quality with no background noise."):

| flag | values (default first) | rendered as |
|---|---|---|
| `--voice` | free name, e.g. Rohit, Laura | sentence subject |
| `--emotion` | one of the 12 below | tone clause + "The intended style is …" |
| `--pitch` | unset, low, moderate, high | "with a low pitch" |
| `--pace` | unset, slow, moderate, fast | "speaks slowly" / "at a fast pace" |
| `--expressivity` | unset, monotone, slightly expressive, expressive | "in an expressive manner" |
| `--noise` | clear, noisy | "with no/noticeable background noise" |
| `--reverb` | close, distant | "distant-sounding" (close is implied) |
| `--quality` | very high, high, basic | "The recording is … quality" |

Emotions come from the shared vocabulary in [Voice conditioning](voice-conditioning.md#voice-conditioning-cross-engine)
— the 12 speaking styles in the indic training set, case-insensitive and
validated.  Each renders an in-distribution clause ("with an angry tone",
"delivering the news", "perfect for narration") plus the trailing style
anchor sentence the training captions used.  The indic card lists 10 officially emotion-tested
languages (Assamese, Bengali, Bodo, Dogri, Kannada, Malayalam, Marathi,
Sanskrit, Nepali, Tamil); elsewhere — including Hindi/Gujarati and the
mini/large English checkpoints — emotion conditioning exists but is
best-effort: validate by ear.  `--description` and the template flags are
mutually exclusive (the CLI errors out rather than silently preferring
one).

`--dtype f16` (~2.5 GB) casts only the decoder matmul weights and
embeddings; the T5 encoder stays f32 (Flan-T5 activations overflow the
f16 range — the well-known T5 fp16 trap), as do norms, biases, snake
alphas, the positional table and all DAC tensors.

Quantized recipes (mini sizes; argmax agreement vs the f32 fixtures over a
200-step teacher-forced trace, f16 scoring 99.6%):

| `--dtype` | size | agree | bulk / tables / heads / T5 |
|---|---|---|---|
| `q8_0` | ~1.16 GB | 98.1% | q8_0 / f16 / f16 / q8_0 |
| `q6_k` | ~0.98 GB | 94.9% | q6_K / q6_K / f16 / q8_0 |

Sub-q6 tiers measured well below this quality floor (q5_0 90.4%, q4_k_m
83.1% on mini; 70.0% / 65.9% on large) and are deliberately not shipped —
reproduce them via `--recipe` if ever needed.  The recipes transfer to
the indic checkpoint unchanged (its own 200-step fixtures: f16 99.6%,
q8_0 98.3%; f32 scores 100%).  All recipes keep the f32
set above untouched and the T5 matmuls at q8_0 (quantized dot products
re-quantize activations per block, so the f16 trap does not apply — T5
must never go f16).  The 9 LM heads never drop below q8_0 (6-bit heads
derail sampled decoding); both tiers lift heads (q8_0 also tables) to
f16 — the dominant quality lever found by grid search
(`scripts/parler/quant-grid.py`; per-tier override via `--recipe`, optional
activation-weighted quantization via `scripts/parler/compute-imatrix.py` +
`--imatrix`).  Non-trivial encodes go through the built ggml library
(`ggml_quantize_chunk` via ctypes; auto-located, or pass `--ggml-lib`), so
build tts-cpp first.  Quantized output diverges from the f32 parity
fixtures by construction — validate by ear;
`PARLER_TEST_REPORT_ONLY=1 ./build/test-parler-t5 MODEL.gguf REF_DIR` and
the corresponding `test-parler-decoder` invocation print stage metrics without
enforcing the f32 tolerance bars.

Digits in the prompt are expanded to English words before tokenization
("12" → "twelve"): parler-v1 ships no text front-end and voices raw digits
badly, so this deliberately diverges from stock HF.  English cardinals
(incl. thousands separators), decimals and ordinals are covered; times,
currency and units are not.  Opt out with `--no-normalize-numbers`
(`EngineOptions::normalize_numbers = false`); the description is never
rewritten.  On indic-class models, ASCII digit runs whose nearest letter
context is an Indic script are instead transliterated to that script's
native digits ("कमरा 12" → "कमरा १२"; all 13 digit-bearing scripts of the
21 languages covered); Latin-context runs still become English words, and
native numerals always pass through untouched.  Best effort by design:
the model only voices native numerals for scripts whose numerals were
frequent in its training text (Devanagari yes; e.g. Gujarati no —
verified by ear); full per-language number-words belong upstream where
the prompt language is known.

Verification: `ctest -R test-parler` runs tokenizer/T5/decoder/delay/DAC/
e2e parity against `.npy` fixtures produced by
`scripts/parler/dump-reference.py` (HF PyTorch reference, greedy).
`test-parler-gguf-load` needs no fixtures: it synthesizes tiny GGUFs and
asserts the loader fails closed on truncated files, missing metadata, and
wrong-typed metadata.  The
greedy token trace matches HF exactly; the DAC waveform matches at
>120 dB SNR. Default decoding is sampled (temperature 1.0, top-k 50, from
`generation_config.json`). End-user `--greedy` and `top_k=1` requests are
repaired to sampling with a warning because argmax does not terminate for this
architecture; use `--seed` for reproducibility. Greedy is retained only inside
the bounded parity harness.

Streaming is available through the `parler::Engine` callback API using
`stream_chunk_frames` and `stream_first_chunk_frames`. Neither `parler-cli` nor
the limited `tts-cli` Parler route exposes those controls.

## Snapdragon Hexagon NPU

Parler-TTS runs on the Snapdragon Hexagon NPU (HTP0; measured on a Snapdragon
8 Elite, Hexagon v79) when it is requested explicitly:
`EngineOptions::backend = "hexagon"` or `--backend hexagon` on `parler-cli`,
`parler-bench` and `parler-fit-params`. There is no fallback: construction
fails when HTP0 is missing, and the automatic `--n-gpu-layers` walk never
selects it. The T5 encoder, the decoder (prefill and every step, flash
attention over an F16 KV cache) and the DAC all run on the NPU; the host only
samples. Weights and the fused decoder stacks go into the Hexagon repack
buffer type. HTP has no K-quant matmul and looks up embedding rows only from
F32/F16 tables, so a `q6_k` GGUF is refused at load: use `q8_0`, `f16` or
`f32`. It needs a `qvac-ext-ggml@speech` build with the Hexagon fixes for
fused F32-precision matmuls and the tanh GELU.

The Hexagon DAC runs each snake as one fused node, each transposed
convolution as one GEMM into columns plus `ggml_col2im_1d` (the kernels are
rearranged at load, about 97 MB more for mini), and its convolutions over F16
im2col columns on HMX, which multiplies in F16 anyway; HTP has no fast F32
im2col.

Accuracy against the HF fixtures, `f32` GGUF (`PARLER_TEST_BACKEND=hexagon`
runs the fixture tests on any named backend):

| Check | Hexagon | CPU | OpenCL (Adreno 830) |
|---|--:|--:|--:|
| T5 cross states, relative error | 5.9e-7 | 2.7e-4 | 1.6e-6 |
| decoder argmax agreement, 360 teacher-forced predictions | 359 | 360 | 360 |
| DAC waveform SNR | 70.5 dB | 126.9 dB | 121.7 dB |
| greedy end-to-end token trace | 429 / 429 | 429 / 429 | 429 / 429 |

The F16 HMX products set the DAC's SNR, well inside the 50 dB accelerator bar.
With the `q8_0` GGUF the decoder agrees with the 200-step `f32` reference on
97.25% of 3,600 teacher-forced predictions on Hexagon and 97.72% on CPU.

Galaxy S25, `parler-mini-v1` `q8_0`, `parler-bench` greedy for 398 decoder
steps (4.54 s of audio), median of 3 warm runs, each backend started from
thermal status 0, ms:

| Backend | T5 | prefill | decode | DAC | total | vs OpenCL |
|---|--:|--:|--:|--:|--:|--:|
| CPU (4 threads) | 178 | 168 | 6,369 | 23,563 | 29,602 | 0.28x |
| OpenCL (Adreno 830) | 71 | 70 | 6,314 | 1,838 | 8,294 | 1.00x |
| Hexagon | 47 | 24 | 5,080 | 1,218 | 6,368 | 1.30x |
| Hexagon, `GGML_HEXAGON_OPPOLL=1` | 43 | 19 | 3,889 | 1,165 | 5,116 | 1.62x |

The same for 998 steps (11.51 s of audio), median of 2 warm runs. OpenCL's
decoder slows from 15.9 to 27.3 ms per step as the context grows; Hexagon's
from 12.8 to 13.8 ms:

| Backend | T5 | prefill | decode | DAC | total | vs OpenCL |
|---|--:|--:|--:|--:|--:|--:|
| CPU (4 threads) | 166 | 159 | 20,750 | 60,064 | 81,139 | 0.40x |
| OpenCL (Adreno 830) | 72 | 72 | 27,290 | 4,908 | 32,341 | 1.00x |
| Hexagon | 48 | 24 | 13,776 | 3,022 | 16,870 | 1.92x |
| Hexagon, `GGML_HEXAGON_OPPOLL=1` | 45 | 20 | 11,685 | 3,159 | 14,909 | 2.17x |

A decode step streams about 375 MB of `q8_0` weights through HVX at about
52 GB/s (7.2 ms of 9.5 ms DSP time at 220 tokens of context); attention adds about 3.3 us per
context token. By default the host blocks on the DSP queue during each step
and the CPU clocks down, so the host work between steps runs slowly;
`GGML_HEXAGON_OPPOLL=1` polls instead, which cuts decoding by a quarter but
keeps one core busy and heats the phone like the CPU backend (thermal status
2 after the benchmark, against 0 with the default wait).
