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
Vulkan, and OpenCL (Adreno): the GPU path (F16 flash attention, fused QKV +
stacked LM heads, DAC upsampling as phase matmuls) is gated on that
allowlist in `src/parler/gguf.cpp`, and any other GPU backend is released at
load and replaced by CPU rather than run unvalidated.  The graph dispatch
uses the shared `sched_dispatch` dual path like the other engines.

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
