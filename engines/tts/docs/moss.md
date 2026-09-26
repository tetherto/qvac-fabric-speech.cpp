# tts engine: MOSS

Part of the [tts engine documentation](../README.md).

## MOSS Delay

[MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) is a family of
delay-pattern TTS models from OpenMOSS: a Qwen3-shaped autoregressive
backbone reads packed rows of one text token plus 32 audio codes and emits
the next row, one text head and 32 audio heads wide, and a
residual-vector-quantised codec turns the code frames into 24 kHz mono audio
at 12.5 frames per second (1920 samples per frame). The 32 audio channels are
staggered one step apart — the delay pattern — so channel `c` of frame `t`
is emitted at step `t + c`. The engine covers the `MossTTSDelay`
architecture, which is shared by MOSS-TTS-v1.5 (single speaker turn) and
MOSS-TTSD (dialogue checkpoints).

Cloning needs no speaker encoder: the codec *encoder* turns a reference wav
into code frames, and those frames are packed, delay-patterned, into the
user section of the prompt.

**Status — CPU and Metal, validated end to end against the released
checkpoints.** `test-moss-generation` pins the delay/de-delay round trip, the
in-loop drain state machine, the sampling primitives, segment extraction, and
prompt packing against the upstream semantics. Against converted
MOSS-TTS-v1.5 and MOSS-Audio-Tokenizer GGUFs, the codec encoder-to-decoder
round trip reaches 0.94 correlation on real audio, and full synthesis on
Metal produces clean speech that terminates generation naturally. The codec
transformers run banded attention: each sliding-window block attends inside
its configured context window, so attention memory scales with the window
instead of the clip length, and the banded output matches the dense
computation on real audio. Numeric reference parity and the remaining
backends are the follow-up to this change.

### Convert

Three GGUFs, because the halves have different lifetimes: the backbone and
the codec decoder are needed for every synthesis, the codec encoder only for
voice cloning. Checkpoints come from
[OpenMOSS/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS/MOSS-TTS-v1.5) (or a
MOSS-TTSD checkpoint with the same architecture) and
[OpenMOSS/MOSS-Audio-Tokenizer](https://huggingface.co/OpenMOSS/MOSS-Audio-Tokenizer)
for the codec.

```sh
# backbone: Qwen3 layers, text embedding + 32 audio embeddings,
# text head + 32 audio heads, tokenizer vocabulary and merges
python scripts/convert-moss-delay-to-gguf.py /path/to/MOSS-TTS-v1.5 \
    --outtype f16 --outfile moss-tts-delay-f16.gguf

# codec: one file per half; each stands alone
python scripts/convert-moss-codec-to-gguf.py /path/to/MOSS-Audio-Tokenizer \
    --outtype f16 \
    --encoder-outfile moss-codec-encoder-f16.gguf \
    --decoder-outfile moss-codec-decoder-f16.gguf
```

The backbone converter flattens the checkpoint's `language_config` into the
root hyperparameters, maps `emb_ext.{i}` to `token_embd_audio.{i}` and
`lm_heads.0` / `lm_heads.k` to `output` / `output_audio.{k-1}`, and exports
the tokenizer as plain `tokenizer.ggml.tokens` / `tokenizer.ggml.merges`
arrays. The codec converter merges weight-norm parametrizations and writes
per-module metadata (patch sizes, transformer shapes, context windows) so the
engine can rebuild the module chain without hardcoded shapes.

### Run

```sh
# text only: backbone + codec decoder
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --text "..." --language en --out out.wav

# streaming: chunks arrive at the terminal as they decode; the WAV is
# identical to the batch render for the same seed
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --text "..." --language en --stream --out out.wav

# cloning: add the codec encoder and a reference wav (must match the
# codec sample rate; stereo is downmixed by averaging)
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --encoder moss-codec-encoder-f16.gguf --ref-audio speaker.wav \
    --text "..." --language en --out out.wav

# two-speaker dialogue (MOSS-TTSD checkpoint): one reference per speaker,
# and the text carries the reference transcripts first, then the turns,
# all tagged [S1]/[S2]
build/moss-cli --backbone moss-ttsd-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --encoder moss-codec-encoder-f16.gguf \
    --dialogue-ref s1.wav --dialogue-ref s2.wav \
    --text "[S1] <s1 transcript> [S2] <s2 transcript> [S1] ... [S2] ..." \
    --language en --out dialogue.wav

# on the GPU: same command plus --gpu
```

`--language` defaults to `zh`, matching the reference pipeline. The speech
is directable: `[pause 2.0s]` markers inside the text insert silences of
roughly the requested length, Pinyin (`ni3 hao3`) and IPA (`/həloʊ/`) notation
inline in the text steer pronunciation, and `--duration-tokens N` asks the
model for a target length of N codec frames (12.5 per second; 0 keeps the
length free). A target length also needs `n_vq - 1` drain steps plus the
`audio_end` and `im_end` rows, so the engine rejects a `duration_tokens`
that does not fit in `max_new_tokens` together with them. Sampling
follows the reference defaults — text temperature 1.5 / top-k 50, audio
temperature 1.7 / top-p 0.8 / top-k 25, repetition penalty 1.0 over the
audio channels, seed 1234 — and every knob is exposed as a flag
(`--text-temperature`, `--audio-top-k`, `--audio-repetition-penalty`, ...).

### Engine notes

The public API is `tts_cpp::moss::Engine` in
[`include/tts-cpp/moss/engine.h`](../include/tts-cpp/moss/engine.h):
construct with `EngineOptions` (model paths, language, sampling, `use_gpu`,
`backends_dir` for builds that load ggml backends at runtime, `context`,
`max_new_tokens`), call `synthesize(text)` for a
`SynthesisResult` (mono PCM, sample rate, frame count, generation and decode
wall times), and `cancel()` from another thread to stop a run. One synthesis
runs at a time per instance; overlapping calls throw, and the engine rejects
a decoder whose quantizer count does not match the backbone's channel count.
`encoder_path` and `reference_audio_path` are needed only for voice cloning:
the reference WAV must already be at the codec sample rate (channels are
averaged; there is no resampling). Each request reseeds the RNG from the
options, so equal requests on one instance produce equal audio.

Dialogue runs in the reference pipeline's continuation mode: each speaker's
reference expands to its own `[Sn]` audio block in the user message, and the
assistant message opens with the concatenated references' codes truncated by
`n_vq - 1` rows, so the model continues the delay pattern seamlessly. The
references also stay in the codec as causal history: the decoder renders them
together with the generated audio and drops their samples, so the first
generated frames decode exactly as the reference pipeline's
decode-then-trim, in batch and in streaming. The codec and the delay model
validate every tensor against the declared geometry at load, and the codec
rejects a sampling rate outside 8-192 kHz; reference WAVs are capped at 60
seconds and at an absolute sample count that does not depend on the model. A backbone may use fewer channels than the codec carries
(MOSS-TTSD uses the first 16 of the tokenizer's 32 residual quantizers):
references are truncated to the backbone's channels and the decoder sums
only those levels.

`synthesize_stream(text, callback)` emits PCM chunks while generation runs:
a frame is complete `n_vq - 1` steps after its row, completed non-pad frames
accumulate, and every `stream_chunk_frames` of them decode incrementally: the
codec keeps each transformer layer's keys and values for its attention window
between chunks, so a chunk decodes only its new frames and the streamed audio
equals the batch render. Codec work stays linear in the output length. The
callback runs on the calling thread and returning false cancels the request;
`SynthesisResult::first_audio_ms` reports the latency to the first chunk and
`pcm` stays empty. The batch path uses the same incremental decode in
60-second windows once a segment exceeds that length, which bounds the
codec's graph size without changing the output.

Generation mirrors the reference loop. The prompt packs the fixed
`<user_inst>` template between `<|im_start|>`/`<|im_end|>` markers and ends
with an `audio_start` seed row; a reference expands to
`audio_start + (frames + n_vq - 1)` user-slot rows carrying the
delay-patterned codes + `audio_end`. During decoding the text channel is
masked to the control tokens valid at each step, audio channel `c` opens at
step `c`, and when the model emits the delay slot the drain state machine
forces delay slots while the channels close one per step and then forces
`audio_end`. The emitted rows are de-delayed and split into segments at
all-pad frames; each segment decodes from a fresh codec context, in batch and
in streaming alike, so one segment never attends to another.

On ARM CPUs ggml accumulates f16 dot products in f16, and the backbone's
feed-forward output exceeds that range; the down projection therefore runs
on inputs scaled by a power of two and rescales its f32 result, which keeps
CPU synthesis finite and leaves GPU results unchanged.

### Test

The suites build with `TTS_CPP_BUILD_TESTS` and need no model downloads.
`test-moss-generation` is pure CPU logic over the generation primitives.
`test-moss-load` builds tiny GGUF fixtures in-test and covers the accept path
plus every load-time rejection for the backbone, the codec, and the engine
pairing, including oversized architecture metadata, the duration budget, and
an f16 feed-forward sum past the half-precision range.
`test-moss-codec-stream` checks on random codec weights that incremental
decoding matches a single decode, that segments decode independently, and
that streaming decodes every frame exactly once. `test-moss-cancel` drives a real synthesis on the fixture models and
cancels it from another thread. `test-moss-cli` covers the flag surface and
runs the CLI end to end against the fixtures. `test-convert-moss` (Python,
registered when an interpreter is available, skips without numpy/gguf)
fabricates tiny checkpoints on disk, runs the converters, and validates the
emitted GGUFs.

## MOSS-SoundEffect

[MOSS-SoundEffect-v2](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
turns a text description into a sound effect of up to 30 seconds. It is a
flow-matching diffusion model: a Qwen3-1.7B text encoder embeds the prompt, a
30-layer Wan-style DiT (adaLN modulation, RoPE self-attention over the latent
frames, cross-attention to the text) predicts the velocity, and a continuous
DAC decoder turns the 128-channel latent (50 frames per second) into 48 kHz
mono audio. It is not a speech model; it shares the MOSS CLI and the Qwen
tokenizer with the Delay engine and nothing else.

**Status — CPU and Metal, validated against the reference pipeline.** With
the same noise, every stage matches the PyTorch fp32 pipeline: the text
encoder, one DiT velocity for the prompt and for the empty negative prompt,
an eight-step guided sampling loop, and the VAE decode. An f16 GGUF reaches
cosine similarity 0.99998 or better on both backends; a q8_0 GGUF reaches
0.9995 per stage and 0.9987 after the eight sampling steps. The CPU path is
correct but slow (about ten seconds per DiT pass on an M1 Ultra, so a
100-step clip takes over half an hour); use `--gpu` where one is available.

### Convert

One GGUF holds the text encoder, the DiT, the VAE decoder, the tokenizer, and
the generation defaults.

```sh
python scripts/convert-moss-sfx-to-gguf.py /path/to/MOSS-SoundEffect-v2.0 \
    --outtype f16 --outfile moss-sfx-v2-f16.gguf
```

`--outtype` picks the type of the linear weights and of the text token
embedding: `f16` (6.4 GB) and `q8_0` (3.5 GB) are validated; `bf16` and `f32`
are also written. Norms, biases, modulation tables, snake parameters, and the
two small 1x1 projections (`dit.patch_embd`, `vae.post_quant`) stay f32, and
the VAE convolution kernels stay f16 because ggml's CPU im2col requires it.
The converter drops the text encoder's unused language-model head, folds the
VAE weight norm into plain kernels, lays out the transposed convolutions as
GEMM columns, and stores `1 / (alpha + 1e-9)` next to each snake `alpha`.

### Run

```sh
build/moss-cli --mode sfx --model moss-sfx-v2-f16.gguf \
    --text "Heavy rain falling on a tin roof with distant thunder." \
    --seconds 10 --gpu --out rain.wav
```

The defaults come from the GGUF and match the reference pipeline: 100 steps,
guidance 4.0, shift 5.0, and an empty negative prompt. `--steps`,
`--guidance`, `--shift`, `--negative-prompt`, and `--seed` (0 by default in
this mode) override them. `--seconds` must be in (0, 30] and is rounded to
0.1 s the way Python's `round` does. There is no streaming in this mode. On an M1 Ultra a
100-step clip takes about two minutes on Metal whatever its length, because
the model always denoises the full 30-second latent and crops the result.

### Engine notes

The public API is `tts_cpp::moss::SoundEffectEngine` in
[`include/tts-cpp/moss/sound_effect.h`](../include/tts-cpp/moss/sound_effect.h):
construct with `SoundEffectOptions` (model path, threads, `use_gpu`,
`backends_dir`), call `generate(request, progress)` for a `SoundEffectResult`
(mono PCM, sample rate, text, diffusion, and decode wall times), and
`cancel()` from another thread or `false` from the progress callback to stop
a run. A request leaves `steps`, `guidance`, and `shift` at zero to use the
model defaults and rejects negative or non-finite values; guidance 1 skips
the negative branch. Prompts are capped at 8192 bytes. One generation runs at
a time per instance; overlapping calls throw.

Generation mirrors the reference pipeline. The prompt gets the training-time
suffix ` duration: X.Xs` and is cleaned the way upstream's `ftfy`-based
cleaner does it: HTML entities are unescaped (to a fixed point, or exactly
twice when the text contains `<`), curly quotes are straightened, full-width
forms and the ideographic space are narrowed, Latin ligatures are split, and
Unicode whitespace collapses to single spaces, so Chinese and typographic
prompts tokenize to the same ids as the Hugging Face tokenizer. Malformed
UTF-8 becomes U+FFFD byte by byte. Not reproduced: Unicode normalization
(NFC), mojibake repair, named entities beyond `&amp; &lt; &gt; &quot; &apos;
&nbsp;` (and named entities without a semicolon), the cp1252 mapping Python
applies to `&#128;`-`&#159;`, numeric references longer than eight digits,
halfwidth katakana and the other halfwidth forms past U+FF5E, and ftfy's
removal of the byte-order mark and control characters. The cleaned prompt is
tokenized up to 512 tokens; the text encoder
runs causally over the real tokens and the padding rows of the 512-token
context stay zero, as upstream. The empty negative prompt is an all-zero
context. The sampler is first-order Euler over the shifted flow-matching
schedule, `sigma' = shift * sigma / (1 + (shift - 1) * sigma)`, with classic
classifier-free guidance. The initial noise comes from a seeded Mersenne
Twister with a Box-Muller transform, so a seed reproduces a clip across runs
on one backend, but does not reproduce the PyTorch pipeline's clip for the
same seed. The DiT graph is built once per request and reused for every step;
the model refuses to run it once another graph has taken the scheduler. The
loader validates every tensor's shape and type against the metadata and bounds
the metadata itself (layer counts, hop length, latent length) so a malformed
GGUF fails with an error instead of aborting inside ggml. The
VAE decodes only the frames that cover the requested length, in 256-frame
windows with 32 frames of context on each side, and the output is cropped to
the exact sample count.

### Test

`test-moss-sfx` builds a tiny random-weight GGUF in-test and needs no
download: the schedule, Euler step, guidance, prompt cleaning against
upstream outputs, Python-compatible rounding, and noise; the load-time
rejections for metadata bounds, tensor shapes, and tensor types; text-encoder
causality, zero padding, and an f16 feed-forward sum past the half-precision
range; DiT conditioning on text and timestep and the scheduler guard;
windowed VAE decoding against a single window; engine determinism, guidance
1, cancellation from the callback and from another thread, overlapping calls,
and request validation; and the CLI's `sfx` and `tts` modes.
`test-convert-moss` also fabricates a tiny MOSS-SoundEffect pipeline and
checks the tensor census, the folded weight norm, the f16 and q8_0 paths, and
the `.pth` VAE checkpoint (when torch is installed). `test-moss-sfx-parity` is the reference check: it
skips unless `MOSS_SFX_MODEL` points at a converted GGUF and
`MOSS_SFX_REFERENCE_DIR` at stage dumps from the PyTorch pipeline
(`MOSS_SFX_GPU=1` runs it on the GPU). The dumps come from
`scripts/dump-moss-sfx-reference.py <checkpoint> <out_dir> --upstream
<MOSS-TTS checkout>`, whose defaults are the prompt, duration, and step count
the test expects.

## MOSS-Transcribe-Diarize

[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
transcribes speech and labels who spoke when, in one pass over the whole
file. A Whisper-medium-shaped encoder (fine-tuned weights) turns each 30 s
window of 16 kHz audio into 1500 frames, a 4x temporal-merge adaptor brings
them to 12.5 tokens per second, and a 0.6B Qwen3 decoder reads those tokens
in place of `<|audio_pad|>` placeholders and writes the transcript as
`[start][Sxx]text[end]` segments. It is a speech-to-text model; it lives next
to the other MOSS engines because it shares the MOSS CLI and the Qwen
tokenizer.

**Status — CPU and Metal, validated against the reference model.** On a
two-minute Spanish reference, every stage matches the Hugging Face model in
fp32: the log-mel front end (cosine 1.000000), the encoder (0.99998 on Metal,
0.9997 on CPU), the adaptor (0.999998), and the prefill logits (1.000000);
the greedy transcript is identical token for token (726 tokens, 20
segments). On a long-form set of concatenated FLEURS clips (2, 5, 10, and 30
minutes), the f16 and q8_0 GGUFs on Metal score the same WER and CER as the
reference model: 2.1-3.9% WER in Spanish and 9.4-12.1% CER in Chinese, with
full coverage. English is not usable on that set (46-62% WER, the model skips
whole spans), and the reference model does the same.

### Convert

One GGUF holds the encoder, the adaptor, the decoder, the Slaney mel filter
bank, the tokenizer, and the prompt defaults.

```sh
python scripts/convert-moss-transcribe-to-gguf.py /path/to/MOSS-Transcribe-Diarize \
    --outtype f16 --outfile moss-transcribe-diarize-f16.gguf
```

`--outtype` picks the type of the linear weights and of the token embedding
(which doubles as the tied output head): `f16` (1.8 GB), `q8_0` (1.0 GB), and
`q5_0` (0.64 GB); `bf16` and `f32` are also written. Rows too short for a
quantization block fall back to f16. Norms, biases, the positional table, and
the mel filters stay f32, and the two encoder convolution kernels stay f16
because ggml's CPU im2col requires it. The converter needs the `tokenizers`
package: it stores the token ids of the default (Chinese) prompt computed by
the Hugging Face tokenizer, because the built-in byte-level tokenizer splits
CJK punctuation differently.

### Run

```sh
build/moss-cli --mode transcribe --model moss-transcribe-diarize-f16.gguf \
    --audio meeting.wav --gpu --out meeting.json
```

The input must be a 16 kHz WAV (stereo is downmixed by averaging). The CLI
prints one `[start-end] Sxx: text` line per segment and, with `--out`, writes
a JSON document with the raw text, the segments, the token counts, and the
stage timings. `--prompt` replaces the default instruction (upstream's
`examples/prompts.md` lists the English and hotword variants; a blank
prompt keeps the default, and a custom prompt goes through the built-in
tokenizer, so CJK punctuation in it may split differently from the Hugging
Face tokenizer) and
`--max-new-tokens` raises the model default of 5120, which is short for
recordings past about fifteen minutes. There is no streaming in this mode.
On an M1 Ultra with Metal and the f16 model, ten minutes of Spanish take
about 95 seconds.

### Engine notes

The public API is `tts_cpp::moss::TranscribeEngine` in
[`include/tts-cpp/moss/transcribe.h`](../include/tts-cpp/moss/transcribe.h):
construct with `TranscribeOptions` (model path, threads, `use_gpu`,
`backends_dir`), call `transcribe(pcm, samples, sample_rate, request,
progress)` for a `TranscribeResult` (text, parsed segments, audio, prompt and
generated token counts, encode, prefill, and decode wall times), and
`cancel()` from another thread or `false` from the progress callback to stop
a run. A request leaves `prompt` empty and `max_new_tokens` at zero to use
the model defaults. Prompts are capped at 8192 bytes. One transcription runs
at a time per instance; overlapping calls throw.

Processing mirrors the reference processor. The audio is cut into
non-overlapping 30 s windows, the last one zero-padded; each window gets a
Whisper log-mel spectrogram (periodic Hann, reflect padding, 80 Slaney bands,
`log10`, clamp to the window maximum minus 8, then `(x + 4) / 4`) and keeps
`(samples - 1) / 1280 + 1` tokens. The prompt is the chat template with the
audio span between `<|audio_start|>` and `<|audio_end|>`; every five seconds
the span carries the elapsed seconds as digit tokens, at the placeholder
positions upstream uses. The decoder runs a single greedy pass up to
`<|im_end|>`, with prefill in 256-token batches and an f16 KV cache (about
112 KiB per token, so a 30-minute file with 16k new tokens needs about 4.4
GB) padded to a multiple of 256 positions. The transcript is decoded with
control tokens skipped and parsed by a port of upstream's streaming
`TranscriptStreamParser`: a segment needs a start timestamp, an `S<digits>`
speaker, text, and an end timestamp no earlier than the start; anything else
stays in the text or is dropped, exactly as upstream does. The loader
validates every tensor's shape and type against the metadata and bounds the
metadata itself, so a malformed GGUF fails with an error instead of aborting
inside ggml.

### Test

`test-moss-transcribe` builds a tiny random-weight GGUF in-test and needs no
download: the mixed-radix FFT against a direct DFT, the log-mel
normalization, the chunk and token arithmetic against the reference
processor's counts, the time-marker placement, the prompt layout for the
default and a custom prompt, byte-level decoding with control tokens
skipped, the transcript parser (edge cases and character-by-character
streaming), the load-time rejections, encoder chunks, decoder prefill in
uneven batches against a single batch, audio injection, engine
end-to-end, cancellation, request validation, and the CLI's `transcribe`
mode. `test-convert-moss` also fabricates a tiny MOSS-Transcribe-Diarize
checkpoint and checks the tensor census, the column biases, the token types,
the stored prompt ids, and the q8_0 path (skipped without `tokenizers`).
`test-moss-transcribe-parity` is the reference check: it skips unless
`MOSS_TRANSCRIBE_MODEL` points at an f16 GGUF (a quantized model passes the
stage checks but its greedy transcript drifts from the fp32 one, although it
scores the same WER) and
`MOSS_TRANSCRIBE_REFERENCE_DIR` at stage dumps from the Hugging Face model
(`MOSS_TRANSCRIBE_GPU=1` runs it on the GPU). The dumps come from
`scripts/dump-moss-transcribe-reference.py <checkpoint> <16 kHz wav>
<out_dir> --upstream <MOSS-Transcribe-Diarize checkout>`.
