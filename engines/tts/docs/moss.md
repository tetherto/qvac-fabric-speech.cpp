# tts engine: MOSS

Part of the [tts engine documentation](../README.md).

## Vulkan validation

All MOSS engines use the shared ggml GPU selector and GPU/CPU scheduler.
With a Vulkan-enabled ggml install, `--gpu` can run MOSS-TTS/TTSD,
MOSS-SoundEffect and MOSS-Speech on Vulkan. For a Vulkan-only development
build, build ggml with `GGML_VULKAN=ON`, `GGML_METAL=OFF` and
`GGML_CUDA=OFF`, install it, and configure the speech engine with that
installation on `CMAKE_PREFIX_PATH`. On a build with several GPU backends,
`TTS_CPP_GPU_BACKEND=vulkan` pins the shared selector for validation:

```sh
TTS_CPP_GPU_BACKEND=vulkan build/moss-cli --gpu --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf --text "Vulkan speech test." --out vulkan.wav
ctest --test-dir build -R '^test-moss-vulkan$' --output-on-failure
```

The test generates small random-weight GGUFs and compares CPU and Vulkan
outputs for Delay prefill/decode/reset, both codec halves, streaming and
reduced-channel dialogue decoding, SoundEffect text/DiT/VAE, Speech's two
LM heads and Whisper-VQ tokenizer. It covers F32, F16 and mixed Q8_0/F16
weights, plus the Speech LM's BF16 export with an F16 tokenizer. It fails
if Vulkan is unavailable instead of accepting a CPU-only run. The existing
manual `tts CI` workflow runs it with `run_gpu=true`; no model downloads
are required for this test.

The same workflow's `run_moss_e2e=true` lane downloads registered checkpoints
and saves generated WAVs, transcripts, logs and model hashes. Cases run
serially because runner services share GPU memory. Speech cases use
`test-moss-speech-e2e` to make two replies on one engine instance, checking
that staged model unloading also works on the next request.

Each Vulkan invocation also takes a host-local lock keyed by the physical
GPU UUID and waits for two consecutive memory readings with at most 1 GiB
unavailable. Admission times out after 15 minutes and records the readings;
foreign processes are never terminated. This coordinates OpenMOSS runs across
runner accounts and avoids starting while another workload occupies the GPU.
Other GPU workloads do not honor this lock and can still start afterward;
before/after process snapshots remain necessary to diagnose contention.
The TTS case also saves a CPU reference with the same checkpoint, text and
seed to investigate backend-dependent generation quality.

Speech cases also run `test-moss-speech-codec-e2e CODEC.gguf OUTPUT_DIR`.
This loads only the codec, comparing CPU and Vulkan on fixed speech tokens,
identical diffusion noise and a short voice reference. It checks finite mel
values, numerical agreement, output length and signal level for both the
single-batch and batched CFG paths, and saves the audio and mel dumps.
The shared S3Gen flow decoder requests F32 flash-attention accumulation:
default F16 accumulation on MoltenVK can amplify errors across solver steps
until the mel becomes non-finite. F16 weights and other Vulkan operations
remain enabled; no `GGML_VK_DISABLE_F16` workaround is required. Non-finite
flow states are reported as synthesis errors instead of decoded as audio.

These generated-model checks pass on Apple M2 through MoltenVK. They do
not establish full-checkpoint speech quality, performance, or Android and
discrete-GPU compatibility. The per-model reference tests below remain the
full-checkpoint validation path; MOSS-Speech's shared S3Gen/HiFT reply decoder
also needs the real codec checkpoint.

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

## MOSS-Speech

[MOSS-Speech](https://huggingface.co/fnlp/MOSS-Speech) answers a spoken
question with speech, without a text step in between. A 9B Qwen3 language
model shares its first 32 layers between two channels, then splits into a
4-layer text branch and a 4-layer audio branch, each with its own KV cache,
final norm, and head. Each position carries a (text, audio) token pair: the
text channel reads the text embedding, and a text placeholder hands the
position to the audio embedding. The companion
[codec](https://huggingface.co/fnlp/MOSS-Speech-Codec) turns user speech into
tokens with a causal Whisper-VQ encoder (12.5 tokens per second, 16384 codes)
and speaks the reply with the CosyVoice2 flow-matching decoder and HiFT
vocoder at 24 kHz, conditioned on a voice prompt and a CAM++ speaker
embedding. The decoder is the same S3Gen stack Chatterbox uses, so the engine
reuses it; the codec only changes the token-to-mel ratio and the speech
vocabulary, which the S3Gen loader now reads from the GGUF.

GPU speech replies run the LM and S3Gen decoder in separate memory phases.
After token generation, the engine releases LM weights and KV/graph buffers
before decoding audio, while retaining tokenizer metadata and backend identity.
The decoder holds a matched S3Gen cache reference only for that decode; other
cache owners retain their references. The next request reloads LM weights from
the same GGUF, so keep the model file available. This reduces peak GPU memory
at the cost of loading weights again between spoken replies. CPU execution
keeps its resident LM weights.

**Status — CPU and Metal, validated against the reference pipeline.** Every
stage matches the PyTorch pipeline on a real question: the log-mel features
(cosine 1.0) and the 45 user speech tokens (45 of 45), the voice-prompt mel
(1.0) and the CAM++ embedding (0.99991), the reply mel from the flow decoder
with the reference noise (0.99977), and the prefill logits of both heads
(text 0.99997, audio 0.999998 in bf16; 0.99994 and 0.999998 in q8_0). With
teacher forcing, greedy decoding picks the reference's audio code at 93.5 %
of the positions in both bf16 and q8_0; the misses are near-ties between
bf16 on MPS and ggml, so free-running greedy replies drift apart after a few
tokens: in bf16 the greedy reply still ends on its own, while in q8_0 it can run
on to the token limit, so keep the default sampling for q8_0. Sampled replies to English questions are
coherent and intelligible in both quantizations. On an M1 Ultra with Metal
the LM runs at about 32 tokens per second in bf16 and 47 in q8_0, faster than
the 12.5 tokens per second of speech it produces; while it speaks, each step
skips the text head, which the audio channel does not read.

### Convert

```sh
# language model: shared trunk, both branches, both heads, tokenizer,
# special tokens, and the default system prompts
python scripts/convert-moss-speech-to-gguf.py /path/to/MOSS-Speech \
    --outtype q8_0 --outfile moss-speech-q8_0.gguf

# codec: Whisper-VQ encoder + CosyVoice2 flow + HiFT + CAM++ + default voice
python scripts/convert-moss-speech-codec-to-gguf.py /path/to/MOSS-Speech-Codec \
    --campplus-gguf cosyvoice3-campplus-f32.gguf \
    --voice-wav /path/to/MOSS-Speech/assets/prompt_en.wav \
    --outtype f16 --outfile moss-speech-codec-f16.gguf
```

The LM converter writes `bf16` (18.2 GB, the default), `q8_0` (9.7 GB),
`f16`, and `f32`; norms stay f32. The codec converter needs torch,
safetensors, transformers, librosa, and soundfile, and takes the CAM++
weights from the CosyVoice3 CAM++ GGUF (the same network the CosyVoice2
frontend runs). `--voice-wav` becomes the default reply voice, stored as mono
float samples at its original rate (at most 60 s). The codec's `--outtype` is
`f16` (1.3 GB, validated), `q8_0`, or `f32`; the conv kernels and the
Whisper-VQ matrices stay f16 in the quantized file.

### Run

```sh
build/moss-cli --mode s2s --model moss-speech-q8_0.gguf \
    --codec moss-speech-codec-f16.gguf --audio question.wav \
    --gpu --out reply.wav
```

`--audio` is the user turn (any rate from 8 to 192 kHz, stereo downmixed).
`--voice ref.wav` (up to 60 s) replaces the default voice, `--system "..."` adds a system
turn, and `--text-reply` asks for a text answer instead of speech (printed to
stdout). Sampling defaults to the upstream values: temperature 0.7, top-p
0.95, top-k 20, seed 0; `--greedy`, `--temperature`, `--top-p`, `--top-k`, and
`--seed` override them. By default, generation continues until the model ends
its reply or the remaining model context is exhausted. `--max-new-tokens N`
sets an explicit reply budget (`0`, the default, uses the remaining context),
and `--max-reply-seconds` cuts the spoken part cleanly by forcing
the end-of-speech code. There is no streaming in this mode.

### Engine notes

The public API is `tts_cpp::moss::SpeechEngine` in
[`include/tts-cpp/moss/speech.h`](../include/tts-cpp/moss/speech.h):
construct with `SpeechOptions` (LM and codec paths, threads, `use_gpu`,
`backends_dir`) and call `respond(request, progress)` with a `SpeechRequest`
that holds the conversation as `SpeechMessage`s. Each message carries either
text or audio (PCM plus sample rate), roles are system, user, and assistant,
and the last message must be the user's. The `SpeechResult` holds the reply
PCM at 24 kHz, the text channel's output, token counts, whether the reply was
truncated by `max_reply_seconds` or `max_new_tokens`, and the encode, prefill, generation, and decode wall times.
`cancel()` from another thread stops a run at the next speech-tokenizer
segment, prefill batch, generation step, or decoder step, and `false` from the
progress callback stops generation; one response runs at a time per
instance. A request holds up to 256 messages, each with up to 16384 bytes of
text or 600 s of audio.

The prompt follows the upstream processor: each turn is
`<|im_start|>role\n` + content + `<|im_end|>\n`, a spoken turn is the
speech-start token, one placeholder row per speech code, and the
end-of-speech code, and when the conversation does not open with a system turn the default
one ("Respond with spoken outputs", or with text outputs for `--text-reply`)
is appended after the turns, as upstream does. The spoken reply opens with
the assistant header and the speech-start token. Generation mirrors the
reference loop: the channel switches to audio after speech-start and back to
text after end-of-speech, audio codes past end-of-speech are masked, the
end-of-speech code is masked for the first nine tokens (upstream's
`min_new_tokens` of 10 counts from one) and `<|im_end|>` on the text channel
for the first ten, as Hugging Face's minimum-length processor does, the same temperature
and top-k/top-p apply to both heads, the text channel is forced to the
placeholder while speaking, and generation stops on `<|im_end|>` or the pad
token. The reply's speech codes are the audio channel from the first
speech row up to the first end-of-speech. A stop row closes the reply and is
dropped; a reply cut by `max_new_tokens` keeps its last row.

The voice prompt is prepared the way the CosyVoice2 frontend does it: speech
tokens from the Whisper-VQ encoder, an 80-band 24 kHz mel trimmed to four
frames per token, and a CAM++ embedding of the 16 kHz audio. The default
voice is prepared once per engine and cached. CAM++ runs on the scalar CPU
path, which is the one that matches the reference; its convolutions now
accumulate along time rows so the compiler vectorizes them, with the same
summation order per output, which cuts the 36-second default voice from
about 24 s to about 4 s on an M1 Ultra. The flow decoder's source noise in
HiFT is random, so the reply waveform is compared with the reference through
its log-mel spectrum. The LM loader validates the geometry, the special
tokens, and every tensor's shape before it allocates; the KV cache is sized
from the prompt plus `max_new_tokens`, rounded up to 256 positions.

### Test

`test-moss-speech` needs no download: it builds a tiny random-weight LM and
a tiny Whisper-VQ GGUF in-test and covers the generation state machine
(channel switches, masked codes, the minimum length, the reply-length cut,
the stop tokens, and reply code extraction), the prompt layout against the
processor's grid, the reply text, the LM load-time rejections, batched
prefill against one batch and against cached steps, the embedding selection
per channel, context overflow, the tokenizer's segment arithmetic, log-mel
normalization, per-segment encoding and its rejections, request validation,
and the CLI's `s2s` flags. `test-convert-moss` fabricates a tiny MOSS-Speech
checkpoint and checks the tensor census, the metadata, the token types, all
four output types, and the rejections, and exercises the codec converter's
Whisper-VQ, CAM++, and default-voice helpers when torch is installed.
`test-moss-speech-parity` is the reference check: it skips unless
`MOSS_SPEECH_MODEL` and `MOSS_SPEECH_REFERENCE_DIR` point at the converted LM
and at dumps of the PyTorch pipeline, and it adds the tokenizer and decoder
stages when `MOSS_SPEECH_CODEC` points at the codec GGUF (`MOSS_SPEECH_GPU=1`
runs it on the GPU, `MOSS_SPEECH_DUMP_MEL=<file>` writes the reply mel). It
prints every number in the status paragraph above and fails below backend-safe
floors: cosine 0.9999 for the user log-mel, 0.999 for the voice mel and the
CAM++ embedding, 0.99 for the reply spectrum and the prefill logits, 95 % of
the user speech tokens, and 90 % teacher-forced agreement. The dumps come from
`scripts/dump-moss-speech-reference.py <MOSS-Speech> <MOSS-Speech-Codec>
<out_dir> --upstream <MOSS-Speech checkout> --user-wav question.wav --device
mps`, which writes the `lm/` and `codec/` folders the test reads.
