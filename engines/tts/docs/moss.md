# tts engine: MOSS Delay

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

# cloning: add the codec encoder and a reference wav (must match the
# codec sample rate; stereo is downmixed by averaging)
build/moss-cli --backbone moss-tts-delay-f16.gguf \
    --decoder moss-codec-decoder-f16.gguf \
    --encoder moss-codec-encoder-f16.gguf --ref-audio speaker.wav \
    --text "..." --language en --out out.wav

# on the GPU: same command plus --gpu
```

`--language` defaults to `zh`, matching the reference pipeline. Sampling
follows the reference defaults — text temperature 1.5 / top-k 50, audio
temperature 1.7 / top-p 0.8 / top-k 25, repetition penalty 1.0 over the
audio channels, seed 1234 — and every knob is exposed as a flag
(`--text-temperature`, `--audio-top-k`, `--audio-repetition-penalty`, ...).

### Engine notes

The public API is `tts_cpp::moss::Engine` in
[`include/tts-cpp/moss/engine.h`](../include/tts-cpp/moss/engine.h):
construct with `EngineOptions` (model paths, language, sampling, `use_gpu`,
`context`, `max_new_tokens`), call `synthesize(text)` for a
`SynthesisResult` (mono PCM, sample rate, frame count, generation and decode
wall times), and `cancel()` from another thread to stop a run. One synthesis
runs at a time per instance; overlapping calls throw, and the engine rejects
a decoder whose quantizer count does not match the backbone's channel count.
`encoder_path` and `reference_audio_path` are needed only for voice cloning:
the reference WAV must already be at the codec sample rate (channels are
averaged; there is no resampling). Each request reseeds the RNG from the
options, so equal requests on one instance produce equal audio.

Generation mirrors the reference loop. The prompt packs the fixed
`<user_inst>` template between `<|im_start|>`/`<|im_end|>` markers and ends
with an `audio_start` seed row; a reference expands to
`audio_start + (frames + n_vq - 1)` user-slot rows carrying the
delay-patterned codes + `audio_end`. During decoding the text channel is
masked to the control tokens valid at each step, audio channel `c` opens at
step `c`, and when the model emits the delay slot the drain state machine
forces delay slots while the channels close one per step and then forces
`audio_end`. The emitted rows are de-delayed, split into segments at all-pad
frames, concatenated, and decoded by the codec.

### Test

The suites build with `TTS_CPP_BUILD_TESTS` and need no model downloads.
`test-moss-generation` is pure CPU logic over the generation primitives.
`test-moss-load` builds tiny GGUF fixtures in-test and covers the accept path
plus every load-time rejection for the backbone, the codec, and the engine
pairing. `test-moss-cancel` drives a real synthesis on the fixture models and
cancels it from another thread. `test-moss-cli` covers the flag surface and
runs the CLI end to end against the fixtures. `test-convert-moss` (Python,
registered when an interpreter is available, skips without numpy/gguf)
fabricates a tiny checkpoint on disk, runs both converters, and validates the
emitted GGUFs.
