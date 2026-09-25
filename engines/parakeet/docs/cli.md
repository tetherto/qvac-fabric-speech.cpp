# parakeet engine: command-line tools

Part of the [parakeet engine documentation](../README.md).

## CLI and microphone examples

The file CLI accepts:

```text
parakeet --model <model.gguf> (--wav <16-kHz-mono.wav> |
         --pcm-in <raw> --pcm-format s16le|f32le --pcm-rate 16000) [options]
```

Useful groups include `--threads`, `--n-gpu-layers`, `--backends-dir`,
`--language`, `--stream`, `--stream-duplex`, context/chunk options,
`--diarization-model`, OpenCL environment controls, `--bench`, `--profile`,
and `--dump-mel`. Run `parakeet --help` for the complete list.
`--require-coreml` (with `--bench`, CTC/IndicConformer, Unified RNN-T, TDT,
EOU, and Nemotron only) fails unless every timed encoder invocation ran on the
[Core ML sidecar](backends.md#core-ml-encoder-sidecar). `--bench`
covers the transcription models only: the diarization path (a Sortformer GGUF
at `--model`) and the attributed path (`--diarization-model`) return before
the bench loop, ignoring the `--bench*` flags — time the invocation externally
to benchmark those.

```bash
build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --n-gpu-layers 1

build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --stream --stream-duplex --emit jsonl

build-parakeet/parakeet \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --diarization-model engines/parakeet/models/diar_sortformer_4spk-v1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# standalone Sortformer diarization: speaker segments only, no ASR model
build-parakeet/parakeet \
  --model engines/parakeet/models/diar_sortformer_4spk-v1.f16.gguf \
  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# IndicConformer: --language is required and selects the token range
build-parakeet/parakeet \
  --model engines/parakeet/models/indic-conformer-600m-multilingual.q8_0.gguf \
  --wav engines/parakeet/test/samples/hi-16k.wav --language hi

# Nemotron: locale alias or auto; empty also selects auto
build-parakeet/parakeet \
  --model engines/parakeet/models/nemotron-3.5-asr-streaming-0.6b.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav --language en-US
```

`live-mic` runs CTC/RNN-T/TDT/EOU/Nemotron transcription or Sortformer diarization.
`live-mic-attributed` combines a CTC/RNN-T/TDT/EOU ASR model with Sortformer.
Both capture 16 kHz mono through miniaudio, accept independent streaming and
backend options, and finalize tail audio on Ctrl-C.

## Memory-fit preflight (`parakeet-fit-params`)

`parakeet-fit-params` projects whether a model fits the device memory
available right now, **without loading any weights**: it reads only GGUF
metadata, drives the same loader/graph builders as a real run through ggml's
size-only allocator APIs, and compares the projection against the device's
free memory. Library API in `<parakeet/fit.h>` (`parakeet::fit_params`);
status semantics follow the SDK's `@qvac/model-fit` contract
(0 = fits, 1 = does not fit, 2 = error — also the exit code).

```bash
# CPU projection for the default 300 s workload
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf

# GPU projection, machine-readable
build-parakeet/parakeet-fit-params \
  --model engines/parakeet/models/parakeet-tdt-0.6b-v3.q8_0.gguf \
  --n-gpu-layers 1 --json
```

The workload (`--audio-seconds`, default 300) sets the longest single
transcribe input to project for. Device-side memory is bounded by the
long-form encoder window, so the device projection saturates once the audio
exceeds one window; host-side buffers (full-input mel, stitched encoder
output) keep growing with it. The projection prices the encoder graph
*cache*, not one graph: a windowed pass keeps up to three window graphs
resident (`run_encoder`'s 3-slot LRU), and all of them are counted.
Sortformer diarization does not window — its device projection grows with
the full input (`O(T^2)` head attention).

Nemotron is fully modelled, and both of its exceptional properties are
priced: its device projection also grows with the audio (the locale-prompt
projection graph runs over the **full** stitched encoder output, not per
window), and its native cache-aware streaming is projected explicitly — the
result covers one offline transcribe **plus** one live streaming session
(step graph, per-chunk pre-encode graph, per-layer channel/time caches and
the other host-resident session state). `--nemotron-chunk-ms` selects the
streaming operating point to project (one of the GGUF's allowed values,
80/160/320/560/1120 on the shipped checkpoint); the default projects the
largest operating point, which bounds every other.

The projection is exact where it can be: `test-fit-params` asserts the
projected weight, encoder-compute, Sortformer-head, and Nemotron
prompt/step/pre-encode bytes equal what a real load/encode/diarize/stream
allocates, byte for byte. For the legacy families (CTC/RNN-T/TDT/EOU) the
projection covers the offline paths; their streaming sessions build smaller
per-chunk graphs but rotate through the same graph cache with
session-dependent keys, so treat the offline projection as a guide, not a
proven bound, for streaming. Nemotron streaming is a modelled part of the
projection, not a guide.
