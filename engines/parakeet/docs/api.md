# parakeet engine: C++ API

Part of the [parakeet engine documentation](../README.md).

## Public C++ API

Include `<parakeet/parakeet.h>` for the complete surface or individual headers.

| API | Purpose |
|---|---|
| `Engine::transcribe` | One-shot WAV transcription for CTC, RNN-T, TDT, EOU, or Nemotron |
| `Engine::transcribe_samples` | One-shot float PCM transcription |
| `Engine::transcribe_stream` / `transcribe_samples_stream` | Chunked callbacks; one offline encode for legacy families, native cache-aware streaming for Nemotron |
| `Engine::stream_start` | Live session; rolling-window re-encoding for legacy families, native cache-aware streaming for Nemotron |
| `StreamSession::feed_pcm_f32` / `feed_pcm_i16` | Push arbitrary-sized PCM blocks |
| `Engine::diarize` / `diarize_samples` | Offline Sortformer diarization |
| `Engine::diarize_start` | Live Sortformer session |
| `transcribe_with_speakers` / `transcribe_samples_with_speakers` | CTC, RNN-T, TDT, or EOU transcription attributed with Sortformer |
| `EngineOptions::prewarm` / `prewarm_audio_seconds` | Run a configurable encoder-only synthetic forward during construction |
| `parakeet_log_set` | Install the host logging sink |
| `parakeet_cli_main` | Embed the same CLI entry point exported by the library |
| `fit_params` | Memory-fit preflight: project whether a GGUF + workload fits the device without loading weights (`<parakeet/fit.h>`) |
| `parakeet_fit_cli_main` | Embed the `parakeet-fit-params` tool entry point |

Minimal one-shot transcription:

```cpp
#include <parakeet/parakeet.h>

#include <cstdio>

int main() {
    parakeet::EngineOptions options;
    options.model_gguf_path = "models/parakeet-ctc-0.6b.q8_0.gguf";

    parakeet::Engine engine(options);
    const auto result = engine.transcribe("audio.wav");
    std::puts(result.text.c_str());
}
```

Mode 3 is duplex rolling-context/sliding-window re-encoding for CTC, RNN-T, TDT,
and EOU. It re-runs the encoder for each window using `left_context_ms` and
`right_lookahead_ms`; it does not maintain an encoder KV cache or convolution
cache. Nemotron Mode 3 uses the cache-aware encoder instead and ignores those
window knobs.

Always call `finalize()` to drain the partial tail. Destroying a
`StreamSession` or `SortformerStreamSession` cancels it and does not finalize
it. `cancel()` stops future work; `Engine::cancel()` may be called from another
thread, but inference calls on one Engine are otherwise not concurrent-safe.
Cancel, join the worker, and only then destroy an Engine; destruction does not
wait for in-flight calls.

`StreamingCallback` and `StreamEventCallback` run synchronously from the API
call processing the audio. `StreamingSegment::is_eou_boundary` and
`StreamEventType::EndOfTurn` mean the EOU model emitted `<EOU>`. They are not
VAD state changes. Sortformer emits `VadStateChanged` from speaker
probabilities. CTC/RNN-T/TDT/Nemotron can opt into RMS energy VAD with
`enable_energy_vad`; configure it with `energy_vad_threshold_db`,
`energy_vad_window_ms`, and `energy_vad_hangover_ms`.

Long-form one-shot transcription uses bounded encoder windows when
`long_form_window_frames` is exceeded and trims
`long_form_context_frames` at seams. Short inputs retain the single-pass path.

## Sortformer AOSC

AOSC is primarily activated by the explicit GGUF metadata
`parakeet.model_variant=sortformer-streaming-v2.1-aosc`. Encoder shape
`n_layers=17` and `n_mels=128` is only a legacy fallback for GGUFs created
before that metadata key existed. Convert current v2.1 checkpoints again to
avoid relying on the heuristic.

`SortformerStreamingOptions` exposes `spkcache_enable`, `spkcache_len`,
`fifo_len`, `chunk_left_context_ms`, `chunk_right_context_ms`, and
`spkcache_update_period`. After `diarize_start()`,
`SortformerStreamSession::aosc_active()` reports whether the session actually
took the AOSC path.

Every non-cancelled `finalize()` drains any trailing partial chunk and then
emits exactly one final synthetic terminator (`speaker_id=-1`,
`is_final=true`, and `start_s == end_s`). Real speaker segments always remain
non-final. Repeated `finalize()` calls are idempotent, while cancellation
suppresses the terminator.
