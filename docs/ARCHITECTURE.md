# Architecture and repository layout

Part of the [speech-stack documentation](../README.md).

## Shared runtime

The umbrella resolves one installed speech-branch ggml before adding the
engines. Every enabled engine consumes that dependency; the ggml tree inside
the Whisper subtree is not compiled by the umbrella. Direct engine builds
may instead stage their own speech-branch checkout, as their build guides describe.

```text
Whisper          Parakeet          TTS          AudioGen
   |                 |             |              |
   +-----------------+-------------+--------------+
                     |
          ggml-speech (qvac-ext-ggml@speech)
                     |
       CPU / Metal / Vulkan / OpenCL / CUDA / Hexagon

Selected model stages may also use optional Apple Core ML sidecars.
```

Backend and sidecar support varies by model. The
[root capability overview](../README.md#supported-models) links to the engine
validation and routing contracts.

## Pipelines

| Engine | High-level flow | Details |
|---|---|---|
| Whisper | Audio → log-mel → encoder/decoder → text; optional VAD and encoder sidecar | [Whisper guide](../third_party/whisper.cpp/README.md) |
| Parakeet | Audio → features → model-specific encoder/head → text, speakers or turn boundary; MOSS uses a separate transcription engine | [Public APIs](../engines/parakeet/docs/api.md), [MOSS](../engines/parakeet/docs/moss-transcribe.md) |
| TTS | Text or speech → model-specific generation → acoustic decoder/codec → PCM; optional LavaSR enhancement | [Pipelines and APIs](../engines/tts/docs/api.md#pipelines) |
| AudioGen | Caption/lyrics or source audio → model-specific conditioning/generation → stereo PCM | [ACE-Step and MiniMax pipelines](../engines/audiogen/docs/pipeline.md) |

## Repository layout

| Path | Role |
|---|---|
| `CMakeLists.txt` | Feature-gated umbrella superbuild |
| `third_party/whisper.cpp/` | Upstream Whisper git subtree; release recorded in [UPSTREAM_PIN](../third_party/whisper.cpp/UPSTREAM_PIN), deliberate deltas in [PATCHES.md](../third_party/whisper.cpp/PATCHES.md) |
| `engines/parakeet/` | Native ASR, diarization, EOU and MOSS transcription |
| `engines/tts/` | Synthesis, cloning, sound effects, speech-to-speech and enhancement |
| `engines/audiogen/` | ACE-Step and MiniMax music generation/editing |
| `engines/*/docs/` | Engine build, model, API, CLI, backend and validation guides |
| `engines/*/docs/history/` | Archived implementation and integration records |
| `scripts/benchmarks/` | Shared performance/quality tooling and operational guides |
| `docs/UPSTREAM-SYNC.md` | Whisper subtree synchronization procedure |
