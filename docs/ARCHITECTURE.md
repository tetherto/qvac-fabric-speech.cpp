# Architecture, pipelines, and repo layout

Part of the [qvac-fabric-speech.cpp documentation](../README.md).

## Architecture

```
+-----------------------------+  +-----------------------------+
| third_party/whisper.cpp     |  | engines/parakeet            |
| speech-to-text              |  | ASR + diarization + EOU     |
+-----------------------------+  +-----------------------------+
+-----------------------------+  +-----------------------------+
| engines/tts                 |  | engines/audiogen            |
| TTS + cloning + enhancement |  | text-to-music               |
+-----------------------------+  +-----------------------------+
                    |                     :
                    v                     : optional encoder sidecar
   ggml-speech (qvac-ext-ggml@speech)     v
                    |                Apple Core ML
   +--------+-------+-------+---------+
   v        v       v       v         v
  CPU     Metal  Vulkan  OpenCL     CUDA
                        (Adreno)
```

Every component consumes one system ggml, so the whole stack shares a single ggml pin and file set. The `ggml/` tree vendored inside the whisper subtree is never compiled.

## Pipelines

```
whisper   wav  -> log-mel -> encoder -> decoder -> text            (+ Silero VAD, + Core ML encoder)
parakeet  wav  -> log-mel -> FastConformer encoder -> CTC | RNN-T | TDT | EOU | Nemotron | Sortformer
                                                   -> text | speaker segments | turn boundary
tts       text -> LM (T3 / Llama / Qwen2.5) -> acoustic tokens -> CFM or flow -> vocoder -> wav
                                                   (+ LavaSR denoise -> bandwidth extension)
audiogen  caption + lyrics -> ACE-Step LM -> FSQ detokenizer -> text encoder
                           -> condition encoder -> DiT flow matching
                           -> Oobleck VAE -> 48 kHz stereo
          short query -> LM inspire (Simple Mode) -> caption + lyrics + metadata
          caption + lyrics -> LM format (Query Rewriting) -> detailed request
                           -> same ACE-Step pipeline
          lyrics + generated audio -> DiT cross-attention probe -> DTW
                           -> synchronized LRC timestamps
          generated codes + request -> teacher-forced LM -> quality score
          audio -> VAE encode -> FSQ tokenize -> LM listener
                           -> metadata + caption + recovered codes
          caption + lyrics -> MiniMax Qwen3 LM -> RVQ depth decoder
                           -> condition encoder -> flow DiT -> vocoder -> stereo
```

## Repo layout

```
CMakeLists.txt              feature-gated umbrella superbuild
third_party/whisper.cpp/    upstream whisper.cpp, vendored as a git subtree,
                            pinned @ v1.9.1 (f049fff9); every QVAC delta is
                            declared in PATCHES.md and enforced by CI
engines/
  parakeet/                 ASR + diarization + end-of-utterance (NVIDIA Parakeet family)
  tts/                      text-to-speech, voice cloning, speech enhancement
  audiogen/                 music generation (ACE-Step, MiniMax-Music3)
docs/UPSTREAM-SYNC.md       how to sync the whisper subtree
```
