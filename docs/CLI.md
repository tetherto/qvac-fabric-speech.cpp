# Command-line tools and examples

Part of the [qvac-fabric-speech.cpp documentation](../README.md).

## Command line tools

| Binary | Engine | Purpose |
|---|---|---|
| `whisper-cli` | whisper | transcribe and translate, with optional Silero VAD |
| `parakeet` | parakeet | transcribe, diarize, detect end-of-utterance, benchmark |
| `tts-cli` | tts | Chatterbox, Supertonic, and Parler synthesis, autodetected from GGUF metadata |
| `parler-cli` | tts | full Parler-TTS flag surface |
| `supertonic-cli` | tts | standalone Supertonic synthesis |
| `cosyvoice-cli` | tts | CosyVoice3 synthesis |
| `audio8-cli` | tts | Audio8 synthesis and zero-shot voice cloning |
| `music-cli` | audiogen | end-to-end text-to-music |
| `acestep-cli` | audiogen | Oobleck VAE decode and roundtrip harness |
| `acestep-quantize` | audiogen | requantize converted ACE-Step or MiniMax-Music3 stage GGUFs |
| `mm3-replay` | audiogen | MiniMax-Music3 generation and parity harness |
| `lavasr-bench` | tts | denoiser and enhancer benchmark |
| `mel2wav` | tts | HiFT mel to wav |

### Whisper

```sh
./third_party/whisper.cpp/models/download-ggml-model.sh base.en
./build/bin/whisper-cli -m third_party/whisper.cpp/models/ggml-base.en.bin \
                        -f third_party/whisper.cpp/samples/jfk.wav
```

### Parakeet

Models are converted from NeMo checkpoints with `download-all-models.sh` and
`convert-nemo-to-gguf.py`. The downloader covers every supported checkpoint,
including the AI4Bharat IndicConformer hybrid; see
[engines/parakeet/README.md](../engines/parakeet/README.md).

The EOU-only artifact path downloads the 120M checkpoint, creates and verifies
an F16 GGUF, creates the Q8_0 runtime GGUF, and compiles the fixed 1101-frame
Core ML sidecar on macOS/Xcode:

```sh
engines/parakeet/scripts/download-all-models.sh eou
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf --quant f16
python engines/parakeet/scripts/verify-gguf-roundtrip.py \
  --nemo engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --gguf engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet_realtime_eou_120m-v1.nemo \
  --hf-repo nvidia/parakeet_realtime_eou_120m-v1 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1.q8_0.gguf --quant q8_0
python engines/parakeet/scripts/export-encoder-coreml.py \
  --gguf engines/parakeet/models/parakeet_realtime_eou_120m-v1.f16.gguf \
  --wav engines/parakeet/test/samples/jfk.wav \
  --palettize-bits 6 --palettize-group-size 16 \
  --out engines/parakeet/models/parakeet_realtime_eou_120m-v1-encoder.mlpackage \
  --compile-dir engines/parakeet/models
```

EOU Core ML is correctness-first: calls use Core ML only at the sidecar's
exact mel-frame shape. Shorter inputs, longer offline inputs, startup/tail
streaming windows, and other mismatching calls fall back to ggml; EOU never
pads a short call to activate Core ML. Use
`PARAKEET_COREML_DISABLE=1` for a forced-ggml comparison, or run the
`parakeet-eou` desktop benchmark family for required-Core-ML versus Metal
timing and normalized JFK WER.

Hybrid RNNT+CTC checkpoints export CTC by default. Pass `--head rnnt` to export
their Transducer branch; conversion validates that the joint output is exactly
vocabulary plus blank. `dump-rnnt-reference.py` selects the same NeMo branch
for token-level parity testing.

```sh
# transcribe (the GGUF metadata selects CTC / RNN-T / TDT / EOU / Nemotron)
./build/engines/parakeet/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav

# transcribe with speaker attribution
./build/engines/parakeet/parakeet --model models/parakeet-tdt-0.6b-v3.q8_0.gguf \
                                  --diarization-model models/diar_sortformer_4spk-v1.f16.gguf \
                                  --wav engines/parakeet/test/samples/diarization-sample-16k.wav

# streaming end-of-utterance, JSONL events
./build/engines/parakeet/parakeet --model models/parakeet_realtime_eou_120m-v1.q8_0.gguf \
                                  --wav engines/parakeet/test/samples/jfk.wav \
                                  --stream --stream-chunk-ms 1500 --emit jsonl
```

### Text-to-speech

GGUF conversion steps and the umbrella/direct/vcpkg build-path matrix are in
[engines/tts/README.md](../engines/tts/README.md). The umbrella build enables this
package with `SPEECH_BUILD_TTS=ON`.

```sh
# Chatterbox Turbo, with voice cloning from a reference wav
./build/engines/tts/tts-cli --model      models/chatterbox-t3-turbo.gguf \
                            --s3gen-gguf models/chatterbox-s3gen.gguf \
                            --reference-audio me.wav \
                            --text "Hello from native C plus plus." --out out.wav

# Chatterbox Multilingual
./build/engines/tts/tts-cli --model      models/chatterbox-t3-mtl-q4_0.gguf \
                            --s3gen-gguf models/chatterbox-s3gen-mtl-q4_0.gguf \
                            --text "Hola, esto es una demostracion multilingue." \
                            --language es --cfm-steps 7 --out out.wav

# Supertonic, preset voice
./build/engines/tts/tts-cli --model models/supertonic2.gguf --voice M1 --language en \
                            --text "The quick brown fox jumps over the lazy dog." --out out.wav

# Parler-TTS, description-conditioned
./build/engines/tts/parler-cli --model models/parler-mini-v1-q8_0.gguf \
                               --description "A female speaker with a calm, clear voice, close up." \
                               --text "Hey, how are you doing today?" --out out.wav

# CosyVoice3
./build/engines/tts/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
                                  --text "Hello from a fully on-device pipeline." --out out.wav

# Audio8; drop --n-gpu-layers to stay on the CPU
./build/engines/tts/audio8-cli --lm models/audio8-lm-q8_0.gguf \
                               --codec-decoder models/audio8-codec-decoder-q8_0.gguf \
                               --text "Hello from Audio8." \
                               --n-gpu-layers 99 --out out.wav
```

`--emotion` and `--pace` work the same way on every engine that supports them;
each CLI lists its own supported values via `--list-emotions` / `--list-paces`.
See [Voice conditioning](../engines/tts/docs/voice-conditioning.md#voice-conditioning-cross-engine).

```sh
./build/engines/tts/parler-cli --model models/parler-indic-q8_0.gguf \
                               --emotion happy --pace moderate \
                               --text "आज मौसम बहुत अच्छा है।" --out out.wav

./build/engines/tts/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
                                  --emotion happy \
                                  --text "Hello from a fully on-device pipeline." --out out.wav
```

### Music generation

AudioGen uses four GGUF files for six runtime weight sets. The DiT file also
contains the FSQ detokenizer and condition encoder; see the
[AudioGen model setup](../engines/audiogen/docs/pipeline.md#model-setup) for the
validated file combinations and the download, conversion, and quantization
steps that produce them.

```sh
./build/engines/audiogen/music-cli --models models/acestep \
                                   --caption "driving synth pop, bright analog leads, 120 bpm" \
                                   --lyrics "[Instrumental]" --dur 8 --gpu --out song.wav
```
