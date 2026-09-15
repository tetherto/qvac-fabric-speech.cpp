# tts engine: fixtures and validation

Part of the [tts engine documentation](../README.md).

## Fixtures and static validation

The checkout includes the model-free voice-clone metric fixtures under
`test/fixtures/voiceclone/v1` and the public-domain JFK voice reference under
`test/reference-audio`. Model-dependent fixtures remain optional:

```bash
python scripts/dump-s3gen-reference.py --out artifacts/s3gen-ref
python scripts/dump-supertonic-reference.py \
  --onnx-dir /path/to/supertonic/onnx \
  --assets-dir /path/to/supertonic/assets \
  --voice-style /path/to/supertonic/assets/voice_styles/M1.json \
  --lang en --out artifacts/supertonic-ref-quick
```

For Supertonic 3, use the same existing dumper interface once per language,
pointing `--onnx-dir`, `--assets-dir`, and `--voice-style` at the v3 bundle:

```bash
python scripts/dump-supertonic-reference.py \
  --onnx-dir /path/to/supertonic-3/onnx \
  --assets-dir /path/to/supertonic-3/assets \
  --voice-style /path/to/supertonic-3/assets/voice_styles/M1.json \
  --lang en --out artifacts/supertonic3-ref-en
```

The current dumper accepts the legacy five-language `--lang` choices. Generate
`en`, `ko`, `es`, `pt`, and `fr` for the registered Supertonic 3 parity suite;
the model-free `test-supertonic-languages` target covers the complete v3
language registry.

For quantized Parler inspection, report-only mode is an environment variable
on the individual harness executable:

```bash
PARLER_TEST_REPORT_ONLY=1 ./build/test-parler-t5 MODEL.gguf REF_DIR
PARLER_TEST_REPORT_ONLY=1 ./build/test-parler-decoder MODEL.gguf REF_DIR
```

Do not treat `ctest -N` or disabled fixture registrations as executed tests.

### Optional: validate against Python references

Every stage of the pipeline has a numerical regression test against
Python-dumped reference tensors:

```bash
./build/test-s3gen models/chatterbox-s3gen.gguf artifacts/s3gen-ref ALL
```

Expected output (rel error per stage):

```
Stage A  speaker_emb_affine    rel ≈ 1e-7
Stage B  input_embedded        rel = 0
Stage C  encoder_embed         rel ≈ 4e-7
Stage D  pre_lookahead         rel ≈ 3e-7
Stage E  enc_block0_out        rel ≈ 1e-7
Stage F  encoder_proj (mu)     rel ≈ 5e-7
Stage G1 time_mixer            rel ≈ 7e-7
Stage G2 cfm_resnet_out        rel ≈ 3e-7
Stage G3 tfm_out               rel ≈ 2e-7
Stage G4 cfm_step0_dxdt        rel ≈ 1e-6
Stage H1 f0                    rel ≈ 4e-6
Stage H3 conv_post             rel ≈ 6e-7
Stage H4 stft                  rel ≈ 8e-3 (boundary-bound)
Stage H5 waveform              rel ≈ 1e-4
```

For T3 bit-exact validation against the Python reference:

```bash
python scripts/reference-t3-turbo.py \
  --text "Hello from ggml." \
  --out-dir artifacts \
  --cpp-bin ./build/tts-cli \
  --cpp-model models/chatterbox-t3-turbo.gguf
```
