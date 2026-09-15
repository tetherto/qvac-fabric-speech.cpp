# parakeet engine: tests and parity

Part of the [parakeet engine documentation](../README.md).

## Tests and NeMo parity

Set up fixtures in this order:

1. Download the required `.nemo` archive.
2. Convert it to the exact q8_0 runtime or f16 parity GGUF filename expected by
   `CMakeLists.txt`.
3. Generate NeMo `.npy` references with the matching `dump-*-reference.py`
   script when the test requires references.
4. Configure after fixtures exist so CTest registrations detect them.
5. Build, inspect `ctest -N`, then run CTest.

```bash
engines/parakeet/scripts/download-all-models.sh

python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet-ctc-0.6b.nemo \
  --out engines/parakeet/models/parakeet-ctc-0.6b.f16.gguf --quant f16

python engines/parakeet/scripts/dump-ctc-reference.py \
  --wav engines/parakeet/test/samples/jfk.wav

# Hybrid checkpoint: select its RNN-T branch in both conversion and NeMo.
python engines/parakeet/scripts/dump-rnnt-reference.py \
  --nemo-model engines/parakeet/models/stt_ka_fastconformer_hybrid_large_pc.nemo \
  --wav engines/parakeet/test/samples/rnnt-ka-16k.wav \
  --out engines/parakeet/artifacts/rnnt-ref

cmake -S engines/parakeet -B build-parakeet -DCMAKE_BUILD_TYPE=Release
cmake --build build-parakeet -j
ctest --test-dir build-parakeet -N
ctest --test-dir build-parakeet --output-on-failure
```

`download-all-models.sh` produces `.nemo` files, not runnable GGUFs. Missing
model, audio, or reference fixtures cause individual tests to be registered as
`DISABLED`, not failed. Configure again after adding a fixture. Each test
carries exactly one label out of `unit`, `fixture`, `cpu`, `gpu`, and `perf`.
The fixture-free suite (the one CI runs) is selected by excluding the
GPU-bound and timing-bound labels:

```bash
ctest --test-dir build-parakeet -LE 'gpu|perf' --output-on-failure
```

`test-rnnt-decoder-parity` is enabled when the hybrid RNN-T GGUF, its WAV, and
the NeMo `token_ids.npy` dump are available. It requires bit-exact token IDs
between the reference and the current shared RNN-T/TDT decoder.

Model-free logic tests (`-L unit`) cover the CTC language mask, mel FFT
parity and per-feature CMVN, RNN-T graph construction, long-form window
planning, Sortformer finalization and probability thresholding, the
streaming energy VAD, the SentencePiece detokenizer, and the NeMo
converter (Python).

Fixture roots are configurable with `PARAKEET_TEST_MODEL_DIR`,
`PARAKEET_TEST_AUDIO_DIR`, and `PARAKEET_TEST_REF_DIR`. `test-gpu-vs-cpu`
(formerly `test-vk-vs-cpu`) is available when Vulkan or Metal is configured.
`verify-gguf-roundtrip.py`,
`ref-encoder-from-gguf.py`, and `streaming-reference.py` support converter and
parity investigation.
