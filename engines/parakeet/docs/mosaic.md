# Mosaic

Mosaic is the production mixed-precision Parakeet TDT model (production manifest version `1`). Its standalone GGUF is **486,693,152 bytes**, SHA256 `9e17114b5773d1b1ed9b6a4fafb0f46427a3bb9a80b77e645c24576bda9188ef`.

The encoder contains 122 standard Q4_0 matrices and 142 standard Q8_0 matrices. Eight decoder matrices use Q8_0. The other 645 tensor payloads and all GGUF metadata remain byte-identical to the retained encoder-source artifact. Quantization uses the source's effective FP16 decoder values directly. No global requantization, network/topology change, dense inference file, custom affine kernel or affine opt-in is required.

## Export and deploy a native GGUF

From `engines/parakeet`, use Python 3.11 with Torch 2.7.0, NumPy 2.2.6, gguf 0.19.0, PyYAML 6.0.3 and sentencepiece 0.2.0:

```sh
python scripts/export-mosaic.py \
  --archive /path/to/parakeet-tdt-0.6b-v3.nemo \
  --output models/parakeet-mosaic.gguf
```

The original NVIDIA archive must have SHA256 `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`. The exporter reconstructs the exact retained encoder-source artifact, verifies its 503,670,112-byte identity, then changes only the eight decoder matrices. It verifies the final whole-file size/hash before producing a production receipt. Existing outputs and receipts are never overwritten. Historical encoder recipe/tensor receipt files remain immutable reconstruction inputs.

If the exact encoder-source artifact is already retained, the Torch-free decoder export is:

```sh
python scripts/export-mosaic.py \
  --source-gguf /path/to/verified-encoder-source.gguf \
  --output models/parakeet-mosaic.gguf
```

For a deployment directory, with native GGUF as the default:

```sh
python scripts/prepare-mosaic.py \
  --gguf models/parakeet-mosaic.gguf --output-dir models/mosaic
```

This copy/verification path needs no Core ML tooling and works on Linux. Inference uses the ordinary speechcpp Parakeet `Engine` API. Select `cpu`, `MTL0`, `CUDA0` or `ROCm0` in `EngineOptions.backend`, with the corresponding GGML backend compiled in. A GPU request must not silently fall back. Set `PARAKEET_COREML_DISABLE=1` for native deployment; the presence of this variable disables Core ML even when its value is `0`. `PARAKEET_AFFINE_FUSED` is unnecessary.

## Optional Apple encoder sidecar

`prepare-mosaic.py --coreml` explicitly generates the matching masked512 FLOAT16 encoder sidecar on macOS, using Torch 2.7.0 and coremltools 9.0. The outputs are `parakeet-mosaic.gguf` and `parakeet-mosaic-encoder.mlmodelc`; the source mlpackage is optional for deployment. Never reuse an encoder sidecar from a different recipe. Loading the full GGUF encoder alongside a sidecar duplicates weight storage/residency.

The opt-in integration lane checks complete short utterances, sidecar input/inventory hashes, actual per-call routing, expected text and repeat stability. This smoke coverage does not transfer the native accuracy qualification or long-form performance results to Core ML, nor prove ANE placement. Native-only export does not require the custom affine GGML API. The historical affine feature in the companion GGML PR remains separately opt-in and separately tested.

## Matched native measurements

17 October 2026, identical model/audio bytes across Apple M4 Metal, Radeon 8060S HIP and one RTX 5090 CUDA. 456 selection recordings (160 clean, 136 difficult, 160 paired noisy at 0 dB), two repetitions with reversed model order, four CPU threads, batch one, eight excluded warmups. Whole-call latency excludes startup; RSS includes it. These are selection-set point estimates, not held-out confirmation or timing confidence intervals.

| Backend | Mean ms/clip | RTF | Clean WER | Difficult WER | Noisy WER |
|---|---:|---:|---:|---:|---:|
| Apple M4 Metal | 71.25 | 0.010272 | 2.2748% | 3.6738% | 9.7875% |
| Radeon 8060S HIP | 30.59 | 0.004410 | 2.2448% | 3.7545% | 9.8474% |
| RTX 5090 CUDA | 9.20 | 0.001327 | 2.2448% | 3.7142% | 9.8174% |

The research runner independently audited 8,208 predictions across Mosaic and standard Q4/Q8 controls. Repeated transcripts within each host were identical; cross-backend floating-point results differ slightly. These runs used frozen research source snapshots, so they are context rather than qualification of a later PR head. Integration CI separately validates the production entry point and public Engine API at its recorded dependency revision.

The 610.6 MB packed-affine research artifact and its earlier performance tables are historical and use a different weight representation. They do not describe this production GGUF.
