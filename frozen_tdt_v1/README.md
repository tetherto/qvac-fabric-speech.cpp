# Frozen TDT hybrid Core ML candidate

The retained candidate is in `experimental-hybrid-fp16/`. The source checkpoint
is `model.pt` (`parakeet_affine_packed_v1`, 427,560,592 bytes). Its first six
encoder blocks have INT8 affine projections; the remaining eighteen have
INT4 projections. Other source tensors include floating-point weights.

## Conversion and preservation

1. Reconstruct the checkpoint's tested FP16 values from its original codes,
   channel minima and steps. The full F16 GGUF (1,407,544,192 bytes) contains
   the encoder and is retained only as a correctness reference.
2. Write an encoder-free deployment GGUF containing the predictor, joint,
   preprocessing data, tokenizer and metadata (38,053,664 bytes).
3. Build the hybrid Core ML encoder: retain 216 original code arrays with
   exact per-channel FP16 lookup tables (162 INT4 and 54 INT8). Store the
   other 72 convolution weights as their exact reconstructed FP16 values.
   No k-means or new quantization is used. Those 72 tensors preserve values,
   but no longer preserve packed storage.
4. Use explicit FP16 computation and disable convolution optimizations that
   would change stored weights, including batch-normalization fusion.
   Compile the package to a Core ML sidecar.

The serialized hybrid's 288 affine tensors were verified bitwise. This is
not a claim that every runtime output is identical: on JFK, encoder relative
L2 difference against the PyTorch reference was 2.85%, with the same transcript.
Broader recognition accuracy is unmeasured. The Core ML compute plan preferred
Neural Engine for all 1,385 reported operations; that is a placement plan,
not a measured breakdown of hardware execution.

## Reproduce

Use Python 3.11, torch 2.7.0, numpy 1.26.4, coremltools 9.0, gguf 0.19.0,
PyYAML and SentencePiece. The lookup operations target macOS 15 / iOS 18;
only macOS execution has been tested.

```sh
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --packed-dir frozen_tdt_v1 --quant f16 \
  --out frozen_tdt_v1/frozen-tdt-v1.f16.gguf

python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --packed-dir frozen_tdt_v1 --quant f16 --encoder-coreml-only \
  --out frozen_tdt_v1/experimental-hybrid-fp16/frozen-tdt-v1.coreml-only.f16.gguf

python engines/parakeet/scripts/export-packed-encoder-coreml.py \
  --packed-dir frozen_tdt_v1 --gguf frozen_tdt_v1/frozen-tdt-v1.f16.gguf \
  --n-mel-frames 1501 --dense-conv-diagnostic --fp16-no-fuse \
  --out frozen_tdt_v1/experimental-hybrid-fp16/frozen-tdt-v1.coreml-only-encoder.mlpackage \
  --compile-dir frozen_tdt_v1/experimental-hybrid-fp16

python engines/parakeet/scripts/verify-packed-coreml-weights.py \
  --package frozen_tdt_v1/experimental-hybrid-fp16/frozen-tdt-v1.coreml-only-encoder.mlpackage
```

## Engine contract

Build with `PARAKEET_COREML=ON`. Keep the compiled encoder next to the
encoder-free GGUF with the matching filename stem. The new `coreml-required`
metadata records the source checkpoint SHA-256. The loader checks the sidecar's
identity and shape and fails if it is missing, incompatible or disabled.
Encoder execution errors do not fall back to an absent ggml encoder.
Unsupported streaming modes fail explicitly. Longer offline audio uses the
existing windowed encoder path (capacity 1501 mel frames, about 15 seconds).
The separate full-model mode retains its existing optional Core ML fallback.
Packed conversion defaults to F16 and requires explicit permission for a
second quantization through `--allow-requantize-packed`.

## Deployment size

| Artifact | Bytes |
| --- | ---: |
| Encoder-free GGUF | 38,053,664 |
| Source Core ML package | 548,519,552 |
| Compiled Core ML sidecar | 548,560,552 |
| GGUF + compiled sidecar | **586,614,216** |
| GGUF + source package | **586,573,216** |

The package and compiled sidecar are alternative distribution forms, not
additive requirements. Directory sizes are logical file bytes, before archive
compression. Keep the reference F16 GGUF out of the deployment download.

Fresh JFK (10 warmups + 20 runs) and test.wav (1 warmup + 3 runs) measurements
are in `benchmark-hybrid/`. Loading, inference and process memory are reported
separately. The previous Core ML packages and HTML reports were removed;
source data and historical raw benchmark results remain available.
