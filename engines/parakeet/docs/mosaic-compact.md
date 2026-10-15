# Parakeet-Mosaic Compact

Final recipe for this experiment: **503,670,112-byte GGUF**, SHA256 `612fcc711d453fd914b04022186f2d71904e094f931e5129fe64c8defd86ab93`.

The Parakeet TDT network is unchanged. 122 encoder matrices use standard Q4_0,142 use standard Q8_0, with32-weight groups. Decoder and other non-target payloads retain the historical Mosaic effective values, including quantized depthwise weights represented in FP16. This is a new effective-weight recipe relative to the610.6MB affine model; do not attribute its results to the old model or fused affine kernel. It uses ordinary GGML kernels and needs no affine opt-in.

## Reproduce

Use Python3.11, Torch2.7.0, NumPy2.2.6, gguf0.19.0, PyYAML6.0.3 and sentencepiece. Start from the original NVIDIA parakeet-tdt-0.6b-v3.nemo identified in the recipe. No company model weights are bundled here.

```sh
python scripts/export-mosaic-compact.py \
  --archive /path/to/parakeet-tdt-0.6b-v3.nemo \
  --output models/mosaic-compact.gguf
```

Run from engines/parakeet. The exporter verifies the source archive, historical recipe and all725 effective-tensor receipts before applying the frozen264-target block32 map. Target matrices are quantized from original FP16, not requantized affine-rounded matrices. It verifies final file size and full SHA256 against the shipped recipe. Separate Q/K/V tensors; no redundant fused-QKV payload. An existing output is never overwritten.

Native Metal testing sets `PARAKEET_COREML_DISABLE=1`. `PARAKEET_AFFINE_FUSED` is unnecessary for Compact. The610.6MB affine model remains available under its original recipe/exporter and benefits from the separate custom GGML operation.

## Optional matching Core ML encoder

Export from this Compact GGUF, not the older affine recipe or a registry encoder:

```sh
python scripts/export-encoder-coreml.py \
  --gguf models/mosaic-compact.gguf --n-mel-frames 512 --precision fp16 \
  --out models/mosaic-compact-encoder.mlpackage --compile-dir models
```

Use coremltools9.0 and Torch2.7.0. No additional palettization was tested for this final recipe. The graph uses mel[128,512] and mel_valid_mask[512]. Remove `PARAKEET_COREML_DISABLE` from the environment entirely to enable automatic sidecar loading: value0 still disables it because the engine checks presence. `PARAKEET_COREML_COMPUTE_UNITS=ALL` was measured; that does not prove ANE placement. The native frontend and Metal decoder stay in speechcpp.

The measured compiled bundle is1,259,381,354bytes, giving a1,763,051,466-byte shipping pair. Compiler metadata/size may vary by host. Source mlpackage is optional for deployment. The loader retains the GGUF encoder, so duplicate storage/residency remains. Never reuse a sidecar from another weight recipe.

## Expanded development evidence — Mac mini M4

Q8 is the primary control; Q5/Q4 are soft context, not rejection gates. Native256clean,256difficult and256paired0dB noisy complete recordings; three fresh-process repetitions, MTL0/fourthreads. Full file-to-text wall excludes loading, compilation, eight warmups and scoring. Selection seed2703 excludes32prior tuning IDs. This is development validation, not held-out qualification.

| Condition | Q8 WER | Compact WER | Q8 median seconds | Compact median seconds |
|---|---:|---:|---:|---:|
| Clean |2.032%|2.239%|20.176|21.408|
| Difficult |4.040%|4.062%|17.772|18.552|
|0dB noise|10.647%|11.287%|20.286|21.593|

Compact is24.76% smaller than669,407,392-byte Q8; median paired latency is6.6–7.0% higher. Clean/difficult reader-cluster WER intervals crosszero (not equivalence); noisy WER delta+0.640pp has95% interval[+0.115,+1.206]. Phase wall clocks locate the remaining gap in decoding; these are not GPU operator traces. The isolated16-step GPU decoder unroll probe preserved transcripts but missed its useful speed threshold and is NOT included.

Separate CoreML scope: ALL complete0.8–4.8second recordings within the frozen selection,99clean/111difficult/99noisy, with fresh native controls and three repetitions:

| Condition | Q8 native WER | Compact+CoreML WER | Q8 median seconds | Hybrid median seconds |
|---|---:|---:|---:|---:|
|Clean|2.590%|2.590%|4.452|3.366|
|Difficult|4.382%|4.084%|4.835|3.731|
|0dB noise|18.919%|19.257%|4.505|3.477|

Hybrid median paired latency falls22.8–24.5% versus nativeQ8. Do not generalize to full-length/long-form audio, infer a fresh Redux result, or claim superior accuracy from tiny aggregate differences. Three timing repetitions are not confidence intervals or extra accuracy samples. Repeated raw transcripts identical, no blanks/unknowns. Native process-lifetime RSS Compact0.655–0.675GB versusQ8 0.812–0.824GB; hybrid0.832–3.107GB. RSS includes startup and does not fully observe driver/ANE residency.

Research evidence: parakeet-research commit a47d333, reports/plan_v3/q8_validation_002. Independent audits rescore4,608native+2,781hybrid predictions, reconstruct768audiofiles, verify12bootstrap intervals and653GGUF tensor dequantizations and compiled/source package hashes. Initial environment routing failure and failed decoder probe retained. No automatic production qualification or changes to historical failed gates.
