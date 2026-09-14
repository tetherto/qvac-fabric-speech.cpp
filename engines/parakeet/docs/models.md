# parakeet engine: models and conversion

Part of the [parakeet engine documentation](../README.md).

## Models and conversion

The engine runs GGUF, not `.nemo`. `scripts/download-all-models.sh` downloads
NeMo archives only; every downloaded checkpoint must still be converted.
Python is required only for conversion, Core ML export, and NeMo parity tooling;
runtime inference remains pure C++.

Create an isolated environment using a Python version supported by the selected
NeMo release:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch gguf numpy pyyaml soundfile librosa sentencepiece \
  "nemo_toolkit[asr]" huggingface_hub
```

Convert the downloaded checkpoint:

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/parakeet-ctc-0.6b.nemo \
  --out engines/parakeet/models/parakeet-ctc-0.6b.q8_0.gguf \
  --quant q8_0
```

The converter defaults to CTC 0.6B and `q8_0`. When `--out` is omitted, it
derives `models/parakeet-ctc-0.6b.<quant>.gguf` from `--quant`. For every other
checkpoint, pass an explicit `--ckpt`, `--hf-repo`, and `--out`; otherwise a
missing local checkpoint can cause the default CTC repository to be downloaded.
Keep the quantization in explicit filenames:

```text
<model>.q8_0.gguf
<model>.f16.gguf
```

Use f16 for numerical parity against NeMo references and q8_0 for normal runtime
fixtures. Small tensors or dimensions unsuitable for block quantization remain
f16. Hybrid RNNT+CTC checkpoints export their CTC branch by default; pass
`--head rnnt` to export the Transducer branch instead. RNN-T conversion rejects
checkpoints whose joint output is not exactly vocabulary plus blank, preventing
a duration-bearing TDT head from being mislabeled as plain RNN-T.

Recorded CTC 0.6B quantization results on an M4 Air CPU:

| Quantization | Size | 20-second encoder | 11-second encoder | Transcript |
|---|---:|---:|---:|---|
| f32 | 2.4 GiB | n/a | n/a | exact |
| f16 | 1.3 GiB | 1221 ms | ~680 ms | bit-equal |
| q8_0 | 697 MiB | 839 ms | 460 ms | bit-equal |
| q5_0 | 453 MiB | 1475 ms | ~650 ms | bit-equal |
| q4_0 | 372 MiB | 1080 ms | 595 ms | bit-equal |

Every supported checkpoint is covered by this local pipeline:
`scripts/download-all-models.sh` fetches the `.nemo` archive (including the
AI4Bharat IndicConformer hybrid) and `scripts/convert-nemo-to-gguf.py` turns it
into the runnable GGUF. The IndicConformer conversion emits the per-language
token ranges the run-time `--language` selection needs:

```bash
python engines/parakeet/scripts/convert-nemo-to-gguf.py \
  --ckpt engines/parakeet/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo \
  --out engines/parakeet/models/indic-conformer-600m-multilingual.q8_0.gguf \
  --quant q8_0
```

To select the same checkpoint's RNN-T branch, add `--head rnnt` and use a
distinct output filename. The auxiliary `ctc_decoder.*` tensors are then
ignored.
