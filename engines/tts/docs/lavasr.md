# tts engine: LavaSR enhancement

Part of the [tts engine documentation](../README.md).

## LavaSR enhancement

LavaSR is post-processing for synthesized or captured speech, not another
text-to-speech engine. The denoiser uses a 16 kHz internal STFT and preserves
the requested output rate; the bandwidth-extension enhancer emits 48 kHz.
Public APIs are installed under `<tts-cpp/lavasr/>`, and `lavasr-bench`
exercises the denoiser, enhancer, or two-stage pipeline. Backend support differs
between the two stages, so use the capability matrix rather than assuming every
compiled ggml backend applies to both.

### Convert

`scripts/download-lavasr-onnx.sh` fetches the LavaSRcpp ONNX release assets
the two converters read (the original torch weights live at
[YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)); both converters
accept `--ftype {f32,f16}`:

```bash
scripts/download-lavasr-onnx.sh --dir checkpoints/lavasr

python3 scripts/convert-lavasr-denoiser-to-gguf.py \
    --denoiser checkpoints/lavasr/denoiser_core_legacy_fixed63.onnx \
    --out models/lavasr-denoiser-f32.gguf
python3 scripts/convert-lavasr-enhancer-to-gguf.py \
    --backbone checkpoints/lavasr/enhancer_backbone.onnx \
    --spec-head checkpoints/lavasr/enhancer_spec_head.onnx \
    --out models/lavasr-enhancer-f32.gguf
```

### Run

`lavasr-bench` runs the two-stage pipeline over a wav and can write both
stage outputs:

```bash
./build/lavasr-bench --denoiser models/lavasr-denoiser-f32.gguf \
                     --enhancer models/lavasr-enhancer-f32.gguf \
                     --in noisy.wav \
                     --out-denoised denoised.wav --out-enhanced enhanced-48k.wav
```

`test-lavasr-gguf-load` needs no fixtures: it synthesizes tiny metadata-only
GGUFs and asserts both stage loaders fail closed on a missing, empty,
truncated, unmarked, cross-architecture, or tensorless file. The denoiser and
enhancer GGUFs differ only by `general.architecture`, so each loader has to
refuse the other's file.
