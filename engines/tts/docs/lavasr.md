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

### Native CPU, OpenCL and Hexagon checks

`lavasr-bench` selects the neural backend explicitly and can run either stage
independently. `--stage pipeline` runs denoising before enhancement. An explicit
backend request fails if the selected device is unavailable; it cannot report a
CPU fallback as an HTP/OpenCL result.

```bash
./build/lavasr-bench --stage denoiser --backend cpu \
    --denoiser models/lavasr-denoiser-f32.gguf --in noisy.wav \
    --verify --runs 3 --warmup 1 --json-out denoiser-cpu.json \
    --out-denoised denoised.wav

./build/lavasr-bench --stage enhancer --backend opencl \
    --enhancer models/lavasr-enhancer-f32.gguf --in noisy.wav \
    --verify --runs 3 --warmup 1 --json-out enhancer-opencl.json \
    --out-enhanced enhanced.wav

GGML_HEXAGON_NDEV=1 GGML_HEXAGON_HOSTBUF=0 ./build/lavasr-bench \
    --stage pipeline --backend hexagon \
    --denoiser models/lavasr-denoiser-f32.gguf \
    --enhancer models/lavasr-enhancer-f32.gguf --in noisy.wav \
    --verify --runs 3 --warmup 1 --json-out pipeline-hexagon.json \
    --out-denoised htp-denoised.wav --out-enhanced htp-enhanced.wav
```

The Hexagon build must include the matching DSP skel and native backend
libraries in its deployment. `--backend hexagon` pins `HTP0`. Legacy
`--dn-gpu-layers`/`--n-gpu-layers` flags and `TTS_CPP_GPU_BACKEND` selection
remain available with `--backend auto`. The existing automatic GPU preference
is preserved; Hexagon is an explicit engine capability.

For the F32 Hexagon checks, use `qvac-ext-ggml@speech` revision
`380c5f9f3e1811dfd88adeb1c19ab887bcf8ab3e`, including its speech kernels and
F32-precision matmul routing. Build that revision with Hexagon enabled, use its
installed CMake package through `CMAKE_PREFIX_PATH`, and deploy the DSP skel
from the same build. The older bundled pin in `scripts/setup-ggml.sh` is not
the dependency used for these Hexagon checks. Record the resolved revision
when using another shared GGML build.

Both F32 and F16 GGUF inputs are supported. The current loaders expand F16
weights to F32, and the neural graphs retain F32 tensors and F32-precision
matrix multiplication. On Hexagon those products use HVX rather than F16 HMX.
An F16 filename does not indicate F16 execution. Hexagon's denoiser graph uses
F32 im2col/matmul convolution, sum/scale means and standard recurrent operations;
its enhancer uses direct depthwise convolution. Each graph is checked against
the selected backend before execution. Resampling, STFT/mel features, denoiser
log features, overlap-add, ISTFT and the FastLR merge remain host audio DSP.

Constructor time, the first processing call, additional warmups and measured
calls are recorded separately. Each measured call reports denoiser, enhancer
and combined seconds. RTF is processing seconds divided by input audio seconds;
lower is faster. Native processing includes the host audio DSP and graph
construction/allocation. WAV I/O, reference verification, constructor/first-call
and warmups are excluded from the steady measurements. Output WAVs use float32
PCM, preserving the measured samples without int16 clipping or quantization.

`--verify` compares denoising against the scalar reference and enhancement
against ggml CPU on the exact same input. The combined mode also compares the
full output to an independently CPU-processed pipeline. Length mismatches,
non-finite output and numerical failures return a nonzero status. JSON includes
actual backend names, graph/precision policy, rates/sample counts, every timing
sample and numerical metrics. Passing the command is correctness evidence;
actual DSP execution still needs a returned Hexagon profile or trace.

For a sustained run, `--min-seconds 300 --runs 1` continues measuring in the
same warmed process until the sum of native processing times reaches at least
five minutes. The final call may cross the target. The JSON records every call
and the actual measured duration. Collect battery/thermal state separately and
match starting conditions before comparing accelerator throughput changes.
