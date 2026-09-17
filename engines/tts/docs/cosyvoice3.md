# tts engine: CosyVoice3

Part of the [tts engine documentation](../README.md).

## CosyVoice3

CosyVoice3 runs a Qwen2.5 speech-token LM, DiT conditional-flow-matching
network, and CausalHiFT vocoder through `tts_cpp::cosyvoice::Engine`. The
validated backends are CPU, Metal (macOS / iOS), OpenCL (Adreno), and Vulkan
(Linux / Windows desktop); `n_gpu_layers > 0` selects the GPU path via the
engine's backend requirement and other GPU backends fall back to CPU. On
Android the requirement stays Metal-or-OpenCL, so Vulkan-only mobile GPUs
(e.g. Mali, Xclipse) keep declining to CPU rather than running unvalidated.
`--vulkan-device N` pins the Vulkan adapter on multi-GPU hosts (`-1`
auto-picks the discrete card with the most free VRAM).

Use `cosyvoice-cli` for end-to-end synthesis. The `cosyvoice-hift`,
`cosyvoice-flow`, and `cosyvoice-llm` executables isolate stages, and
`cosyvoice-bench` reports per-stage timing. The streaming callback currently
chunks PCM only after the full utterance has been generated, so it preserves
the callback shape but does not reduce first-audio latency.

```bash
./build/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
  --text "Hello from CosyVoice3." --out out.wav
```

### Convert

Everything the engine loads converts from the official
[Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
checkpoint plus the 3D-Speaker CAM++ torch checkpoint;
`scripts/download-cosyvoice3-checkpoint.sh` fetches both (Hugging Face CLI:
`pip install -U huggingface_hub`). The LM/flow/HiFT converters need the
[Conversion tooling](build.md#conversion-tooling) environment with `torch`; the
tokenizer converter additionally needs `pip install s3tokenizer` (or
`--s3tokenizer-repo` at a checkout of it).

```bash
scripts/download-cosyvoice3-checkpoint.sh --dir checkpoints/cosyvoice3

python3 scripts/convert-cosyvoice3-llm-to-gguf.py \
    --llm checkpoints/cosyvoice3/llm.pt \
    --outfile cosyvoice3-llm-f32.gguf --dtype f32
python3 scripts/convert-cosyvoice3-flow-to-gguf.py \
    --flow checkpoints/cosyvoice3/flow.pt \
    --config checkpoints/cosyvoice3/cosyvoice3.yaml \
    --outfile cosyvoice3-flow-f32.gguf --dtype f32
python3 scripts/convert-cosyvoice3-hift-to-gguf.py \
    --hift checkpoints/cosyvoice3/hift.pt \
    --outfile cosyvoice3-hift-f32.gguf --dtype f32

# voice-cloning add-on (required only when reference_audio is used)
python3 scripts/convert-s3tokenizer-v3-to-gguf.py \
    --onnx checkpoints/cosyvoice3/speech_tokenizer_v3.onnx \
    --out cosyvoice3-s3tok-f16.gguf --outtype f16
python3 scripts/convert-campplus-to-gguf.py \
    --ckpt checkpoints/cosyvoice3/campplus_cn_common.bin \
    --out cosyvoice3-campplus-f32.gguf
```

The LM converter accepts `--dtype {f32,f16,q8_0,q4_0}`, the flow converter
`--dtype {f32,f16,bf16,q8_0,q4_0}`, and HiFT `f32`/`f16` (its f0 predictor
stays f32 either way). The recommended desktop GPU tier is LM `q8_0` + flow
`q8_0` + HiFT `f16`, except on Metal, where flow `f16` wins: the DiT there
is compute-bound at the GPU's f16 GEMM rate, so `q8_0` saves no time and
costs a little accuracy (M3 Ultra, pinned trajectory: flow+HiFT f16 1.71 s
vs q8_0 1.83 s). On CPU the float flow tiers beat `q8_0` once ggml is
built with tinyBLAS (`GGML_LLAMAFILE=ON`, the bundled-ggml default): flow
`bf16` is fastest on AVX512-BF16 hosts (Zen 4/5, recent Xeon), flow `f16`
elsewhere, both with HiFT `f16`. The CPU LM decode is weight-bandwidth
bound, so LM `q4_0` roughly halves it against `q8_0` (measured 7.5 ->
4.1 ms/token) where its output quality is acceptable. A quantized HiFT tier
would gain nothing: like the flow's `q8_0`, quantization applies only to 2-D
matmul weights, and the vocoder is convolutions end to end. Avoid LM `f16`:
the engine reads the embedding tables as f32, so a f16 LM is not loadable
today — use the quantized LM tiers instead.
In `q8_0`/`q4_0` mode the flow converter quantizes only the 2D matmul
weights (conv kernels, norms, biases, the token embedding and the baked
`rand_noise` stay float), and it always writes the per-block attention
projections pre-fused as one `to_qkv` tensor; the engine also still loads
older GGUFs with separate `to_q`/`to_k`/`to_v` tensors. `bf16` mode stores
those same 2D matmul weights as bf16 and the conv kernels as f16, keeping
the kernel-typed f16 im2col path. Every reduced-precision tier is gated
against the f32 reference by `test-cosyvoice-{flow,hift}-tier-*`, which
needs only the two GGUFs staged (no PyTorch fixture). Measured deviations
on the gate's synthetic inputs (mel cosine / max abs): flow `f16`
0.99996 / 0.25, `bf16` 0.99967 / 0.99, `q8_0` 0.99983 / 0.66, `q4_0`
0.98659 / 4.84 — prefer `q8_0` over `q4_0` where the flow size allows.
The HiFT leg (`f16` waveform cosine 0.999966, max abs 0.0012) pins f0 so
the gate measures weight precision rather than sine-phase noise. The
thresholds registered in CMakeLists.txt carry margin over these values.

The LM converter applies the same attention fusion to each layer
(`qkv_proj`, rows q ++ k ++ v): one matvec feeds all three heads per decode
step. Row-wise quantization makes the fused tensor bit-identical to the
separate ones, and the engine still loads older GGUFs with separate
`q/k/v_proj` tensors.

`cosyvoice-cli --flow-cut-prompt` enables an opt-in flow shortcut that treats
the voice-prompt frames as attention conditioning only (the same design
cosyvoice.cpp uses by default). It cuts DiT time by roughly the prompt's share
of the mel sequence but deviates from the PyTorch reference, so it is off by
default; `test-cosyvoice-flow-cut` pins its output against the reference mel
with a looser bound.

The engine will not construct without a baked default voice (`voice.gguf`): it
packs the four prompt tensors of one reference utterance, computed on the
upstream PyTorch stack. The dump scripts load the upstream CosyVoice3 Python
model, so they need a full Hugging Face snapshot (including `campplus.onnx`
and the yaml/config set — the converter inputs fetched above are not enough)
plus a [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) checkout with its
dependencies on `PYTHONPATH`:

```bash
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir checkpoints/Fun-CosyVoice3-0.5B

PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \
python3 scripts/dump-cosyvoice3-reference.py \
    --model-dir checkpoints/Fun-CosyVoice3-0.5B \
    --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --out-dir artifacts/cv3-ref
PYTHONPATH=CosyVoice:CosyVoice/third_party/Matcha-TTS \
python3 scripts/dump-cosyvoice3-llm-reference.py \
    --model-dir checkpoints/Fun-CosyVoice3-0.5B \
    --prompt-audio CosyVoice/asset/zero_shot_prompt.wav \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --out-dir artifacts/llm-ref

python3 scripts/bake-cosyvoice3-voice.py \
    --prompt-stok  artifacts/llm-ref/prompt_stok.npy \
    --prompt-token artifacts/cv3-ref/prompt_token.npy \
    --prompt-feat  artifacts/cv3-ref/prompt_feat.npy \
    --embedding    artifacts/cv3-ref/embedding.npy \
    --prompt-text "希望你以后能够做的比我还好呦。" \
    --outfile voice.gguf
```

Any 5-15 s reference clip works in place of `zero_shot_prompt.wav`; pass its
verbatim transcript as `--prompt-text`. Assemble the model directory the
engine scans:

```bash
python3 scripts/assemble-cosyvoice3-model.py \
    --llm cosyvoice3-llm-f32.gguf --flow cosyvoice3-flow-f32.gguf \
    --hift cosyvoice3-hift-f32.gguf --voice voice.gguf \
    --vocab checkpoints/cosyvoice3/CosyVoice-BlankEN/vocab.json \
    --merges checkpoints/cosyvoice3/CosyVoice-BlankEN/merges.txt \
    --s3tok cosyvoice3-s3tok-f16.gguf --campplus cosyvoice3-campplus-f32.gguf \
    --out models/cosyvoice3-0.5b
```

### Voice cloning

With no reference audio the engine speaks with the baked default voice
(`voice.gguf`). Zero-shot / cross-lingual cloning runs fully natively — the
reference wav is tokenized by a ggml port of `speech_tokenizer_v3` (12-block
FSMN encoder + FSQ, converted by `scripts/convert-s3tokenizer-v3-to-gguf.py`),
the 192-d speaker embedding comes from the CAM++ port
(`scripts/convert-campplus-to-gguf.py`; the same 3D-Speaker checkpoint
CosyVoice ships as `campplus.onnx`), and the prompt mel from the shared
matcha-compatible mel extractor. Both GGUFs are required whenever
`reference_audio` is set (`cosyvoice3-s3tok*.gguf`,
`cosyvoice3-campplus*.gguf`, auto-resolved from the model dir); a missing
model, an unreadable wav, or an out-of-range duration (0.5-30 s hard limits,
5-15 s recommended) fails construction rather than silently keeping the baked
voice.

The reference transcript selects the mode, mirroring the upstream frontends:

```bash
# zero-shot: transcript given, LM prompted with the reference speech tokens
# (best fidelity when synthesizing the reference's own language)
./build/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
  --reference-audio me.wav --prompt-text "verbatim transcript of me.wav" \
  --text "Same language as the reference." --out out.wav

# cross-lingual: no transcript, timbre-only conditioning through the flow
./build/cosyvoice-cli --model-dir models/cosyvoice3-0.5b \
  --reference-audio me.wav --text "Any other language." --out out.wav
```

The bake runs once at engine construction (roughly a second of CPU for the
tokenizer + CAM++ + mel on a short clip; the tokenizer graph rides the
engine's GPU backend when one is selected). Instruct mode composes with a
cloned voice: the instruction drives dialect/style while the cloned tensors
supply the timbre.
