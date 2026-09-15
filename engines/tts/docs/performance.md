# tts engine: performance deep dive

Part of the [tts engine documentation](../README.md).

The authoritative Linux x86-64 numbers come from CI (table below); the
Apple-silicon figures are maintainer measurements (no CI Metal runner). Shared
setup for the Apple rows:

- Text: *"Hello from native C plus plus. This audio was generated end
  to end on CPU using ggml."*
- Reference voice: `test/reference-audio/jfk.wav` (11 s mono 16 kHz)
- Seed: 42, warm 3-run average, inference only (excludes model load)

### speech-cpp CI (2026-09-07, CPU + macOS)

Fresh CPU baseline from `speech-benchmark-desktop.yml` on the hosted-Linux and
self-hosted macOS runners (5 timed runs + 1 warmup); parler on macOS runs on
Metal (`MTL0`).

| Engine | Runner | Backend | Median wall ms | Median RTF | Peak RSS MiB |
|---|---|---|--:|--:|--:|
| Chatterbox: chatterbox-t3-turbo-q8_0 | linux | CPU | 8078 | — | 1835 |
| Chatterbox: chatterbox-t3-turbo-q8_0 | macos | CPU | 2626 | — | 1746 |
| Supertonic: supertonic3-q8_0 | linux | (CPU) | 708 | 0.484 | 876 |
| Supertonic: supertonic3-q8_0 | macos | (CPU) | 49 | 0.0340 | 584 |
| Parler: parler-mini-v1-q8_0 | linux | CPU | 249362 | 18.8 | 1964 |
| Parler: parler-mini-v1-q8_0 | macos | MTL0 (Metal) | 12204 | 0.582 | 1307 |
| Cosyvoice: cosyvoice3-llm-q8_0 | linux | CPU | 67695 | 18.0 | 1416 |
| Cosyvoice: cosyvoice3-llm-q8_0 | macos | CPU | 46825 | 12.3 | 1467 |

`—` RTF for chatterbox because it is text-driven variable output.

Source: [workflow run 34113144218](https://github.com/tetherto/qvac-fabric-speech.cpp/actions/runs/34113144218) (2026-09-07).

### Mac Studio M3 Ultra (96 GB unified memory)

| Implementation                        | Backend         | T3 gen             | S3Gen+HiFT gen | Total inference | RTF   | vs real-time |
|---------------------------------------|-----------------|-------------------:|---------------:|----------------:|------:|-------------:|
| **`tts-cpp` Q4_0**                    | **Metal**       |  573 ms / 155 tok  |    412 ms      |   **985 ms**    | 0.16  | **6.4×**     |
| `tts-cpp` Q4_0                        | CPU (NEON+Accel)| 2 045 ms / 178 tok |  5 523 ms      |    7 568 ms     | 1.05  | 0.96×        |

On Apple Silicon the Metal build runs end to end at **6.4× real time**; the
CPU-only build stays just under real time.

### Multilingual (Apple M4, F16 weights)

Same prompt + seed run through both variants on the same M4, for apples-to-
apples comparison.  MTL is 30 transformer layers vs Turbo's 24 plus CFG on
both T3 and CFM (2 forward passes per step), and it samples standard CFM
for 10 Euler steps instead of Turbo's meanflow 2.

| Config                              | T3 infer            | S3Gen infer | Audio | **RTF** |
|-------------------------------------|---------------------:|-------------:|------:|--------:|
| Turbo, Metal                        |  788 ms /  73 tok   |    768 ms    | 3.04 s| 0.51    |
| Turbo, CPU 4t                       | 1 721 ms /  73 tok  |  3 334 ms    | 3.04 s| 1.66    |
| Multilingual, Metal *(batched CFM)* | 1 865 ms /  61 tok  |  2 247 ms    | 2.56 s| 1.61    |
| Multilingual, CPU 4t *(2-call CFM)*³| 3 210 ms /  85 tok  | 25 660 ms    | 3.52 s| 8.20    |

The MTL Metal path packs the CFG cond+uncond into a single batch=2
decoder forward (`use_b2 = !ggml_backend_is_cpu(...)`), since kernel
dispatch overhead amortises well across the bigger workload; on ggml-cpu
the extra permute+cont ops that a batched attention block needs regress
throughput, so CPU keeps the two-call path.  See
[`PROGRESS.md §3.19`](../PROGRESS.md) for the measurement and a discussion
of where the MTL slowdown lives relative to Turbo.

³ Re-measured on the same M4 host after commit `6d9b42b` restored the
CFG combine on the non-`use_b2` (CPU) CFM path.  The previous row in
this table (`2 711 ms / 71 tok`, `8 029 ms`, `RTF 3.63`) was captured
while CPU was silently running only the conditional CFM pass — i.e.
half the CFM compute and no classifier-free guidance steering.  S3Gen
wall-time roughly doubled (one extra forward call per CFM step) and
RTF went from 3.63 → 8.20.  The remeasurement here uses a different
local reference wav (the original `jfk.wav` was not present on the
host) — that accounts for the slightly larger token count (85 vs the
original 71); the per-audio-second cost ratio is what shifted, ~2×,
in line with restoring the missing pass.  Turbo and the Metal
multilingual row are
unaffected — they always carried the CFG combine.

### Multilingual (Mac Studio M3 Ultra, after §3.21 optimisation pass)

Same Spanish prompt (`"Hola, esto es una demostración multilingüe."`,
`--language es`), `jfk.wav` voice, seed 42, greedy (`--temp 0 --top-k 1`),
3 warm runs averaged.  T3 is now CFG-batched into a single Metal forward
(B=2, mirrors S3Gen's `use_b2`); MLP uses `ggml_swiglu_split` so the 30
SiLU+Mul element-wise pairs collapse into one fused Metal kernel per
layer.  The new `--cfm-steps N` flag exposes the standard CFM step count
(default 10); N=7 is the recommended quality knee (log-mel cosine vs N=10
= **0.995**).

| Config                              | T3 infer           | S3Gen infer | Audio | **RTF** |
|-------------------------------------|-------------------:|------------:|------:|--------:|
| MTL, Metal Q4_0, `--cfm-steps 7`    |  478 ms /  84 tok  |    576 ms   | 3.48 s|  0.30   |
| MTL, Metal Q4_0 (default N=10)      |  482 ms /  84 tok  |    730 ms   | 3.48 s|  0.35   |
| MTL, Metal F16, `--cfm-steps 7`     |  579 ms /  89 tok  |    586 ms   | 3.68 s|  0.32   |
| MTL, Metal F16 (default N=10)       |  613 ms /  89 tok  |    752 ms   | 3.68 s|  0.37   |

Compared to the M4 multilingual numbers above, the M3 Ultra hits
**RTF 0.30** on Q4_0 — a 4.6× speedup.  The CFG-batching alone drops T3
by 42–45% (see PROGRESS.md §3.21 for the full bench matrix and the
NEGATIVE results for F16 KV cache and SwiGLU on F16).

### Multilingual (M3 Ultra, post §3.24–§3.31 Metal kernel portfolio)

Same prompt, voice, seed as §3.21 above.  Adds, on top of §3.21:

- **§3.24** — HiFT conv-kernel F16 quantisation (64 tensors).
- **§3.26** — `kernel_mul_mv_f32_f16{,_4,_short}` Metal kernel variants
  to unblock 21 more HiFT `source_*` F16 tensors
  (GGUF shrinks **754 → 747 MB**, WAV cos 1.000000 vs §3.24).
- **§3.27** — `kernel_mul_mm` + `ADD(bias)` [+ `ADD(residual)`] fusion
  for the CFM transformer Q4_0 mat-muls (1820 saved `ggml_add`
  dispatches per synth).
- **§3.28** — extends the fusion to absorb `GELU_ERF` (CFM FF ff0
  activation path; 1120 additional saved dispatches).
- **§3.30** — `test-metal-ops` fused-mul_mm parity harness + bias-only
  direct-store variant.
- **§3.31** — iOS-arm64 cross-build portability +
  `scripts/bench-m4-validation.sh` for M4 hand-off.

5-invocation averages (`default N=10` CFM — compare to the §3.21 N=10 row):

| Config                              | T3 infer            | S3Gen infer | Audio | **RTF** |
|-------------------------------------|--------------------:|------------:|------:|--------:|
| MTL, Metal Q4_0 + HiFT F16 v2 (§3.28) |  433 ms / 84 tok  |    706 ms   | 3.48 s| **0.33** |
| MTL, Metal Q4_0 baseline (§3.21 N=10) |  482 ms / 84 tok  |    730 ms   | 3.48 s|  0.35    |
| **Δ §3.21 → §3.28**                   |  **−49 ms / −10.2 %** | **−24 ms / −3.3 %** | — | **−0.02** |

WAV is byte-exact deterministic across runs (md5
`d8a1b22375dbcb2259c686426a7d76c5` ×5).  Parity harness
`test-metal-ops` passes 14 gates (3 base + 3 conv_transpose_1d + 8
fused `mul_mm`).  The 1088-line ggml-metal patch backing these
kernel changes is shipped pre-applied by the `ggml-speech` vcpkg
port (qvac-ext-ggml/speech branch); the standalone chatterbox.cpp
repo carries it under `patches/ggml-metal-chatterbox-ops.patch`
against pinned ggml `58c38058`.  All §3.24–§3.30 kernel changes
cross-compile cleanly for iOS-arm64 (portability verified; runtime
measurement deferred until an M4 / iPhone / iPad run of
[`scripts/bench-m4-validation.sh`](../scripts/bench-m4-validation.sh)).

M3 Ultra CFM time specifically drops from 541.9 ms → 534.0 ms
(**−1.5 %**) — modest on this chip because per-dispatch overhead
is very low; expected to be larger on bandwidth-limited silicon
(M4 / A-series) where each saved `ggml_add` dispatch is worth more
relative to compute.

### Reproducing these numbers

```bash
# Build tts-cpp, then:
./build/tts-cli \
    --model       models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf  models/chatterbox-s3gen.gguf \
    --reference-audio test/reference-audio/jfk.wav \
    --text "Hello from native C plus plus. This audio was generated end to end on CPU using ggml." \
    --out /tmp/bench.wav \
    --seed 42 \
    --n-gpu-layers 99   # 0 or omit for CPU
```

The binary prints both the per-stage timings and `BENCH:` lines that
scripts can scrape.  Note: the binary also prints an inner
`=== pipeline: … RTF=… ===` line — that RTF covers **only the
S3Gen + HiFT phase** (the timer around `s3gen_synthesize_to_wav`, which
runs after T3 is already done).  The tables above report the full
end-to-end number (T3_INFER + S3GEN_INFER).

`gen_RTF = (T3_INFER_MS + S3GEN_INFER_MS) / AUDIO_MS`

Token counts vary slightly across backends because the CPU-side
sampler reads logits that come out of different float-reduction orders
per backend; per-token T3 cost is the directly-comparable figure.
Full development history and older backend combinations (F16 vs
Q4_0 / Q5_0 / Q8_0, plus other machines) are in
[`PROGRESS.md §3.10 / §3.13`](../PROGRESS.md).

### Streaming mode — low-latency playback

For interactive use cases, the binary can emit audio **chunk-by-chunk**
as it's generated instead of waiting for the whole sentence to finish.
Any non-zero `--stream-chunk-tokens N` turns streaming on.

**Flags:**

- `--stream-chunk-tokens N` — main knob; N speech tokens per chunk
  (25 ≈ 1 s of audio, 50 ≈ 2 s).
- `--stream-first-chunk-tokens N` — override the *first* chunk's size
  so first-audio-out lands early while later chunks stay big and keep
  overall RTF low.  Typical: 10.
- `--stream-cfm-steps N` — CFM Euler step count. Turbo defaults to 2 and
  supports 1 or 2; one step trades some quality for lower cost.
  Multilingual uses standard CFM, and streaming requests below its model
  timestep count are floored to 10.
- `--out -` — emit raw `s16le` mono @ 24 kHz to stdout instead of
  writing a wav file, so the output can be piped straight into a
  player.

**Recommended low-latency preset for interactive use:**

```bash
brew install sox      # one-time, for the `play` command

./build/tts-cli \
    --model      models/chatterbox-t3-turbo.gguf \
    --s3gen-gguf models/chatterbox-s3gen.gguf \
    --text       "Hello from streaming Chatterbox." \
    --stream-first-chunk-tokens 10 \
    --stream-chunk-tokens       25 \
    --stream-cfm-steps          1 \
    --n-gpu-layers              99 \
    --out - \
  | play -q -t raw -r 24000 -b 16 -e signed -c 1 -
```

`play` ships with `sox` and routes straight to CoreAudio.  If you
prefer, the same stdout stream works with `ffplay -f s16le -ar 24000
-ch_layout mono -nodisp -i -` or piped through a Python
`sounddevice.play()` one-liner; on some macOS 26 builds ffplay's SDL
output is silent for raw piped audio, so `sox play` is the safest
default.

You can also drop the `--out -` to get a regular wav:

```bash
./build/tts-cli … --stream-chunk-tokens 50 --out out.wav
afplay out.wav
```

**Latency and throughput** on an Apple M4 with the Metal backend and
the preset above, feeding the sentence *"Hello from streaming
Chatterbox, I am John and I work in Google since 2010. I love to go
out with my friends, eat some pizza and also drink some wine. I also
love to travel around the world alone."* (produces 317 speech tokens,
~12.7 s of audio):

| metric | value |
|---|---|
| first-audio-out latency | **279 ms** |
| chunk 1 (10-token bootstrap) | RTF 0.99 |
| chunks 2–13 (steady-state, 25 tokens each) | **RTF 0.30 – 0.63** |
| chunk 14 (tail finalise) | RTF 1.42 |
| total wall time | 11.5 s for 12.7 s of audio |
| overall RTF | **0.90** |

The steady-state RTFs stay comfortably below 1.0, so the streamer
sustainably pushes audio faster than real-time playback consumes it.
Chunk 1 is small by design so first audio lands in ~280 ms; the final
chunk is short and relatively slow (fixed encoder/CFM overhead
amortised over only 0.4 s of audio).

For the full journal of how streaming got there — bit-exact CFM parity,
`cache_source` + `trim_fade` port, `--out -` stdout wiring, per-chunk
tuning — see [`PROGRESS.md §B1`](../PROGRESS.md).
