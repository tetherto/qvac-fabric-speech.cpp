# tts engine: Chatterbox

Part of the [tts engine documentation](../README.md).

## Chatterbox

End-to-end inference on a short sentence with voice cloning from an 11 s
reference wav (T3 + S3Gen + HiFT, warm runs, excludes model load):

**Turbo:**

| Backend                              | Wall      | `RTF`  | vs real-time |
|--------------------------------------|----------:|-------:|-------------:|
| Vulkan (CI · Linux x86-64, Q4_0)     |   410 ms  | 0.099  | **10.1×**    |
| Metal (Mac Studio M3 Ultra, Q4_0)    |   985 ms  | 0.16   | 6.4×         |
| CPU (CI · Linux x86-64, Q4_0)        | 5 841 ms  | 1.54   | 0.65×        |
| CPU (Mac Studio M3 Ultra, NEON)      | 7 568 ms  | 1.05   | 0.96×        |

> Rows are independent warm runs on different machines/backends, so **`RTF`** —
> not wall time — is the comparable metric; wall time tracks each run's own
> generated audio length. The Linux x86-64 rows are CI numbers (see the
> [Performance](../README.md#performance) section for the full CI table + provenance).

**Multilingual** (same Spanish prompt, seed 42, built-in voice):

| Backend                              | Wall      | `RTF` | vs real-time |
|--------------------------------------|----------:|------:|-------------:|
| **Metal (M3 Ultra, Q4_0, `--cfm-steps 7`)** | **1.05 s**| **0.30** | **3.3×**     |
| Metal (M3 Ultra, Q4_0)               |  **1.22 s** | 0.35  | 2.9×        |
| Metal (M3 Ultra, F16, `--cfm-steps 7`)| 1.16 s   |  0.32  | 3.2×        |
| Metal (M3 Ultra, F16)                |  1.41 s   |  0.38  | 2.6×        |
| **Metal (M4, Q4_0)**                 |  **3.0 s**| 1.37  | 0.73×        |
| Metal (M4, F16)                      |   4.0 s   | 1.65  | 0.61×        |
| CPU (M4, 4t NEON, Q4_0)              |  10.7 s²  | 4.32² | 0.23×        |
| CPU (M4, 4t NEON, F16)               |  17.1 s²  | 6.70² | 0.15×        |

The M3 Ultra rows reflect the §3.21 optimisation pass — CFG cond+uncond
batched into one Metal forward (B=2) on T3, the new `--cfm-steps N` knob
on the standard 10-step CFM (N=7 is the recommended quality knee, log-mel
cosine vs N=10 = **0.995**), and `ggml_swiglu_split` on the Llama MLP.
The M4 rows are kept for continuity with §3.19/§3.20.

² **CPU multilingual rows re-measured after restoring CFG on the
non-Metal CFM path.**  The previous numbers in this row (`6.0 s / 2.69`
and `7.8 s / 3.24`) were captured between Apr 23–May 4 while the
non-`use_b2` branch in the CFM step loop was silently running only the
conditional pass — i.e. half the CFM compute, no classifier-free
guidance steering on CPU/Vulkan/CUDA.  Fixed in commit `6d9b42b`; the
two rows above are now end-to-end re-measurements on the same M4 host
with CFG correctly applied (12 CFM steps × 2 forward calls).
S3Gen wall-time roughly doubled, RTF went up ~2×.  Metal rows are
unaffected (the `use_b2` branched path always carried the CFG combine).

See the [full benchmark](../README.md#performance) section below for the CI benchmark
table, or [`PROGRESS.md`](../PROGRESS.md) for the full chronological
development journal — every numerical-parity stage and optimization pass
(T3 Flash Attention, KV-cache layout rework, Metal kernel patches,
CAMPPlus + VoiceEncoder + S3TokenizerV2 ported to ggml graphs, mel
extraction via STFT matmul, T3 Q4/Q5/Q8 quantization, the multilingual
Llama-520M port + CFG dual-cache (§3.19), and the shared S3Gen weight-
quantisation pass that ships in this repo (§3.20)).

---
