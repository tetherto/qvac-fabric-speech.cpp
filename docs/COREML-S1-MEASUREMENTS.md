# Core ML load time: S1 measurements (QVAC-27561, unit S1c)

Measured with `scripts/coreml/bench-load-time.py` on the workloads pinned in `scripts/coreml/bench-manifest.json` (PLAN §5.3). These are the numbers S1b and S6 consume; the last section maps each one to the plan item it settles.

- Not measured yet: the Benchmark stage runs `bench-load-time.py run` under `lane quiet` and regenerates this file with `bench-load-time.py report`.

Reproduce on an otherwise idle M2 (macOS 15.7.3, AC power), with the sidecars exported as `bench-manifest.json` `exporter` describes and a `tts-cli` built with `-DTTS_CPP_COREML=ON`:

```bash
scripts/coreml/bench-load-time.py build-driver
python3 -B scripts/coreml/test_bench_load_time.py     # hermetic, includes --dry-run
scripts/coreml/bench-load-time.py smoke                # fresh-host.sh identity and e5rt cache clean-up
scripts/coreml/bench-load-time.py run --tts-cli <build>/tts-cli --out <dir>
scripts/coreml/bench-load-time.py report --results <dir> --out docs/COREML-S1-MEASUREMENTS.md
```

## Multifunction spec sizes (§2.5)

What a template would embed is the program (`model.mil` compiled, `model.mlmodel` in the package), not the weights. MB = 10^6 bytes.

| Variant | Set | Stage | Functions | model.mil | zlib | LZFSE | model.mlmodel | zlib | LZFSE | weight.bin |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| — | | | | | | | | | | |

## Width set: 3 vs 7 (O8)

Cold = sum of first loads on a fresh identity (ANE compile included), all functions of the set incl. HiFT. Warm = median per-function load + first prediction in a new process. Synthesis = warm `tts-cli` on the benchmark prompt (median of the runs).

| Variant | Set | Functions | Cold total (s) | Warm load+predict, median over functions (ms) | Synthesis wall (ms) | S3Gen (ms) |
|---|---|--:|--:|--:|--:|--:|
| — | | | | | | |

## `expected_warm_load_ms` per function (§2.9)

A warm load is a new process on an identity whose ANE plans are cached; the hint's `warm_load_ms` is load plus the warm-up prediction, which is what this column reports. Prediction = median of the later predictions.

| Variant | Set | Stage | Function | Cold load (ms) | Warm load (ms) | expected_warm_load_ms (median) | Max | Prediction (ms) |
|---|---|---|---|--:|--:|--:|--:|--:|
| — | | | | | | | | |

## Do loaded functions share weight memory? (§2.13)

All functions of one stage loaded and held in one process; footprint increment after each load + prediction. 'shared' when functions 2..n add less than 25% of a weight.bin each.

| Variant | Set | Stage | weight.bin (MB) | First function (MB) | Each later function, median (MB) | One function alone (MB) | Verdict |
|---|---|---|--:|--:|--:|--:|---|
| — | | | | | | | |

## Per-variant Metal cost constants for `s3gen_estimate_metal_ms` (§3.1.1)

Least-squares `ms = a + b × frames` over the length sweep on Metal (`CHATTERBOX_COREML_DISABLE=1`), one S3Gen call per sentence after a discarded warm-up call. Encoder against `T_mu`, the rest against `T_mel`. Worst ratio = the largest measured / fitted.

| Variant | Stage | a (ms) | b (ms/frame) | R² | Points | Frames | Worst ratio |
|---|---|--:|--:|--:|--:|---|--:|
| — | | | | | | | |

## Time to ready, with and without the admission gate (§2.9, §5.3)

Every function (encoder and CFM widths plus HiFT) requested at `minimal` in one construction group with `wait=construct`, on a fresh identity with the artifacts at a fresh path. `generate` = a stand-in prepare job per artifact (copy the package, `compileModelAtURL`), as on-device generation will; gated = the S1a manager, ungated = the same two queues without the gate. Prepare wait = issue to prepare start; load gate wait = artifact available to load start.

| Variant | Config | Runs | All ready | Time to ready median (s) | MAD (s) | Min–max (s) | Worst prepare wait (s) | Worst load wait (s) | deferred(memory) (s) | Cached startup (s) |
|---|---|--:|---|--:|--:|---|--:|--:|--:|--:|
| — | | | | | | | | | | |

## Available-memory reading under normal load (§2.9 memory admission)

Not measured yet.

## Warm synthesis and encoder-fallback contention (§3.1.1)

`tts-cli`, benchmark prompt, built-in voice, seed 42, q8_0, a new process per run after one priming run. `coreml-no-encoder` keeps CFM and HiFT on Core ML (so S3Gen still overlaps T3) and the encoder on Metal: the T3 slowdown against `coreml-w7` is the contention S1b inherits when the encoder falls back.

| Variant | Config | Runs | Wall (ms) | T3 (ms) | S3Gen (ms) | Peak footprint (MB) |
|---|---|--:|--:|--:|--:|--:|
| — | | | | | | |

## Where each number goes

| Number | Consumer | Plan |
|---|---|---|
| Spec sizes | S6 template size budget; LZFSE escape route if over budget | §2.5 |
| 3 vs 7 widths (cold total, warm load, synthesis) | S6 width set; O8 sign-off | §3.1.1 Generation, §6.2 O8 |
| `expected_warm_load_ms` per function | `template.json` for S6; bounded wait before a hint exists | §2.9 |
| Weight sharing verdict | `F_res` charging in the Core ML envelope | §2.13 |
| Metal cost constants | `s3gen_estimate_metal_ms` in `chatterbox_coreml.cpp` (S1b) | §3.1.1 |
| Time to ready with / without the gate, worst prepare wait | admission-gate decision; S1b "time to all 15 ready" | §2.9, §5.3 |
| Available-memory reading | confirms memory admission does not defer cold loads on the bench host | §2.9 |
| Encoder-fallback contention | S1b overlap expectations | §3.1.1 |

