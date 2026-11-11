# Core ML load time measurements

Measured with `scripts/coreml/bench-load-time.py` on the workloads pinned in `scripts/coreml/bench-manifest.json`. These numbers set the Chatterbox Core ML runtime's constants (Metal cost model, warm-load hints, width set, weight-sharing accounting); the last section says where each one goes.

- Not measured yet: run `bench-load-time.py run` on an otherwise idle machine, then regenerate this file with `bench-load-time.py report`.

## Setup and usage

Requirements:

- Apple silicon, macOS 15 or later, Xcode command-line tools (`xcrun clang++`) and Python 3.9 or later. The published numbers come from an otherwise idle M2 on macOS 15.7.3; `run` refuses battery power unless `--allow-battery` is given.
- The Chatterbox GGUFs pinned in `bench-manifest.json` (`variants`, with SHA-256 and Hugging Face revisions) under the models root: `--models-root DIR`, or `QVAC_BENCH_MODELS`.
- The Core ML sidecars for each width set in `sidecar_sets`, exported from the pinned f32 GGUFs with `engines/tts/scripts/export-chatterbox-coreml.py` as `exporter` in the manifest describes (`--frames` selects the widths, `--compile` writes the `.mlmodelc` next to the `.mlpackage`).
- A `tts-cli` built with `-DTTS_CPP_COREML=ON` (see `engines/tts/docs/chatterbox.md`).

Commands:

```bash
python3 -B scripts/coreml/test_bench_load_time.py     # hermetic tests, no Core ML needed
scripts/coreml/bench-load-time.py --dry-run --check-models   # manifest, pinned files and the run plan
scripts/coreml/bench-load-time.py build-driver
scripts/coreml/bench-load-time.py smoke                # fresh-host.sh identity and e5rt cache clean-up
scripts/coreml/bench-load-time.py run --tts-cli <build>/tts-cli --out <dir>   # all measurements
scripts/coreml/bench-load-time.py run ttr --variant turbo --out <dir>   # one measurement, one variant
scripts/coreml/bench-load-time.py report --results <dir> --out docs/COREML-LOAD-TIME.md
scripts/coreml/bench-load-time.py cleanup              # remove leftover bench identities and caches
```

`run` writes one JSON object per line to `<dir>/results.jsonl`; `report` renders this file from it. Cold runs use a fresh `fresh-host.sh` identity and a fresh copy of the sidecars, and delete both afterwards.

## Multifunction spec sizes

What a template would embed is the program (`model.mil` compiled, `model.mlmodel` in the package), not the weights. MB = 10^6 bytes.

| Variant | Set | Stage | Functions | model.mil | zlib | LZFSE | model.mlmodel | zlib | LZFSE | weight.bin |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| — | | | | | | | | | | |

## Width set: 3 vs 7

Cold = sum of first loads on a fresh identity (ANE compile included), all functions of the set incl. HiFT. Warm = median per-function load + first prediction in a new process. Synthesis = warm `tts-cli` on the benchmark prompt (median of the runs).

| Variant | Set | Functions | Cold total (s) | Warm load+predict, median over functions (ms) | Synthesis wall (ms) | S3Gen (ms) |
|---|---|--:|--:|--:|--:|--:|
| — | | | | | | |

## `expected_warm_load_ms` per function

A warm load is a new process on an identity whose ANE plans are cached; the hint's `warm_load_ms` is load plus the warm-up prediction, which is what this column reports. Prediction = median of the later predictions.

| Variant | Set | Stage | Function | Cold load (ms) | Warm load (ms) | expected_warm_load_ms (median) | Max | Prediction (ms) |
|---|---|---|---|--:|--:|--:|--:|--:|
| — | | | | | | | | |

## Do loaded functions share weight memory?

All functions of one stage loaded and held in one process, per compute-unit policy. Per metric: the increment of the first function, the median of each later one, and the last function loaded alone (MB). `footprint` = phys_footprint (what fit charges); `resident` adds file-backed weight pages; `wired` = host wired memory, where the ANE keeps a loaded program; `available` = the host's available-memory drop (noisy). 'shared' when functions 2..n add less than 25% of a weight.bin each.

| Variant | Set | Stage | Units | weight.bin (MB) | Metric | First | Each later | Alone | Verdict |
|---|---|---|---|--:|---|--:|--:|--:|---|
| — | | | | | | | | | |

## Per-variant Metal cost constants for `s3gen_estimate_metal_ms`

Least-squares `ms = a + b × frames` over the length sweep on Metal (`CHATTERBOX_COREML_DISABLE=1`), one S3Gen call per sentence after a discarded warm-up call. Encoder, CFM and the whole pipeline against `T_mu` (prompt + new frames), HiFT against `T_mel` (new frames). Worst ratio = the largest measured / fitted.

| Variant | Stage | a (ms) | b (ms/frame) | R² | Points | Frames | Worst ratio |
|---|---|--:|--:|--:|--:|---|--:|
| — | | | | | | | |

## Time to ready, with and without the admission gate

Every function (encoder and CFM widths plus HiFT) requested at `minimal` in one construction group with `wait=construct`, on a fresh identity with the artifacts at a fresh path. `generate` = a stand-in prepare job per artifact (copy the package, `compileModelAtURL`), as on-device generation will; gated = the shared runtime's Manager, ungated = the same two queues without the gate. Prepare wait = issue to prepare start; load gate wait = artifact available to load start.

| Variant | Config | Runs | All ready | Time to ready median (s) | MAD (s) | Min–max (s) | Worst prepare wait (s) | Worst load wait (s) | deferred(memory) (s) | Cached startup (s) |
|---|---|--:|---|--:|--:|---|--:|--:|--:|--:|
| — | | | | | | | | | | |

## Available-memory reading under normal load

Not measured yet.

## Warm synthesis and encoder-fallback contention

`tts-cli`, benchmark prompt, built-in voice, seed 42, q8_0, a new process per run after one priming run. `coreml-no-encoder` keeps CFM and HiFT on Core ML (so S3Gen still overlaps T3) and the encoder on Metal: the T3 slowdown against `coreml-w7` is the contention the runtime pays when the encoder falls back.

| Variant | Config | Runs | Wall (ms) | T3 (ms) | S3Gen (ms) | Peak footprint (MB) |
|---|---|--:|--:|--:|--:|--:|
| — | | | | | | |

## Where each number goes

| Number | Where it goes |
|---|---|
| Spec sizes | The multifunction template size budget; LZFSE compression if a template is over it |
| 3 vs 7 widths (cold total, warm load, synthesis) | The width set the Chatterbox generation templates ship |
| `expected_warm_load_ms` per function | Each template's `template.json`; the bounded wait before a hint exists |
| Weight sharing verdict | How resident functions are charged against the Core ML memory envelope |
| Metal cost constants | `s3gen_estimate_metal_ms` in the Chatterbox Core ML runtime |
| Time to ready with / without the gate, worst prepare wait | Whether the admission gate stays on; the Chatterbox "time to all 15 ready" target |
| Available-memory reading | Confirms memory admission does not defer cold loads on the bench host |
| Encoder-fallback contention | Expected overlap when the Chatterbox encoder runs on Metal |

