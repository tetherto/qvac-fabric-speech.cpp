# Weekly desktop benchmark on the shared Tailscale machines

`weekly.sh` benchmarks the speech engines every Sunday evening on the team's shared machines, without any
developer laptop in the loop: cron on each machine builds the current engine and ggml heads for every backend the
machine has, provisions the pinned models, runs the matrix in `families.json` through
`scripts/benchmarks/run-family.sh`, and writes a Markdown report with the numbers, the memory peaks, the exact
machine and toolchain versions and the sha256 of every model file. The results everyone reads are the
Speech Bench hub and one dashboard per machine, refreshed from the machines' summaries every Monday (see
`skills/speech-bench/SKILL.md`); e-mail delivery exists but is off unless `MAIL_TO` is set.

| machine | host | backends | schedule |
| --- | --- | --- | --- |
| DGX Spark GB10 | `spark-d9ca` | cpu, cuda | Sunday 18:00 CET (17:00 on the UTC clock) |
| Linux x64, 2x RTX 5090 | `qvac-dev-linux-x64` | cpu, cuda, vulkan | Sunday 18:00 CET (17:00 on the UTC clock) |
| Strix Halo | `qvac-dev-strix-0` | cpu, vulkan | Sunday 18:00 CET (17:00 on the UTC clock) |
| Mac Studio M3 Ultra | `qvac-dev-mac-arm64` | cpu, metal | Sunday 18:00 CET |
| Mac mini M4 | `mac-mini-m4-gpu` | cpu, metal | Sunday 18:00 CET |

The two weekly benches on the shared machines run on different days so that neither disturbs the other: the
fabric-llm engine bench on Saturday from 18:00 CET, the speech bench on Sunday from 18:00 CET (a full run takes
about 6 to 7 hours). Both also hold the same `/tmp/gpu-bench.lock` on the Mac; this run waits for the lock (up to
`LOCK_WAIT_HOURS`, 12 by default) and for every other GPU job to end before it starts, then holds the lock while it
benchmarks. See [`skills/speech-bench/SKILL.md`](../../../skills/speech-bench/SKILL.md) for the coordination rules
and the weekly artifact.

## What is measured

`families.json` lists one family per model and quantization requested for the desktop matrix:

| task | family | weights |
| --- | --- | --- |
| ASR | `parakeet-tdt` | Parakeet TDT 0.6B v3 Q8_0 |
| ASR | `parakeet-unified` | Parakeet Unified English 0.6B Q8_0 |
| ASR | `transcribepsy` | TranscribePsy-250M-EU-sub Q8_0 (HuggingFace, pinned revision) |
| ASR | `nemotron` | Nemotron 3.5 ASR Streaming 0.6B F16 |
| TTS | `supertonic3` | Supertonic 3 Q8_0, every component and voice in one GGUF |
| TTS | `audio8-lm-{f16,q8_0,q4_0}-dec-{f16,q8_0}` | Audio8 TTS Preview 0.6B, LM x codec decoder grid |
| TTS cloning | `audio8-clone-enc-{f16,q8_0}` | Audio8 with the codec encoder, cloning the JFK clip |
| TTS | `moss-tts-f16`, `moss-tts-q8_0` | MOSS-TTS v1.5 8.5B: F16 backbone, and transformer blocks requantized to Q8_0; decoder F16 |
| TTS cloning | `moss-tts-clone-f16` | MOSS-TTS with the F16 tokenizer encoder and the JFK clip at 24 kHz |
| Music | `acestep-turbo-q8_0`, `acestep-turbo-q4_k_m`, `acestep-sft-q8_0`, `acestep-base-q4_k_m` | ACE-Step 1.5 DiT variants with Qwen3-Embedding-0.6B Q8_0, 5Hz LM 0.6B Q8_0 and the BF16 VAE |

Every family is a `run-family.sh` spec (same fields as `scripts/benchmarks/families.json`) plus `runs`, `warmup`,
`cpu_runs`, `cpu_warmup`, `timeout_minutes` and a `provision` list that says where each model file comes from.
ASR rows report the native bench's inference time, RTF and WER on the 11 s JFK clip; TTS rows report the process
wall clock including model load, the generated duration and the TTS intelligibility WER (pinned Whisper tiny on
CPU); music rows report the process wall clock of a 30 s render at seed 42. Memory is the peak RSS from
`/usr/bin/time` and, on NVIDIA backends, the rise of `nvidia-smi` memory.used over the run.

Two requested cuts are not in the registry: MOSS-TTS Q8_0 is produced on the machine by requantizing the `blk.*`
tensors of the F16 backbone with `engines/tts/scripts/requantize-gguf.py`, and the ACE-Step Base DiT ships only as
Q4_K_M, so that row is Q4_K_M and says so.

## Files

| file | what it does |
| --- | --- |
| `weekly.sh` | entry point: lock, update sources, fetch models, build and bench every backend, summarize, report, e-mail |
| `lib.sh` | shared functions: config, lock, source update, per-backend builds, watchdog, GPU sampler, run selection |
| `families.json` | the matrix (see above) |
| `fetch-models.sh`, `fetch-registry.js`, `prepare-derived.py` | model provisioning: registry client, HuggingFace, repo fixtures, requantized and resampled files |
| `machine-info.sh`, `collect-pins.sh` | `machine.json` (hardware, OS, driver, toolchain) and `pins.json` (engine and ggml commits, model sha256 and origins) |
| `summarize.py` | run directory -> `summary.json` |
| `report.py` | `summary.json` (+ previous run) -> `report.md` with the changes over 5% and whether a pin moved |
| `build-dashboard-data.py`, `comparisons.json`, `engines.json` | a folder of `summary.json` files -> per machine the JSON files its dashboard reads (`<host>/runs.json`, `history.json`, `coverage.json`, `engines.json`, `machine.json`, `commentary.json`) and `hub.json` for the hub; `comparisons.json` holds the dated competitor rows of the campaign (qvac RTF and competitor-over-qvac time ratio per device and backend), `engines.json` the engines with their scope |
| `dashboard/index.html`, `dashboard/app.js`, `dashboard/app.css`, `dashboards.json` | the page every machine dashboard runs (overview, a tab per task with a speed chart per model, backend and input and a week-over-week line per backend, runs and coverage) and the links of the hub and the dashboards |
| `build-landing.py`, `dashboard/hub.css` | `hub.json` -> the Speech Bench hub: per task every machine with a link into its dashboard's tab, the quality flags, the engines and how the bench runs |
| `build-artifact.py` | a folder of `summary.json` files -> one self-contained HTML report (tables per task, history and pins per machine) for reading offline, without the dashboards |
| `nsdate`, `scheduler.sh` | nanosecond clock shim for macOS before 26, and the user-space scheduler for hosts without cron |
| `narrate.py` | optional: asks Claude for a ten-line narrative placed above the report in the e-mail |
| `send-report.py` | SMTP delivery with `report.md` and `summary.json` attached |
| `install.sh`, `config.example.env` | deployment and the per-machine configuration |
| `test-desktop-weekly.sh`, `test_*.py` | the test suite (runs in the desktop benchmark workflow's self-check) |

## Layout on a machine

Everything lives under `BENCH_HOME` (`~/speech-bench` by default):

```
tools/                 this directory, copied by install.sh
config.env             per-machine settings (see config.example.env)
src/speech, src/ggml   shallow checkouts of the engine and ggml branches
builds/<backend>/      ggml, ggml-install and speech build trees, plus BUILT with the commit pair they came from
registry-client/       the npm registry client, private to this account
venv/                  python venv with gguf and numpy for the requantized cut, private to this account
models/store/          one copy of every model file, with .sha256 and .source sidecars
models/families/<f>/   symlinks the driver reads as --models-root
runs/<date>/           weekly.log, machine.json, pins.json, results/<backend>/<family>.{json,log,gpu.csv},
                       summary.json, report.md, narrative.md, mail-body.md, DONE or FAILED
```

`MODELS_DIR` defaults to the private `models/` above. It can point at the shared model folders instead
(`/home/shared/models/qvac-speech/bench` on the Linux box, `/Users/Shared/models/qvac-speech/bench` on the Mac) so
that every account reuses one copy, at a price: every account that can write there can replace a model file or its
`.sha256` sidecar, so the report's pins are only as trustworthy as that folder's writers. Only data goes to
`MODELS_DIR`; the registry client, the venv, the checkouts and the builds always stay under `BENCH_HOME`.

## Install

On each machine, as your own user (nothing needs root):

```sh
scp -r scripts/benchmarks/desktop-weekly <host>:speech-bench-src
ssh <host> 'bash speech-bench-src/install.sh --hour 18'   # --hour 17 on the UTC Linux box
```

`install.sh` copies the tools to `~/speech-bench/tools`, seeds `~/speech-bench/config.env` and installs the cron
line (Sunday 18:00 machine time by default; `--weekday`, `--hour`, `--minute` change it, `--no-cron` skips it).
Then edit `config.env` for the machine's settings (`MODELS_DIR`, `BACKENDS`, `CUDA_ARCH` and the other keys in
`config.example.env`); mail and the Claude narrative are optional and stay off without their keys. Any secret
lives in a file with mode 0600 that the config points at, never in `config.env` itself.

Prerequisites already present on the machines: git, cmake, ninja, jq, python3, node, curl; CUDA 13.3 and the
Vulkan SDK on the Linux boxes; Xcode command line tools and Homebrew on the Macs. The registry client is installed
by `fetch-models.sh` into `registry-client/` under the bench home from npm. Per-machine settings (`BACKENDS`,
`CUDA_ARCH`, `CUDA_HOME`, `MEMORY_SAMPLER=meminfo` for unified-memory NVIDIA hosts, `SKIP_FAMILIES` for the 16 GB
Mac mini) are listed in `config.example.env` and in the skill. A host without cron (the Strix Halo) gets
`scheduler.sh` from `install.sh` instead of a crontab line; it does not survive a reboot. A macOS host whose
`/bin/date` lacks `%N` (macOS 15) gets `run-family.sh` routed through the `nsdate` shim after every source update.

Try the pipeline before the first Sunday:

```sh
ssh <host> 'nohup ~/speech-bench/tools/weekly.sh --quick --no-mail < /dev/null > /dev/null 2>&1 &'
```

`--quick` runs `QUICK_FAMILIES` (Parakeet TDT, Supertonic 3 and one Audio8 pair) with one timed iteration into
`runs/<date>-quick`; `--families a,b` and `--backends a,b` narrow a run further, `--no-update` keeps the current
checkouts, `--label NAME` keeps several runs of one day apart. Watch `runs/<date>-quick/weekly.log` and wait for
`DONE` or `FAILED`; a family's own output is in `results/<backend>/<family>.log`.

## Where the results go

Each run leaves `report.md` and `summary.json` under `runs/<date>/` on the machine. The speech-bench skill (a
scheduled Monday task) copies every machine's summaries into a history folder, turns them into the dashboards' data
files with `build-dashboard-data.py`, uploads each machine's files to that machine's dashboard and republishes the
hub from `build-landing.py`, so the hub and its dashboards are the place to look. E-mail is optional and off by default: with `MAIL_TO`, `MAIL_FROM` and the SMTP keys set in
`config.env` (tether.io mail is Google Workspace, so `smtp.gmail.com:465` with an app password in a mode 0600 file),
`send-report.py` also mails the report with `report.md` and `summary.json` attached, and a failed run mails a
one-paragraph notice.

## Narrative by Claude

With `ANTHROPIC_API_KEY_FILE` set, `narrate.py` sends `report.md` to the Messages API (model
`NARRATIVE_MODEL`, `claude-opus-5-5` by default, with the server-side refusal fallback enabled) and asks for at
most twelve plain lines: the overall state, every failed cell, every metric that moved by more than the threshold
with the most likely cause given the pins that changed, and what to look at this week. The call uses the standard
library over HTTPS because the bench machines carry no SDK; any error or a missing key skips the narrative and the
report still goes out.

## Reading a report

- **Pins** names the engine and ggml commits, the CUDA architecture, the registry client, the TTS scorer pin and
  every model file with its size, sha256 and origin (registry key, HuggingFace URL, repo fixture or derived cut).
- **Results on &lt;backend&gt;** has one row per family: status, wall median with its min and max, speed (seconds of
  audio per second of wall clock), peak RSS, GPU memory rise and peak, quality, the backend the driver observed in
  the logs, and the iteration count.
- **Speed across backends** is the same matrix transposed for a quick look.
- **Changes** compares every ok cell with the previous finished run: wall median and peak RSS moves over 5%, and
  whether the engine, ggml or model pins changed in between. A slower cell with unchanged pins usually means
  another user loaded the machine; a changed pin points at a commit or a model cut.
- **Problems** lists build failures, timeouts and every non-ok cell with the driver's note.

## Adding or changing a model

Add a family to `families.json` with the driver fields, the weekly fields and its `provision` list, keep registry
keys exact (`qvac_models_compiled/ggml/<key>`), pin HuggingFace files to a revision, and list a derived file after
the file it is made from. `test-desktop-weekly.sh` checks that every `${MODEL_DIR}` file in `args` is provisioned
and that requantized cuts follow their source. Families run in the order of the file.

## Shared-machine rules

- Both machines are shared. The run waits for the lock and for other users' GPU work; never delete someone else's
  lock or kill their processes.
- Keep `BENCH_HOME` small; a shared `MODELS_DIR` saves disk but trusts every writer of that folder.
- No root: everything installs under `BENCH_HOME` and the shared model folder.
- One weekly run per machine is enough; run on demand with `--quick` or `--families` during working hours only
  after checking `who` and `nvidia-smi`.

## Mobile

The matrix and the report are host-agnostic: a mobile lane would keep `families.json` and `report.py`, replace the
build and run steps with the Android or iOS harnesses, and feed `summary.json` rows from the device. AWS Device
Farm can host that lane later; nothing here assumes a particular device host beyond `run-family.sh`.

## Tests

```sh
scripts/benchmarks/desktop-weekly/test-desktop-weekly.sh
```

Checks that every script parses, that `families.json` is complete and self-consistent, exercises the lock, the
watchdog, the build markers, run selection and the offline model provisioning path against temporary
directories, and runs the Python unit tests for `summarize.py`, `report.py`, `narrate.py`, `send-report.py` and
`prepare-derived.py`. The desktop benchmark workflow runs the same script in its plan job.
