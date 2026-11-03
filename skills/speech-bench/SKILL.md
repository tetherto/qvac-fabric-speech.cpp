---
name: speech-bench
description: Run, collect and publish the weekly desktop benchmark of the qvac-fabric-speech.cpp engines (Parakeet, TranscribePsy, Nemotron, Supertonic, Audio8, MOSS-TTS, ACE-Step) on the team's shared Tailscale machines, the RTX 5090 Linux box and the Mac Studio M3 Ultra, and keep the weekly results artifact current. Use when asked to benchmark the speech engines, check this week's speech numbers, rerun a disturbed cell, deploy the bench on a machine, or update the speech benchmark page.
argument-hint: collect | publish | run <linux|mac> [quick] | deploy <linux|mac>
---

# Speech weekly benchmark on the shared Tailscale machines

Every Sunday from 18:00 CET, cron on each shared machine runs `scripts/benchmarks/desktop-weekly/weekly.sh`
under the installing user's own account: it updates the engine `master` and ggml `speech` heads, provisions the
pinned models, builds every backend the machine has, drives `scripts/benchmarks/run-family.sh` over the matrix
in `families.json`, and writes `~/speech-bench/runs/<date>/summary.json` plus `report.md`. No developer
machine is involved in the measurement. The dashboards behind the weekly landing page, linked from the team channel's canvas,
are where everyone reads the numbers. This skill is the Claude side: on-demand runs, collecting the summaries, building the
results page from them and publishing it, and the rules that keep two automated benches and twenty-odd people from
disturbing each other on the same hardware.

| path | what it is |
| --- | --- |
| `scripts/benchmarks/desktop-weekly/README.md` | the full description of the bench, its layout on a machine and its configuration |
| `scripts/benchmarks/desktop-weekly/weekly.sh` | the entry point cron runs; `--quick`, `--families`, `--backends`, `--no-update`, `--no-mail`, `--label` |
| `scripts/benchmarks/desktop-weekly/families.json` | the matrix: 20 families with their driver spec, iteration counts, timeouts and model provisioning |
| `scripts/benchmarks/desktop-weekly/report.py` | `summary.json` (+ previous run) -> Markdown report with the changes over 5% and whether a pin moved |
| `scripts/benchmarks/desktop-weekly/build-dashboard-data.py` | a folder of `summary.json` files -> the JSON files the task dashboards read (latest result per lane, every result, campaign rows, runs) |
| `scripts/benchmarks/desktop-weekly/build-landing.py` | a folder of `summary.json` files -> the landing page (dashboard links, every machine's newest run, how to read it) |
| `scripts/benchmarks/desktop-weekly/dashboard/index.html`, `dashboards.json` | the page every task dashboard runs, and the links of the landing page and the three dashboards |
| `scripts/benchmarks/desktop-weekly/install.sh` | deploys the tools to `~/speech-bench/tools` and installs the crontab line |

## Machines, schedule and coordination

| machine | host | backends | clock | config.env | notes |
| --- | --- | --- | --- | --- | --- |
| DGX Spark (GB10, 128 GB unified) | `spark-d9ca` | cpu, cuda | UTC | `BACKENDS=cpu,cuda`, `CUDA_ARCH=121a-real`, `CUDA_HOME=/usr/local/cuda-13.3`, `MEMORY_SAMPLER=meminfo` | nvidia-smi reports no memory.used; weights load at about 0.5 GB/s |
| RTX 5090 box (2x 32 GB, GPU 0 used) | `qvac-dev-linux-x64` | cpu, cuda, vulkan | UTC | `MODELS_DIR=/home/shared/models/qvac-speech/bench` | 22-account `dev` group can write the shared model folder |
| Strix Halo (Radeon 8060S, 128 GB) | `qvac-dev-strix-0` | cpu, vulkan | UTC | `BACKENDS=cpu,vulkan` | no cron: `install.sh` starts `scheduler.sh`, which does not survive a reboot |
| Mac Studio M3 Ultra (96 GB) | `qvac-dev-mac-arm64` | cpu, metal | CET | `MODELS_DIR=/Users/Shared/models/qvac-speech/bench` | shares Saturday with the fabric-llm bench |
| Mac mini M4 (16 GB) | `mac-mini-m4-gpu` | cpu, metal | CET | `SKIP_FAMILIES=moss-tts-f16,moss-tts-q8_0,moss-tts-clone-f16` | macOS 15: `/bin/date` has no %N, the bench routes `run-family.sh` through `nsdate` |

Every machine runs the speech bench on Sunday from 18:00 CET (17:00 on the UTC clocks: `install.sh --hour 17`
there, `--hour 18` on the Macs), about 6 to 7 hours. The fabric-llm bench (Aman's `engine-bench`) owns Saturday
from 18:00 CET on the Mac Studio, the Strix Halo and the DGX Spark. The bench folder is `~/speech-bench` everywhere,
with runs in `runs/<date>` and models in `models/` unless `MODELS_DIR` says otherwise.

Two automated benches share three of the machines. They are kept apart by **day** (LLM on Saturday, speech on Sunday, both
from 18:00 CET) and by the **lock** `/tmp/gpu-bench.lock`: each bench waits while the file exists or another
user's GPU job runs, creates the file atomically when it starts measuring, and removes it when it is done. A
contaminated run is one where something else ran on the machine during the measurement; the report flags it as a
wall-clock move with unchanged pins. The rules for every Claude session and every person:

- **Never start a run on either machine from Saturday 17:00 CET to Monday 02:00 CET**, which covers both
  weekly benches with margin. Rerun a disturbed cell on a weekday instead.
- **Check before any on-demand run:** `who`, `ps -eo user,pcpu,etime,comm --sort=-pcpu | head` (macOS:
  `ps -Ao user,pcpu,etime,comm -r | head`), `nvidia-smi` on linux, and `ls -l /tmp/gpu-bench.lock`. If someone
  else is running GPU work, wait or ask them in the channel. Never delete someone else's lock and never kill
  another user's process.
- **Ask your user before a run longer than about 30 minutes during European working hours.** `--quick` is
  about 3 minutes once models and builds are in place; a full `weekly.sh` is 6 to 7 hours.
- **One weekly schedule per bench per machine.** Do not add a second crontab line or a second scheduled task
  that runs the bench; change the hour with `install.sh --hour` if the day split ever moves.
- **Do not change the other bench's files or lock.** If the LLM bench's schedule moves, agree the new split with
  its owner first and update this file and the README together.
- **No root, no `sudo`, nothing installed outside `~/speech-bench` and the shared model folder.**

## Procedure

Arguments: `collect`, `publish`, `run <linux|mac> [quick]` or `deploy <linux|mac>`. Without an argument, ask
which one. SSH as yourself with the host names above.

### collect

Copy every finished run's summary into the local history folder, one file per run, named by host and run:

```sh
H=qvac-dev-linux-x64   # and spark-d9ca, qvac-dev-strix-0, qvac-dev-mac-arm64, mac-mini-m4-gpu
HISTORY=${SPEECH_BENCH_HISTORY:-$HOME/speech-bench-history}
mkdir -p $HISTORY/summaries/$H
for run in $(ssh $H 'cd ~/speech-bench/runs && for d in */; do [ -e "$d/DONE" ] && echo "${d%/}"; done'); do
  scp -q $H:speech-bench/runs/$run/summary.json $HISTORY/summaries/$H/$run.json
done
```

The history folder is the input of `build-dashboard-data.py` and `build-landing.py`. Keep it; a run pruned on the machine (the bench keeps
12) stays in the page's history only through this folder. Also copy `report.md` of the newest run when the user
wants to read it.

### publish

Build the dashboards' data files and the landing page from the history folder:

```sh
HISTORY=${SPEECH_BENCH_HISTORY:-$HOME/speech-bench-history}
D=scripts/benchmarks/desktop-weekly
python3 $D/build-dashboard-data.py $HISTORY/summaries -o $HISTORY/dashboard-data
python3 $D/build-landing.py $HISTORY/summaries -o $HISTORY/speech-weekly-landing.html
```

The links are in `$D/dashboards.json`: `landing` (the page linked from the canvas) and one dashboard per task,
`asr`, `tts` and `music`. Each dashboard is a Dashboard-type artifact whose page reads four attached files, the
datasets `latest`, `weekly`, `campaign` and `runs`. Refresh them in place, never as new artifacts:

1. For each task, upload the four files of `$HISTORY/dashboard-data/<task>/` (`runs.json` lives one level up and
   is the same for every task) with the Artifact tool: `url` = that dashboard, `asset: true`, `file_path` = the
   file. Each upload answers with a `/_blob/<id>` url.
2. Point the datasets at the new files with the ArtifactData tool: read `datasets/<id>` (for its `version`), then
   `update` it with `if_version`, `source.url` = the new blob url, `source.name` = the file name,
   `updated: {at: <UTC ISO>, by: "Claude (<run date>)"}` and `edit: {via: "claude", at: <UTC ISO>}`. The
   `campaign` dataset is `fixed: true`; upload it only when `comparisons.json` changed.
3. Only when `$D/dashboard/index.html` changed: read `files/index.html` and `set` it to
   `{"text": <the file>, "edit": {...}}` with `if_version`, on all three dashboards.
4. Read the landing artifact (`landing` in `dashboards.json`) with the Artifact tool, then publish
   `$HISTORY/speech-weekly-landing.html` with the same `url` (no icon). If the publish is refused because a newer
   version is live, publish again: the page is rebuilt from the history folder and contains everything.

Each dashboard shows, for one task, a card per machine and backend with the newest result of every model (speed in
seconds of audio per second of wall clock, wall time, peak RSS or WER by the metric buttons; GPU lanes first, the
lane buttons switch to CPU), the campaign comparison against other engines coloured per engine (transcribe.cpp,
audio.cpp and LocalAI parakeet.cpp for Parakeet TDT; audio.cpp for Supertonic 3 and Audio8; acestep.cpp for
ACE-Step), the week-over-week line per lane for a chosen model, and the run log; the Sources panel shows every
file and row behind a number. Tell the user what changed since the previous week: failed cells, moves over 5%,
and which pins moved. To refresh a comparison, re-run the campaign's measurement for that model and lane and
replace the row (`qvac_rtf`, `ratio`, versions, `measured`, `source`) in `comparisons.json`; TranscribePsy,
Nemotron and MOSS-TTS have no campaign rows yet. `build-artifact.py` still writes a single self-contained HTML
report from the same folder for reading offline.

### run

On demand, outside the protected window and after the checks above. Start it detached so it survives the SSH
session, and wait for `DONE` or `FAILED` without polling in the foreground:

```sh
ssh <host> 'nohup /bin/bash ~/speech-bench/tools/weekly.sh --quick --no-mail --label manual < /dev/null > /dev/null 2>&1 &'
```

`--quick` runs three families with one timed iteration into `runs/<date>-quick-manual`; leave `--quick` out for
the full matrix, or narrow it with `--families a,b` and `--backends a,b`. The log is `runs/<run>/weekly.log`; a
family's own output is `results/<backend>/<family>.log`. A labelled or quick run never becomes the baseline of
the next weekly report, and the page lists it under its own name. To rerun a disturbed cell of a weekly run,
delete that cell's `results/<backend>/<family>.json` on the machine and rerun `weekly.sh --no-update` with the
same date (it resumes, keeping the ok cells and rebuilding nothing when the commits did not move).

### deploy

```sh
scp -r scripts/benchmarks/desktop-weekly <host>:speech-bench-src
ssh <host> 'bash speech-bench-src/install.sh --hour 18 && rm -rf speech-bench-src'   # --hour 17 on linux (UTC)
```

Then set the machine's keys in `~/speech-bench/config.env` (see the machines table) and try
`weekly.sh --quick --no-mail` once. Results reach people through the artifact page, not by mail; `MAIL_TO` and
`ANTHROPIC_API_KEY_FILE` stay unset unless someone asks for mail or the narrative. The README has the prerequisites
and the layout.

## Reading a run

`summary.json` (schema 1): `run` (name, host, started, finished, status ok/partial/failed), `machine`,
`pins` (engine and ggml commits, CUDA arch, registry client, venv packages, every model file with bytes, sha256
and origin), `backends`, `families`, `results` (one row per backend and family: status, observed backend,
wall median/min/max in ms, rtf, audio seconds, speed_x, peak RSS MiB, GPU memory peak and rise MiB, quality
kind and value, runs, notes), `counts`, `build_failures`, `problems`.

- Speed is seconds of audio per second of wall clock. ASR and Supertonic rows time inference only; Audio8,
  MOSS and ACE-Step rows are process wall clock including model load, so their speed is a throughput figure, not
  a latency.
- A slower cell with unchanged pins points at load from another user; a changed pin names the engine, ggml or
  model cut to look at. Report every number as measured and say what was disturbed or not like for like.
- Two rows are not what the ticket asked for because the registry has no such cut: ACE-Step Base runs at
  Q4_K_M, and MOSS-TTS Q8_0 is the F16 backbone with its transformer blocks requantized on the machine.

## Rules

- Benchmarks only: do not commit, push or open pull requests from a benchmark session.
- Publish only to the artifact named above, and only when the user asks or the scheduled task runs.
- Never start a run inside the weekend window, never touch other users' files, locks or processes, and never run
  anything as root.
