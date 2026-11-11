# Speech bench results

Results of the weekly desktop speech benchmark on the shared lab machines, and the workflow that publishes the
Speech Bench site from them. The machines push here themselves when their Sunday run ends; nothing else writes.

| Path | What it holds |
| --- | --- |
| `summaries/<host>/<run>.json` | the summary of a finished run: every cell with its timings, memory, quality, pins and model hashes |
| `reports/<host>/<run>.md` | the Markdown report of that run |
| `failures/<host>/<run>.txt` | the reason a run failed and the end of its log |
| `raw/` | copies kept for reference, such as a summary that was later corrected |
| `report-rows.json` | the rows copied from the comparison report of 2026-11-04 |

`.github/workflows/publish-site.yml` runs on every push. Once every machine has delivered the week's summary or
failure notice, it builds the site with `build-site.py` from qvac-fabric-speech.cpp and deploys it to this
repository's GitHub Pages site, which only people with read access to this repository can open. Every Monday at
08:00 UTC it publishes whatever has arrived, and "Run workflow" on the Actions tab publishes on demand.

Each machine pushes with a deploy key of its own; its `~/speech-bench/config.env` names this repository in
`RESULTS_REPO` and the key in `RESULTS_KEY_FILE`. This README and the workflow come from
`scripts/benchmarks/desktop-weekly/results-repo/` in qvac-fabric-speech.cpp; change them there.
