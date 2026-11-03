#!/usr/bin/env python3
"""Write the landing page of the weekly speech bench: links to the task dashboards, the state of every machine's
newest stored run, and how to read the numbers.

usage: build-landing.py <summaries dir> -o landing.html [--dashboards dashboards.json]

dashboards.json maps the task keys asr, tts and music to their dashboard links; the landing page links each one
with the models it covers and which machines have a result for it.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib

import importlib.util

HERE = pathlib.Path(__file__).resolve().parent
SHA_SHORT = 10
TASK_CARDS = [
    ("asr", "ASR", "speech to text", "Parakeet TDT 0.6B v3 Q8_0, Parakeet Unified English 0.6B Q8_0, TranscribePsy-250M-EU-sub Q8_0, Nemotron 3.5 ASR Streaming 0.6B F16; against transcribe.cpp, audio.cpp and LocalAI parakeet.cpp from the campaign"),
    ("tts", "TTS", "text to speech and voice cloning", "Supertonic 3 Q8_0, Audio8 TTS Preview 0.6B (LM F16, Q8_0, Q4_0 by codec decoder F16, Q8_0, cloning with encoder F16, Q8_0), MOSS-TTS v1.5 8.5B F16, Q8_0 blocks and cloning; against audio.cpp from the campaign"),
    ("music", "Music", "music generation", "ACE-Step 1.5 Turbo Q8_0 and Q4_K_M, SFT Q8_0, Base Q4_K_M; against acestep.cpp from the campaign"),
]
STATUS_CLASS = {"ok": "good", "partial": "warn", "failed": "bad"}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


data = load_module("build_dashboard_data", "build-dashboard-data.py")


def esc(text):
    return html.escape(str(text if text is not None else ""), quote=True)


def newest_per_host(summaries):
    newest = {}
    for summary in summaries:
        newest[summary["_host"]] = summary
    return [newest[host] for host in sorted(newest, key=lambda h: data.machine_label(h))]


def machines_with_task(summaries, task_key):
    hosts = {s["_host"] for s in summaries for r in s.get("results", []) if data.TASK_KEYS.get(r.get("task", "")) == task_key and r.get("status") == "ok"}
    return sorted(data.machine_label(h) for h in hosts)


def task_card(summaries, dashboards, key, name, kind, models):
    url = dashboards.get(key, "")
    machines = machines_with_task(summaries, key)
    coverage = f"results from {', '.join(machines)}" if machines else "no stored result yet"
    link = f'<a class="card-link" href="{esc(url)}">Open the {esc(name)} dashboard</a>' if url else '<span class="note">dashboard link not set</span>'
    return (f'<article class="card"><h2>{esc(name)} <span class="sub">{esc(kind)}</span></h2><p class="note">{esc(models)}</p>'
            f'<p class="note">{esc(coverage)}</p>{link}</article>')


def task_cards(summaries, dashboards):
    return "".join(task_card(summaries, dashboards, *card) for card in TASK_CARDS)


def machine_tile(summary):
    run, counts, pins = summary.get("run", {}), summary.get("counts", {}), summary.get("pins", {})
    engine, ggml = (pins.get("engine") or {}).get("sha", "")[:SHA_SHORT], (pins.get("ggml") or {}).get("sha", "")[:SHA_SHORT]
    status = run.get("status", "")
    machine = summary.get("machine", {})
    hardware = " · ".join(x for x in [machine.get("os_version") or machine.get("os", ""), machine.get("cpu", ""), (machine.get("gpu") or {}).get("name", "")] if x)
    return (f'<div class="tile"><span class="tile-host">{esc(data.machine_label(summary["_host"]))}<span class="sub">{esc(summary["_host"])}</span></span>'
            f'<span class="tile-status {STATUS_CLASS.get(status, "")}">{esc(status)}</span>'
            f'<span class="tile-big num">{counts.get("ok", 0)}<small>/{counts.get("expected", 0)} cells ok</small></span>'
            f'<span class="sub">{esc(run.get("name", ""))} · {esc(", ".join(summary.get("backends", [])))} · engine <code>{esc(engine)}</code> · ggml <code>{esc(ggml)}</code></span>'
            f'<span class="sub">{esc(hardware)}</span></div>')


def machine_tiles(summaries):
    return "".join(machine_tile(summary) for summary in newest_per_host(summaries))


def problems_list(summaries):
    items = [(data.machine_label(s["_host"]), p) for s in newest_per_host(summaries) for p in (s.get("problems") or [])]
    if not items:
        return '<p class="note">none in the newest runs</p>'
    return "<ul class=\"problems\">" + "".join(f"<li><b>{esc(m)}</b>: {esc(p)}</li>" for m, p in items) + "</ul>"


def render(summaries, dashboards, built_at):
    return f'''<title>Speech Fabric Weekly Bench</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {{ --bg: #f6f7f5; --surface: #ffffff; --fg: #1d2420; --fg-2: #5a645e; --line: #d9ded9; --accent: #1f6f78; --good: #1b7f4a; --warn: #b86a00; --bad: #b3261e;
  --display: "Barlow Semi Condensed", "Arial Narrow", sans-serif; --body: "Source Sans 3", "Segoe UI", sans-serif; --mono: "JetBrains Mono", Menlo, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #141817; --surface: #1d2321; --fg: #e8ebe8; --fg-2: #a7b0aa; --line: #333b38; --accent: #6cc3cc; --good: #52c27a; --warn: #e3a24a; --bad: #ef7a73; color-scheme: dark; }} }}
:root[data-theme="dark"] {{ --bg: #141817; --surface: #1d2321; --fg: #e8ebe8; --fg-2: #a7b0aa; --line: #333b38; --accent: #6cc3cc; --good: #52c27a; --warn: #e3a24a; --bad: #ef7a73; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 15px; line-height: 1.45; margin: 0;
  padding-block: 24px 48px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
main {{ max-width: 1100px; margin: 0 auto; display: grid; gap: 28px; }}
h1, h2, h3 {{ font-family: var(--display); text-wrap: balance; margin: 0; line-height: 1.1; }}
h1 {{ font-size: 2.4rem; }} h2 {{ font-size: 1.6rem; color: var(--accent); }} h2 .sub {{ display: inline; font-family: var(--mono); font-size: .8rem; margin-left: 8px; color: var(--fg-2); }}
h3 {{ font-size: 1.1rem; text-transform: uppercase; letter-spacing: .06em; color: var(--fg-2); }}
p {{ margin: 0; max-width: 72ch; }} .note {{ color: var(--fg-2); font-size: .95rem; }}
header {{ display: grid; gap: 10px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 14px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; display: grid; gap: 8px; min-width: 0; align-content: start; }}
.card-link {{ font-family: var(--display); font-size: 1.1rem; font-weight: 700; color: var(--accent); text-decoration: none; margin-top: 4px; }}
.card-link:hover {{ text-decoration: underline; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)); gap: 12px; }}
.tile {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; display: grid; gap: 4px; min-width: 0; }}
.tile-host {{ font-weight: 600; }} .tile-status {{ font-family: var(--display); font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }}
.tile-big {{ font-size: 1.8rem; font-weight: 600; }} .tile-big small {{ font-size: .9rem; color: var(--fg-2); font-weight: 400; margin-left: 4px; }}
.good {{ color: var(--good); }} .warn {{ color: var(--warn); }} .bad {{ color: var(--bad); }}
.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }} .sub {{ display: block; color: var(--fg-2); font-size: .82rem; overflow-wrap: anywhere; }}
section {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; min-width: 0; display: grid; gap: 10px; }}
section.plain {{ background: transparent; border: 0; padding: 0; }}
ul {{ margin: 0; padding-left: 20px; }} li {{ margin: 4px 0; }} code {{ font-family: var(--mono); font-size: .85em; }}
a {{ color: var(--accent); }} :focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<main>
<header><h1>Speech Fabric Weekly Bench</h1>
<p>Every Sunday from 18:00 CET the shared Tailscale machines benchmark the qvac-fabric-speech.cpp engines at the engine master and ggml speech heads of that night, with no developer machine involved. The numbers live in one dashboard per task, each with a card per machine and backend, the campaign comparison against other engines, the week-over-week lines and the run log. The fabric-llm engine bench owns Saturday on the same machines. Last build {esc(built_at)}.</p></header>
<section class="plain"><div class="cards">{task_cards(summaries, dashboards)}</div></section>
<section><h2>Machines <span class="sub">newest stored run of each</span></h2><div class="tiles">{machine_tiles(summaries)}</div>
<h3>Problems reported by the newest runs</h3>{problems_list(summaries)}</section>
<section id="method"><h2>How to read it</h2>
<ul>
<li><b>Speed</b> is seconds of audio per second of wall clock: above 1 is faster than real time. ASR and Supertonic time inference only; Audio8, MOSS-TTS and ACE-Step time the whole process including model load, so their speed is a throughput figure, not a latency.</li>
<li><b>Memory</b> is the process peak RSS; on the CUDA boxes the GPU memory rise during the cell is sampled with nvidia-smi (the DGX Spark and the Jetson sample system memory instead, as their unified memory is not reported per process).</li>
<li><b>Quality</b> is the WER of the fixed clip for ASR and the WER of pinned Whisper tiny over the generated clip for TTS, outside the timed region.</li>
<li><b>Lanes</b> are machine and backend: CUDA on the DGX Spark, the RTX 5090 box and the Jetson, Vulkan on the Strix Halo (and the 5090 and Jetson), Metal on the Macs, plus a CPU lane on every machine. Each dashboard shows the GPU lanes first; the lane buttons switch to CPU.</li>
<li><b>Other engines</b> are the rows of the QVAC speech engines campaign (Gustavo Araujo, October 2026 edition): both engines on the same device with the same inputs, dated, not on the weekly clock. Compare engines within a device card, not with the weekly cards.</li>
<li><b>Deviations from the ticket:</b> ACE-Step Base runs at Q4_K_M because the registry has no Q8_0 Base cut, and MOSS-TTS Q8_0 is the F16 backbone with its transformer blocks requantized on the machine.</li>
<li><b>Where it comes from:</b> <code>scripts/benchmarks/desktop-weekly</code> in <code>qvac-fabric-speech.cpp</code> (the bench), its <code>speech-bench</code> skill (collecting and publishing), and each machine's <code>~/speech-bench/runs/&lt;run&gt;/summary.json</code>. The numbers reach the dashboards through their attached files, uploaded on Monday morning; the Sources panel of each dashboard shows every file and row.</li>
</ul></section>
</main>
'''


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--dashboards", default=str(HERE / "dashboards.json"))
    return parser.parse_args(argv)


def load_dashboards(path):
    if not pathlib.Path(path).is_file():
        return {}
    with open(path) as handle:
        return json.load(handle)


def main(argv=None):
    args = parse_args(argv)
    summaries = data.load_summaries(args.summaries)
    if not summaries:
        raise SystemExit(f"no summary.json files under {args.summaries}")
    built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pathlib.Path(args.out).write_text(render(summaries, load_dashboards(args.dashboards), built_at))
    print(f"wrote {args.out} from {len(summaries)} runs")


if __name__ == "__main__":
    main()
