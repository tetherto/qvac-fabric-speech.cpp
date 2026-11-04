#!/usr/bin/env python3
"""Write the hub page of the weekly speech bench from hub.json (the output of build-dashboard-data.py).

usage: build-landing.py <hub.json> -o landing.html

The hub lists, per task, every machine with a link into its dashboard's tab (how many models, which backends, how many
other engines), then the quality flags, the engines with where they were measured, what each dashboard shows and how the
bench runs.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import pathlib

MACHINE_ORDER = ["DGX Spark", "RTX 5090", "Strix Halo", "Jetson Orin", "Mac Studio M3 Ultra", "Mac mini M4"]
TASKS = [("tts", "TTS and voice cloning", "Speech synthesis from a fixed English prompt and voice cloning from the JFK clip: Supertonic 3, the Audio8 LM and codec cuts, MOSS-TTS."),
         ("asr", "ASR", "Speech to text on the 11 s JFK clip with its reference transcript: Parakeet TDT and Unified, TranscribePsy, Nemotron 3.5 streaming."),
         ("music", "Music generation", "A 30 s render as a fresh process including model load: ACE-Step 1.5 Turbo, SFT and Base.")]
BACKEND_LABELS = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan", "metal": "Metal", "coreml": "Core ML"}
ENGINE_CLASSES = {"qvac-fabric-speech.cpp": "e-qvac", "audio.cpp": "e-audiocpp", "acestep.cpp": "e-acestep", "LocalAI / parakeet.cpp": "e-localai",
                  "transcribe.cpp": "e-transcribe"}
QUALITY_FLAG_PCT = 10
HERE = pathlib.Path(__file__).resolve().parent
STYLE_FILE = HERE / "dashboard" / "hub.css"


def esc(text):
    return html.escape(str(text if text is not None else ""), quote=True)


def fmt_speed(value):
    if value is None:
        return "n/a"
    return f"{value:,.0f}×" if value >= 100 else f"{value:.1f}×" if value >= 10 else f"{value:.2f}×"


def machine_rank(machine):
    return MACHINE_ORDER.index(machine) if machine in MACHINE_ORDER else len(MACHINE_ORDER)


def backends_text(backends):
    return ", ".join(BACKEND_LABELS.get(b, b) for b in backends)


def machine_link(machine, task_key):
    counts = machine["tasks"].get(task_key, {})
    if not counts.get("rows") or not machine.get("url"):
        return f'<li><span class="off"><b>{esc(machine["machine"])}</b><small>{"collecting" if not counts.get("rows") else "dashboard link not set"}</small></span></li>'
    others = counts.get("others", 0)
    detail = f'{counts["models"]} model{"s" if counts["models"] != 1 else ""} · {esc(backends_text(counts["backends"]))} · {others} other engine{"s" if others != 1 else ""}'
    return f'<li><a href="{esc(machine["url"])}#{task_key}" target="_blank" rel="noopener"><b>{esc(machine["machine"])}</b><small>{detail}</small></a></li>'


def task_box(hub, task_key, title, blurb):
    machines = sorted(hub["machines"], key=lambda m: machine_rank(m["machine"]))
    return (f'<section class="tbox" id="{task_key}"><h2>{esc(title)}</h2><p>{esc(blurb)}</p><ul>'
            + "".join(machine_link(m, task_key) for m in machines) + "</ul></section>")


def task_boxes(hub):
    return '<div class="boxes">' + "".join(task_box(hub, *task) for task in TASKS) + "</div>"


def engine_span(engine):
    return f'<span class="eng {ENGINE_CLASSES.get(engine, "e-other")}"><i></i>{esc(engine)}</span>'


def flag_item(flag):
    cloning = " (cloning)" if flag["task"] == "TTS cloning" else ""
    return (f'<li>{engine_span(flag["engine"])} <b>{esc(flag["model"])}</b> ({esc(flag["variant"])}) on {esc(flag["machine"])} {esc(flag["backend"])}{cloning}: '
            f'intelligibility WER <b>{flag["wer_pct"]:.1f}%</b> at {esc(fmt_speed(flag["speed_x"]))} ({esc(flag["wer_kind"])}).</li>')


def quality_section(hub):
    flags = hub.get("quality_flags", [])
    body = '<ul class="flags">' + "".join(flag_item(f) for f in flags) + "</ul>" if flags else f'<p class="muted small">No TTS row reaches the {QUALITY_FLAG_PCT}% intelligibility WER flag this week.</p>'
    return (f'<section class="block" id="quality"><h2>Quality flags</h2><p class="muted small">Fast results that come with a quality problem: TTS rows with '
            f'Whisper-tiny intelligibility WER at or above {QUALITY_FLAG_PCT}%. Intelligibility WER does not measure naturalness or speaker similarity.</p>{body}</section>')


def engine_row(engine):
    return (f'<tr><td>{engine_span(engine["engine"])}</td><td>{esc(engine["status"])}</td><td>{esc(engine["measured_on"] or "none")}</td>'
            f'<td>{esc(engine["details"])}<br><span class="muted">Source: {esc(engine["source"])}</span></td></tr>')


def engines_table(title, rows):
    if not rows:
        return ""
    return (f'<h3>{esc(title)}</h3><div class="tbl"><table><thead><tr><th>Engine</th><th>This week</th><th>Measured on</th><th>Scope and caveats</th></tr></thead>'
            f'<tbody>{"".join(engine_row(e) for e in rows)}</tbody></table></div>')


def engines_section(hub):
    engines = hub.get("engines", [])
    qvac = [e for e in engines if e["role"] == "QVAC engine"]
    prior = [e for e in engines if e["role"] == "prior-report competitor"]
    new = [e for e in engines if e["role"] not in ("QVAC engine", "prior-report competitor")]
    return f'<section class="block" id="engines"><h2>Engines</h2>{engines_table("Weekly engine", qvac)}{engines_table("From the previous comparison", prior)}{engines_table("New competitors", new)}</section>'


def dashboard_links(hub):
    machines = sorted((m for m in hub["machines"] if m.get("url")), key=lambda m: machine_rank(m["machine"]))
    return ", ".join(f'<a href="{esc(m["url"])}" target="_blank" rel="noopener">{esc(m["machine"])}</a>' for m in machines)


def pins_lines(hub):
    machines = sorted(hub["machines"], key=lambda m: machine_rank(m["machine"]))
    return "".join(f'<li><b>{esc(m["machine"])}</b>: run {esc(m["newest_run"])} ({esc(m["status"])}, {m["ok"]} of {m["expected"]} cells) at {esc(m["pins"])}.</li>' for m in machines)


def method_section(hub):
    return f'''<section class="method" id="method">
    <div>
      <h3>What each dashboard shows</h3>
      <ul>
        <li>{dashboard_links(hub)}: an overview of the machine (status, a summary of the week, quality flags, which models ran on which backend, gaps, engines), then ASR, TTS and music tabs and a Runs and coverage tab.</li>
        <li>Each task tab has one section per model: a speed chart per backend and input, the weekly charts with QVAC's variants and the campaign charts with every engine in its color, and a details panel with precision per row, timed runs and warmups, generated duration, a qualified time ratio and comments.</li>
        <li>A week-over-week line per backend follows each task, with buttons for speed, latency, RTF, host memory, device memory and WER.</li>
        <li>Every chart opens a Sources panel with exact values, source file, engine commits, model hashes, command, configuration and caveats; the Runs and coverage tab lists every row and every cell that was skipped or failed.</li>
      </ul>
    </div>
    <div>
      <h3>How it runs</h3>
      <ul>
        <li>Every Sunday from 18:00 CET each shared machine updates the engine master and ggml speech heads, builds every backend it has, provisions the pinned models and runs the matrix of <code>scripts/benchmarks/desktop-weekly/families.json</code>; a Claude task collects the summaries on Monday morning and refreshes these pages.</li>
        <li>Weekly rows and campaign rows are never in one chart: the campaign rows keep the build, inputs and dates of the QVAC speech engines report (October 2026 edition) and are compared only among themselves.</li>
        <li>ASR, Supertonic and Nemotron rows time inference only; Audio8, MOSS-TTS and ACE-Step rows are whole fresh processes including model load from a warm filesystem cache.</li>
        <li>Quality checks are narrow: ASR WER on one English clip, TTS intelligibility on one prompt. Music quality and speaker similarity are unmeasured.</li>
        <li>Deviations from the ticket: ACE-Step Base runs at Q4_K_M because the registry has no Q8_0 Base cut; MOSS-TTS Q8_0 is the F16 backbone with its transformer blocks requantized on the machine.</li>
        <li>Newest runs and pins: <ul>{pins_lines(hub)}</ul></li>
      </ul>
    </div>
  </section>'''


def render(hub, built_at):
    style = STYLE_FILE.read_text()
    machines = len(hub["machines"])
    return f'''<title>Speech Bench</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
{style}
</style>
<div class="wrap">
  <header>
    <div class="eyebrow">QVAC speech · weekly bench · newest run {esc(hub["measured"])} · {machines} machines · {hub["accepted"]} accepted measurements · page built {esc(built_at)}</div>
    <h1>Speech Bench</h1>
    <p class="lede">QVAC (qvac-fabric-speech.cpp) every week on the shared machines, with the campaign rows against other speech engines. Pick a task, then a machine: each machine dashboard shows, per model and backend, the weekly numbers and how QVAC compared with the other engines on that machine.</p>
  </header>
  {task_boxes(hub)}
  <details class="more">
    <summary>Quality flags, engines and how it runs</summary>
  {quality_section(hub)}
  {engines_section(hub)}
  {method_section(hub)}
  </details>
</div>
'''


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hub")
    parser.add_argument("-o", "--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.hub) as handle:
        hub = json.load(handle)
    built_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pathlib.Path(args.out).write_text(render(hub, built_at))
    print(f"wrote {args.out} for {len(hub['machines'])} machines")


if __name__ == "__main__":
    main()
