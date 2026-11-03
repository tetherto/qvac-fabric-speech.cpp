#!/usr/bin/env python3
"""Write the dataset files the task dashboards read, from the stored summaries and comparisons.json.

usage: build-dashboard-data.py <summaries dir> -o <out dir> [--comparisons comparisons.json]

For every task (ASR, TTS with cloning, Music) it writes three JSON files of row objects under <out dir>/<task key>/:
weekly.json (one row per machine, backend and family of every stored run), latest.json (the newest result of every
machine, backend and family, flagged current when it comes from that machine's newest run) and campaign.json (the
paired competitor rows with speeds in seconds of audio per second of wall clock), plus runs.json with one row per stored run (pins,
status, counts) shared by every task. The dashboards attach these files as their datasets; the Monday task uploads
new versions after each Sunday run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import re

TASK_KEYS = {"ASR": "asr", "TTS": "tts", "TTS cloning": "tts", "Music": "music"}
MACHINE_LABELS = {"spark-d9ca": "DGX Spark", "qvac-dev-linux-x64": "RTX 5090", "qvac-dev-strix-0": "Strix Halo",
                  "qvac-dev-strix-1": "Strix Halo (1)", "qvac-dev-linux-arm64": "Jetson Orin Nano",
                  "qvac-dev-mac-arm64": "Mac Studio M3 Ultra", "mac-mini-m4-gpu": "Mac mini M4"}
FAMILY_LABELS = {"parakeet-tdt": "Parakeet TDT 0.6B v3 Q8_0", "parakeet-unified": "Parakeet Unified EN 0.6B Q8_0",
                 "transcribepsy": "TranscribePsy 250M Q8_0", "nemotron": "Nemotron 3.5 streaming 0.6B F16",
                 "supertonic3": "Supertonic 3 Q8_0", "audio8-lm-f16-dec-f16": "Audio8 LM F16, decoder F16",
                 "audio8-lm-f16-dec-q8_0": "Audio8 LM F16, decoder Q8_0", "audio8-lm-q8_0-dec-f16": "Audio8 LM Q8_0, decoder F16",
                 "audio8-lm-q8_0-dec-q8_0": "Audio8 LM Q8_0, decoder Q8_0", "audio8-lm-q4_0-dec-f16": "Audio8 LM Q4_0, decoder F16",
                 "audio8-lm-q4_0-dec-q8_0": "Audio8 LM Q4_0, decoder Q8_0", "audio8-clone-enc-f16": "Audio8 cloning, encoder F16",
                 "audio8-clone-enc-q8_0": "Audio8 cloning, encoder Q8_0", "moss-tts-f16": "MOSS-TTS 8.5B F16",
                 "moss-tts-q8_0": "MOSS-TTS 8.5B Q8_0 blocks", "moss-tts-clone-f16": "MOSS-TTS 8.5B cloning F16",
                 "acestep-turbo-q8_0": "ACE-Step Turbo Q8_0", "acestep-turbo-q4_k_m": "ACE-Step Turbo Q4_K_M",
                 "acestep-sft-q8_0": "ACE-Step SFT Q8_0", "acestep-base-q4_k_m": "ACE-Step Base Q4_K_M"}
FAMILY_ORDER = list(FAMILY_LABELS)
MIB_PER_GIB = 1024.0
PERCENT = 100.0
SHA_SHORT = 10
SPEED_DIGITS = 4
RTF_DIGITS = 6
RTF_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def load_summaries(folder):
    found = []
    for path in sorted(pathlib.Path(folder).glob("*/*.json")):
        with path.open() as handle:
            summary = json.load(handle)
        summary["_host"] = path.parent.name
        found.append(summary)
    return sorted(found, key=lambda s: (s["_host"], s.get("run", {}).get("started") or ""))


def load_comparisons(path):
    if not path or not pathlib.Path(path).is_file():
        return []
    with open(path) as handle:
        return json.load(handle).get("rows", [])


def machine_label(host):
    return MACHINE_LABELS.get(host, host)


def speed_from_rtf(rtf):
    return None if not rtf else round(1 / rtf, SPEED_DIGITS)


def gib(mib):
    return None if mib is None else round(mib / MIB_PER_GIB, 2)


def quality_fields(row):
    quality = row.get("quality") or {}
    kind = {"wer": "WER", "tts_wer": "TTS WER"}.get(quality.get("kind"), "")
    value = quality.get("value")
    return {"quality": kind, "quality_pct": None if value is None else round(value * PERCENT, 2)}


def family_label(family, row):
    return FAMILY_LABELS.get(family) or f"{row.get('model_label', family)} {row.get('weights', '')}".strip()


def family_order(family):
    return FAMILY_ORDER.index(family) if family in FAMILY_ORDER else len(FAMILY_ORDER)


def weekly_row(summary, row):
    run, pins = summary.get("run", {}), summary.get("pins", {})
    out = {"host": summary["_host"], "machine": machine_label(summary["_host"]), "run": run.get("name", ""),
           "label": family_label(row["family"], row), "order": family_order(row["family"]),
           "started": run.get("started", ""), "day": (run.get("started") or "")[:10], "backend": row["backend"],
           "lane": f"{machine_label(summary['_host'])} {row['backend']}", "family": row["family"], "task": row.get("task", ""),
           "model": row.get("model_label", row["family"]), "weights": row.get("weights", ""), "status": row.get("status", ""),
           "wall_ms": row.get("wall_ms_median"), "wall_min_ms": row.get("wall_ms_min"), "wall_max_ms": row.get("wall_ms_max"),
           "speed_x": row.get("speed_x"), "rtf": None if not row.get("speed_x") else round(1 / row["speed_x"], 5),
           "audio_s": row.get("audio_seconds"), "rss_mib": row.get("peak_rss_mib"), "rss_gib": gib(row.get("peak_rss_mib")),
           "gpu_mem_rise_mib": row.get("gpu_memory_rise_mib"), "gpu_mem_rise_gib": gib(row.get("gpu_memory_rise_mib")),
           "gpu_mem_peak_mib": row.get("gpu_memory_peak_mib"), "observed_backend": row.get("observed_backend"), "runs": row.get("runs"),
           "notes": row.get("notes", ""), "engine": (pins.get("engine") or {}).get("sha", "")[:SHA_SHORT],
           "ggml": (pins.get("ggml") or {}).get("sha", "")[:SHA_SHORT]}
    out.update(quality_fields(row))
    return out


def weekly_rows(summaries, task_key):
    return [weekly_row(summary, row) for summary in summaries for row in summary.get("results", [])
            if TASK_KEYS.get(row.get("task", "")) == task_key]


def newest_run_name(summaries, host):
    return max((s for s in summaries if s["_host"] == host), key=lambda s: s.get("run", {}).get("started") or "")["run"].get("name", "")


def latest_cells(rows, summaries):
    newest = {}
    for row in rows:
        newest[(row["host"], row["backend"], row["family"])] = row
    for row in newest.values():
        row["current"] = row["run"] == newest_run_name(summaries, row["host"])
    return list(newest.values())


def rtf_pair(text):
    numbers = RTF_PATTERN.findall(text or "")
    return (float(numbers[0]) if numbers else None, float(numbers[1]) if len(numbers) > 1 else None)


def campaign_row(row):
    qvac_short, qvac_long = rtf_pair(row.get("qvac_rtf", ""))
    ratio_short, ratio_long = rtf_pair(row.get("ratio", ""))
    return {"task": row.get("task", ""), "model": row.get("model", ""), "quantization": row.get("quantization", ""),
            "label": f"{row.get('model', '')} {row.get('quantization', '')}".strip(), "engine": row.get("engine", ""), "engine_version": row.get("engine_version", ""), "qvac_version": row.get("qvac_version", ""),
            "host": row.get("host", ""), "device": row.get("device", ""), "backend": row.get("backend", ""),
            "lane": f"{row.get('device', '')} {row.get('backend', '')}", "qvac_rtf_short": qvac_short, "qvac_rtf_long": qvac_long,
            "ratio_short": ratio_short, "ratio_long": ratio_long,
            "competitor_rtf_short": rounded_rtf(competitor_rtf(qvac_short, ratio_short)),
            "competitor_rtf_long": rounded_rtf(competitor_rtf(qvac_long, ratio_long)),
            "qvac_speed_short": speed_from_rtf(qvac_short), "qvac_speed_long": speed_from_rtf(qvac_long),
            "competitor_speed_short": speed_from_rtf(competitor_rtf(qvac_short, ratio_short)),
            "competitor_speed_long": speed_from_rtf(competitor_rtf(qvac_long, ratio_long)),
            "measured": row.get("measured", ""), "source": row.get("source", ""), "harness": row.get("harness", "")}


def competitor_rtf(qvac_rtf, ratio):
    return None if qvac_rtf is None or ratio is None else qvac_rtf * ratio


def rounded_rtf(rtf):
    return None if rtf is None else round(rtf, RTF_DIGITS)


def campaign_rows(comparisons, task_key):
    return [campaign_row(row) for row in comparisons if TASK_KEYS.get(row.get("task", "")) == task_key]


def run_row(summary):
    run, counts, pins = summary.get("run", {}), summary.get("counts", {}), summary.get("pins", {})
    return {"host": summary["_host"], "machine": machine_label(summary["_host"]), "run": run.get("name", ""),
            "started": run.get("started", ""), "finished": run.get("finished", ""), "status": run.get("status", ""),
            "ok": counts.get("ok", 0), "expected": counts.get("expected", 0), "backends": ",".join(summary.get("backends", [])),
            "engine": (pins.get("engine") or {}).get("sha", ""), "engine_date": (pins.get("engine") or {}).get("commit_date", ""),
            "ggml": (pins.get("ggml") or {}).get("sha", ""), "ggml_date": (pins.get("ggml") or {}).get("commit_date", ""),
            "problems": "; ".join(summary.get("problems") or [])}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")


def build(summaries, comparisons, out_dir):
    written = {}
    for task_key in sorted(set(TASK_KEYS.values())):
        folder = out_dir / task_key
        weekly = weekly_rows(summaries, task_key)
        files = {"weekly": weekly, "latest": latest_cells(weekly, summaries),
                 "campaign": campaign_rows(comparisons, task_key)}
        for name, rows in files.items():
            write_rows(folder / f"{name}.json", rows)
        written[task_key] = {name: len(rows) for name, rows in files.items()}
    write_rows(out_dir / "runs.json", [run_row(summary) for summary in summaries])
    return written


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--comparisons", default=str(pathlib.Path(__file__).resolve().parent / "comparisons.json"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summaries = load_summaries(args.summaries)
    if not summaries:
        raise SystemExit(f"no summary.json files under {args.summaries}")
    written = build(summaries, load_comparisons(args.comparisons), pathlib.Path(args.out))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} wrote {json.dumps(written)} and runs.json ({len(summaries)} runs) under {args.out}")


if __name__ == "__main__":
    main()
