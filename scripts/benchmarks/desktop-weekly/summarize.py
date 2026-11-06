#!/usr/bin/env python3
"""Collect one runs/<date>/ directory into summary.json.

usage: summarize.py --run-dir DIR --families families.json --backends cpu,cuda --families-selected a,b [--started ISO]

Reads machine.json, pins.json, results/<backend>/<family>.json and the nvidia-smi samples next to them,
and writes <run dir>/summary.json with one row per (backend, family), run-level counts and the problem list.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib

SCHEMA_VERSION = 1
MS_PER_SECOND = 1000.0
STATUS_OK = "ok"
STATUS_NOT_RUN = "not-run"
STATUS_BUILD_FAILED = "build-failed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
GPU_CSV_MEMORY_COLUMN = 1
GPU_CSV_UTILIZATION_COLUMN = 2


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_optional_json(path):
    return load_json(path) if path.is_file() else {}


def load_families(path):
    return {name: spec for name, spec in load_json(path).items() if not name.startswith("_")}


def split_list(text):
    return [item for item in text.split(",") if item]


def result_path(run_dir, backend, family):
    return run_dir / "results" / backend / f"{family}.json"


def read_result(run_dir, backend, family):
    path = result_path(run_dir, backend, family)
    return load_json(path) if path.is_file() else None


def parse_sample(line):
    fields = [field.strip() for field in line.split(",")]
    try:
        return float(fields[GPU_CSV_MEMORY_COLUMN]), float(fields[GPU_CSV_UTILIZATION_COLUMN])
    except (IndexError, ValueError):
        return None


def parse_samples(text):
    return [sample for sample in map(parse_sample, text.splitlines()) if sample is not None]


def gpu_memory(run_dir, backend, family):
    path = run_dir / "results" / backend / f"{family}.gpu.csv"
    samples = parse_samples(path.read_text()) if path.is_file() else []
    if not samples:
        return {"gpu_memory_peak_mib": None, "gpu_memory_rise_mib": None, "gpu_util_max": None}
    used = [memory for memory, _ in samples]
    return {"gpu_memory_peak_mib": max(used), "gpu_memory_rise_mib": max(used) - min(used),
            "gpu_util_max": max(util for _, util in samples)}


def tts_metadata(result):
    return (result.get("tts_intelligibility") or {}).get("audio_metadata") or {}


def audio_seconds(spec, result):
    if spec.get("audio_duration_seconds") is not None:
        return spec["audio_duration_seconds"]
    return tts_metadata(result).get("duration_seconds")


def speed_x(seconds, wall_ms):
    if not seconds or not wall_ms:
        return None
    return seconds * MS_PER_SECOND / wall_ms


def quality(result):
    if result.get("wer_median") is not None:
        return {"kind": "wer", "value": result["wer_median"]}
    tts = result.get("tts_intelligibility") or {}
    if tts.get("wer") is not None:
        return {"kind": "tts_wer", "value": tts["wer"]}
    return None


def base_row(backend, family, spec):
    return {"backend": backend, "family": family, "task": spec.get("task", ""), "model_label": spec.get("model_label", family),
            "weights": spec.get("weights", "")}


def unrun_row(backend, family, spec, status):
    row = base_row(backend, family, spec)
    row.update({"status": status, "observed_backend": None, "runner": None, "wall_ms_median": None, "wall_ms_min": None,
                "wall_ms_max": None, "rtf_median": None, "audio_seconds": None, "speed_x": None, "peak_rss_mib": None,
                "gpu_memory_peak_mib": None, "gpu_memory_rise_mib": None, "gpu_util_max": None, "quality": None, "runs": 0,
                "notes": "", "encoder_backend": None, "baseline_inference_ms_median": None})
    return row


def measured_row(backend, family, spec, result, gpu):
    row = base_row(backend, family, spec)
    seconds = audio_seconds(spec, result)
    row.update({"status": result.get("status", "unknown"), "observed_backend": result.get("backend"),
                "runner": result.get("runner"), "wall_ms_median": result.get("wall_ms_median"),
                "wall_ms_min": result.get("wall_ms_min"), "wall_ms_max": result.get("wall_ms_max"),
                "rtf_median": result.get("rtf_median"), "audio_seconds": seconds,
                "speed_x": speed_x(seconds, result.get("wall_ms_median")), "peak_rss_mib": result.get("peak_rss_mib"),
                "quality": quality(result), "runs": result.get("runs", 0), "notes": result.get("notes", ""),
                "encoder_backend": result.get("encoder_backend"), "baseline_inference_ms_median": result.get("baseline_inference_ms_median")})
    row.update(gpu)
    return row


def row_for(run_dir, backend, family, spec, build_failed):
    if build_failed:
        return unrun_row(backend, family, spec, STATUS_BUILD_FAILED)
    result = read_result(run_dir, backend, family)
    if result is None:
        return unrun_row(backend, family, spec, STATUS_NOT_RUN)
    return measured_row(backend, family, spec, result, gpu_memory(run_dir, backend, family))


def family_runs_on_backend(spec, backend):
    allowed = spec.get("backends")
    return not allowed or backend in allowed


def collect_rows(run_dir, backends, families, specs, build_failures):
    return [row_for(run_dir, backend, family, specs[family], backend in build_failures)
            for backend in backends for family in families if family_runs_on_backend(specs[family], backend)]


def count_rows(rows):
    ok = sum(1 for row in rows if row["status"] == STATUS_OK)
    return {"expected": len(rows), "ok": ok, "failed": len(rows) - ok}


def overall_status(counts):
    if counts["ok"] == counts["expected"]:
        return STATUS_OK
    return STATUS_PARTIAL if counts["ok"] else STATUS_FAILED


def row_problem(row):
    label = f"{row['family']} on {row['backend']}"
    note = f": {row['notes']}" if row.get("notes") else ""
    return f"{label} {row['status']}{note}"


def problems(rows, build_failures):
    found = [f"build failed on {backend}" for backend in build_failures]
    found += [row_problem(row) for row in rows if row["status"] != STATUS_OK]
    return found


def read_build_failures(run_dir):
    path = run_dir / "build-failures.txt"
    return split_list(path.read_text().replace("\n", ",")) if path.is_file() else []


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_summary(run_dir, specs, backends, families, started):
    machine = load_optional_json(run_dir / "machine.json")
    build_failures = read_build_failures(run_dir)
    rows = collect_rows(run_dir, backends, families, specs, build_failures)
    counts = count_rows(rows)
    return {"schema": SCHEMA_VERSION,
            "run": {"name": run_dir.name, "host": machine.get("hostname", ""), "started": started, "finished": utc_now(),
                    "status": overall_status(counts)},
            "machine": machine, "pins": load_optional_json(run_dir / "pins.json"), "backends": backends,
            "families": families, "results": rows, "counts": counts, "build_failures": build_failures,
            "problems": problems(rows, build_failures)}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--backends", required=True)
    parser.add_argument("--families-selected", required=True)
    parser.add_argument("--started", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = pathlib.Path(args.run_dir)
    summary = build_summary(run_dir, load_families(args.families), split_list(args.backends),
                            split_list(args.families_selected), args.started)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['run']['status']}: {summary['counts']['ok']}/{summary['counts']['expected']} ok")


if __name__ == "__main__":
    main()
