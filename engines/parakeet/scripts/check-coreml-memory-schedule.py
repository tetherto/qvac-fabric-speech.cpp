#!/usr/bin/env python3
"""Checks overlapped memory schedules against the templates' memory envelope.

Reads the JSON reports of nemotron-diarization-coreml-memory (one per run,
generated schedule) and the templates they exercised, prints one table row per
run plus the host's maxima, and fails when a measurement exceeds what the
envelope charges:

  - the run's peak footprint increment over the ggml-only run of the same
    passes, against the Core ML bytes fit_params charged for the same options;
  - the largest prepare (from the start of a capacity's generate phase to the
    end of its compile), against the template's prepare transient
    generation_peak_bytes + 8 MiB + compile_peak_bytes;
  - the largest load and warm increment, against load_peak_bytes for the
    resolved compute units;
  - ANECompilerService's peak resident size during the loads, against
    ane_compile_reserve_bytes (the runtime's cold-load memory admission).

A null constant means the runtime's default: compile 0.5 x weight.bin; load
0.25 x weight.bin for cpu_and_ane and 1.0 x for the other unit classes; the
compiler reserve 1.6 x weight.bin.

  check-coreml-memory-schedule.py --template-dir DIR [--template-dir DIR ...] RUN.json [RUN.json ...]
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

MIB = 1 << 20
PREPARE_FIXED = 8 * MIB


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def charged(template, units):
    t, m = template, template["memory"]
    weights = int(t.get("weight_bin_size", 0))
    compile_peak = m.get("compile_peak_bytes")
    compile_peak = weights // 2 if compile_peak is None else int(compile_peak)
    load = (m.get("load_peak_bytes") or {}).get(units)
    if load is None:
        load = weights // 4 if units == "cpu_and_ane" else weights
    reserve = m.get("ane_compile_reserve_bytes")
    reserve = int(1.6 * weights) if reserve is None else int(reserve)
    prepare = int(t.get("generation_peak_bytes", 0)) + PREPARE_FIXED + compile_peak
    return prepare, int(load), reserve


def mib(v):
    return f"{v / MIB:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template-dir", action="append", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()

    templates = {}
    for d in a.template_dir:
        t = json.loads((Path(d) / "template.json").read_text())
        templates[t["stage"]] = t
    host = f"{sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}, macOS {sh(['sw_vers', '-productVersion'])}"
    if platform.system() != "Darwin":
        host = platform.platform()

    failures = []
    worst = {"increment": 0, "envelope": 0, "prepare": 0, "load": 0, "ane": 0}
    limit = {"prepare": None, "load": None, "reserve": None}
    print(f"host: {host}")
    print("| Run | Units | Peak increment (MiB) | Fit charge (MiB) | Largest prepare (MiB) | Charged prepare (MiB) "
          "| Largest load+warm (MiB) | Charged load (MiB) | ANECompilerService peak (MiB) | Reserve (MiB) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path in a.runs:
        r = json.loads(Path(path).read_text())
        if r.get("mode") != "coreml" or not r.get("baseline_bytes"):
            failures.append(f"{path}: not a Core ML run with a ggml-only baseline")
            continue
        units = {s["stage"]: s.get("compute_units") for s in r.get("status", {}).get("stages", [])}
        segs = {}
        for s in r["segments"]:
            segs.setdefault(s["stage"], {})[s["phase"]] = s
        prepare = load = ane = 0
        prepare_lim = load_lim = reserve_lim = None
        for stage, ph in segs.items():
            if stage not in templates:
                failures.append(f"{path}: no template for {stage}")
                continue
            p_lim, l_lim, r_lim = charged(templates[stage], units.get(stage) or "cpu_and_ane")
            if "generate" in ph:
                end = ph.get("compile", ph["generate"])
                inc = max(ph["generate"]["peak"], end["peak"]) - ph["generate"]["before"]
                prepare = max(prepare, inc)
                if inc > p_lim:
                    failures.append(f"{path}: {stage} prepare +{mib(inc)} MiB over the charged {mib(p_lim)} MiB")
            if "load" in ph:
                inc = ph["load"]["increment"]
                load = max(load, inc)
                if inc > l_lim:
                    failures.append(f"{path}: {stage} load +{mib(inc)} MiB over the charged {mib(l_lim)} MiB")
                peak = ph["load"]["ane_compiler_peak"]
                ane = max(ane, peak)
                if peak > r_lim:
                    failures.append(f"{path}: {stage} ANECompilerService {mib(peak)} MiB over the reserve "
                                    f"{mib(r_lim)} MiB")
            prepare_lim = p_lim if prepare_lim is None else min(prepare_lim, p_lim)
            load_lim = l_lim if load_lim is None else min(load_lim, l_lim)
            reserve_lim = r_lim if reserve_lim is None else min(reserve_lim, r_lim)
        increment = r["peak_increment_over_baseline"]
        envelope = r["fit_coreml_bytes"]
        if increment > envelope:
            failures.append(f"{path}: peak increment +{mib(increment)} MiB over the fit charge {mib(envelope)} MiB")
        if r.get("ggml_passes_after_ready", 0) and r.get("coreml_passes", 0) == 0:
            failures.append(f"{path}: no pass ran on Core ML")
        unit_names = ",".join(sorted({u for u in units.values() if u}))
        print(f"| {Path(path).name} | {unit_names} | {mib(increment)} | {mib(envelope)} | {mib(prepare)} | "
              f"{mib(prepare_lim or 0)} | {mib(load)} | {mib(load_lim or 0)} | {mib(ane)} | {mib(reserve_lim or 0)} |")
        worst = {"increment": max(worst["increment"], increment), "envelope": envelope,
                 "prepare": max(worst["prepare"], prepare), "load": max(worst["load"], load),
                 "ane": max(worst["ane"], ane)}
        limit = {"prepare": prepare_lim, "load": load_lim, "reserve": reserve_lim}
    print(f"host maxima: peak increment {mib(worst['increment'])} MiB (charge {mib(worst['envelope'])}), "
          f"prepare {mib(worst['prepare'])} MiB (charged {mib(limit['prepare'] or 0)}), "
          f"load+warm {mib(worst['load'])} MiB (charged {mib(limit['load'] or 0)}), "
          f"ANECompilerService {mib(worst['ane'])} MiB (reserve {mib(limit['reserve'] or 0)})")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
