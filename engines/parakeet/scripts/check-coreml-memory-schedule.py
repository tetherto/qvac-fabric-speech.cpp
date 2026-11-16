#!/usr/bin/env python3
"""Checks overlapped memory schedules against the templates' memory envelope.

Reads the JSON reports of nemotron-diarization-coreml-memory (one per run,
generated schedule) and the templates they exercised, prints one table row per
run plus the maxima over the runs, and fails when a measurement exceeds what
the envelope charges:

  - the run's peak footprint increment over the ggml-only run of the same
    passes, against the Core ML bytes fit_params charged for the same options;
  - each prepare (from the start of a capacity's generate phase to the end of
    its compile), against the template's prepare transient
    generation_peak_bytes + 8 MiB + compile_peak_bytes;
  - each load and warm increment, against load_peak_bytes for the resolved
    compute units;
  - each capacity's resident step (the settled footprint once it is ready,
    less the level before it), against the template's resident part
    runtime_weight_bytes + 2 x I/O + state_bytes + workspace_bytes, and the
    run's steady growth over ggml against the sum of those parts;
  - ANECompilerService's growth during the loads (its peak less its size when
    the first load started), against ane_compile_reserve_bytes (the runtime's
    cold-load memory admission).

Every template stage must show a generate and a load segment in every run, or
the stage checks would pass without measuring anything; a stage seen more than
once in a phase is checked at its largest value.

A null constant means the runtime's default: runtime weights = weight.bin;
compile 0.5 x weight.bin; load 0.25 x weight.bin for cpu_and_ane and 1.0 x for
the other unit classes; the compiler reserve 1.6 x weight.bin.

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


def dtype_bytes(dtype):
    return {"fp16": 2, "fp64": 8}.get(dtype, 4)


def io_bytes(template):
    total = 0
    for fn in template.get("functions", []):
        for side in ("inputs", "outputs"):
            for x in fn.get(side, []):
                n = dtype_bytes(x.get("dtype", "fp32"))
                for d in x.get("shape", []):
                    n *= int(d)
                total += n
    return total


def charged(template, units):
    """(prepare, load, reserve, resident) the envelope charges one stage."""
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
    resident = (int(m.get("runtime_weight_bytes") or weights) + 2 * io_bytes(t) + int(t.get("state_bytes", 0))
                + int(m.get("workspace_bytes") or 0))
    return prepare, int(load), reserve, resident


def mib(v):
    return f"{v / MIB:.1f}"


def prepares(segs):
    """The increment of each prepare: a generate segment and the compile
    segments after it, up to the next generate or load."""
    out, current = [], None
    for s in sorted(segs, key=lambda s: s["t0"]):
        if s["phase"] == "generate":
            if current:
                out.append(current)
            current = {"before": s["before"], "peak": s["peak"]}
        elif s["phase"] == "compile" and current:
            current["peak"] = max(current["peak"], s["peak"])
        elif s["phase"] == "load" and current:
            out.append(current)
            current = None
    if current:
        out.append(current)
    return [max(0, p["peak"] - p["before"]) for p in out]


def check_run(path, r, templates):
    """Returns (row, failures) for one report."""
    failures = []
    if r.get("mode") != "coreml" or not r.get("baseline_bytes"):
        return None, [f"{path}: not a Core ML run with a ggml-only baseline"]
    units = {s["stage"]: s.get("compute_units") for s in r.get("status", {}).get("stages", [])}
    by_stage = {}
    for s in r.get("segments", []):
        by_stage.setdefault(s["stage"], []).append(s)
    for stage in sorted(set(by_stage) - set(templates)):
        failures.append(f"{path}: no template for {stage}")

    row = {"prepare": 0, "load": 0, "resident": 0, "prepare_lim": None, "load_lim": None, "resident_lim": None,
           "reserve_lim": None, "resident_sum": 0}
    for stage, template in sorted(templates.items()):
        segs = by_stage.get(stage, [])
        phases = {s["phase"] for s in segs}
        for phase in ("generate", "load"):
            if phase not in phases:
                failures.append(f"{path}: {stage} has no {phase} segment")
        p_lim, l_lim, r_lim, res_lim = charged(template, units.get(stage) or "cpu_and_ane")
        row["resident_sum"] += res_lim
        for key, lim in (("prepare_lim", p_lim), ("load_lim", l_lim), ("reserve_lim", r_lim),
                         ("resident_lim", res_lim)):
            row[key] = lim if row[key] is None else min(row[key], lim)
        inc = max(prepares(segs), default=0)
        row["prepare"] = max(row["prepare"], inc)
        if inc > p_lim:
            failures.append(f"{path}: {stage} prepare +{mib(inc)} MiB over the charged {mib(p_lim)} MiB")
        loads = [s for s in segs if s["phase"] == "load"]
        inc = max((s["increment"] for s in loads), default=0)
        row["load"] = max(row["load"], inc)
        if inc > l_lim:
            failures.append(f"{path}: {stage} load +{mib(inc)} MiB over the charged {mib(l_lim)} MiB")
        if any("resident_increment" not in s for s in loads):
            failures.append(f"{path}: {stage} load segment without resident_increment (an older tool's report)")
        inc = max((s.get("resident_increment", 0) for s in loads), default=0)
        row["resident"] = max(row["resident"], inc)
        if inc > res_lim:
            failures.append(f"{path}: {stage} resident step +{mib(inc)} MiB over the charged {mib(res_lim)} MiB")

    loads = [s for s in r.get("segments", []) if s["phase"] == "load"]
    peak = max((s["ane_compiler_peak"] for s in loads), default=0)
    if "ane_compiler_before_loads" not in r:
        failures.append(f"{path}: no ane_compiler_before_loads (an older tool's report)")
    growth = max(0, peak - int(r.get("ane_compiler_before_loads", 0)))
    row["ane_growth"], row["ane_peak"] = growth, peak
    if row["reserve_lim"] is not None and growth > row["reserve_lim"]:
        failures.append(f"{path}: ANECompilerService grew {mib(growth)} MiB during the loads, over the reserve "
                        f"{mib(row['reserve_lim'])} MiB")
    steady = int(r.get("steady_increment_over_baseline", 0))
    row["steady"] = steady
    if steady > row["resident_sum"]:
        failures.append(f"{path}: steady growth +{mib(steady)} MiB over the charged resident parts "
                        f"{mib(row['resident_sum'])} MiB")
    row["increment"], row["envelope"] = r["peak_increment_over_baseline"], r["fit_coreml_bytes"]
    if row["increment"] > row["envelope"]:
        failures.append(f"{path}: peak increment +{mib(row['increment'])} MiB over the fit charge "
                        f"{mib(row['envelope'])} MiB")
    if r.get("ggml_passes_after_ready", 0) and r.get("coreml_passes", 0) == 0:
        failures.append(f"{path}: no pass ran on Core ML")
    row["units"] = ",".join(sorted({u for u in units.values() if u}))
    return row, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template-dir", action="append", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)

    templates = {}
    for d in a.template_dir:
        t = json.loads((Path(d) / "template.json").read_text())
        templates[t["stage"]] = t
    host = f"{sh(['sysctl', '-n', 'machdep.cpu.brand_string'])}, macOS {sh(['sw_vers', '-productVersion'])}"
    if platform.system() != "Darwin":
        host = platform.platform()

    failures = []
    rows = []
    print(f"host: {host}")
    print("| Run | Units | Peak increment (MiB) | Fit charge (MiB) | Largest prepare (MiB) | Charged prepare (MiB) "
          "| Largest load+warm (MiB) | Charged load (MiB) | Largest resident step (MiB) | Charged resident (MiB) "
          "| Steady growth (MiB) | ANECompilerService growth (MiB) | Reserve (MiB) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for path in a.runs:
        row, f = check_run(path, json.loads(Path(path).read_text()), templates)
        failures += f
        if row is None:
            continue
        rows.append(row)
        print(f"| {Path(path).name} | {row['units']} | {mib(row['increment'])} | {mib(row['envelope'])} | "
              f"{mib(row['prepare'])} | {mib(row['prepare_lim'] or 0)} | {mib(row['load'])} | "
              f"{mib(row['load_lim'] or 0)} | {mib(row['resident'])} | {mib(row['resident_lim'] or 0)} | "
              f"{mib(row['steady'])} | {mib(row['ane_growth'])} (peak {mib(row['ane_peak'])}) | "
              f"{mib(row['reserve_lim'] or 0)} |")
    if rows:
        worst = {k: max(r[k] for r in rows) for k in ("increment", "prepare", "load", "resident", "steady",
                                                       "ane_growth")}
        print(f"maxima: peak increment {mib(worst['increment'])} MiB, prepare {mib(worst['prepare'])} MiB, "
              f"load+warm {mib(worst['load'])} MiB, resident step {mib(worst['resident'])} MiB, "
              f"steady growth {mib(worst['steady'])} MiB, ANECompilerService growth {mib(worst['ane_growth'])} MiB")
    for f in failures:
        print(f"FAIL {f}")
    print("PASS" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
