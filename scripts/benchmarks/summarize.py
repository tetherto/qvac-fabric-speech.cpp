#!/usr/bin/env python3
# Aggregate the per-(family, runner) result.json files produced by
# scripts/benchmarks/run-family.sh into a single markdown table.
#
# Called from .github/workflows/benchmark-desktop.yml's summarize job:
#
#   summarize.py --results-dir all-results \
#                --out speech-benchmark-findings.md
#
# Also writes to $GITHUB_STEP_SUMMARY when the env var is set so the table
# is visible on the workflow run page without downloading the artifact.
#
# Column shape mirrors engines/audiogen/benchmarks/comparison/reports/*.md
# (the closest in-repo precedent for this project's benchmark tables):
# Family | Model | Runner | OS | Backend | Wall ms (median/min/max) | RTF
# (median) | Peak RSS MiB | Runs | Status | Notes.

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any


def load_results(root: pathlib.Path) -> list[dict[str, Any]]:
    """Walk `root` for every result.json; return a list sorted for a stable table."""
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            with path.open() as f:
                results.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            # A missing / malformed file is a bench-job crash — surface it as
            # a row rather than silently dropping data.
            results.append({
                "family": path.parent.name,
                "model": "?",
                "runner": "?",
                "os": "?",
                "backend": "?",
                "wall_ms_median": None,
                "rtf_median": None,
                "peak_rss_mib": None,
                "runs": 0,
                "status": "artifact-corrupt",
                "notes": f"could not parse {path}: {e}",
            })
    return sorted(
        results,
        key=lambda r: (r.get("family", ""), r.get("runner", ""), r.get("model", "")),
    )


def fmt_ms(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def fmt_rtf(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def fmt_rss(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.0f}"
    except (TypeError, ValueError):
        return "—"


def render_markdown(results: list[dict[str, Any]]) -> str:
    """Single fixed-column table. Empty result set still produces a valid table."""
    header = (
        "| Family | Model | Runner | OS | Backend | Median wall ms | Min | Max "
        "| Median RTF | Peak RSS MiB | Runs | Status | Notes |\n"
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    rows = []
    for r in results:
        rows.append(
            "| {family} | {model} | {runner} | {os} | {backend} | {median} | {min} | {max} "
            "| {rtf} | {rss} | {runs} | {status} | {notes} |".format(
                family=r.get("family", "?"),
                model=r.get("model", "?"),
                runner=r.get("runner", "?"),
                os=r.get("os", "?"),
                backend=r.get("backend", "?") or "unknown",
                median=fmt_ms(r.get("wall_ms_median")),
                min=fmt_ms(r.get("wall_ms_min")),
                max=fmt_ms(r.get("wall_ms_max")),
                rtf=fmt_rtf(r.get("rtf_median")),
                rss=fmt_rss(r.get("peak_rss_mib")),
                runs=r.get("runs", 0),
                status=r.get("status", "?"),
                notes=(r.get("notes") or "").replace("|", "\\|"),
            )
        )
    if not rows:
        rows.append("| _no results collected_ | | | | | | | | | | | | |")
    return header + "\n".join(rows) + "\n"


def render_legend() -> str:
    """Small key describing what each metric means, so a reader landing on
    the workflow run page without context can interpret the numbers."""
    return (
        "\n**Metric key**\n"
        "- **Median wall ms** — end-to-end wall-clock median across `--runs` timed "
        "iterations (warmup runs discarded). Includes model load + graph build for "
        "time-wrapped families; native `*-bench` binaries report their own totals.\n"
        "- **Median RTF** — real-time factor; wall / audio-seconds. `< 1.0` means "
        "faster than real-time. Blank when the family's output length isn't fixed "
        "(text-driven TTS, chatterbox, audio8).\n"
        "- **Peak RSS MiB** — maximum resident set size across all timed runs, via "
        "`/usr/bin/time` (GNU `-v` on Linux, BSD `-l` on macOS).\n"
        "- **Backend** — from the bench binary's JSON when it reports one, else the "
        "engine's `using <NAME> backend` / `backend: <NAME>` log lines, else ggml's "
        "own GPU compute-init logs; a green run with no GPU-engagement evidence is "
        "reported as `CPU`. `unknown` appears only on failed runs, whose logs may be "
        "truncated mid-init.\n"
        "- **Status** — `ok` / `not-in-registry` (whisper `small`) / `missing-model` "
        "(minimax — no S3 path yet) / `build-failed` / `run-failed` / `fetch-failed` / "
        "`infrastructure-failed` (a pre-bench workflow step — AWS OIDC, ggml build, "
        "cmake configure — did not complete).\n"
        "- **Linux vs macOS ggml build**: hosted Linux is a portable ggml build "
        "(no `-march=native`, no GPU backend) — its numbers are a **floor**, "
        "not a runner-optimal peak. Self-hosted macOS gets Metal + native "
        "codegen. Comparing the two columns as apples-to-apples is misleading; "
        "compare each family's Linux row across dispatches for CPU perf, "
        "the macOS row for Metal perf.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    # `expected_count` is emitted by the workflow's `plan` job. When set, we
    # also fail if fewer rows than planned made it to the summary — that's
    # the "cell died before writing its pre-flight seed" case, which the
    # `not ok_rows` gate below can't see because the missing row contributes
    # zero to `results`.
    #
    # The workflow calls this with `--expected-count "$EXPECTED_COUNT"`; if
    # the plan job failed before setting the output, EXPECTED_COUNT expands
    # to an empty string. Accept "" as "unset" so summarize can still render
    # whatever artifacts it does find (a plan-failed dispatch has already
    # turned the workflow red — summarize failing on top would just be
    # noise).
    def _opt_int(v: str) -> int | None:
        return int(v) if v else None
    ap.add_argument("--expected-count", type=_opt_int, default=None)
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1

    results = load_results(args.results_dir)

    title = "# Speech benchmark findings\n\n"
    subtitle = (
        f"Rows collected: **{len(results)}**. "
        "Wall-clock median across `--runs` timed iterations "
        "(warmup runs discarded).\n\n"
    )
    body = title + subtitle + render_markdown(results) + render_legend()

    args.out.write_text(body)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(body)
    print(f"wrote {args.out} ({len(results)} rows)")

    # Loudness gate. `bench` runs with `continue-on-error: true` so one
    # family failure doesn't skip the summary — but if EVERY cell failed
    # (misconfigured S3 role, wrong binary paths, ggml build broken) the
    # workflow would otherwise finish green with a table full of red
    # statuses that nobody looks at. Failing the summarize job here makes
    # the whole dispatch red so the reader has to look at the summary.
    #
    # Two independent trip conditions:
    #   (a) rows are collected but none succeeded — every cell wrote a
    #       failure result.
    #   (b) fewer rows than planned — a cell died before its pre-flight
    #       seed could write result.json (workflow env / harness bug).
    #       Needs --expected-count from the workflow's plan job.
    ok_rows = [r for r in results if r.get("status") == "ok"]
    if results and not ok_rows:
        print(
            f"FAIL: {len(results)} rows collected but 0 have status=ok. "
            "Every cell failed — see the row-level notes.",
            file=sys.stderr,
        )
        return 1
    if args.expected_count is not None and len(results) < args.expected_count:
        print(
            f"FAIL: expected {args.expected_count} rows from the plan "
            f"but only {len(results)} arrived. "
            f"{args.expected_count - len(results)} cell(s) died before "
            "writing a result.json.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
