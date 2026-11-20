#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

STEM = "diar_streaming_sortformer_4spk-v2.1"
MODEL_NAME = STEM + ".q8_0.gguf"
ARMS = ("metal", "legacy", "diarizer")
FIXTURES = ("abcba", "abcdba")
CHUNKS_MS = (1000, 2000)
ROUTE = re.compile(r"diarization-route: pass=(\d+) coreml=([01]) frames=(\d+)")
COREML_ENV_PREFIX = "PARAKEET_COREML_"
TIMEOUT_SECONDS = 600
COLLAR_MS = 250


def load_scorer():
    spec = importlib.util.spec_from_file_location("compute_der", Path(__file__).with_name("compute-der.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_routes(log, arm):
    routes = [(int(index), int(coreml), int(frames)) for index, coreml, frames in ROUTE.findall(log)]
    if not routes or [row[0] for row in routes] != list(range(1, len(routes) + 1)):
        raise ValueError("missing or incomplete per-pass routing evidence")
    expected = int(arm != "metal")
    if any(coreml != expected or frames <= 0 for _, coreml, frames in routes):
        raise ValueError(f"unexpected backend or empty pass in {arm}")
    return {"network_passes": len(routes), "coreml_passes": sum(row[1] for row in routes)}


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset_manifest(root):
    return [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
             "sha256": hash_file(path)} for path in sorted(root.rglob("*")) if path.is_file()]


def stage_arm(assets, destination, arm):
    destination.mkdir(parents=True, exist_ok=False)
    model = assets / MODEL_NAME
    if not model.is_file() or model.stat().st_size == 0:
        raise ValueError(f"missing model: {model}")
    (destination / MODEL_NAME).symlink_to(model.resolve())
    names = {"metal": (), "legacy": (STEM + "-encoder", STEM + "-encoder-bypass-pre-encode"),
             "diarizer": (STEM + "-diarizer-410",)}[arm]
    for name in names:
        bundle = assets / (name + ".mlmodelc")
        if not (bundle / "model.mil").is_file() or not (bundle / "weights/weight.bin").is_file():
            raise ValueError(f"missing or incomplete bundle: {bundle}")
        (destination / bundle.name).symlink_to(bundle.resolve(), target_is_directory=True)
    return destination / MODEL_NAME


def arm_environment(arm):
    environment = {key: value for key, value in os.environ.items() if not key.startswith(COREML_ENV_PREFIX)}
    environment["PARAKEET_COREML_COMPUTE_UNITS"] = "cpu_and_ane"
    if arm == "metal":
        environment["PARAKEET_COREML_DISABLE"] = "1"
    return environment


def run_case(args, model, arm, fixture, chunk, iteration):
    name = f"{arm}-{fixture}-{chunk}ms-{iteration}"
    base = args.out / name
    command = [str(args.binary.resolve()), "--model", str(model.absolute()),
               "--wav", str(args.samples / (fixture + ".wav")), "--stream", "--stream-chunk-ms", str(chunk),
               "--emit", "jsonl", "--backend", "metal", "--n-gpu-layers", "999", "--verbose"]
    base.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
    with base.with_suffix(".jsonl").open("w") as stdout, base.with_suffix(".log").open("w") as stderr:
        subprocess.run(command, env=arm_environment(arm), stdout=stdout, stderr=stderr,
                       timeout=TIMEOUT_SECONDS, check=True)
    routing = validate_routes(base.with_suffix(".log").read_text(), arm)
    scorer = load_scorer()
    hypothesis = scorer.parse_hypothesis_jsonl(base.with_suffix(".jsonl").read_text())
    reference = scorer.parse_rttm((args.samples / (fixture + ".rttm")).read_text())
    score = scorer.compute_der(hypothesis, reference, collar_ms=COLLAR_MS)
    result = {"arm": arm, "fixture": fixture, "chunk_ms": chunk, "run": iteration, **routing, **score}
    base.with_suffix(".score.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_cases(args, staging):
    results = []
    for arm in ARMS:
        model = stage_arm(args.assets, staging / arm, arm)
        for fixture in FIXTURES:
            for chunk in CHUNKS_MS:
                for iteration in range(1, args.runs + 1):
                    results.append(run_case(args, model, arm, fixture, chunk, iteration))
    return results


def summarize(results):
    groups = {}
    for row in results:
        key = (row["arm"], row["fixture"], row["chunk_ms"])
        groups.setdefault(key, []).append(row["der"])
    return [{"arm": arm, "fixture": fixture, "chunk_ms": chunk, "der_runs": values,
             "der_median": statistics.median(values)} for (arm, fixture, chunk), values in groups.items()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"assets": asset_manifest(args.assets), "binary_sha256": hash_file(args.binary),
                "samples": asset_manifest(args.samples), "collar_ms": COLLAR_MS}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="sortformer-preflight-") as staging:
        results = run_cases(args, Path(staging))
    (args.out / "summary.json").write_text(json.dumps(summarize(results), indent=2) + "\n")


if __name__ == "__main__":
    main()
