#!/usr/bin/env bash
# Test suite for run-family.sh + families.json.
#
# Two layers:
#   1. Spec pins — assertions against the real families.json that encode the
#      regressions this harness has already hit (sortformer benching through a
#      CLI mode that ignores --bench, cosyvoice missing its GPU flag, HF
#      families without a pinned ref/sha256).
#   2. Driver behavior — a synthetic spec plus stub bench binaries and stub
#      aws/curl on PATH exercise fetch retries, both bench kinds, backend
#      attribution, and the failure-reporting paths without models, engines,
#      or network.
#
# Runs standalone (`scripts/benchmarks/test-run-family.sh`) and as the
# workflow's plan-job self-check. Needs bash, jq, and a `date` with %N —
# the same prerequisites run-family.sh itself enforces.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/run-family.sh"
REAL_SPEC="$HERE/families.json"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { PASS=$((PASS + 1)); echo "ok $PASS - $*"; }

# ---- 1. static checks on the shipped files ----------------------------------

bash -n "$DRIVER" || fail "run-family.sh does not parse"
ok "run-family.sh parses"

jq -e . "$REAL_SPEC" > /dev/null || fail "families.json is not valid JSON"
ok "families.json is valid JSON"

bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select((.value.bench_kind // "") == "" or (.value.binary // "") == "" or (.value.cmake_target // "") == "")
  | .key' "$REAL_SPEC")"
[[ -z "$bad" ]] || fail "families missing bench_kind/binary/cmake_target: $bad"
ok "every family declares bench_kind, binary, cmake_target"

jq -e '.sortformer.bench_kind == "time-wrapped"' "$REAL_SPEC" > /dev/null \
  || fail "sortformer must be time-wrapped: parakeet-cli's diarize path returns before the --bench loop and emits no --bench-json"
jq -e '[.sortformer.args[] | select(startswith("--bench"))] | length == 0' "$REAL_SPEC" > /dev/null \
  || fail "sortformer args carry --bench flags the diarize path ignores"
jq -e '.sortformer.models | length == 1' "$REAL_SPEC" > /dev/null \
  || fail "pure-diarization sortformer needs exactly its own GGUF"
jq -e '.sortformer.audio_duration_seconds == 11.0' "$REAL_SPEC" > /dev/null \
  || fail "sortformer RTF divides by the jfk.wav duration"
ok "sortformer benches the diarize path time-wrapped"

jq -e '.cosyvoice.args | index("--n-gpu-layers") != null' "$REAL_SPEC" > /dev/null \
  || fail "cosyvoice args lost --n-gpu-layers (Metal offload on the macOS runner)"
ok "cosyvoice requests GPU offload"

jq -e '.parakeet.correctness.kind == "wer"' "$REAL_SPEC" > /dev/null \
  || fail "parakeet must declare a correctness.kind='wer' block — the pilot for the WER-scoring path"
jq -e '.parakeet.correctness.reference | test("parakeet-expected-jfk-output\\.txt$")' "$REAL_SPEC" > /dev/null \
  || fail "parakeet correctness.reference must point at the checked-in JFK expected transcript"
parakeet_ref_repo_rel="$(jq -r '.parakeet.correctness.reference' "$REAL_SPEC")"
[[ -f "$HERE/../../$parakeet_ref_repo_rel" ]] \
  || fail "parakeet correctness.reference file missing: $parakeet_ref_repo_rel"
ok "parakeet declares a WER correctness block pointing at a real reference file"

# compute-wer.py: shipped self-test must pass. Catches accidental regressions
# in the normalizer / DP without needing a real bench run.
python3 "$HERE/compute-wer.py" --self-test > /dev/null \
  || fail "compute-wer.py --self-test failed"
ok "compute-wer.py self-test passes"

jq -e '.vad.source == "huggingface" and (.vad.hf_repo | length > 0) and (.vad.hf_ref | test("^[0-9a-f]{40}$"))' "$REAL_SPEC" > /dev/null \
  || fail "vad family must pin an HF repo at a full commit sha"
jq -e '.vad.models[0].sha256 | test("^[0-9a-f]{64}$")' "$REAL_SPEC" > /dev/null \
  || fail "vad model must pin a sha256"
jq -e '.vad.audio_duration_seconds == 11.0' "$REAL_SPEC" > /dev/null \
  || fail "vad RTF divides by the jfk.wav duration"
ok "vad family pins its HF source and checksum"

hf_bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
  | select(.value.source == "huggingface")
  | select((.value.hf_repo // "") == "" or (.value.hf_ref // "") == "")
  | .key' "$REAL_SPEC")"
[[ -z "$hf_bad" ]] || fail "huggingface families without hf_repo/hf_ref: $hf_bad"
ok "every huggingface family pins repo and ref"

# ---- 2. driver behavior against a synthetic spec -----------------------------

SPEC="$TMP/spec.json"
BUILD="$TMP/build"
MODELS="$TMP/models"
AUDIO="$TMP/audio"
OUT="$TMP/out"
STUBBIN="$TMP/stubbin"
STATE="$TMP/state"
mkdir -p "$BUILD/bin" "$MODELS" "$AUDIO" "$OUT" "$STUBBIN" "$STATE"

if command -v sha256sum > /dev/null; then
  hello_sha="$(printf 'hello' | sha256sum | awk '{print $1}')"
else
  hello_sha="$(printf 'hello' | shasum -a 256 | awk '{print $1}')"
fi

REF_DIR="$TMP/refs"
mkdir -p "$REF_DIR"
printf 'the quick brown fox' > "$REF_DIR/ref-perfect.txt"
printf 'the quick brown fox' > "$REF_DIR/ref-2subs.txt"    # same ref; the stub hyp will differ

jq -n --arg sha "$hello_sha" --arg ref_perfect "$REF_DIR/ref-perfect.txt" --arg ref_2subs "$REF_DIR/ref-2subs.txt" '{
  "nat-ok": {
    bench_kind: "native", binary: "bin/nat-ok", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: null, notes: "n"
  },
  "nat-nojson": {
    bench_kind: "native", binary: "bin/nat-nojson", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: null, notes: "n"
  },
  "wer-perfect": {
    bench_kind: "native", binary: "bin/nat-transcript-perfect", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "expects wer_median=0.0"
  },
  "wer-nonzero": {
    bench_kind: "native", binary: "bin/nat-transcript-wrong", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_2subs, normalizer: "english"},
    notes: "expects wer_median>0"
  },
  "wer-notranscript": {
    bench_kind: "native", binary: "bin/nat-ok", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "nat-ok emits no .transcript field; correctness must be gracefully skipped"
  },
  "wer-badref": {
    bench_kind: "native", binary: "bin/nat-transcript-perfect", cmake_target: "x",
    args: ["${JSON_OUT}"], audio_duration_seconds: 11.0,
    correctness: {kind: "wer", reference: "/does/not/exist.txt", normalizer: "english"},
    notes: "missing reference file — correctness skipped, perf still ok"
  },
  "wer-tw-warns": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0,
    correctness: {kind: "wer", reference: $ref_perfect, normalizer: "english"},
    notes: "time-wrapped families cannot score correctness; driver should warn"
  },
  "tw-marker": {
    bench_kind: "time-wrapped", binary: "bin/tw-marker", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0, notes: "n"
  },
  "tw-evidence": {
    bench_kind: "time-wrapped", binary: "bin/tw-evidence", cmake_target: "x",
    args: [], audio_duration_seconds: 1.0, notes: "n"
  },
  "tw-silent": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "s3-retry": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    s3_prefix: "stub", models: ["d/model.bin"],
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "s3-fail": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    s3_prefix: "stub", models: ["d/never.bin"],
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "hf-file": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    source: "huggingface", hf_repo: "stub/repo", hf_ref: "0000000000000000000000000000000000000000",
    models: [{name: "m.bin", sha256: $sha}],
    args: [], audio_duration_seconds: null, notes: "n"
  },
  "whisper": {
    bench_kind: "time-wrapped", binary: "bin/tw-silent", cmake_target: "x",
    source: "huggingface", hf_repo: "stub/repo", hf_ref: "0000000000000000000000000000000000000000",
    models_by_size: {tiny: {name: "ggml-tiny.bin"}},
    args: [], audio_duration_seconds: null, notes: "n"
  }
}' > "$SPEC"

cat > "$BUILD/bin/nat-ok" <<'STUB'
#!/usr/bin/env bash
printf '{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU"}' > "$1"
STUB

cat > "$BUILD/bin/nat-nojson" <<'STUB'
#!/usr/bin/env bash
echo "[0.00-2.00] speaker_0"
exit 0
STUB

cat > "$BUILD/bin/tw-marker" <<'STUB'
#!/usr/bin/env bash
echo "  backend: Metal (kv_attn_type=f16)"
STUB

cat > "$BUILD/bin/tw-evidence" <<'STUB'
#!/usr/bin/env bash
echo "ggml_metal_library_init: using embedded metal library" >&2
STUB

cat > "$BUILD/bin/tw-silent" <<'STUB'
#!/usr/bin/env bash
exit 0
STUB

# Parakeet-shape JSON emitters for correctness tests. The driver expands
# ${JSON_OUT} into args; both binaries take that as their sole arg.
cat > "$BUILD/bin/nat-transcript-perfect" <<'STUB'
#!/usr/bin/env bash
cat > "$1" <<'EOJ'
{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU","transcript":"the quick brown fox"}
EOJ
STUB

cat > "$BUILD/bin/nat-transcript-wrong" <<'STUB'
#!/usr/bin/env bash
cat > "$1" <<'EOJ'
{"stages":{"tot":{"median_ms":100,"min_ms":90,"max_ms":110}},"rtf":{"median":0.5},"backend":"StubGPU","transcript":"the SLOW brown RED fox"}
EOJ
STUB

cat > "$STUBBIN/aws" <<'STUB'
#!/usr/bin/env bash
# Invoked as: aws s3 cp <s3url> <dest> --no-progress
count_file="$STATE_DIR/aws-calls"
n="$(cat "$count_file" 2>/dev/null || echo 0)"
n=$(( n + 1 ))
echo "$n" > "$count_file"
case "$3" in
  */model.bin) if [[ "$n" -ge 3 ]]; then printf 'weights' > "$4"; exit 0; else exit 1; fi ;;
  *) exit 1 ;;
esac
STUB

cat > "$STUBBIN/curl" <<'STUB'
#!/usr/bin/env bash
# Only the -o <dest> pair matters to the test.
dest=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "-o" ]]; then dest="$2"; shift 2; else shift; fi
done
[[ -n "$dest" ]] || exit 2
printf 'hello' > "$dest"
STUB

chmod +x "$BUILD"/bin/* "$STUBBIN"/*

run_driver() {
  # run_driver <family> <out.json> <stderr-log> [extra env as VAR=VALUE...]
  local family="$1" out="$2" errlog="$3"; shift 3
  env -i PATH="$STUBBIN:$PATH" HOME="$TMP" "$@" \
    bash "$DRIVER" \
      --family "$family" --runs 2 --warmup 0 \
      --build-dir "$BUILD" --models-root "$MODELS" \
      --audio-dir "$AUDIO" --runner test \
      --out "$out" 2> "$errlog"
}

# native, JSON emitted: stats + rtf + backend come from the bench JSON.
run_driver nat-ok "$OUT/nat-ok.json" "$OUT/nat-ok.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "StubGPU" and .wall_ms_median == 100 and .rtf_median == 0.5' \
  "$OUT/nat-ok.json" > /dev/null || fail "nat-ok: $(cat "$OUT/nat-ok.json")"
ok "native bench JSON drives stats, rtf, and backend"

# native, exit 0 without JSON (the sortformer failure shape): run-failed and
# the child's output is dumped to the step log instead of vanishing.
run_driver nat-nojson "$OUT/nat-nojson.json" "$OUT/nat-nojson.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "run-failed"' "$OUT/nat-nojson.json" > /dev/null \
  || fail "nat-nojson status: $(cat "$OUT/nat-nojson.json")"
grep -q 'did not emit the --json-out file' "$OUT/nat-nojson.err" \
  || fail "nat-nojson: missing no-JSON diagnosis in driver stderr"
grep -q 'speaker_0' "$OUT/nat-nojson.err" \
  || fail "nat-nojson: child stdout not dumped to the step log"
ok "silent no-JSON native bench reports run-failed with dumped child output"

# time-wrapped, explicit `backend:` marker on stdout (supertonic/lavasr shape).
run_driver tw-marker "$OUT/tw-marker.json" "$OUT/tw-marker.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "Metal" and .rtf_median != null' \
  "$OUT/tw-marker.json" > /dev/null || fail "tw-marker: $(cat "$OUT/tw-marker.json")"
ok "time-wrapped run scrapes 'backend:' markers from stdout"

# time-wrapped, ggml compute-init evidence on stderr (acestep-on-Metal shape).
run_driver tw-evidence "$OUT/tw-evidence.json" "$OUT/tw-evidence.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "Metal"' "$OUT/tw-evidence.json" > /dev/null \
  || fail "tw-evidence: $(cat "$OUT/tw-evidence.json")"
ok "ggml GPU compute-init evidence attributes the backend"

# time-wrapped, no marker at all: a green run defaults to CPU, not unknown.
run_driver tw-silent "$OUT/tw-silent.json" "$OUT/tw-silent.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .backend == "CPU" and .rtf_median == null' \
  "$OUT/tw-silent.json" > /dev/null || fail "tw-silent: $(cat "$OUT/tw-silent.json")"
ok "green run without GPU evidence is labeled CPU"

# S3 fetch: two transient failures, third attempt lands — one bench, 3 calls.
mkdir -p "$STATE/retry"
run_driver s3-retry "$OUT/s3-retry.json" "$OUT/s3-retry.err" \
  BENCH_FAMILIES_JSON="$SPEC" MODEL_S3_BUCKET=stub-bucket \
  BENCH_RETRY_SLEEP_S=0 STATE_DIR="$STATE/retry"
jq -e '.status == "ok"' "$OUT/s3-retry.json" > /dev/null \
  || fail "s3-retry: $(cat "$OUT/s3-retry.json")"
[[ "$(cat "$STATE/retry/aws-calls")" == "3" ]] \
  || fail "s3-retry: expected 3 aws attempts, got $(cat "$STATE/retry/aws-calls")"
[[ -f "$MODELS/s3-retry/model.bin" ]] || fail "s3-retry: model not staged"
ok "transient S3 resets are retried until the copy lands"

# S3 fetch: permanent failure exhausts the retries and reports fetch-failed
# with no partial file left behind.
mkdir -p "$STATE/fail"
run_driver s3-fail "$OUT/s3-fail.json" "$OUT/s3-fail.err" \
  BENCH_FAMILIES_JSON="$SPEC" MODEL_S3_BUCKET=stub-bucket \
  BENCH_RETRY_SLEEP_S=0 STATE_DIR="$STATE/fail"
jq -e '.status == "fetch-failed"' "$OUT/s3-fail.json" > /dev/null \
  || fail "s3-fail: $(cat "$OUT/s3-fail.json")"
[[ "$(cat "$STATE/fail/aws-calls")" == "3" ]] \
  || fail "s3-fail: expected 3 aws attempts, got $(cat "$STATE/fail/aws-calls")"
[[ ! -f "$MODELS/s3-fail/never.bin" ]] || fail "s3-fail: partial file left behind"
ok "exhausted S3 retries report fetch-failed without a partial file"

# HuggingFace fetch for a non-whisper family: {name, sha256} entries resolve,
# the checksum is enforced, and the model label uses the file name.
run_driver hf-file "$OUT/hf-file.json" "$OUT/hf-file.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .model == "hf-file: m.bin"' "$OUT/hf-file.json" > /dev/null \
  || fail "hf-file: $(cat "$OUT/hf-file.json")"
[[ -f "$MODELS/hf-file/m.bin" ]] || fail "hf-file: model not staged"
ok "non-whisper HF family fetches by name with checksum verification"

# whisper keeps its size-keyed path: a size absent from models_by_size still
# reports not-in-registry (exit-66 contract) rather than failing the cell.
env -i PATH="$STUBBIN:$PATH" HOME="$TMP" BENCH_FAMILIES_JSON="$SPEC" \
  bash "$DRIVER" \
    --family whisper --whisper-size base --runs 2 --warmup 0 \
    --build-dir "$BUILD" --models-root "$MODELS" \
    --audio-dir "$AUDIO" --runner test \
    --out "$OUT/whisper-66.json" 2> "$OUT/whisper-66.err"
jq -e '.status == "not-in-registry" and .model == "whisper-base"' \
  "$OUT/whisper-66.json" > /dev/null || fail "whisper-66: $(cat "$OUT/whisper-66.json")"
ok "whisper size missing from the registry reports not-in-registry"

# ---- correctness scoring (WER path) -----------------------------------------
# The perf portion of every cell here is the nat-transcript-{perfect,wrong}
# stub or nat-ok — all emit a valid perf JSON. The tests below only assert
# the correctness fields the driver merged into result.json.

run_driver wer-perfect "$OUT/wer-perfect.json" "$OUT/wer-perfect.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == 0.0 and .correctness_kind == "wer"
       and (.correctness_reference | test("ref-perfect\\.txt$"))' \
  "$OUT/wer-perfect.json" > /dev/null \
  || fail "wer-perfect: $(cat "$OUT/wer-perfect.json")"
ok "correctness: exact-match transcript scores WER 0.0"

run_driver wer-nonzero "$OUT/wer-nonzero.json" "$OUT/wer-nonzero.err" BENCH_FAMILIES_JSON="$SPEC"
# 4 ref words, 1 substitution + 1 insertion = 2 edits / 4 words = 0.5
jq -e '.status == "ok" and .wer_median == 0.5 and .correctness_kind == "wer"' \
  "$OUT/wer-nonzero.json" > /dev/null \
  || fail "wer-nonzero: $(cat "$OUT/wer-nonzero.json")"
ok "correctness: 1 sub + 1 insertion scores WER 0.5"

run_driver wer-notranscript "$OUT/wer-notranscript.json" "$OUT/wer-notranscript.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == null and .correctness_kind == null' \
  "$OUT/wer-notranscript.json" > /dev/null \
  || fail "wer-notranscript: $(cat "$OUT/wer-notranscript.json")"
grep -q 'has no .transcript field' "$OUT/wer-notranscript.err" \
  || fail "wer-notranscript: missing transcript diagnosis not surfaced"
ok "correctness: bench JSON without .transcript => wer_median=null + diagnostic"

run_driver wer-badref "$OUT/wer-badref.json" "$OUT/wer-badref.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == null and .correctness_kind == null' \
  "$OUT/wer-badref.json" > /dev/null \
  || fail "wer-badref: $(cat "$OUT/wer-badref.json")"
grep -q 'reference file not found' "$OUT/wer-badref.err" \
  || fail "wer-badref: missing-reference diagnosis not surfaced"
ok "correctness: missing reference file => wer_median=null + diagnostic (perf still ok)"

run_driver wer-tw-warns "$OUT/wer-tw-warns.json" "$OUT/wer-tw-warns.err" BENCH_FAMILIES_JSON="$SPEC"
jq -e '.status == "ok" and .wer_median == null' \
  "$OUT/wer-tw-warns.json" > /dev/null \
  || fail "wer-tw-warns: $(cat "$OUT/wer-tw-warns.json")"
grep -q "declares correctness but bench_kind='time-wrapped'" "$OUT/wer-tw-warns.err" \
  || fail "wer-tw-warns: time-wrapped correctness warning not printed"
ok "correctness: time-wrapped family with a correctness block warns and is skipped"

echo "all $PASS checks passed"
