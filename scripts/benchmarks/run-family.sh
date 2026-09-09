#!/usr/bin/env bash
# Driver invoked by .github/workflows/speech-benchmark-desktop.yml for one
# (family, runner) matrix cell. Also runnable locally:
#
#   scripts/benchmarks/run-family.sh \
#       --family whisper --runs 3 --build-dir build \
#       --models-root $PWD/bench-models --out result.json
#
# Contract: emits one JSON file (--out) with the shape
#
#   {
#     "family":         "whisper",
#     "model":          "whisper-tiny",
#     "runner":         "linux",
#     "os":             "Linux",
#     "backend":        "CUDA" | "Metal" | "Vulkan" | "OpenCL" | "CPU" | "unknown"
#                       ("unknown" only on failed runs — a green run with no
#                        GPU-engagement evidence is reported as "CPU"),
#     "wall_ms_median": 1234.5,
#     "wall_ms_min":    1210.0,
#     "wall_ms_max":    1301.2,
#     "rtf_median":     0.612,        # null when the family's rtf can't be measured
#     "peak_rss_mib":   1024.5,       # null on runners without /usr/bin/time -v/-l
#     "runs":           3,
#     "status":         "ok" | "not-in-registry" | "missing-model"
#                       | "build-failed" | "run-failed" | "fetch-failed"
#                       | "infrastructure-failed" (only written by the
#                         workflow's pre-flight seed, never by this driver),
#     "notes":          "..."
#   }
#
# The summarizer (scripts/benchmarks/summarize.py) reads these and produces
# the single markdown table for $GITHUB_STEP_SUMMARY.

set -euo pipefail

FAMILY=""
RUNS=3
WARMUP=1
BUILD_DIR="build"
MODELS_ROOT="$PWD/bench-models"
OUT="result.json"
RUNNER_LABEL="${RUNNER_OS:-unknown}"
AUDIO_DIR="engines/parakeet/test/samples"
WHISPER_SIZE="tiny"

usage() {
  cat <<EOF >&2
usage: $0 --family FAMILY [--runs N] [--warmup N]
          [--build-dir DIR] [--models-root DIR] [--audio-dir DIR]
          [--runner LABEL] [--whisper-size tiny|base|small]
          --out result.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family)         FAMILY="$2"; shift 2 ;;
    --runs)           RUNS="$2"; shift 2 ;;
    --warmup)         WARMUP="$2"; shift 2 ;;
    --build-dir)      BUILD_DIR="$2"; shift 2 ;;
    --models-root)    MODELS_ROOT="$2"; shift 2 ;;
    --audio-dir)      AUDIO_DIR="$2"; shift 2 ;;
    --runner)         RUNNER_LABEL="$2"; shift 2 ;;
    --whisper-size)   WHISPER_SIZE="$2"; shift 2 ;;
    --out)            OUT="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$FAMILY" ]]; then usage; exit 2; fi

# ---- read families.json -----------------------------------------------------
# BENCH_FAMILIES_JSON overrides the spec path so the driver's own test suite
# (test-run-family.sh) can exercise it against synthetic families and stub
# binaries without touching the real spec.
FAMILIES_JSON="${BENCH_FAMILIES_JSON:-$(dirname "$0")/families.json}"
if ! [[ -f "$FAMILIES_JSON" ]]; then
  echo "families.json not found at $FAMILIES_JSON" >&2; exit 1
fi
if ! command -v jq >/dev/null; then
  echo "jq is required" >&2; exit 1
fi

# Local-run guard: on pre-macOS-26 the system date's %N returns the literal
# "N" rather than nanoseconds, awk parses only the leading digits, and every
# time-wrapped measurement silently reports ~0.0 ms. Check once at start
# with the absolute /bin/date path (matches the invocation in now_ns()
# below) and fail loudly instead — the runners are known-good, but the
# driver is documented as locally runnable.
if ! [[ "$(/bin/date +%N 2>/dev/null)" =~ ^[0-9]+$ ]]; then
  echo "run-family.sh: /bin/date +%N does not return nanoseconds on this host" >&2
  echo "  (got: '$(/bin/date +%N 2>/dev/null)')." >&2
  echo "  Time-wrapped bench timings would be silently zero. Bailing." >&2
  exit 1
fi

spec_field() {
  jq -r --arg family "$FAMILY" --arg field "$1" '.[$family][$field] // ""' "$FAMILIES_JSON"
}
spec_field_raw() {
  # For fields that may be null (audio_duration_seconds) — return jq's raw
  # "null" instead of an empty string so downstream can distinguish.
  jq -r --arg family "$FAMILY" --arg field "$1" '.[$family][$field]' "$FAMILIES_JSON"
}

BENCH_KIND="$(spec_field bench_kind)"
BINARY_REL="$(spec_field binary)"
S3_PREFIX="$(spec_field s3_prefix)"
NOTES="$(spec_field notes)"
AUDIO_DURATION_S="$(spec_field_raw audio_duration_seconds)"   # "null" or a number

if [[ -z "$BENCH_KIND" ]]; then
  echo "family '$FAMILY' not found in families.json" >&2; exit 1
fi

# Config sanity: correctness is honored for both bench kinds — native families
# feed compute-wer.py their .transcript field from --json-out; time-wrapped
# families feed the last run's captured stdout. Anything else (TTS-style
# families with no transcript at all, or a hypothetical third bench kind)
# is flagged but not fatal — the perf portion still works.
if jq -e --arg family "$FAMILY" '.[$family].correctness // empty' \
     "$FAMILIES_JSON" > /dev/null 2>&1; then
  case "$BENCH_KIND" in
    native|time-wrapped) : ;;
    *)
      echo "warning: family '$FAMILY' declares correctness but bench_kind='$BENCH_KIND'" >&2
      echo "  correctness scoring requires a hypothesis source; block will be ignored." >&2
      ;;
  esac
fi

MODEL_DIR="$MODELS_ROOT/$FAMILY"
mkdir -p "$MODEL_DIR"

# ---- output-JSON emitter (used from every exit path) ------------------------
BACKEND="unknown"     # populated from JSON or stderr scrape on a successful run
PEAK_RSS_MIB="null"   # tracked across runs; max seen
RTF_MEDIAN="null"     # from bench JSON (native) or computed (time-wrapped w/ audio_duration_seconds)
# Correctness fields — populated only when the family carries a `correctness`
# block AND the run produced enough output to score against a reference. All
# three stay null on families without a correctness spec so the result.json
# schema is stable (summarize.py renders "—" for null).
WER_MEDIAN="null"          # numeric WER in [0,1] or null
CORRECTNESS_KIND="null"    # "wer" (currently the only kind) or null
CORRECTNESS_REF="null"     # repo-relative path to the reference file or null

emit_json() {
  local status="$1" median="${2:-null}" wmin="${3:-null}" wmax="${4:-null}" extra="${5:-}"
  local uname_s; uname_s="$(uname -s)"
  jq -n \
    --arg  family "$FAMILY" \
    --arg  model  "$MODEL_LABEL" \
    --arg  runner "$RUNNER_LABEL" \
    --arg  os     "$uname_s" \
    --arg  backend "$BACKEND" \
    --argjson wall_median "$median" \
    --argjson wall_min    "$wmin" \
    --argjson wall_max    "$wmax" \
    --argjson rtf_median  "$RTF_MEDIAN" \
    --argjson peak_rss    "$PEAK_RSS_MIB" \
    --argjson runs        "$RUNS" \
    --argjson wer_median  "$WER_MEDIAN" \
    --argjson correctness_kind "$CORRECTNESS_KIND" \
    --argjson correctness_reference "$CORRECTNESS_REF" \
    --arg  status "$status" \
    --arg  notes  "$NOTES$extra" \
    '{family:$family, model:$model, runner:$runner, os:$os, backend:$backend,
      wall_ms_median:$wall_median, wall_ms_min:$wall_min, wall_ms_max:$wall_max,
      rtf_median:$rtf_median, peak_rss_mib:$peak_rss,
      wer_median:$wer_median, correctness_kind:$correctness_kind,
      correctness_reference:$correctness_reference,
      runs:$runs, status:$status, notes:$notes}' > "$OUT"
  echo "wrote $OUT"
  cat "$OUT" >&2
}

# The `source` field per family (default "s3"): "s3" for the tetherto S3
# registry via aws s3 cp, "huggingface" for `curl` against a pinned HF
# revision. Vanilla whisper GGMLs (tiny/base/small) live on HuggingFace,
# not S3 — the whisper family sets source="huggingface" and provides
# hf_repo / hf_ref / models_by_size[<size>] = {name, sha256?} entries.
SOURCE="$(spec_field source)"
[[ -n "$SOURCE" ]] || SOURCE="s3"

# ---- resolve MODEL_LABEL + placeholders -------------------------------------
# models[] entries may be strings or objects ({s3, as} for renamed S3 keys,
# {name, sha256?} for HuggingFace files) — the label wants the filename the
# bench actually loads.
if [[ "$FAMILY" == "whisper" ]]; then
  MODEL_LABEL="whisper-$WHISPER_SIZE"
else
  first_model="$(jq -r --arg f "$FAMILY" \
    '.[$f].models[0] // "" | if type == "object" then (.name // .as // .s3 // "") else . end' \
    "$FAMILIES_JSON")"
  MODEL_LABEL="$FAMILY: ${first_model##*/}"
fi

# ---- fetch models from HuggingFace (curl, optional sha256 verification) ----
# whisper picks one size-keyed entry from models_by_size; every other HF
# family lists its files in models[] as {name, sha256?} objects (or plain
# name strings), resolved against the family's pinned hf_repo@hf_ref.
fetch_from_hf() {
  local repo ref entries
  repo="$(spec_field hf_repo)"
  ref="$(spec_field hf_ref)"
  if [[ -z "$repo" || -z "$ref" ]]; then
    echo "$FAMILY: missing hf_repo / hf_ref in families.json" >&2
    return 1
  fi

  if [[ "$FAMILY" == "whisper" ]]; then
    local wname wsha
    wname="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size].name // ""' "$FAMILIES_JSON")"
    wsha="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size].sha256 // ""' "$FAMILIES_JSON")"
    if [[ -z "$wname" || "$wname" == "null" ]]; then
      echo "$FAMILY-$WHISPER_SIZE: no HF model name in registry (not-in-registry)"
      return 66
    fi
    entries="$wname"$'\t'"$wsha"
  else
    entries="$(jq -r --arg family "$FAMILY" \
      '.[$family].models[]? | if type == "object" then "\(.name // "")\t\(.sha256 // "")" else "\(.)\t" end' \
      "$FAMILIES_JSON")"
    if [[ -z "$entries" ]]; then
      return 65      # same shape as the S3 path: no files pinned yet
    fi
  fi

  local line name sha256 dest url
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    name="${line%%$'\t'*}"
    sha256="${line#*$'\t'}"
    if [[ -z "$name" ]]; then
      echo "$FAMILY: HF model entry without a name in families.json" >&2
      return 1
    fi
    dest="$MODEL_DIR/$name"
    if [[ -f "$dest" ]]; then continue; fi
    url="https://huggingface.co/${repo}/resolve/${ref}/${name}"
    echo "fetch $url -> $dest"
    # curl -f: fail on HTTP >= 400 rather than writing an HTML error page and
    # returning success. -L: follow redirects (HF uses them). --retry rides
    # out transient network resets. Explicit || return $? so a fetch failure
    # propagates instead of the loop swallowing it (bash's `set -e` is
    # disabled inside a function called with `||`).
    curl -fL -o "$dest" "$url" --silent --show-error \
      --retry 3 --retry-delay 5 || return $?

    if [[ -n "$sha256" && "$sha256" != "null" ]]; then
      if command -v sha256sum >/dev/null; then
        echo "$sha256  $dest" | sha256sum -c - || return $?
      elif command -v shasum >/dev/null; then
        echo "$sha256  $dest" | shasum -a 256 -c - || return $?
      fi
    fi
  done <<< "$entries"
  return 0
}

# aws s3 cp retries API-level throttles internally, but a mid-stream
# connection reset (acestep's ~5 GB shard has hit ConnectionResetError(104))
# aborts the transfer and leaves a partial file. Retry the whole copy,
# removing the partial first so a truncated file can never satisfy the
# already-downloaded check on the next attempt or the next dispatch.
s3_cp_retry() {
  local s3url="$1" dest="$2"
  local attempt rc=1
  for attempt in 1 2 3; do
    if aws s3 cp "$s3url" "$dest" --no-progress; then
      return 0
    else
      rc=$?
    fi
    rm -f "$dest"
    if [[ $attempt -lt 3 ]]; then
      echo "aws s3 cp $s3url failed (exit $rc, attempt $attempt/3) — retrying" >&2
      # BENCH_RETRY_SLEEP_S: test hook (test-run-family.sh sets 0).
      sleep $(( attempt * ${BENCH_RETRY_SLEEP_S:-5} ))
    fi
  done
  return $rc
}

# ---- fetch models from S3 (skipped when models[] empty / no bucket) --------
fetch_from_s3() {
  local bucket="${MODEL_S3_BUCKET:-}"
  if [[ -z "$bucket" ]]; then
    echo "MODEL_S3_BUCKET not set — skipping fetch (assuming local models present)" >&2
    return 0
  fi

  # `mapfile -t` is bash 4+, so the macOS self-hosted runner would break if
  # /usr/bin/env bash resolved to the system 3.2. Use a portable read loop.
  # Each row is "s3-key<TAB>rename-to" — rename-to is empty for string
  # entries and the engine-expected filename for object entries like
  # {"s3": "2026-07-23/voice-en.gguf", "as": "voice.gguf"} (cosyvoice's
  # baked voice is stored language-suffixed in S3 but the engine looks
  # for the canonical `voice.gguf`).
  local -a keys=()
  local line
  while IFS= read -r line; do keys+=("$line"); done < <(jq -r --arg family "$FAMILY" \
    '.[$family].models[]? | if type == "string" then "\(.)\t" else "\(.s3)\t\(.as // "")" end' \
    "$FAMILIES_JSON")
  if [[ ${#keys[@]} -eq 0 ]]; then
    return 65      # minimax shape: no S3 path
  fi

  for row in "${keys[@]}"; do
    local key="${row%%$'\t'*}"
    local rename_to="${row#*$'\t'}"
    local basename="${rename_to:-${key##*/}}"
    local dest="$MODEL_DIR/$basename"
    if [[ -f "$dest" ]]; then continue; fi
    local s3url="s3://$bucket/qvac_models_compiled/ggml/$S3_PREFIX/$key"
    echo "fetch $s3url -> $dest"
    # `|| return $?` so an S3 failure (404, denied, network) propagates as a
    # non-zero exit from the function — otherwise `set -e` is disabled by
    # the caller's `||` guard and the loop continues past a missing model,
    # eventually running the bench binary against nothing and mis-reporting
    # `run-failed` for what's actually a `fetch-failed`.
    s3_cp_retry "$s3url" "$dest" || return $?
  done
  return 0
}

fetch_models() {
  case "$SOURCE" in
    huggingface) fetch_from_hf ;;
    s3|"")       fetch_from_s3 ;;
    *) echo "unknown source '$SOURCE' for family '$FAMILY'" >&2; return 1 ;;
  esac
}

# ---- pick the model path for whisper (single -m arg) ------------------------
MODEL_PATH=""
if [[ "$FAMILY" == "whisper" ]]; then
  wname="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size].name // ""' "$FAMILIES_JSON")"
  if [[ -n "$wname" && "$wname" != "null" ]]; then
    MODEL_PATH="$MODEL_DIR/$wname"
  fi
fi

fetch_status=0
fetch_models || fetch_status=$?
if   [[ $fetch_status -eq 66 ]]; then
  emit_json "not-in-registry" null null null " (whisper $WHISPER_SIZE size not in the model registry)"; exit 0
elif [[ $fetch_status -eq 65 ]]; then
  emit_json "missing-model"   null null null " (no S3 path in the registry — follow-up ticket)"; exit 0
elif [[ $fetch_status -ne 0 ]]; then
  emit_json "fetch-failed"    null null null " (model fetch failed with status $fetch_status)"; exit 0
fi

# ---- native-mode bench JSON path (resolved before args expansion) ----------
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
JSON_OUT="$tmp_dir/native.json"

# ---- build the argv array from the JSON-array `args` field -----------------
# families.json stores `args` as a JSON array so multi-word values (e.g.
# "benchmark run") stay a single argv token — a plain space-separated string
# would be word-split by the shell into ['benchmark], [run'] on
# expansion. Every element is passed through the same placeholder
# substitution table.
expand_placeholder() {
  local s="$1"
  s="${s//\$\{MODEL_DIR\}/$MODEL_DIR}"
  s="${s//\$\{MODELS_ROOT\}/$MODELS_ROOT}"
  s="${s//\$\{AUDIO_DIR\}/$AUDIO_DIR}"
  s="${s//\$\{MODEL_PATH\}/${MODEL_PATH:-}}"
  s="${s//\$\{RUNS\}/$RUNS}"
  s="${s//\$\{WARMUP\}/$WARMUP}"
  s="${s//\$\{JSON_OUT\}/$JSON_OUT}"
  printf '%s' "$s"
}

BENCH_ARGS=()
_line=""
while IFS= read -r _line; do
  BENCH_ARGS+=("$(expand_placeholder "$_line")")
done < <(jq -r --arg family "$FAMILY" '.[$family].args[]?' "$FAMILIES_JSON")

BINARY="$BUILD_DIR/$BINARY_REL"
if ! [[ -x "$BINARY" ]]; then
  echo "binary not found or not executable: $BINARY" >&2
  emit_json "build-failed" null null null " (binary '$BINARY' missing)"; exit 0
fi

# ---- /usr/bin/time wrapper (portable RSS capture) --------------------------
# Linux GNU time: --verbose prints 'Maximum resident set size (kbytes): NNN'.
# macOS BSD time: -l prints '  NNN  maximum resident set size' in BYTES.
# Both write to stderr. We capture stderr into a file and parse after the run.
have_gnu_time() {
  /usr/bin/time --version >/dev/null 2>&1
}

wrap_time() {
  # Prepend the correct time invocation. When neither is available, skip.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    /usr/bin/time -l "$@"
  elif have_gnu_time; then
    /usr/bin/time -v "$@"
  else
    "$@"
  fi
}

parse_rss_from_time_stderr() {
  # $1: path to a stderr log produced by wrap_time.
  # Emits the peak RSS in MiB (or empty when none can be parsed).
  local file="$1"
  local kb bytes
  # GNU time: kbytes
  kb="$(grep -oE 'Maximum resident set size \(kbytes\): [0-9]+' "$file" 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -n "$kb" ]]; then
    awk -v k="$kb" 'BEGIN { printf "%.1f", k / 1024.0 }'
    return
  fi
  # BSD time: bytes on a line ending 'maximum resident set size'
  bytes="$(grep -oE '[0-9]+  *maximum resident set size' "$file" 2>/dev/null | grep -oE '^[0-9]+' | head -1 || true)"
  if [[ -n "$bytes" ]]; then
    awk -v b="$bytes" 'BEGIN { printf "%.1f", b / (1024.0 * 1024.0) }'
    return
  fi
  echo ""
}

# ---- backend extraction ----------------------------------------------------
# Attribution chain over the run's stdout AND stderr, most explicit first:
#   1. `using <NAME> backend`  — parakeet-cli / whisper / tts-cli style line.
#   2. `backend: <NAME>`       — the *-bench binaries print this on stdout
#      (matches lavasr's "denoiser backend: <NAME>" too).
#   3. GPU-engagement evidence — ggml's own compute-init logs, independent of
#      engine verbosity: creating a Metal compute context always logs
#      ggml_metal_library_init, Vulkan compiles shaders, OpenCL compiles its
#      program. Device *enumeration* alone (which also happens on CPU runs of
#      GPU-enabled builds) does not print these, so enumeration cannot
#      mis-label a CPU run.
# When every pattern misses, the caller labels a green run "CPU": with no
# compute-init evidence, nothing ran on a GPU. The residual risk is an engine
# that installs a silent ggml log handler AND prints no backend line of its
# own; none of the benched binaries do. Failed runs keep "unknown" — their
# logs may be truncated mid-init.
parse_backend_from_logs() {
  local hit f
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    hit="$(grep -oE 'using [A-Za-z0-9_]+ backend' "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)"
    if [[ -n "$hit" ]]; then echo "$hit"; return; fi
  done
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    hit="$(grep -oE 'backend: [A-Za-z0-9_]+' "$f" 2>/dev/null | head -1 | awk '{print $2}' || true)"
    if [[ -n "$hit" ]]; then echo "$hit"; return; fi
  done
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    if grep -q  'ggml_metal_library_init'            "$f" 2>/dev/null; then echo "Metal";  return; fi
    if grep -q  'ggml_vulkan: Compiling shaders'     "$f" 2>/dev/null; then echo "Vulkan"; return; fi
    if grep -qiE 'ggml_opencl:.*(compil|program)'    "$f" 2>/dev/null; then echo "OpenCL"; return; fi
  done
  echo ""
}

# ---- correctness scoring (WER) ---------------------------------------------
# When a family in families.json carries a `correctness` block, score the
# hypothesis transcript against the checked-in reference and echo
# "wer|kind|reference" (or empty when scoring is skipped). Callers set the
# emitter's WER_MEDIAN / CORRECTNESS_* vars from the returned values.
#
# Two hypothesis sources, one per bench kind:
#   mode="bench-json" <path>  — native families whose --json-out carries
#                               .transcript (parakeet).
#   mode="text-file"  <path>  — time-wrapped families whose CLI prints its
#                               transcript on stdout (whisper -nt).
#
# All error paths are non-fatal: a missing reference file, missing transcript
# in the bench JSON, missing stdout capture, or compute-wer.py failure logs
# to stderr and returns empty so the perf portion of the run still succeeds
# — a green benchmark with correctness=null tells the reader "perf is fine,
# correctness didn't run" instead of dragging the whole cell to run-failed.
score_correctness() {
  local mode="$1" src_path="$2"
  local spec_kind spec_ref spec_norm
  spec_kind="$(jq -r --arg family "$FAMILY" \
    '.[$family].correctness.kind // ""'       "$FAMILIES_JSON")"
  spec_ref="$(jq  -r --arg family "$FAMILY" \
    '.[$family].correctness.reference // ""'  "$FAMILIES_JSON")"
  spec_norm="$(jq -r --arg family "$FAMILY" \
    '.[$family].correctness.normalizer // "english"' "$FAMILIES_JSON")"

  # No correctness block — silent no-op (the common case).
  if [[ -z "$spec_kind" ]]; then return 0; fi

  if [[ "$spec_kind" != "wer" ]]; then
    echo "$FAMILY: unsupported correctness.kind '$spec_kind' (only 'wer' today)" >&2
    return 0
  fi

  # Resolve the reference path against the repo root (two levels up from
  # $(dirname "$0")/../..), or accept it as absolute. This mirrors how the
  # workflow invokes us with the repo checked out at $PWD.
  local ref_path="$spec_ref"
  if [[ "$ref_path" != /* ]]; then
    ref_path="$(cd "$(dirname "$0")/../.." && pwd)/$spec_ref"
  fi
  if ! [[ -f "$ref_path" ]]; then
    echo "$FAMILY: correctness reference file not found: $ref_path" >&2
    return 0
  fi

  # Build the --hypothesis-* args for compute-wer.py. Both modes preserve the
  # "present-but-empty transcript => WER 1.0" invariant: bench-json checks
  # field presence separately from truthiness so `.transcript=""` reaches
  # compute-wer.py rather than being conflated with field-absent; text-file
  # passes even an empty stdout capture through, which compute-wer.py scores
  # as a full miss when the reference is non-empty.
  local -a hyp_args=()
  case "$mode" in
    bench-json)
      local hyp
      if ! jq -e 'has("transcript") and (.transcript | type == "string")' \
             "$src_path" > /dev/null 2>&1; then
        echo "$FAMILY: bench JSON has no .transcript field — correctness skipped" >&2
        return 0
      fi
      hyp="$(jq -r '.transcript' "$src_path" 2>/dev/null || true)"
      hyp_args=(--hypothesis-text "$hyp")
      ;;
    text-file)
      if ! [[ -f "$src_path" ]]; then
        echo "$FAMILY: hypothesis stdout capture missing: $src_path — correctness skipped" >&2
        return 0
      fi
      hyp_args=(--hypothesis-file "$src_path")
      ;;
    *)
      echo "$FAMILY: internal error: unknown score_correctness mode '$mode'" >&2
      return 0
      ;;
  esac

  local wer_out wer
  # Prefer the caller-python interpreter that ran the workflow's other Python
  # steps (python3). No venv assumed — compute-wer.py imports only stdlib.
  if ! wer_out="$(python3 "$(dirname "$0")/compute-wer.py" \
       "${hyp_args[@]}" \
       --reference       "$ref_path" \
       --normalizer      "$spec_norm" 2>&1)"; then
    echo "$FAMILY: compute-wer.py failed: $wer_out" >&2
    return 0
  fi
  wer="$(echo "$wer_out" | jq -r '.wer // empty' 2>/dev/null || true)"
  if [[ -z "$wer" ]]; then
    echo "$FAMILY: could not parse WER from compute-wer.py output: $wer_out" >&2
    return 0
  fi

  # Emit the tuple the caller expects. Reference is echoed as the
  # families.json-declared (repo-relative) path so the summary table stays
  # stable across runners with different absolute checkout paths.
  echo "$wer|$spec_kind|$spec_ref"
}

# Nanosecond timestamp. macOS 26 / arm64's /bin/date does support %N despite
# being a GNU extension; the script-start guard above bails loudly if we
# land on an older date that would silently return zeros. Invokes /bin/date
# by absolute path (matching the guard) so a stray coreutils `date` earlier
# on PATH can't diverge from what the guard actually verified.
now_ns() {
  /bin/date +%s%N
}

# ---- native bench: one invocation, parse the emitted JSON ------------------
# Native JSON schema tolerance — different bench binaries put the same
# information in different keys. We fall back through the known layouts:
#   * .stages.tot|.total|.e2e     -- supertonic/parler/cosyvoice
#   * .inference_ms               -- parakeet-cli --bench
#   * .end_to_end_ms              -- alternate parakeet naming (defensive)
# RTF likewise: .rtf.median (supertonic-shape) vs .rtf_median (parakeet).
# Backend likewise: .backend at top level (parakeet) — otherwise fall back
# to stderr parsing.
run_native() {
  local native_json="$1"; shift
  local stderr_log="$1"; shift
  # Ensure stderr_log exists even if wrap_time fails to write it, so the
  # caller can safely `parse_backend_from_logs "$stderr_log" ...` on either
  # success or failure paths.
  : > "$stderr_log"

  if wrap_time "$BINARY" "${BENCH_ARGS[@]}" \
      > "$stderr_log.stdout" 2>> "$stderr_log"; then
    :
  else
    # Surface the child's actual failure to the step log so debugging a
    # run-failed cell doesn't require local repro.
    { echo "--- native bench failed; last 30 lines of stdout ---"
      tail -n 30 "$stderr_log.stdout" 2>/dev/null || true
      echo "--- last 30 lines of stderr ---"
      tail -n 30 "$stderr_log"        2>/dev/null || true
      echo "--- end ---"
    } >&2
    return 1
  fi

  if ! [[ -s "$native_json" ]]; then
    # Exit 0 without the JSON means the binary took a code path that never
    # reaches its bench writer (this is how the sortformer diarization cells
    # failed invisibly) — dump the child's output so the step log shows what
    # actually ran, exactly like the non-zero-exit branch above.
    { echo "native bench exited 0 but did not emit the --json-out file"
      echo "--- last 30 lines of stdout ---"
      tail -n 30 "$stderr_log.stdout" 2>/dev/null || true
      echo "--- last 30 lines of stderr ---"
      tail -n 30 "$stderr_log"        2>/dev/null || true
      echo "--- end ---"
    } >&2
    return 1
  fi

  local median wmin wmax rtf backend
  median="$(jq -r '(.stages.tot.median_ms // .stages.total.median_ms // .stages.e2e.median_ms // .inference_ms.median // .end_to_end_ms.median // empty)' "$native_json" 2>/dev/null || true)"
  wmin="$(jq   -r '(.stages.tot.min_ms    // .stages.total.min_ms    // .stages.e2e.min_ms    // .inference_ms.min    // .end_to_end_ms.min    // empty)' "$native_json" 2>/dev/null || true)"
  wmax="$(jq   -r '(.stages.tot.max_ms    // .stages.total.max_ms    // .stages.e2e.max_ms    // .inference_ms.max    // .end_to_end_ms.max    // empty)' "$native_json" 2>/dev/null || true)"
  rtf="$(jq    -r '(.rtf.median // .rtf_median // empty)' "$native_json" 2>/dev/null || true)"
  backend="$(jq -r '(.backend // empty)' "$native_json" 2>/dev/null || true)"

  # Echo "median|min|max|rtf|backend" for the caller.
  echo "${median}|${wmin}|${wmax}|${rtf}|${backend}"
}

# ---- time-wrapped bench: N invocations, we take the median ------------------
run_one_time_wrapped() {
  local iter="$1" stderr_log="$2" rss_log="$3"
  # stderr_log MUST exist by function-exit — the caller cat's it into the
  # combined log even on failure. Touch first, then let wrap_time's redirect
  # append its stderr.
  : > "$stderr_log"
  : > "$rss_log"
  local start_ns end_ns
  start_ns="$(now_ns)"
  if ! wrap_time "$BINARY" "${BENCH_ARGS[@]}" > "$stderr_log.stdout" 2> "$rss_log"; then
    cat "$rss_log" >> "$stderr_log"
    tail -20 "$stderr_log" >&2
    return 1
  fi
  end_ns="$(now_ns)"
  cat "$rss_log" >> "$stderr_log"
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.1f", (e - s) / 1000000.0 }'
}

# ---- run --------------------------------------------------------------------
case "$BENCH_KIND" in
  native)
    stderr_log="$tmp_dir/stderr.log"
    parsed=""
    if ! parsed="$(run_native "$JSON_OUT" "$stderr_log")"; then
      BACKEND="$(parse_backend_from_logs "$stderr_log" "$stderr_log.stdout")"
      BACKEND="${BACKEND:-unknown}"
      emit_json "run-failed" null null null " (native bench invocation failed)"
      exit 0
    fi
    IFS='|' read -r n_med n_min n_max n_rtf n_backend <<< "$parsed"
    [[ -n "$n_med" ]] || n_med="null"
    [[ -n "$n_min" ]] || n_min="null"
    [[ -n "$n_max" ]] || n_max="null"
    if [[ -n "$n_rtf" ]]; then RTF_MEDIAN="$n_rtf"; fi

    # Prefer the backend the bench JSON declared (parakeet reports the
    # post-fallback active backend); fall back to log scraping, then to CPU —
    # the run finished green with no GPU-engagement evidence.
    if [[ -n "$n_backend" ]]; then
      BACKEND="$n_backend"
    else
      BACKEND="$(parse_backend_from_logs "$stderr_log" "$stderr_log.stdout")"
      BACKEND="${BACKEND:-CPU}"
    fi
    rss="$(parse_rss_from_time_stderr "$stderr_log")"
    [[ -n "$rss" ]] && PEAK_RSS_MIB="$rss"

    # Correctness scoring — no-op unless the family declares a correctness
    # block. Runs after perf is captured so a scoring failure never
    # downgrades a green perf run.
    corr_out=""
    corr_out="$(score_correctness bench-json "$JSON_OUT" || true)"
    if [[ -n "$corr_out" ]]; then
      IFS='|' read -r c_wer c_kind c_ref <<< "$corr_out"
      [[ -n "$c_wer"  ]] && WER_MEDIAN="$c_wer"
      [[ -n "$c_kind" ]] && CORRECTNESS_KIND="$(printf '%s' "$c_kind" | jq -R .)"
      [[ -n "$c_ref"  ]] && CORRECTNESS_REF="$(printf '%s'  "$c_ref"  | jq -R .)"
    fi

    emit_json "ok" "$n_med" "$n_min" "$n_max"
    ;;

  time-wrapped)
    for i in $(seq 1 "$WARMUP"); do
      echo "warmup $i/$WARMUP" >&2
      wu_stderr="$tmp_dir/warmup-$i.err"
      wu_rss="$tmp_dir/warmup-$i.rss"
      if ! run_one_time_wrapped "$i" "$wu_stderr" "$wu_rss" >/dev/null; then
        BACKEND="$(parse_backend_from_logs "$wu_stderr" "$wu_stderr.stdout")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (warmup failed)"; exit 0
      fi
    done

    declare -a wall_ms=()
    max_rss_seen=""
    combined_stderr="$tmp_dir/combined.err"
    combined_stdout="$tmp_dir/combined.out"
    : > "$combined_stderr"
    : > "$combined_stdout"
    for i in $(seq 1 "$RUNS"); do
      echo "run $i/$RUNS" >&2
      r_stderr="$tmp_dir/run-$i.err"
      r_rss="$tmp_dir/run-$i.rss"
      if ! ms="$(run_one_time_wrapped "$i" "$r_stderr" "$r_rss")"; then
        cat "$r_stderr" >> "$combined_stderr"
        cat "$r_stderr.stdout" >> "$combined_stdout" 2>/dev/null || true
        BACKEND="$(parse_backend_from_logs "$combined_stderr" "$combined_stdout")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (timed run $i failed)"; exit 0
      fi
      wall_ms+=("$ms")
      cat "$r_stderr" >> "$combined_stderr"
      cat "$r_stderr.stdout" >> "$combined_stdout" 2>/dev/null || true
      rss_i="$(parse_rss_from_time_stderr "$r_stderr")"
      if [[ -n "$rss_i" ]]; then
        if [[ -z "$max_rss_seen" ]] || awk -v a="$rss_i" -v b="$max_rss_seen" 'BEGIN{exit !(a>b)}'; then
          max_rss_seen="$rss_i"
        fi
      fi
    done

    stats="$(printf '%s\n' "${wall_ms[@]}" | jq -s '{
      median: (sort | if length%2==1 then .[length/2|floor] else (.[length/2-1] + .[length/2]) / 2 end),
      min: min,
      max: max
    }')"
    med="$(echo "$stats" | jq '.median')"
    mn="$(echo "$stats" | jq '.min')"
    mx="$(echo "$stats" | jq '.max')"

    BACKEND="$(parse_backend_from_logs "$combined_stderr" "$combined_stdout")"
    BACKEND="${BACKEND:-CPU}"
    if [[ -n "$max_rss_seen" ]]; then PEAK_RSS_MIB="$max_rss_seen"; fi

    # Compute RTF only when families.json declared audio_duration_seconds.
    if [[ "$AUDIO_DURATION_S" != "null" && -n "$AUDIO_DURATION_S" ]]; then
      RTF_MEDIAN="$(awk -v m="$med" -v s="$AUDIO_DURATION_S" 'BEGIN { printf "%.3f", m / (s * 1000.0) }')"
    fi

    # Correctness scoring — no-op unless the family declares a correctness
    # block. The hypothesis is the last successful run's captured stdout
    # ($r_stderr.stdout from the final loop iteration, still on disk in
    # $tmp_dir). Runs after perf is captured so a scoring failure never
    # downgrades a green perf run.
    corr_out=""
    corr_out="$(score_correctness text-file "$r_stderr.stdout" || true)"
    if [[ -n "$corr_out" ]]; then
      IFS='|' read -r c_wer c_kind c_ref <<< "$corr_out"
      [[ -n "$c_wer"  ]] && WER_MEDIAN="$c_wer"
      [[ -n "$c_kind" ]] && CORRECTNESS_KIND="$(printf '%s' "$c_kind" | jq -R .)"
      [[ -n "$c_ref"  ]] && CORRECTNESS_REF="$(printf '%s'  "$c_ref"  | jq -R .)"
    fi

    emit_json "ok" "$med" "$mn" "$mx"
    ;;

  *)
    emit_json "run-failed" null null null " (unknown bench_kind '$BENCH_KIND')"
    exit 1
    ;;
esac
