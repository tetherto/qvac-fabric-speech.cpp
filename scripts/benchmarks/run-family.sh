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
#     "backend":        "CUDA" | "Metal" | "Vulkan" | "OpenCL" | "unknown",
#     "encoder_backend":"coreml-all" | "ggml-metal" | ... | "unknown",
#     "encoder_coreml_all_runs": true | false | null,
#     "encoder_ms_median": 32.1,       # native engines that report it
#     "baseline_encoder_ms_median": 121.3, # Darwin Core ML comparisons
#     "encoder_speedup": 3.78,         # baseline / Core ML
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
FAMILIES_JSON="$(dirname "$0")/families.json"
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

MODEL_DIR="$MODELS_ROOT/$FAMILY"
mkdir -p "$MODEL_DIR"

# ---- output-JSON emitter (used from every exit path) ------------------------
BACKEND="unknown"     # populated from JSON or stderr scrape on a successful run
ENCODER_BACKEND="unknown"
ENCODER_COREML_ALL_RUNS="null"
ENCODER_MS_MEDIAN="null"
BASELINE_BACKEND=""
BASELINE_ENCODER_BACKEND=""
BASELINE_ENCODER_MS_MEDIAN="null"
BASELINE_INFERENCE_MS_MEDIAN="null"
BASELINE_RTF_MEDIAN="null"
ENCODER_SPEEDUP="null"
INFERENCE_SPEEDUP="null"
PEAK_RSS_MIB="null"   # tracked across runs; max seen
RTF_MEDIAN="null"     # from bench JSON (native) or computed (time-wrapped w/ audio_duration_seconds)

emit_json() {
  local status="$1" median="${2:-null}" wmin="${3:-null}" wmax="${4:-null}" extra="${5:-}"
  local uname_s; uname_s="$(uname -s)"
  jq -n \
    --arg  family "$FAMILY" \
    --arg  model  "$MODEL_LABEL" \
    --arg  runner "$RUNNER_LABEL" \
    --arg  os     "$uname_s" \
    --arg  backend "$BACKEND" \
    --arg  encoder_backend "$ENCODER_BACKEND" \
    --argjson encoder_coreml_all_runs "$ENCODER_COREML_ALL_RUNS" \
    --argjson encoder_ms_median "$ENCODER_MS_MEDIAN" \
    --arg  baseline_backend "$BASELINE_BACKEND" \
    --arg  baseline_encoder_backend "$BASELINE_ENCODER_BACKEND" \
    --argjson baseline_encoder_ms_median "$BASELINE_ENCODER_MS_MEDIAN" \
    --argjson baseline_inference_ms_median "$BASELINE_INFERENCE_MS_MEDIAN" \
    --argjson baseline_rtf_median "$BASELINE_RTF_MEDIAN" \
    --argjson encoder_speedup "$ENCODER_SPEEDUP" \
    --argjson inference_speedup "$INFERENCE_SPEEDUP" \
    --argjson wall_median "$median" \
    --argjson wall_min    "$wmin" \
    --argjson wall_max    "$wmax" \
    --argjson rtf_median  "$RTF_MEDIAN" \
    --argjson peak_rss    "$PEAK_RSS_MIB" \
    --argjson runs        "$RUNS" \
    --arg  status "$status" \
    --arg  notes  "$NOTES$extra" \
    '{family:$family, model:$model, runner:$runner, os:$os, backend:$backend,
      encoder_backend:$encoder_backend,
      encoder_coreml_all_runs:$encoder_coreml_all_runs,
      encoder_ms_median:$encoder_ms_median,
      baseline_backend:(if $baseline_backend == "" then null else $baseline_backend end),
      baseline_encoder_backend:(if $baseline_encoder_backend == "" then null else $baseline_encoder_backend end),
      baseline_encoder_ms_median:$baseline_encoder_ms_median,
      baseline_inference_ms_median:$baseline_inference_ms_median,
      baseline_rtf_median:$baseline_rtf_median,
      encoder_speedup:$encoder_speedup, inference_speedup:$inference_speedup,
      wall_ms_median:$wall_median, wall_ms_min:$wall_min, wall_ms_max:$wall_max,
      rtf_median:$rtf_median, peak_rss_mib:$peak_rss,
      runs:$runs, status:$status, notes:$notes}' > "$OUT"
  echo "wrote $OUT"
  cat "$OUT" >&2
}

# ---- resolve MODEL_LABEL + placeholders -------------------------------------
if [[ "$FAMILY" == "whisper" ]]; then
  MODEL_LABEL="whisper-$WHISPER_SIZE"
else
  first_model="$(jq -r --arg f "$FAMILY" '.[$f].models[0] // ""' "$FAMILIES_JSON")"
  MODEL_LABEL="$FAMILY: ${first_model##*/}"
fi

# ---- fetch models from S3 (skipped when models[] empty / no bucket) --------
fetch_models() {
  local bucket="${MODEL_S3_BUCKET:-}"
  if [[ -z "$bucket" ]]; then
    echo "MODEL_S3_BUCKET not set — skipping remote fetch (assuming local models present)" >&2
  fi

  # `mapfile -t` is bash 4+, so the macOS self-hosted runner would break if
  # /usr/bin/env bash resolved to the system 3.2. Use a portable read loop.
  local -a keys=()
  local line
  if [[ "$FAMILY" == "whisper" ]]; then
    while IFS= read -r line; do keys+=("$line"); done < <(jq -r --arg size "$WHISPER_SIZE" \
      '.whisper.models_by_size[$size][]?' "$FAMILIES_JSON")
    if [[ ${#keys[@]} -eq 0 ]]; then
      echo "$FAMILY-$WHISPER_SIZE: no S3 key in registry (not-in-registry)"
      return 66
    fi
  else
    while IFS= read -r line; do keys+=("$line"); done < <(jq -r --arg family "$FAMILY" \
      '.[$family].models[]?' "$FAMILIES_JSON")
    if [[ ${#keys[@]} -eq 0 ]]; then
      return 65      # minimax shape: no S3 path
    fi
  fi

  for key in "${keys[@]}"; do
    local basename="${key##*/}"
    local dest="$MODEL_DIR/$basename"
    if [[ -f "$dest" ]]; then continue; fi
    if [[ -z "$bucket" ]]; then
      echo "required local model is missing: $dest" >&2
      return 1
    fi
    local s3url="s3://$bucket/qvac_models_compiled/ggml/$S3_PREFIX/$key"
    echo "fetch $s3url -> $dest"
    aws s3 cp "$s3url" "$dest" --no-progress
  done

  return 0
}

# ---- pick the model path for whisper (single -m arg) ------------------------
MODEL_PATH=""
if [[ "$FAMILY" == "whisper" ]]; then
  wkey="$(jq -r --arg size "$WHISPER_SIZE" '.whisper.models_by_size[$size][0] // ""' "$FAMILIES_JSON")"
  if [[ -n "$wkey" ]]; then
    MODEL_PATH="$MODEL_DIR/${wkey##*/}"
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

# The Core ML bundle is derived from the exact F16 model under test. A restored
# Actions cache makes this a no-op on subsequent runs; the first run exports and
# compiles locally on the Apple Silicon benchmark host.
if [[ "$(uname -s)" == "Darwin" ]] &&
   [[ "$(spec_field coreml_compare_on_darwin)" == "true" ]]; then
  coreml_sidecar="$MODEL_DIR/parakeet-tdt-0.6b-v3-encoder.mlmodelc"
  if [[ ! -d "$coreml_sidecar" ]]; then
    coreml_python="${PARAKEET_COREML_PYTHON:-python3}"
    if [[ ! -x "$coreml_python" ]] && ! command -v "$coreml_python" >/dev/null 2>&1; then
      emit_json "run-failed" null null null " (Core ML exporter Python not found: $coreml_python)"
      exit 0
    fi
    coreml_package="$tmp_dir/parakeet-tdt-0.6b-v3-encoder.mlpackage"
    echo "generating Core ML encoder sidecar from the benchmark GGUF" >&2
    if ! "$coreml_python" engines/parakeet/scripts/export-encoder-coreml.py \
        --gguf "$MODEL_DIR/parakeet-tdt-0.6b-v3.f16.gguf" \
        --n-mel-frames 1501 \
        --palettize-bits 6 \
        --palettize-group-size 16 \
        --out "$coreml_package" \
        --compile-dir "$MODEL_DIR"; then
      emit_json "run-failed" null null null " (Core ML sidecar generation failed)"
      exit 0
    fi
  fi
  if [[ ! -d "$coreml_sidecar" ]]; then
    emit_json "run-failed" null null null " (Core ML exporter did not produce $coreml_sidecar)"
    exit 0
  fi
fi

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
build_bench_args() {
  BENCH_ARGS=()
  local line=""
  while IFS= read -r line; do
    BENCH_ARGS+=("$(expand_placeholder "$line")")
  done < <(jq -r --arg family "$FAMILY" '.[$family].args[]?' "$FAMILIES_JSON")
}
build_bench_args

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
# The engines log a `using <NAME> backend` line only when verbose/bench flags
# are set (parakeet-cli --verbose, native *-bench binaries always). If the
# marker is absent we return "unknown" rather than defaulting to "CPU" —
# assuming CPU based on the mere presence of /usr/bin/time stderr would
# mis-label every GPU run whose engine happened to log at a lower verbosity.
parse_backend_from_stderr() {
  local file="$1"
  local hit
  hit="$(grep -oE 'using [A-Za-z0-9]+ backend' "$file" 2>/dev/null | head -1 | awk '{print $2}' || true)"
  if [[ -n "$hit" ]]; then
    echo "$hit"; return
  fi
  echo ""
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
  # caller can safely `parse_backend_from_stderr "$stderr_log"` on either
  # success or failure paths.
  : > "$stderr_log"

  if wrap_time "$BINARY" "${BENCH_ARGS[@]}" \
      > "$stderr_log.stdout" 2>> "$stderr_log"; then
    :
  else
    return 1
  fi

  if ! [[ -s "$native_json" ]]; then
    echo "native bench did not emit --json-out file" >&2
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
    coreml_compare="$(spec_field coreml_compare_on_darwin)"
    if [[ "$(uname -s)" == "Darwin" && "$coreml_compare" == "true" ]]; then
      coreml_json="$tmp_dir/native-coreml.json"
      coreml_stderr="$tmp_dir/coreml.err"
      JSON_OUT="$coreml_json"
      build_bench_args
      BENCH_ARGS+=("--require-coreml")
      if ! run_native "$coreml_json" "$coreml_stderr" >/dev/null; then
        tail -40 "$coreml_stderr" >&2 || true
        BACKEND="$(parse_backend_from_stderr "$coreml_stderr")"
        BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (required Core ML benchmark failed; verify the sidecar and runner compatibility)"
        exit 0
      fi

      if [[ "$(jq -r '.encoder_coreml_all_runs // false' "$coreml_json")" != "true" ]] ||
         [[ "$(jq -r '.encoder_backend // ""' "$coreml_json")" != coreml-* ]]; then
        emit_json "run-failed" null null null " (benchmark completed without Core ML on every encoder invocation)"
        exit 0
      fi

      baseline_json="$tmp_dir/native-metal.json"
      baseline_stderr="$tmp_dir/metal.err"
      JSON_OUT="$baseline_json"
      build_bench_args
      if ! PARAKEET_COREML_DISABLE=1 run_native "$baseline_json" "$baseline_stderr" >/dev/null; then
        tail -40 "$baseline_stderr" >&2 || true
        BACKEND="$(jq -r '.backend // "unknown"' "$coreml_json")"
        ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$coreml_json")"
        ENCODER_COREML_ALL_RUNS="true"
        emit_json "run-failed" null null null " (forced-ggml baseline benchmark failed)"
        exit 0
      fi

      if [[ "$(jq -r '.encoder_coreml_all_runs // false' "$baseline_json")" == "true" ]]; then
        emit_json "run-failed" null null null " (PARAKEET_COREML_DISABLE was ignored by the baseline run)"
        exit 0
      fi
      if [[ "$(jq -r '.transcript // ""' "$coreml_json")" != "$(jq -r '.transcript // ""' "$baseline_json")" ]]; then
        emit_json "run-failed" null null null " (Core ML and forced-ggml transcripts differ)"
        exit 0
      fi

      BACKEND="$(jq -r '.backend // "unknown"' "$coreml_json")"
      ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$coreml_json")"
      ENCODER_COREML_ALL_RUNS="true"
      ENCODER_MS_MEDIAN="$(jq -r '.encoder_ms.median // null' "$coreml_json")"
      BASELINE_BACKEND="$(jq -r '.backend // "unknown"' "$baseline_json")"
      BASELINE_ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$baseline_json")"
      BASELINE_ENCODER_MS_MEDIAN="$(jq -r '.encoder_ms.median // null' "$baseline_json")"
      BASELINE_INFERENCE_MS_MEDIAN="$(jq -r '.inference_ms.median // null' "$baseline_json")"
      BASELINE_RTF_MEDIAN="$(jq -r '.rtf_median // null' "$baseline_json")"
      coreml_inference="$(jq -r '.inference_ms.median // null' "$coreml_json")"
      RTF_MEDIAN="$(jq -r '.rtf_median // null' "$coreml_json")"

      if [[ "$ENCODER_MS_MEDIAN" != "null" && "$BASELINE_ENCODER_MS_MEDIAN" != "null" ]]; then
        ENCODER_SPEEDUP="$(jq -n --argjson base "$BASELINE_ENCODER_MS_MEDIAN" --argjson active "$ENCODER_MS_MEDIAN" 'if $active > 0 then $base / $active else null end')"
      fi
      if [[ "$coreml_inference" != "null" && "$BASELINE_INFERENCE_MS_MEDIAN" != "null" ]]; then
        INFERENCE_SPEEDUP="$(jq -n --argjson base "$BASELINE_INFERENCE_MS_MEDIAN" --argjson active "$coreml_inference" 'if $active > 0 then $base / $active else null end')"
      fi

      coreml_rss="$(parse_rss_from_time_stderr "$coreml_stderr")"
      baseline_rss="$(parse_rss_from_time_stderr "$baseline_stderr")"
      if [[ -n "$coreml_rss" && -n "$baseline_rss" ]]; then
        PEAK_RSS_MIB="$(awk -v a="$coreml_rss" -v b="$baseline_rss" 'BEGIN { print (a > b ? a : b) }')"
      elif [[ -n "$coreml_rss" ]]; then
        PEAK_RSS_MIB="$coreml_rss"
      elif [[ -n "$baseline_rss" ]]; then
        PEAK_RSS_MIB="$baseline_rss"
      fi

      artifact_dir="$(dirname "$OUT")"
      mkdir -p "$artifact_dir"
      cp "$coreml_json" "$artifact_dir/parakeet-tdt-coreml-native.json"
      cp "$baseline_json" "$artifact_dir/parakeet-tdt-metal-native.json"

      coreml_min="$(jq -r '.inference_ms.min // null' "$coreml_json")"
      coreml_max="$(jq -r '.inference_ms.max // null' "$coreml_json")"
      emit_json "ok" "$coreml_inference" "$coreml_min" "$coreml_max"
      exit 0
    fi

    stderr_log="$tmp_dir/stderr.log"
    parsed=""
    if ! parsed="$(run_native "$JSON_OUT" "$stderr_log")"; then
      BACKEND="$(parse_backend_from_stderr "$stderr_log")"
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
    # post-fallback active backend); fall back to stderr scraping.
    if [[ -n "$n_backend" ]]; then
      BACKEND="$n_backend"
    else
      BACKEND="$(parse_backend_from_stderr "$stderr_log")"
      BACKEND="${BACKEND:-unknown}"
    fi
    ENCODER_BACKEND="$(jq -r '.encoder_backend // "unknown"' "$JSON_OUT")"
    ENCODER_COREML_ALL_RUNS="$(jq -r '.encoder_coreml_all_runs // null' "$JSON_OUT")"
    ENCODER_MS_MEDIAN="$(jq -r '.encoder_ms.median // null' "$JSON_OUT")"
    rss="$(parse_rss_from_time_stderr "$stderr_log")"
    [[ -n "$rss" ]] && PEAK_RSS_MIB="$rss"
    emit_json "ok" "$n_med" "$n_min" "$n_max"
    ;;

  time-wrapped)
    for i in $(seq 1 "$WARMUP"); do
      echo "warmup $i/$WARMUP" >&2
      wu_stderr="$tmp_dir/warmup-$i.err"
      wu_rss="$tmp_dir/warmup-$i.rss"
      if ! run_one_time_wrapped "$i" "$wu_stderr" "$wu_rss" >/dev/null; then
        BACKEND="$(parse_backend_from_stderr "$wu_stderr")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (warmup failed)"; exit 0
      fi
    done

    declare -a wall_ms=()
    max_rss_seen=""
    combined_stderr="$tmp_dir/combined.err"
    : > "$combined_stderr"
    for i in $(seq 1 "$RUNS"); do
      echo "run $i/$RUNS" >&2
      r_stderr="$tmp_dir/run-$i.err"
      r_rss="$tmp_dir/run-$i.rss"
      if ! ms="$(run_one_time_wrapped "$i" "$r_stderr" "$r_rss")"; then
        cat "$r_stderr" >> "$combined_stderr"
        BACKEND="$(parse_backend_from_stderr "$combined_stderr")"; BACKEND="${BACKEND:-unknown}"
        emit_json "run-failed" null null null " (timed run $i failed)"; exit 0
      fi
      wall_ms+=("$ms")
      cat "$r_stderr" >> "$combined_stderr"
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

    BACKEND="$(parse_backend_from_stderr "$combined_stderr")"
    BACKEND="${BACKEND:-unknown}"
    if [[ -n "$max_rss_seen" ]]; then PEAK_RSS_MIB="$max_rss_seen"; fi

    # Compute RTF only when families.json declared audio_duration_seconds.
    if [[ "$AUDIO_DURATION_S" != "null" && -n "$AUDIO_DURATION_S" ]]; then
      RTF_MEDIAN="$(awk -v m="$med" -v s="$AUDIO_DURATION_S" 'BEGIN { printf "%.3f", m / (s * 1000.0) }')"
    fi

    emit_json "ok" "$med" "$mn" "$mx"
    ;;

  *)
    emit_json "run-failed" null null null " (unknown bench_kind '$BENCH_KIND')"
    exit 1
    ;;
esac
