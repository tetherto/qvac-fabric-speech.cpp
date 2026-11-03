#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0

fail() { echo "FAIL: $*" >&2; exit 1; }
ok() { PASS=$((PASS + 1)); echo "ok $PASS - $*"; }

check_scripts_parse() {
  local script
  for script in "$HERE"/*.sh; do bash -n "$script" || fail "$script does not parse"; done
  python3 -c 'import ast, sys; [ast.parse(open(path).read(), path) for path in sys.argv[1:]]' "$HERE"/*.py || fail "a python script does not parse"
  if command -v node > /dev/null; then node --check "$HERE/fetch-registry.js" || fail "fetch-registry.js does not parse"; fi
  ok "every script parses"
}

check_spec_fields() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
    | select([.value.bench_kind, .value.binary, .value.cmake_target, .value.task, .value.model_label, .value.weights] | any(. == null))
    | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families without bench_kind/binary/cmake_target/task/model_label/weights: $bad"
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not)
    | select([.value.runs, .value.warmup, .value.cpu_runs, .value.cpu_warmup, .value.timeout_minutes] | any(. == null))
    | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families without runs/warmup/cpu_runs/cpu_warmup/timeout_minutes: $bad"
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | select(.value.runs < 1 or .value.cpu_runs < 1 or .value.timeout_minutes < 1) | .key' "$HERE/families.json")"
  [ -z "$bad" ] || fail "families with zero runs or timeout: $bad"
  ok "every family declares the weekly fields"
}

check_args_are_provisioned() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key as $family
    | ([.value.provision[] | (.registry // .huggingface.name // .fixture // .derived.name) | split("/") | last]) as $files
    | [.value.args[] | select(startswith("${MODEL_DIR}/")) | sub("^\\$\\{MODEL_DIR\\}/"; "")] | map(select(. as $f | $files | index($f) | not))
    | select(length > 0) | "\($family): \(join(","))"' "$HERE/families.json")"
  [ -z "$bad" ] || fail "args reference files no provision entry supplies: $bad"
  ok "every \${MODEL_DIR} file in args is provisioned"
}

check_derived_sources() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key as $family | .value.provision as $p
    | range(0; $p | length) as $i | $p[$i].derived // empty | select(.kind == "requantize") | .source as $source
    | select(([$p[0:$i][] | .registry // "" | split("/") | last] | index($source)) == null) | "\($family): \($source)"' "$HERE/families.json")"
  [ -z "$bad" ] || fail "requantize sources listed after or without their registry file: $bad"
  ok "requantized files follow their source in the provision list"
}

check_huggingface_pins() {
  local bad
  bad="$(jq -r 'to_entries[] | select(.key | startswith("_") | not) | .key as $family
    | .value.provision[] | .huggingface // empty | select((.revision // "" | length) != 40 or (.sha256 // "" | length) != 64) | "\($family): \(.name)"' "$HERE/families.json")"
  [ -z "$bad" ] || fail "HuggingFace files without a 40-char revision and a 64-char sha256: $bad"
  ok "every HuggingFace file pins a revision and a sha256"
}

make_repo() {
  local dir="$1"
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m init
}

setup_home() {
  export BENCH_HOME="$TMP/home" CONFIG_FILE="$TMP/none" KILL_GRACE_SECONDS=1 DEADLINE_POLL_SECONDS=1 LOCK_POLL_SECONDS=1
  mkdir -p "$BENCH_HOME/runs"
  make_repo "$BENCH_HOME/src/speech"
  make_repo "$BENCH_HOME/src/ggml"
  mkdir -p "$BENCH_HOME/src/speech/engines/parakeet/test/samples"
  cp "$REPO/engines/parakeet/test/samples/jfk.wav" "$BENCH_HOME/src/speech/engines/parakeet/test/samples/jfk.wav"
  git -C "$BENCH_HOME/src/speech" add -A
  git -C "$BENCH_HOME/src/speech" -c user.name=test -c user.email=test@example.com commit -q -m fixture
}

check_lib_backends() {
  . "$HERE/lib.sh"
  load_config
  local expected="cpu,cuda,vulkan"
  [ "$OS_NAME" = "Darwin" ] && expected="cpu,metal"
  [ "$(resolve_backends)" = "$expected" ] || fail "auto backends on $OS_NAME: $(resolve_backends)"
  BACKENDS="cpu,cuda"
  [ "$(resolve_backends)" = "cpu,cuda" ] || fail "BACKENDS override ignored"
  BACKENDS="auto"
  ok "backends resolve per OS and from BACKENDS"
}

check_private_tool_dirs() {
  MODELS_DIR="$TMP/shared-models"
  load_config
  [ "$MODELS_DIR" = "$TMP/shared-models" ] || fail "load_config ignored MODELS_DIR"
  [ "$REGISTRY_CLIENT_DIR" = "$BENCH_HOME/registry-client" ] || fail "registry client dir left the bench home: $REGISTRY_CLIENT_DIR"
  [ "$MODELS_VENV" = "$BENCH_HOME/venv" ] || fail "venv left the bench home: $MODELS_VENV"
  case "$REGISTRY_CLIENT_DIR$MODELS_VENV" in *"$MODELS_DIR"*) fail "tool dirs inside MODELS_DIR" ;; esac
  printf 'BACKENDS=cpu\n' > "$TMP/config-backends.env"
  [ "$(BACKENDS=cuda CONFIG_FILE="$TMP/config-backends.env" bash -c '. "$0"; load_config; echo "$BACKENDS"' "$HERE/lib.sh")" = "cuda" ] || fail "config.env overrode an exported BACKENDS"
  MODELS_DIR="$BENCH_HOME/models"
  load_config
  ok "executable tool directories stay under the private bench home and exported BACKENDS wins over config.env"
}

check_lib_iterations() {
  [ "$(family_runs parakeet-tdt cuda)" = 10 ] || fail "parakeet-tdt gpu runs"
  [ "$(family_runs parakeet-tdt cpu)" = 5 ] || fail "parakeet-tdt cpu runs"
  [ "$(family_warmup moss-tts-f16 cpu)" = 0 ] || fail "moss-tts-f16 cpu warmup"
  [ "$(family_field acestep-turbo-q8_0 timeout_minutes)" = 120 ] || fail "acestep timeout"
  ok "iterations come from the backend-specific fields"
}

check_lib_targets() {
  local targets
  targets="$(cmake_targets)"
  for target in whisper-cli parakeet-cli supertonic-bench audio8-cli moss-cli music-cli; do
    case " $targets " in *" $target "*) ;; *) fail "cmake target $target missing from '$targets'" ;; esac
  done
  ok "cmake targets cover every family plus whisper-cli"
}

check_lib_lock() {
  gpu_busy() { return 1; }
  GPU_LOCK="$TMP/lock"
  LOCK_WAIT_HOURS=0
  echo "someone" > "$GPU_LOCK"
  if wait_for_gpu; then fail "wait_for_gpu ignored a held lock"; fi
  if take_gpu_lock; then fail "take_gpu_lock overwrote a held lock"; fi
  [ "$(cat "$GPU_LOCK")" = "someone" ] || fail "held lock content changed"
  if acquire_gpu_lock; then fail "acquire_gpu_lock ignored a held lock"; fi
  rm -f "$GPU_LOCK"
  wait_for_gpu || fail "wait_for_gpu failed with a free GPU"
  acquire_gpu_lock || fail "acquire_gpu_lock failed on a free GPU"
  [ -f "$GPU_LOCK" ] || fail "take_gpu_lock wrote nothing"
  grep -q "speech weekly bench" "$GPU_LOCK" || fail "lock content"
  release_gpu_lock
  [ ! -e "$GPU_LOCK" ] || fail "release_gpu_lock left the file"
  [ -z "$(trap -p EXIT)" ] || fail "release_gpu_lock left the EXIT trap armed"
  mkfifo "$GPU_LOCK"
  [ -z "$(lock_holder)" ] || fail "lock_holder read a FIFO"
  rm -f "$GPU_LOCK"
  [ "$(unset USER LOGNAME; current_user)" = "$(id -un)" ] || fail "current_user without USER and LOGNAME"
  GPU_LOCK="$TMP/no-such-dir/lock"
  if acquire_gpu_lock; then fail "acquire_gpu_lock succeeded on an unwritable lock path"; fi
  GPU_LOCK="$TMP/lock"
  trap 'rm -rf "$TMP"' EXIT
  ok "lock is taken atomically, released once, read only from regular files, and acquisition is bounded"
}

check_lib_deadline() {
  local status=0
  run_with_deadline 0 sleep 30 || status=$?
  [ "$status" = "$DEADLINE_EXIT_CODE" ] || fail "run_with_deadline returned $status for a timed-out command"
  run_with_deadline 1 true || fail "run_with_deadline failed a quick command"
  status=0
  run_with_deadline 1 false || status=$?
  [ "$status" = 1 ] || fail "run_with_deadline hid the exit code ($status)"
  ok "run_with_deadline kills on timeout and passes exit codes through"
}

check_failed_result() {
  write_failed_result moss-tts-f16 cuda timeout "$TMP/failed.json"
  [ "$(jq -r .status "$TMP/failed.json")" = "timeout" ] || fail "failed result status"
  [ "$(jq -r .family "$TMP/failed.json")" = "moss-tts-f16" ] || fail "failed result family"
  ok "write_failed_result emits a driver-shaped row"
}

make_run() {
  mkdir -p "$BENCH_HOME/runs/$1"
  echo '{"run":{}}' > "$BENCH_HOME/runs/$1/summary.json"
  [ "${2:-done}" = "done" ] && touch "$BENCH_HOME/runs/$1/DONE"
  return 0
}

check_previous_and_prune() {
  make_run 2026-01-04
  make_run 2026-01-11
  make_run 2026-01-12-quick
  make_run 2026-01-13-fix1
  make_run 2026-01-18 pending
  [ "$(previous_summary "$BENCH_HOME/runs/2026-01-18")" = "$BENCH_HOME/runs/2026-01-11/summary.json" ] || fail "previous of a pending run skips quick and labelled runs"
  [ "$(previous_summary "$BENCH_HOME/runs/2026-01-11")" = "$BENCH_HOME/runs/2026-01-04/summary.json" ] || fail "previous skips the current run"
  KEEP_RUNS=2
  prune_old_runs
  [ ! -d "$BENCH_HOME/runs/2026-01-04" ] || fail "prune kept the oldest run"
  [ ! -d "$BENCH_HOME/runs/2026-01-12-quick" ] || fail "prune kept an old quick run"
  [ -d "$BENCH_HOME/runs/2026-01-18" ] || fail "prune removed a recent run"
  ok "previous_summary and prune_old_runs walk runs/ newest first"
}

check_build_marker() {
  mkdir -p "$BUILDS_DIR/cpu"
  if build_is_current cpu; then fail "build_is_current without a marker"; fi
  build_signature cpu > "$(build_marker cpu)"
  build_is_current cpu || fail "build_is_current with a matching marker"
  git -C "$ENGINE_SRC" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m bump
  if build_is_current cpu; then fail "build_is_current after the engine moved"; fi
  mkdir -p "$BUILDS_DIR/cuda"
  build_signature cuda > "$(build_marker cuda)"
  build_is_current cuda || fail "build_is_current for cuda with a fresh marker"
  CUDA_ARCH=89-real
  if build_is_current cuda; then fail "build_is_current after CUDA_ARCH changed"; fi
  CUDA_ARCH=120a-real
  mkdir -p "$TMP/rebuild/results/cuda"
  touch "$TMP/rebuild/results/cuda/old.json"
  discard_backend_results "$TMP/rebuild" cuda
  [ ! -e "$TMP/rebuild/results/cuda/old.json" ] || fail "discard_backend_results kept old rows"
  ok "build markers track the commits and the build options, and a rebuild discards that backend's rows"
}

check_fetch_models_offline() {
  cat > "$TMP/families-test.json" <<'SPEC'
{
  "clone-test": {
    "cmake_target": "x", "bench_kind": "time-wrapped", "binary": "x", "models": [], "args": [],
    "task": "TTS", "model_label": "x", "weights": "x", "runs": 1, "warmup": 0, "cpu_runs": 1, "cpu_warmup": 0, "timeout_minutes": 1,
    "provision": [
      {"fixture": "engines/parakeet/test/samples/jfk.wav"},
      {"derived": {"name": "jfk-24k.wav", "kind": "resample", "source": "engines/parakeet/test/samples/jfk.wav", "rate": 24000}}
    ]
  }
}
SPEC
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/fetch-models.sh" clone-test > "$TMP/fetch.log" 2>&1 || fail "fetch-models.sh failed: $(cat "$TMP/fetch.log")"
  [ -L "$BENCH_HOME/models/families/clone-test/jfk.wav" ] || fail "fixture not linked"
  [ -L "$BENCH_HOME/models/families/clone-test/jfk-24k.wav" ] || fail "derived file not linked"
  [ -s "$BENCH_HOME/models/store/jfk.wav.sha256" ] || fail "sha256 sidecar"
  grep -q "^fixture engines/parakeet" "$BENCH_HOME/models/store/jfk.wav.source" || fail "source sidecar"
  grep -q "^derived resample 24000" "$BENCH_HOME/models/store/jfk-24k.wav.source" || fail "derived source sidecar"
  ok "fetch-models.sh provisions fixtures and derived files offline"
}

check_fetch_models_reprovisions_on_origin_change() {
  local before after
  before="$(cat "$BENCH_HOME/models/store/jfk.wav.source")"
  git -C "$ENGINE_SRC" -c user.name=test -c user.email=test@example.com commit -q --allow-empty -m moved
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/fetch-models.sh" clone-test > "$TMP/fetch2.log" 2>&1 || fail "second fetch-models.sh failed"
  after="$(cat "$BENCH_HOME/models/store/jfk.wav.source")"
  [ "$before" != "$after" ] || fail "fixture origin not refreshed after the engine moved"
  printf 'stale' > "$BENCH_HOME/models/store/jfk-24k.wav"
  rm -f "$BENCH_HOME/models/store/jfk-24k.wav.source"
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/fetch-models.sh" clone-test > "$TMP/fetch3.log" 2>&1 || fail "third fetch-models.sh failed"
  [ "$(wc -c < "$BENCH_HOME/models/store/jfk-24k.wav" | tr -d ' ')" -gt 1000 ] || fail "a file without an origin sidecar was kept"
  ok "files are re-provisioned when their recorded origin differs or is missing"
}

check_weekly_helpers() {
  (
    . "$HERE/weekly.sh"
    RUN_DIR="$TMP/run-helpers"
    mkdir -p "$RUN_DIR/results/cpu"
    touch "$RUN_DIR/DONE" "$RUN_DIR/build-failures.txt" "$RUN_DIR/narrative.md" "$RUN_DIR/summary.json"
    echo '{"status":"ok"}' > "$RUN_DIR/results/cpu/good.json"
    echo '{"status":"timeout"}' > "$RUN_DIR/results/cpu/bad.json"
    touch "$RUN_DIR/results/cpu/bad.log"
    clear_run_outputs
    [ ! -e "$RUN_DIR/build-failures.txt" ] && [ ! -e "$RUN_DIR/narrative.md" ] && [ ! -e "$RUN_DIR/summary.json" ] || { echo "clear_run_outputs kept stale outputs" >&2; exit 1; }
    [ -e "$RUN_DIR/results/cpu/good.json" ] || { echo "clear_run_outputs dropped an ok row" >&2; exit 1; }
    [ ! -e "$RUN_DIR/results/cpu/bad.json" ] && [ ! -e "$RUN_DIR/results/cpu/bad.log" ] || { echo "clear_run_outputs kept a failed row" >&2; exit 1; }
    [ -z "$(existing_attachments)" ] || { echo "existing_attachments listed missing files" >&2; exit 1; }
    touch "$RUN_DIR/report.md"
    [ "$(existing_attachments | tr '\n' ' ')" = "--attach $RUN_DIR/report.md " ] || { echo "existing_attachments: $(existing_attachments)" >&2; exit 1; }
  ) || fail "weekly.sh helpers"
  ok "weekly.sh clears stale outputs, drops failed rows for a rerun and attaches only existing files"
}

check_info_scripts() {
  bash "$HERE/machine-info.sh" > "$TMP/machine.json" || fail "machine-info.sh failed"
  [ "$(jq -r .hostname "$TMP/machine.json")" = "$(hostname -s)" ] || fail "machine.json hostname"
  jq -e '.tools.cmake != null and .gpu != null' "$TMP/machine.json" > /dev/null || fail "machine.json shape"
  FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/collect-pins.sh" clone-test > "$TMP/pins.json" || fail "collect-pins.sh failed"
  [ "$(jq -r .engine.sha "$TMP/pins.json")" = "$(git -C "$ENGINE_SRC" rev-parse HEAD)" ] || fail "pins engine sha"
  [ "$(jq -r '.models[] | select(.name == "jfk-24k.wav") | .present' "$TMP/pins.json")" = "true" ] || fail "pins model entry"
  ok "machine-info.sh and collect-pins.sh emit the expected JSON"
}

check_nvcc_release() {
  mkdir -p "$TMP/cuda/bin"
  printf '#!/bin/sh\nprintf "nvcc: NVIDIA (R) Cuda compiler driver\\nCuda compilation tools, release 13.3, V13.3.48\\n"\n' > "$TMP/cuda/bin/nvcc"
  chmod +x "$TMP/cuda/bin/nvcc"
  CUDA_HOME="$TMP/cuda" bash "$HERE/machine-info.sh" > "$TMP/machine-cuda.json" || fail "machine-info.sh with a stub nvcc"
  [ "$(jq -r .tools.nvcc "$TMP/machine-cuda.json")" = "13.3" ] || fail "nvcc release parse: $(jq -r .tools.nvcc "$TMP/machine-cuda.json")"
  BACKENDS=cpu FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/collect-pins.sh" clone-test > "$TMP/pins-cpu.json" || fail "collect-pins.sh without cuda"
  [ "$(jq -r .build.cuda_arch "$TMP/pins-cpu.json")" = "" ] || fail "cuda_arch recorded without a cuda backend"
  BACKENDS=cpu,cuda FAMILIES_JSON="$TMP/families-test.json" bash "$HERE/collect-pins.sh" clone-test > "$TMP/pins-cuda.json" || fail "collect-pins.sh with cuda"
  [ "$(jq -r .build.cuda_arch "$TMP/pins-cuda.json")" = "120a-real" ] || fail "cuda_arch missing with a cuda backend"
  ok "nvcc release and CUDA arch pins follow the toolchain and the backends"
}

check_input_validation() {
  if bash "$HERE/weekly.sh" --label 'x /' > /dev/null 2>&1; then fail "weekly.sh accepted an unsafe label"; fi
  if BENCH_HOME="$TMP/install" bash "$HERE/install.sh" --no-cron --hour 99 > /dev/null 2>&1; then fail "install.sh accepted hour 99"; fi
  if BENCH_HOME="$TMP/bad home" bash "$HERE/install.sh" --no-cron > /dev/null 2>&1; then fail "install.sh accepted a bench home with a space"; fi
  printf 'secret\n' > "$TMP/secret-open"
  chmod 644 "$TMP/secret-open"
  printf 'secret\n' > "$TMP/secret-private"
  chmod 600 "$TMP/secret-private"
  [ -z "$(read_secret "$TMP/secret-open" 2>/dev/null)" ] || fail "read_secret returned a world-readable secret"
  [ "$(read_secret "$TMP/secret-private" 2>/dev/null)" = "secret" ] || fail "read_secret rejected a private secret"
  [ "$(public_repo_url "https://x-access-token:abc@github.com/o/r.git")" = "https://github.com/o/r.git" ] || fail "public_repo_url kept the credential"
  [ "$(public_repo_url "git@github.com:o/r.git")" = "git@github.com:o/r.git" ] || fail "public_repo_url changed an ssh url"
  ok "unsafe labels, schedules, bench homes, open secret files and repo credentials are rejected or stripped"
}

check_entry_points() {
  bash "$HERE/weekly.sh" --help > /dev/null 2>&1 || fail "weekly.sh --help"
  bash "$HERE/install.sh" --help > /dev/null 2>&1 || fail "install.sh --help"
  BENCH_HOME="$TMP/install" bash "$HERE/install.sh" --no-cron > /dev/null || fail "install.sh --no-cron"
  [ -x "$TMP/install/tools/weekly.sh" ] || fail "install.sh did not copy the tools"
  [ -f "$TMP/install/config.env" ] || fail "install.sh did not seed config.env"
  [ "$(ls -l "$TMP/install/config.env" | cut -c5-10)" = "------" ] || fail "config.env is not private"
  ok "entry points answer --help and install.sh deploys without cron"
}

check_python_tests() {
  local test
  for test in "$HERE"/test_*.py; do python3 -B "$test" > "$TMP/pytest.log" 2>&1 || fail "$(basename "$test"): $(tail -20 "$TMP/pytest.log")"; done
  ok "python unit tests pass"
}

main() {
  check_scripts_parse
  check_spec_fields
  check_args_are_provisioned
  check_derived_sources
  check_huggingface_pins
  setup_home
  check_lib_backends
  check_private_tool_dirs
  check_lib_iterations
  check_lib_targets
  check_lib_lock
  check_lib_deadline
  check_failed_result
  check_previous_and_prune
  check_build_marker
  check_fetch_models_offline
  check_fetch_models_reprovisions_on_origin_change
  check_weekly_helpers
  check_info_scripts
  check_nvcc_release
  check_input_validation
  check_entry_points
  check_python_tests
  echo "all $PASS checks passed"
}

main "$@"
