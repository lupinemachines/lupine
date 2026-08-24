#!/usr/bin/env bash
# Run llama.cpp's test-backend-ops (every ggml op on the CUDA backend, checked
# against the CPU backend) through the lupine client shim against a remote
# server. ggml-cuda is a real consumer of the VMM pool, CUDA graphs, cuBLAS,
# and async copies, which is why it is worth more than another sample.
#
# Each op is a unit with its own server on SERVER_PORT_BASE + index. Ops
# listed in test/llama.cpp/known_failures.txt are marked SKIP:known.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_CPP_URL="${LLAMA_CPP_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b10589}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$repo_root/test/llama.cpp/llama.cpp}"
LLAMA_CPP_BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-$LLAMA_CPP_DIR/build}"
LLAMA_CPP_BIN="${LLAMA_CPP_BIN:-}"
LLAMA_CPP_ARCH="${LLAMA_CPP_ARCH:-${CUDA_SAMPLES_ARCH:-89}}"
LLAMA_CPP_CMAKE_ARGS="${LLAMA_CPP_CMAKE_ARGS:-}"
LLAMA_CPP_SKIP_LIST="${LLAMA_CPP_SKIP_LIST:-}"
LLAMA_CPP_KNOWN_FAILURES="${LLAMA_CPP_KNOWN_FAILURES:-$repo_root/test/llama.cpp/known_failures.txt}"
BUILD_LLAMA="${BUILD_LLAMA:-auto}"
BUILD_ONLY="${BUILD_ONLY:-0}"
JOBS="${JOBS:-$(nproc)}"
SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-20500}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-45}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-llama-${USER:-lupine}-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"
SERVER_LD_LIBRARY_PATH="${SERVER_LD_LIBRARY_PATH:-}"

LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
TEST_TIMEOUT="${TEST_TIMEOUT:-600}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/llama.cpp/results/$(date +%Y%m%d-%H%M%S)}"

usage() {
  cat <<EOF
Usage: $0 [op ...]
Ops are ggml op names as printed by test-backend-ops --list-ops (ADD, MUL_MAT).
Default: every op.
Environment:
  LLAMA_CPP_DIR        Clone path. Default: $LLAMA_CPP_DIR
  LLAMA_CPP_REF        Tag/commit to check out. Default: $LLAMA_CPP_REF
  LLAMA_CPP_BUILD_DIR  CMake build path. Default: <LLAMA_CPP_DIR>/build
  LLAMA_CPP_BIN        Prebuilt test-backend-ops; skips clone and build.
  LLAMA_CPP_ARCH       CMAKE_CUDA_ARCHITECTURES. Default: $LLAMA_CPP_ARCH
  LLAMA_CPP_CMAKE_ARGS Extra CMake configure args.
  BUILD_LLAMA          auto, 1, or 0. Default: auto (build when the binary is missing).
  BUILD_ONLY           1 to clone/build and exit before running.
  LLAMA_CPP_SKIP_LIST  Comma or space separated ops to mark SKIP:disabled.
  LLAMA_CPP_KNOWN_FAILURES
                       Ops to mark SKIP:known. Default: $LLAMA_CPP_KNOWN_FAILURES
  SERVER_SSH_TARGET    GPU host. Default: $SERVER_SSH_TARGET
  SERVER_PORT_BASE     First per-op server port. Default: $SERVER_PORT_BASE
  TEST_TIMEOUT         Per-op timeout in seconds. Default: $TEST_TIMEOUT
  RESULTS_DIR          Output directory. Default: test/llama.cpp/results/<timestamp>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$LLAMA_CPP_BIN" ]]; then
  LLAMA_CPP_BIN="$LLAMA_CPP_BUILD_DIR/bin/test-backend-ops"
  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    rm -rf "$LLAMA_CPP_DIR"
    git clone --quiet --depth 1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_URL" "$LLAMA_CPP_DIR"
  fi
  git config --global --add safe.directory "$LLAMA_CPP_DIR"
  if [[ "$BUILD_LLAMA" == "1" || ( "$BUILD_LLAMA" == "auto" && ! -x "$LLAMA_CPP_BIN" ) ]]; then
    # Only the test binary and the backends it needs. FlashAttention template
    # instances are the bulk of ggml-cuda's compile time and run server-side,
    # so they are left out; test-backend-ops reports those cases unsupported.
    # libggml-cuda.so is linked against the libcuda stub, so the executable
    # link must not insist on resolving driver symbols (same as upstream's
    # cuda.Dockerfile).
    # CUDA 11 toolkits use the v1 cuGraphExecUpdate form, which fails through
    # the shim (#601); graphs stay off there so the ops still get covered.
    graphs=ON
    if [[ "$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n1)" == "11" ]]; then
      graphs=OFF
    fi
    # shellcheck disable=SC2086
    cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CUDA_FA=OFF -DGGML_CUDA_GRAPHS="$graphs" \
      -DCMAKE_CUDA_ARCHITECTURES="$LLAMA_CPP_ARCH" \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      $LLAMA_CPP_CMAKE_ARGS
    cmake --build "$LLAMA_CPP_BUILD_DIR" --parallel "$JOBS" --target test-backend-ops
  fi
fi
if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi

LUPINE_LIB_DIR="$(cd "$(dirname "$LUPINE_LIB")" && pwd)"

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done
if [[ ! -x "$LLAMA_CPP_BIN" ]]; then
  # The image build already reported why; keep the lane green but visible.
  mkdir -p "$RESULTS_DIR"
  echo "BUILD-FAILED test-backend-ops missing at $LLAMA_CPP_BIN" | tee "$RESULTS_DIR/summary.txt"
  exit 0
fi

OPS=()
if [[ $# -gt 0 ]]; then
  OPS=("$@")
else
  # libggml-cuda.so needs libcuda.so.1 at load time even to list ops.
  while IFS= read -r op; do
    OPS+=("$op")
  done < <(LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" "$LLAMA_CPP_BIN" --list-ops \
    | sed -nE 's/^[[:space:]]*([A-Z0-9_]+)[[:space:]]*$/\1/p' | sort -u)
fi

known_failures=()
if [[ -f "$LLAMA_CPP_KNOWN_FAILURES" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -n "$line" ]] && known_failures+=("$line")
  done < "$LLAMA_CPP_KNOWN_FAILURES"
fi

ssh_with_timeout() {
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "$@"
}

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi

cleanup_remote_bin() {
  if [[ "$SERVER_UPLOAD" == "1" && "$SERVER_REMOTE_CLEANUP" == "1" ]]; then
    ssh_with_timeout "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_bin EXIT

stop_remote_server() {
  local pidfile="$1"
  local server_log="$2"

  ssh_with_timeout "
    if [ -f '$pidfile' ]; then
      pid=\$(cat '$pidfile' 2>/dev/null || true)
      if [ -n \"\$pid\" ]; then
        kill \"\$pid\" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
          kill -0 \"\$pid\" >/dev/null 2>&1 || break
          sleep 0.1
        done
        kill -9 \"\$pid\" >/dev/null 2>&1 || true
      fi
    fi
    rm -f '$pidfile' '$server_log'
  " >/dev/null 2>&1 || true
}

op_in_list() {
  local op="$1"
  shift
  local item=""
  for item in "$@"; do
    [[ "$item" == "$op" ]] && return 0
  done
  return 1
}

tsv="$RESULTS_DIR/results.tsv"
mkdir -p "$RESULTS_DIR"
: > "$tsv"
pass=0
fail=0
skip=0

echo "llama.cpp $LLAMA_CPP_REF test-backend-ops, ${#OPS[@]} ops" >&2

for i in "${!OPS[@]}"; do
  op="${OPS[$i]}"
  port=$((SERVER_PORT_BASE + i))
  log="$RESULTS_DIR/$op.log"
  server_log="/tmp/lupine-llama-$port.log"
  pidfile="/tmp/lupine-llama-$port.pid"
  op_start_seconds="$SECONDS"
  server_environment="LUPINE_PORT=$port"
  if [[ -n "$SERVER_LD_LIBRARY_PATH" ]]; then
    printf -v server_environment 'LD_LIBRARY_PATH=%q %s' \
      "$SERVER_LD_LIBRARY_PATH" "$server_environment"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] llama.cpp op $((i + 1))/${#OPS[@]}: $op" >&2

  # shellcheck disable=SC2206
  disabled=(${LLAMA_CPP_SKIP_LIST//,/ })
  if op_in_list "$op" "${disabled[@]}"; then
    skip=$((skip + 1))
    printf '%s\t%s\t%s\n' "$op" "SKIP:disabled" "disabled by LLAMA_CPP_SKIP_LIST" | tee -a "$tsv"
    continue
  fi
  if op_in_list "$op" "${known_failures[@]}"; then
    skip=$((skip + 1))
    printf '%s\t%s\t%s\n' "$op" "SKIP:known" "listed in $(basename "$LLAMA_CPP_KNOWN_FAILURES")" | tee -a "$tsv"
    continue
  fi

  stop_remote_server "$pidfile" "$server_log"
  ssh_with_timeout \
    "rm -f '$server_log' '$pidfile'; $server_environment nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null & echo \$! >'$pidfile'; sleep 0.25"

  set +e
  timeout --kill-after=5s "$TEST_TIMEOUT" env \
    LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
    LUPINE_SERVER="$SERVER_HOST:$port" \
    LD_PRELOAD="$LUPINE_LIB" \
    "$LLAMA_CPP_BIN" test -b CUDA0 -o "$op" >"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  # Per-case lines end in OK / FAIL / "not supported [CUDA0]"; the summary is
  # "N/M tests passed" followed by "Backend CUDA0: OK|FAIL".
  plain="$(sed -E 's/\x1b\[[0-9;]*m//g' "$log")"
  counts="$(printf '%s\n' "$plain" | sed -nE 's/^[[:space:]]*([0-9]+\/[0-9]+) tests passed.*/\1/p' | tail -n1)"
  unsupported="$(printf '%s\n' "$plain" | grep -c 'not supported' || true)"
  failed_cases="$(printf '%s\n' "$plain" | { grep -E ': FAIL$' || true; } | sed -E 's/^[[:space:]]*//; s/: FAIL$//' | head -n3 | tr '\n' ' ')"

  if [[ "$rc" == "0" ]]; then
    status="PASS"
    pass=$((pass + 1))
  else
    status="FAIL:$rc"
    fail=$((fail + 1))
  fi

  signature="${counts:+$counts passed }${unsupported:+$unsupported unsupported }$failed_cases"
  if [[ "$rc" != "0" && -z "$failed_cases" ]]; then
    signature="$signature$(printf '%s\n' "$plain" | { grep -E 'LUPINE|error|Error|Aborted|Segmentation' || true; } | head -n2 | tr '\n' ' ')"
  fi
  signature="$(printf '%s' "$signature" | sed -E 's/[[:space:]]+/ /g; s/ $//' | cut -c1-240)"
  if [[ -z "$signature" && "$rc" == "124" ]]; then
    signature="timed out after ${TEST_TIMEOUT}s"
  fi
  printf '%s\t%s\t%s\n' "$op" "$status" "$signature" | tee -a "$tsv"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] llama.cpp op $op -> $status in $((SECONDS - op_start_seconds))s" >&2
done

{
  echo "PASS $pass"
  echo "FAIL $fail"
  echo "SKIP $skip"
  echo "TOTAL $((pass + fail + skip))"
  echo "RESULTS $tsv"
} | tee "$RESULTS_DIR/summary.txt"

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
