#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-20100}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-45}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-pytorch-${USER:-lupine}-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"
PYTORCH_SKIP_LIST="${PYTORCH_SKIP_LIST:-}"

LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
PYTHON_BIN="${PYTHON_BIN:-$repo_root/.venv-pytorch312/bin/python}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
TEST_TIMEOUT="${TEST_TIMEOUT:-90}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/pytorch/results/$(date +%Y%m%d-%H%M%S)}"

TESTS=(
  discover
  tensor_ops
  matmul
  fft
  cudnn_conv
  sparse_mm
  linalg_solve
  autograd_step
  compile_elementwise
  microgpt_train
)

if [[ $# -gt 0 ]]; then
  TESTS=("$@")
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing python: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$LUPINE_LIB" ]]; then
  echo "missing shim: $LUPINE_LIB" >&2
  exit 1
fi
if [[ ! -x "$SERVER_LOCAL_BIN" ]]; then
  echo "missing server binary: $SERVER_LOCAL_BIN" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

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

test_disabled() {
  local test_name="$1"
  local disabled=""

  # shellcheck disable=SC2206
  local disabled_tests=(${PYTORCH_SKIP_LIST//,/ })
  for disabled in "${disabled_tests[@]}"; do
    if [[ "$disabled" == "$test_name" ]]; then
      return 0
    fi
  done

  return 1
}

tsv="$RESULTS_DIR/results.tsv"
: > "$tsv"
pass=0
fail=0
skip=0

for i in "${!TESTS[@]}"; do
  test_name="${TESTS[$i]}"
  port=$((SERVER_PORT_BASE + i))
  log="$RESULTS_DIR/$test_name.log"
  server_log="/tmp/lupine-pytorch-$port.log"
  pidfile="/tmp/lupine-pytorch-$port.pid"
  test_start_seconds="$SECONDS"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PyTorch test $((i + 1))/${#TESTS[@]}: $test_name" >&2

  if test_disabled "$test_name"; then
    status="SKIP:disabled"
    skip=$((skip + 1))
    signature="disabled by PYTORCH_SKIP_LIST"
    printf '%s\t%s\t%s\n' "$test_name" "$status" "$signature" | tee -a "$tsv"
    continue
  fi

  stop_remote_server "$pidfile" "$server_log"

  ssh_with_timeout \
    "rm -f '$server_log' '$pidfile'; LUPINE_PORT=$port nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null & echo \$! >'$pidfile'; sleep 0.25"

  set +e
  timeout --kill-after=5s "$TEST_TIMEOUT" env \
    LD_LIBRARY_PATH="$repo_root/build:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
    LUPINE_SERVER="$SERVER_HOST:$port" \
    LD_PRELOAD="$LUPINE_LIB" \
    "$PYTHON_BIN" "$repo_root/test/pytorch_lupine_tests.py" "$test_name" \
    >"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  if [[ "$rc" == "0" ]]; then
    status="PASS"
    pass=$((pass + 1))
  else
    status="FAIL:$rc"
    fail=$((fail + 1))
  fi

  signature="$(tr '\n' ' ' < "$log" | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
  if [[ -z "$signature" && "$rc" == "124" ]]; then
    signature="timed out after ${TEST_TIMEOUT}s"
  fi
  printf '%s\t%s\t%s\n' "$test_name" "$status" "$signature" | tee -a "$tsv"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PyTorch test $test_name -> $status in $((SECONDS - test_start_seconds))s" >&2
done

{
  echo "PASS $pass"
  echo "FAIL $fail"
  echo "SKIP $skip"
  echo "TOTAL $((pass + fail + skip))"
  echo "RESULTS $RESULTS_DIR/results.tsv"
} | tee "$RESULTS_DIR/summary.txt"

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
