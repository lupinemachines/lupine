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
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-pytorch-${USER:-lupine}-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"

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

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi

cleanup_remote_bin() {
  if [[ "$SERVER_UPLOAD" == "1" && "$SERVER_REMOTE_CLEANUP" == "1" ]]; then
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_bin EXIT

tsv="$RESULTS_DIR/results.tsv"
: > "$tsv"
pass=0
fail=0

for i in "${!TESTS[@]}"; do
  test_name="${TESTS[$i]}"
  port=$((SERVER_PORT_BASE + i))
  log="$RESULTS_DIR/$test_name.log"
  server_log="/tmp/lupine-pytorch-$port.log"
  pidfile="/tmp/lupine-pytorch-$port.pid"

  ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
    "if [ -f '$pidfile' ]; then kill \$(cat '$pidfile') >/dev/null 2>&1 || true; fi; pkill -f -- '$SERVER_REMOTE_BIN' >/dev/null 2>&1 || true; rm -f '$server_log' '$pidfile'" \
    >/dev/null 2>&1 || true

  ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
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

  ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
    "if [ -f '$pidfile' ]; then kill \$(cat '$pidfile') >/dev/null 2>&1 || true; fi; pkill -f -- '$SERVER_REMOTE_BIN' >/dev/null 2>&1 || true; rm -f '$pidfile'" \
    >/dev/null 2>&1 || true

  if [[ "$rc" == "0" ]]; then
    status="PASS"
    pass=$((pass + 1))
  else
    status="FAIL:$rc"
    fail=$((fail + 1))
  fi

  signature="$(tr '\n' ' ' < "$log" | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
  printf '%s\t%s\t%s\n' "$test_name" "$status" "$signature" | tee -a "$tsv"
done

{
  echo "PASS $pass"
  echo "FAIL $fail"
  echo "TOTAL $((pass + fail))"
  echo "RESULTS $RESULTS_DIR/results.tsv"
} | tee "$RESULTS_DIR/summary.txt"
