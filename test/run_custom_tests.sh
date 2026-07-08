#!/usr/bin/env bash
# Build and run the repo's custom driver-API tests (test/test_*.cu) through the
# lupine client shim against a remote server. Mirrors run_cuda_samples.sh's
# server management and env conventions so the GPU integration job can invoke it
# the same way. Each test is a standalone driver-API program that exits non-zero
# on failure; this script fails if any test fails to build or run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT="${SERVER_PORT:-${SERVER_PORT_BASE:-14990}}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-custom-server-$$}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
LUPINE_LIB_DIR="$(cd "$(dirname "$LUPINE_LIB")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"
CUSTOM_TEST_BIN_DIR="${CUSTOM_TEST_BIN_DIR:-}"
if [[ -n "${BUILD_DIR:-}" ]]; then
  owns_build_dir=0
else
  BUILD_DIR="$(mktemp -d)"
  owns_build_dir=1
fi

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done

server_pid=""
cleanup() {
  [[ -n "$server_pid" ]] && ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
    "kill $server_pid 2>/dev/null; rm -f '$SERVER_REMOTE_BIN' /tmp/lupine-custom-server.log" \
    >/dev/null 2>&1 || true
  [[ "$owns_build_dir" == "1" ]] && rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi
server_pid="$(ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
  "LUPINE_PORT=$SERVER_PORT nohup '$SERVER_REMOTE_BIN' >/tmp/lupine-custom-server.log 2>&1 </dev/null & echo \$!")"
sleep 1

pass=0 fail=0
shopt -s nullglob
tests=("$repo_root"/test/test_*.cu)
if [[ ${#tests[@]} -eq 0 ]]; then
  echo "no test/test_*.cu found" >&2
  exit 0
fi

for src in "${tests[@]}"; do
  name="$(basename "$src" .cu)"
  exe="$BUILD_DIR/$name"
  if [[ -n "$CUSTOM_TEST_BIN_DIR" ]]; then
    exe="$CUSTOM_TEST_BIN_DIR/$name"
    if [[ ! -x "$exe" ]]; then
      echo "missing prebuilt custom test: $exe" >&2
      fail=$((fail + 1))
      continue
    fi
    echo "=== using prebuilt $name ==="
  else
    echo "=== building $name ==="
    if ! "$NVCC" --cudart=shared -Wno-deprecated-gpu-targets "$src" -o "$exe" \
         -lcuda -lcublas -L"$CUDA_HOME/lib64/stubs" 2>&1; then
      echo "BUILD FAILED: $name" >&2
      fail=$((fail + 1))
      continue
    fi
  fi
  echo "=== running $name against $SERVER_HOST:$SERVER_PORT ==="
  if timeout --kill-after=5s "$TEST_TIMEOUT" env \
       LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
       LUPINE_SERVER="$SERVER_HOST:$SERVER_PORT" \
       "$exe"; then
    echo "PASS: $name"
    pass=$((pass + 1))
  else
    echo "FAIL: $name (exit $?)" >&2
    fail=$((fail + 1))
  fi
done

echo ""
echo "custom tests: $pass passed, $fail failed"
[[ "$fail" -eq 0 ]]
