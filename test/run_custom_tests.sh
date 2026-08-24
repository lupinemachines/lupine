#!/usr/bin/env bash
# Build and run the repo's custom driver-API tests (test/test_*.cu) through the
# lupine client shim against a remote server. Mirrors run_cuda_samples.sh's
# server management and env conventions so the GPU integration job can invoke it
# the same way. Each test is a standalone driver-API program that exits non-zero
# on failure; this script fails if any test fails to build or run.
#
# Tests build and run TEST_JOBS at a time. Each test gets its own server
# instance on SERVER_PORT_BASE + index so a test that wedges or crashes its
# server cannot poison the others.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-${SERVER_PORT:-14990}}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-45}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-custom-server-$$}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LD_LIBRARY_PATH="${SERVER_LD_LIBRARY_PATH:-}"
LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
LUPINE_LIB_DIR="$(cd "$(dirname "$LUPINE_LIB")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"
CUDA_SAMPLES_ARCH="${CUDA_SAMPLES_ARCH:-}"
TEST_JOBS="${TEST_JOBS:-$(nproc)}"
if [[ -n "${BUILD_DIR:-}" ]]; then
  owns_build_dir=0
else
  BUILD_DIR="$(mktemp -d)"
  owns_build_dir=1
fi

case "$TEST_JOBS" in
  ''|*[!0-9]*|0)
    echo "TEST_JOBS must be a positive integer: $TEST_JOBS" >&2
    exit 1
    ;;
esac

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done

ssh_with_timeout() {
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "$@"
}

# Pidfiles and server logs share a $$-scoped prefix so an aborted run can be
# swept remotely with one ssh command.
remote_prefix="/tmp/lupine-custom-$$"

cleanup() {
  ssh_with_timeout "
    for f in $remote_prefix-*.pid; do
      [ -f \"\$f\" ] || continue
      pid=\$(cat \"\$f\" 2>/dev/null || true)
      [ -n \"\$pid\" ] && kill \"\$pid\" >/dev/null 2>&1
      rm -f \"\$f\"
    done
    rm -f $remote_prefix-*.log '$SERVER_REMOTE_BIN'
  " >/dev/null 2>&1 || true
  [[ "$owns_build_dir" == "1" ]] && rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

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

start_remote_server() {
  local pidfile="$1"
  local server_log="$2"
  local port="$3"
  local attempt
  local server_environment="LUPINE_PORT=$port"

  if [[ -n "$SERVER_LD_LIBRARY_PATH" ]]; then
    printf -v server_environment 'LD_LIBRARY_PATH=%q %s' \
      "$SERVER_LD_LIBRARY_PATH" "$server_environment"
  fi

  for attempt in 1 2 3; do
    stop_remote_server "$pidfile" "$server_log"
    if ssh_with_timeout "
      rm -f '$server_log' '$pidfile'
      $server_environment nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null &
      echo \$! >'$pidfile'
      sleep 0.5
      test -s '$pidfile'
    "; then
      return 0
    fi
    sleep "$attempt"
  done

  return 1
}

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi

shopt -s nullglob
tests=("$repo_root"/test/test_*.cu)
if [[ ${#tests[@]} -eq 0 ]]; then
  echo "no test/test_*.cu found" >&2
  exit 0
fi

run_one_test() {
  local i="$1"
  local src="${tests[$i]}"
  local name
  name="$(basename "$src" .cu)"
  local exe="$BUILD_DIR/$name"
  local log="$BUILD_DIR/$name.log"
  local result_file="$BUILD_DIR/.result-$i"
  local port=$((SERVER_PORT_BASE + i))
  local pidfile="$remote_prefix-$port.pid"
  local server_log="$remote_prefix-$port.log"
  local start_seconds="$SECONDS"
  local rc=0

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] custom test $((i + 1))/${#tests[@]}: $name" >&2

  # nvcc targets one architecture by default (sm_75 on 13.3) and leaves every
  # other device to a PTX JIT, which the test host's driver refuses when it is
  # older than the toolkit -- the launch then silently does nothing and the
  # test reports a stale value. Emit a cubin for every architecture nvcc knows
  # unless CUDA_SAMPLES_ARCH pins one.
  local arch_arg="-arch=all"
  if [[ -n "$CUDA_SAMPLES_ARCH" ]]; then
    arch_arg="-arch=sm_$CUDA_SAMPLES_ARCH"
  fi

  if ! "$NVCC" --cudart=shared -Wno-deprecated-gpu-targets "$arch_arg" \
       "$src" -o "$exe" \
       -lcuda -lcublas -L"$CUDA_HOME/lib64/stubs" >"$log" 2>&1; then
    echo "FAIL:build" > "$result_file"
    return 0
  fi

  if ! start_remote_server "$pidfile" "$server_log" "$port"; then
    echo "FAIL:server-start" > "$result_file"
    return 0
  fi

  set +e
  timeout --kill-after=5s "$TEST_TIMEOUT" env \
    LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
    LUPINE_SERVER="$SERVER_HOST:$port" \
    "$exe" >>"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  if [[ "$rc" == "0" ]]; then
    echo "PASS" > "$result_file"
  else
    echo "FAIL:$rc" > "$result_file"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] custom test $name -> $(cat "$result_file") in $((SECONDS - start_seconds))s" >&2
}

# Keep TEST_JOBS tests in flight; launch the next one as soon as any finishes.
active=0
for i in "${!tests[@]}"; do
  run_one_test "$i" &
  active=$((active + 1))
  if (( active >= TEST_JOBS )); then
    wait -n || true
    active=$((active - 1))
  fi
done
while (( active > 0 )); do
  wait -n || true
  active=$((active - 1))
done

pass=0 fail=0
for i in "${!tests[@]}"; do
  name="$(basename "${tests[$i]}" .cu)"
  result_file="$BUILD_DIR/.result-$i"
  if [[ -s "$result_file" ]]; then
    status="$(cat "$result_file")"
  else
    status="FAIL:internal"
  fi

  echo "=== $name: $status ==="
  [[ -f "$BUILD_DIR/$name.log" ]] && cat "$BUILD_DIR/$name.log"

  if [[ "$status" == "PASS" ]]; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
  fi
done

echo ""
echo "custom tests: $pass passed, $fail failed"
[[ "$fail" -eq 0 ]]
