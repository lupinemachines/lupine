#!/usr/bin/env bash
# Build and run one custom driver-API test through the lupine client shim.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${LIST_TESTS:-0}" == "1" ]]; then
  printf '%s\n' "$repo_root"/test/test_*.cu | xargs -n1 basename -s .cu
  exit 0
fi

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
CUDA_SAMPLES_ARCH="${CUDA_SAMPLES_ARCH:-}"
if [[ -n "${BUILD_DIR:-}" ]]; then
  owns_build_dir=0
else
  BUILD_DIR="$(mktemp -d)"
  owns_build_dir=1
fi

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done

ssh_with_timeout() {
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "$@"
}

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

name="${1:?custom test name required}"
name="${name%.cu}"
src="$repo_root/test/$name.cu"
exe="$BUILD_DIR/$name"
port="$SERVER_PORT_BASE"
pidfile="/tmp/lupine-custom-$port.pid"
server_log="/tmp/lupine-custom-$port.log"

cleanup() {
  stop_remote_server "$pidfile" "$server_log"
  [[ "$SERVER_UPLOAD" == "1" ]] && ssh_with_timeout "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  [[ "$owns_build_dir" == "1" ]] && rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

arch_arg="-arch=all"
[[ -n "$CUDA_SAMPLES_ARCH" ]] && arch_arg="-arch=sm_$CUDA_SAMPLES_ARCH"
"$NVCC" --cudart=shared -Wno-deprecated-gpu-targets "$arch_arg" \
  "$src" -o "$exe" -lcuda -lcublas -L"$CUDA_HOME/lib64/stubs"

start_remote_server "$pidfile" "$server_log" "$port"
env LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
  LUPINE_SERVER="$SERVER_HOST:$port" "$exe"
