#!/usr/bin/env bash
# Builds test/test_hip_runtime.cpp with the host compiler and runs it through
# the HIP shim against a local lupine_driver_server on this machine's AMD GPU.
# With --native it links the ROCm runtime directly instead, to compare.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$repo_root/build}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-14990}"

native=0
[[ "${1:-}" == "--native" ]] && native=1

source_file="${HIP_TEST_SOURCE:-$repo_root/test/test_hip_runtime.cpp}"
exe="$BUILD_DIR/$(basename "${source_file%.*}")"
lib_dir="$BUILD_DIR"
[[ "$native" == "1" ]] && lib_dir="$ROCM_PATH/lib"
compiler=(g++)
if [[ "$source_file" == *.hip ]]; then
  compiler=("$ROCM_PATH/bin/hipcc" "--offload-arch=${HIP_TEST_ARCH:-native}")
  # WSL cannot service HIP printf hostcalls, which require PCIe atomics.
  if [[ "$(uname -r)" == *[Mm]icrosoft* ]]; then
    compiler+=(-mprintf-kind=buffered)
  fi
fi
"${compiler[@]}" -std=c++17 -Wall -I"$ROCM_PATH/include" "$source_file" \
  -o "$exe" -L"$lib_dir" -lamdhip64

if [[ "$native" == "1" ]]; then
  LD_LIBRARY_PATH="$lib_dir" "$exe"
  exit
fi

server_log="/tmp/lupine-hip-$SERVER_PORT_BASE.log"
LUPINE_PORT="$SERVER_PORT_BASE" LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}" \
  "$BUILD_DIR/lupine_driver_server" >"$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in $(seq 50); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$SERVER_PORT_BASE") 2>/dev/null; then
    break
  fi
  sleep 0.1
done

LUPINE_SERVER="127.0.0.1:$SERVER_PORT_BASE" LD_LIBRARY_PATH="$lib_dir" \
  LUPINE_HIP_SHIM=1 "$exe" || { cat "$server_log"; exit 1; }
