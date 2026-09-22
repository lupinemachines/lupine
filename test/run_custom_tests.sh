#!/usr/bin/env bash
# Build and run one custom CUDA test through the lupine client shims.
# BUILD_ONLY=1 builds without a server; BUILD_TESTS=0 reuses the executable.
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
# cuDNN ships outside the toolkit: a directory with its include/ and lib/, such
# as an nvidia-cudnn wheel's nvidia/cudnn.
CUDNN_HOME="${CUDNN_HOME:-}"
# NCCL too, such as an nvidia-nccl wheel's nvidia/nccl.
NCCL_HOME="${NCCL_HOME:-}"
# nvSHMEM too, such as an nvidia-nvshmem wheel's nvidia/nvshmem.
NVSHMEM_HOME="${NVSHMEM_HOME:-}"
# cuSPARSELt too, such as an nvidia-cusparselt wheel's nvidia/cusparselt.
CUSPARSELT_HOME="${CUSPARSELT_HOME:-}"
BUILD_ONLY="${BUILD_ONLY:-0}"
BUILD_TESTS="${BUILD_TESTS:-1}"
if [[ -n "${BUILD_DIR:-}" ]]; then
  owns_build_dir=0
else
  BUILD_DIR="$(mktemp -d)"
  owns_build_dir=1
fi

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done

# shellcheck source=test/integration/remote_server.sh
source "$repo_root/test/integration/remote_server.sh"

if [[ "$SERVER_UPLOAD" == "1" && "$BUILD_ONLY" != "1" ]]; then
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
# A test whose virtual devices have to straddle two servers says so, and gets a
# second one on the port above its own. Both are pinned to a single GPU, so
# ordinal 1 is the second connection's first device however many the host has.
second_port=""
if grep -q 'LUPINE_TEST_CONNECTIONS 2' "$src"; then
  second_port="$((port + 1))"
  second_pidfile="/tmp/lupine-custom-$second_port.pid"
  second_server_log="/tmp/lupine-custom-$second_port.log"
fi

cleanup() {
  if [[ "$BUILD_ONLY" != "1" ]]; then
    if [[ -n "${RESULTS_DIR:-}" ]]; then
      mkdir -p "$RESULTS_DIR"
      ssh_with_timeout "cat '$server_log'" >"$RESULTS_DIR/server.log" 2>&1 || true
      if [[ -n "$second_port" ]]; then
        ssh_with_timeout "cat '$second_server_log'" >"$RESULTS_DIR/server-2.log" 2>&1 || true
      fi
    fi
    stop_remote_server "$pidfile" "$server_log"
    if [[ -n "$second_port" ]]; then
      stop_remote_server "$second_pidfile" "$second_server_log"
    fi
    [[ "$SERVER_UPLOAD" == "1" ]] && ssh_with_timeout "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  fi
  if [[ "$owns_build_dir" == "1" ]]; then
    rm -rf "$BUILD_DIR"
  fi
}
trap cleanup EXIT

cudnn_args=()
if grep -q '#include <cudnn.h>' "$src"; then
  cudnn_lib=""
  if [[ -n "$CUDNN_HOME" ]]; then
    cudnn_lib="$(ls "$CUDNN_HOME"/lib/libcudnn.so.[0-9]* 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "$cudnn_lib" ]]; then
    echo "SKIP: $name needs cuDNN; set CUDNN_HOME"
    exit 0
  fi
  cudnn_args=(-I"$CUDNN_HOME/include" -L"$CUDNN_HOME/lib" -l:"$(basename "$cudnn_lib")")
fi
if grep -q '#include <nccl.h>' "$src"; then
  if [[ -z "$NCCL_HOME" || ! -e "$NCCL_HOME/lib/libnccl.so.2" ]]; then
    echo "SKIP: $name needs NCCL; set NCCL_HOME"
    exit 0
  fi
  cudnn_args+=(-I"$NCCL_HOME/include" -L"$NCCL_HOME/lib" -l:libnccl.so.2)
fi
if grep -q '#include <nvshmem_host.h>' "$src"; then
  if [[ -z "$NVSHMEM_HOME" || ! -e "$NVSHMEM_HOME/lib/libnvshmem_host.so.3" ]]; then
    echo "SKIP: $name needs nvSHMEM; set NVSHMEM_HOME"
    exit 0
  fi
  cudnn_args+=(-I"$NVSHMEM_HOME/include" -L"$NVSHMEM_HOME/lib" -l:libnvshmem_host.so.3)
fi

nvjitlink_args=()
if grep -q '#include <nvJitLink.h>' "$src"; then
  if [[ ! -e "$CUDA_HOME/lib64/libnvJitLink.so" ]]; then
    echo "SKIP: $name needs nvJitLink, which this toolkit does not have"
    exit 0
  fi
  nvjitlink_args=(-lnvJitLink)
fi

cusparselt_args=()
if grep -q '#include <cusparseLt.h>' "$src"; then
  if [[ -z "$CUSPARSELT_HOME" || ! -e "$CUSPARSELT_HOME/lib/libcusparseLt.so.0" ]]; then
    echo "SKIP: $name needs cuSPARSELt; set CUSPARSELT_HOME"
    exit 0
  fi
  cusparselt_args=(-I"$CUSPARSELT_HOME/include" -L"$CUSPARSELT_HOME/lib" -l:libcusparseLt.so.0)
fi

cufile_args=()
if grep -q '#include <cufile.h>' "$src"; then
  if [[ ! -e "$CUDA_HOME/lib64/libcufile.so" ]]; then
    echo "SKIP: $name needs cuFile, which this toolkit does not have"
    exit 0
  fi
  cufile_args=(-lcufile)
fi

cupti_args=()
if grep -q '#include <cupti.h>' "$src"; then
  if [[ ! -e "$CUDA_HOME/lib64/libcupti.so" ]]; then
    echo "SKIP: $name needs CUPTI, which this toolkit does not have"
    exit 0
  fi
  cupti_args=(-lcupti)
fi

if [[ "$BUILD_TESTS" == "1" ]]; then
  mkdir -p "$BUILD_DIR"
  arch_arg="-arch=all"
  [[ -n "$CUDA_SAMPLES_ARCH" ]] && arch_arg="-arch=sm_$CUDA_SAMPLES_ARCH"
  "$NVCC" --cudart=shared -Wno-deprecated-gpu-targets "$arch_arg" \
    "$src" -o "$exe" -lcuda -lcublas -lcublasLt -lcufft -lcusolver -lcusolverMg -lcurand -lnvrtc -lnvjpeg -lcusparse -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps -ldl -L"$CUDA_HOME/lib64/stubs" \
    "${cudnn_args[@]}" "${nvjitlink_args[@]}" "${cusparselt_args[@]}" "${cufile_args[@]}" "${cupti_args[@]}"
fi
[[ -x "$exe" ]] || { echo "missing custom test executable: $exe" >&2; exit 1; }
if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi
if [[ "${LUPINE_TEST_VARIANT:-}" == "driver-only" ]] && grep -q '#include <cufile.h>' "$src"; then
  # NVIDIA's libcufile drives the local GPU and its nvidia-fs driver from the
  # client, which a GPU-less client has neither of; only the cuFile shim runs.
  echo "SKIP: $name runs cuFile on the client, which needs more than the driver shim"
  exit 0
fi
if [[ "${LUPINE_TEST_VARIANT:-}" == "driver-only" ]] && grep -q '#include <cupti.h>' "$src"; then
  # NVIDIA's libcupti attaches to the local driver to profile the local
  # process, and a GPU-less client has neither; only the CUPTI shim runs.
  echo "SKIP: $name runs CUPTI on the client, which needs more than the driver shim"
  exit 0
fi

servers="$SERVER_HOST:$port"
if [[ -n "$second_port" ]]; then
  start_remote_server "$pidfile" "$server_log" "$port" "CUDA_VISIBLE_DEVICES=0"
  start_remote_server "$second_pidfile" "$second_server_log" "$second_port" \
    "CUDA_VISIBLE_DEVICES=0"
  servers="$servers,$SERVER_HOST:$second_port"
else
  start_remote_server "$pidfile" "$server_log" "$port"
fi
if [[ -n "${RESULTS_DIR:-}" ]]; then
  mkdir -p "$RESULTS_DIR"
  env LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${CUDNN_HOME:+$CUDNN_HOME/lib:}${NCCL_HOME:+$NCCL_HOME/lib:}${NVSHMEM_HOME:+$NVSHMEM_HOME/lib:}${CUSPARSELT_HOME:+$CUSPARSELT_HOME/lib:}${LD_LIBRARY_PATH:-}" \
    LUPINE_SERVER="$servers" "$exe" 2>&1 | tee "$RESULTS_DIR/client.log"
else
  env LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${CUDNN_HOME:+$CUDNN_HOME/lib:}${NCCL_HOME:+$NCCL_HOME/lib:}${NVSHMEM_HOME:+$NVSHMEM_HOME/lib:}${CUSPARSELT_HOME:+$CUSPARSELT_HOME/lib:}${LD_LIBRARY_PATH:-}" \
    LUPINE_SERVER="$servers" "$exe"
fi
