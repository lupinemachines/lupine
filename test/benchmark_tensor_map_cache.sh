#!/usr/bin/env bash
# Synthetic RPC/cache benchmark. Requires a real GPU for context/allocation,
# but the server encoder is a fixture and produces no hardware TMA descriptor.
# Usage: bash test/benchmark_tensor_map_cache.sh BASE_BUILD OPT_BUILD
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline="$(realpath "${1:?baseline build directory required}")"
optimized="$(realpath "${2:?optimized build directory required}")"
cuda="${CUDA_HOME:-/usr/local/cuda}"
scratch="$(mktemp -d /tmp/lupine-tensor-map-bench.XXXXXX)"
port="${LUPINE_TENSOR_MAP_BENCH_PORT:-14928}"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  printf 'Benchmark artifacts: %s\n' "$scratch"
}
trap cleanup EXIT
"${CXX:-c++}" -shared -fPIC -O3 -I"$cuda/include" \
  "$repo_root/test/tensor_map_encode_fixture.cpp" -o "$scratch/encoder.so"
"${CXX:-c++}" -shared -fPIC -O3 -I"$cuda/include" \
  "$repo_root/test/tensor_map_no_local_fixture.cpp" -o "$scratch/no-local.so"
"$cuda/bin/nvcc" -std=c++17 -O3 \
  "$repo_root/test/test_tensor_map_encode_cache.cu" \
  -L"$cuda/lib64/stubs" -lcuda -o "$scratch/test"
for mode in baseline optimized; do
  if [[ "$mode" == baseline ]]; then build="$baseline"; else build="$optimized"; fi
  env CUDA_VISIBLE_DEVICES="${LUPINE_TENSOR_MAP_BENCH_GPU:-0}" \
    LUPINE_PORT="$port" LD_PRELOAD="$scratch/encoder.so" \
    "$build/lupine_driver_server" >"$scratch/$mode-server.log" 2>&1 &
  server_pid=$!
  for attempt in {1..100}; do
    kill -0 "$server_pid" || { cat "$scratch/$mode-server.log"; exit 1; }
    if grep -q 'Server listening' "$scratch/$mode-server.log"; then break; fi
    sleep 0.05
  done
  for repetition in {1..5}; do
    printf '%s run %s: ' "$mode" "$repetition"
    env LUPINE_REAL_LIBCUDA="$scratch/no-local.so" LD_LIBRARY_PATH="$build" \
      LUPINE_SERVER="127.0.0.1:$port" LUPINE_TENSOR_MAP_BENCH_CALLS=10000 \
      LUPINE_RPC_STATS="$scratch/$mode-$repetition.tsv" "$scratch/test"
    awk '$1 == 1386204641 {print "  encode RPCs:", $2}' \
      "$scratch/$mode-$repetition.tsv"
    expected=10008
    if [[ "$mode" == optimized ]]; then expected=6; fi
    awk -v expected="$expected" \
      '$1 == 1386204641 {found=1; if ($2 != expected) exit 1} END {if (!found) exit 1}' \
      "$scratch/$mode-$repetition.tsv"
  done
  env LUPINE_REAL_LIBCUDA="$scratch/no-local.so" LD_LIBRARY_PATH="$build" \
    LUPINE_SERVER="127.0.0.1:$port" LUPINE_TENSOR_MAP_TEST_EVICTION=1 \
    "$scratch/test"
  kill "$server_pid"
  wait "$server_pid" || true
  server_pid=""
done
