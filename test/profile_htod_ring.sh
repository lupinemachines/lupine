#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="$(mktemp "${TMPDIR:-/tmp}/profile-htod-ring.XXXXXX")"
trap 'rm -f "$binary"' EXIT

nvcc="${NVCC:-}"
if [[ -z "$nvcc" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    nvcc="$(command -v nvcc)"
  else
    nvcc="${CUDA_PATH:-/usr/local/cuda}/bin/nvcc"
  fi
fi

"$nvcc" -O2 -std=c++17 "$script_dir/profile_htod_ring.cu" \
  -o "$binary" -lcuda
"$binary"
