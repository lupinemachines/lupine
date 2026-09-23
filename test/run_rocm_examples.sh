#!/usr/bin/env bash
# Leaf CMake projects from ROCm/rocm-examples.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/rocm_runner.py" rocm-examples "$@"
