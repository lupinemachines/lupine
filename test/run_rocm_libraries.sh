#!/usr/bin/env bash
# rocBLAS/hipBLAS conformance tests and benchmarks from rocm-libraries.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/rocm_runner.py" rocm-libraries "$@"
