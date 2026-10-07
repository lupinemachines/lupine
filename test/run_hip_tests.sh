#!/usr/bin/env bash
# HIP Catch2 feature directories and standalone hip-tests samples.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/rocm_runner.py" hip-tests "$@"
