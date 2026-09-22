#!/usr/bin/env bash
# Opt-in AMD GPU test; use --native for the same executable on local ROCm.
set -euo pipefail
test_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HIP_TEST_SOURCE="$test_dir/test_hip_hello_world.hip"
output="$(mktemp)"
trap 'rm -f "$output"' EXIT
"$test_dir/run_hip_runtime_test.sh" "$@" | tee "$output"
# Successful exit alone is insufficient: a broken printf path can lose the
# device lines. Thread scheduling may change their order, so compare counts.
python3 - "$output" <<'PY'
from collections import Counter
from pathlib import Path
import sys

expected = Counter({
    "Hello from the host!": 1,
    "Hello from HIP block 0 thread 0!": 3,
    "Hello from HIP block 0 thread 1!": 1,
    "Hello from HIP block 1 thread 0!": 1,
    "Hello from HIP block 1 thread 1!": 1,
    "PASS": 1,
})
actual = Counter(Path(sys.argv[1]).read_text().splitlines())
if actual != expected:
    raise SystemExit(f"Unexpected device stdout: missing={expected - actual}, extra={actual - expected}")
PY
