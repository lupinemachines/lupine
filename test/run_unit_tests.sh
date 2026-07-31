#!/usr/bin/env bash
# Build and run the repo's unit tests (the *_test executables registered with
# ctest). Mirrors run_custom_tests.sh conventions: env overrides, and any
# arguments select individual tests by name. Needs no GPU, server, or network;
# new *_test targets in CMakeLists.txt are picked up automatically.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-$repo_root/build}"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  echo "missing cmake build dir: $BUILD_DIR (configure lupine first)" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  regex="^($(IFS='|'; echo "$*"))$"
else
  regex="_test$"
fi

cmake --build "$BUILD_DIR" --parallel "$JOBS"
ctest --test-dir "$BUILD_DIR" --output-on-failure -R "$regex"
