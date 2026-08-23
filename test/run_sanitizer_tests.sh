#!/usr/bin/env bash
# Build and run the CPU-only *_test targets under a sanitizer. These exercise
# the transport, decoder, framing, ipc, and checkpoint code with no GPU,
# server, or network, so they run cheaply in CI. The nvcc-compiled driver and
# server are never built here (nvcc does not accept -fsanitize).
#
# Usage: run_sanitizer_tests.sh <sanitizer>
#   <sanitizer> is a -fsanitize value, e.g. "address,undefined" or "thread".
set -euo pipefail

sanitizer="${1:?usage: run_sanitizer_tests.sh <address,undefined|thread>}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_root/build-sanitize-${sanitizer//,/-}}"
jobs="${JOBS:-$(nproc)}"

# Sanitized CPU test targets and the ctest names they register.
targets=(h2_test h2_unknown_cuda_version_test checkpoint_test ipc_test
         server_checkpoint_test test_lupinecr_provider)
test_regex='^(h2_test|h2_unknown_cuda_version_test|checkpoint_test|ipc_test|server_checkpoint_test|server_checkpoint_missing_provider_test)$'

cmake -S "$repo_root" -B "$build_dir" -DLUPINE_SANITIZE="$sanitizer" >/dev/null
cmake --build "$build_dir" --parallel "$jobs" --target "${targets[@]}"

# Symbolized, fail-fast output; a leak or race aborts the test.
common="halt_on_error=1:abort_on_error=1:print_stacktrace=1:symbolize=1"
export ASAN_OPTIONS="${common}:detect_leaks=1:strict_string_checks=1"
export UBSAN_OPTIONS="${common}:print_summary=1"
export TSAN_OPTIONS="${common}:second_deadlock_stack=1"
export LSAN_OPTIONS="print_suppressions=0"
supp="$repo_root/test/sanitizer_suppressions.txt"
if [[ -f "$supp" ]]; then
  export LSAN_OPTIONS="${LSAN_OPTIONS}:suppressions=$supp"
fi

# ThreadSanitizer's shadow memory is incompatible with high-entropy ASLR, so
# run its tests with randomization disabled (setarch -R propagates to the
# ctest children). Other sanitizers run normally.
runner=()
if [[ "$sanitizer" == *thread* ]] && command -v setarch >/dev/null; then
  runner=(setarch -R)
fi
"${runner[@]}" ctest --test-dir "$build_dir" --output-on-failure -R "$test_regex"
