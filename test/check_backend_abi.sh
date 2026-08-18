#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:?usage: check_backend_abi.sh BUILD_DIR}"

exports() {
  nm -D --defined-only -P "$1" | awk '{ print $1 }'
}

require_export() {
  local library="$1"
  local symbol="$2"
  if ! exports "$library" | grep -Fx "$symbol" >/dev/null; then
    echo "$library does not export $symbol" >&2
    exit 1
  fi
}

check_soname() {
  local library="$1"
  local soname="$2"
  if ! readelf -d "$library" | grep -F "Library soname: [$soname]" \
    >/dev/null; then
    echo "$library does not retain SONAME $soname" >&2
    exit 1
  fi
}

check_no_internal_exports() {
  local library="$1"
  if exports "$library" | grep -E \
    '^(rpc_|lupine_client_transport_|lupine_(cuda|nvml|hip)_server_|get_(cuda|nvml|hip)_handler)' \
    >/dev/null; then
    echo "$library exposes an internal RPC, transport, or registry symbol" >&2
    exit 1
  fi
}

check_export_allowlist() {
  local library="$1"
  local pattern="$2"
  local unexpected
  unexpected="$(exports "$library" | grep -Ev "$pattern" || true)"
  if [[ -n "$unexpected" ]]; then
    echo "$library exposes symbols outside its public ABI:" >&2
    echo "$unexpected" >&2
    exit 1
  fi
}

check_export_snapshot() {
  local library="$1"
  local expected="$2"
  local actual
  actual="$(exports "$library" | sort -u | sha256sum | awk '{ print $1 }')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$library export set changed: expected $expected, got $actual" >&2
    exit 1
  fi
}

check_no_cuda_dependency() {
  local library="$1"
  if readelf -d "$library" | grep -E \
    'Shared library: \[(libcuda\.so|libamdhip64\.so)' >/dev/null; then
    echo "$library has an unexpected accelerator-shim dependency" >&2
    exit 1
  fi
}

cuda="$build_dir/libcuda.so.1"
nvml="$build_dir/libnvidia-ml.so.1"
test -f "$cuda"
test -f "$nvml"

check_soname "$cuda" libcuda.so.1
check_soname "$nvml" libnvidia-ml.so.1
require_export "$cuda" cuInit
require_export "$cuda" dlsym
require_export "$cuda" lupine_checkpoint_drain_cuda_calls
require_export "$nvml" nvmlInit_v2
require_export "$nvml" nvmlDeviceGetCount_v2
check_no_internal_exports "$cuda"
check_no_internal_exports "$nvml"
check_export_allowlist "$cuda" \
  '^(cu[A-Z][A-Za-z0-9_]*|dlsym|lupine_checkpoint_(drain_cuda_calls|resume_captures|resume_cuda_calls|wait_for_captures))$'
check_export_allowlist "$nvml" '^nvml[A-Z][A-Za-z0-9_]*$'
check_export_snapshot "$cuda" \
  b5c738c0c66ff110671270fd3491a996e14b364a89f1c53720c1b9418102710e
check_export_snapshot "$nvml" \
  2a8fd4454aac4146d0e812fb1fc1600144f1c53bab0e99edf76fe12c0ee4507f
check_no_cuda_dependency "$cuda"
check_no_cuda_dependency "$nvml"

if [[ -f "$build_dir/libamdhip64.so.1" ]]; then
  hip="$build_dir/libamdhip64.so.1"
  check_soname "$hip" libamdhip64.so.1
  require_export "$hip" hipInit
  require_export "$hip" hipGetDeviceCount
  check_no_internal_exports "$hip"
  check_export_allowlist "$hip" '^hip[A-Z][A-Za-z0-9_]*$'
  check_no_cuda_dependency "$hip"
fi
