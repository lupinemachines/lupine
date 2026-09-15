#!/usr/bin/env bash
# Return success only after verifying and loading the expected dependency image.
# The remaining arguments are the SSH command, its options, and its destination.
set -euo pipefail
cache_uri="$1"
dependency_image="$2"
remote_archive="$3"
shift 3
remote=("$@")
cleanup() {
  "${remote[@]}" "rm -f '$remote_archive'"
}
trap cleanup EXIT

expected_sha="$(
  gcloud storage objects describe "$cache_uri" \
    --raw --format='value(metadata.sha256)' 2>/dev/null || true
)"
[[ -n "$expected_sha" ]] || exit 1
gcloud storage cp "$cache_uri" - | "${remote[@]}" "cat > '$remote_archive'"
actual_sha="$("${remote[@]}" "sha256sum '$remote_archive' | cut -d' ' -f1")"
[[ "$actual_sha" == "$expected_sha" ]] || exit 1
"${remote[@]}" "sudo docker load -i '$remote_archive' && sudo docker image inspect '$dependency_image' >/dev/null"
gcloud storage objects update "$cache_uri" \
  --custom-time="$(date -u +%Y-%m-%dT%H:%M:%SZ)" || \
  echo "Could not refresh dependency-image cache lifetime: $cache_uri" >&2
