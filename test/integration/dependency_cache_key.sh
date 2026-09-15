#!/usr/bin/env bash
# Hash dependency inputs only. Call from the repository root; source changes
# are copied into a new final image after restoring this dependency image.
set -euo pipefail
if [[ $# != 5 ]]; then
  echo "usage: $0 cache-version base-image samples-ref pytorch-index cuda-arch" >&2
  exit 1
fi
{
  printf '%s\n' "$@"
  sha256sum test/integration/Dockerfile.dependencies \
    test/run_cuda_library_samples.sh test/run_cuda_python_tests.sh
} | sha256sum | cut -d' ' -f1
