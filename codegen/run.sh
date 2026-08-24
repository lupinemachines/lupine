#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
image_name=lupine-codegen:cuda-13.3.1

docker build --file "$script_dir/Dockerfile" --tag "$image_name" "$script_dir"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$repo_root:/workspace" \
  --workdir /workspace \
  "$image_name" "$@"
