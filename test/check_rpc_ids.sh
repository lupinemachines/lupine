#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
expected_file="$repo_root/test/rpc_ids.sha256"
ids_file="$repo_root/codegen/gen_rpc_ids.h"

expected="$(awk 'NR == 1 { print $1 }' "$expected_file")"
actual="$(awk '/^#define (RPC_|LUPINE_RPC_)/ && \
                    $2 !~ /^(RPC|LUPINE_RPC)_hip/ { print }' "$ids_file" \
  | sha256sum | awk '{ print $1 }')"

if [[ "$actual" != "$expected" ]]; then
  echo "RPC ID snapshot changed: expected $expected, got $actual" >&2
  echo "Review wire compatibility before updating test/rpc_ids.sha256." >&2
  exit 1
fi
