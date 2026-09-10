#!/usr/bin/env bash
set -euo pipefail
exec python3 "$(dirname "$0")/cleanup.py" "${1:?Terraform directory required}"
