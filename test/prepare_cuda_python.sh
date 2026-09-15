#!/usr/bin/env bash
# Install matching bindings and a shallow checkout of their upstream tests.
# This is an explicit setup step; LIST_TESTS never performs installation.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$repo_root/.venv-pytorch312/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_PYTHON_URL="${CUDA_PYTHON_URL:-https://github.com/NVIDIA/cuda-python.git}"
CUDA_PYTHON_REF="${CUDA_PYTHON_REF:-}"
CUDA_PYTHON_DIR="${CUDA_PYTHON_DIR:-$repo_root/test/cuda-python/cuda-python}"

cuda_major() {
  local release=""
  if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
    release="$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n1)"
  fi
  if [[ -z "$release" && -f "$CUDA_HOME/version.json" ]]; then
    release="$(sed -nE 's/.*"cuda"[^0-9]*([0-9]+)\..*/\1/p' "$CUDA_HOME/version.json" | head -n1)"
  fi
  printf '%s\n' "$release"
}

# cuda-bindings was split out of cuda-python at 12.6.1; the 11.8 line still
# ships as cuda-python with the cuda.cuda/cuda.cudart module layout.
major="$(cuda_major)"
case "$major" in
  11) package=cuda-python; spec="cuda-python==11.8.*" ;;
  12) package=cuda-bindings; spec="cuda-bindings==12.*" ;;
  13) package=cuda-bindings; spec="cuda-bindings==13.*" ;;
  *) echo "cannot map CUDA toolkit major '$major' to a cuda-python release" >&2; exit 1 ;;
esac

"$PYTHON_BIN" -m pip install --quiet "$spec" numpy pytest pytest-benchmark
installed="$("$PYTHON_BIN" -c "from importlib.metadata import version; print(version('$package'))")"
if [[ -z "$CUDA_PYTHON_REF" ]]; then
  CUDA_PYTHON_REF="v$installed"
fi

mkdir -p "$(dirname "$CUDA_PYTHON_DIR")"
if [[ ! -d "$CUDA_PYTHON_DIR/.git" ]]; then
  git clone --quiet --depth 1 --branch "$CUDA_PYTHON_REF" \
    "$CUDA_PYTHON_URL" "$CUDA_PYTHON_DIR"
elif [[ "$(git -C "$CUDA_PYTHON_DIR" describe --tags --exact-match 2>/dev/null || true)" != "$CUDA_PYTHON_REF" ]]; then
  git -C "$CUDA_PYTHON_DIR" fetch --quiet --depth 1 origin "refs/tags/$CUDA_PYTHON_REF:refs/tags/$CUDA_PYTHON_REF"
  git -C "$CUDA_PYTHON_DIR" checkout --quiet "$CUDA_PYTHON_REF"
fi

