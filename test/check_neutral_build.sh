#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:?usage: check_neutral_build.sh BUILD_DIR}"
compile_commands="$build_dir/compile_commands.json"
core="$build_dir/liblupine_rpc_core.a"

test -f "$compile_commands"
test -f "$core"

neutral_commands="$(mktemp)"
trap 'rm -f "$neutral_commands"' EXIT
command_line=""
while IFS= read -r line; do
  if [[ "$line" == *'"command":'* ]]; then
    command_line="$line"
  elif [[ "$line" =~ /(rpc|h2|compress|transport|rpc_server)\.cpp\" ]]; then
    printf '%s\n' "$command_line" >>"$neutral_commands"
  fi
done <"$compile_commands"

if grep -Eq '/usr/local/cuda|/opt/rocm|(^|[ /])nvcc([ /]|$)|hipcc' \
  "$neutral_commands"; then
  echo "neutral source received an accelerator include path or compiler" >&2
  cat "$neutral_commands" >&2
  exit 1
fi

if nm -u "$core" | grep -Eq \
  '\b(CU[A-Za-z_]|cuda[A-Za-z_]|hip[A-Za-z_]|nvml[A-Za-z_])'; then
  echo "neutral RPC core imports an accelerator symbol" >&2
  exit 1
fi
