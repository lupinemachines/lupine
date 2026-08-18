#!/usr/bin/env bash
set -euo pipefail

server="${1:?server executable required}"
runtime_dir="${2:?fake runtime directory required}"
client_test="${3:?client test executable required}"
shim="${4:?HIP shim required}"
port="$((30000 + $$ % 20000))"
server_log="$(mktemp)"

LUPINE_PORT="$port" \
LD_LIBRARY_PATH="$runtime_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$server" >"$server_log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" >/dev/null 2>&1 || true
  wait "$server_pid" >/dev/null 2>&1 || true
  rm -f "$server_log"
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 100); do
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    exec 3>&-
    ready=1
    break
  fi
  sleep 0.05
done

if [[ "$ready" != 1 ]] || ! kill -0 "$server_pid" 2>/dev/null; then
  cat "$server_log" >&2
  exit 1
fi

if ! LUPINE_SERVER="127.0.0.1:$port" "$client_test" "$shim"; then
  cat "$server_log" >&2
  exit 1
fi
