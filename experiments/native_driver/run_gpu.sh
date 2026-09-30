#!/bin/bash
# End-to-end feasibility test on an existing, isolated L4/driver-580 GPU host.
# Usage: ./run_gpu.sh USER@GPU_HOST SSH_KEY
# The CPU client runs real libcuda; the GPU host runs only the raw-device server.
# This tests execution, not backend replacement or checkpoint/restore.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 USER@GPU_HOST SSH_KEY" >&2
  exit 2
fi
source_root=$(cd -- "$(dirname -- "$0")" && pwd)
target=$1
key=$2
work=$(mktemp -d /tmp/lupine-native-driver.XXXXXX)
ssh_args=(-i "$key" -o StrictHostKeyChecking=accept-new)
remote=$(ssh "${ssh_args[@]}" "$target" 'mktemp -d /tmp/lupine-native-driver.XXXXXX')
[[ "$remote" =~ ^/tmp/lupine-native-driver\.[A-Za-z0-9]+$ ]] || exit 2
tunnel=
cleanup() {
  result=$?
  trap - EXIT
  if [ -n "$tunnel" ]; then kill "$tunnel" 2>/dev/null || true; fi
  ssh "${ssh_args[@]}" "$target" "sudo cat '$remote/server.log'" > "$work/server.log" 2>/dev/null || true
  ssh "${ssh_args[@]}" "$target" "if test -f '$remote/server.pid'; then sudo kill \$(cat '$remote/server.pid') 2>/dev/null || true; fi; rm -rf '$remote'" || true
  echo "Result: $result. Artifacts: $work"
  exit "$result"
}
trap cleanup EXIT
cmake -S "$source_root" -B "$work/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$work/build" --parallel 2
scp "${ssh_args[@]}" "$work/build/native_driver_server" "$source_root/probe.ptx" "$target:$remote/"
ssh "${ssh_args[@]}" "$target" bash -s -- "$remote" <<'REMOTE'
set -euo pipefail
work=$1
mkdir -p "$work/metadata/pci"
for pci in /sys/bus/pci/devices/*; do
  if [ "$(cat "$pci/vendor")" = 0x10de ]; then
    mkdir "$work/metadata/pci/${pci##*/}"
    cat "$pci/config" > "$work/metadata/pci/${pci##*/}/config"
  fi
done
cat /proc/modules > "$work/metadata/nvidia-modules"
cat /proc/driver/nvidia/params > "$work/metadata/nvidia-params"
cat /sys/module/nvidia/initstate > "$work/metadata/nvidia-initstate"
cp -L /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$work/libcuda.so.1"
/usr/local/cuda/bin/ptxas "$work/probe.ptx" -arch=sm_89 -o "$work/probe.cubin"
# userfaultfd must service copy_from_user faults, not just user-mode faults.
sudo sh -c "nohup '$work/native_driver_server' >'$work/server.log' 2>&1 </dev/null & echo \$! >'$work/server.pid'"
REMOTE
scp "${ssh_args[@]}" -r "$target:$remote/metadata" "$target:$remote/libcuda.so.1" "$target:$remote/probe.cubin" "$work/"
ssh "${ssh_args[@]}" -o ExitOnForwardFailure=yes -N -L 127.0.0.1:16130:127.0.0.1:16130 "$target" &
tunnel=$!
sleep 1
timeout 300 env LUPINE_NATIVE_METADATA="$work/metadata" \
  LD_PRELOAD="$work/build/libnative_driver_boundary.so:$work/build/libnative_driver_client.so" \
  "$work/build/native_driver_probe" "$work/libcuda.so.1" "$work/probe.cubin" > "$work/client.log" 2>&1
cat "$work/client.log"
