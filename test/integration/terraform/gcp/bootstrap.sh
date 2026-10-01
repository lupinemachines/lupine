#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

runtime_ready() {
  python3 -c 'import ctypes; ctypes.CDLL("libnghttp2.so.14")' > /dev/null 2>&1
}

# The pinned images normally include these already. Avoid competing with their
# first-boot package jobs just to mark installed dependencies as manual.
if runtime_ready; then
  touch /var/lib/lupine-startup-complete
  exit 0
fi

for attempt in {1..12}; do
  if apt-get -o DPkg::Lock::Timeout=20 update && \
      apt-get -o DPkg::Lock::Timeout=20 install -y --no-install-recommends libnghttp2-14 python3 && \
      runtime_ready; then
    touch /var/lib/lupine-startup-complete
    exit 0
  fi
  sleep 5
done
touch /var/lib/lupine-startup-failed
exit 1
