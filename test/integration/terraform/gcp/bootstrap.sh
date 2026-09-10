#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libnghttp2-14 python3
touch /var/lib/lupine-startup-complete
