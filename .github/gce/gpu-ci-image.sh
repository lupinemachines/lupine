#!/usr/bin/env bash
# Provisions the GPU integration VM image. Runs once as the startup script of a
# builder VM; the image is named by a hash of this file and the base image, so
# editing it rebuilds the image on the next run.
set -euxo pipefail
trap 'echo "lupine-image-build: failed" > /dev/ttyS0; poweroff' ERR
export DEBIAN_FRONTEND=noninteractive

# Zonal GCE Ubuntu mirrors stall for tens of minutes in some fallback zones.
find /etc/apt \( -name '*.sources' -o -name '*.list' \) -exec sed -i -E \
  's#https?://[a-z0-9-]+\.gce\.clouds\.archive\.ubuntu\.com#http://archive.ubuntu.com#g' {} +
apt_opts=(-o Acquire::Retries=5 -o Acquire::http::Timeout=30)

# The dependency cache is a tarball of the data root, so every image must live
# under /var/lib/docker rather than containerd's store.
mkdir -p /etc/docker
echo '{"features": {"containerd-snapshotter": false}}' > /etc/docker/daemon.json

apt-get "${apt_opts[@]}" update
apt-get "${apt_opts[@]}" install -y --no-install-recommends \
  ca-certificates curl docker.io iproute2 libnghttp2-14 procps zstd

# Driver 580 predates CUDA 13.3; the CUDA 13.3 lane points the server at this
# forward-compat library. It installs under /usr/local/cuda-13.3/compat and is
# unused by the other lanes.
curl --fail --location --retry 5 --output /tmp/cuda-compat.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-compat-13-3_610.57.04-1ubuntu1_amd64.deb
echo "5150dbbe6376c29a53ec01730aa1b335f192d5965e95be71a448a719f10044a4  /tmp/cuda-compat.deb" |
  sha256sum --check
apt-get "${apt_opts[@]}" install -y --no-install-recommends /tmp/cuda-compat.deb
rm -f /tmp/cuda-compat.deb

apt-get clean
rm -rf /var/lib/apt/lists/*
echo "lupine-image-build: ok" > /dev/ttyS0
poweroff
