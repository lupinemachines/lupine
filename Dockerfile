ARG CUDA_VERSION=13.3.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG ROCM_VERSION=7.2.4
ARG ROCM_SDK_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
ARG ROCM_SDK_PLATFORM=linux/amd64

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS cuda-sdk

FROM --platform=${ROCM_SDK_PLATFORM} ${ROCM_SDK_IMAGE} AS rocm-sdk

FROM cuda-sdk AS cuda-ops

WORKDIR /opt/lupine

COPY ops/smemcpy.cu ops/smemcpy.h ops/smemcpy_dispatch.h /opt/lupine/ops/

# Precompile CUDA operations before entering the SDK-neutral builder. Include
# native code for every architecture accepted by this toolkit and PTX for the
# oldest one as a forward-compatible fallback.
RUN set -eux; \
    set --; \
    oldest=""; \
    for code in $(nvcc --list-gpu-code); do \
      arch="${code#sm_}"; \
      if [ -z "$oldest" ] || [ "$arch" -lt "$oldest" ]; then \
        oldest="$arch"; \
      fi; \
      set -- "$@" \
        "--generate-code=arch=compute_${arch},code=sm_${arch}"; \
    done; \
    test -n "$oldest"; \
    set -- "$@" \
      "--generate-code=arch=compute_${oldest},code=compute_${oldest}"; \
    mkdir -p /opt/lupine-precompiled-ops/cuda; \
    nvcc -std=c++17 --fatbin "$@" \
      -I/opt/lupine /opt/lupine/ops/smemcpy.cu \
      -o /tmp/lupine_smemcpy.fatbin; \
    bin2c --const --name lupine_smemcpy_fatbin \
      /tmp/lupine_smemcpy.fatbin \
      > /opt/lupine-precompiled-ops/cuda/smemcpy.cpp; \
    printf '\nextern "C" const void *lupine_cuda_smemcpy_image() {\n  return lupine_smemcpy_fatbin;\n}\n' \
      >> /opt/lupine-precompiled-ops/cuda/smemcpy.cpp; \
    rm /tmp/lupine_smemcpy.fatbin

FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release
ARG CUDA_VERSION

# Device operations are precompiled in their own SDK stages. The main builder
# needs only API headers, link-time stubs, and the combined operation directory,
# so CUDA and ROCm compiler SDKs never have to coexist here.
COPY --from=cuda-sdk /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=cuda-sdk /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
COPY --from=cuda-ops /opt/lupine-precompiled-ops/ /opt/lupine-precompiled-ops/
COPY --from=rocm-sdk /opt/rocm/include/ /opt/rocm/include/

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    libnghttp2-dev \
    libssl-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lupine

COPY . /opt/lupine

RUN cmake -S /opt/lupine -B /opt/lupine/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_CUDA_DRIVER_LIBRARY="${CUDA_HOME}/lib64/stubs/libcuda.so" \
      -DLUPINE_CUDA_VERSION_OVERRIDE="${CUDA_VERSION}" \
      -DLUPINE_PRECOMPILED_OPS=/opt/lupine-precompiled-ops

FROM builder AS client-build

RUN cmake --build /opt/lupine/build --parallel \
      --target lupine_cuda_client lupine_nvml_client lupine_hip_client

RUN test -e /opt/lupine/build/libcuda.so.1 \
    && test -e /opt/lupine/build/libnvidia-ml.so.1 \
    && test -e /opt/lupine/build/libamdhip64.so.1 \
    && ln -sf libcuda.so.1 /opt/lupine/build/libcuda.so \
    && ln -sf libnvidia-ml.so.1 /opt/lupine/build/libnvidia-ml.so \
    && ln -sf libamdhip64.so.1 /opt/lupine/build/libamdhip64.so \
    && ! nm -D --defined-only /opt/lupine/build/libcuda.so.1 \
      | awk '{print $3}' \
      | grep -E '^cuda'

FROM builder AS server-build

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server

RUN test -x /opt/lupine/build/lupine_driver_server

FROM cuda-sdk AS nvidia-utils

ARG DEBIAN_FRONTEND=noninteractive
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=

# Ubuntu periodically turns an older nvidia-utils-NNN into an empty
# transitional package (Depends on a newer NNN, no binaries of its own) as
# driver branches age out, so the pinned NVIDIA_UTILS_PACKAGE can silently
# stop shipping nvidia-smi. Try the pin first, then fall back to whichever
# nvidia-utils-NNN (newest first) actually contains it.
RUN set -eux; \
    apt-get update; \
    mkdir -p /tmp/nvidia-utils; \
    cd /tmp/nvidia-utils; \
    try_nvidia_utils() { \
      rm -f ./*.deb; \
      rm -rf /tmp/nvidia-utils/root; \
      apt-get download "$1" >/dev/null 2>&1 || return 1; \
      dpkg-deb -x ./*.deb /tmp/nvidia-utils/root || return 1; \
      test -x /tmp/nvidia-utils/root/usr/bin/nvidia-smi; \
    }; \
    found=""; \
    if [ -n "$NVIDIA_UTILS_VERSION" ]; then \
      try_nvidia_utils "${NVIDIA_UTILS_PACKAGE}=${NVIDIA_UTILS_VERSION}" && found=1; \
    else \
      try_nvidia_utils "${NVIDIA_UTILS_PACKAGE}" && found=1; \
    fi; \
    if [ -z "$found" ]; then \
      for pkg in $(apt-cache search --names-only '^nvidia-utils-[0-9]+$' | awk '{print $1}' | sort -t- -k3 -rn); do \
        if try_nvidia_utils "$pkg"; then found=1; break; fi; \
      done; \
    fi; \
    test -n "$found"; \
    cp /tmp/nvidia-utils/root/usr/bin/nvidia-smi /nvidia-smi; \
    chmod +x /nvidia-smi; \
    rm -rf /var/lib/apt/lists/* /tmp/nvidia-utils

FROM ubuntu:${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE client runtime with CUDA, NVML, and HIP shims"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-rocm-${ROCM_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libgcc-s1 \
    libnghttp2-14 \
    libstdc++6 \
    # libssl3 on jammy, libssl3t64 on noble.
    && (apt-get install -y --no-install-recommends libssl3 || apt-get install -y --no-install-recommends libssl3t64) \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nvidia-utils /nvidia-smi /usr/bin/nvidia-smi

COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so.1
COPY --from=client-build /opt/lupine/build/libamdhip64.so.1 /opt/lupine/lib/libamdhip64.so.1

RUN ln -sf /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so \
    && ln -sf /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so \
    && ln -sf /opt/lupine/lib/libamdhip64.so.1 /opt/lupine/lib/libamdhip64.so

ENV LUPINE_LIBCUDA=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIB=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIBHIP=/opt/lupine/lib/libamdhip64.so.1
ENV LD_LIBRARY_PATH=/opt/lupine/lib

ENTRYPOINT []
CMD ["bash"]

FROM client AS client-slim

ARG CUDA_VERSION
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.description="LUPINE SDK-free client runtime with CUDA, NVML, and HIP shims"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-rocm-${ROCM_VERSION}-ubuntu${UBUNTU_VERSION}-slim"

FROM ubuntu:${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG AMDGPU_INSTALL_VERSION=7.2.4.70204-1
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_VERSION
ARG LUPINE_REQUIRE_CLIENT_BUNDLES=0
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-server"
LABEL org.opencontainers.image.description="LUPINE CUDA and ROCm server runtime"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-rocm-${ROCM_VERSION}-ubuntu${UBUNTU_VERSION}"

# NVIDIA's container runtime supplies the host driver ahead of the compatibility
# library in LD_LIBRARY_PATH. The compatibility package also lets the unified
# binary start on AMD-only hosts, where no NVIDIA driver is mounted.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libgcc-s1 \
    libnghttp2-14 \
    libstdc++6 \
    wget \
    # libssl3 on jammy, libssl3t64 on noble.
    && (apt-get install -y --no-install-recommends libssl3 || apt-get install -y --no-install-recommends libssl3t64) \
    && arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) cuda_repo_arch=x86_64 ;; \
         arm64) cuda_repo_arch=sbsa ;; \
         *) echo "Unsupported CUDA architecture: $arch" >&2; exit 1 ;; \
       esac \
    && ubuntu_repo="ubuntu$(printf '%s' "${UBUNTU_VERSION}" | tr -d '.')" \
    && cuda_series="$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "-" $2}')" \
    && wget -q \
         "https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_repo}/${cuda_repo_arch}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb" \
         -O /tmp/cuda-keyring.deb \
    && apt-get install -y --no-install-recommends /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends "cuda-compat-${cuda_series}" \
    && cuda_series_dot="$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "." $2}')" \
    && ln -sfn "cuda-${cuda_series_dot}" /usr/local/cuda \
    && if [ "$arch" = amd64 ]; then \
         . /etc/os-release; \
         wget -q \
           "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/${VERSION_CODENAME}/amdgpu-install_${AMDGPU_INSTALL_VERSION}_all.deb" \
           -O /tmp/amdgpu-install.deb; \
         apt-get install -y --no-install-recommends /tmp/amdgpu-install.deb; \
         apt-get update; \
         apt-get install -y --no-install-recommends hip-runtime-amd; \
         apt-get purge -y amdgpu-install; \
       fi \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*.deb

COPY --from=server-build /opt/lupine/build/lupine_driver_server /opt/lupine/bin/lupine_driver_server
COPY client-bundles/ /opt/lupine/client-bundles/

RUN set -eux; \
    if [ "${LUPINE_REQUIRE_CLIENT_BUNDLES}" = 1 ]; then \
      for platform in \
        linux/amd64 linux/arm64 \
        macos/amd64 macos/arm64 \
        windows/amd64 windows/arm64; do \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip"; \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip.etag"; \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip.digest"; \
      done; \
    fi

RUN chmod +x /opt/lupine/bin/lupine_driver_server

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/opt/rocm/lib
ENV LUPINE_PORT=14833
ENV LUPINE_CLIENT_BUNDLE_DIR=/opt/lupine/client-bundles
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]
