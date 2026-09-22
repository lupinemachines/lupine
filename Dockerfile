ARG CUDA_VERSION=13.3.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG ROCM_VERSION=7.2.4
ARG ROCM_SDK_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
ARG ROCM_SDK_PLATFORM=linux/amd64

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS cuda-sdk

FROM --platform=${ROCM_SDK_PLATFORM} ${ROCM_SDK_IMAGE} AS rocm-sdk

FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim@sha256:0b074d1ae15f5c3f1861354917d356e5afbd5a4c53c1190e81ad2f2add46e45b AS uv

FROM cuda-sdk AS cuda-ops

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/lupine

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY ops/precompile.cmake ops/cuda_smemcpy.cu ops/smemcpy.h ops/smemcpy_dispatch.h /opt/lupine/ops/

RUN cmake \
      -DLUPINE_PRECOMPILED_OPS=/opt/lupine-precompiled-ops \
      -P /opt/lupine/ops/precompile.cmake

FROM cuda-sdk AS nccl-headers

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION

# NCCL ships outside the toolkit too. Its header comes from the newest release
# built for this CUDA series, the one the server image installs.
RUN apt-get update \
    && cuda_series_dot="$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "." $2}')" \
    && nccl_version="$(apt-cache madison libnccl-dev | awk -v s="+cuda${cuda_series_dot}" 'index($3, s) {print $3; exit}')" \
    && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
         "libnccl2=${nccl_version}" "libnccl-dev=${nccl_version}" \
    && mkdir -p /opt/nccl/include \
    && cp -L /usr/include/nccl.h /opt/nccl/include/ \
    && rm -rf /var/lib/apt/lists/*

FROM cuda-sdk AS nvshmem-headers

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION

# nvSHMEM ships outside the toolkit too, in a package per CUDA major that puts
# its headers under a directory of their own. The CUDA 11 releases predate
# nvshmem_host.h and are packaged for x86_64 alone, so that series is skipped
# and its empty directory leaves the shim out of the build.
RUN cuda_major="${CUDA_VERSION%%.*}" \
    && mkdir -p /opt/nvshmem/include \
    && if [ "$cuda_major" -ge 12 ]; then \
         apt-get update \
         && apt-get install -y --no-install-recommends "libnvshmem3-dev-cuda-${cuda_major}" \
         && cp -rL "/usr/include/nvshmem_${cuda_major}/." /opt/nvshmem/include/ \
         && rm -rf /var/lib/apt/lists/*; \
       fi
FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release
ARG LUPINE_CLIENT_BUNDLE_INPUT
ARG CUDA_VERSION

# Device operations are precompiled in their own SDK stages. The main builder
# needs only API headers, link-time stubs, and the combined operation directory,
# so CUDA and ROCm compiler SDKs never have to coexist here.
COPY --from=cuda-sdk /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=nccl-headers /opt/nccl/include/ /usr/local/cuda/include/
# nvSHMEM's headers keep directories of their own (device/, host/, non_abi/),
# so they stay beside the toolkit's rather than inside them.
COPY --from=nvshmem-headers /opt/nvshmem/include/ /opt/nvshmem/include/
COPY --from=cuda-sdk /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
COPY --from=cuda-ops /opt/lupine-precompiled-ops/ /opt/lupine-precompiled-ops/
COPY --from=rocm-sdk /opt/rocm/include/ /opt/rocm/include/
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    libnghttp2-dev \
    libssl-dev \
    ninja-build \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lupine

COPY . /opt/lupine

RUN cmake -S /opt/lupine -B /opt/lupine/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_CUDA_DRIVER_LIBRARY="${CUDA_HOME}/lib64/stubs/libcuda.so" \
      -DLUPINE_CUDA_VERSION_OVERRIDE="${CUDA_VERSION}" \
      -DLUPINE_CLIENT_BUNDLE_INPUT="${LUPINE_CLIENT_BUNDLE_INPUT}" \
      -DLUPINE_NVSHMEM_INCLUDE_DIR=/opt/nvshmem/include \
      -DLUPINE_PRECOMPILED_OPS=/opt/lupine-precompiled-ops

FROM builder AS client-build

# nvSHMEM is header-gated, so its target exists only where the CUDA major
# carries the package.
RUN cmake --build /opt/lupine/build --parallel \
      --target lupine_cuda_client lupine_nccl_client lupine_nvml_client \
               lupine_hip_client \
    && if [ "${CUDA_VERSION%%.*}" -ge 12 ]; then \
         cmake --build /opt/lupine/build --parallel --target lupine_nvshmem_client; \
       fi

FROM builder AS server-build

ARG LUPINE_CLIENT_BUNDLE_INPUT

RUN test -n "${LUPINE_CLIENT_BUNDLE_INPUT}"

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server

FROM ubuntu:${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE client runtime with CUDA driver, NCCL, nvSHMEM, NVML, and HIP shims"
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
    requested="$NVIDIA_UTILS_PACKAGE"; \
    if [ -n "$NVIDIA_UTILS_VERSION" ]; then \
      requested="${NVIDIA_UTILS_PACKAGE}=${NVIDIA_UTILS_VERSION}"; \
    fi; \
    if ! try_nvidia_utils "$requested"; then \
      for pkg in $(apt-cache search --names-only '^nvidia-utils-[0-9]+$' | awk '{print $1}' | sort -t- -k3 -rn); do \
        if try_nvidia_utils "$pkg"; then break; fi; \
      done; \
    fi; \
    cp /tmp/nvidia-utils/root/usr/bin/nvidia-smi /usr/bin/nvidia-smi; \
    rm -rf /var/lib/apt/lists/* /tmp/nvidia-utils

COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libnccl.so* /opt/lupine/lib/
# The bracket keeps the COPY valid on toolkits without an nvSHMEM shim.
COPY --from=client-build /opt/lupine/build/libnvshmem_hos[t].so* /opt/lupine/lib/
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

FROM ubuntu:${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG AMDGPU_INSTALL_VERSION=7.2.4.70204-1
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_VERSION
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
    && nccl_version="$(apt-cache madison libnccl2 | awk -v s="+cuda$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "." $2}')" 'index($3, s) {print $3; exit}')" \
    && apt-get install -y --no-install-recommends "cuda-compat-${cuda_series}" "libnccl2=${nccl_version}" \
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

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/opt/rocm/lib
ENV LUPINE_PORT=14833
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]
