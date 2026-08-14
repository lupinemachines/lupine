# Builds the AMD HIP backend: the libamdhip64.so client shim and the
# HIP-enabled server. The server still links CUDA::cuda_driver (HIP is an
# additional backend in the same binary), so the base is an NVIDIA CUDA devel
# image with the ROCm HIP headers layered on top. This keeps the CUDA-only
# Dockerfile and its build targets untouched; the HIP path is opt-in via the
# ROCm/HIP SDK headers being present (see CMakeLists.txt).
ARG CUDA_VERSION=13.1.0
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS hip-builder

ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        cmake \
        gnupg \
        libnghttp2-dev \
        libssl-dev \
        ninja-build \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ROCm HIP headers. The hip-dev package installs the ROCm 6.x headers (with the
# R0600 versioning the generated code targets) under /opt/rocm/include, and an
# older stub set under /usr/include. CMake's find_path uses NO_CMAKE_SYSTEM_PATH
# so it binds /opt/rocm/include and ignores /usr/include.
RUN mkdir -p /etc/apt/keyrings \
    && wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && . /etc/os-release \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4 ${UBUNTU_CODENAME} main" \
       > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends hip-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lupine
COPY . /opt/lupine

RUN cmake -S /opt/lupine -B /opt/lupine/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server lupine_hip

RUN test -x /opt/lupine/build/lupine_driver_server \
    && test -e /opt/lupine/build/libamdhip64.so.1