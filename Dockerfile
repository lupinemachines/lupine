ARG CUDA_VERSION=13.1.0
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG CUDA_RUNTIME_IMAGE_FLAVOR=runtime

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    libnghttp2-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lupine

COPY . /opt/lupine

RUN cmake -S /opt/lupine -B /opt/lupine/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"

FROM builder AS client-build

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver lupine_nvml

RUN test -e /opt/lupine/build/libcuda.so.1 \
    && test -e /opt/lupine/build/libnvidia-ml.so.1 \
    && ln -sf libcuda.so.1 /opt/lupine/build/libcuda.so \
    && ln -sf libnvidia-ml.so.1 /opt/lupine/build/libnvidia-ml.so \
    && ! nm -D --defined-only /opt/lupine/build/libcuda.so.1 \
      | awk '{print $3}' \
      | grep -E '^cuda'

FROM builder AS server-build

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server

RUN test -x /opt/lupine/build/lupine_driver_server

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS nvidia-utils

ARG DEBIAN_FRONTEND=noninteractive
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=

RUN set -eux; \
    apt-get update; \
    mkdir -p /tmp/nvidia-utils; \
    cd /tmp/nvidia-utils; \
    if [ -n "$NVIDIA_UTILS_VERSION" ]; then \
      apt-get download "${NVIDIA_UTILS_PACKAGE}=${NVIDIA_UTILS_VERSION}"; \
    else \
      apt-get download "${NVIDIA_UTILS_PACKAGE}"; \
    fi; \
    dpkg-deb -x ./*.deb /tmp/nvidia-utils/root; \
    cp /tmp/nvidia-utils/root/usr/bin/nvidia-smi /nvidia-smi; \
    chmod +x /nvidia-smi; \
    rm -rf /var/lib/apt/lists/* /tmp/nvidia-utils

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE client runtime with driver-only libcuda shim"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libnghttp2-14 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nvidia-utils /nvidia-smi /usr/bin/nvidia-smi

COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so.1

RUN ln -sf /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so \
    && ln -sf /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so

ENV LUPINE_LIBCUDA=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIB=/opt/lupine/lib/libcuda.so.1
ENV LD_LIBRARY_PATH=/opt/lupine/lib:${LD_LIBRARY_PATH}

ENTRYPOINT []
CMD ["bash"]

FROM ubuntu:${UBUNTU_VERSION} AS client-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE slim client runtime with driver-only libcuda shim"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}-slim"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libgcc-s1 \
    libnghttp2-14 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nvidia-utils /nvidia-smi /usr/bin/nvidia-smi
COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so.1

RUN ln -sf /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so \
    && ln -sf /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so

ENV LUPINE_LIBCUDA=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIB=/opt/lupine/lib/libcuda.so.1
ENV LD_LIBRARY_PATH=/opt/lupine/lib

ENTRYPOINT []
CMD ["bash"]

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-server"
LABEL org.opencontainers.image.description="LUPINE server runtime"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libnghttp2-14 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=server-build /opt/lupine/build/lupine_driver_server /opt/lupine/bin/lupine_driver_server

RUN chmod +x /opt/lupine/bin/lupine_driver_server

ENV LUPINE_PORT=14833
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]
