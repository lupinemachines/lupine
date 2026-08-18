FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

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

RUN cmake -S /opt/lupine -B /opt/lupine/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE \
      -DLUPINE_BUILD_CUDA=OFF \
      -DLUPINE_BUILD_NVML=OFF \
      -DLUPINE_BUILD_HIP=ON \
      -DLUPINE_BUILD_SERVER=ON \
    && cmake --build /opt/lupine/build --parallel \
    && ctest --test-dir /opt/lupine/build --output-on-failure

RUN test -x /opt/lupine/build/lupine_driver_server \
    && test -e /opt/lupine/build/libamdhip64.so.1
