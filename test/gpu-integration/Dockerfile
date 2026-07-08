# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_SAMPLE_SKIP_LIST=""
ARG CUDA_SAMPLES_REF=""
ARG PYTORCH_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_LIB_DIR=/usr/local/cuda/lib64 \
    LUPINE_BUILD_DIR=/opt/lupine/build \
    LUPINE_CUSTOM_TEST_BIN_DIR=/opt/lupine/custom-tests \
    CUDA_SAMPLES_DIR=/opt/cuda-samples-src \
    CUDA_SAMPLES_BUILD_DIR=/opt/cuda-samples-build \
    CUDA_SAMPLES_ARCH=89 \
    CCACHE_DIR=/opt/lupine/ccache \
    CCACHE_BASEDIR=/src \
    CCACHE_COMPILERCHECK=content \
    CCACHE_COMPRESS=true \
    PATH=/usr/local/cuda/bin:/opt/venv-pytorch/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      ccache \
      cmake \
      curl \
      freeglut3-dev \
      git \
      libgl-dev \
      libglu1-mesa-dev \
      libnghttp2-dev \
      libx11-dev \
      libxi-dev \
      libxmu-dev \
      ninja-build \
      openssh-client \
      python3 \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv-pytorch \
    && /opt/venv-pytorch/bin/pip install --upgrade pip \
    && /opt/venv-pytorch/bin/pip install numpy \
    && /opt/venv-pytorch/bin/pip install --index-url "$PYTORCH_INDEX_URL" torch

WORKDIR /src
COPY . /src

RUN --mount=type=cache,target=/opt/lupine/ccache \
    cmake -S /src -B "$LUPINE_BUILD_DIR" \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDAToolkit_ROOT="$CUDA_HOME" \
      -DCMAKE_LIBRARY_PATH="$CUDA_HOME/lib64/stubs" \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    && cmake --build "$LUPINE_BUILD_DIR" --parallel \
      --target lupine_driver lupine_nvml lupine_driver_server \
    && ln -sf libcuda.so.1 "$LUPINE_BUILD_DIR/libcuda.so" \
    && ln -sf libnvidia-ml.so.1 "$LUPINE_BUILD_DIR/libnvidia-ml.so" \
    && mkdir -p "$LUPINE_CUSTOM_TEST_BIN_DIR" \
    && for src in /src/test/test_*.cu; do \
         name="$(basename "$src" .cu)"; \
         nvcc --cudart=shared -Wno-deprecated-gpu-targets "$src" \
           -o "$LUPINE_CUSTOM_TEST_BIN_DIR/$name" \
           -lcuda -lcublas -L"$CUDA_HOME/lib64/stubs"; \
       done \
    && CUDA_SAMPLES_REF="$CUDA_SAMPLES_REF" \
       CUDA_SAMPLE_SKIP_LIST="$CUDA_SAMPLE_SKIP_LIST" \
       LUPINE_LIB="$LUPINE_BUILD_DIR/libcuda.so.1" \
       SERVER_LOCAL_BIN="$LUPINE_BUILD_DIR/lupine_driver_server" \
       CUDA_SAMPLES_CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" \
       BUILD_ONLY=1 BUILD_SAMPLES=auto SAMPLE_SUITE=extended \
       /src/test/run_cuda_samples.sh \
    && { ccache --show-stats || true; }
