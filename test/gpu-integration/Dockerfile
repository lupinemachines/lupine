# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDA_SAMPLE_SKIP_LIST=""
ARG CUDA_SAMPLES_REF=""

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    CUDA_LIB_DIR=/usr/local/cuda/lib64 \
    CUDA_SAMPLES_DIR=/opt/cuda-samples-src \
    CUDA_SAMPLES_BUILD_DIR=/opt/cuda-samples-build \
    CUDA_SAMPLES_ARCH=89 \
    CCACHE_DIR=/opt/lupine/ccache \
    CCACHE_BASEDIR=/src \
    CCACHE_COMPILERCHECK=content \
    CCACHE_COMPRESS=true \
    PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      ccache \
      cmake \
      curl \
      freeglut3-dev \
      git \
      libfreeimage-dev \
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

WORKDIR /src
COPY . /src

RUN --mount=type=cache,target=/opt/lupine/ccache \
    CUDA_SAMPLES_REF="$CUDA_SAMPLES_REF" \
       CUDA_SAMPLE_SKIP_LIST="$CUDA_SAMPLE_SKIP_LIST" \
       CUDA_SAMPLES_CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" \
       BUILD_ONLY=1 BUILD_SAMPLES=auto SAMPLE_SUITE=extended \
       /src/test/run_cuda_samples.sh \
    && { ccache --show-stats || true; }
