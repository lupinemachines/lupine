FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 AS cuda-sdk

FROM rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054 AS rocm-sdk

FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim@sha256:0b074d1ae15f5c3f1861354917d356e5afbd5a4c53c1190e81ad2f2add46e45b

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cuda-sdk /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=rocm-sdk /opt/rocm/include/ /opt/rocm/include/

ENV UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /workspace

ENTRYPOINT ["uv", "run", "codegen/codegen.py"]
