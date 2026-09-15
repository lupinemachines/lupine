FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 AS cuda-sdk

FROM rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054 AS rocm-sdk

# cuDNN ships outside the toolkit. Its header package is pinned so the
# annotation file always preprocesses against the same declarations.
FROM cuda-sdk AS cudnn-sdk
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcudnn9-headers-cuda-13=9.26.0.51-1 \
    && mkdir -p /opt/cudnn/include \
    && cp -L /usr/include/*-linux-gnu/cudnn*.h /opt/cudnn/include/ \
    && rm -rf /var/lib/apt/lists/*

# NCCL ships outside the toolkit too; the devel image's copy is replaced by a
# pinned release for the same reason.
FROM cuda-sdk AS nccl-sdk
RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
         libnccl2=2.31.2-1+cuda13.3 libnccl-dev=2.31.2-1+cuda13.3 \
    && mkdir -p /opt/nccl/include \
    && cp -L /usr/include/nccl.h /opt/nccl/include/ \
    && rm -rf /var/lib/apt/lists/*

FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim@sha256:0b074d1ae15f5c3f1861354917d356e5afbd5a4c53c1190e81ad2f2add46e45b

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cuda-sdk /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=cudnn-sdk /opt/cudnn/include/ /usr/local/cuda/include/
COPY --from=nccl-sdk /opt/nccl/include/ /usr/local/cuda/include/
COPY --from=rocm-sdk /opt/rocm/include/ /opt/rocm/include/

ENV UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /workspace

ENTRYPOINT ["uv", "run", "codegen/codegen.py"]
