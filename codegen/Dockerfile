FROM ghcr.io/astral-sh/uv:0.9.18@sha256:5713fa8217f92b80223bc83aac7db36ec80a84437dbc0d04bbc659cae030d8c9 AS uv

FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON=3.12.3 \
    UV_PYTHON_INSTALL_DIR=/tmp/uv-python \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /workspace

ENTRYPOINT ["uv", "run", "codegen/codegen.py"]
