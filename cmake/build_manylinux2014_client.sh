#!/usr/bin/env bash

set -euo pipefail

mode="${1:-}"
case "$mode" in
  complete)
    cuda_packages=(nvidia-cuda-runtime nvidia-nvml-dev)
    nvml=ON
    ;;
  runtime)
    cuda_packages=(nvidia-cuda-runtime)
    nvml=OFF
    ;;
  *)
    echo "usage: $0 {complete|runtime}" >&2
    exit 2
    ;;
esac

deps=/opt/lupine-deps
headers="$PWD/headers-manylinux2014"
sdk="$PWD/cuda-sdk-manylinux2014"
build="$PWD/build"
cudart_build="$PWD/build-cudart"

cmake -E remove_directory "$headers"
cmake -E remove_directory "$sdk"
cmake -E remove_directory "$build"
cmake -E remove_directory "$cudart_build"
cmake -E make_directory "$headers" "$sdk"

python3 -m pip download \
  --dest "$headers" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation py \
  --python-version 3 \
  --abi none \
  --no-deps \
  "${cuda_packages[@]}"
for wheel in "$headers"/*.whl; do
  unzip -q "$wheel" -d "$sdk"
done

cmake -S . -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLUPINE_BUILD_SERVER=OFF \
  -DLUPINE_BUILD_NVML="$nvml" \
  -DLUPINE_BUILD_HIP=OFF \
  -DCMAKE_PREFIX_PATH="$sdk/nvidia/cu13" \
  -DNGHTTP2_INCLUDE_DIR="$deps/include" \
  -DNGHTTP2_LIBRARY="$deps/lib/libnghttp2.a" \
  -DOPENSSL_ROOT_DIR="$deps" \
  -DOPENSSL_USE_STATIC_LIBS=TRUE \
  -DOPENSSL_SSL_LIBRARY="$deps/lib/libssl.a" \
  -DOPENSSL_CRYPTO_LIBRARY="$deps/lib/libcrypto.a"

targets=(lupine_cuda_client)
if [[ "$mode" == complete ]]; then
  targets+=(lupine_nvml)
fi
cmake --build "$build" --parallel --target "${targets[@]}"

cmake -S python/cudart -B "$cudart_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLUPINE_CUDA_INCLUDE_DIR="$sdk/nvidia/cu13/include" \
  -DLUPINE_CLIENT_DIR="$build"
cmake --build "$cudart_build" --parallel

libraries=(
  "$build/libcuda.so.1"
  "$cudart_build/libcudart.so.13"
)
if [[ "$mode" == complete ]]; then
  libraries+=("$build/libnvidia-ml.so.1")
fi
python3 cmake/verify_manylinux2014.py "${libraries[@]}"
