#!/usr/bin/env bash

set -euo pipefail

if [[ "${AUDITWHEEL_POLICY:-}" != manylinux2014 ]]; then
  echo "this script must run inside a manylinux2014 container" >&2
  exit 2
fi

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

readonly nghttp2_version=1.70.0
readonly nghttp2_sha256=aa317e2cf9dca6afa0aed68f8fad6ff303ec6982e25a78c75c0b65e2b9b3ded5
readonly openssl_version=3.5.8
readonly openssl_sha256=a8f84a39918ec6415ce765d9b429d313ba97b8143169c172e734b9514464f5b2
readonly static_runtime_flags="-static-libgcc -static-libstdc++"
readonly private_archives="libcrypto.a:libgcc.a:libgcc_eh.a:libnghttp2.a:libssl.a:libstdc++.a"

export PATH="/opt/python/cp312-cp312/bin:$PATH"

deps=/opt/lupine-deps
sources="$(mktemp -d)"
headers="$PWD/headers-manylinux2014"
sdk="$PWD/cuda-sdk-manylinux2014"
build="$PWD/build"
cudart_build="$PWD/build-cudart"
trap 'rm -rf "$sources"' EXIT

download() {
  local url="$1"
  local sha256="$2"
  local output="$3"
  curl -fL "$url" -o "$output"
  printf '%s  %s\n' "$sha256" "$output" | sha256sum -c -
}

# OpenSSL's build uses Perl modules outside the manylinux base package.
yum -q install -y perl-core

# Link current, PIC builds of nghttp2 and OpenSSL into the client shims instead
# of depending on copies supplied by the eventual host process.
download \
  "https://github.com/nghttp2/nghttp2/releases/download/v${nghttp2_version}/nghttp2-${nghttp2_version}.tar.gz" \
  "$nghttp2_sha256" \
  "$sources/nghttp2.tar.gz"
tar -C "$sources" -xzf "$sources/nghttp2.tar.gz"
(
  cd "$sources/nghttp2-${nghttp2_version}"
  CFLAGS=-fPIC ./configure \
    --prefix="$deps" \
    --libdir="$deps/lib" \
    --enable-lib-only \
    --enable-static \
    --disable-shared
  make -s -j"$(getconf _NPROCESSORS_ONLN)"
  make -s install
)

download \
  "https://github.com/openssl/openssl/releases/download/openssl-${openssl_version}/openssl-${openssl_version}.tar.gz" \
  "$openssl_sha256" \
  "$sources/openssl.tar.gz"
tar -C "$sources" -xzf "$sources/openssl.tar.gz"
(
  cd "$sources/openssl-${openssl_version}"
  perl ./Configure \
    --prefix="$deps" \
    --openssldir=/etc/ssl \
    --libdir=lib \
    no-module \
    no-shared \
    no-tests \
    no-zlib \
    -fPIC
  make -s -j"$(getconf _NPROCESSORS_ONLN)"
  make -s install_sw
)

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
  "-DCMAKE_SHARED_LINKER_FLAGS=$static_runtime_flags -Wl,--exclude-libs,$private_archives" \
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
  "-DCMAKE_SHARED_LINKER_FLAGS=$static_runtime_flags" \
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
