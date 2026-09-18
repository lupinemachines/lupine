#!/usr/bin/env bash
# Build and run NVIDIA/CUDALibrarySamples (cuBLAS, cuBLASLt, cuFFT, cuRAND,
# cuSOLVER, cuSPARSE, cuSPARSELt, cuTENSOR, cuEST, nvJPEG, nvJPEG2000, NPP,
# cuPQC, MathDx) through the lupine client shim against a remote server. Every
# leaf directory with a CMakeLists.txt is a standalone CMake project; every
# executable it produces is a unit with its own server on
# SERVER_PORT_BASE + index.
#
# Libraries that ship outside the toolkit (cuDSS, nvCOMP, the *Mp
# multi-process variants) are not selected by default. cuPQC and MathDx are
# exceptions: both are device-side, so they add no host library to shim. cuPQC
# needs CUPQC_HOME pointed at an unpacked SDK; MathDx's package is a tarball
# this script unpacks next to the checkout, and it needs CUDA 13. cuSPARSELt
# ships outside the toolkit too and needs CUSPARSELT_HOME; without one its
# samples stay unbuilt. They call cusparseLtGetErrorString, which 0.6.x does
# not declare and 0.7.x declares but does not export, so they need 0.8.0 or
# newer.
# cuTENSOR ships outside the toolkit as well and needs no shim either: its
# samples run on the client against NVIDIA's own libcutensor and only their
# driver traffic crosses the wire, so this script unpacks its redist archive
# the way it does MathDx's.
# cuEST ships outside the toolkit as well and needs no shim either: its samples
# run on the client against NVIDIA's own libcuest and only their driver traffic
# crosses the wire, so this script unpacks its redist archive the way it does
# MathDx's. Its wheel carries Python bindings only.
# nvJPEG2000 ships outside the toolkit as well and needs no shim either: its
# samples run on the client against NVIDIA's own libnvjpeg2k and only their
# driver traffic crosses the wire, so this script unpacks its redist archive
# the way it does MathDx's.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIBRARY_SAMPLES_URL="${LIBRARY_SAMPLES_URL:-https://github.com/NVIDIA/CUDALibrarySamples.git}"
LIBRARY_SAMPLES_REF="${LIBRARY_SAMPLES_REF:-3437729}"
LIBRARY_SAMPLES_DIR="${LIBRARY_SAMPLES_DIR:-$repo_root/test/cuda-library-samples/CUDALibrarySamples}"
LIBRARY_SAMPLES_BUILD_DIR="${LIBRARY_SAMPLES_BUILD_DIR:-$LIBRARY_SAMPLES_DIR/build}"
LIBRARY_SAMPLES_LIBS="${LIBRARY_SAMPLES_LIBS:-cuBLAS cuBLASLt cuFFT cuRAND cuSOLVER cuSPARSE cuSPARSELt cuTENSOR cuEST nvJPEG nvJPEG2000 NPP cuPQC MathDx}"
LIBRARY_SAMPLES_ARCH="${LIBRARY_SAMPLES_ARCH:-${CUDA_SAMPLES_ARCH:-89}}"
LIBRARY_SAMPLES_CMAKE_ARGS="${LIBRARY_SAMPLES_CMAKE_ARGS:-}"
LIBRARY_SAMPLES_SKIP_LIST="${LIBRARY_SAMPLES_SKIP_LIST:-}"
LIBRARY_SAMPLES_KNOWN_FAILURES="${LIBRARY_SAMPLES_KNOWN_FAILURES:-$repo_root/test/cuda-library-samples/known_failures.txt}"
# 26.06 is the release the samples at this ref require (cuBLASDx 0.7.1,
# cuFFTDx 1.7.3, cuSolverDx 0.5.0) and it is published for CUDA 13 only: the
# newest package with a CUDA 12 build, 25.12.1, is a generation behind every
# one of those minimums, and each example project fatal-errors on a package
# below its own.
MATHDX_URL="${MATHDX_URL:-https://developer.nvidia.com/downloads/compute/cublasdx/redist/cublasdx/cuda13/nvidia-mathdx-26.06.1-cuda13.tar.gz}"
MATHDX_CUDA_MIN_MAJOR=13
MATHDX_DIR="${MATHDX_DIR:-$(dirname "$LIBRARY_SAMPLES_DIR")/mathdx}"
MATHDX_HOME="${MATHDX_HOME:-$MATHDX_DIR/nvidia/mathdx/26.06}"
# The cuTENSOR redist archive, unlike the wheel, carries the unversioned
# libcutensor.so the samples link, so nothing has to be staged for it.
CUTENSOR_VERSION="${CUTENSOR_VERSION:-2.8.0.6}"
CUTENSOR_HOME="${CUTENSOR_HOME:-$(dirname "$LIBRARY_SAMPLES_DIR")/cutensor}"
CUEST_VERSION="${CUEST_VERSION:-0.2.2.2}"
CUEST_HOME="${CUEST_HOME:-$(dirname "$LIBRARY_SAMPLES_DIR")/cuest}"
NVJPEG2K_VERSION="${NVJPEG2K_VERSION:-0.11.0.51}"
NVJPEG2K_HOME="${NVJPEG2K_HOME:-$(dirname "$LIBRARY_SAMPLES_DIR")/nvjpeg2k}"
BUILD_SAMPLES="${BUILD_SAMPLES:-auto}"
BUILD_ONLY="${BUILD_ONLY:-0}"
JOBS="${JOBS:-$(nproc)}"
SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-21000}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-45}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-libsamples-${USER:-lupine}-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"
SERVER_LD_LIBRARY_PATH="${SERVER_LD_LIBRARY_PATH:-}"

LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
CUPQC_HOME="${CUPQC_HOME:-}"
CUSPARSELT_HOME="${CUSPARSELT_HOME:-}"
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-180}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/cuda-library-samples/results/$(date +%Y%m%d-%H%M%S)}"
nvjpeg_assets="${NVJPEG_ASSETS_DIR:-${TMPDIR:-/tmp}/lupine-nvjpeg-assets}"
nvjpeg2k_assets="${NVJPEG2K_ASSETS_DIR:-${TMPDIR:-/tmp}/lupine-nvjpeg2000-assets}"

usage() {
  cat <<EOF
Usage: $0 [sample-dir ...]
Sample dirs are relative to the checkout (cuBLAS/Level-3/gemm). Default: every
leaf CMake project under LIBRARY_SAMPLES_LIBS.
Environment:
  LIBRARY_SAMPLES_DIR        Clone path. Default: $LIBRARY_SAMPLES_DIR
  LIBRARY_SAMPLES_REF        Commit to check out. Default: $LIBRARY_SAMPLES_REF
  LIBRARY_SAMPLES_BUILD_DIR  Build root. Default: <LIBRARY_SAMPLES_DIR>/build
  LIBRARY_SAMPLES_LIBS       Library roots to select. Default: $LIBRARY_SAMPLES_LIBS
  LIBRARY_SAMPLES_ARCH       CMAKE_CUDA_ARCHITECTURES. Default: $LIBRARY_SAMPLES_ARCH
  LIBRARY_SAMPLES_CMAKE_ARGS Extra CMake configure args.
  CUPQC_HOME                 Unpacked cuPQC SDK root. Without it the cuPQC
                             samples have no headers or archives and are dropped.
  LIBRARY_SAMPLES_SKIP_LIST  Comma or space separated units to mark SKIP:disabled.
  LIBRARY_SAMPLES_KNOWN_FAILURES
                             Units to mark SKIP:known. Default: $LIBRARY_SAMPLES_KNOWN_FAILURES
  CUSPARSELT_HOME            cuSPARSELt root holding include/ and lib/. Without
                             one the cuSPARSELt samples are not built.
  MATHDX_URL                 MathDx package to unpack. Default: $MATHDX_URL
  MATHDX_HOME                Unpacked MathDx package. Default: $MATHDX_HOME
  CUTENSOR_HOME              Unpacked cuTENSOR archive. Default: $CUTENSOR_HOME
  CUEST_HOME                 Unpacked cuEST archive. Default: $CUEST_HOME
  NVJPEG2K_HOME              Unpacked nvJPEG2000 archive. Default: $NVJPEG2K_HOME
  BUILD_SAMPLES              auto, 1, or 0. Default: auto (build dirs without a build).
  BUILD_ONLY                 1 to clone/build and exit before running.
  JOBS                       Parallel sample builds. Default: $JOBS
  SERVER_SSH_TARGET          GPU host. Default: $SERVER_SSH_TARGET
  SERVER_PORT_BASE           First per-unit server port. Default: $SERVER_PORT_BASE
  SAMPLE_TIMEOUT             Per-unit timeout in seconds. Default: $SAMPLE_TIMEOUT
  RESULTS_DIR                Output directory. Default: test/cuda-library-samples/results/<timestamp>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cuda_version="$(sed -nE 's/^#define CUDA_VERSION ([0-9]+).*/\1/p' "$CUDA_HOME/include/cuda.h" 2>/dev/null | head -n1 || true)"
if [[ " $LIBRARY_SAMPLES_LIBS " == *" MathDx "* &&
      -n "$cuda_version" && $((cuda_version / 1000)) -lt "$MATHDX_CUDA_MIN_MAJOR" ]]; then
  libs=()
  for lib in $LIBRARY_SAMPLES_LIBS; do
    [[ "$lib" == "MathDx" ]] || libs+=("$lib")
  done
  LIBRARY_SAMPLES_LIBS="${libs[*]}"
  echo "MathDx needs CUDA $MATHDX_CUDA_MIN_MAJOR or newer; its samples contribute no units on $((cuda_version / 1000)).$((cuda_version % 1000 / 10))" >&2
fi

if [[ "$BUILD_SAMPLES" != "0" ]]; then
  mkdir -p "$(dirname "$LIBRARY_SAMPLES_DIR")"
  if [[ ! -d "$LIBRARY_SAMPLES_DIR/.git" ]]; then
    rm -rf "$LIBRARY_SAMPLES_DIR"
    git clone --quiet "$LIBRARY_SAMPLES_URL" "$LIBRARY_SAMPLES_DIR"
  fi
  git config --global --add safe.directory "$LIBRARY_SAMPLES_DIR"
  if [[ "$(git -C "$LIBRARY_SAMPLES_DIR" rev-parse --short=7 HEAD)" != "$(git -C "$LIBRARY_SAMPLES_DIR" rev-parse --short=7 "$LIBRARY_SAMPLES_REF" 2>/dev/null || true)" ]]; then
    git -C "$LIBRARY_SAMPLES_DIR" fetch --quiet origin
    git -C "$LIBRARY_SAMPLES_DIR" checkout --quiet "$LIBRARY_SAMPLES_REF"
  fi
  # The multi-GPU cuFFT samples read the result back before the plan's stream
  # has finished, which returns stale chunks on a busy GPU. Fix proposed in
  # NVIDIA/CUDALibrarySamples#367; drop this once LIBRARY_SAMPLES_REF includes it.
  patch_file="$(mktemp)"
  cat >"$patch_file" <<'PATCH'
--- a/cuFFT/1d_mgpu_c2c/1d_mgpu_c2c_example.cpp
+++ b/cuFFT/1d_mgpu_c2c/1d_mgpu_c2c_example.cpp
@@ -105,6 +105,11 @@
     // Execute the plan
     CUFFT_CALL(cufftXtExecDescriptor(plan, indesc, indesc, CUFFT_FORWARD));
 
+#if CUFFT_VERSION >= 10400
+    // The transform runs on the plan's stream; wait for it before reading back
+    CUDA_RT_CALL(cudaStreamSynchronize(stream));
+#endif
+
     // Copy output data to CPU
     CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(h_data_out.data()),
                              reinterpret_cast<void *>(indesc), CUFFT_COPY_DEVICE_TO_HOST));
--- a/cuFFT/3d_mgpu_c2c/3d_mgpu_c2c_example.cpp
+++ b/cuFFT/3d_mgpu_c2c/3d_mgpu_c2c_example.cpp
@@ -105,6 +105,11 @@
     // Execute the plan
     CUFFT_CALL(cufftXtExecDescriptor(plan, indesc, indesc, CUFFT_FORWARD));
 
+#if CUFFT_VERSION >= 10400
+    // The transform runs on the plan's stream; wait for it before reading back
+    CUDA_RT_CALL(cudaStreamSynchronize(stream));
+#endif
+
     // Copy output data to CPU
     CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(h_data_out.data()),
                              reinterpret_cast<void *>(indesc), CUFFT_COPY_DEVICE_TO_HOST));
--- a/cuFFT/3d_mgpu_r2c_c2r/3d_mgpu_r2c_c2r_example.cpp
+++ b/cuFFT/3d_mgpu_r2c_c2r/3d_mgpu_r2c_c2r_example.cpp
@@ -155,6 +155,11 @@
     // Execute the plan_r2c
     CUFFT_CALL(cufftXtExecDescriptor(plan_r2c, indesc, indesc, CUFFT_FORWARD));
 
+#if CUFFT_VERSION >= 10400
+    // The transform runs on the plan's stream; wait for it before reading back
+    CUDA_RT_CALL(cudaStreamSynchronize(stream));
+#endif
+
     // Scale complex results
     float scale{2.f};
     scaleComplex(indesc, scale, h_data_out.size(), gpus.size());
@@ -162,6 +167,11 @@
     // Execute the plan_c2r
     CUFFT_CALL(cufftXtExecDescriptor(plan_c2r, indesc, indesc, CUFFT_INVERSE));
 
+#if CUFFT_VERSION >= 10400
+    // The transform runs on the plan's stream; wait for it before reading back
+    CUDA_RT_CALL(cudaStreamSynchronize(stream));
+#endif
+
     // Copy output data to CPU
     CUFFT_CALL(cufftXtMemcpy(plan_c2r, (void *)h_data_out.data(), (void *)indesc,
                              CUFFT_COPY_DEVICE_TO_HOST));
PATCH
  if ! git -C "$LIBRARY_SAMPLES_DIR" apply --reverse --check "$patch_file" 2>/dev/null; then
    git -C "$LIBRARY_SAMPLES_DIR" apply "$patch_file"
  fi
  rm -f "$patch_file"

  # cuPQC ships a flat Makefile, not a CMake project, so discovery would not
  # see it. The archives hold LTO-IR only: the samples have to be compiled and
  # linked with -dlto, which CMake emits for INTERPROCEDURAL_OPTIMIZATION.
  if [[ -n "$CUPQC_HOME" ]]; then
    cat >"$LIBRARY_SAMPLES_DIR/cuPQC/CMakeLists.txt" <<'CUPQC_CMAKE'
cmake_minimum_required(VERSION 3.25)
project(cupqc_examples LANGUAGES CXX CUDA)

set(CUPQC_HOME "$ENV{CUPQC_HOME}" CACHE PATH "Unpacked cuPQC SDK root")

file(GLOB cupqc_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(src IN LISTS cupqc_sources)
  get_filename_component(name "${src}" NAME_WE)
  add_executable(${name} "${src}")
  set_target_properties(${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INTERPROCEDURAL_OPTIMIZATION ON)
  target_compile_features(${name} PRIVATE cxx_std_17 cuda_std_17)
  target_include_directories(${name} PRIVATE "${CUPQC_HOME}/include" "${CUPQC_HOME}/include/cupqc")
  target_link_directories(${name} PRIVATE "${CUPQC_HOME}/lib")
  target_link_libraries(${name} PRIVATE cupqc cuhash)
endforeach()
CUPQC_CMAKE
  elif [[ " $LIBRARY_SAMPLES_LIBS " == *" cuPQC "* ]]; then
    echo "cuPQC selected but CUPQC_HOME is unset; its samples contribute no units" >&2
  fi

  # MathDx is headers plus device-side fatbins and static archives, so the
  # package only has to be unpacked. Where it cannot be (no network), the
  # MathDx samples fail to configure and the run reports them
  # SKIP:build-failed like any other.
  if [[ " $LIBRARY_SAMPLES_LIBS " == *" MathDx "* && ! -d "$MATHDX_HOME" ]]; then
    mkdir -p "$MATHDX_DIR"
    curl -fsSL "$MATHDX_URL" | tar -xz -C "$MATHDX_DIR" --strip-components=1 || true
  fi

  # cuTENSOR's archive is CUDA-major-specific. Where it cannot be fetched -- no
  # network, or a major NVIDIA does not build it for -- the cuTENSOR samples
  # fail to configure and the run reports them SKIP:build-failed like any other.
  if [[ " $LIBRARY_SAMPLES_LIBS " == *" cuTENSOR "* && ! -d "$CUTENSOR_HOME" ]]; then
    mkdir -p "$CUTENSOR_HOME"
    curl -fsSL "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-${CUTENSOR_VERSION}_cuda$((${cuda_version:-0} / 1000))-archive.tar.xz" \
      | tar -xJ -C "$CUTENSOR_HOME" --strip-components=1 || true
  fi

  # cuEST's archive is CUDA-major-specific. Where it cannot be fetched -- no
  # network, or a major NVIDIA does not build it for -- the cuEST samples fail
  # to configure and the run reports them SKIP:build-failed like any other.
  if [[ " $LIBRARY_SAMPLES_LIBS " == *" cuEST "* && ! -d "$CUEST_HOME" ]]; then
    mkdir -p "$CUEST_HOME"
    curl -fsSL "https://developer.download.nvidia.com/compute/cuest/redist/libcuest/linux-x86_64/libcuest-linux-x86_64-${CUEST_VERSION}_cuda$((${cuda_version:-0} / 1000))-archive.tar.xz" \
      | tar -xJ -C "$CUEST_HOME" --strip-components=1 || true
  fi

  # nvJPEG2000's archive is CUDA-major-specific. Where it cannot be fetched --
  # no network, or a major NVIDIA does not build it for -- the nvJPEG2000
  # samples fail to configure and the run reports them SKIP:build-failed like
  # any other.
  if [[ " $LIBRARY_SAMPLES_LIBS " == *" nvJPEG2000 "* && ! -d "$NVJPEG2K_HOME" ]]; then
    mkdir -p "$NVJPEG2K_HOME"
    curl -fsSL "https://developer.download.nvidia.com/compute/nvjpeg2000/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-${NVJPEG2K_VERSION}_cuda$((${cuda_version:-0} / 1000))-archive.tar.xz" \
      | tar -xJ -C "$NVJPEG2K_HOME" --strip-components=1 || true
  fi
fi

# A sample is a directory with a CMakeLists.txt and no CMake project beneath
# it; umbrella lists (cuBLASLt/CMakeLists.txt) are skipped so one sample that
# fails to build does not take its siblings with it.
SAMPLES=()
if [[ $# -gt 0 ]]; then
  SAMPLES=("$@")
else
  while IFS= read -r d; do
    SAMPLES+=("$d")
  done < <(
    cd "$LIBRARY_SAMPLES_DIR"
    # MathDx inverts the leaf rule: one project per library builds every
    # example, and the nested projects under it are the LTO-only ones its
    # CUFFTDX_EXAMPLES_LTO option adds, off by default.
    # shellcheck disable=SC2086
    find $LIBRARY_SAMPLES_LIBS -name CMakeLists.txt -printf '%h\n' \
      | grep -vE '(^|/)(cmake|utils)(/|$)' | grep -vE '^MathDx/[^/]+/' \
      | grep -vE '^cuEST/(c_examples/|python_examples)' | sort \
      | awk '{ if (prev != "" && index($0, prev "/") == 1) { skip[prev] = 1 } ; prev = $0; lines[n++] = $0 }
             END { for (i = 0; i < n; i++) if (!(lines[i] in skip)) print lines[i] }'
  )
fi
if [[ "${LIST_TESTS:-0}" == "1" ]]; then
  printf '%s\n' "${SAMPLES[@]}"
  exit 0
fi

# The cuSPARSELt samples link $CUSPARSELT_PATH/lib64/libcusparseLt.so, a layout
# no current release ships: the wheels carry lib/libcusparseLt.so.0 under
# cusparselt/, nvidia/cusparselt/ or nvidia/cu13/. Stage the SONAME under the
# name the samples ask for.
CUSPARSELT_STAGE="$LIBRARY_SAMPLES_BUILD_DIR/cusparselt-root"
if [[ -n "$CUSPARSELT_HOME" ]]; then
  mkdir -p "$CUSPARSELT_STAGE/lib64"
  ln -sfn "$CUSPARSELT_HOME/include" "$CUSPARSELT_STAGE/include"
  ln -sfn "$CUSPARSELT_HOME/lib/libcusparseLt.so.0" \
    "$CUSPARSELT_STAGE/lib64/libcusparseLt.so"
fi

build_sample() {
  local sample="$1"
  local build="$LIBRARY_SAMPLES_BUILD_DIR/$sample"
  local log="$build.build.log"
  mkdir -p "$(dirname "$build")"
  local configure_args=()
  local build_args=()
  if [[ "$sample" == cuSPARSELt/* ]]; then
    if [[ -z "$CUSPARSELT_HOME" ]]; then
      echo "BUILD SKIPPED $sample: set CUSPARSELT_HOME"
      return 1
    fi
    configure_args=(-DCUSPARSELT_PATH="$CUSPARSELT_STAGE")
    # Each cuSPARSELt sample also builds the same source against
    # libcusparseLt_static.a, which no cuSPARSELt wheel ships. Naming the
    # shared executable keeps the missing archive out of the ninja graph.
    build_args=(--target "$(basename "$sample")_example")
  elif [[ "$sample" == MathDx/* ]]; then
    # Each MathDx library keeps its own architecture cache variable, defaulted
    # to 80-real upstream, that CMAKE_CUDA_ARCHITECTURES does not reach; its
    # kernels are the sample, so they have to be built for the server's GPU.
    configure_args=(-Dmathdx_ROOT="$MATHDX_HOME"
      -D"$(basename "$sample" | tr '[:lower:]' '[:upper:]')_CUDA_ARCHITECTURES=$LIBRARY_SAMPLES_ARCH-real")
  elif [[ "$sample" == cuTENSOR ]]; then
    configure_args=(-DCUTENSOR_ROOT="$CUTENSOR_HOME")
  elif [[ "$sample" == cuEST/* ]]; then
    configure_args=(-DCUEST_INCLUDE_DIR="$CUEST_HOME/include"
      -DCUEST_LIB_DIR="$CUEST_HOME/lib")
  elif [[ "$sample" == nvJPEG2000/* ]]; then
    configure_args=(-DNVJPEG2K_PATH="$NVJPEG2K_HOME")
  fi
  # Use the shared runtime by default. Explicit upstream cudart_static links
  # (the nvJPEG multi-instance examples) remain driver/static-runtime coverage.
  # The cuSPARSE lists link bare cudart/cusparse/cuda names and compile host
  # .cpp files that include cuda_fp16.h; LIBRARY_PATH and CPATH are how the
  # toolkit reaches those lines without patching the samples.
  # shellcheck disable=SC2086
  if LIBRARY_PATH="$CUDA_LIB_DIR:$CUDA_LIB_DIR/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
     CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}" \
      cmake -S "$LIBRARY_SAMPLES_DIR/$sample" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
        -DCMAKE_CUDA_ARCHITECTURES="$LIBRARY_SAMPLES_ARCH" \
        -DCUDAToolkit_ROOT="$CUDA_HOME" \
        "${configure_args[@]}" \
        $LIBRARY_SAMPLES_CMAKE_ARGS >"$log" 2>&1 \
     && LIBRARY_PATH="$CUDA_LIB_DIR:$CUDA_LIB_DIR/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
     CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}" \
      cmake --build "$build" --parallel 2 "${build_args[@]}" >>"$log" 2>&1; then
    echo "built $sample"
  else
    echo "BUILD FAILED $sample: $(grep -m1 -iE 'error' "$log" | cut -c1-160)"
    return 1
  fi
}
export -f build_sample
export LIBRARY_SAMPLES_DIR LIBRARY_SAMPLES_BUILD_DIR LIBRARY_SAMPLES_ARCH LIBRARY_SAMPLES_CMAKE_ARGS CUDA_HOME CUDA_LIB_DIR CUPQC_HOME CUSPARSELT_HOME CUSPARSELT_STAGE MATHDX_HOME CUTENSOR_HOME CUEST_HOME NVJPEG2K_HOME

if [[ "$BUILD_SAMPLES" != "0" ]]; then
  to_build=()
  for sample in "${SAMPLES[@]}"; do
    if [[ "$BUILD_SAMPLES" == "1" || ! -f "$LIBRARY_SAMPLES_BUILD_DIR/$sample/build.ninja" ]]; then
      to_build+=("$sample")
    fi
  done
  if [[ ${#to_build[@]} -gt 0 ]]; then
    echo "building ${#to_build[@]} library samples with $JOBS jobs" >&2
    printf '%s\n' "${to_build[@]}" | xargs -P "$JOBS" -I{} bash -c 'build_sample "$1" || true' _ {} >&2
  fi
fi
if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi

LUPINE_LIB_DIR="$(cd "$(dirname "$LUPINE_LIB")" && pwd)"

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done

# Units are <sample>/<executable>; one sample can produce several.
UNITS=()
for sample in "${SAMPLES[@]}"; do
  build="$LIBRARY_SAMPLES_BUILD_DIR/$sample"
  found=0
  if [[ -d "$build" ]]; then
    while IFS= read -r exe; do
      UNITS+=("$sample/$exe")
      found=1
    done < <(cd "$build" && find . -type f -perm -u+x -not -path './CMakeFiles/*' -not -name '*.so*' -not -name '*.a' -printf '%P\n' | sort)
  fi
  [[ "$found" == "1" ]] || UNITS+=("$sample/")
done

# Arguments and working directory for samples that read input files. The
# MultipleInstances samples need the decoder's JPEGs, and the encoder needs
# BMPs that nvjpegDecoder writes with -o, so it runs after the decoder.
unit_argv() {
  local images="$LIBRARY_SAMPLES_DIR/nvJPEG/nvJPEG-Decoder/input_images/"
  # The nvCOMPDx samples compress a file the caller names; their build writes
  # the two its own CTest arguments use, and these mirror them.
  local nvcompdx_data="$LIBRARY_SAMPLES_BUILD_DIR/MathDx/nvCOMPDx"
  # The cuEST examples take a geometry and one or more basis sets on the
  # command line; these are the files and pairings its run_all.sh uses.
  local cuest_data="$LIBRARY_SAMPLES_DIR/cuEST/data"
  local xyz="$cuest_data/geometry/h2o.xyz"
  local xyz_ecp="$cuest_data/geometry/ch2i2.xyz"
  local gbs="$cuest_data/basis_set/def2-svp.gbs"
  local aux_gbs="$cuest_data/basis_set/def2-universal-jkfit.gbs"
  local ecp_gbs="$cuest_data/basis_set/def2-svp-ecp.gbs"
  # The nvJPEG2000 decoders scan their input directory recursively, so one path
  # covers all three .jp2 fixtures; the encoder takes BMPs, which the plain
  # decoder writes with -o, so it runs after it in the sorted unit order.
  local jp2_images="$LIBRARY_SAMPLES_DIR/nvJPEG2000/nvJPEG2000-Decoder/images/"
  case "$1" in
    nvJPEG/nvJPEG-Decoder/*) printf '%s\0' -i "$images" -b 2 -o "$nvjpeg_assets/nvjpeg-decoded" ;;
    nvJPEG/nvJPEG-Decoder-Backend-ROI/*) printf '%s\0' -i "$images" -b 2 ;;
    nvJPEG/nvJPEG-Decoder-MultipleInstances/*) printf '%s\0' -i "$images" -s 2 -j 2 -r 1 ;;
    nvJPEG/nvJPEG-Encoder-MultipleInstances/*) printf '%s\0' -i "$nvjpeg_assets/nvjpeg-decoded/" -s 2 -j 2 -r 1 ;;
    nvJPEG/Image-Resize/*) printf '%s\0' -i "$images" -o "$nvjpeg_assets/nvjpeg-resized" ;;
    nvJPEG/Image-Resize-WaterMark/*) printf '%s\0' -i "$LIBRARY_SAMPLES_DIR/nvJPEG/Image-Resize-WaterMark/input_images/" -o "$nvjpeg_assets/nvjpeg-watermarked" ;;
    NPP/nppCanny/*) printf '%s\0' example_input.png ;;
    MathDx/nvCOMPDx/01_introduction/*) printf '%s\0' -f "$nvcompdx_data/random_file_64KiB.bin" -o "$nvcompdx_data/random_file_64KiB.lz4" ;;
    MathDx/nvCOMPDx/02_lz4_gpu/*) printf '%s\0' -f "$nvcompdx_data/random_file_64KiB.bin" ;;
    MathDx/nvCOMPDx/03_lz4_gpu_and_cpu/*|MathDx/nvCOMPDx/05_lz4_cpu_and_nvrtc/*) printf '%s\0' -f "$nvcompdx_data/random_file_100MiB.bin" ;;
    MathDx/nvCOMPDx/04_ans_gpu/ans_gpu_compression_decompression) printf '%s\0' -t uint8 -f "$nvcompdx_data/random_file_100MiB.bin" ;;
    cuEST/*/xc_grid_general) printf '%s\0' "$xyz" ;;
    cuEST/*/ao_basis_general) printf '%s\0' "$xyz" "$gbs" ;;
    cuEST/*/ecp_atom) printf '%s\0' "$xyz_ecp" "$ecp_gbs" ;;
    cuEST/*/3_density_fitting/*) printf '%s\0' "$xyz" "$gbs" "$aux_gbs" ;;
    cuEST/*/5_effective_core_potentials/*) printf '%s\0' "$xyz_ecp" "$gbs" "$ecp_gbs" ;;
    cuEST/*/2_one_electron_integrals/*|cuEST/*/4_exchange_correlation/*|cuEST/*/6_pcm/*)
      printf '%s\0' "$xyz" "$gbs" ;;
    nvJPEG2000/nvJPEG2000-Decoder/*) printf '%s\0' -i "$jp2_images" -b 2 -o "$nvjpeg2k_assets/decoded" ;;
    nvJPEG2000/nvJPEG2000-Decoder-Pipelined/*|nvJPEG2000/nvJPEG2000-Decoder-Tile-Partial/*) printf '%s\0' -i "$jp2_images" -b 2 ;;
    nvJPEG2000/nvJPEG2000-Encoder/*) printf '%s\0' -i "$nvjpeg2k_assets/decoded" -b 2 -o "$nvjpeg2k_assets/encoded" ;;
  esac
}
# NPP samples open their inputs relative to the working directory: findContour
# and nppCanny from the sample dir, the rest via ../images/; the watermark
# sample reads NVLogo.jpg from the sample dir.
unit_cwd() {
  case "$1" in
    NPP/findContour/*|NPP/nppCanny/*|nvJPEG/Image-Resize-WaterMark/*) printf '%s\n' "$LIBRARY_SAMPLES_DIR/$(dirname "$1")" ;;
    NPP/*) printf '%s\n' "$LIBRARY_SAMPLES_DIR/$(dirname "$1")/images" ;;
    *) printf '%s\n' "$LIBRARY_SAMPLES_BUILD_DIR/$(dirname "$1")" ;;
  esac
}

known_failures=()
if [[ -f "$LIBRARY_SAMPLES_KNOWN_FAILURES" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -n "$line" ]] && known_failures+=("$line")
  done < "$LIBRARY_SAMPLES_KNOWN_FAILURES"
fi

# shellcheck source=test/integration/remote_server.sh
source "$repo_root/test/integration/remote_server.sh"

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi

cleanup_remote_bin() {
  if [[ "$SERVER_UPLOAD" == "1" && "$SERVER_REMOTE_CLEANUP" == "1" ]]; then
    ssh_with_timeout "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_bin EXIT

in_list() {
  local needle="$1"
  shift
  local item=""
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

mkdir -p "$RESULTS_DIR" \
  "$nvjpeg_assets/nvjpeg-decoded" \
  "$nvjpeg_assets/nvjpeg-resized" \
  "$nvjpeg_assets/nvjpeg-watermarked" \
  "$nvjpeg2k_assets/decoded" \
  "$nvjpeg2k_assets/encoded"
tsv="$RESULTS_DIR/results.tsv"
: > "$tsv"
pass=0
fail=0
skip=0
# shellcheck disable=SC2206
disabled=(${LIBRARY_SAMPLES_SKIP_LIST//,/ })

echo "CUDALibrarySamples $LIBRARY_SAMPLES_REF, ${#SAMPLES[@]} samples, ${#UNITS[@]} units" >&2

for i in "${!UNITS[@]}"; do
  unit="${UNITS[$i]}"
  # Units run one server at a time, so ports only have to differ from their
  # neighbours; rotating inside ten keeps a sample with many executables --
  # every MathDx one -- within the window test/integration/CMakeLists.txt
  # allocates it, instead of walking into the next entry's.
  port=$((SERVER_PORT_BASE + i % 10))
  log="$RESULTS_DIR/${unit//\//_}.log"
  server_log="/tmp/lupine-libsamples-$port.log"
  pidfile="/tmp/lupine-libsamples-$port.pid"
  unit_start_seconds="$SECONDS"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] library sample $((i + 1))/${#UNITS[@]}: $unit" >&2

  if [[ "$unit" == */ ]]; then
    skip=$((skip + 1))
    signature="$(grep -m1 -iE 'error' "$LIBRARY_SAMPLES_BUILD_DIR/${unit%/}.build.log" 2>/dev/null | cut -c1-200 || true)"
    printf '%s\t%s\t%s\n' "${unit%/}" "SKIP:build-failed" "$signature" | tee -a "$tsv"
    continue
  fi
  if in_list "$unit" "${disabled[@]}"; then
    skip=$((skip + 1))
    printf '%s\t%s\t%s\n' "$unit" "SKIP:disabled" "disabled by LIBRARY_SAMPLES_SKIP_LIST" | tee -a "$tsv"
    continue
  fi
  if in_list "$unit" "${known_failures[@]}"; then
    skip=$((skip + 1))
    printf '%s\t%s\t%s\n' "$unit" "SKIP:known" "listed in $(basename "$LIBRARY_SAMPLES_KNOWN_FAILURES")" | tee -a "$tsv"
    continue
  fi

  argv=()
  while IFS= read -r -d '' a; do
    argv+=("$a")
  done < <(unit_argv "$unit")
  cwd="$(unit_cwd "$unit")"
  exe="$LIBRARY_SAMPLES_BUILD_DIR/$unit"

  start_remote_server "$pidfile" "$server_log" "$port"

  set +e
  (
    cd "$cwd"
    timeout --kill-after=5s "$SAMPLE_TIMEOUT" env \
      LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${CUSPARSELT_HOME:+$CUSPARSELT_HOME/lib:}$CUTENSOR_HOME/lib:$CUEST_HOME/lib:$NVJPEG2K_HOME/lib:${LD_LIBRARY_PATH:-}" \
      LUPINE_SERVER="$SERVER_HOST:$port" \
      LD_PRELOAD="$LUPINE_LIB" \
      "$exe" "${argv[@]}"
  ) >"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  # cuSPARSE and cuFFT samples report wrong results as "test FAILED" /
  # "FAILED with L2 error" but still exit 0.
  if [[ "$rc" == "0" ]] && grep -qE '\bFAILED\b' "$log"; then
    status="FAIL:output"
    fail=$((fail + 1))
  elif [[ "$rc" == "0" ]]; then
    status="PASS"
    pass=$((pass + 1))
  else
    status="FAIL:$rc"
    fail=$((fail + 1))
  fi
  signature="$({ grep -iE 'LUPINE|error|fail|unsupported|not supported|Aborted|Segmentation' "$log" || true; } | head -n3 | tr '\n' ' ')"
  [[ -n "$signature" ]] || signature="$(tail -n1 "$log" | tr -d '\0')"
  signature="$(printf '%s' "$signature" | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
  if [[ -z "$signature" && "$rc" == "124" ]]; then
    signature="timed out after ${SAMPLE_TIMEOUT}s"
  fi
  printf '%s\t%s\t%s\n' "$unit" "$status" "$signature" | tee -a "$tsv"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] library sample $unit -> $status in $((SECONDS - unit_start_seconds))s" >&2
done

{
  echo "PASS $pass"
  echo "FAIL $fail"
  echo "SKIP $skip"
  echo "TOTAL $((pass + fail + skip))"
  echo "RESULTS $tsv"
} | tee "$RESULTS_DIR/summary.txt"

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
