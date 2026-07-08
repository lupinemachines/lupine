#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_SAMPLES_URL="${CUDA_SAMPLES_URL:-https://github.com/NVIDIA/cuda-samples.git}"
CUDA_SAMPLES_REF="${CUDA_SAMPLES_REF:-}"
DEFAULT_CUDA_SAMPLES_DIR="$repo_root/test/cuda-samples/cuda-samples"
if [[ -n "${HOME:-}" && -d "$HOME/cuda-samples/.git" ]]; then
  DEFAULT_CUDA_SAMPLES_DIR="$HOME/cuda-samples"
fi
CUDA_SAMPLES_DIR="${CUDA_SAMPLES_DIR:-$DEFAULT_CUDA_SAMPLES_DIR}"
CUDA_SAMPLES_BUILD_DIR="${CUDA_SAMPLES_BUILD_DIR:-$CUDA_SAMPLES_DIR/build}"
CUDA_SAMPLES_BIN="${CUDA_SAMPLES_BIN:-}"
CUDA_SAMPLES_CMAKE_ARGS="${CUDA_SAMPLES_CMAKE_ARGS:-}"
CUDA_SAMPLES_ARCH="${CUDA_SAMPLES_ARCH:-}"
BUILD_SAMPLES="${BUILD_SAMPLES:-auto}"
BUILD_ONLY="${BUILD_ONLY:-0}"
JOBS="${JOBS:-$(nproc)}"
CUDA_SAMPLE_JOBS="${CUDA_SAMPLE_JOBS:-1}"
SAMPLE_SUITE="${SAMPLE_SUITE:-compliance}"

SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-14900}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-45}"
CUDA_SAMPLE_SKIP_LIST="${CUDA_SAMPLE_SKIP_LIST:-}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-lupine-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"

LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-120}"
LONG_SAMPLE_TIMEOUT="${LONG_SAMPLE_TIMEOUT:-600}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/cuda-samples/results/$(date +%Y%m%d-%H%M%S)}"

CORE_SAMPLES=(
  deviceQuery deviceQueryDrv topologyQuery
  vectorAdd vectorAddDrv vectorAdd_nvrtc
  asyncAPI cudaOpenMP clock clock_nvrtc matrixMul matrixMulDrv matrixMul_nvrtc
  inlinePTX inlinePTX_nvrtc
  simpleAssert simpleAssert_nvrtc
  simpleAttributes simpleCallback simpleDrvRuntime simplePrintf simpleTemplates
  simpleAtomicIntrinsics simpleAtomicIntrinsics_nvrtc simpleStreams simpleMultiCopy simpleMultiGPU
  simpleOccupancy simpleCooperativeGroups simpleIPC
  FunctionPointers
  simpleCubemapTexture simpleLayeredTexture simpleSurfaceWrite
  simpleTexture simpleTextureDrv simplePitchLinearTexture
  mergeSort reduction reductionMultiBlockCG scan sortingNetworks histogram scalarProd transpose
  BlackScholes BlackScholes_nvrtc binomialOptions binomialOptions_nvrtc SobolQRNG quasirandomGenerator
  quasirandomGenerator_nvrtc
  simpleCudaGraphs streamOrderedAllocation cudaCompressibleMemory simpleZeroCopy alignedTypes LargeKernelParameter
  vectorAddMMAP
  simple simpleHyperQ simpleVoteIntrinsics simpleAWBarrier binaryPartitionCG
  globalToShmemAsyncCopy shfl_scan threadFenceReduction warpAggregatedAtomicsCG
  cdpSimplePrint cdpSimpleQuicksort cdpAdvancedQuicksort cdpQuadtree cdpBezierTessellation
  newdelete
  StreamPriorities
  cudaTensorCoreGemm tf32TensorCoreGemm bf16TensorCoreGemm fp16ScalarProduct
  dmmaTensorCoreGemm immaTensorCoreGemm
  convolutionFFT2D convolutionSeparable convolutionTexture dwtHaar1D dxtc eigenvalues fastWalshTransform FDTD3d
  HSOpticalFlow
  MC_EstimatePiP MC_EstimatePiQ MC_EstimatePiInlineP MC_EstimatePiInlineQ
  MC_SingleAsianOptionP MonteCarloMultiGPU
  cudaGraphsPerfScaling graphConditionalNodes graphMemoryNodes graphMemoryFootprint jacobiCudaGraphs
  dct8x8 lineOfSight nbody recursiveGaussian stereoDisparity
  Mandelbrot SobelFilter bicubicTexture bilateralFilter bindlessTexture boxFilter
  imageDenoising marchingCubes particles simpleGL smokeParticles
  simpleTexture3D volumeFiltering volumeRender
  radixSortThrust segmentationTreeThrust template interval
  ptxgen ptxjit matrixMulDynlinkJIT threadMigration
)

LIBRARY_SAMPLES=(
  simpleCUBLAS matrixMulCUBLAS simpleCUBLAS_LU batchCUBLAS
  simpleCUFFT simpleCUFFT_callback
  oceanFFT
  cuSolverDn_LinearSolver cuSolverSp_LinearSolver cuSolverSp_LowlevelCholesky
  cuSolverSp_LowlevelQR cuSolverRf
  conjugateGradient conjugateGradientPrecond
  conjugateGradientCudaGraphs conjugateGradientUM conjugateGradientMultiBlockCG
  MersenneTwisterGP11213
  nvJPEG nvJPEG_encoder
  NV12toBGRandResize
  randomFog
  jitLto
  watershedSegmentationNPP
  boxFilterNPP
  cannyEdgeDetectorNPP
  FilterBorderControlNPP
)

DEFAULT_SAMPLES=(
  "${CORE_SAMPLES[@]}"
  "${LIBRARY_SAMPLES[@]}"
)

usage() {
  cat <<EOF
Usage: $0 [sample ...]

Environment:
  CUDA_SAMPLES_DIR     Clone/work tree path. Default: $DEFAULT_CUDA_SAMPLES_DIR
  CUDA_SAMPLES_BUILD_DIR CMake build path. Default: <CUDA_SAMPLES_DIR>/build.
  CUDA_SAMPLES_BIN     Legacy executable path override. CMake builds are resolved per sample.
  CUDA_SAMPLES_CMAKE_ARGS Extra args passed to CMake configure for CUDA 13 samples.
  CUDA_SAMPLES_ARCH    Compile device code only for this compute capability
                       (e.g. 75). Samples whose hardcoded arch list excludes it
                       keep their upstream list.
  CUDA_SAMPLES_REF     Optional branch/tag/commit to checkout after clone.
  BUILD_SAMPLES        auto, 1, or 0. Default: auto.
  BUILD_ONLY           1 to clone/build selected samples and exit before running.
  CUDA_SAMPLE_JOBS     Number of samples to run concurrently. Default: $CUDA_SAMPLE_JOBS.
  SAMPLE_SUITE         compliance, core, libraries, or extended when no samples are given.
                       Default: compliance.
  SAMPLE_TIMEOUT       Default per-sample execution timeout in seconds. Default: $SAMPLE_TIMEOUT.
  LONG_SAMPLE_TIMEOUT  Timeout for known long-running compliance samples. Default: $LONG_SAMPLE_TIMEOUT.
  SERVER_SSH_TARGET    GPU host SSH target. Default: kevin@inferable-node-008.
  SERVER_PORT_BASE     First per-sample server port. Default: 14900.
  SSH_COMMAND_TIMEOUT  Timeout for each SSH/SCP control command. Default: $SSH_COMMAND_TIMEOUT.
  CUDA_SAMPLE_SKIP_LIST Comma or space separated samples to mark SKIP:disabled.
  LUPINE_LIB            Client shim. Default: $repo_root/build/libcuda.so.1.
  RESULTS_DIR          Output directory. Default: test/cuda-samples/results/<timestamp>.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$BUILD_ONLY" != "1" ]]; then
  if [[ ! -x "$LUPINE_LIB" ]]; then
    echo "missing shim: $LUPINE_LIB" >&2
    exit 1
  fi

  runtime_exports="$(nm -D --defined-only "$LUPINE_LIB" | awk '{print $3}' | grep -E '^cuda' || true)"
  if [[ -n "$runtime_exports" ]]; then
    echo "shim exports CUDA Runtime API symbols; keep this driver-only:" >&2
    echo "$runtime_exports" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$CUDA_SAMPLES_DIR")" "$RESULTS_DIR"

if [[ ! -d "$CUDA_SAMPLES_DIR/.git" ]]; then
  if [[ -e "$CUDA_SAMPLES_DIR" ]]; then
    rm -rf "$CUDA_SAMPLES_DIR"
  fi
  git clone "$CUDA_SAMPLES_URL" "$CUDA_SAMPLES_DIR"
fi
git config --global --add safe.directory "$CUDA_SAMPLES_DIR"

detect_cuda_samples_ref() {
  local release=""
  local major=""
  local minor=""
  local patch=""

  if command -v nvcc >/dev/null 2>&1; then
    release="$(nvcc --version | sed -nE 's/.*release ([0-9]+)\.([0-9]+)(\.([0-9]+))?.*/\1 \2 \4/p' | head -n1)"
  fi
  if [[ -z "$release" && -f /usr/local/cuda/version.json ]]; then
    release="$(sed -nE 's/.*"cuda"[^0-9]*([0-9]+)\.([0-9]+)(\.([0-9]+))?.*/\1 \2 \4/p' /usr/local/cuda/version.json | head -n1)"
  fi
  if [[ -z "$release" ]]; then
    return 0
  fi

  read -r major minor patch <<<"$release"
  case "$major.$minor.$patch" in
    12.4.1)
      printf '%s\n' v12.4.1
      ;;
    11.8.*)
      printf '%s\n' v11.8
      ;;
    11.7.*)
      printf '%s\n' v11.6
      ;;
    12.4.*)
      printf '%s\n' v12.4
      ;;
    12.5.*|12.6.*|12.7.*)
      printf '%s\n' v12.5
      ;;
    12.8.*)
      printf '%s\n' v12.8
      ;;
    12.9.*)
      printf '%s\n' v12.9
      ;;
    13.0.*)
      printf '%s\n' v13.0
      ;;
    13.1.*)
      printf '%s\n' v13.1
      ;;
  esac
}

if [[ -z "$CUDA_SAMPLES_REF" ]]; then
  CUDA_SAMPLES_REF="$(detect_cuda_samples_ref)"
fi

if [[ -n "$CUDA_SAMPLES_REF" ]]; then
  git -C "$CUDA_SAMPLES_DIR" fetch --tags origin
  git -C "$CUDA_SAMPLES_DIR" checkout "$CUDA_SAMPLES_REF"
  git -C "$CUDA_SAMPLES_DIR" reset --hard "$CUDA_SAMPLES_REF"
fi

# The samples hardcode set(CMAKE_CUDA_ARCHITECTURES ...) per CMakeLists, so a
# -D flag cannot narrow them. Rewrite lists that include the requested arch to
# just that arch; samples restricted to other archs are left as upstream built
# them.
if [[ -n "$CUDA_SAMPLES_ARCH" ]]; then
  find "$CUDA_SAMPLES_DIR" -name CMakeLists.txt -exec sed -i -E \
    "s/set\(CMAKE_CUDA_ARCHITECTURES ([0-9 ]* )?$CUDA_SAMPLES_ARCH( [0-9 ]*)?\)/set(CMAKE_CUDA_ARCHITECTURES $CUDA_SAMPLES_ARCH)/" {} +
fi

cmake_samples=0
if [[ -f "$CUDA_SAMPLES_DIR/CMakeLists.txt" && ( -d "$CUDA_SAMPLES_DIR/cpp" || -d "$CUDA_SAMPLES_DIR/Samples" ) ]]; then
  cmake_samples=1
fi

if [[ -z "$CUDA_SAMPLES_BIN" ]]; then
  if [[ "$cmake_samples" == "1" ]]; then
    CUDA_SAMPLES_BIN="$CUDA_SAMPLES_BUILD_DIR"
  else
    CUDA_SAMPLES_BIN="$CUDA_SAMPLES_DIR/bin/x86_64/linux/release"
  fi
fi

resolve_sample_exe() {
  local sample="$1"
  local exe=""

  if [[ -x "$CUDA_SAMPLES_BIN/$sample" && ! -d "$CUDA_SAMPLES_BIN/$sample" ]]; then
    printf '%s\n' "$CUDA_SAMPLES_BIN/$sample"
    return 0
  fi

  if [[ "$cmake_samples" == "1" ]]; then
    exe="$(find "$CUDA_SAMPLES_BUILD_DIR" -type f -name "$sample" -perm /111 2>/dev/null | head -n1 || true)"
    if [[ -n "$exe" ]]; then
      printf '%s\n' "$exe"
      return 0
    fi
  fi

  return 1
}

resolve_sample_srcdir() {
  local sample="$1"
  local dir=""

  if [[ -d "$CUDA_SAMPLES_DIR/Samples" ]]; then
    dir="$(find "$CUDA_SAMPLES_DIR/Samples" -mindepth 2 -maxdepth 2 -type d -name "$sample" 2>/dev/null | head -n1 || true)"
    if [[ -n "$dir" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
  fi

  if [[ -d "$CUDA_SAMPLES_DIR/cpp" ]]; then
    dir="$(find "$CUDA_SAMPLES_DIR/cpp" -mindepth 2 -maxdepth 2 -type d -name "$sample" 2>/dev/null | head -n1 || true)"
    if [[ -n "$dir" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
  fi

  return 1
}

sample_args() {
  local sample="$1"

  case "$sample" in
    FDTD3d)
      printf '%s\0' --qatest
      ;;
    batchCUBLAS)
      printf '%s\0' -m32 -n32 -k32 -N1
      ;;
    cuSolverRf)
      printf '%s\0' "-file=$(resolve_sample_srcdir cuSolverRf)/lap2D_5pt_n100.mtx"
      ;;
    bicubicTexture)
      printf '%s\0' -mode=0 -file=data/0_nearest.ppm
      ;;
    bilateralFilter)
      printf '%s\0' -radius=5 -file=data/ref_05.ppm
      ;;
    bindlessTexture)
      printf '%s\0' -file=data/ref_bindlessTexture.bin
      ;;
    boxFilter)
      printf '%s\0' -radius=14 -file=data/ref_14.ppm
      ;;
    eigenvalues)
      printf '%s\0' -matrix-size=128 -iters-timing=1
      ;;
    FunctionPointers)
      printf '%s\0' -mode=0 -file=data/ref_orig.pgm
      ;;
    imageDenoising)
      printf '%s\0' -kernel=0 -file=data/ref_passthru.ppm
      ;;
    Mandelbrot)
      printf '%s\0' -mode=0 -file=data/Mandelbrot_fp32.ppm
      ;;
    marchingCubes)
      printf '%s\0' -dump=0 -file=data/posArray.bin
      ;;
    matrixMul|matrixMul_nvrtc)
      printf '%s\0' -wA=32 -hA=32 -wB=32 -hB=32
      ;;
    matrixMulCUBLAS)
      printf '%s\0' -sizemult=1
      ;;
    nbody)
      printf '%s\0' -benchmark -numbodies=4096 -i=1
      ;;
    NV12toBGRandResize)
      printf '%s\0' \
        -input=data/test640x480.nv12 \
        -width=640 \
        -height=480 \
        -dst_width=320 \
        -dst_height=240 \
        -batch=1
      ;;
    oceanFFT)
      printf '%s\0' -qatest
      ;;
    particles)
      printf '%s\0' -file=data/ref_particles.bin
      ;;
    ptxgen)
      printf '%s\0' test.ll
      ;;
    randomFog)
      printf '%s\0' -qatest
      ;;
    reduction|threadFenceReduction)
      printf '%s\0' -n=1024 -threads=64 -maxblocks=16
      ;;
    recursiveGaussian)
      printf '%s\0' -benchmark
      ;;
    simpleGL)
      printf '%s\0' -file=data/ref_simpleGL.bin
      ;;
    smokeParticles)
      printf '%s\0' -qatest
      ;;
    SobelFilter)
      printf '%s\0' -mode=0 -file=data/ref_orig.pgm
      ;;
    simpleTexture3D)
      printf '%s\0' -file=data/ref_texture3D.bin
      ;;
    transpose)
      printf '%s\0' -dimX=512 -dimY=512
      ;;
    volumeFiltering)
      printf '%s\0' -file=data/ref_volumefilter.ppm
      ;;
    volumeRender)
      printf '%s\0' --file=ref_volume.ppm
      ;;
  esac
}

sample_workdir() {
  local sample="$1"
  local sample_exe="$2"

  case "$sample" in
    bicubicTexture|bilateralFilter|bindlessTexture|boxFilter|FunctionPointers|\
      imageDenoising|Mandelbrot|marchingCubes|nbody|NV12toBGRandResize|\
      oceanFFT|particles|ptxgen|randomFog|recursiveGaussian|simpleGL|\
      smokeParticles|SobelFilter|simpleTexture3D|volumeFiltering|volumeRender)
      printf '%s\n' "$(resolve_sample_srcdir "$sample")"
      return 0
      ;;
  esac

  dirname "$sample_exe"
}

prepare_sample_runtime_files() {
  local sample="$1"
  local sample_cwd="$2"
  local sample_srcdir=""

  case "$sample" in
    *_nvrtc)
      sample_srcdir="$(resolve_sample_srcdir "$sample" || true)"
      for dir in "$sample_cwd" "$sample_srcdir"; do
        [[ -n "$dir" ]] || continue
        if [[ -d "$CUDA_HOME/include/nv" && ! -e "$dir/nv" ]]; then
          cp -a "$CUDA_HOME/include/nv" "$dir/nv"
        fi
        if [[ -d "$CUDA_HOME/include/cuda" && ! -e "$dir/cuda" ]]; then
          cp -a "$CUDA_HOME/include/cuda" "$dir/cuda"
        fi
      done
      ;;
  esac
}

explicit_samples=0
build_full_sample_tree=0
samples=("$@")
if [[ ${#samples[@]} -eq 0 ]]; then
  case "$SAMPLE_SUITE" in
    compliance)
      samples=("${DEFAULT_SAMPLES[@]}")
      ;;
    extended|all|default)
      samples=("${DEFAULT_SAMPLES[@]}")
      build_full_sample_tree=1
      ;;
    core)
      samples=("${CORE_SAMPLES[@]}")
      ;;
    libraries|library)
      samples=("${LIBRARY_SAMPLES[@]}")
      ;;
    *)
      echo "unknown SAMPLE_SUITE: $SAMPLE_SUITE" >&2
      echo "expected one of: compliance, core, libraries, extended" >&2
      exit 1
      ;;
  esac
else
  explicit_samples=1
fi
selected_sample_build=0
if [[ "$build_full_sample_tree" != "1" && ${#samples[@]} -gt 0 ]]; then
  selected_sample_build=1
fi

needs_build=0
if [[ "$BUILD_SAMPLES" == "1" ]]; then
  needs_build=1
elif [[ "$BUILD_SAMPLES" == "auto" ]]; then
  for sample in "${samples[@]}"; do
    if ! resolve_sample_exe "$sample" >/dev/null; then
      needs_build=1
      break
    fi
  done
fi

if [[ "$needs_build" == "1" ]]; then
  if [[ "$cmake_samples" == "1" ]]; then
    if [[ "$selected_sample_build" == "1" ]]; then
      for sample in "${samples[@]}"; do
        sample_srcdir="$(resolve_sample_srcdir "$sample" || true)"
        if [[ -z "$sample_srcdir" ]]; then
          echo "missing sample source dir: $sample" >&2
          continue
        fi
        sample_build_dir="$CUDA_SAMPLES_BUILD_DIR/selected/$sample"
        build_log="$RESULTS_DIR/.build-$sample.log"
        : > "$build_log"
        if [[ ! -f "$sample_build_dir/CMakeCache.txt" ]]; then
          # shellcheck disable=SC2086
          cmake -S "$sample_srcdir" -B "$sample_build_dir" -DCMAKE_BUILD_TYPE=Release $CUDA_SAMPLES_CMAKE_ARGS >>"$build_log" 2>&1
        fi
        cmake --build "$sample_build_dir" --parallel "$JOBS" --target "$sample" >>"$build_log" 2>&1 || true
      done
    else
      if [[ ! -f "$CUDA_SAMPLES_BUILD_DIR/CMakeCache.txt" ]]; then
        # shellcheck disable=SC2086
        cmake -S "$CUDA_SAMPLES_DIR" -B "$CUDA_SAMPLES_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $CUDA_SAMPLES_CMAKE_ARGS
      fi
      cmake --build "$CUDA_SAMPLES_BUILD_DIR" --parallel "$JOBS"
    fi
  else
    if [[ "$selected_sample_build" == "1" ]]; then
      for sample in "${samples[@]}"; do
        sample_srcdir="$(resolve_sample_srcdir "$sample" || true)"
        if [[ -z "$sample_srcdir" ]]; then
          echo "missing sample source dir: $sample" >&2
          continue
        fi
        if [[ ! -f "$sample_srcdir/Makefile" && ! -f "$sample_srcdir/makefile" && ! -f "$sample_srcdir/GNUmakefile" ]]; then
          echo "missing sample Makefile: $sample" >&2
          continue
        fi
        make -C "$sample_srcdir" -j"$JOBS" || echo "sample build failed: $sample" >&2
      done
    else
      make -C "$CUDA_SAMPLES_DIR" -j"$JOBS"
    fi
  fi
fi

if [[ "$BUILD_ONLY" == "1" ]]; then
  exit 0
fi

case "$CUDA_SAMPLE_JOBS" in
  ''|*[!0-9]*)
    echo "CUDA_SAMPLE_JOBS must be a positive integer: $CUDA_SAMPLE_JOBS" >&2
    exit 1
    ;;
esac
if (( CUDA_SAMPLE_JOBS < 1 )); then
  echo "CUDA_SAMPLE_JOBS must be a positive integer: $CUDA_SAMPLE_JOBS" >&2
  exit 1
fi

if [[ ! -x "$SERVER_LOCAL_BIN" ]]; then
  echo "missing server binary: $SERVER_LOCAL_BIN" >&2
  exit 1
fi

ssh_with_timeout() {
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "$@"
}

sample_disabled() {
  local sample="$1"
  local disabled=""

  # shellcheck disable=SC2206
  local disabled_samples=(${CUDA_SAMPLE_SKIP_LIST//,/ })
  for disabled in "${disabled_samples[@]}"; do
    if [[ "$disabled" == "$sample" ]]; then
      return 0
    fi
  done

  return 1
}

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi

cleanup_remote_bin() {
  if [[ "$SERVER_UPLOAD" == "1" && "$SERVER_REMOTE_CLEANUP" == "1" ]]; then
    ssh_with_timeout \
      "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_bin EXIT

stop_remote_server() {
  local pidfile="$1"
  local server_log="$2"

  ssh_with_timeout "
    if [ -f '$pidfile' ]; then
      pid=\$(cat '$pidfile' 2>/dev/null || true)
      if [ -n \"\$pid\" ]; then
        kill \"\$pid\" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5 6 7 8 9 10; do
          kill -0 \"\$pid\" >/dev/null 2>&1 || break
          sleep 0.1
        done
        kill -9 \"\$pid\" >/dev/null 2>&1 || true
      fi
    fi
    rm -f '$pidfile' '$server_log'
  " >/dev/null 2>&1 || true
}

sample_timeout() {
  case "$1" in
    simpleStreams|scan|LargeKernelParameter|HSOpticalFlow|jacobiCudaGraphs|radixSortThrust|segmentationTreeThrust|batchCUBLAS|cuSolverRf|conjugateGradientPrecond|watershedSegmentationNPP)
      printf '%s\n' "$LONG_SAMPLE_TIMEOUT"
      ;;
    *)
      printf '%s\n' "$SAMPLE_TIMEOUT"
      ;;
  esac
}

tsv="$RESULTS_DIR/results.tsv"
summary="$RESULTS_DIR/summary.txt"
: > "$tsv"

pass=0
fail=0
skip=0

run_sample() {
  local i="$1"
  local result_file="$2"
  local sample="${samples[$i]}"
  local port=$((SERVER_PORT_BASE + i))
  local log="$RESULTS_DIR/$sample.log"
  local server_log="/tmp/lupine-samples-$port.log"
  local pidfile="/tmp/lupine-samples-$port.pid"
  local sample_start_seconds="$SECONDS"
  local sample_exe=""
  local sample_cwd=""
  local timeout_seconds=""
  local rc=0
  local status=""
  local signature=""
  local sample_argv=()

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] CUDA sample $((i + 1))/${#samples[@]}: $sample" >&2

  if sample_disabled "$sample"; then
    status="SKIP:disabled"
    signature="disabled by CUDA_SAMPLE_SKIP_LIST"
    printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" > "$result_file"
    return 0
  fi

  sample_exe="$(resolve_sample_exe "$sample" || true)"
  if [[ -z "$sample_exe" ]]; then
    if [[ -z "$(resolve_sample_srcdir "$sample" || true)" ]]; then
      if [[ "$explicit_samples" == "1" ]]; then
        status="FAIL:missing"
      else
        status="SKIP:missing"
      fi
      signature="missing sample source dir: $sample"
    else
      if [[ "$explicit_samples" == "1" ]]; then
        status="FAIL:build-failed"
      else
        status="SKIP:build-failed"
      fi
      build_log="$RESULTS_DIR/.build-$sample.log"
      signature="$(grep -iE 'will not build|fatal error|:[[:space:]]*error:|No rule to make target|not found' "$build_log" 2>/dev/null | tr '\n' ' ' | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
      if [[ -z "$signature" ]]; then
        signature="sample source present but executable not produced: $sample"
      fi
    fi
    printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" > "$result_file"
    return 0
  fi
  sample_cwd="$(sample_workdir "$sample" "$sample_exe")"
  timeout_seconds="$(sample_timeout "$sample")"
  if ! prepare_sample_runtime_files "$sample" "$sample_cwd"; then
    status="FAIL:prepare"
    signature="failed to prepare runtime files"
    printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" > "$result_file"
    return 0
  fi
  sample_argv=()
  while IFS= read -r -d '' arg; do
    sample_argv+=("$arg")
  done < <(sample_args "$sample")

  stop_remote_server "$pidfile" "$server_log"

  if ! ssh_with_timeout \
    "rm -f '$server_log' '$pidfile'; LUPINE_PORT=$port nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null & echo \$! >'$pidfile'; sleep 0.25"; then
    status="FAIL:ssh"
    signature="failed to start remote server on port $port"
    printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" > "$result_file"
    return 0
  fi

  set +e
  (
    cd "$sample_cwd"
    timeout --kill-after=5s "$timeout_seconds" env \
      LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
      LUPINE_SERVER="$SERVER_HOST:$port" \
      LD_PRELOAD="$LUPINE_LIB" \
      "$sample_exe" "${sample_argv[@]}"
  ) >"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  if [[ "$rc" == "0" ]]; then
    status="PASS"
  elif [[ "$rc" == "2" ]]; then
    status="SKIP:waived"
  else
    status="FAIL:$rc"
  fi

  signature="$(tr '\n' ' ' < "$log" | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
  if [[ -z "$signature" && "$rc" == "124" ]]; then
    signature="timed out after ${timeout_seconds}s"
  fi
  printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" > "$result_file"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] CUDA sample $sample -> $status in $((SECONDS - sample_start_seconds))s" >&2
}

result_files=()
pids=()

wait_for_batch() {
  local pid=""

  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
  pids=()
}

for i in "${!samples[@]}"; do
  result_file="$RESULTS_DIR/.sample-$i.tsv"
  result_files[$i]="$result_file"
  run_sample "$i" "$result_file" &
  pids+=("$!")

  if (( ${#pids[@]} >= CUDA_SAMPLE_JOBS )); then
    wait_for_batch
  fi
done

wait_for_batch

for i in "${!samples[@]}"; do
  result_file="${result_files[$i]}"
  sample="${samples[$i]}"

  if [[ -s "$result_file" ]]; then
    IFS=$'\t' read -r sample status signature < "$result_file"
  else
    status="FAIL:internal"
    signature="sample worker did not write result"
  fi

  case "$status" in
    PASS)
      pass=$((pass + 1))
      ;;
    SKIP:*)
      skip=$((skip + 1))
      ;;
    *)
      fail=$((fail + 1))
      ;;
  esac

  printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" | tee -a "$tsv"
  rm -f "$result_file"
done

{
  echo "PASS $pass"
  echo "SKIP $skip"
  echo "FAIL $fail"
  echo "TOTAL $((pass + skip + fail))"
  echo "RESULTS $tsv"
} | tee "$summary"

# Coverage: how much of the pinned cuda-samples catalog this run exercises.
if [[ "$cmake_samples" == "1" && -d "$CUDA_SAMPLES_DIR/Samples" ]]; then
  declare -A _enabled=()
  for _s in "${samples[@]}"; do _enabled["$_s"]=1; done

  graphics=0; ipc=0; mgpu=0; um=0; other=0; covered=0; catalog_total=0
  while IFS= read -r _c; do
    [[ -n "$_c" ]] || continue
    catalog_total=$((catalog_total + 1))
    if [[ -n "${_enabled[$_c]:-}" ]]; then
      covered=$((covered + 1))
    elif [[ "$_c" =~ (D3D|GL|GLES|Vulkan|vulkan|EGL|NvSci|NvMedia|cuDLA|MPI|freeImage|Tegra|fluids|postProcess|CUDA2GL|SLI) ]]; then
      graphics=$((graphics + 1))
    elif [[ " memMapIPCDrv simpleIPC streamOrderedAllocationIPC " == *" $_c "* ]]; then
      ipc=$((ipc + 1))
    elif [[ " simpleP2P p2pBandwidthLatencyTest streamOrderedAllocationP2P conjugateGradientMultiDeviceCG simpleCUFFT_MGPU simpleCUFFT_2d_MGPU " == *" $_c "* ]]; then
      mgpu=$((mgpu + 1))
    elif [[ " UnifiedMemoryPerf UnifiedMemoryStreams systemWideAtomics uvmlite " == *" $_c "* ]]; then
      um=$((um + 1))
    else
      other=$((other + 1))
    fi
  done < <(find "$CUDA_SAMPLES_DIR/Samples" -mindepth 2 -maxdepth 2 -type d -printf '%f\n' 2>/dev/null | sort -u)

  _cov_line="CUDA sample coverage: $covered/$catalog_total catalog samples enabled ($CUDA_SAMPLES_REF)"
  _ne_line="Not enabled: $((catalog_total - covered)) (graphics $graphics, ipc $ipc, multi-gpu $mgpu, unified-memory $um, other $other)"
  echo ""
  echo "$_cov_line"
  echo "$_ne_line"
  printf '%s\n%s\n' "$_cov_line" "$_ne_line" > "$RESULTS_DIR/coverage.txt"
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
