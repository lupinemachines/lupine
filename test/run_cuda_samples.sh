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
BUILD_SAMPLES="${BUILD_SAMPLES:-auto}"
BUILD_ONLY="${BUILD_ONLY:-0}"
JOBS="${JOBS:-$(nproc)}"
SAMPLE_SUITE="${SAMPLE_SUITE:-compliance}"

SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-14900}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-lupine-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"

LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-20}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/cuda-samples/results/$(date +%Y%m%d-%H%M%S)}"

CORE_SAMPLES=(
  deviceQuery deviceQueryDrv topologyQuery
  vectorAdd vectorAddDrv vectorAdd_nvrtc
  asyncAPI cudaOpenMP clock clock_nvrtc matrixMul matrixMulDrv matrixMul_nvrtc
  inlinePTX inlinePTX_nvrtc
  simpleAssert simpleAssert_nvrtc
  simpleAttributes simpleCallback simpleDrvRuntime simplePrintf simpleTemplates
  simpleAtomicIntrinsics simpleAtomicIntrinsics_nvrtc simpleStreams simpleMultiCopy simpleMultiGPU
  simpleOccupancy simpleCooperativeGroups
  simpleCubemapTexture simpleLayeredTexture simpleSurfaceWrite
  simpleTexture simpleTextureDrv simplePitchLinearTexture
  mergeSort reduction reductionMultiBlockCG scan sortingNetworks histogram scalarProd transpose
  BlackScholes BlackScholes_nvrtc binomialOptions binomialOptions_nvrtc SobolQRNG quasirandomGenerator
  quasirandomGenerator_nvrtc
  simpleCudaGraphs streamOrderedAllocation cudaCompressibleMemory simpleZeroCopy alignedTypes LargeKernelParameter
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
  MC_SingleAsianOptionP
  cudaGraphsPerfScaling graphConditionalNodes graphMemoryNodes graphMemoryFootprint jacobiCudaGraphs
  dct8x8 lineOfSight nbody recursiveGaussian stereoDisparity
  simpleTexture3D
  radixSortThrust segmentationTreeThrust template interval
  dsl ptxgen ptxjit matrixMulDynlinkJIT threadMigration
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
  jitLto
  watershedSegmentationNPP
)

COMPLIANCE_SAMPLES=(
  c++11_cuda
  vectorAdd
  batchCUBLAS
  simplePrintf
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
  CUDA_SAMPLES_REF     Optional branch/tag/commit to checkout after clone.
  BUILD_SAMPLES        auto, 1, or 0. Default: auto.
  BUILD_ONLY           1 to clone/build selected samples and exit before running.
  SAMPLE_SUITE         compliance, core, libraries, or extended when no samples are given.
                       Default: compliance.
  SERVER_SSH_TARGET    GPU host SSH target. Default: kevin@inferable-node-008.
  SERVER_PORT_BASE     First per-sample server port. Default: 14900.
  LUPINE_LIB            Client shim. Default: $repo_root/build/libcuda.so.1.
  RESULTS_DIR          Output directory. Default: test/cuda-samples/results/<timestamp>.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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

mkdir -p "$(dirname "$CUDA_SAMPLES_DIR")" "$RESULTS_DIR"

if [[ ! -d "$CUDA_SAMPLES_DIR/.git" ]]; then
  git clone "$CUDA_SAMPLES_URL" "$CUDA_SAMPLES_DIR"
fi

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
    exe="$(find "$CUDA_SAMPLES_BUILD_DIR" -type f -name "$sample" -perm -111 2>/dev/null | head -n1 || true)"
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
    nbody)
      printf '%s\0' -benchmark -numbodies=4096 -i=1
      ;;
    oceanFFT)
      printf '%s\0' -qatest
      ;;
    ptxgen)
      printf '%s\0' test.ll
      ;;
    recursiveGaussian)
      printf '%s\0' -benchmark
      ;;
    simpleTexture3D)
      printf '%s\0' -file=data/ref_texture3D.bin
      ;;
  esac
}

sample_workdir() {
  local sample="$1"
  local sample_exe="$2"

  case "$sample" in
    nbody|oceanFFT|ptxgen|recursiveGaussian|simpleTexture3D)
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
      samples=("${COMPLIANCE_SAMPLES[@]}")
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
    if [[ ! -f "$CUDA_SAMPLES_BUILD_DIR/CMakeCache.txt" ]]; then
      # shellcheck disable=SC2086
      cmake -S "$CUDA_SAMPLES_DIR" -B "$CUDA_SAMPLES_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release $CUDA_SAMPLES_CMAKE_ARGS
    fi
    if [[ "$selected_sample_build" == "1" ]]; then
      for sample in "${samples[@]}"; do
        cmake --build "$CUDA_SAMPLES_BUILD_DIR" --parallel "$JOBS" --target "$sample" || true
      done
    else
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

if [[ ! -x "$SERVER_LOCAL_BIN" ]]; then
  echo "missing server binary: $SERVER_LOCAL_BIN" >&2
  exit 1
fi

if [[ "$SERVER_UPLOAD" == "1" ]]; then
  scp -q "${SSH_ARGS[@]}" "$SERVER_LOCAL_BIN" "$SERVER_SSH_TARGET:$SERVER_REMOTE_BIN"
fi

cleanup_remote_bin() {
  if [[ "$SERVER_UPLOAD" == "1" && "$SERVER_REMOTE_CLEANUP" == "1" ]]; then
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
      "rm -f '$SERVER_REMOTE_BIN'" >/dev/null 2>&1 || true
  fi
}
trap cleanup_remote_bin EXIT

stop_remote_server() {
  local pidfile="$1"
  local server_log="$2"

  ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "
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

tsv="$RESULTS_DIR/results.tsv"
summary="$RESULTS_DIR/summary.txt"
: > "$tsv"

pass=0
fail=0
skip=0

for i in "${!samples[@]}"; do
  sample="${samples[$i]}"
  port=$((SERVER_PORT_BASE + i))
  log="$RESULTS_DIR/$sample.log"
  server_log="/tmp/lupine-samples-$port.log"
  pidfile="/tmp/lupine-samples-$port.pid"

  sample_exe="$(resolve_sample_exe "$sample" || true)"
  if [[ -z "$sample_exe" ]]; then
    if [[ "$explicit_samples" == "1" ]]; then
      status="FAIL:missing"
      fail=$((fail + 1))
    else
      status="SKIP:missing"
      skip=$((skip + 1))
    fi
    signature="missing executable: $sample"
    printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" | tee -a "$tsv"
    continue
  fi
  sample_cwd="$(sample_workdir "$sample" "$sample_exe")"
  prepare_sample_runtime_files "$sample" "$sample_cwd"
  sample_argv=()
  while IFS= read -r -d '' arg; do
    sample_argv+=("$arg")
  done < <(sample_args "$sample")

  stop_remote_server "$pidfile" "$server_log"

  ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" \
    "rm -f '$server_log' '$pidfile'; LUPINE_PORT=$port nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null & echo \$! >'$pidfile'; sleep 0.25"

  set +e
  (
    cd "$sample_cwd"
    timeout --kill-after=5s "$SAMPLE_TIMEOUT" env \
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
    pass=$((pass + 1))
  elif [[ "$rc" == "2" ]]; then
    status="SKIP:waived"
    skip=$((skip + 1))
  else
    status="FAIL:$rc"
    fail=$((fail + 1))
  fi

  signature="$(tr '\n' ' ' < "$log" | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
  printf '%s\t%s\t%s\n' "$sample" "$status" "$signature" | tee -a "$tsv"
done

{
  echo "PASS $pass"
  echo "SKIP $skip"
  echo "FAIL $fail"
  echo "TOTAL $((pass + skip + fail))"
  echo "RESULTS $tsv"
} | tee "$summary"

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
