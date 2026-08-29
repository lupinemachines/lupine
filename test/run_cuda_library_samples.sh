#!/usr/bin/env bash
# Build and run NVIDIA/CUDALibrarySamples (cuBLAS, cuBLASLt, cuFFT, cuRAND,
# cuSOLVER, cuSPARSE, nvJPEG, NPP) through the lupine client shim against a
# remote server. Every leaf directory with a CMakeLists.txt is a standalone
# CMake project; every executable it produces is a unit with its own server on
# SERVER_PORT_BASE + index.
#
# Libraries that ship outside the toolkit (cuTENSOR, cuDSS, nvCOMP, cuSPARSELt,
# MathDx, the *Mp multi-process variants) are not selected by default.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LIBRARY_SAMPLES_URL="${LIBRARY_SAMPLES_URL:-https://github.com/NVIDIA/CUDALibrarySamples.git}"
LIBRARY_SAMPLES_REF="${LIBRARY_SAMPLES_REF:-3437729}"
LIBRARY_SAMPLES_DIR="${LIBRARY_SAMPLES_DIR:-$repo_root/test/cuda-library-samples/CUDALibrarySamples}"
LIBRARY_SAMPLES_BUILD_DIR="${LIBRARY_SAMPLES_BUILD_DIR:-$LIBRARY_SAMPLES_DIR/build}"
LIBRARY_SAMPLES_LIBS="${LIBRARY_SAMPLES_LIBS:-cuBLAS cuBLASLt cuFFT cuRAND cuSOLVER cuSPARSE nvJPEG NPP}"
LIBRARY_SAMPLES_ARCH="${LIBRARY_SAMPLES_ARCH:-${CUDA_SAMPLES_ARCH:-89}}"
LIBRARY_SAMPLES_CMAKE_ARGS="${LIBRARY_SAMPLES_CMAKE_ARGS:-}"
LIBRARY_SAMPLES_SKIP_LIST="${LIBRARY_SAMPLES_SKIP_LIST:-}"
LIBRARY_SAMPLES_KNOWN_FAILURES="${LIBRARY_SAMPLES_KNOWN_FAILURES:-$repo_root/test/cuda-library-samples/known_failures.txt}"
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
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-180}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/cuda-library-samples/results/$(date +%Y%m%d-%H%M%S)}"
nvjpeg_assets="${TMPDIR:-/tmp}/lupine-nvjpeg-assets"

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
  LIBRARY_SAMPLES_SKIP_LIST  Comma or space separated units to mark SKIP:disabled.
  LIBRARY_SAMPLES_KNOWN_FAILURES
                             Units to mark SKIP:known. Default: $LIBRARY_SAMPLES_KNOWN_FAILURES
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
    # shellcheck disable=SC2086
    find $LIBRARY_SAMPLES_LIBS -name CMakeLists.txt -printf '%h\n' \
      | grep -vE '(^|/)(cmake|utils)(/|$)' | sort \
      | awk '{ if (prev != "" && index($0, prev "/") == 1) { skip[prev] = 1 } ; prev = $0; lines[n++] = $0 }
             END { for (i = 0; i < n; i++) if (!(lines[i] in skip)) print lines[i] }'
  )
fi
if [[ "${LIST_TESTS:-0}" == "1" ]]; then
  printf '%s\n' "${SAMPLES[@]}"
  exit 0
fi

build_sample() {
  local sample="$1"
  local build="$LIBRARY_SAMPLES_BUILD_DIR/$sample"
  local log="$build.build.log"
  mkdir -p "$(dirname "$build")"
  # The cuSPARSE lists link bare cudart/cusparse/cuda names and compile host
  # .cpp files that include cuda_fp16.h; LIBRARY_PATH and CPATH are how the
  # toolkit reaches those lines without patching the samples.
  # shellcheck disable=SC2086
  if LIBRARY_PATH="$CUDA_LIB_DIR:$CUDA_LIB_DIR/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
     CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}" \
      cmake -S "$LIBRARY_SAMPLES_DIR/$sample" -B "$build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$LIBRARY_SAMPLES_ARCH" \
        -DCUDAToolkit_ROOT="$CUDA_HOME" \
        $LIBRARY_SAMPLES_CMAKE_ARGS >"$log" 2>&1 \
     && LIBRARY_PATH="$CUDA_LIB_DIR:$CUDA_LIB_DIR/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}" \
     CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}" \
      cmake --build "$build" --parallel 2 >>"$log" 2>&1; then
    echo "built $sample"
  else
    echo "BUILD FAILED $sample: $(grep -m1 -iE 'error' "$log" | cut -c1-160)"
    return 1
  fi
}
export -f build_sample
export LIBRARY_SAMPLES_DIR LIBRARY_SAMPLES_BUILD_DIR LIBRARY_SAMPLES_ARCH LIBRARY_SAMPLES_CMAKE_ARGS CUDA_HOME CUDA_LIB_DIR

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
  case "$1" in
    nvJPEG/nvJPEG-Decoder/*) printf '%s\0' -i "$images" -b 2 -o "$nvjpeg_assets/nvjpeg-decoded" ;;
    nvJPEG/nvJPEG-Decoder-Backend-ROI/*) printf '%s\0' -i "$images" -b 2 ;;
    nvJPEG/nvJPEG-Decoder-MultipleInstances/*) printf '%s\0' -i "$images" -s 2 -j 2 -r 1 ;;
    nvJPEG/nvJPEG-Encoder-MultipleInstances/*) printf '%s\0' -i "$nvjpeg_assets/nvjpeg-decoded/" -s 2 -j 2 -r 1 ;;
    nvJPEG/Image-Resize/*) printf '%s\0' -i "$images" -o "$nvjpeg_assets/nvjpeg-resized" ;;
    nvJPEG/Image-Resize-WaterMark/*) printf '%s\0' -i "$LIBRARY_SAMPLES_DIR/nvJPEG/Image-Resize-WaterMark/input_images/" -o "$nvjpeg_assets/nvjpeg-watermarked" ;;
    NPP/nppCanny/*) printf '%s\0' example_input.png ;;
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

ssh_with_timeout() {
  timeout --kill-after=5s "$SSH_COMMAND_TIMEOUT" \
    ssh "${SSH_ARGS[@]}" "$SERVER_SSH_TARGET" "$@"
}

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

in_list() {
  local needle="$1"
  shift
  local item=""
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

unit_output_failure() {
  local unit="$1"
  local log="$2"
  local residual=""

  case "$unit" in
    cuSOLVER/Xgeev/*)
      if grep -qiE '(^|[^[:alnum:]_])[-+]?(nan|inf(inity)?)([^[:alnum:]_]|$)' "$log"; then
        printf '%s\n' "non-finite Xgeev result"
        return 0
      fi
      case "$unit" in
        */cusolver_Xgeev_example1|*/cusolver_Xgeev_example3)
          residual="$(awk -F'= ' '/\|A\*VR - VR\*diag\(W\)\|/ { print $2 }' "$log" | awk '{ print $1 }' | tail -n1)"
          if [[ -z "$residual" ]]; then
            printf '%s\n' "missing Xgeev residual"
            return 0
          fi
          if ! awk -v value="$residual" 'BEGIN {
            exit !(value ~ /^[+]?[0-9]+([.][0-9]*)?([Ee][-+]?[0-9]+)?$/ &&
                   value + 0 >= 0 && value + 0 <= 1e-10)
          }'; then
            printf 'Xgeev residual exceeds tolerance: %s\n' "$residual"
            return 0
          fi
          ;;
      esac
      ;;
  esac
  return 1
}

mkdir -p "$RESULTS_DIR" \
  "$nvjpeg_assets/nvjpeg-decoded" \
  "$nvjpeg_assets/nvjpeg-resized" \
  "$nvjpeg_assets/nvjpeg-watermarked"
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
  port=$((SERVER_PORT_BASE + i))
  log="$RESULTS_DIR/${unit//\//_}.log"
  server_log="/tmp/lupine-libsamples-$port.log"
  pidfile="/tmp/lupine-libsamples-$port.pid"
  unit_start_seconds="$SECONDS"
  server_environment="LUPINE_PORT=$port"
  if [[ -n "$SERVER_LD_LIBRARY_PATH" ]]; then
    printf -v server_environment 'LD_LIBRARY_PATH=%q %s' \
      "$SERVER_LD_LIBRARY_PATH" "$server_environment"
  fi
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

  stop_remote_server "$pidfile" "$server_log"
  ssh_with_timeout \
    "rm -f '$server_log' '$pidfile'; $server_environment nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null & echo \$! >'$pidfile'; sleep 0.25"

  set +e
  (
    cd "$cwd"
    timeout --kill-after=5s "$SAMPLE_TIMEOUT" env \
      LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
      LUPINE_SERVER="$SERVER_HOST:$port" \
      LD_PRELOAD="$LUPINE_LIB" \
      "$exe" "${argv[@]}"
  ) >"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  # Some samples report invalid results while still exiting successfully.
  output_failure=""
  if [[ "$rc" == "0" ]] && output_failure="$(unit_output_failure "$unit" "$log")"; then
    status="FAIL:output"
    fail=$((fail + 1))
  elif [[ "$rc" == "0" ]] && grep -qE '\bFAILED\b' "$log"; then
    status="FAIL:output"
    fail=$((fail + 1))
  elif [[ "$rc" == "0" ]]; then
    status="PASS"
    pass=$((pass + 1))
  else
    status="FAIL:$rc"
    fail=$((fail + 1))
  fi
  signature="$output_failure"
  if [[ -z "$signature" ]]; then
    signature="$({ grep -iE 'LUPINE|error|fail|unsupported|not supported|Aborted|Segmentation' "$log" || true; } | head -n3 | tr '\n' ' ')"
  fi
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
