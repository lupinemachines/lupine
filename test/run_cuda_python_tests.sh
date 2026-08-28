#!/usr/bin/env bash
# Run NVIDIA/cuda-python's cuda_bindings test suite and examples through the
# lupine client shim against a remote server. The bindings call the driver API
# directly (no cudart in between for most tests), so this checks entry points
# and struct round-trips that cuda-samples never reach.
#
# Each pytest module and each example is a unit with its own server on
# SERVER_PORT_BASE + index. If CUDA_PYTHON_KNOWN_FAILURES names an existing
# file, its pytest node ids are passed to pytest as --deselect.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_PYTHON_URL="${CUDA_PYTHON_URL:-https://github.com/NVIDIA/cuda-python.git}"
CUDA_PYTHON_REF="${CUDA_PYTHON_REF:-}"
CUDA_PYTHON_DIR="${CUDA_PYTHON_DIR:-$repo_root/test/cuda-python/cuda-python}"
CUDA_PYTHON_SKIP_LIST="${CUDA_PYTHON_SKIP_LIST:-}"
CUDA_PYTHON_KNOWN_FAILURES="${CUDA_PYTHON_KNOWN_FAILURES:-$repo_root/test/cuda-python/known_failures.txt}"
CUDA_PYTHON_INSTALL="${CUDA_PYTHON_INSTALL:-auto}"
SERVER_HOST="${SERVER_HOST:-inferable-node-008}"
SERVER_USER="${SERVER_USER:-kevin}"
SERVER_SSH_TARGET="${SERVER_SSH_TARGET:-$SERVER_USER@$SERVER_HOST}"
SERVER_PORT_BASE="${SERVER_PORT_BASE:-20300}"
SSH_OPTS="${SSH_OPTS:-}"
# shellcheck disable=SC2206
SSH_ARGS=($SSH_OPTS)
SSH_COMMAND_TIMEOUT="${SSH_COMMAND_TIMEOUT:-45}"
SERVER_UPLOAD="${SERVER_UPLOAD:-1}"
SERVER_LOCAL_BIN="${SERVER_LOCAL_BIN:-$repo_root/build/lupine_driver_server}"
SERVER_REMOTE_BIN="${SERVER_REMOTE_BIN:-/tmp/lupine-driver-server-cuda-python-${USER:-lupine}-$$}"
SERVER_REMOTE_CLEANUP="${SERVER_REMOTE_CLEANUP:-1}"
SERVER_LD_LIBRARY_PATH="${SERVER_LD_LIBRARY_PATH:-}"

LUPINE_LIB="${LUPINE_LIB:-$repo_root/build/libcuda.so.1}"
LUPINE_LIB_DIR="$(cd "$(dirname "$LUPINE_LIB")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$repo_root/.venv-pytorch312/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/usr/local/cuda/lib64}"
TEST_TIMEOUT="${TEST_TIMEOUT:-300}"
RESULTS_DIR="${RESULTS_DIR:-$repo_root/test/cuda-python/results/$(date +%Y%m%d-%H%M%S)}"

usage() {
  cat <<EOF
Usage: $0 [unit ...]
Units are pytest modules (test_cuda.py) or example paths relative to the
examples directory (0_Introduction/vector_add_drv.py). Default: every
test_*.py except test_cufile.py and test_examples.py, plus every example.
Environment:
  CUDA_PYTHON_DIR        Clone path. Default: $CUDA_PYTHON_DIR
  CUDA_PYTHON_REF        Tag to check out. Default: v<installed bindings version>.
  CUDA_PYTHON_INSTALL    auto installs the bindings matching the CUDA toolkit
                         major plus pytest into PYTHON_BIN's environment; 0 skips.
  CUDA_PYTHON_SKIP_LIST  Comma or space separated units to mark SKIP:disabled.
  CUDA_PYTHON_KNOWN_FAILURES
                         pytest node ids (relative to the tests or examples
                         directory) to --deselect when the file exists.
                         Default: $CUDA_PYTHON_KNOWN_FAILURES
  PYTHON_BIN             Interpreter. Default: $PYTHON_BIN
  SERVER_SSH_TARGET      GPU host. Default: $SERVER_SSH_TARGET
  SERVER_PORT_BASE       First per-unit server port. Default: $SERVER_PORT_BASE
  TEST_TIMEOUT           Per-unit timeout in seconds. Default: $TEST_TIMEOUT
  RESULTS_DIR            Output directory. Default: test/cuda-python/results/<timestamp>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for f in "$SERVER_LOCAL_BIN" "$LUPINE_LIB"; do
  [[ -e "$f" ]] || { echo "missing build artifact: $f (build lupine first)" >&2; exit 1; }
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing python: $PYTHON_BIN" >&2
  exit 1
fi

cuda_major() {
  local release=""
  if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
    release="$("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n1)"
  fi
  if [[ -z "$release" && -f "$CUDA_HOME/version.json" ]]; then
    release="$(sed -nE 's/.*"cuda"[^0-9]*([0-9]+)\..*/\1/p' "$CUDA_HOME/version.json" | head -n1)"
  fi
  printf '%s\n' "$release"
}

# cuda-bindings was split out of cuda-python at 12.6.1; the 11.8 line still
# ships as cuda-python with the cuda.cuda/cuda.cudart module layout.
major="$(cuda_major)"
case "$major" in
  11) package=cuda-python; spec="cuda-python==11.8.*" ;;
  12) package=cuda-bindings; spec="cuda-bindings==12.*" ;;
  13) package=cuda-bindings; spec="cuda-bindings==13.*" ;;
  *) echo "cannot map CUDA toolkit major '$major' to a cuda-python release" >&2; exit 1 ;;
esac

if [[ "$CUDA_PYTHON_INSTALL" != "0" ]]; then
  "$PYTHON_BIN" -m pip install --quiet "$spec" numpy pytest pytest-benchmark
fi
installed="$("$PYTHON_BIN" -c "from importlib.metadata import version; print(version('$package'))")"
if [[ -z "$CUDA_PYTHON_REF" ]]; then
  CUDA_PYTHON_REF="v$installed"
fi

mkdir -p "$RESULTS_DIR"
if [[ "$CUDA_PYTHON_INSTALL" != "0" ]]; then
  mkdir -p "$(dirname "$CUDA_PYTHON_DIR")"
  if [[ ! -d "$CUDA_PYTHON_DIR/.git" ]]; then
    rm -rf "$CUDA_PYTHON_DIR"
    git clone --quiet "$CUDA_PYTHON_URL" "$CUDA_PYTHON_DIR"
  fi
  git config --global --add safe.directory "$CUDA_PYTHON_DIR"
  if [[ "$(git -C "$CUDA_PYTHON_DIR" describe --tags --exact-match 2>/dev/null || true)" != "$CUDA_PYTHON_REF" ]]; then
    git -C "$CUDA_PYTHON_DIR" fetch --quiet --tags origin
    git -C "$CUDA_PYTHON_DIR" checkout --quiet "$CUDA_PYTHON_REF"
  fi
fi

tests_dir=""
for d in cuda_bindings/tests cuda/tests; do
  if [[ -d "$CUDA_PYTHON_DIR/$d" ]]; then
    tests_dir="$CUDA_PYTHON_DIR/$d"
    break
  fi
done
if [[ -z "$tests_dir" ]]; then
  echo "no tests directory in $CUDA_PYTHON_DIR at $CUDA_PYTHON_REF" >&2
  exit 1
fi
examples_dir=""
for d in cuda_bindings/examples examples; do
  if [[ -d "$CUDA_PYTHON_DIR/$d" ]]; then
    examples_dir="$CUDA_PYTHON_DIR/$d"
    break
  fi
done

# test_cufile needs GPUDirect Storage; nvvm/nvjitlink/nvfatbin are host-only
# libraries whose tests track the toolkit minor rather than the driver;
# test_examples is covered per example below so one wedged example cannot
# take the whole module down.
UNITS=()
while IFS= read -r m; do
  UNITS+=("$m")
done < <(cd "$tests_dir" && ls test_*.py | grep -vE '^test_(cufile|examples|nvvm|nvjitlink|nvfatbin)\.py$' | sort)
# Examples are *_test.py pytest files before 13.3 (next to common/ helpers and
# a numba plugin) and plain scripts from 13.3 on.
if [[ -n "$examples_dir" ]]; then
  if compgen -G "$examples_dir/*/*_test.py" >/dev/null; then
    example_glob='*_test.py'
  else
    example_glob='*.py'
  fi
  while IFS= read -r e; do
    UNITS+=("$e")
  done < <(cd "$examples_dir" && find . -name "$example_glob" -not -path './common/*' -not -name conftest.py -not -name '__init__.py' -not -name 'numba_*' -printf '%P\n' | sort)
fi
if [[ $# -gt 0 ]]; then
  UNITS=("$@")
fi

if [[ "${LIST_TESTS:-0}" == "1" ]]; then
  printf '%s\n' "${UNITS[@]}"
  exit 0
fi

deselect_args=()
if [[ -f "$CUDA_PYTHON_KNOWN_FAILURES" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    line="${line//[[:space:]]/}"
    [[ -n "$line" ]] && deselect_args+=(--deselect "$line")
  done < "$CUDA_PYTHON_KNOWN_FAILURES"
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

unit_disabled() {
  local unit="$1"
  local disabled=""
  # shellcheck disable=SC2206
  local disabled_units=(${CUDA_PYTHON_SKIP_LIST//,/ })
  for disabled in "${disabled_units[@]}"; do
    if [[ "$disabled" == "$unit" ]]; then
      return 0
    fi
  done
  return 1
}

tsv="$RESULTS_DIR/results.tsv"
: > "$tsv"
pass=0
fail=0
skip=0

echo "cuda-python $CUDA_PYTHON_REF ($package $installed), ${#UNITS[@]} units, $((${#deselect_args[@]} / 2)) known failures deselected" >&2

for i in "${!UNITS[@]}"; do
  unit="${UNITS[$i]}"
  port=$((SERVER_PORT_BASE + i))
  log="$RESULTS_DIR/${unit//\//_}.log"
  server_log="/tmp/lupine-cuda-python-$port.log"
  pidfile="/tmp/lupine-cuda-python-$port.pid"
  unit_start_seconds="$SECONDS"
  server_environment="LUPINE_PORT=$port"
  if [[ -n "$SERVER_LD_LIBRARY_PATH" ]]; then
    printf -v server_environment 'LD_LIBRARY_PATH=%q %s' \
      "$SERVER_LD_LIBRARY_PATH" "$server_environment"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cuda-python unit $((i + 1))/${#UNITS[@]}: $unit" >&2

  if unit_disabled "$unit"; then
    status="SKIP:disabled"
    skip=$((skip + 1))
    printf '%s\t%s\t%s\n' "$unit" "$status" "disabled by CUDA_PYTHON_SKIP_LIST" | tee -a "$tsv"
    continue
  fi

  # Examples are standalone scripts from 13.3 on and pytest files before that.
  # Scripts exit 100 when the machine cannot run them (peer GPUs, toolkit
  # version); pytest exits 5 when every test in a module was skipped/deselected.
  # --rootdir keeps node ids relative to the tests directory so the known
  # failures file can name them without the cuda_bindings/tests prefix.
  pytest_cmd=("$PYTHON_BIN" -m pytest -q --no-header -p no:cacheprovider -p no:randomly)
  if [[ "$unit" == test_*.py ]]; then
    cwd="$tests_dir"
    cmd=("${pytest_cmd[@]}" --rootdir "$tests_dir" "${deselect_args[@]}" "$unit")
  elif [[ "$unit" == *_test.py ]]; then
    cwd="$examples_dir"
    cmd=("${pytest_cmd[@]}" --rootdir "$examples_dir" "${deselect_args[@]}" "$unit")
  else
    cwd="$examples_dir"
    cmd=("$PYTHON_BIN" "$unit")
  fi

  stop_remote_server "$pidfile" "$server_log"
  ssh_with_timeout \
    "rm -f '$server_log' '$pidfile'; $server_environment nohup '$SERVER_REMOTE_BIN' >'$server_log' 2>&1 < /dev/null & echo \$! >'$pidfile'; sleep 0.25"

  set +e
  (
    cd "$cwd"
    timeout --kill-after=5s "$TEST_TIMEOUT" env \
      LD_LIBRARY_PATH="$LUPINE_LIB_DIR:$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}" \
      LUPINE_SERVER="$SERVER_HOST:$port" \
      LD_PRELOAD="$LUPINE_LIB" \
      CUDA_BINDINGS_SKIP_EXAMPLE=100 \
      "${cmd[@]}"
  ) >"$log" 2>&1
  rc=$?
  set -e

  stop_remote_server "$pidfile" "$server_log"

  case "$rc" in
    0) status="PASS"; pass=$((pass + 1)) ;;
    5) status="SKIP:empty"; skip=$((skip + 1)) ;;
    100) status="SKIP:unsupported"; skip=$((skip + 1)) ;;
    *) status="FAIL:$rc"; fail=$((fail + 1)) ;;
  esac

  if [[ "${cmd[2]:-}" == pytest ]]; then
    signature="$({ grep -E '^(FAILED|ERROR) ' "$log" || true; } | sed -E 's/ - .*//' | tr '\n' ' ')$(tail -n1 "$log" | tr -d '\0')"
  else
    signature="$({ grep -E 'LUPINE|Error|error|Traceback' "$log" || true; } | head -n3 | tr '\n' ' ')"
    [[ -n "$signature" ]] || signature="$(tail -n1 "$log" | tr -d '\0')"
  fi
  signature="$(printf '%s' "$signature" | sed -E 's/[[:space:]]+/ /g' | cut -c1-240)"
  if [[ -z "$signature" && "$rc" == "124" ]]; then
    signature="timed out after ${TEST_TIMEOUT}s"
  fi
  printf '%s\t%s\t%s\n' "$unit" "$status" "$signature" | tee -a "$tsv"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cuda-python unit $unit -> $status in $((SECONDS - unit_start_seconds))s" >&2
done

{
  echo "PASS $pass"
  echo "FAIL $fail"
  echo "SKIP $skip"
  echo "TOTAL $((pass + fail + skip))"
  echo "DESELECTED $((${#deselect_args[@]} / 2))"
  echo "RESULTS $tsv"
} | tee "$RESULTS_DIR/summary.txt"

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
