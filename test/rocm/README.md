# Upstream HIP / ROCm runners

These opt-in runners follow the CUDA runners' per-unit server, timeout, log,
`results.tsv`, `summary.txt`, and `known_failures.txt` conventions. They are not
registered in the root CTest project or any GitHub Actions workflow. They wire
up the upstream suites in [#659](https://github.com/lupinemachines/lupine/issues/659);
they do not implement the missing HIP APIs.

| Runner | Selection / unit |
| --- | --- |
| `test/run_hip_tests.sh` | Each `catch/unit/<feature>` directory runs its upstream CTest cases against one server. Also discovers leaf CMake projects under `samples/`, with one server per executable. |
| `test/run_rocm_examples.sh` | Leaf CMake projects under `HIP-Basic`, `Libraries`, and `Applications`, with one server per executable. |
| `test/run_rocm_libraries.sh` | `rocblas/rocblas-test`, `rocblas/rocblas-bench`, `hipblas/hipblas-test`, and `hipblas/hipblas-bench` from `rocm-libraries/projects`. Builds clients against installed ROCm libraries using `SKIP_LIBRARY=ON`. |

All three default to `rocm-7.2.0`. Override the suite's `*_REF` to match the
installed SDK. The library tier initially covers rocBLAS and hipBLAS as
described in #659; other monorepo projects need their own build/argument adapters.
The pinned hip-tests release has no `catch/unit/env` directory; use `LIST_TESTS=1`
to see the available feature directories.

## Local use

Requirements: Linux with `ldd`, Python 3.9+, Git, CMake 3.21+, Ninja, a ROCm SDK, and an AMD
GPU. Library clients additionally need GTest, a Fortran compiler, an ILP64
(64-bit integer) BLAS/LAPACK implementation, pkg-config, and Python PyYAML. On Ubuntu these
are provided by `libgtest-dev`, `gfortran`, `libopenblas64-pthread-dev`, and
`pkg-config`, and `python3-yaml`. The default reference library is OpenBLAS;
`pkg-config openblas64` supplies its matching headers and library directory.
Use `PKG_CONFIG_PATH` for a custom install, or override `BLAS_INCLUDE_DIR`,
`CMAKE_LIBRARY_PATH`, and `BLA_VENDOR` through `ROCM_LIBRARIES_CMAKE_ARGS`.
Upstream CMake can fetch rocm-cmake when it is not installed.

```sh
# A HIP-only Lupine build for the local GPU.
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLUPINE_BUILD_CUDA=OFF -DLUPINE_BUILD_NVML=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build --target lupine_hip_client lupine_driver_server -j8

# Native baseline on this host, then the same binaries through local Lupine.
RUN_NATIVE=1 test/run_rocm_examples.sh HIP-Basic/hello_world HIP-Basic/matrix_multiplication
BUILD_SAMPLES=0 test/run_rocm_examples.sh HIP-Basic/hello_world HIP-Basic/matrix_multiplication

# Feature directory and small standalone HIP program.
RUN_NATIVE=1 HIP_TESTS_CTEST_ARGS='-R ^Unit_hipGetDeviceCount' \
  test/run_hip_tests.sh catch/unit/device
RUN_NATIVE=1 test/run_hip_tests.sh samples/0_Intro/square

# Library clients: the full suite is expensive; select one executable to start.
RUN_NATIVE=1 test/run_rocm_libraries.sh hipblas/hipblas-bench
```

Native mode removes inherited preloads and Lupine endpoints and ignores the
Lupine known-failure list. `ROCM_LD_LIBRARY_PATH` supplies any additional native
runtime directories; by default only `$ROCM_PATH/lib` is used. Compile, discovery,
and server processes also use this native environment.

Lupine mode creates a private directory mapping the HIP runtime names `.so`,
`.so.1`, `.so.6`, and `.so.7` to the selected shim and enables eager symbol
resolution. It also checks executable dependencies with `ldd`, rejecting a
native HIP runtime selected through an old `DT_RPATH` or a different SONAME.
This prevents the current `.so.1` shim from silently falling back to the system
runtime. Missing `__hip*` registration or other HIP entry points
are expected failures until the backend work in #659 lands. Each unit gets a
fresh server on `SERVER_PORT_BASE + index`, and both client and server process
groups are reaped on completion, timeout, or interruption.

## Options

Each suite has its own prefix: `HIP_TESTS`, `ROCM_EXAMPLES`, or `ROCM_LIBRARIES`.

| Variable | Default / behavior |
| --- | --- |
| `<PREFIX>_URL`, `_REF` | Upstream Git URL and `rocm-7.2.0`. |
| `<PREFIX>_DIR`, `_BUILD_DIR` | `test/<suite>/<suite>` and `<checkout>/build`. |
| `<PREFIX>_CMAKE_ARGS` | Extra configure arguments, parsed with shell quoting. |
| `<PREFIX>_SKIP_LIST` | Comma/space separated exact unit names; records `SKIP:disabled`. |
| `<PREFIX>_KNOWN_FAILURES` | Optional `test/<suite>/known_failures.txt`; add entries only for confirmed Lupine failures, with an exact unit and a tracking issue URL in a comment. A missing file means no known-failure skips. |
| `ROCM_PATH` | `/opt/rocm`. |
| `ROCM_ARCH` | Semicolon separated gfx architectures; detected from the local GPU if unset. Set this explicitly for a remote GPU or a machine without a GPU. |
| `ROCM_EXAMPLES_ROOTS` | `HIP-Basic Libraries Applications`. |
| `ROCM_LIBRARIES_LIBS` | `rocblas hipblas`. |
| `HIP_TESTS_CTEST_ARGS` | Extra arguments such as `-R ^Unit_hipGetDeviceCount`. |
| `ROCM_LIBRARIES_GTEST_FILTER` | `quick*`; override with `*` for full conformance. Zero matched tests are failures. |
| `ROCM_LIBRARIES_BENCH_ARGS` | `-f gemm -r f32_r -m 32 -n 32 -k 32 -i 1 -j 0 -v 1` (small GEMM with CPU validation). |
| `BUILD_TESTS`, `BUILD_SAMPLES` | `auto`: configure and incrementally build selected targets. `0`: reuse builds without fetching/configuring. `1`: explicitly enable building. `BUILD_SAMPLES` applies to rocm-examples; `BUILD_TESTS` to the other two. |
| `BUILD_ONLY` | `1` builds and exits; build failures return nonzero. |
| `LIST_TESTS` | `1` prints project selections without configuring/building or starting a server. Can clone if building is enabled and the checkout is missing. |
| `JOBS` | CMake build parallelism, at most 8 by default. |
| `RUN_NATIVE` | `1` runs against the native runtime without a server. Default `0`. |
| `LUPINE_LIB` | `build/libamdhip64.so.1`. |
| `SERVER_MODE` | `local` (default) or `ssh`. |
| `SERVER_LOCAL_BIN` | `build/lupine_driver_server`. |
| `SERVER_HOST`, `SERVER_SSH_TARGET` | `127.0.0.1`; SSH target defaults to `SERVER_HOST`. |
| `SERVER_PORT_BASE` | `23000`. Use different ranges for concurrent runner invocations. |
| `SERVER_REMOTE_BIN` | Unique temporary path; required with `SERVER_UPLOAD=0`. |
| `SERVER_UPLOAD`, `SERVER_REMOTE_CLEANUP` | Both default to `1` for SSH mode. |
| `SSH_OPTS`, `SSH_COMMAND_TIMEOUT` | SSH/scp options; 45-second control-command timeout. |
| `SERVER_LD_LIBRARY_PATH` | Native ROCm path on the server; defaults to the native client library path. |
| `TEST_TIMEOUT` | 300 seconds per unit; falls back to `SAMPLE_TIMEOUT` if set. |
| `RESULTS_DIR` | `test/<suite>/results/<timestamp>`. |

Results include a log and server log for each unit. Configure/build logs live
under the build directory. A build failure records `SKIP:build-failed` for that
project and makes the runner return nonzero; an old executable is never run
after a failed rebuild. Known failures must first be reproduced natively on the
same host before attributing them to Lupine. An invocation with only explicitly
disabled or known-failing units returns 77, which the optional CTest project
reports as skipped.

For remote runs, set `SERVER_MODE=ssh SERVER_HOST=<amd-host>` and optionally
`SERVER_SSH_TARGET=<user>@<amd-host>`. The executable runs locally, so the client
needs version-compatible ROCm libraries and `ROCM_ARCH` must target the server.
Programs requiring interactive input, external data, multiple processes, or
extra dependencies may need an upstream-specific invocation before they can
be included in a useful passing subset.

ROCm 7.2's `rocblas-bench` initializes ROCm SMI even when frequency reporting
is disabled. On WSL its native run can abort with `RSMI_STATUS_INIT_ERROR`
before GEMM runs. This is a native host limitation, not a Lupine known failure;
the runner reports it as a failure and leaves the native log available.

## Optional CTest registration

Prepare the selected suites using `BUILD_ONLY=1`, then configure this separate
CTest project. Export checkout/build overrides for both configure and test
commands if using nondefault paths.

```sh
BUILD_ONLY=1 test/run_hip_tests.sh catch/unit/device
cmake -S test/rocm -B build/rocm-integration -DROCM_SUITES=hip-tests
RUN_NATIVE=1 HIP_TESTS_CTEST_ARGS='-R ^Unit_hipGetDeviceCount' \
  ctest --test-dir build/rocm-integration -R 'hip-tests.catch/unit/device$' --output-on-failure
```

CTest registers available project selections without building them. It reserves
16 ports per project so `ctest -j` invocations have separate server ports. Select
only prepared projects with `-R`; unbuilt selections fail visibly. Configure
with `-DROCM_SUITES='hip-tests;rocm-examples;rocm-libraries'` to include all three.

The harness itself has CPU-only regression checks:

```sh
python3 -m unittest discover -s test -p test_rocm_runner.py -v
```
