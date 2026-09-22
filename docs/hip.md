# HIP hello-world

The HIP shim can run compiler-registered kernels with no arguments on the
server's AMD GPU, including `HIP-Basic/hello_world` from ROCm/rocm-examples.
Device `printf` output is returned to the client's stdout.

## Build and test locally

With ROCm installed at `/opt/rocm`, build the backend and run the opt-in test:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLUPINE_BUILD_CUDA=OFF -DLUPINE_BUILD_NVML=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build --parallel
HIP_TEST_ARCH=gfx1100 test/run_hip_hello_world_test.sh --native
HIP_TEST_ARCH=gfx1100 test/run_hip_hello_world_test.sh
```

Use the server GPU's architecture in place of `gfx1100`. The second test starts
a local server, links the client against the shim, and verifies device output,
stream launches, module reload after device reset, and rejection of a kernel
requiring an argument. GPU tests are not added to CI; `hip_registration_test`
only tests local registration, launch stacks, and bundle validation.

For an existing binary, start `build/lupine_driver_server` with the native ROCm
library directory in its `LD_LIBRARY_PATH`, then run the binary with
`LUPINE_SERVER=host:port` and the absolute `build` directory in its own
`LD_LIBRARY_PATH`. The build provides the installed SDK's runtime filename
(such as `libamdhip64.so.7`) as an alias of the shim, alongside `.so` and `.so.1`.
The client does not need access to a local GPU.

## WSL device printf

On the tested WSL host (ROCm 7.2, RX 7900 XTX), the default hostcall-based
`printf` fails with `hipErrorIllegalState` and the native runtime diagnostic
`Pcie atomics not enabled, hostcall not supported`. Compile device code with
`-mprintf-kind=buffered`. This is a compile-time choice: the server cannot repair
a binary already compiled to use hostcalls. AMD documents this alternative in
the [ROCm release notes](https://rocm.docs.amd.com/en/docs-6.0.0/about/CHANGELOG.html#non-hostcall-hip-printf).
The local test script selects buffered printf automatically on WSL.

For the upstream runner introduced in PR #905, configure hello-world with:

```sh
export ROCM_ARCH=gfx1100
export ROCM_EXAMPLES_CMAKE_ARGS='-DCMAKE_HIP_FLAGS=-mprintf-kind=buffered'
RUN_NATIVE=1 test/run_rocm_examples.sh HIP-Basic/hello_world
test/run_rocm_examples.sh HIP-Basic/hello_world
```

Allow the runner to rebuild after changing compiler flags. Check the output,
not just the exit status: the upstream example does not check the launch error
and can return zero while printing only its two host lines. A successful run
prints ten lines, including one `Hello world from device kernel block B thread T!`
for each of `(B,T) = (0,0), (0,1), (1,0), (1,1)`. Their order can vary.

## Current kernel support

- Registration is local and lazy. The first launch uploads the uncompressed
  Clang offload bundle and resolves its kernel by name on the server. The
  runtime selects the GPU's code object. Raw ELF and compressed bundles are
  not supported by this upload path yet.
- Kernels must have no explicit arguments. Native HIP validates the signature
  and returns `hipErrorInvalidValue` for a kernel requiring arguments. Explicit
  module launches with `kernelParams` or `extra` return `hipErrorNotSupported`.
- Launches synchronize their stream before responding so device stdout is
  captured throughout execution. This implementation does not provide
  asynchronous kernel launches or kernel graph capture.
- Compiler registrations cache modules per device and session. Reset and
  connection closure invalidate that cache. Uploaded compiler modules remain
  on the server until session teardown; unregistering a binary does not open a
  connection from an ELF destructor. Explicit `hipModuleUnload` releases its
  uploaded image after native unload.
- As with the existing HIP forwarding API, handles used by a thread must belong
  to its current device's server. Kernel arguments and registered device globals
  need further implementation before general HIP programs can run.
