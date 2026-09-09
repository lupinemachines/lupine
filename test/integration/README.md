# GPU integration variants

Each CUDA-sample, library-sample, CUDA-Python, PyTorch, and custom test runs twice:

- `driver-only`: LUPINE's `libcuda` with NVIDIA's CUDA runtime.
- `driver-runtime`: LUPINE's `libcuda` and `libcudart`.

Build both shim targets once and set `LUPINE_LIB` to the resulting
`libcuda.so.1` before configuring this directory. CMake makes isolated symlink
directories for the two library search paths; it does not rebuild either shim.
The driver-only directory contains only `libcuda.so` and `libcuda.so.1`.
Inherited `LD_LIBRARY_PATH` and `LD_PRELOAD` are replaced for each test so a
runtime shim from the caller's environment cannot leak into driver-only runs.

Both variants reuse the same sample binaries. Custom executables are compiled
once per CTest invocation by setup fixtures, then run unchanged in both variants.
Sample builds select the shared CUDA runtime (`CMAKE_CUDA_RUNTIME_LIBRARY=Shared`
or nvcc's `--cudart shared`) so the loader can switch runtimes. Previously built
static-runtime executables must be rebuilt once with that setting; they cannot
exercise runtime interposition merely by changing `LD_LIBRARY_PATH`.
Ports, results, and nvJPEG-generated assets are separate. The two runs of a
sample cannot execute concurrently because some samples write in their shared
working directory.

The GCE workflow builds once and runs both variants in the same container and
CTest invocation. It also uploads the lane's NVIDIA runtime for the server,
independently of which client variant is running. Use
`ctest --test-dir <build> -L driver-only` or
`-L driver-runtime` to select a variant; the required custom-build fixtures run
automatically.
