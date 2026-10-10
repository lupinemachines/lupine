# lupine Python package

CUDA on any host. The configured LUPINE server publishes its compatible native
client for Linux (x86_64, aarch64), macOS (universal2), and Windows (amd64,
arm64): the CUDA **driver API** (`libcuda` / `nvcuda.dll`) and **NVML**. The
program's own CUDA runtime and libraries run against it: on Linux and Windows
x86_64 the CUDA `torch` wheels bundle them, and a natively compiled program
brings its own. NCCL and nvSHMEM cannot work through the driver alone, so
their shims are loaded when the bundle names them and no native copy (an
`nvidia-nccl-*` wheel or a system library) is installed.

ARM64 macOS client bundles additionally carry NVIDIA's CUDA 13.3 runtime, NVRTC, and cuBLAS,
converted from its Linux binary by the vendored
[machso](https://github.com/kevmo314/machso) compiler. The runtime runs locally
and makes driver calls through Lupine. It loads with local visibility so its
embedded Linux libc cannot replace Python's native libc. Intel macOS bundles
continue to provide the native driver and NVML only.

The wheel itself is pure Python: a small PyTorch adapter and the loader that
resolves that client. No NVIDIA driver, CUDA toolkit, or container runtime is
needed on the client.

```python
import lupine

with lupine.connect(host="gpu-host:14833") as session:
    import torch

    device = session.device()
    x = torch.arange(8, device=device, dtype=torch.float32)
    print((x * 2).cpu())  # tensor([0., 2., 4., 6., ...]) — computed remotely
```

## How it works

```
torch / any CUDA binary ─▶ selected shims ──RPC──▶ lupine server ─▶ GPU
```

`lupine.connect()` exports `LUPINE_SERVER`, downloads and verifies the exact
client object selected by that server, and preloads it with global visibility
before CUDA initializes. `lupine.cloud()` first binds through the stable cloud
API, then uses the returned regional gateway for both bundle discovery and the
native HTTP/2 connection, so:

- **PyTorch builds with CUDA** keep their normal `torch.device("cuda:N")`
  dispatch; every driver call their bundled CUDA libraries make lands on the
  LUPINE driver shim.
- **Natively compiled CUDA code** (nvcc/clang binaries) resolves the driver
  shim directly.
- **CPU-only PyTorch builds** do not gain the built-in CUDA backend by linking.
  On macOS, the custom Lupine device supplies GPU operator implementations.
- **macOS** uses the native Lupine PyTorch backend described below. Operators
  run through local dylibs and GPU kernels; no torch worker or container is used.
- **Native arm64 Python on Windows** gets the driver and NVML; run an x64
  Python for a CUDA PyTorch (see below).

## ARM64 macOS CUDA runtime

With a server whose ARM64 macOS bundle includes the converted runtime, Python
can call the runtime directly. Set `LUPINE_SERVER` before starting Python:

```python
import ctypes
import lupine

libraries = lupine.load_native(missing_ok=False)
runtime = ctypes.CDLL(libraries["libcudart.dylib"], mode=ctypes.RTLD_LOCAL)
runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
runtime.cudaGetDeviceCount.restype = ctypes.c_int
count = ctypes.c_int()
assert runtime.cudaGetDeviceCount(ctypes.byref(count)) == 0
print(count.value)
```

The native-client workflow prepares the pinned ARM64 Linux runtime, NVRTC, cuBLAS, and their
dependency closure on Ubuntu, then links and signs it in separate macOS steps.
`third_party/machso/UPSTREAM` records the vendored revision. The preparation
script also accepts repeatable `--package name==version` arguments. It
resolves their wheel dependencies, records exact versions and wheel hashes in
`packages.lock`, and extracts every ELF library as an additional root. Use
`--requirements packages.lock` to replay those inputs with hash verification. It discovers
their dependency closure from ELF metadata, and builds one shared libc/TLS
image. Late-loaded helper SONAMEs resolve to this image automatically. The pure Python
wheel still gets its native objects from the selected server. This runtime
does not supply PyTorch's compiled CUDA operators. Native macOS PyTorch can
use the custom Lupine backend below.

The GPU integration checks in `tests/test_macos_runtime.py` cover discovery,
allocation, copies, memset, and coexistence with native CPU PyTorch. Run them
on ARM64 macOS with `LUPINE_SERVER` configured and
`LUPINE_TEST_MACOS_RUNTIME=1`; an optional `LUPINE_LIBDIR` selects a local
client bundle. These checks do not validate PyTorch CUDA operators or kernel
launches.

Additional CUDA packages can be prepared with the same pipeline. For example,
`--package nvidia-cufft==12.3.0.29 --package nvidia-curand==10.4.3.29
--package nvidia-nvjitlink==13.3.33` includes the FFT/RNG libraries and pins
cuFFT's declared linker dependency. `tests/test_macos_cuda_libraries.py` checks
GPU FFT/RNG results and clean process teardown when run on ARM64 macOS with
`LUPINE_TEST_MACOS_CUDA_LIBRARIES=1` and a matching `LUPINE_LIBDIR`.

## Native Lupine PyTorch backend (experimental)

`lupine.connect()` on ARM64 macOS starts the native PyTorch backend and
runs supported operators through the dylibs, without a torch worker or container. Use
`session.device()` or `device="lupine"`. This backend uses PyTorch's
PrivateUse1 registration, leaving its built-in CUDA and MPS device names
intact. Call `lupine.connect()` before importing torch.

```python
import lupine

with lupine.connect(host="gpu-host:14833") as session:
    import torch
    a = torch.arange(12, device=session.device(), dtype=torch.float32).reshape(3, 4)
    a.requires_grad_()
    b = torch.ones((4, 2), device=session.device())
    loss = ((a @ b) * 2).sum()
    loss.backward()
    print(a.grad.cpu())
    print(torch.lupine.get_device_name())
```

The backend reuses the device guards, hooks, allocator handles, and metadata
kernels introduced in commit `f64a5b8bd`. Each handle now owns a real CUDA
allocation. The converted runtime handles memory and copies; GPU PTX kernels
handle tensor arithmetic through `libcuda.dylib`. Converted NVRTC compiles the
shipped CUDA source locally on the Mac; older runtime bundles use the included
PTX. Matrix multiplication uses converted cuBLAS when the bundle provides it,
with a basic GPU kernel for older bundles. It does not require Linux libtorch.
The host extension is compiled against the installed native torch; it requires a C++ compiler
(Xcode Command Line Tools) and `ninja`.

The initial scope is one GPU, at most eight dimensions, float32 arithmetic,
int64 transfers/fill/arange, broadcasting, views, in-place arithmetic, matrix
multiplication, sum/mean, and backward passes composed from those operators.
It also supports seeded GPU uniform/normal RNG, common unary functions,
ReLU/sigmoid/tanh and their gradients, softmax/log-softmax and their gradients,
batched matrix multiplication, concatenation, layer normalization with
gradients, grouped 2D convolution and its gradients, math attention with
float masks (including fully masked rows) and no dropout, and linear layer training.
Unsupported operators raise `NotImplementedError`. Mixed precision,
transposed/1D/3D convolution, fused attention, CUDA graphs, and `torch.compile`
are not supported.
Kernels run synchronously; convolution currently uses basic GPU kernels, and
older bundles without cuBLAS also use a basic matrix multiplication kernel;
this is a correctness prototype rather than a performance implementation.

PTX is checked in alongside `lupine/_backend/native/kernels.cu`, targeting
compute capability 7.5 or newer. Regenerate it on a CUDA 13.3 Linux build host:

```sh
nvcc --ptx --gpu-architecture=compute_75 --fmad=false \
  -o lupine/_backend/native/kernels.ptx lupine/_backend/native/kernels.cu
```

Run `tests/test_native_torch.py` separately on the Mac with
`LUPINE_TEST_NATIVE_TORCH=1` and the server and bundle configured. It checks GPU arithmetic, strided views, exact int64
copies, reductions, matrix multiplication and its gradients against CPU
PyTorch results, and explicit rejection of unsupported operators.

## Windows on ARM

NVIDIA publishes no CUDA runtime for Windows arm64, so a CUDA PyTorch there
is the x64 `torch` wheel with its bundled NVIDIA DLLs, running in an **x64
Python** under the OS's built-in x64 emulation. The loader keys the bundle on
the interpreter, not the machine: an x64 Python selects the `windows/amd64`
client (`platform.machine()` reports the machine's `ARM64` for it, which
would pick the wrong one), a native arm64 Python selects `windows/arm64`,
which carries the driver and NVML. The shim is user mode and the GPU
work is remote, so emulated CPU speed is acceptable.

```powershell
# Windows arm64: install an x64 Python (python.org "Windows installer (64-bit)"),
# then the x64 CUDA torch wheel and lupine into it.
py -3.12-64 -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\pip install lupine
```

A native arm64 Python can still use the driver shim directly (ctypes, or a
program built against the driver API); `session.device()` on one explains
that CUDA torch needs the x64 interpreter.

## API

- `lupine.connect(host=..., port=...)` — declare servers and load the native
  shims. `host` accepts one host or a list; `LUPINE_SERVER` (comma-separated)
  is the default. Returns a `Session`; usable with or without `with`.
- `session.devices()` / `session.device(i=0)` — Lupine devices on macOS,
  or CUDA devices for a CUDA-enabled PyTorch on Linux/Windows.
- `lupine.load_native()` / `lupine.libdir()` — load/inspect the selected
  shims without torch.
- `LUPINE_LIBDIR` — load shims from a custom directory (e.g. a newer build).
  The directory is filtered by the same names as a bundle: the driver and
  NVML load, and NCCL and nvSHMEM when present and not installed natively.
- `TRITON_LIBCUDA_PATH` — defaults to the selected shim directory so
  `torch.compile` can link Triton's launcher; an explicit value is preserved.

The package depends on nothing but the standard library.

## Automatic bootstrap

Authenticate once and install the opt-in extra:

```sh
uvx lupine login
# Or, when lupine is installed: python -m lupine login
pip install "lupine[auto]"
python existing_torch_program.py
```

The companion package installs a lazy Python startup hook. Immediately before
the first CUDA consumer is imported, it reads the credential shared with the
Lupine CLI, acquires and binds a lease, starts its heartbeat, and preloads the
server-selected native client. If no credential exists it leaves Python running
and prints a hint to run one of the login commands above. For CI and other
headless environments, set `LUPINE_API_TOKEN`.

An explicit `LUPINE_SERVER` still wins. An externally managed
`LUPINE_SESSION` must be accompanied by its bind-selected `LUPINE_SERVER`. Use
`LUPINE_GPU_TYPE`, `LUPINE_GPU_COUNT`, and `LUPINE_REGION` to constrain automatic
placement, and `LUPINE_AUTO=0` to disable the hook for one process.

On macOS use `session.device()` or `device="lupine"` with the native backend.
Linux and Windows CUDA consumers use their compiled CUDA backend.

## Authenticated cloud API

Applications that want explicit lease lifetime can use the main package without
the automatic extra:

```python
import lupine

with lupine.cloud(gpu_type="RTX_4090") as session:
    import torch

    value = torch.ones(4, device=session.device())
```

`lupine.cloud()` uses `LUPINE_API_TOKEN` first, then the credential stored by
`lupine login`, and releases its process-owned lease when the context exits.
The bearer token is used only for coordinator API calls. Bind returns the
regional gateway used for bundle and native RPC traffic, which carry the lease
ID in `LUPINE_SESSION`.

## Layout

```
lupine/
  __init__.py    Session / connect() adapter
  _bundles.py    server bundle resolution, verification, and caching
  _native.py     shim discovery + preloading
  _backend/      native PyTorch device registration, dylib bindings, and GPU kernels
```

No native object ships in the wheel. Server workflows publish the clients
for CMake to embed, and the bundle manifest names the shims it carries; the
loader loads the driver, NVML, and the conditional NCCL / nvSHMEM shims by
those names.
