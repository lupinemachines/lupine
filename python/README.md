# lupine Python package

CUDA on any host. The configured LUPINE server publishes its compatible native
client for Linux (x86_64, aarch64), macOS (universal2), and Windows (amd64,
arm64): the CUDA **driver API** (`libcuda` / `nvcuda.dll`) and **NVML**. The
program's own CUDA runtime and libraries run against it: on Linux and Windows
x86_64 the CUDA `torch` wheels bundle them, and a natively compiled program
brings its own. NCCL and nvSHMEM cannot work through the driver alone, so
their shims are loaded when the bundle names them and no native copy (an
`nvidia-nccl-*` wheel or a system library) is installed.

ARM64 macOS client bundles additionally carry NVIDIA's CUDA 13.3 runtime,
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
- **CPU-only PyTorch builds** cannot gain a CUDA backend by linking (the
  backend is compiled out); use the driver shim directly via ctypes, or run
  such workloads in a container against the same server.
- **macOS** defaults to the worker torch backend: the program's own torch
  gets a device whose operators execute in a same-version CUDA torch running
  elsewhere. `LUPINE_TORCH_BACKEND=native` instead selects the experimental
  dylib-backed Lupine device described below.
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

The native-client workflow prepares the pinned ARM64 Linux runtime and its
dependency closure on Ubuntu, then links and signs it in separate macOS steps.
`third_party/machso/UPSTREAM` records the vendored revision. The pure Python
wheel still gets its native objects from the selected server. This runtime
does not supply PyTorch's compiled CUDA operators. Native macOS PyTorch can
instead use the custom Lupine backend below; the worker backend remains
available for broader operator coverage.

The GPU integration checks in `tests/test_macos_runtime.py` cover discovery,
allocation, copies, memset, and coexistence with native CPU PyTorch. Run them
on ARM64 macOS with `LUPINE_SERVER` configured and
`LUPINE_TEST_MACOS_RUNTIME=1`; an optional `LUPINE_LIBDIR` selects a local
client bundle. These checks do not validate PyTorch CUDA operators or kernel
launches.

## Native Lupine PyTorch backend (experimental)

Set `LUPINE_TORCH_BACKEND=native` on ARM64 macOS to run supported PyTorch
operators through the dylibs, without a torch worker or container. Use
`session.device()` or `device="lupine"`. This backend uses PyTorch's
PrivateUse1 registration, leaving its built-in CUDA and MPS device names
intact. Call `lupine.connect()` before importing torch.

```python
import os
os.environ["LUPINE_TORCH_BACKEND"] = "native"
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
handle tensor arithmetic through `libcuda.dylib`. It needs neither a Linux
libtorch nor converted cuBLAS/cuDNN for these operators. The host extension
is compiled against the installed native torch; it requires a C++ compiler
(Xcode Command Line Tools) and `ninja`, as the worker backend does.

The initial scope is one GPU, at most eight dimensions, float32 arithmetic,
int64 transfers/fill/arange, broadcasting, views, in-place arithmetic, matrix
multiplication, sum/mean, and backward passes composed from those operators.
Unsupported operators raise `NotImplementedError`. RNG, mixed precision,
convolutions, attention, CUDA graphs, and `torch.compile` are not supported.
Kernels run synchronously and matrix multiplication uses a simple kernel;
this is a correctness prototype rather than a performance implementation.

PTX is checked in alongside `lupine/_backend/native/kernels.cu`, targeting
compute capability 7.5 or newer. Regenerate it on a CUDA 13.3 Linux build host:

```sh
nvcc --ptx --gpu-architecture=compute_75 --fmad=false \
  -o lupine/_backend/native/kernels.ptx lupine/_backend/native/kernels.cu
```

Run `tests/test_native_torch.py` separately on the Mac with
`LUPINE_TEST_NATIVE_TORCH=1`, `LUPINE_TORCH_BACKEND=native`, and the server
and bundle configured. It checks GPU arithmetic, strided views, exact int64
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

## Torch backend (macOS, or `LUPINE_TORCH_BACKEND=1`)

PyTorch has no CUDA build for macOS, so the driver shims alone cannot give a
Mac torch a GPU. `lupine.connect()` on macOS instead connects to a
**worker**, a CUDA PyTorch of the same major.minor release as the host torch
running against the driver shims, and gives the host torch a device whose
operators execute there:

```
host torch ──pickled aten ops, fire-and-forget──▶ lupine-torch-worker ──driver shims──▶ lupine server ─▶ GPU
```

- Tensors on the device are host-side metadata (sizes, strides, dtype,
  views, autograd) over a storage handle the worker owns. Every operator
  reaches one Python fallback (`lupine._backend.forward`, registered with
  `torch.library`). Its first call with a given argument metadata is a round
  trip: the worker runs it and reports what each result is (a new storage,
  an argument, a view of one), and that report is the operator's plan for
  those shapes; later calls build their results from the plan over
  host-assigned handles and are forwarded fire-and-forget, applied by the
  worker in issue order. Only `.item()`, copies to the CPU,
  `torch.cuda.synchronize()` and operators whose output shape depends on
  data (`nonzero`, `masked_select`, `unique`, boolean indexing) wait for a
  reply. Storage release rides along with the next call. A call travels as
  `pickle` with tensors replaced by handle descriptors, over a plain socket
  (`lupine._backend.transport`); the worker (`lupine._worker`) is pure
  Python and resolves each op as `torch.ops.<ns>.<name>.<overload>`.
- On a torch with no CUDA build the backend owns the in-tree `cuda` device:
  `torch.device("cuda")`, `.cuda()`, `torch.cuda.*`, `torch.autocast("cuda")`
  and CUDA graphs (`torch.cuda.graph`, captured and replayed by the worker)
  work unchanged, and the program prints what it prints on CUDA torch. torch
  refuses to rename its PrivateUse1 backend to an in-tree name, so this is the
  only way a program's `"cuda"` strings can keep working. A host torch that
  does have a CUDA build keeps `cuda` for the driver path and reaches the
  backend as `torch.device("lupine")`.
- `LUPINE_TORCH_BACKEND=1` selects the backend on any platform; the worker is
  then a subprocess of `LUPINE_WORKER_PYTHON`, an interpreter with a CUDA
  torch of the same major.minor release. `LUPINE_WORKER=host:port` attaches
  to a worker started by hand (`lupine-torch-worker --listen host:port` on a
  Linux machine with that torch); on macOS, which has no CUDA torch to run as
  a subprocess, that is the only way until the container worker lands (its
  own PR, branch `python/torch-worker-container`), and `lupine.connect()`
  says so.
- What is compiled is the device registration alone
  (`lupine/_backend/csrc`, a few hundred lines against libtorch: allocator,
  device guard, hooks, generator and the metadata kernels), because torch's
  stable ABI has no way to register a device. It is built with the CPython
  limited API, so one artifact serves one torch release on one platform for
  every Python, and it comes two ways: a JIT build on first use, cached
  under `~/.cache/lupine/torch-backend/<torch version>-<platform>/` (needs a
  C++ compiler, the Xcode Command Line Tools on macOS or gcc/clang on Linux,
  and `pip install ninja`; `python -m lupine._backend.ext` runs it by hand),
  or a prebuilt `_C.abi3.so` from `python python/lupine/_backend/setup.py
  build_ext --inplace` (or the abi3 wheel `bdist_wheel` makes), found ahead
  of the cache. Host and worker torch must share a major.minor release
  (`2.12.1+cpu` with `2.12.0+cu130` works); a mismatch is refused when the
  backend connects, and an operator the worker's torch does not have is
  refused at that operator, naming both versions.

Limits: CUDA extensions compiled against the host torch (flash-attn, apex,
hand-written kernels) cannot run, since the host has no CUDA; `torch.compile`
runs eagerly; sparse and quantized layouts are not supported; explicit
`torch.Generator` objects are not forwarded (`torch.manual_seed` is);
`torch.Generator(device="cuda")` is unavailable on a torch without CUDA.
`LUPINE_TORCH_PROFILE=<path>` writes per-op host timings (and the message
count), `LUPINE_WORKER_TORCH_PROFILE=<path>` the worker's per-op decode and
run timings, `LUPINE_TORCH_TRACE=1` logs every forwarded op, and
`LUPINE_WORKER_RPC_STATS=<path>` gives the worker's `LUPINE_RPC_STATS`.
`LUPINE_TEST_TORCH_BACKEND=1 pytest tests/test_backend.py` runs the backend
end to end against a worker on this machine's CPU torch (no GPU needed).

## API

- `lupine.connect(host=..., port=...)` — declare servers and load the native
  shims. `host` accepts one host or a list; `LUPINE_SERVER` (comma-separated)
  is the default. Returns a `Session`; usable with or without `with`.
- `session.devices()` / `session.device(i=0)` — `torch.device("cuda:N")`
  objects from LUPINE's virtual topology across all servers.
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

As with the explicit API, PyTorch must have a compiled CUDA backend; a CPU-only
PyTorch build cannot gain one at runtime.

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
  _backend/      torch backend: Python fallback, wire, transport, device module; C++ device registration
  _worker.py     lupine-torch-worker entry point
  _guest.py      worker provisioning (subprocess) and attach by address
```

No native object ships in the wheel. Server workflows publish the clients
for CMake to embed, and the bundle manifest names the shims it carries; the
loader loads the driver, NVML, and the conditional NCCL / nvSHMEM shims by
those names.
