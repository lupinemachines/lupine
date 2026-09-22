# lupine Python package

CUDA on any host. The configured LUPINE server publishes its compatible native
client for Linux (x86_64, aarch64), macOS (universal2), and Windows (amd64,
arm64): the CUDA **driver API** (`libcuda` / `nvcuda.dll`) and **NVML**. The
program's own CUDA runtime and libraries run against it: on Linux and Windows
x86_64 the CUDA `torch` wheels bundle them, and a natively compiled program
brings its own. NCCL and nvSHMEM cannot work through the driver alone, so
their shims are loaded when the bundle names them and no native copy (an
`nvidia-nccl-*` wheel or a system library) is installed.

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
- **macOS** has no CUDA torch, so `lupine.connect()` there loads the torch
  backend instead (see below): the program's own torch gets a device whose
  operators execute in a same-version CUDA torch in an arm64 Linux guest.
- **Native arm64 Python on Windows** gets the driver and NVML; run an x64
  Python for a CUDA PyTorch (see below).

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
Mac torch a GPU. `lupine.connect()` on macOS instead starts a **worker**, a
CUDA PyTorch of the *same release* as the host torch running in an arm64
Linux container against the driver shims, and loads a native torch backend
into the host interpreter:

```
host torch ──boxed aten ops, fire-and-forget──▶ lupine-torch-worker ──driver shims──▶ lupine server ─▶ GPU
```

- Tensors on the device are host-side metadata (sizes, strides, dtype,
  views, autograd) over a storage handle the worker owns. Every operator
  reaches one boxed fallback. Its first call with a given argument metadata
  is a round trip: the worker runs it and reports what each result is (a new
  storage, an argument, a view of one), and that report is the operator's
  plan for those shapes; later calls build their results from the plan over
  host-assigned handles and are forwarded fire-and-forget, applied by the
  worker in issue order. Only `.item()`, copies to the CPU,
  `torch.cuda.synchronize()` and operators whose output shape depends on
  data (`nonzero`, `masked_select`, `unique`, boolean indexing) wait for a
  reply. Storage release rides along with the next call.
- On a torch with no CUDA build the backend owns the in-tree `cuda` device:
  `torch.device("cuda")`, `.cuda()`, `torch.cuda.*`, `torch.autocast("cuda")`
  and CUDA graphs (`torch.cuda.graph`, captured and replayed by the worker)
  work unchanged, and the program prints what it prints on CUDA torch. torch
  refuses to rename its PrivateUse1 backend to an in-tree name, so this is the
  only way a program's `"cuda"` strings can keep working. A host torch that
  does have a CUDA build keeps `cuda` for the driver path and reaches the
  backend as `torch.device("lupine")`.
- `LUPINE_TORCH_BACKEND=1` selects the backend on any platform; the worker is
  then a subprocess of `LUPINE_WORKER_PYTHON`, an interpreter with the CUDA
  torch of the same release (Linux, no container). `LUPINE_WORKER=host:port`
  attaches to a worker started by hand (`lupine-torch-worker --listen ...`).
- On macOS the worker image is `ghcr.io/lupinemachines/lupine-pytorch-worker:torch<release>-cu<xyz>`,
  selected from the host torch release and the CUDA version the server
  advertises; `LUPINE_WORKER_IMAGE` overrides it and `LUPINE_WORKER_RUNTIME`
  picks `container` (Apple Container), `docker`, `podman` or `nerdctl`
  (default: the first one installed). The worker's port is published on the
  host loopback. The client bundle inside the guest comes from the server's
  bundle negotiation like any other client.
- The extension in `lupine/_backend` is built per torch release from a
  repository checkout (`python python/lupine/_backend/setup.py build_ext --inplace`,
  needs `libnghttp2`); the worker image builds it the same way. Host and
  worker torch releases must match exactly (`2.12.1+cpu` and `2.12.1+cu130`
  do); the boxed operator schema is the wire contract.

Limits: CUDA extensions compiled against the host torch (flash-attn, apex,
hand-written kernels) cannot run, since the host has no CUDA; `torch.compile`
runs eagerly; sparse and quantized layouts are not supported; explicit
`torch.Generator` objects are not forwarded (`torch.manual_seed` is);
`torch.Generator(device="cuda")` is unavailable on a torch without CUDA.
`LUPINE_TORCH_PROFILE=<path>` writes per-op host timings, `LUPINE_TORCH_TRACE=1`
logs every forwarded op, and `LUPINE_WORKER_RPC_STATS=<path>` gives the
worker's `LUPINE_RPC_STATS`.

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
  _backend/      torch backend: host kernels + worker dispatch (C++), device module
  _worker.py     lupine-torch-worker entry point
  _guest.py      worker provisioning (subprocess or container) and image selection
  container.py   Apple Container / Docker / Podman / nerdctl launch
```

No native object ships in the wheel. Server workflows publish the clients
for CMake to embed, and the bundle manifest names the shims it carries; the
loader loads the driver, NVML, and the conditional NCCL / nvSHMEM shims by
those names.
