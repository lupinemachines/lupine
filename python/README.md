# lupine Python package

CUDA on any host. The configured LUPINE server publishes its compatible native
client — CUDA **driver API** (`libcuda` / `nvcuda.dll`) and **NVML**, plus the
complete runtime set needed by non-Python clients — for Linux (x86_64,
aarch64), macOS (universal2), and Windows (amd64, arm64). The Python wheel
contains only the portable CUDA **runtime API** translation stubs (`libcudart`
/ `cudart64_13.dll`) and a small PyTorch adapter. No NVIDIA software, CUDA
toolkit, or container runtime is needed on the client.

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
  dispatch; every CUDA call lands on the LUPINE shims.
- **Natively compiled CUDA code** (nvcc/clang binaries) resolves the
  selected shims directly — including on platforms where no NVIDIA
  runtime has ever shipped.
- **CPU-only PyTorch builds** cannot gain a CUDA backend by linking (the
  backend is compiled out); use the shims directly via ctypes, or run such
  workloads in a container against the same server.

## API

- `lupine.connect(host=..., port=...)` — declare servers and load the native
  shims. `host` accepts one host or a list; `LUPINE_SERVER` (comma-separated)
  is the default. Returns a `Session`; usable with or without `with`.
- `session.devices()` / `session.device(i=0)` — `torch.device("cuda:N")`
  objects from LUPINE's virtual topology across all servers.
- `lupine.load_native()` / `lupine.libdir()` — load/inspect the selected
  shims without torch.
- `LUPINE_LIBDIR` — load shims from a custom directory (e.g. a newer build).
- `TRITON_LIBCUDA_PATH` — defaults to the selected shim directory so
  `torch.compile` can link Triton's launcher; an explicit value is preserved.
- `LUPINE_DISABLE_LOCAL=0` — include local GPUs (when present) in the
  topology ahead of the remote ones.

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

The hook does not change `LUPINE_DISABLE_LOCAL`. As with the explicit API,
PyTorch must have a compiled CUDA backend; a CPU-only PyTorch build cannot
gain one at runtime.

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
  _native.py     platform shim discovery + preloading
  _libs/         (in wheels) per-platform CUDA runtime stubs
```

`build.py stage-runtime <tag> <dir>` copies exactly one runtime stub into the
wheel staging tree. The Python workflow never stages complete clients. A
separate client-bundle workflow passes its driver/runtime/NVML artifacts
directly to `build.py bundles`; they never enter the Python package tree.
