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
before CUDA initializes, so:

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

Install the opt-in extra to preload LUPINE before application imports run:

```sh
pip install "lupine[auto]"
lupine run -- python existing_torch_program.py
```

The CLI acquires a lease and exports `LUPINE_SESSION` plus its regional
`LUPINE_SERVER`. The companion package installs a Python startup hook that
resolves and preloads the selected native client before the application imports
PyTorch. An explicit `LUPINE_SERVER` always wins; otherwise the hook uses
`https://api.lupine.sh`, whose bootstrap route redirects the bound session to
its regional gateway. Set `LUPINE_AUTO=0` to disable the hook for one process.

The hook does not change `LUPINE_DISABLE_LOCAL`. As with the explicit API,
PyTorch must have a compiled CUDA backend; a CPU-only PyTorch build cannot
gain one at runtime.

## Layout

```
lupine/
  __init__.py    Session / connect() adapter
  _native.py     platform shim discovery + preloading
  _libs/         (in wheels) per-platform CUDA runtime stubs
```

`build.py stage <tag> <dir>` assembles `lupine/_libs/<tag>/` from
build outputs (CI does this for every platform). Server bundles consume the
complete staged client, while the wheel build includes only the runtime stubs.
CI builds the native shims from the repository root and `python/cudart`.
