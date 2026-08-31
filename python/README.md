# lupine Python package

CUDA on any host. The wheel bundles the LUPINE native client shims —
CUDA **driver API** (`libcuda` / `nvcuda.dll`), CUDA **runtime API**
(`libcudart` / `cudart64_13.dll`), and **NVML** — for Linux (x86_64,
aarch64), macOS (universal2), and Windows (amd64, arm64), plus a small
PyTorch adapter. No NVIDIA software, CUDA toolkit, or container runtime
is needed on the client.

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
torch / any CUDA binary ─▶ bundled shims ──RPC──▶ lupine server ─▶ GPU
```

`lupine.connect()` exports `LUPINE_SERVER` and preloads the bundled shims
with global visibility before CUDA initializes, so:

- **PyTorch builds with CUDA** keep their normal `torch.device("cuda:N")`
  dispatch; every CUDA call lands on the LUPINE shims.
- **Natively compiled CUDA code** (nvcc/clang binaries) resolves the
  bundled shims directly — including on platforms where no NVIDIA
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
- `lupine.load_native()` / `lupine.libdir()` — load/inspect the bundled
  shims without torch.
- `LUPINE_LIBDIR` — load shims from a custom directory (e.g. a newer build).
- `TRITON_LIBCUDA_PATH` — defaults to the bundled Linux shim directory so
  `torch.compile` can link Triton's launcher; an explicit value is preserved.
- `LUPINE_DISABLE_LOCAL=0` — include local GPUs (when present) in the
  topology ahead of the remote ones.

The package depends on nothing but the standard library.

## Automatic bootstrap

Install the opt-in extra to make ordinary Python processes default to the
hosted LUPINE demo without calling the LUPINE API:

```sh
pip install "lupine[auto]"
python existing_torch_program.py
```

The companion package installs a Python startup hook that sets
`LUPINE_SERVER=demo.lupinemachines.com:14833` only when the application has
not configured a server, then preloads the bundled native shims before the
application imports PyTorch. An explicit `LUPINE_SERVER` always wins. Set
`LUPINE_AUTO=0` to disable the hook for one process.

The hook does not change `LUPINE_DISABLE_LOCAL`. As with the explicit API,
PyTorch must have a compiled CUDA backend; a CPU-only PyTorch build cannot
gain one at runtime.

## Layout

```
lupine/
  __init__.py    Session / connect() adapter
  _native.py     platform shim discovery + preloading
  _libs/         (in wheels) per-platform native shims
```

`build.py stage <tag> <dir>` assembles `lupine/_libs/<tag>/` from
build outputs (CI does this for every platform); `build.py build`
builds the wheel. CI builds the native shims from the repository root
and `python/cudart`.
