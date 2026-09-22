"""Host side of the LUPINE torch backend.

The user's ``torch`` gets a device whose operators execute in a same-version
CUDA torch running in the worker (``lupine._worker``). Tensors are host-side
metadata over a remote storage handle; one boxed fallback forwards every
operator to the worker over the LUPINE RPC layer, fire-and-forget, with the
result metadata decided here by the operator's meta kernel. Synchronous
round trips happen only where torch itself waits: ``.item()``, copies to
the CPU, ``synchronize()``.

When the host torch has no CUDA build (every macOS torch, or a CPU wheel)
the kernels are registered on the in-tree CUDA dispatch key, so
``torch.device("cuda")`` and the ``torch.cuda`` module resolve to the
backend; torch refuses to rename PrivateUse1 to an in-tree device name, so
this is the only way the program's ``"cuda"`` strings can keep working. A
host torch that does have a CUDA build keeps ``cuda`` for its own driver
path and reaches the backend as the PrivateUse1 device ``torch.device("lupine")``.

The extension is built from a repository checkout (see ``setup.py``)::

    python python/lupine/_backend/setup.py build_ext --inplace
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_started: dict[str, Any] = {}


def _preload_torch_libraries() -> None:
    """Load libtorch for the extension without importing torch.

    On a torch with no CUDA build the extension has to supply the CUDA
    hooks before torch's own import first asks for them (they are created
    once, on first use), so the extension must load before ``import torch``.
    """

    import ctypes
    import importlib.util
    import sys

    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    lib = Path(next(iter(spec.submodule_search_locations))) / "lib"
    suffix = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    for name in ("libc10", "libtorch_cpu", "libtorch", "libtorch_python"):
        path = lib / f"{name}{suffix}"
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def _extension() -> Any:
    _preload_torch_libraries()
    try:
        from . import _C
    except ImportError as exc:
        from .. import LupineError

        raise LupineError(
            "the lupine torch backend extension is not built for this torch; "
            "run `python python/lupine/_backend/setup.py build_ext --inplace` "
            f"from a repository checkout ({exc})"
        ) from exc
    return _C


def base_version(version: str) -> str:
    """``2.12.1+cu130`` -> ``2.12.1``: the boxed schema is the wire contract."""

    return version.split("+", 1)[0]


def is_dual() -> bool:
    """Whether the backend also owns the ``cuda`` device in this process."""

    return bool(_started.get("dual"))


def is_started() -> bool:
    return bool(_started)


def info() -> dict[str, Any]:
    return dict(_started.get("info", {}))


def start(address: str) -> dict[str, Any]:
    """Connect the backend to a worker at ``host:port`` and register it."""

    if _started:
        if _started["address"] != address:
            from .. import LupineError

            raise LupineError(
                "the torch backend is already connected to a different worker"
            )
        return info()
    from .. import LupineError

    ext = _extension()
    import torch

    dual = torch.version.cuda is None
    # The worker serves no client bundle; a stale selector from an earlier
    # driver-path session in this process must not be offered to it.
    for name in ("LUPINE_CLIENT_ETAG", "LUPINE_CLIENT_PLATFORM"):
        os.environ.pop(name, None)
    worker = json.loads(ext.connect(address, dual))
    host_version = base_version(torch.__version__)
    worker_version = base_version(worker.get("torch", ""))
    if host_version != worker_version:
        raise LupineError(
            f"torch version mismatch: this process runs torch {torch.__version__} "
            f"and the worker runs {worker.get('torch')}; the worker must run the "
            "same torch release"
        )
    _started.update({"address": address, "dual": dual, "info": worker})

    from . import device as device_module

    if dual:
        # torch treats a named PrivateUse1 backend as the accelerator ahead of
        # everything else (autograd streams, torch.accelerator), so while the
        # backend owns the cuda key it must not also be named.
        device_module.install_as_cuda()
    else:
        torch.utils.rename_privateuse1_backend("lupine")
        torch._register_device_module("lupine", device_module)
        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
    device_module.forward_context_settings()
    device_module.keep_compile_eager()
    return info()
