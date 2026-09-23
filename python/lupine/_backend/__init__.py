"""Host side of the LUPINE torch backend.

The user's ``torch`` gets a device whose operators execute in a CUDA torch
of the same major.minor release running in the worker (``lupine._worker``).
Tensors are host-side metadata over a storage handle the worker owns; a
Python boxed fallback (``forward``) sends every operator over a plain
socket (``transport``) pickled with tensor references (``wire``). The first
call of an operator with a given argument metadata is a round trip that
teaches the host what the results look like; later calls build their results
from that plan over host-assigned handles and go fire-and-forget.
Synchronous round trips otherwise happen only where torch itself waits
(``.item()``, copies to the CPU, ``synchronize()``) and for operators whose
output shape depends on data.

The C++ extension (``ext``, ``csrc``) is the device registration only:
allocator, device guard, hooks, generator and the metadata kernels, which
libtorch has no stable ABI for and so are built per torch release.

When the host torch has no CUDA build (every macOS torch, or a CPU wheel)
the kernels are registered on the in-tree CUDA dispatch key, so
``torch.device("cuda")`` and the ``torch.cuda`` module resolve to the
backend; torch refuses to rename PrivateUse1 to an in-tree device name, so
this is the only way the program's ``"cuda"`` strings can keep working. A
host torch that does have a CUDA build keeps ``cuda`` for its own driver
path and reaches the backend as the PrivateUse1 device ``torch.device("lupine")``.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

_started: dict[str, Any] = {}


def _extension() -> Any:
    from . import ext

    return ext.load()


def release_of(version: str) -> tuple[int, int]:
    """``2.12.1+cu130`` -> ``(2, 12)``: the operator schema is the wire
    contract, and it only changes between minor releases."""

    parts = version.split("+", 1)[0].split(".")
    return int(parts[0]), int(parts[1])


def is_dual() -> bool:
    """Whether the backend also owns the ``cuda`` device in this process."""

    return bool(_started.get("dual"))


def is_started() -> bool:
    return bool(_started)


def info() -> dict[str, Any]:
    return dict(_started.get("info", {}))


def start(address: str) -> dict[str, Any]:
    """Connect the backend to a worker at ``host:port`` and register it."""

    from .. import LupineError

    if _started:
        if _started["address"] != address:
            raise LupineError("the torch backend is already connected to a different worker")
        return info()
    from . import ext, transport

    client = transport.Client.connect(address)
    worker = client.hello
    host_version = ext.torch_version()
    if release_of(host_version) != release_of(worker.get("torch", "0.0")):
        client.close()
        raise LupineError(
            f"torch version mismatch: this process runs torch {host_version} and "
            f"the worker runs {worker.get('torch')}; the worker must run a torch of the "
            "same major.minor release"
        )
    C = _extension()
    from . import forward

    import torch

    dual = torch.version.cuda is None
    # The worker serves no client bundle; a stale selector from an earlier
    # driver-path session in this process must not be offered to it.
    for name in ("LUPINE_CLIENT_ETAG", "LUPINE_CLIENT_PLATFORM"):
        os.environ.pop(name, None)
    backend = forward.install(client, dual, C)
    try:
        C.register(int(worker["device_count"]), dual, backend.release, backend.synchronize)
    except RuntimeError as exc:
        client.close()
        raise LupineError(str(exc)) from exc
    atexit.register(C.unregister)
    _started.update({"address": address, "dual": dual, "info": worker, "backend": backend})

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


def close() -> None:
    """Drops the worker connection; a subprocess worker then exits."""

    backend = _started.get("backend")
    if backend is not None:
        backend.close()
