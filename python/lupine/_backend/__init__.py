"""The native Lupine PyTorch backend and its process-wide device state.

Metadata and autograd live in native PyTorch. Storage and arithmetic use
local CUDA dylibs and GPU kernels through the Lupine driver client.
"""

from __future__ import annotations

from typing import Any

_started: dict[str, Any] = {}


def _extension() -> Any:
    from . import ext

    return ext.load()


def is_started() -> bool:
    backend = _started.get("backend")
    return backend is not None and not backend.runtime.closed


def info() -> dict[str, Any]:
    return dict(_started.get("info", {}))


def start() -> dict[str, Any]:
    from . import native

    return native.start()


def close() -> None:
    backend = _started.get("backend")
    if backend is not None:
        _extension().unregister()
        backend.close()
