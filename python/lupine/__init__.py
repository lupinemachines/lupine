"""PyTorch adapter helpers for LUPINE.

The adapter returns ordinary ``torch.device("cuda:N")`` objects. PyTorch stays
on its built-in CUDA dispatch path while LUPINE handles CUDA driver calls below
it.
"""

from __future__ import annotations

import os
import ctypes
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_PORT = 14833


class LupineError(RuntimeError):
    """Raised when the LUPINE adapter cannot select a usable device."""


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise LupineError("PyTorch is required to use LUPINE devices.") from exc
    return torch


def _cuda_initialized() -> bool:
    try:
        return bool(_torch().cuda.is_initialized())
    except LupineError:
        return False


def _has_native_cuda_backend() -> bool:
    try:
        return _torch().version.cuda is not None
    except LupineError:
        return False


def _require_mutable_config() -> None:
    if _cuda_initialized():
        raise LupineError("connect to LUPINE before PyTorch initializes CUDA")


def _normalize_server(host: str, port: int | None = None) -> str:
    host = str(host).strip()
    if not host:
        raise LupineError("host must not be empty")
    if port is not None:
        return f"{host}:{int(port)}"
    if host.startswith("[") and "]:" in host:
        return host
    if host.count(":") == 1:
        return host
    return f"{host}:{DEFAULT_PORT}"


def _normalize_hosts(host: str | Sequence[str], port: int | None = None) -> tuple[str, ...]:
    if isinstance(host, str):
        servers = (_normalize_server(host, port),)
    else:
        servers = tuple(_normalize_server(item, port) for item in host)
    if not servers:
        raise LupineError("at least one LUPINE host is required")
    if len(set(servers)) != len(servers):
        raise LupineError("LUPINE hosts must be unique")
    return servers


def _set_server_env(servers: Sequence[str]) -> None:
    os.environ["LUPINE_SERVER"] = ",".join(servers)


def _default_libcuda() -> Path | None:
    override = os.environ.get("LUPINE_LIBCUDA")
    if override:
        return Path(override)
    repo_candidate = Path(__file__).resolve().parents[2] / "build" / "libcuda.so.1"
    if repo_candidate.exists():
        return repo_candidate
    return None


def _load_libcuda(path: str | os.PathLike[str] | None) -> None:
    libcuda = Path(path) if path is not None else _default_libcuda()
    if libcuda is None:
        return
    if not libcuda.exists():
        raise LupineError(f"LUPINE libcuda does not exist: {libcuda}")
    ctypes.CDLL(str(libcuda), mode=ctypes.RTLD_GLOBAL)


def _servers_from_env() -> tuple[str, ...]:
    value = os.environ.get("LUPINE_SERVER", "")
    return tuple(server.strip() for server in value.split(",") if server.strip())


def _cuda_device_count(*, require_available: bool = False) -> int:
    count = int(_torch().cuda.device_count())
    if require_available and count <= 0:
        raise LupineError(
            "PyTorch does not see any CUDA devices. Check that the LUPINE "
            "client library is selected and LUPINE_SERVER is configured."
        )
    return count


def _cuda_device(index: int) -> Any:
    return _torch().device("cuda", int(index))


def _checked_cuda_device(index: int, count: int) -> Any:
    index = int(index)
    if index < 0 or index >= count:
        raise LupineError(
            f"CUDA device index {index} is out of range for {count} devices"
        )
    return _cuda_device(index)


@dataclass
class Session:
    """A process-local LUPINE connection declaration."""

    servers: tuple[str, ...]
    require_available: bool = False
    libcuda: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if not self.servers:
            raise LupineError("at least one LUPINE host is required")
        self._topology_queried = False

    def __enter__(self) -> "Session":
        _require_mutable_config()
        self._previous_server = os.environ.get("LUPINE_SERVER")
        configured = _servers_from_env()
        if configured and configured != self.servers:
            raise LupineError(
                "LUPINE_SERVER is already configured differently; start a new "
                "process or pass the same hosts to lupine.connect()."
            )
        if not configured:
            _set_server_env(self.servers)
        _load_libcuda(self.libcuda)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if not _cuda_initialized() and not self._topology_queried:
            self._restore_env()
        return False

    def _restore_env(self) -> None:
        if getattr(self, "_previous_server", None) is None:
            os.environ.pop("LUPINE_SERVER", None)
        else:
            os.environ["LUPINE_SERVER"] = self._previous_server

    def _device_count(self, *, require_available: bool) -> int:
        # Lupine fixes its process-wide virtual device table on the first
        # topology query, even though PyTorch does not mark CUDA initialized.
        self._topology_queried = True
        return _cuda_device_count(require_available=require_available)

    def devices(self, *, require_available: bool | None = None) -> list[Any]:
        """Return every GPU in LUPINE's native virtual device topology."""

        check = self.require_available if require_available is None else require_available
        count = self._device_count(require_available=check)
        return [_cuda_device(index) for index in range(count)]

    def device(self, index: int = 0, *, require_available: bool | None = None) -> Any:
        """Return one GPU from LUPINE's native virtual device topology."""

        check = self.require_available if require_available is None else require_available
        count = self._device_count(require_available=check)
        return _checked_cuda_device(index, count)


def connect(
    *,
    host: str | Sequence[str],
    port: int | None = None,
    require_available: bool = False,
    libcuda: str | os.PathLike[str] | None = None,
) -> Any:
    """Create a LUPINE session for one or more remote GPU hosts.

    Use the session before any PyTorch CUDA operation:

    ``with lupine.connect(host=["a:14833", "b:14833"]) as s:``

    ``s.devices()`` then returns every CUDA ordinal in LUPINE's native virtual
    device topology.

    On macOS with a CPU-only PyTorch build, ``connect()`` automatically returns
    a sidecar session backed by Apple's container runtime.
    """

    servers = _normalize_hosts(host, port)
    if not _has_native_cuda_backend():
        if sys.platform != "darwin":
            raise LupineError(
                "PyTorch is not compiled with CUDA and automatic LUPINE sidecar "
                "fallback is only supported on macOS."
            )
        if len(servers) != 1:
            raise LupineError("automatic LUPINE sidecar fallback supports one host")
        if libcuda is not None:
            raise LupineError("libcuda is only supported with native CUDA PyTorch")
        return sidecar(server=servers[0])

    return Session(
        servers=servers,
        require_available=require_available,
        libcuda=libcuda,
    )


def devices(*, require_available: bool = True) -> list[Any]:
    """Return devices for the current ``LUPINE_SERVER`` environment."""

    if not _servers_from_env():
        raise LupineError("LUPINE_SERVER is not configured")
    count = _cuda_device_count(require_available=require_available)
    return [_cuda_device(index) for index in range(count)]


def device(index: int = 0, *, require_available: bool = True) -> Any:
    """Return one device for the current ``LUPINE_SERVER`` environment."""

    if not _servers_from_env():
        raise LupineError("LUPINE_SERVER is not configured")
    count = _cuda_device_count(require_available=require_available)
    return _checked_cuda_device(index, count)


def servers() -> tuple[str, ...]:
    """Return configured LUPINE servers from ``LUPINE_SERVER``."""

    return _servers_from_env()


def is_configured() -> bool:
    """Return true when ``LUPINE_SERVER`` names at least one server."""

    return bool(servers())


def is_available() -> bool:
    """Return true when PyTorch sees at least one CUDA device."""

    try:
        torch = _torch()
    except LupineError:
        return False
    return bool(torch.cuda.is_available())


def device_count() -> int:
    """Return the native virtual CUDA device count visible to PyTorch."""

    return _cuda_device_count()


def current_device() -> int:
    """Return PyTorch's current CUDA device index."""

    torch = _torch()
    return int(torch.cuda.current_device())


def synchronize(index: int = 0) -> None:
    """Synchronize a LUPINE-backed CUDA device."""

    torch = _torch()
    torch.cuda.synchronize(_cuda_device(index))


def sidecar(
    server: str | None = None,
    *,
    image: str | None = None,
    runtime: str = "auto",
    platform: str = "linux/arm64",
    rosetta: bool = False,
    env: dict[str, str] | None = None,
) -> Any:
    """Create a session-scoped sidecar PyTorch worker frontend."""

    from .sidecar import DEFAULT_IMAGE, sidecar as _sidecar

    return _sidecar(
        server=server,
        image=image or DEFAULT_IMAGE,
        runtime=runtime,
        platform=platform,
        rosetta=rosetta,
        env=env,
    )


__all__ = [
    "DEFAULT_PORT",
    "LupineError",
    "Session",
    "connect",
    "current_device",
    "device",
    "device_count",
    "devices",
    "is_available",
    "is_configured",
    "sidecar",
    "servers",
    "synchronize",
]
