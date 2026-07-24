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
    if not host.startswith("[") and host.count(":") > 1:
        host = f"[{host}]"
    if port is not None:
        return f"{host}:{int(port)}"
    if host.startswith("[") and "]:" in host:
        return host
    if host.count(":") == 1:
        return host
    return f"{host}:{DEFAULT_PORT}"


def _normalize_hosts(host: str | Sequence[str], port: int | None = None) -> tuple[str, ...]:
    if isinstance(host, str):
        return (_normalize_server(host, port),)
    return tuple(_normalize_server(item, port) for item in host)


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


@dataclass
class Session:
    """A process-local LUPINE connection declaration."""

    servers: tuple[str, ...]
    libcuda: str | os.PathLike[str] | None = None

    def __enter__(self) -> "Session":
        self._previous_server = os.environ.get("LUPINE_SERVER")
        if not self.servers:
            return self
        _require_mutable_config()
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
        self._restore_env()
        return False

    def _restore_env(self) -> None:
        if getattr(self, "_previous_server", None) is None:
            os.environ.pop("LUPINE_SERVER", None)
        else:
            os.environ["LUPINE_SERVER"] = self._previous_server

    def devices(self) -> list[Any]:
        """Return every GPU in LUPINE's native virtual device topology."""

        if not self.servers:
            return []
        torch = _torch()
        count = int(torch.cuda.device_count())
        return [torch.device("cuda", index) for index in range(count)]


def connect(
    *,
    host: str | Sequence[str],
    port: int | None = None,
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
    if not servers:
        return Session(servers=servers, libcuda=libcuda)
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
        libcuda=libcuda,
    )


def devices() -> list[Any]:
    """Return devices in PyTorch's current native CUDA topology."""

    torch = _torch()
    count = int(torch.cuda.device_count())
    return [torch.device("cuda", index) for index in range(count)]


def servers() -> tuple[str, ...]:
    """Return configured LUPINE servers from ``LUPINE_SERVER``."""

    return _servers_from_env()


def is_configured() -> bool:
    """Return true when ``LUPINE_SERVER`` names at least one server."""

    return bool(servers())


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
    "devices",
    "is_configured",
    "sidecar",
    "servers",
]
