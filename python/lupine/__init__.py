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
from urllib.parse import urlsplit, urlunsplit

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
    if host.startswith(("http://", "https://")):
        parsed = urlsplit(host)
        if not parsed.netloc:
            raise LupineError("host must include a server name")
        netloc = parsed.netloc
        if port is not None or (parsed.scheme == "http" and parsed.port is None):
            endpoint_port = int(port) if port is not None else DEFAULT_PORT
            hostname = parsed.hostname or ""
            if ":" in hostname and not hostname.startswith("["):
                hostname = f"[{hostname}]"
            netloc = f"{hostname}:{endpoint_port}"
        return urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
        )
    if not host.startswith("[") and host.count(":") > 1:
        host = f"[{host}]"
    has_port = "]:" in host if host.startswith("[") else host.count(":") == 1
    if port is not None:
        # An explicit port replaces the one in the host, as it does for URLs.
        return f"{host.rsplit(':', 1)[0] if has_port else host}:{int(port)}"
    if has_port:
        return host
    return f"{host}:{DEFAULT_PORT}"


def _normalize_hosts(
    host: str | Sequence[str], port: int | None = None
) -> tuple[str, ...]:
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

    def device(self, index: int = 0) -> Any:
        """Return one GPU from LUPINE's native virtual device topology."""

        torch = _torch()
        count = int(torch.cuda.device_count()) if self.servers else 0
        return torch.device("cuda", range(count)[index])


def _create_sidecar(
    server: str | None,
    *,
    image: str | None = None,
    runtime: str = "auto",
    platform: str | None = None,
    rosetta: bool = False,
    env: dict[str, str] | None = None,
) -> Any:
    from .sidecar import sidecar as create_sidecar

    return create_sidecar(
        server=server,
        image=image,
        runtime=runtime,
        platform=platform,
        rosetta=rosetta,
        env=env,
    )


def connect(
    *,
    host: str | Sequence[str] | None = None,
    port: int | None = None,
    libcuda: str | os.PathLike[str] | None = None,
    sidecar: bool | None = None,
) -> Any:
    """Create a LUPINE session for one or more remote GPU hosts.

    Use the session before any PyTorch CUDA operation:

    ``with lupine.connect(host=["a:14833", "b:14833"]) as s:``

    ``host`` defaults to ``LUPINE_SERVER``, which is how a launcher such as
    ``lupine run`` hands the session its already-bound hosts.

    ``s.devices()`` then returns every CUDA ordinal in LUPINE's native virtual
    device topology.

    ``sidecar=None`` automatically selects the sidecar on macOS when PyTorch
    has no native CUDA backend. Pass ``sidecar=True`` to force the sidecar or
    ``sidecar=False`` to force the native CUDA path.
    """

    if sidecar is not None and not isinstance(sidecar, bool):
        raise TypeError("sidecar must be True, False, or None")

    if host is None:
        host = _servers_from_env()
        if not host:
            raise LupineError("pass host=... or set LUPINE_SERVER")

    servers = _normalize_hosts(host, port)
    if not servers:
        if sidecar is True:
            raise LupineError("sidecar mode requires exactly one host")
        return Session(servers=servers, libcuda=libcuda)

    has_native_cuda = _has_native_cuda_backend()
    use_sidecar = sidecar is True or (
        sidecar is None and sys.platform == "darwin" and not has_native_cuda
    )
    if use_sidecar:
        if len(servers) != 1:
            raise LupineError("sidecar mode requires exactly one host")
        if libcuda is not None:
            raise LupineError("libcuda is only supported with sidecar=False")
        return _create_sidecar(server=servers[0])

    if not has_native_cuda:
        if sidecar is False:
            raise LupineError(
                "PyTorch is not compiled with CUDA and sidecar=False disables "
                "the LUPINE sidecar."
            )
        raise LupineError(
            "PyTorch is not compiled with CUDA and automatic LUPINE sidecar "
            "selection is only supported on macOS; pass sidecar=True to force it."
        )

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
    platform: str | None = None,
    rosetta: bool = False,
    env: dict[str, str] | None = None,
) -> Any:
    """Create a session-scoped sidecar PyTorch worker frontend."""

    return _create_sidecar(
        server=server,
        image=image,
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
