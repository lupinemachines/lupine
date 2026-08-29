"""PyTorch adapter for LUPINE-backed CUDA devices.

``lupine.connect(...)`` points the process at one or more LUPINE GPU
servers and preloads the bundled native shims (driver API, runtime API,
NVML) so ordinary CUDA consumers — PyTorch included — transparently run on
the remote GPUs:

.. code-block:: python

    import lupine

    with lupine.connect(host="gpu-host:14833") as session:
        import torch

        x = torch.arange(8, device=session.device(), dtype=torch.float32)
        print((x * 2).cpu())

The adapter returns ordinary ``torch.device("cuda:N")`` objects; PyTorch
keeps its built-in CUDA dispatch path while the LUPINE shims handle the
driver and runtime calls underneath it. No NVIDIA software is required on
the client — the wheel bundles shims for Linux, macOS, and Windows.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
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


def _torch_has_cuda() -> bool:
    try:
        return _torch().version.cuda is not None
    except LupineError:
        return False


def _normalize_server(host: str, port: int | None = None) -> str:
    from urllib.parse import urlsplit, urlunsplit

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


def _servers_from_env() -> tuple[str, ...]:
    value = os.environ.get("LUPINE_SERVER", "")
    return tuple(server.strip() for server in value.split(",") if server.strip())


@dataclass
class Session:
    """A process-local LUPINE connection declaration.

    Entering the session exports ``LUPINE_SERVER`` and loads the bundled
    native shims; both stay in effect for the process after exit (library
    unloading is not possible once CUDA state references the shims).
    """

    servers: tuple[str, ...]
    _previous_server: str | None = field(default=None, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def __enter__(self) -> "Session":
        if not self.servers:
            return self
        if self._loaded:
            # Already active (e.g. connect() entered before returning the
            # session); re-entering must not reload or reset state.
            return self
        configured = _servers_from_env()
        if configured and configured != self.servers:
            raise LupineError(
                "LUPINE_SERVER is already configured differently; start a new "
                "process or pass the same hosts to lupine.connect()."
            )
        self._previous_server = os.environ.get("LUPINE_SERVER")
        if not configured:
            os.environ["LUPINE_SERVER"] = ",".join(self.servers)
        from . import _native

        _native.load(missing_ok=False)
        self._loaded = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if getattr(self, "_previous_server", None) is None:
            os.environ.pop("LUPINE_SERVER", None)
        else:
            os.environ["LUPINE_SERVER"] = self._previous_server
        return False

    def devices(self) -> list[Any]:
        """Return every GPU in LUPINE's virtual device topology."""

        if not self.servers:
            return []
        if not self._loaded and not _native_loaded():
            raise LupineError(
                "Session is not active; use 'with lupine.connect(...) as s:'."
            )
        torch = _torch()
        count = int(torch.cuda.device_count())
        return [torch.device("cuda", index) for index in range(count)]

    def device(self, index: int = 0) -> Any:
        """Return one GPU from LUPINE's virtual device topology."""

        torch = _torch()
        count = int(torch.cuda.device_count()) if self.servers else 0
        if index >= count or index < -count:
            raise LupineError(f"device index {index} out of range ({count} devices)")
        return torch.device("cuda", range(count)[index])


def _native_loaded() -> bool:
    from . import _native

    return bool(_native.loaded())


def connect(
    *,
    host: str | Sequence[str] | None = None,
    port: int | None = None,
) -> Session:
    """Create a LUPINE session for one or more remote GPU hosts.

    Use the session before any CUDA operation (including ``import torch``
    on platforms where PyTorch eagerly resolves CUDA symbols):

    ``with lupine.connect(host=["a:14833", "b:14833"]) as s:``

    ``host`` defaults to ``LUPINE_SERVER``, which is how a launcher such as
    ``lupine run`` hands the session its already-bound hosts.

    ``s.devices()`` then returns every CUDA ordinal in LUPINE's virtual
    device topology.
    """

    if host is None:
        host = _servers_from_env()
        if not host:
            raise LupineError("pass host=... or set LUPINE_SERVER")

    servers = _normalize_hosts(host, port)
    if not servers:
        return Session(servers=servers)

    session = Session(servers=servers)
    # Bind immediately: the caller may import torch right after connect()
    # without using the context manager form.
    session.__enter__()
    session._previous_server = None  # keep LUPINE_SERVER configured on exit
    return session


def devices() -> list[Any]:
    """Return devices in PyTorch's current LUPINE-backed CUDA topology."""

    torch = _torch()
    count = int(torch.cuda.device_count())
    return [torch.device("cuda", index) for index in range(count)]


def servers() -> tuple[str, ...]:
    """Return configured LUPINE servers from ``LUPINE_SERVER``."""

    return _servers_from_env()


def is_configured() -> bool:
    """Return true when ``LUPINE_SERVER`` names at least one server."""

    return bool(servers())


def load_native(*, missing_ok: bool = True) -> dict[str, str]:
    """Load the bundled native shims; see :mod:`lupine._native`."""

    from . import _native

    return _native.load(missing_ok=missing_ok)


def libdir() -> Any:
    """Directory holding the bundled native shims, if any."""

    from . import _native

    return _native.libdir()


__all__ = [
    "DEFAULT_PORT",
    "LupineError",
    "Session",
    "connect",
    "devices",
    "is_configured",
    "libdir",
    "load_native",
    "servers",
]
