"""PyTorch adapter helpers for LUPINE.

The adapter returns ordinary ``torch.device("cuda:N")`` objects. PyTorch stays
on its built-in CUDA dispatch path while LUPINE handles CUDA driver calls below
it.
"""

import os
import ctypes
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

DEFAULT_PORT = 14833
_loaded_libcuda: Any | None = None
SnapshotType = Literal["rw", "r", "w"]
SNAPSHOT_TYPES = get_args(SnapshotType)


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
    global _loaded_libcuda
    libcuda = Path(path) if path is not None else _default_libcuda()
    if libcuda is None:
        return
    if not libcuda.exists():
        raise LupineError(f"LUPINE libcuda does not exist: {libcuda}")
    _loaded_libcuda = ctypes.CDLL(str(libcuda), mode=ctypes.RTLD_GLOBAL)


def _snapshot_lib() -> Any:
    global _loaded_libcuda
    if _loaded_libcuda is not None:
        return _loaded_libcuda
    libcuda = _default_libcuda()
    if libcuda is not None:
        _loaded_libcuda = ctypes.CDLL(str(libcuda), mode=ctypes.RTLD_GLOBAL)
        return _loaded_libcuda
    _loaded_libcuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
    return _loaded_libcuda


def _clear_snapshot_load() -> None:
    lib = _snapshot_lib()
    func = getattr(lib, "lupine_snapshot_load", None)
    if func is None:
        raise LupineError("LUPINE client library does not expose snapshot load")
    func.argtypes = [ctypes.c_char_p]
    func.restype = ctypes.c_int
    if int(func(None)) != 0:
        raise LupineError("LUPINE snapshot clear load failed")


def _disconnect() -> None:
    """Tear down the client connection so the next CUDA call reconnects.

    Scopes the connection to a ``connect()`` context: exiting disconnects, and
    re-entering re-opens a fresh connection (re-sending the snapshot bootstrap,
    so a restore happens). Uses the already-loaded client library; best-effort,
    so it is a no-op if the library or the symbol is unavailable.
    """
    lib = _loaded_libcuda
    if lib is None:
        return
    func = getattr(lib, "lupine_rpc_disconnect", None)
    if func is None:
        return
    func.argtypes = []
    func.restype = None
    func()




def _snapshot_synchronize() -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _servers_from_env() -> tuple[str, ...]:
    value = os.environ.get("LUPINE_SERVER", "")
    return tuple(server.strip() for server in value.split(",") if server.strip())


def _cuda_device(index: int, *, require_available: bool = False) -> Any:
    torch = _torch()
    index = int(index)
    if require_available:
        count = int(torch.cuda.device_count())
        if count <= 0:
            raise LupineError(
                "PyTorch does not see any CUDA devices. Check that the LUPINE "
                "client library is selected and LUPINE_SERVER is configured."
            )
        if index < 0 or index >= count:
            raise LupineError(f"CUDA device index {index} is out of range for {count} devices")
    return torch.device("cuda", index)


@dataclass
class Session:
    """A process-local LUPINE connection declaration."""

    servers: tuple[str, ...]
    require_available: bool = False
    libcuda: str | os.PathLike[str] | None = None
    snapshot_id: str | None = None
    snapshot_type: SnapshotType = "rw"

    def __post_init__(self) -> None:
        if not self.servers:
            raise LupineError("at least one LUPINE host is required")
        if self.snapshot_type not in SNAPSHOT_TYPES:
            raise LupineError("snapshot_type must be one of 'rw', 'r', or 'w'")

    def __enter__(self) -> "Session":
        self._previous_server = os.environ.get("LUPINE_SERVER")
        configured = _servers_from_env()
        if configured and configured != self.servers:
            raise LupineError(
                "LUPINE_SERVER is already configured differently; start a new "
                "process or pass the same hosts to lupine.connect()."
            )
        if not configured:
            # Configuring the server for the first time must precede PyTorch
            # CUDA initialization. Re-entering with the same server is fine: the
            # connection is scoped to the context and reconnects on entry.
            _require_mutable_config()
            _set_server_env(self.servers)
        _load_libcuda(self.libcuda)
        if self.snapshot_id is not None and "r" in self.snapshot_type:
            load_snapshot(self.snapshot_id)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        try:
            if self.snapshot_id is not None and "w" in self.snapshot_type:
                save_snapshot_and_exit(self.snapshot_id)
        finally:
            # Scope the connection to the context: drop it so a later context
            # re-opens a fresh connection (and restores, if it reads).
            _disconnect()
            if not _cuda_initialized():
                self._restore_env()
            if self.snapshot_id is not None:
                _clear_snapshot_load()
        return False

    def _restore_env(self) -> None:
        if getattr(self, "_previous_server", None) is None:
            os.environ.pop("LUPINE_SERVER", None)
        else:
            os.environ["LUPINE_SERVER"] = self._previous_server

    def devices(self, *, require_available: bool | None = None) -> list[Any]:
        """Return all declared LUPINE GPUs as ``torch.device("cuda:N")``."""

        check = self.require_available if require_available is None else require_available
        return [
            _cuda_device(index, require_available=check)
            for index in range(len(self.servers))
        ]

    def device(self, index: int = 0, *, require_available: bool | None = None) -> Any:
        """Return one declared LUPINE GPU as ``torch.device("cuda:N")``."""

        index = int(index)
        if index < 0 or index >= len(self.servers):
            raise LupineError(
                f"LUPINE device index {index} is out of range for {len(self.servers)} hosts"
            )
        check = self.require_available if require_available is None else require_available
        return _cuda_device(index, require_available=check)

def connect(
    *,
    host: str | Sequence[str],
    port: int | None = None,
    require_available: bool = False,
    libcuda: str | os.PathLike[str] | None = None,
    snapshot_id: str | None = None,
    snapshot_type: SnapshotType = "rw",
) -> Any:
    """Create a LUPINE session for one or more remote GPU hosts.

    Use the session before any PyTorch CUDA operation:

    ``with lupine.connect(host=["a:14833", "b:14833"]) as s:``

    ``s.devices()`` then returns ``[torch.device("cuda:0"), torch.device("cuda:1")]``.

    When ``snapshot_id`` is set, ``snapshot_type`` selects how the snapshot is
    used (writing the device snapshot is the expensive step):

    - ``"rw"`` (default): restore the snapshot on enter if it exists, and save
      it again on exit.
    - ``"r"``: restore on enter but never save on exit. Use this to reuse a
      snapshot repeatedly (e.g. serving inference from a preloaded model)
      without paying the save cost each time.
    - ``"w"``: start fresh (do not restore) and save on exit. Use this to
      create or refresh a snapshot.

    On macOS with a CPU-only PyTorch build, ``connect()`` automatically returns
    a sidecar session backed by Apple's container runtime.
    """

    servers = _normalize_hosts(host, port)
    if snapshot_type not in SNAPSHOT_TYPES:
        raise LupineError("snapshot_type must be one of 'rw', 'r', or 'w'")

    if not _has_native_cuda_backend():
        if sys.platform != "darwin":
            raise LupineError(
                "PyTorch is not compiled with CUDA and automatic LUPINE sidecar "
                "fallback is only supported on macOS."
            )
        if snapshot_id is not None:
            raise LupineError("snapshot_id is only supported with native CUDA PyTorch")
        if len(servers) != 1:
            raise LupineError("automatic LUPINE sidecar fallback supports one host")
        if libcuda is not None:
            raise LupineError("libcuda is only supported with native CUDA PyTorch")
        return sidecar(server=servers[0])

    return Session(
        servers=servers,
        require_available=require_available,
        libcuda=libcuda,
        snapshot_id=snapshot_id,
        snapshot_type=snapshot_type,
    )


def devices(*, require_available: bool = True) -> list[Any]:
    """Return devices for the current ``LUPINE_SERVER`` environment."""

    servers = _servers_from_env()
    if not servers:
        raise LupineError("LUPINE_SERVER is not configured")
    return [
        _cuda_device(index, require_available=require_available)
        for index in range(len(servers))
    ]


def device(index: int = 0, *, require_available: bool = True) -> Any:
    """Return one device for the current ``LUPINE_SERVER`` environment."""

    return devices(require_available=require_available)[int(index)]


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
    """Return PyTorch's CUDA device count."""

    torch = _torch()
    return int(torch.cuda.device_count())


def current_device() -> int:
    """Return PyTorch's current CUDA device index."""

    torch = _torch()
    return int(torch.cuda.current_device())


def synchronize(index: int = 0) -> None:
    """Synchronize a LUPINE-backed CUDA device."""

    torch = _torch()
    torch.cuda.synchronize(_cuda_device(index, require_available=False))


def load_snapshot(snapshot_id: str) -> None:
    """Load a server-side snapshot on the next LUPINE connection."""

    lib = _snapshot_lib()
    func = lib.lupine_snapshot_load
    func.argtypes = [ctypes.c_char_p]
    func.restype = ctypes.c_int
    if int(func(snapshot_id.encode("utf-8"))) != 0:
        raise LupineError(f"LUPINE snapshot load failed for {snapshot_id!r}")


def save_snapshot_and_exit(snapshot_id: str) -> None:
    """Save this server-side session snapshot and close the server worker."""

    _snapshot_synchronize()
    lib = _snapshot_lib()
    func = lib.lupine_snapshot_save_and_exit
    func.argtypes = [ctypes.c_char_p]
    func.restype = ctypes.c_int
    if int(func(snapshot_id.encode("utf-8"))) != 0:
        raise LupineError(f"LUPINE snapshot save failed for {snapshot_id!r}")


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
    "load_snapshot",
    "save_snapshot_and_exit",
    "sidecar",
    "servers",
    "synchronize",
]
