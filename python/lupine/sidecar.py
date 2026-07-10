from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import types
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


_BACKEND_NAME = "lupine"
_ACTIVE_SESSION: "SidecarSession | None" = None
_REGISTERED = False
DEFAULT_IMAGE = "lupine-pytorch-worker:cuda-13.1.0"


class SidecarError(RuntimeError):
    """Raised when the sidecar PyTorch worker fails."""


def _result_handles(value: Any) -> set[int]:
    if isinstance(value, dict):
        if value.get("type") == "tensor":
            return {int(value["handle"])}
        handles: set[int] = set()
        for item in value.values():
            handles.update(_result_handles(item))
        return handles
    if isinstance(value, list):
        handles = set()
        for item in value:
            handles.update(_result_handles(item))
        return handles
    return set()


def _finalize_sidecar_handle(session_ref: Any, handle: int) -> None:
    session = session_ref()
    if session is not None:
        try:
            session._release_handle(handle, suppress_errors=True)
        except BaseException:
            pass


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> Any:
    return getattr(torch, name)


def _normalize_device(device: Any = None) -> str:
    if device is None:
        return f"{_BACKEND_NAME}:0"
    parsed = torch.device(device)
    if parsed.type != _BACKEND_NAME:
        raise SidecarError(f"expected {_BACKEND_NAME} device, got {parsed}")
    return str(parsed)


def _contains_sidecar(value: Any) -> bool:
    if isinstance(value, SidecarTensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_sidecar(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_sidecar(item) for item in value)
    return False


def _contains_lupine_device(value: Any) -> bool:
    if isinstance(value, torch.device):
        return value.type == _BACKEND_NAME
    if isinstance(value, Mapping):
        return any(_contains_lupine_device(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_lupine_device(item) for item in value)
    return False


def _session_from(value: Any) -> "SidecarSession | None":
    if isinstance(value, SidecarTensor):
        return value._lupine_session
    if isinstance(value, Mapping):
        for item in value.values():
            session = _session_from(item)
            if session is not None:
                return session
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            session = _session_from(item)
            if session is not None:
                return session
    return None


def _op_name(func: Any) -> dict[str, str]:
    overload = func.__name__
    if "." in overload:
        packet, overload = overload.split(".", 1)
    else:
        packet = func.overloadpacket.__name__
        overload = "default"
    return {"packet": packet, "overload": overload}


def _system_running(output: str) -> bool:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return "running" in output.lower()
    return payload.get("status") == "running"


def _ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        torch.utils.rename_privateuse1_backend(_BACKEND_NAME)
    except RuntimeError as exc:
        if _BACKEND_NAME not in str(exc):
            raise

    module = types.SimpleNamespace(
        is_available=lambda: _ACTIVE_SESSION is not None,
        device_count=lambda: 1 if _ACTIVE_SESSION is not None else 0,
        current_device=lambda: 0,
        _is_in_bad_fork=lambda: False,
        manual_seed_all=lambda seed: None,
    )
    try:
        torch._register_device_module(_BACKEND_NAME, module)
    except RuntimeError as exc:
        if "already" not in str(exc):
            raise
    torch.utils.generate_methods_for_privateuse1_backend()
    _REGISTERED = True


class SidecarTensor(torch.Tensor):
    """Tensor wrapper that owns one GPU object handle in the sidecar worker.

    Call :meth:`close` (or use the tensor as a context manager) for prompt
    release. Garbage collection provides a best-effort fallback, and closing
    the owning session releases every handle still live in that session.
    """

    @staticmethod
    def __new__(
        cls,
        *,
        session: "SidecarSession",
        handle: int,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any = None,
    ) -> "SidecarTensor":
        _ensure_registered()
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=torch.device(_normalize_device(device)),
            layout=torch.strided,
            requires_grad=False,
        )

    def __init__(
        self,
        *,
        session: "SidecarSession",
        handle: int,
        shape: tuple[int, ...],
        dtype: Any,
        device: Any = None,
    ) -> None:
        self._lupine_session = session
        self._lupine_handle = int(handle)
        self._lupine_closed = False
        session._adopt_handle(self._lupine_handle)
        self._lupine_finalizer = weakref.finalize(
            self,
            _finalize_sidecar_handle,
            weakref.ref(session),
            self._lupine_handle,
        )

    @property
    def closed(self) -> bool:
        """Whether this wrapper no longer owns a usable worker handle."""

        return self._lupine_closed or self._lupine_session.closed

    def close(self) -> None:
        """Release this tensor's worker handle. Safe to call more than once."""

        if self._lupine_closed:
            return
        self._lupine_closed = True
        self._lupine_finalizer.detach()
        self._lupine_session._release_handle(self._lupine_handle)

    def __enter__(self) -> "SidecarTensor":
        if self.closed:
            raise SidecarError(f"sidecar tensor handle {self._lupine_handle} is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        state = ", closed=True" if self.closed else ""
        return (
            f"SidecarTensor(handle={self._lupine_handle}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}{state})"
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if func.overloadpacket.__name__ == "detach":
            if args[0].closed:
                raise SidecarError(
                    f"sidecar tensor handle {args[0]._lupine_handle} is closed"
                )
            return args[0]
        session = _session_from((args, kwargs))
        if session is None:
            raise SidecarError(f"sidecar LUPINE dispatch could not find a session for {func}")
        return session.forward(func, args, kwargs)


class SidecarDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, session: "SidecarSession") -> None:
        super().__init__()
        self._session = session

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if _contains_sidecar((args, kwargs)) or _contains_lupine_device(kwargs):
            return self._session.forward(func, args, kwargs)
        return func(*args, **kwargs)


_WORKER = r"""
import json
import sys
import traceback

import torch

objects = {}
next_handle = 1


def store(tensor, created_handles):
    global next_handle
    if tensor.device.type != "cuda":
        return {
            "type": "tensor_data",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "data": tensor.tolist(),
        }
    handle = next_handle
    next_handle += 1
    objects[handle] = tensor
    created_handles.append(handle)
    return {
        "type": "tensor",
        "handle": handle,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if isinstance(value, dict) and "__tuple__" in value:
        return tuple(decode(item) for item in value["__tuple__"])
    if isinstance(value, dict) and "__sidecar_tensor__" in value:
        handle = int(value["__sidecar_tensor__"])
        if handle not in objects:
            raise RuntimeError(f"sidecar tensor handle {handle} is released or unknown")
        return objects[handle]
    if isinstance(value, dict) and "__dtype__" in value:
        return getattr(torch, value["__dtype__"])
    if isinstance(value, dict) and "__device__" in value:
        return torch.device(value["__device__"])
    if isinstance(value, dict) and "__layout__" in value:
        return getattr(torch, value["__layout__"])
    if isinstance(value, dict) and "__memory_format__" in value:
        return getattr(torch, value["__memory_format__"])
    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    return value


def encode(value, created_handles):
    if isinstance(value, torch.Tensor):
        return store(value, created_handles)
    if isinstance(value, torch.Size):
        return {"type": "tuple", "items": list(value)}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [encode(item, created_handles) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [encode(item, created_handles) for item in value],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {
                key: encode(item, created_handles) for key, item in value.items()
            },
        }
    return {"type": "value", "value": value}


def resolve(packet, overload):
    overload_packet = getattr(torch.ops.aten, packet)
    if overload == "default":
        return overload_packet.default
    return getattr(overload_packet, overload)


def release_handles(handles):
    released = 0
    for handle in set(int(handle) for handle in handles):
        if objects.pop(handle, None) is not None:
            released += 1
    return released


def collect_handles(value):
    handles = []
    if isinstance(value, dict) and "__sidecar_tensor__" in value:
        handles.append(int(value["__sidecar_tensor__"]))
        return handles
    if isinstance(value, dict) and value.get("type") == "tensor":
        handles.append(int(value["handle"]))
        return handles
    if isinstance(value, list):
        for item in value:
            handles.extend(collect_handles(item))
        return handles
    if isinstance(value, dict):
        for item in value.values():
            handles.extend(collect_handles(item))
    return handles


def stats():
    cuda_available = torch.cuda.is_available()
    return {
        "live_handles": len(objects),
        "cuda_memory_allocated": torch.cuda.memory_allocated() if cuda_available else 0,
        "cuda_memory_reserved": torch.cuda.memory_reserved() if cuda_available else 0,
    }


def handle(request, created_handles):
    op = request["op"]
    if op == "ping":
        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    if op == "call":
        func = resolve(request["packet"], request["overload"])
        args = decode(request.get("args", []))
        kwargs = decode(request.get("kwargs", {}))
        return encode(func(*args, **kwargs), created_handles)
    if op == "release":
        handles = request.get("handles")
        if handles is None:
            handles = collect_handles(request.get("value"))
        return {
            "released": release_handles(handles),
            "live_handles": len(objects),
        }
    if op == "stats":
        return stats()
    raise RuntimeError(f"unknown op: {op}")


for line in sys.stdin:
    created_handles = []
    try:
        response = {
            "ok": True,
            "result": handle(json.loads(line), created_handles),
        }
        serialized = json.dumps(response)
    except Exception as exc:
        release_handles(created_handles)
        response = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        serialized = json.dumps(response)
    print(serialized, flush=True)
"""


@dataclass
class ContainerRuntime:
    """Launch a Linux worker with Apple's `container` CLI."""

    image: str = DEFAULT_IMAGE
    server: str | None = None
    platform: str = "linux/arm64"
    rosetta: bool = False
    env: dict[str, str] = field(default_factory=dict)

    def _container(self) -> str:
        if sys.platform != "darwin":
            raise SidecarError("Apple container sidecars are only supported on macOS")
        container = shutil.which("container")
        if container is None:
            raise SidecarError(
                "Apple container CLI is not installed. Install Apple's `container` "
                "runtime with `brew install --cask container`, or download the signed "
                "installer from https://github.com/apple/container/releases. After "
                "installing, run `container system start` once to initialize it."
            )
        return container

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._container(), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def prepare(self) -> None:
        status = self._run(["system", "status", "--format", "json"])
        if status.returncode != 0 or not _system_running(status.stdout):
            started = self._run(["system", "start"])
            if started.returncode != 0:
                raise SidecarError(
                    "Apple container services are not running and automatic startup failed:\n"
                    f"{(started.stderr or started.stdout).strip()}"
                )

        inspected = self._run(["image", "inspect", self.image])
        if inspected.returncode == 0:
            return

        pulled = self._run(
            ["image", "pull", "--progress", "none", "--platform", self.platform, self.image]
        )
        if pulled.returncode != 0:
            raise SidecarError(
                f"LUPINE sidecar image {self.image!r} is not cached and automatic "
                "pull failed:\n"
                f"{(pulled.stderr or pulled.stdout).strip()}"
            )

    def command(self, script: str) -> list[str]:
        container = self._container()
        cmd = [
            container,
            "run",
            "--rm",
            "--interactive",
            "--progress",
            "none",
            "--platform",
            self.platform,
        ]
        if self.rosetta:
            cmd.append("--rosetta")
        environment = dict(self.env)
        if self.server:
            environment["LUPINE_SERVER"] = self.server
        for key, value in environment.items():
            cmd.extend(["--env", f"{key}={value}"])
        cmd.extend([self.image, "python3", "-u", "-c", script])
        return cmd


@dataclass
class SidecarSession:
    """Session-scoped macOS frontend for a local Linux CUDA PyTorch sidecar.

    Returned ``SidecarTensor`` objects own their worker handles. Closing the
    session releases all outstanding handles before stopping the worker.
    """

    server: str
    image: str = DEFAULT_IMAGE
    runtime: str = "auto"
    platform: str = "linux/arm64"
    rosetta: bool = False
    env: dict[str, str] = field(default_factory=dict)
    _proc: Any = field(init=False, default=None, repr=False)
    _lock: Any = field(init=False, default_factory=threading.RLock, repr=False)
    _mode: Any = field(init=False, default=None, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _live_handles: set[int] = field(init=False, default_factory=set, repr=False)
    _atexit_registered: bool = field(init=False, default=False, repr=False)
    info: dict[str, Any] = field(init=False, default_factory=dict)

    def __enter__(self) -> "SidecarSession":
        global _ACTIVE_SESSION
        if self._closed:
            raise SidecarError("a closed LUPINE sidecar session cannot be reopened")
        if _ACTIVE_SESSION is not None:
            raise SidecarError("a LUPINE sidecar session is already active")
        _ensure_registered()
        runtime = "container" if self.runtime == "auto" and sys.platform == "darwin" else self.runtime
        if runtime != "container":
            raise SidecarError("only runtime='container' is implemented")
        launcher = ContainerRuntime(
            image=self.image,
            server=self.server,
            platform=self.platform,
            rosetta=self.rosetta,
            env=self.env,
        )
        launcher.prepare()
        self._proc = subprocess.Popen(
            launcher.command(textwrap.dedent(_WORKER)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self.info = self._request({"op": "ping"})
            if not self.info.get("cuda_available"):
                raise SidecarError(f"sidecar worker has no CUDA device: {self.info}")
            _ACTIVE_SESSION = self
            self._mode = SidecarDispatchMode(self)
            self._mode.__enter__()
            atexit.register(self.close)
            self._atexit_registered = True
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        global _ACTIVE_SESSION
        with self._lock:
            if self._closed and self._proc is None:
                return
            if _ACTIVE_SESSION is self:
                _ACTIVE_SESSION = None
            mode = self._mode
            self._mode = None
            try:
                if mode is not None:
                    mode.__exit__(None, None, None)
            finally:
                proc = self._proc
                if self._live_handles and proc is not None and proc.poll() is None:
                    try:
                        self._request(
                            {
                                "op": "release",
                                "handles": sorted(self._live_handles),
                            }
                        )
                    except BaseException:
                        pass
                self._terminate_worker_locked()
                if self._atexit_registered:
                    atexit.unregister(self.close)
                    self._atexit_registered = False

    def _terminate_worker_locked(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._proc = None
        self._closed = True
        self._live_handles.clear()

    @property
    def closed(self) -> bool:
        """Whether the worker session has been closed."""

        return self._closed

    @property
    def live_handle_count(self) -> int:
        """Number of worker tensor handles currently owned by this session."""

        with self._lock:
            return len(self._live_handles)

    def stats(self) -> dict[str, int]:
        """Return worker handle and CUDA allocator counts for diagnostics."""

        result = self._request({"op": "stats"})
        if not isinstance(result, dict):
            raise SidecarError(f"sidecar returned invalid stats: {result!r}")
        return {key: int(value) for key, value in result.items()}

    def _adopt_handle(self, handle: int) -> None:
        with self._lock:
            if self._closed:
                raise SidecarError("sidecar session is closed")
            if handle in self._live_handles:
                raise SidecarError(f"sidecar tensor handle {handle} is duplicated")
            self._live_handles.add(handle)

    def _release_handle(self, handle: int, *, suppress_errors: bool = False) -> bool:
        with self._lock:
            if handle not in self._live_handles:
                return False
            try:
                self._request({"op": "release", "handles": [handle]})
            except BaseException:
                self._terminate_worker_locked()
                if suppress_errors:
                    return False
                raise
            self._live_handles.discard(handle)
            return True

    def device(self, index: int = 0) -> Any:
        if index != 0:
            raise SidecarError("sidecar prototype exposes one LUPINE device")
        return torch.device(f"{_BACKEND_NAME}:0")

    def _request(self, payload: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed:
                raise SidecarError("sidecar session is closed")
            try:
                request = json.dumps(payload) + "\n"
            except (TypeError, ValueError) as exc:
                raise SidecarError(f"sidecar request is not JSON serializable: {exc}") from exc

            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                self._terminate_worker_locked()
                raise SidecarError("sidecar worker pipes are closed")
            try:
                proc.stdin.write(request)
                proc.stdin.flush()
                line = proc.stdout.readline()
            except BaseException as exc:
                self._terminate_worker_locked()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise SidecarError(f"sidecar transport failed: {exc}") from exc

            if not line:
                stderr = ""
                returncode = proc.poll()
                if returncode is not None and proc.stderr is not None:
                    stderr = proc.stderr.read()
                self._terminate_worker_locked()
                raise SidecarError(
                    f"sidecar worker exited with code {returncode}: {stderr}"
                )
            try:
                response = json.loads(line)
            except (json.JSONDecodeError, TypeError) as exc:
                self._terminate_worker_locked()
                raise SidecarError(f"sidecar returned invalid JSON: {exc}") from exc
            if not isinstance(response, dict) or "ok" not in response:
                self._terminate_worker_locked()
                raise SidecarError(f"sidecar returned invalid response: {response!r}")
            if not response.get("ok"):
                raise SidecarError(response.get("traceback") or response.get("error"))
            if "result" not in response:
                self._terminate_worker_locked()
                raise SidecarError(f"sidecar returned invalid response: {response!r}")
            return response["result"]

    def _wrap(self, result: dict[str, Any]) -> SidecarTensor:
        return SidecarTensor(
            session=self,
            handle=result["handle"],
            shape=tuple(result["shape"]),
            dtype=_dtype_from_name(result["dtype"]),
            device=self.device(),
        )

    def _encode(self, value: Any) -> Any:
        if isinstance(value, SidecarTensor):
            if value._lupine_session is not self:
                raise SidecarError("cannot mix tensors from different sidecar sessions")
            if value.closed or value._lupine_handle not in self._live_handles:
                raise SidecarError(
                    f"sidecar tensor handle {value._lupine_handle} is closed"
                )
            return {"__sidecar_tensor__": value._lupine_handle}
        if isinstance(value, torch.dtype):
            return {"__dtype__": _dtype_name(value)}
        if isinstance(value, torch.device):
            device = "cuda:0" if value.type == _BACKEND_NAME else str(value)
            return {"__device__": device}
        if isinstance(value, torch.layout):
            return {"__layout__": str(value).removeprefix("torch.")}
        if isinstance(value, torch.memory_format):
            return {"__memory_format__": str(value).removeprefix("torch.")}
        if isinstance(value, torch.Size):
            return {"__tuple__": [self._encode(item) for item in value]}
        if isinstance(value, tuple):
            return {"__tuple__": [self._encode(item) for item in value]}
        if isinstance(value, list):
            return [self._encode(item) for item in value]
        if isinstance(value, Mapping):
            return {key: self._encode(item) for key, item in value.items()}
        return value

    def _decode(self, value: Any) -> Any:
        kind = value.get("type") if isinstance(value, dict) else None
        if kind == "tensor":
            return self._wrap(value)
        if kind == "tensor_data":
            return torch.tensor(value["data"], dtype=_dtype_from_name(value["dtype"]))
        if kind == "tuple":
            return tuple(self._decode(item) for item in value["items"])
        if kind == "list":
            return [self._decode(item) for item in value["items"]]
        if kind == "dict":
            return {key: self._decode(item) for key, item in value["items"].items()}
        if kind == "value":
            return value["value"]
        raise SidecarError(f"sidecar returned unsupported result: {value!r}")

    def _discard_result_handles(self, result: Any) -> None:
        try:
            handles = _result_handles(result)
            if handles:
                self._request({"op": "release", "handles": sorted(handles)})
                self._live_handles.difference_update(handles)
        except BaseException:
            self._terminate_worker_locked()

    def forward(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        with self._lock:
            if self._closed:
                raise SidecarError("sidecar session is closed")
            op = _op_name(func)
            payload = {
                "op": "call",
                "packet": op["packet"],
                "overload": op["overload"],
                "args": self._encode(args),
                "kwargs": self._encode(kwargs),
            }
            no_result = object()
            result = no_result
            try:
                result = self._request(payload)
                return self._decode(result)
            except SidecarError:
                if result is not no_result:
                    self._discard_result_handles(result)
                raise
            except BaseException:
                if result is no_result:
                    self._terminate_worker_locked()
                else:
                    self._discard_result_handles(result)
                raise


def sidecar(
    server: str | None = None,
    *,
    image: str = DEFAULT_IMAGE,
    runtime: str = "auto",
    platform: str = "linux/arm64",
    rosetta: bool = False,
    env: dict[str, str] | None = None,
) -> SidecarSession:
    server = server or os.environ.get("LUPINE_SERVER")
    if not server:
        raise SidecarError("pass server=... or set LUPINE_SERVER")
    return SidecarSession(
        server=server,
        image=image,
        runtime=runtime,
        platform=platform,
        rosetta=rosetta,
        env=dict(env or {}),
    )
