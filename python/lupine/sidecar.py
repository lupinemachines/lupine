import atexit
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import types
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

    def __repr__(self) -> str:
        return (
            f"SidecarTensor(handle={self._lupine_handle}, "
            f"shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"
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


def store(tensor):
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
        return objects[int(value["__sidecar_tensor__"])]
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


def encode(value):
    if isinstance(value, torch.Tensor):
        return store(value)
    if isinstance(value, torch.Size):
        return {"type": "tuple", "items": list(value)}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [encode(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [encode(item) for item in value]}
    if isinstance(value, dict):
        return {"type": "dict", "items": {key: encode(item) for key, item in value.items()}}
    return {"type": "value", "value": value}


def resolve(packet, overload):
    overload_packet = getattr(torch.ops.aten, packet)
    if overload == "default":
        return overload_packet.default
    return getattr(overload_packet, overload)


def release(value):
    if isinstance(value, dict) and "__sidecar_tensor__" in value:
        objects.pop(int(value["__sidecar_tensor__"]), None)
        return
    if isinstance(value, list):
        for item in value:
            release(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            release(item)


def handle(request):
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
        return encode(func(*args, **kwargs))
    if op == "release":
        release(request["value"])
        return True
    raise RuntimeError(f"unknown op: {op}")


for line in sys.stdin:
    try:
        response = {"ok": True, "result": handle(json.loads(line))}
    except Exception as exc:
        response = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(response), flush=True)
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
    """Session-scoped macOS frontend for a local Linux CUDA PyTorch sidecar."""

    server: str
    image: str = DEFAULT_IMAGE
    runtime: str = "auto"
    platform: str = "linux/arm64"
    rosetta: bool = False
    env: dict[str, str] = field(default_factory=dict)

    def __enter__(self) -> "SidecarSession":
        global _ACTIVE_SESSION
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
        self._lock = threading.Lock()
        self.info = self._request({"op": "ping"})
        if not self.info.get("cuda_available"):
            raise SidecarError(f"sidecar worker has no CUDA device: {self.info}")
        _ACTIVE_SESSION = self
        self._mode = SidecarDispatchMode(self)
        self._mode.__enter__()
        atexit.register(self.close)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        global _ACTIVE_SESSION
        if _ACTIVE_SESSION is self:
            _ACTIVE_SESSION = None
        mode = getattr(self, "_mode", None)
        if mode is not None:
            mode.__exit__(None, None, None)
            self._mode = None
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._proc = None

    def device(self, index: int = 0) -> Any:
        if index != 0:
            raise SidecarError("sidecar prototype exposes one LUPINE device")
        return torch.device(f"{_BACKEND_NAME}:0")

    def _request(self, payload: dict[str, Any]) -> Any:
        with self._lock:
            proc = self._proc
            if proc.stdin is None or proc.stdout is None:
                raise SidecarError("sidecar worker pipes are closed")
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        if not line:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()
            raise SidecarError(f"sidecar worker exited with code {proc.poll()}: {stderr}")
        response = json.loads(line)
        if not response.get("ok"):
            raise SidecarError(response.get("traceback") or response.get("error"))
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

    def forward(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        op = _op_name(func)
        result = self._request(
            {
                "op": "call",
                "packet": op["packet"],
                "overload": op["overload"],
                "args": self._encode(args),
                "kwargs": self._encode(kwargs),
            }
        )
        return self._decode(result)


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
