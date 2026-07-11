from __future__ import annotations

import atexit
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


_BACKEND_NAME = "lupine"
_ACTIVE_SESSION: "SidecarSession | None" = None
_REGISTERED = False
_TENSOR_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_IMAGE = "lupine-pytorch-worker:cuda-13.1.0"


class SidecarError(RuntimeError):
    """Raised when the sidecar PyTorch worker fails."""


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> Any:
    return getattr(torch, name)


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _tensor_bytes(tensor: Any) -> memoryview:
    size = _tensor_nbytes(tensor)
    if size == 0:
        return memoryview(b"")
    storage = (ctypes.c_ubyte * size).from_address(tensor.data_ptr())
    return memoryview(storage).cast("B")


def _write_all(stream: Any, data: Any) -> None:
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None or written <= 0:
            raise SidecarError("sidecar stream stopped accepting data")
        offset += written


def _write_tensor(
    stream: Any, tensor: Any, *, chunk_size: int = _TENSOR_CHUNK_BYTES
) -> None:
    data = _tensor_bytes(tensor)
    for offset in range(0, len(data), chunk_size):
        _write_all(stream, data[offset : offset + chunk_size])


def _read_tensor(
    stream: Any, tensor: Any, *, chunk_size: int = _TENSOR_CHUNK_BYTES
) -> None:
    data = _tensor_bytes(tensor)
    offset = 0
    while offset < len(data):
        end = min(offset + chunk_size, len(data))
        read = stream.readinto(data[offset:end])
        if read is None or read <= 0:
            raise SidecarError(
                "sidecar tensor stream ended before the tensor was complete"
            )
        offset += read


def _cpu_tensor_metadata(tensor: Any) -> dict[str, Any]:
    if tensor.device.type != "cpu":
        raise SidecarError(
            "sidecar arguments must be CPU tensors or SidecarTensor objects, "
            f"got {tensor.device}"
        )
    if tensor.layout != torch.strided:
        raise SidecarError(
            f"sidecar CPU tensor arguments require strided layout, got {tensor.layout}"
        )
    if tensor.is_quantized:
        raise SidecarError("quantized CPU tensor arguments are not supported")
    if tensor.is_conj():
        raise SidecarError(
            "sidecar CPU tensor arguments cannot be lazy conjugate views; "
            "call resolve_conj() first"
        )
    if tensor.is_neg():
        raise SidecarError(
            "sidecar CPU tensor arguments cannot be lazy negative views; "
            "call resolve_neg() first"
        )
    if not tensor.is_contiguous():
        raise SidecarError(
            "sidecar CPU tensor arguments must be contiguous; call contiguous() first"
        )

    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "layout": "strided",
        "device": "cpu",
        "byteorder": sys.byteorder,
        "nbytes": _tensor_nbytes(tensor),
    }


def _decode_cpu_tensor(payload: Mapping[str, Any], stream: Any) -> Any:
    if payload.get("device") != "cpu" or payload.get("layout") != "strided":
        raise SidecarError("sidecar returned an invalid CPU tensor description")
    if payload.get("byteorder") != sys.byteorder:
        raise SidecarError(
            "sidecar CPU tensor byte order does not match the local process"
        )

    try:
        shape = tuple(int(dimension) for dimension in payload["shape"])
        dtype = _dtype_from_name(payload["dtype"])
        size = int(payload["nbytes"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SidecarError("sidecar returned malformed CPU tensor data") from exc
    if any(dimension < 0 for dimension in shape):
        raise SidecarError("sidecar returned a CPU tensor with a negative dimension")

    element_count = math.prod(shape)
    expected_bytes = element_count * torch.empty((), dtype=dtype).element_size()
    if size != expected_bytes:
        raise SidecarError(
            "sidecar returned the wrong number of bytes for a CPU tensor: "
            f"expected {expected_bytes}, got {size}"
        )
    tensor = torch.empty(shape, dtype=dtype)
    _read_tensor(stream, tensor)
    return tensor


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


_WORKER_PATH = Path(__file__).with_name("_sidecar_worker.py")


def _worker_source() -> str:
    return _WORKER_PATH.read_text(encoding="utf-8")


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
            launcher.command(_worker_source()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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

    def _request(
        self,
        payload: dict[str, Any],
        input_tensors: Sequence[tuple[Any, Mapping[str, Any]]] = (),
        *,
        decode_result: bool = False,
    ) -> Any:
        with self._lock:
            proc = getattr(self, "_proc", None)
            if proc is None:
                raise SidecarError("sidecar worker pipes are closed")
            if proc.stdin is None or proc.stdout is None:
                raise SidecarError("sidecar worker pipes are closed")
            request = dict(payload)
            request["tensor_streams"] = [metadata for _, metadata in input_tensors]
            try:
                header = json.dumps(request).encode("utf-8") + b"\n"
                _write_all(proc.stdin, header)
                for tensor, _ in input_tensors:
                    _write_tensor(proc.stdin, tensor)
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    raise SidecarError("sidecar worker closed its response stream")
                response = json.loads(line)
                output_tensors = [
                    _decode_cpu_tensor(metadata, proc.stdout)
                    for metadata in response.pop("tensor_streams", [])
                ]
            except Exception as exc:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                self._proc = None
                stderr = b""
                if proc.stderr is not None:
                    stderr = proc.stderr.read()
                detail = stderr.decode("utf-8", errors="replace").strip()
                suffix = f": {detail}" if detail else ""
                raise SidecarError(f"sidecar tensor transport failed{suffix}") from exc
            if not response.get("ok"):
                raise SidecarError(response.get("traceback") or response.get("error"))
            if decode_result:
                return self._decode(response["result"], output_tensors)
            return response["result"]

    def _wrap(self, result: dict[str, Any]) -> SidecarTensor:
        return SidecarTensor(
            session=self,
            handle=result["handle"],
            shape=tuple(result["shape"]),
            dtype=_dtype_from_name(result["dtype"]),
            device=self.device(),
        )

    def _upload_cpu_tensor(self, tensor: Any, dtype: Any) -> SidecarTensor:
        metadata = _cpu_tensor_metadata(tensor)
        result = self._request(
            {
                "op": "upload",
                "dtype": _dtype_name(dtype),
                "device": "cuda:0",
            },
            [(tensor, metadata)],
        )
        return self._wrap(result)

    def _copy_from_cpu(self, destination: SidecarTensor, source: Any) -> None:
        metadata = _cpu_tensor_metadata(source)
        self._request(
            {
                "op": "copy_from_cpu",
                "handle": destination._lupine_handle,
            },
            [(source, metadata)],
        )

    def _download_tensor(self, tensor: SidecarTensor, dtype: Any) -> Any:
        return self._request(
            {
                "op": "download",
                "handle": tensor._lupine_handle,
                "dtype": _dtype_name(dtype),
            },
            decode_result=True,
        )

    def _encode(self, value: Any, tensors: list[tuple[Any, Mapping[str, Any]]]) -> Any:
        if isinstance(value, SidecarTensor):
            return {"__sidecar_tensor__": value._lupine_handle}
        if isinstance(value, torch.Tensor):
            stream_index = len(tensors)
            tensors.append((value, _cpu_tensor_metadata(value)))
            return {"__cpu_tensor__": stream_index}
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
            return {"__tuple__": [self._encode(item, tensors) for item in value]}
        if isinstance(value, tuple):
            return {"__tuple__": [self._encode(item, tensors) for item in value]}
        if isinstance(value, list):
            return [self._encode(item, tensors) for item in value]
        if isinstance(value, Mapping):
            return {key: self._encode(item, tensors) for key, item in value.items()}
        return value

    def _decode(self, value: Any, tensors: Sequence[Any]) -> Any:
        kind = value.get("type") if isinstance(value, dict) else None
        if kind == "tensor":
            return self._wrap(value)
        if kind == "tensor_data":
            stream_index = int(value.get("stream", -1))
            if stream_index < 0 or stream_index >= len(tensors):
                raise SidecarError(
                    f"sidecar returned invalid CPU tensor stream {stream_index}"
                )
            return tensors[stream_index]
        if kind == "tuple":
            return tuple(self._decode(item, tensors) for item in value["items"])
        if kind == "list":
            return [self._decode(item, tensors) for item in value["items"]]
        if kind == "dict":
            return {
                key: self._decode(item, tensors) for key, item in value["items"].items()
            }
        if kind == "value":
            return value["value"]
        raise SidecarError(f"sidecar returned unsupported result: {value!r}")

    def forward(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        op = _op_name(func)
        if op["packet"] == "_to_copy" and len(args) == 1:
            source = args[0]
            device = kwargs.get("device")
            target = torch.device(device) if device is not None else None
            layout = kwargs.get("layout", torch.strided)
            memory_format = kwargs.get("memory_format")
            direct_layout = layout == torch.strided and memory_format in (
                None,
                torch.preserve_format,
                torch.contiguous_format,
            )
            if (
                direct_layout
                and isinstance(source, torch.Tensor)
                and not isinstance(source, SidecarTensor)
                and target is not None
                and target.type == _BACKEND_NAME
            ):
                return self._upload_cpu_tensor(
                    source, kwargs.get("dtype") or source.dtype
                )
            if (
                direct_layout
                and isinstance(source, SidecarTensor)
                and target is not None
                and target.type == "cpu"
                and not kwargs.get("pin_memory", False)
            ):
                return self._download_tensor(
                    source, kwargs.get("dtype") or source.dtype
                )
        if (
            op["packet"] == "copy_"
            and len(args) >= 2
            and isinstance(args[0], SidecarTensor)
            and isinstance(args[1], torch.Tensor)
            and not isinstance(args[1], SidecarTensor)
        ):
            self._copy_from_cpu(args[0], args[1])
            return args[0]

        input_tensors: list[tuple[Any, Mapping[str, Any]]] = []
        return self._request(
            {
                "op": "call",
                "packet": op["packet"],
                "overload": op["overload"],
                "args": self._encode(args, input_tensors),
                "kwargs": self._encode(kwargs, input_tensors),
            },
            input_tensors,
            decode_result=True,
        )


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
