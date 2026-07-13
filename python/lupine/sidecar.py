from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .tensor import (
    _BACKEND_NAME as _BACKEND_NAME,
    SidecarDispatchMode as SidecarDispatchMode,
    SidecarError as SidecarError,
    SidecarTensor as SidecarTensor,
    _cpu_tensor_metadata as _cpu_tensor_metadata,
    _decode_cpu_tensor as _decode_cpu_tensor,
    _dtype_from_name as _dtype_from_name,
    _dtype_name as _dtype_name,
    _ensure_registered as _ensure_registered,
    _get_active_session,
    _set_active_session,
    _write_all as _write_all,
    _write_tensor as _write_tensor,
)


DEFAULT_IMAGE = "lupine-pytorch-worker:cuda-13.1.0"


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


_TENSOR_PATH = Path(__file__).with_name("tensor.py")
_WORKER_PATH = Path(__file__).with_name("worker.py")


def _worker_source() -> str:
    tensor_source = _TENSOR_PATH.read_text(encoding="utf-8")
    worker_source = _WORKER_PATH.read_text(encoding="utf-8")
    return (
        "import sys\n"
        "import types\n\n"
        "package = types.ModuleType('lupine')\n"
        "package.__path__ = []\n"
        "sys.modules['lupine'] = package\n\n"
        "def load_module(name, source):\n"
        "    module = types.ModuleType(name)\n"
        "    module.__file__ = name.replace('.', '/') + '.py'\n"
        "    module.__package__ = 'lupine'\n"
        "    sys.modules[name] = module\n"
        "    setattr(package, name.rsplit('.', 1)[-1], module)\n"
        "    exec(compile(source, module.__file__, 'exec'), module.__dict__)\n"
        "    return module\n\n"
        f"tensor = load_module('lupine.tensor', {tensor_source!r})\n"
        f"worker = load_module('lupine.worker', {worker_source!r})\n"
        "worker.main()\n"
    )


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
            [
                "image",
                "pull",
                "--progress",
                "none",
                "--platform",
                self.platform,
                self.image,
            ]
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
        if _get_active_session() is not None:
            raise SidecarError("a LUPINE sidecar session is already active")
        _ensure_registered()
        runtime = (
            "container"
            if self.runtime == "auto" and sys.platform == "darwin"
            else self.runtime
        )
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
        _set_active_session(self)
        self._mode = SidecarDispatchMode(self)
        self._mode.__enter__()
        atexit.register(self.close)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if _get_active_session() is self:
            _set_active_session(None)
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
