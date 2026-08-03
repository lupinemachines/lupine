from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import torch
from packaging.version import InvalidVersion, Version

from .container import prepare_runtime
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

_WORKER_REGISTRY_HOST = "ghcr.io"
_WORKER_REPOSITORY = "lupinemachines/lupine-pytorch-worker"
_WORKER_IMAGE_REPOSITORY = f"{_WORKER_REGISTRY_HOST}/{_WORKER_REPOSITORY}"
DEFAULT_IMAGE = f"{_WORKER_IMAGE_REPOSITORY}:cuda-13.1.0"
_CUDA_VERSION_HEADER = "x-lupine-cuda-version"
_SESSION_HEADER = "x-lupine-session"


def _server_cuda_version(server: str) -> Version:
    url = server
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    url = f"{url.rstrip('/')}/"
    # A hosted gateway routes on the lease and drops any connection whose first
    # HEADERS frame omits it, so the probe has to carry it too.
    headers = {}
    session = os.environ.get("LUPINE_SESSION")
    if session:
        headers[_SESSION_HEADER] = session
    try:
        with httpx.Client(
            http1=False,
            http2=True,
            timeout=httpx.Timeout(10.0, connect=5.0),
        ) as client:
            response = client.head(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SidecarError(f"could not query LUPINE server {server!r}: {exc}") from exc

    version = response.headers.get(_CUDA_VERSION_HEADER)
    if version is None:
        raise SidecarError(
            f"LUPINE server {server!r} did not advertise "
            f"{_CUDA_VERSION_HEADER}; pass image= explicitly"
        )
    try:
        return Version(version.strip())
    except InvalidVersion as exc:
        raise SidecarError(
            f"server returned invalid {_CUDA_VERSION_HEADER}: {version!r}"
        ) from exc


def _registry_json(path: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        response = httpx.get(
            f"https://{_WORKER_REGISTRY_HOST}{path}",
            headers=headers,
            timeout=10,
        )
    except httpx.HTTPError as exc:
        raise SidecarError(
            f"could not query sidecar image registry: {exc}; pass image= explicitly"
        ) from exc

    if response.status_code != 200:
        detail = response.text.strip()
        raise SidecarError(
            f"could not query sidecar image registry: HTTP {response.status_code}: "
            f"{detail}; pass image= explicitly"
        )
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise SidecarError(
            "sidecar image registry returned invalid JSON; pass image= explicitly"
        ) from exc
    if not isinstance(payload, dict):
        raise SidecarError(
            "sidecar image registry returned an invalid response; "
            "pass image= explicitly"
        )
    return payload


def _worker_images() -> tuple[tuple[Version, str], ...]:
    query = urlencode(
        {
            "service": _WORKER_REGISTRY_HOST,
            "scope": f"repository:{_WORKER_REPOSITORY}:pull",
        }
    )
    token = _registry_json(f"/token?{query}").get("token")
    if not isinstance(token, str) or not token:
        raise SidecarError(
            "sidecar image registry did not return an access token; "
            "pass image= explicitly"
        )

    payload = _registry_json(
        f"/v2/{_WORKER_REPOSITORY}/tags/list?n=1000",
        {"Authorization": f"Bearer {token}"},
    )
    tags = payload.get("tags")
    if not isinstance(tags, list):
        raise SidecarError(
            "sidecar image registry did not return image tags; pass image= explicitly"
        )

    images: list[tuple[Version, str]] = []
    for tag in tags:
        if not isinstance(tag, str) or not tag.startswith("cuda-"):
            continue
        version_text = tag.removeprefix("cuda-")
        try:
            version = Version(version_text)
        except InvalidVersion:
            continue
        if (
            version.epoch
            or version.pre is not None
            or version.post is not None
            or version.dev is not None
            or version.local is not None
            or len(version.release) != 3
            or str(version) != version_text
        ):
            continue
        images.append((version, f"{_WORKER_IMAGE_REPOSITORY}:{tag}"))

    if not images:
        raise SidecarError(
            "sidecar image registry has no CUDA-versioned images; "
            "pass image= explicitly"
        )
    return tuple(sorted(images, reverse=True))


def _worker_image_for_server(server: str) -> str:
    server_version = _server_cuda_version(server)
    for required_version, image in _worker_images():
        if required_version <= server_version:
            return image
    raise SidecarError(
        f"no published sidecar worker supports server CUDA {server_version}; "
        "pass image= explicitly"
    )


def _op_name(func: Any) -> dict[str, str]:
    overload = func.__name__
    if "." in overload:
        packet, overload = overload.split(".", 1)
    else:
        packet = func.overloadpacket.__name__
        overload = "default"
    return {"packet": packet, "overload": overload}


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
class SidecarSession:
    """Session-scoped frontend for a containerized CUDA PyTorch worker."""

    server: str
    image: str | None = None
    runtime: str = "auto"
    platform: str | None = None
    rosetta: bool = False
    env: dict[str, str] = field(default_factory=dict)
    _pending_release: list[int] = field(default_factory=list, init=False, repr=False)

    def _worker_image(self) -> str:
        return self.image or _worker_image_for_server(self.server)

    def __enter__(self) -> "SidecarSession":
        if _get_active_session() is not None:
            raise SidecarError("a LUPINE sidecar session is already active")
        _ensure_registered()
        image = self._worker_image()
        self.image = image
        launcher = prepare_runtime(
            self.runtime,
            image=image,
            server=self.server,
            platform=self.platform,
            rosetta=self.rosetta,
            env=self.env,
        )
        self._proc = subprocess.Popen(
            launcher.command(_worker_source()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._lock = threading.Lock()
        try:
            self.info = self._request({"op": "ping"})
            if not self.info.get("cuda_available"):
                raise SidecarError(f"sidecar worker has no CUDA device: {self.info}")
            _set_active_session(self)
            mode = SidecarDispatchMode(self)
            mode.__enter__()
            self._mode = mode
        except BaseException:
            # __exit__ never runs for a failed __enter__, so stop the worker here.
            self.close()
            raise
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

    def _release_handle(self, handle: int) -> None:
        self._pending_release.append(handle)

    def _flush_released(self) -> None:
        # Sent from _request, never from the finalizer, so garbage collection
        # cannot re-enter the transport lock. pop() keeps an append that lands
        # mid-drain.
        handles = []
        while self._pending_release:
            handles.append(self._pending_release.pop())
        if handles and getattr(self, "_proc", None) is not None:
            values = [{"__sidecar_tensor__": handle} for handle in handles]
            self._send({"op": "release", "value": values})

    def _request(
        self,
        payload: dict[str, Any],
        input_tensors: Sequence[tuple[Any, Mapping[str, Any]]] = (),
        *,
        decode_result: bool = False,
    ) -> Any:
        self._flush_released()
        return self._send(payload, input_tensors, decode_result=decode_result)

    def _send(
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
            except (TypeError, ValueError) as exc:
                # Nothing reached the worker yet, so the session stays usable.
                raise SidecarError(
                    f"sidecar cannot send this operation to the worker: {exc}; "
                    "arguments must be tensors, numbers, strings, dtypes, "
                    "devices, layouts, or memory formats. Build the value on "
                    "CPU and move it with .to(device) instead."
                ) from exc
            try:
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
    image: str | None = None,
    runtime: str = "auto",
    platform: str | None = None,
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
