"""Tensor wrappers and binary transport for the LUPINE sidecar."""

from __future__ import annotations

import ctypes
import math
import sys
import types
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any

import torch


_BACKEND_NAME = "lupine"
_ACTIVE_SESSION: Any | None = None
_TENSOR_CHUNK_BYTES = 8 * 1024 * 1024


class SidecarError(RuntimeError):
    """Raised when the sidecar PyTorch worker fails."""


def _get_active_session() -> Any:
    return _ACTIVE_SESSION


def _set_active_session(session: Any) -> None:
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = session


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


def _session_from(value: Any) -> Any:
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


@cache
def _ensure_registered() -> None:
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


class SidecarTensor(torch.Tensor):
    @staticmethod
    def __new__(
        cls,
        *,
        session: Any,
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
        session: Any,
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
            raise SidecarError(
                f"sidecar LUPINE dispatch could not find a session for {func}"
            )
        return session.forward(func, args, kwargs)


class SidecarDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, session: Any) -> None:
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
