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


class TensorStreamError(SidecarError):
    """Raised when a framed tensor stream cannot be read completely."""


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
            raise TensorStreamError(
                "sidecar tensor stream ended before the tensor was complete"
            )
        offset += read


def _discard_bytes(stream: Any, size: int) -> None:
    buffer = bytearray(min(size, _TENSOR_CHUNK_BYTES))
    view = memoryview(buffer)
    remaining = size
    while remaining:
        count = min(remaining, len(view))
        read = stream.readinto(view[:count])
        if read is None or read <= 0:
            raise TensorStreamError(
                "sidecar tensor stream ended before the tensor was complete"
            )
        remaining -= read


def _tensor_stream_metadata(tensor: Any, dtype: Any) -> dict[str, Any]:
    element_size = torch.empty((), dtype=dtype).element_size()
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(dtype),
        "layout": "strided",
        "device": "cpu",
        "byteorder": sys.byteorder,
        "nbytes": int(tensor.numel()) * int(element_size),
    }


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

    return _tensor_stream_metadata(tensor, tensor.dtype)


def _validate_cpu_result_tensor(tensor: Any) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"cannot return a tensor on {tensor.device} from the sidecar")
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise ValueError(
            "sidecar CPU tensor results require unquantized strided layout"
        )
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError("sidecar CPU tensor results cannot be lazy views")
    if not tensor.is_contiguous():
        raise ValueError("sidecar CPU tensor results must be contiguous")


def _parse_tensor_metadata(
    payload: Mapping[str, Any],
) -> tuple[tuple[int, ...], Any, int, int, int]:
    if payload.get("device") != "cpu" or payload.get("layout") != "strided":
        raise TensorStreamError("sidecar returned an invalid CPU tensor description")
    if payload.get("byteorder") != sys.byteorder:
        raise TensorStreamError(
            "sidecar CPU tensor byte order does not match the local process"
        )

    try:
        shape = tuple(int(dimension) for dimension in payload["shape"])
        dtype = _dtype_from_name(payload["dtype"])
        size = int(payload["nbytes"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise TensorStreamError("sidecar returned malformed CPU tensor data") from exc
    if any(dimension < 0 for dimension in shape):
        raise TensorStreamError(
            "sidecar returned a CPU tensor with a negative dimension"
        )

    element_count = math.prod(shape)
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = element_count * element_size
    if size != expected_bytes:
        raise TensorStreamError(
            "sidecar returned the wrong number of bytes for a CPU tensor: "
            f"expected {expected_bytes}, got {size}"
        )
    return shape, dtype, element_count, element_size, size


def _decode_cpu_tensor(payload: Mapping[str, Any], stream: Any) -> Any:
    shape, dtype, _, _, size = _parse_tensor_metadata(payload)
    try:
        tensor = torch.empty(shape, dtype=dtype)
    except Exception:
        _discard_bytes(stream, size)
        raise
    _read_tensor(stream, tensor)
    return tensor


def _read_cuda_tensor(
    stream: Any,
    payload: Mapping[str, Any],
    *,
    dtype: Any = None,
    device: Any = "cuda:0",
    destination: Any = None,
) -> Any:
    shape, source_dtype, element_count, element_size, size = _parse_tensor_metadata(
        payload
    )
    consumed = 0
    try:
        direct = (
            destination is not None
            and tuple(destination.shape) == shape
            and destination.is_contiguous()
        )
        if direct:
            tensor = destination
        else:
            target_dtype = source_dtype if dtype is None else dtype
            tensor = torch.empty(shape, dtype=target_dtype, device=device)

        flat = tensor.reshape(-1)
        chunk_elements = max(1, _TENSOR_CHUNK_BYTES // element_size)
        staging = bytearray(min(size, chunk_elements * element_size))
        staging_view = memoryview(staging)
        for element_offset in range(0, element_count, chunk_elements):
            count = min(chunk_elements, element_count - element_offset)
            chunk_bytes = count * element_size
            offset = 0
            while offset < chunk_bytes:
                read = stream.readinto(staging_view[offset:chunk_bytes])
                if read is None or read <= 0:
                    raise TensorStreamError(
                        "sidecar tensor stream ended before the tensor was complete"
                    )
                offset += read
            consumed += chunk_bytes
            source = torch.frombuffer(staging, dtype=source_dtype, count=count)
            flat[element_offset : element_offset + count].copy_(source)

        if destination is not None and not direct:
            destination.copy_(tensor)
            return destination
        return tensor
    except TensorStreamError:
        raise
    except Exception:
        _discard_bytes(stream, size - consumed)
        raise


def _write_cuda_tensor(stream: Any, tensor: Any, dtype: Any) -> None:
    source = tensor.detach().reshape(-1)
    element_size = torch.empty((), dtype=dtype).element_size()
    chunk_elements = max(1, _TENSOR_CHUNK_BYTES // element_size)
    staging = torch.empty(
        min(source.numel(), chunk_elements), dtype=dtype, device="cpu"
    )
    for offset in range(0, source.numel(), chunk_elements):
        count = min(chunk_elements, source.numel() - offset)
        chunk = staging[:count]
        chunk.copy_(source[offset : offset + count])
        _write_tensor(stream, chunk)


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
