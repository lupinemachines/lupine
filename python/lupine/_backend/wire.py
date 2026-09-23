"""What a call looks like on the wire, shared by the host and the worker.

A call is ``pickle`` of ``(name, args, kwargs)`` with every tensor replaced by
a reference into a table sent alongside; a table entry is a descriptor
(storage handle, storage bytes, device index, dtype, sizes, strides, offset)
so either end rebuilds the view over the storage it owns for that handle. A
CPU tensor travels by value. ``torch.device`` values are renamed to ``cuda``
for the worker; dtypes, layouts and memory formats pickle as themselves.
Results come back the same way plus, per table entry, what the tensor is
relative to the arguments (see ``forward``).
"""

from __future__ import annotations

import ctypes
import io
import pickle
import threading
from collections.abc import Callable
from typing import Any

import torch

# Request kinds. OP and COPY_FROM_HOST and EXEC are fire-and-forget; the rest
# wait for a response.
OP = 1
OP_SYNC = 2
COPY_TO_HOST = 3
COPY_FROM_HOST = 4
EXEC = 5
EVAL = 6
SYNC = 7

# Response statuses.
OK = 0
ERROR = 1
MISSING_OP = 2

# Result alias kinds the worker reports.
FRESH = 0
INPUT = 1
INPUT_STORAGE = 2

Desc = tuple[int, int, int, str, tuple[int, ...], tuple[int, ...], int]

_DTYPE_NAME = {getattr(torch, n): f"torch.{n}" for n in dir(torch) if isinstance(getattr(torch, n), torch.dtype)}
_DTYPE = {name: dtype for dtype, name in _DTYPE_NAME.items()}
_local = threading.local()


def dtype_name(dtype: torch.dtype) -> str:
    return _DTYPE_NAME[dtype]


def dtype_of(name: str) -> torch.dtype:
    return _DTYPE[name]


def cpu_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    if nbytes == 0:
        return b""
    return bytes((ctypes.c_char * nbytes).from_address(t.data_ptr()))


def cpu_buffer(t: torch.Tensor) -> Any:
    """A writable buffer over a contiguous CPU tensor's bytes, without copying."""

    nbytes = t.numel() * t.element_size()
    return (ctypes.c_char * nbytes).from_address(t.data_ptr())


# --- pickling ---------------------------------------------------------------


def _at(index: int) -> torch.Tensor:
    return _local.table[index]


def _cpu(dtype: str, sizes: tuple[int, ...], data: bytes) -> torch.Tensor:
    if not data:
        return torch.empty(sizes, dtype=dtype_of(dtype))
    return torch.frombuffer(bytearray(data), dtype=dtype_of(dtype)).reshape(sizes)


def _generator(_: torch.Generator) -> Any:
    raise RuntimeError(
        "lupine: explicit torch.Generator objects are not supported on the "
        "lupine backend; use torch.manual_seed"
    )


class _Dispatch(dict):
    """A pickler dispatch table: tensors and devices by the caller's rule,
    tensor subclasses learned on first sight, everything else pickled as is."""

    def __init__(self, tensor: Callable[[torch.Tensor], Any], device: Callable[[torch.device], Any]):
        super().__init__()
        self[torch.Tensor] = tensor
        self[torch.nn.Parameter] = tensor
        self[torch.device] = device
        self[torch.Generator] = _generator

    def __missing__(self, cls: type) -> Any:
        if issubclass(cls, torch.Tensor):
            self[cls] = self[torch.Tensor]
            return self[cls]
        raise KeyError(cls)


def dumps(value: Any, tensor: Callable[[torch.Tensor], Any], device: Callable[[torch.device], Any]) -> bytes:
    """Pickles ``value`` with tensors and devices reduced by the given rules."""

    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = _Dispatch(tensor, device)
    pickler.dump(value)
    return buffer.getvalue()


def loads(data: bytes, table: list[Any]) -> Any:
    """Unpickles ``data`` with its tensor references resolved against ``table``."""

    _local.table = table
    try:
        return pickle.loads(data)
    finally:
        _local.table = None


def table_reference(table: list[torch.Tensor]) -> Callable[[torch.Tensor], Any]:
    """The tensor rule that files every tensor in ``table`` and pickles its index."""

    def reduce(t: torch.Tensor) -> Any:
        table.append(t)
        return (_at, (len(table) - 1,))

    return reduce


def cpu_value(t: torch.Tensor) -> Any:
    return (_cpu, (dtype_name(t.dtype), tuple(t.shape), cpu_bytes(t)))


def device_as(target: str, types: tuple[str, ...]) -> Callable[[torch.device], Any]:
    """The device rule that renames the backend's device to the worker's."""

    def reduce(d: torch.device) -> Any:
        if d.type in types:
            return (torch.device, (target,) if d.index is None else (target, d.index))
        return d.__reduce__()

    return reduce


# --- frames -----------------------------------------------------------------


def pack_meta(frees: list[int], table: list[Any], expected: Any) -> bytes:
    """The part of a request in front of the pickled call: the storages the
    host released since its last request, the argument descriptors, and
    what the request kind needs (a replay's expected results)."""

    return pickle.dumps((frees, table, expected), protocol=pickle.HIGHEST_PROTOCOL)


def unpack_meta(data: bytes) -> tuple[list[int], list[Any], Any]:
    return pickle.loads(data)


def result_tensors(value: Any) -> list[torch.Tensor]:
    """The tensors of an op's return value in order: a tensor, or a tuple or
    list whose items are tensors, tensor lists and scalars (which contribute
    nothing). Both ends walk a result this way, so a replay's expected
    descriptors line up with the worker's results."""

    if isinstance(value, torch.Tensor):
        return [value]
    flat: list[torch.Tensor] = []
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                flat.append(item)
            elif isinstance(item, (tuple, list)):
                flat.extend(item)
    return flat
