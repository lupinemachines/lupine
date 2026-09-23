"""The ``torch.lupine`` device module, also installed as ``torch.cuda`` when
the host torch has no CUDA build.

Only what must live on the host is defined here: the device the backend
selects and counts (``is_available``, ``device_count``, ``current_device``,
``set_device``, ``device``, ``_lazy_init``, ``_is_in_bad_fork``, ...),
streams and events (stubs: the worker runs every op on one stream in issue
order), CUDA graphs (captured and replayed by the worker's torch),
``synchronize``, and the installation over ``torch.cuda``. Every other
``torch.cuda`` function (device properties and capabilities, memory
statistics, RNG seeds and states, ``current_blas_handle``, ``empty_cache``,
``mem_get_info``, ...) is resolved by ``__getattr__`` on the worker's real
``torch.cuda``: the call travels pickled, runs with the host's current device
selected, and its picklable result comes back (RNG states as CPU tensors).
"""

from __future__ import annotations

import base64
import functools
import gc
import inspect
import pickle
import warnings
import weakref
from typing import Any

import torch

from . import _extension, forward, is_dual, is_started

_DEVICE_NAME = "lupine"


def _C() -> Any:
    return _extension()


def _device_type() -> str:
    return "cuda" if is_dual() else _DEVICE_NAME


def _index(device: Any) -> int:
    if device is None:
        return _C().current_device()
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type not in ("cuda", _DEVICE_NAME):
            raise ValueError(f"expected a {_device_type()} device, got {device}")
        return _C().current_device() if device.index is None else device.index
    raise TypeError(f"invalid device: {device!r}")


def _eval(code: str) -> Any:
    return forward.current().eval_(code)


def _exec(code: str) -> None:
    forward.current().exec_(code)


# --- forwarding to the worker's torch.cuda ----------------------------------


class _Properties:
    """``torch.cuda.get_device_properties`` as the worker reports it."""

    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)

    def __repr__(self) -> str:
        return (
            f"_CudaDeviceProperties(name='{self.name}', major={self.major}, "
            f"minor={self.minor}, total_memory={self.total_memory >> 20}MB, "
            f"multi_processor_count={self.multi_processor_count})"
        )


def _to_worker(value: Any) -> Any:
    if isinstance(value, torch.device) and value.type in ("cuda", _DEVICE_NAME):
        return torch.device("cuda") if value.index is None else torch.device("cuda", value.index)
    return value


def _forwarded(name: str) -> Any:
    def call(*args: Any, **kwargs: Any) -> Any:
        payload = ([_to_worker(a) for a in args], {k: _to_worker(v) for k, v in kwargs.items()})
        encoded = base64.b64encode(pickle.dumps(payload)).decode()
        result = _eval(f"_cuda({name!r}, {encoded!r}, {_C().current_device()})")
        return pickle.loads(base64.b64decode(result))

    call.__name__ = name
    return call


def __getattr__(name: str) -> Any:
    if name.startswith("_") or not inspect.isfunction(getattr(torch.cuda, name, None)):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _forwarded(name)


# --- devices ----------------------------------------------------------------


def is_available() -> bool:
    return is_started() and _C().device_count() > 0


def is_initialized() -> bool:
    return is_started()


def init() -> None:
    pass


def _lazy_init() -> None:
    pass


def _is_in_bad_fork() -> bool:
    return False


def device_count() -> int:
    return _C().device_count() if is_started() else 0


def current_device() -> int:
    return _C().current_device()


def set_device(device: Any) -> None:
    index = _index(device)
    if index >= 0:
        _C().set_device(index)


def _exchange_device(device: int) -> int:
    return _C().exchange_device(device)


def _maybe_exchange_device(device: int) -> int:
    return _C().exchange_device(device)


class device:
    """Context manager selecting a device, as ``torch.cuda.device`` does."""

    def __init__(self, device: Any):
        self.idx = -1 if device is None else _index(device)
        self.prev_idx = -1

    def __enter__(self) -> None:
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, *args: object) -> bool:
        self.idx = _maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    def __init__(self, obj: Any):
        super().__init__(obj.get_device() if obj.device.type != "cpu" else -1)


# Autocast asks the PrivateUse1 device module for this; torch.cuda has no
# such function to forward to.
def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def synchronize(device: Any = None) -> None:
    forward.current().synchronize()


# --- streams and events -----------------------------------------------------


class Stream:
    """A stream stub: the worker runs every op on one stream in issue order."""

    def __init__(self, device: Any = None, priority: int = 0, **kwargs: Any):
        self.device = torch.device(_device_type(), _index(device))
        self.device_index = self.device.index
        self.device_type = 1
        self.stream_id = 0
        self.priority = priority
        self.cuda_stream = 0

    def wait_stream(self, stream: Stream) -> None:
        pass

    def wait_event(self, event: Event) -> None:
        pass

    def record_event(self, event: Event | None = None) -> Event:
        event = Event() if event is None else event
        event.record(self)
        return event

    def synchronize(self) -> None:
        synchronize()

    def query(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Stream) and other.device == self.device

    def __hash__(self) -> int:
        return hash((self.device, 0))

    def __repr__(self) -> str:
        return f"<lupine.Stream device={self.device}>"


ExternalStream = Stream


class Event:
    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        self.enable_timing = enable_timing
        self._recorded = False

    def record(self, stream: Stream | None = None) -> None:
        self._recorded = True

    def wait(self, stream: Stream | None = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        synchronize()

    def elapsed_time(self, end_event: Event) -> float:
        return 0.0

    def ipc_handle(self) -> bytes:
        raise RuntimeError("lupine: events cannot be shared between processes")


class StreamContext:
    def __init__(self, stream: Stream | None):
        self.stream = stream

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: object) -> None:
        pass


def stream(stream: Stream | None) -> StreamContext:
    return StreamContext(stream)


def set_stream(stream: Stream) -> None:
    pass


def current_stream(device: Any = None) -> Stream:
    return Stream(device)


def default_stream(device: Any = None) -> Stream:
    return Stream(device)


# --- CUDA graphs ------------------------------------------------------------


def _release_graph(graph_id: int) -> None:
    if is_started():
        _exec(f"_reset({graph_id})")


class CUDAGraph:
    """A graph captured and replayed by the worker's torch.

    Between ``capture_begin`` and ``capture_end`` the worker runs every op on
    a capture stream, so the same ops the host issues eagerly are recorded
    there; ``replay`` is one message.
    """

    _next_id = 0
    capturing = False

    def __init__(self, keep_graph: bool = False):
        self._id = CUDAGraph._next_id
        CUDAGraph._next_id += 1
        weakref.finalize(self, _release_graph, self._id)

    def capture_begin(self, pool: Any = None, capture_error_mode: str = "global") -> None:
        _exec(f"_capture_begin({self._id}, {capture_error_mode!r}, {pool!r})")
        CUDAGraph.capturing = True

    def capture_end(self) -> None:
        CUDAGraph.capturing = False
        _exec(f"_capture_end({self._id})")

    def replay(self) -> None:
        _exec(f"_replay({self._id})")

    def reset(self) -> None:
        _exec(f"_reset({self._id})")

    def pool(self) -> None:
        return None

    def enable_debug_mode(self) -> None:
        pass

    def debug_dump(self, debug_path: str) -> None:
        pass



class graph:
    def __init__(self, cuda_graph: CUDAGraph, pool: Any = None, stream: Stream | None = None,
                 capture_error_mode: str = "global"):
        self.cuda_graph = cuda_graph
        self.pool = pool
        self.capture_error_mode = capture_error_mode

    def __enter__(self) -> None:
        synchronize()
        gc.collect()
        _forwarded("empty_cache")()
        self.cuda_graph.capture_begin(self.pool, self.capture_error_mode)

    def __exit__(self, *args: object) -> None:
        self.cuda_graph.capture_end()


def is_current_stream_capturing() -> bool:
    return CUDAGraph.capturing


# --- installation -----------------------------------------------------------

# What this module defines for torch.cuda; its other public functions are
# forwarded to the worker.
_HOST_NAMES = (
    "is_available", "is_initialized", "init", "_lazy_init", "_is_in_bad_fork",
    "device_count", "current_device", "set_device", "_exchange_device",
    "_maybe_exchange_device", "device", "device_of", "synchronize", "Stream",
    "ExternalStream", "Event", "StreamContext", "stream", "set_stream",
    "current_stream", "default_stream", "CUDAGraph", "graph",
    "is_current_stream_capturing",
)


def install_as_cuda() -> None:
    """Route ``torch.cuda`` to the backend on a torch with no CUDA build."""

    import torch.cuda
    import torch.cuda.random

    this = globals()
    for name in _HOST_NAMES:
        setattr(torch.cuda, name, this[name])
    for module in (torch.cuda, torch.cuda.random):
        for name, value in list(vars(module).items()):
            if name not in _HOST_NAMES and not name.startswith("_") and inspect.isfunction(value):
                setattr(module, name, _forwarded(name))
    torch.cuda._initialized = True
    torch.cuda.graphs.CUDAGraph = CUDAGraph
    torch.cuda.graphs.graph = graph


_FORWARDED_ATTRIBUTES = {
    "torch.backends.cuda.matmul": (
        "allow_tf32", "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction", "fp32_precision",
    ),
    "torch.backends.cudnn": (
        "enabled", "benchmark", "deterministic", "allow_tf32", "benchmark_limit",
        "conv.fp32_precision", "rnn.fp32_precision",
    ),
}


def _forward_attribute_setter(path: str, names: tuple[str, ...]) -> None:
    target: Any = torch
    for part in path.split(".")[1:]:
        target = getattr(target, part)
    cls = type(target)
    original = cls.__setattr__

    def __setattr__(self: Any, name: str, value: Any) -> None:
        original(self, name, value)
        if self is target and name in names:
            _exec(f"{path}.{name} = {value!r}")

    cls.__setattr__ = __setattr__


def forward_context_settings() -> None:
    """Numerics flags set on the host are mirrored on the worker."""

    for path, names in _FORWARDED_ATTRIBUTES.items():
        _forward_attribute_setter(path, names)

    original_precision = torch.set_float32_matmul_precision

    @functools.wraps(original_precision)
    def set_float32_matmul_precision(precision: str) -> None:
        original_precision(precision)
        _exec(f"torch.set_float32_matmul_precision({precision!r})")

    torch.set_float32_matmul_precision = set_float32_matmul_precision

    original_deterministic = torch.use_deterministic_algorithms

    @functools.wraps(original_deterministic)
    def use_deterministic_algorithms(mode: bool, *, warn_only: bool = False) -> None:
        original_deterministic(mode, warn_only=warn_only)
        _exec(f"torch.use_deterministic_algorithms({bool(mode)}, warn_only={bool(warn_only)})")

    torch.use_deterministic_algorithms = use_deterministic_algorithms


def keep_compile_eager() -> None:
    """``torch.compile`` has no inductor backend here; programs stay eager."""

    warned = False

    def compile(model: Any = None, *args: Any, **kwargs: Any) -> Any:
        nonlocal warned
        if not warned:
            warned = True
            warnings.warn(
                "torch.compile is not supported on the lupine torch backend; "
                "running eagerly",
                stacklevel=2,
            )
        if model is None:
            return lambda fn: fn
        return model

    torch.compile = compile
