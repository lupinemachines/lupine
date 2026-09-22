"""The ``torch.lupine`` device module, also installed as ``torch.cuda`` when
the host torch has no CUDA build.

Everything here is host-side state or a message to the worker: streams and
events are stubs because the worker runs every op on one stream in issue
order; CUDA graphs are captured and replayed by the worker's own torch.
"""

from __future__ import annotations

import base64
import functools
import gc
import json
import warnings
from typing import Any

import torch

from . import _extension, is_dual, is_started

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
    return json.loads(_C().eval(code))


def _exec(code: str) -> None:
    _C().exec(code)


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


class _Properties:
    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)

    def __repr__(self) -> str:
        return (
            f"_CudaDeviceProperties(name='{self.name}', major={self.major}, "
            f"minor={self.minor}, total_memory={self.total_memory >> 20}MB, "
            f"multi_processor_count={self.multi_processor_count})"
        )


@functools.lru_cache(maxsize=None)
def get_device_properties(device: Any = None) -> _Properties:
    from . import info

    return _Properties(info()["devices"][_index(device)])


def get_device_name(device: Any = None) -> str:
    return get_device_properties(device).name


def get_device_capability(device: Any = None) -> tuple[int, int]:
    props = get_device_properties(device)
    return props.major, props.minor


def get_arch_list() -> list[str]:
    return _eval("torch.cuda.get_arch_list()")


def is_bf16_supported(including_emulation: bool = True) -> bool:
    return bool(_eval(f"torch.cuda.is_bf16_supported({bool(including_emulation)})"))


def can_device_access_peer(device: Any, peer_device: Any) -> bool:
    return bool(
        _eval(f"torch.cuda.can_device_access_peer({_index(device)}, {_index(peer_device)})")
    )


def get_amp_supported_dtype() -> list[torch.dtype]:
    return [torch.float16, torch.bfloat16]


def synchronize(device: Any = None) -> None:
    _C().synchronize()


def ipc_collect() -> None:
    pass


# --- memory -----------------------------------------------------------------


def empty_cache() -> None:
    _exec("torch.cuda.empty_cache()")


def _memory_query(name: str, device: Any) -> Any:
    return _eval(f"torch.cuda.{name}({_index(device)})")


def memory_allocated(device: Any = None) -> int:
    return int(_memory_query("memory_allocated", device))


def max_memory_allocated(device: Any = None) -> int:
    return int(_memory_query("max_memory_allocated", device))


def memory_reserved(device: Any = None) -> int:
    return int(_memory_query("memory_reserved", device))


def max_memory_reserved(device: Any = None) -> int:
    return int(_memory_query("max_memory_reserved", device))


def reset_peak_memory_stats(device: Any = None) -> None:
    _exec(f"torch.cuda.reset_peak_memory_stats({_index(device)})")


def reset_max_memory_allocated(device: Any = None) -> None:
    reset_peak_memory_stats(device)


def memory_stats(device: Any = None) -> dict[str, Any]:
    return dict(_memory_query("memory_stats", device))


def mem_get_info(device: Any = None) -> tuple[int, int]:
    free, total = _memory_query("mem_get_info", device)
    return int(free), int(total)


# --- random -----------------------------------------------------------------


def manual_seed(seed: int) -> None:
    _exec(f"torch.cuda.manual_seed({int(seed)})")


def manual_seed_all(seed: int) -> None:
    _exec(f"torch.cuda.manual_seed_all({int(seed)})")


def seed() -> None:
    _exec("torch.cuda.seed()")


def seed_all() -> None:
    _exec("torch.cuda.seed_all()")


def initial_seed() -> int:
    return int(_eval("torch.cuda.initial_seed()"))


def get_rng_state(device: Any = None) -> torch.Tensor:
    encoded = _eval(
        "__import__('base64').b64encode("
        f"torch.cuda.get_rng_state({_index(device)}).numpy().tobytes()).decode()"
    )
    return torch.frombuffer(bytearray(base64.b64decode(encoded)), dtype=torch.uint8).clone()


def set_rng_state(new_state: torch.Tensor, device: Any = None) -> None:
    encoded = base64.b64encode(new_state.cpu().numpy().tobytes()).decode()
    _exec(
        "torch.cuda.set_rng_state(torch.frombuffer(bytearray(__import__('base64')"
        f".b64decode({encoded!r})), dtype=torch.uint8).clone(), {_index(device)})"
    )


def get_rng_state_all() -> list[torch.Tensor]:
    return [get_rng_state(i) for i in range(device_count())]


def set_rng_state_all(new_states: list[torch.Tensor]) -> None:
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


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


def current_blas_handle() -> int:
    return 0


# --- CUDA graphs ------------------------------------------------------------


class CUDAGraph:
    """A graph captured and replayed by the worker's torch.

    Between ``capture_begin`` and ``capture_end`` the worker runs every op on
    a capture stream, so the same ops the host issues eagerly are recorded
    there; ``replay`` is one message.
    """

    _next_id = 0

    def __init__(self, keep_graph: bool = False):
        self._id = CUDAGraph._next_id
        CUDAGraph._next_id += 1

    def capture_begin(self, pool: Any = None, capture_error_mode: str = "global") -> None:
        if pool is not None:
            raise RuntimeError("lupine: shared graph memory pools are not supported")
        _exec(f"_capture_begin({self._id}, {capture_error_mode!r})")

    def capture_end(self) -> None:
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

    def __del__(self) -> None:
        try:
            if is_started():
                _exec(f"_reset({self._id})")
        except Exception:
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
        empty_cache()
        self.cuda_graph.capture_begin(self.pool, self.capture_error_mode)

    def __exit__(self, *args: object) -> None:
        self.cuda_graph.capture_end()


def graph_pool_handle() -> None:
    raise RuntimeError("lupine: shared graph memory pools are not supported")


def is_current_stream_capturing() -> bool:
    return bool(_eval("_capturing()"))


# --- installation -----------------------------------------------------------

_CUDA_MODULE_NAMES = (
    "is_available", "is_initialized", "init", "_lazy_init",
    "_is_in_bad_fork", "device_count", "current_device", "set_device",
    "_exchange_device", "_maybe_exchange_device", "device", "device_of",
    "get_device_properties", "get_device_name", "get_device_capability",
    "get_arch_list", "is_bf16_supported", "can_device_access_peer", "synchronize",
    "ipc_collect", "empty_cache", "memory_allocated", "max_memory_allocated",
    "memory_reserved", "max_memory_reserved", "reset_peak_memory_stats",
    "reset_max_memory_allocated", "memory_stats", "mem_get_info", "manual_seed",
    "manual_seed_all", "seed", "seed_all", "initial_seed", "get_rng_state",
    "set_rng_state", "get_rng_state_all", "set_rng_state_all", "Stream",
    "ExternalStream", "Event", "StreamContext", "stream", "set_stream",
    "current_stream", "default_stream", "current_blas_handle", "CUDAGraph",
    "graph", "graph_pool_handle", "is_current_stream_capturing",
)

_RANDOM_NAMES = (
    "manual_seed", "manual_seed_all", "seed", "seed_all", "initial_seed",
    "get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all",
)


def install_as_cuda() -> None:
    """Route ``torch.cuda`` to the backend on a torch with no CUDA build."""

    import torch.cuda
    import torch.cuda.random

    this = globals()
    for name in _CUDA_MODULE_NAMES:
        setattr(torch.cuda, name, this[name])
    for name in _RANDOM_NAMES:
        setattr(torch.cuda.random, name, this[name])
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
