"""torch.lupine: devices, synchronization, and a single synchronous stream."""

from __future__ import annotations

from typing import Any

import torch

from . import _extension, _started, is_started


def _index(device: Any) -> int:
    if device is None:
        return current_device()
    if isinstance(device, int):
        return device
    value = torch.device(device)
    if value.type != "lupine":
        raise ValueError(f"expected a lupine device, got {value}")
    return current_device() if value.index is None else value.index


def _backend():
    if not is_started():
        raise RuntimeError("Lupine is not connected")
    return _started["backend"]


def __getattr__(name: str):
    if name.startswith("_") or not hasattr(torch.cuda, name):
        raise AttributeError(name)
    return lambda *args, **kwargs: _backend().cuda_call(name, *args, **kwargs)


class _Properties:
    def __init__(self, values):
        self.__dict__.update(values)

    def __repr__(self):
        return (
            f"LupineDevice(name={self.name!r}, capability={self.major}.{self.minor}, "
            f"memory={self.total_memory})"
        )


def is_available():
    return is_started() and device_count() > 0


def is_initialized():
    return is_started()


def init():
    _backend()


def _lazy_init():
    init()


def _is_in_bad_fork():
    return False


def device_count():
    return _extension().device_count() if is_started() else 0


def current_device():
    return _extension().current_device()


def set_device(device):
    _extension().set_device(_index(device))


def _exchange_device(index):
    return _extension().exchange_device(index)


def _maybe_exchange_device(index):
    return _exchange_device(index)


class device:
    def __init__(self, value):
        self.index = _index(value)

    def __enter__(self):
        self.previous = _exchange_device(self.index)
        return self

    def __exit__(self, *args):
        _exchange_device(self.previous)


def synchronize(device=None):
    if _index(device) != 0:
        raise ValueError("Lupine currently supports device 0")
    _backend().synchronize()


def get_amp_supported_dtype():
    # There are no AMP kernels yet.
    return []


class Stream:
    def __init__(self, device=None, priority=0, **kwargs):
        index = _index(device)
        if index != 0 or priority != 0 or kwargs:
            raise NotImplementedError("Lupine currently uses one default stream")
        self.device = torch.device("lupine", index)
        self.device_index = index
        self.device_type = torch._C._get_privateuse1_backend_name()
        self.stream_id = 0
        self.cuda_stream = 0

    def synchronize(self):
        synchronize(self.device)

    def query(self):
        return True  # Every native launch synchronizes before returning.

    def wait_stream(self, stream):
        stream.synchronize()

    def wait_event(self, event):
        event.synchronize()


def current_stream(device=None):
    return Stream(device)


def default_stream(device=None):
    return Stream(device)


def manual_seed_all(seed):
    _backend().manual_seed(seed)


def manual_seed(seed):
    manual_seed_all(seed)


def initial_seed():
    return _backend().seed
