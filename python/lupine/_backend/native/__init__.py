"""Experimental PyTorch kernels backed by the native Mac CUDA dylibs.

Reuses the historical backend's device registration and metadata-only
storages. Each storage handle owns a real CUDA allocation here; every
supported tensor operation launches a GPU kernel. Unsupported operators
raise, rather than silently running arithmetic on the CPU.
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
import math

from .runtime import Runtime, ct, I, Q, P, F


class Backend:
    native = True

    def __init__(self, C):
        import torch

        self.torch = torch
        self.C = C
        self.kind = "lupine"
        self.runtime = Runtime()
        self.allocations = {}
        self.libraries = []

    def dtype(self, tensor):
        if tensor.dtype == self.torch.float32:
            return 0
        if tensor.dtype == self.torch.int64:
            return 1
        raise NotImplementedError(
            f"Lupine native supports float32 and int64, got {tensor.dtype}"
        )

    def floats(self, *tensors):
        for tensor in tensors:
            if tensor.ndim > 8:
                raise NotImplementedError(
                    "Lupine native supports at most eight dimensions"
                )
            if tensor.dtype != self.torch.float32:
                raise NotImplementedError(
                    "Lupine native arithmetic currently supports float32"
                )

    def ours(self, tensor):
        return isinstance(tensor, self.torch.Tensor) and tensor.device.type == self.kind

    def pointer(self, tensor):
        self.dtype(tensor)
        if not self.ours(tensor) or tensor.device.index != 0:
            raise ValueError("Lupine native expected a tensor on device 0")
        storage = tensor.untyped_storage()
        handle = storage.data_ptr() >> 20
        size = storage.nbytes()
        existing = self.allocations.get(handle)
        if existing is None or size > existing[1]:
            pointer = self.runtime.malloc(size)
            if existing is not None:
                try:
                    self.runtime.check(
                        self.runtime.copy(pointer, existing[0], existing[1], 3),
                        "cudaMemcpy(grow storage)",
                    )
                except BaseException:
                    self.runtime.free(pointer)
                    raise
                self.runtime.check(
                    self.runtime.free(existing[0]), "cudaFree(grow storage)"
                )
            self.allocations[handle] = (pointer, size)
        return P(
            self.allocations[handle][0]
            + tensor.storage_offset() * tensor.element_size()
        )

    def descriptor(self, tensor, shape=None):
        shape = tuple(tensor.shape if shape is None else shape)
        if len(shape) > 8:
            raise NotImplementedError("Lupine native supports at most eight dimensions")
        if len(tensor.shape) > len(shape):
            raise ValueError("cannot broadcast to fewer dimensions")
        pad = len(shape) - tensor.ndim
        sizes = (1,) * pad + tuple(tensor.shape)
        strides = (0,) * pad + tuple(tensor.stride())
        if any(size != target and size != 1 for size, target in zip(sizes, shape)):
            raise ValueError("tensor sizes do not broadcast")
        strides = tuple(
            0 if size == 1 else stride for size, stride in zip(sizes, strides)
        )
        return (
            list(shape)
            + [1] * (8 - len(shape))
            + list(strides)
            + [0] * (8 - len(shape))
        )

    def empty(self, shape, dtype=None, device=None):
        return self.torch.empty(
            shape, dtype=dtype or self.torch.float32, device=device or self.kind
        )

    def copy_(self, destination, source, non_blocking=False):
        torch = self.torch
        if self.ours(destination):
            if any(
                size > 1 and stride == 0
                for size, stride in zip(destination.shape, destination.stride())
            ):
                raise ValueError(
                    "cannot write to an expanded tensor with overlapping elements"
                )
            if (
                self.ours(source)
                and destination.untyped_storage().data_ptr()
                == source.untyped_storage().data_ptr()
            ):
                if (
                    destination.shape == source.shape
                    and destination.stride() == source.stride()
                    and destination.storage_offset() == source.storage_offset()
                    and destination.dtype == source.dtype
                ):
                    return destination
                raise NotImplementedError(
                    "Lupine native does not support copies between aliased views"
                )
        with self.runtime.active(), ExitStack() as stack:
            if not self.ours(destination):
                if destination.device.type != "cpu" or not self.ours(source):
                    raise NotImplementedError(
                        "Lupine native copies only CPU and its own device"
                    )
                packed = self.empty(destination.shape, destination.dtype)
                self.copy_(packed, source)
                host = torch.empty(destination.shape, dtype=destination.dtype)
                self.runtime.check(
                    self.runtime.copy(
                        host.data_ptr(),
                        self.pointer(packed),
                        host.numel() * host.element_size(),
                        2,
                    ),
                    "cudaMemcpy(DtoH)",
                )
                destination.copy_(host)
                return destination
            if not self.ours(source):
                if source.device.type != "cpu":
                    raise NotImplementedError(
                        "Lupine native cannot copy another accelerator"
                    )
                host = source.contiguous()
                source_pointer = self.runtime.malloc(host.numel() * host.element_size())
                stack.callback(self.runtime.free, source_pointer)
                self.runtime.check(
                    self.runtime.copy(
                        source_pointer,
                        host.data_ptr(),
                        host.numel() * host.element_size(),
                        1,
                    ),
                    "cudaMemcpy(HtoD)",
                )
                source = host
                source_pointer = P(source_pointer)
            else:
                source_pointer = self.pointer(source)
            destination_pointer = self.pointer(destination)
            od = stack.enter_context(
                self.runtime.uploaded(self.descriptor(destination))
            )
            id = stack.enter_context(
                self.runtime.uploaded(self.descriptor(source, destination.shape))
            )
            self.runtime.kernel(
                "copy_tensor",
                destination.numel(),
                destination_pointer,
                source_pointer,
                od,
                id,
                I(destination.ndim),
                Q(destination.numel()),
                I(self.dtype(destination)),
                I(self.dtype(source)),
            )
            return destination

    def copy_from(self, source, destination, non_blocking=False):
        return self.copy_(destination, source, non_blocking)

    def scalar(self, tensor):
        host = self.torch.empty((), dtype=tensor.dtype)
        self.copy_(host, tensor)
        return host.item()

    def fill(self, tensor, value):
        dtype = self.dtype(tensor)
        integer = int(value) if dtype == 1 else 0
        if dtype == 1 and not -(2**63) <= integer < 2**63:
            raise OverflowError("fill value does not fit int64")
        with self.runtime.active(), self.runtime.uploaded(
            self.descriptor(tensor)
        ) as desc:
            self.runtime.kernel(
                "fill_tensor",
                tensor.numel(),
                self.pointer(tensor),
                desc,
                I(tensor.ndim),
                Q(tensor.numel()),
                I(dtype),
                F(value),
                Q(integer),
            )
        return tensor

    def arange(
        self,
        start,
        end,
        step=1,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        out=None,
    ):
        if step == 0 or not all(math.isfinite(float(x)) for x in (start, end, step)):
            raise ValueError("arange requires finite start/end and a nonzero step")
        if (end - start) * step < 0:
            raise ValueError("arange bounds are inconsistent with the step")
        if layout not in (None, self.torch.strided) or pin_memory:
            raise NotImplementedError(
                "Lupine native arange requires unpinned strided storage"
            )
        n = max(0, math.ceil((end - start) / step))
        if out is not None:
            result = out
            if result.numel() != n:
                result.resize_((n,))
            if not result.is_contiguous():
                raise NotImplementedError("Lupine native arange.out must be contiguous")
        else:
            dtype = dtype or (
                self.torch.float32
                if any(isinstance(x, float) for x in (start, end, step))
                else self.torch.int64
            )
            result = self.empty((n,), dtype, device)
        with self.runtime.active():
            self.runtime.kernel(
                "arange_tensor",
                n,
                self.pointer(result),
                Q(n),
                I(self.dtype(result)),
                F(start),
                F(step),
                Q(int(start)),
                Q(int(step)),
            )
        return result

    def binary(self, name, a, b, *, alpha=1, out=None):
        torch = self.torch
        self.floats(a)
        scalar = (
            not isinstance(b, torch.Tensor) or b.device.type == "cpu" and b.ndim == 0
        )
        if scalar:
            value = b.item() if isinstance(b, torch.Tensor) else b
            shape = a.shape
        else:
            self.floats(b)
            shape = torch.broadcast_shapes(a.shape, b.shape)
            value = 0
        result = self.empty(shape)
        with self.runtime.active(), ExitStack() as stack:
            ad = stack.enter_context(self.runtime.uploaded(self.descriptor(a, shape)))
            bd = (
                ad
                if scalar
                else stack.enter_context(
                    self.runtime.uploaded(self.descriptor(b, shape))
                )
            )
            self.runtime.kernel(
                "binary_tensor",
                result.numel(),
                self.pointer(result),
                self.pointer(a),
                P() if scalar else self.pointer(b),
                ad,
                bd,
                I(len(shape)),
                Q(result.numel()),
                I(("add", "sub", "mul", "div").index(name)),
                I(scalar),
                F(value),
                F(alpha),
            )
        if out is not None:
            if tuple(out.shape) != tuple(shape):
                out.resize_(shape)
            return self.copy_(out, result)
        return result

    def unary(self, tensor, op):
        self.floats(tensor)
        result = self.empty(tensor.shape)
        with self.runtime.active(), self.runtime.uploaded(
            self.descriptor(tensor)
        ) as desc:
            self.runtime.kernel(
                "unary_tensor",
                result.numel(),
                self.pointer(result),
                self.pointer(tensor),
                desc,
                I(tensor.ndim),
                Q(tensor.numel()),
                I(op),
            )
        return result

    def mm(self, a, b, *, out=None):
        self.floats(a, b)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("mm requires compatible matrices")
        m, k = a.shape
        n = b.shape[1]
        result = self.empty((m, n))
        with self.runtime.active():
            self.runtime.kernel(
                "matmul_tensor",
                m * n,
                self.pointer(result),
                self.pointer(a),
                self.pointer(b),
                Q(m),
                Q(n),
                Q(k),
                *(Q(x) for x in (*a.stride(), *b.stride())),
            )
        return self.copy_(out, result) if out is not None else result

    def reduce(
        self, tensor, dim=None, keepdim=False, *, dtype=None, mean=False, out=None
    ):
        self.floats(tensor)
        if dtype not in (None, self.torch.float32):
            raise NotImplementedError("Lupine native reductions currently use float32")
        dimensions = (
            tuple(range(tensor.ndim))
            if dim is None or dim == [] or dim == ()
            else ((dim,) if isinstance(dim, int) else tuple(dim))
        )
        if any(d < -tensor.ndim or d >= tensor.ndim for d in dimensions):
            raise IndexError("reduction dimension out of range")
        dimensions = tuple(d % tensor.ndim for d in dimensions)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("duplicate reduction dimension")
        outer = [i for i in range(tensor.ndim) if i not in dimensions]
        shape = (
            tuple(1 if i in dimensions else s for i, s in enumerate(tensor.shape))
            if keepdim
            else (tuple(tensor.shape[i] for i in outer))
        )
        result = self.empty(shape)

        def descriptor(axes):
            return (
                [tensor.shape[i] for i in axes]
                + [1] * (8 - len(axes))
                + [tensor.stride()[i] for i in axes]
                + [0] * (8 - len(axes))
            )

        with self.runtime.active(), self.runtime.uploaded(
            descriptor(outer)
        ) as od, self.runtime.uploaded(descriptor(dimensions)) as id:
            self.runtime.kernel(
                "sum_tensor",
                result.numel(),
                self.pointer(result),
                self.pointer(tensor),
                od,
                id,
                I(len(outer)),
                I(len(dimensions)),
                Q(result.numel()),
                Q(math.prod(tensor.shape[i] for i in dimensions)),
                I(mean),
            )
        return self.copy_(out, result) if out is not None else result

    def fallback(self, op, *args, **kwargs):
        raise NotImplementedError(
            f"Lupine native GPU kernel is not implemented for {op}"
        )

    def release(self, handle):
        with self.runtime.active():
            allocation = self.allocations.pop(handle, None)
            if allocation is not None:
                self.runtime.check(
                    self.runtime.free(allocation[0]), "cudaFree(storage)"
                )

    def synchronize(self):
        self.runtime.synchronize()

    def cuda_call(self, name, *args, **kwargs):
        if name in (
            "get_device_name",
            "get_device_properties",
            "get_device_capability",
        ):
            from ..device import _Properties, _index

            if _index(args[0] if args else kwargs.get("device")) != 0:
                raise ValueError("Lupine native supports device 0")
            with self.runtime.active():
                values = []
                for attribute in (75, 76, 16):
                    value = I()
                    self.runtime.check(
                        self.runtime.attribute(ct.byref(value), attribute, 0),
                        "cudaDeviceGetAttribute",
                    )
                    values.append(value.value)
                free, total = ct.c_size_t(), ct.c_size_t()
                self.runtime.check(
                    self.runtime.mem_info(ct.byref(free), ct.byref(total)),
                    "cudaMemGetInfo",
                )
                buffer = ct.create_string_buffer(256)
                get_name = self.runtime.bind(
                    self.runtime.driver, "cuDeviceGetName", P, I, I
                )
                self.runtime.check(get_name(buffer, 256, 0), "cuDeviceGetName")
            properties = _Properties(
                dict(
                    name=buffer.value.decode(),
                    major=values[0],
                    minor=values[1],
                    multi_processor_count=values[2],
                    total_memory=total.value,
                )
            )
            if name == "get_device_name":
                return properties.name
            if name == "get_device_capability":
                return properties.major, properties.minor
            return properties
        if name in (
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
        ):
            # There is no caching allocator in this prototype.
            if name.startswith("max_"):
                raise NotImplementedError(
                    "Lupine native does not track peak memory yet"
                )
            with self.runtime.active():
                return sum(size for pointer, size in self.allocations.values())
        if name == "empty_cache":
            return None
        raise NotImplementedError(f"Lupine native does not implement torch.cuda.{name}")

    def close(self):
        if self.runtime.closed:
            return
        with self.runtime.active():
            self.runtime.check(self.runtime.sync(), "cudaDeviceSynchronize")
            for pointer, size in self.allocations.values():
                self.runtime.check(self.runtime.free(pointer), "cudaFree(storage)")
            self.allocations.clear()
            self.runtime.check(
                self.runtime.unload(self.runtime.module), "cuModuleUnload"
            )
            self.runtime.closed = True

    def install(self):
        from torch.library import Library

        key = "PrivateUse1"
        fallback = Library("_", "IMPL")
        fallback.fallback(self.fallback, key)
        aten = Library("aten", "IMPL", key)
        aten.impl("copy_", self.copy_)
        aten.impl("_copy_from", self.copy_from)
        aten.impl("_local_scalar_dense", self.scalar)
        aten.impl("fill_.Scalar", self.fill)
        aten.impl("zero_", lambda tensor: self.fill(tensor, 0))
        aten.impl("arange", lambda end, **kw: self.arange(0, end, **kw))
        aten.impl(
            "arange.start", lambda start, end, **kw: self.arange(start, end, **kw)
        )
        aten.impl("arange.start_step", self.arange)
        aten.impl("arange.out", lambda end, **kw: self.arange(0, end, **kw))
        aten.impl("arange.start_out", self.arange)
        for name in ("add", "sub", "mul", "div"):

            def binary(a, b, _name=name, **kw):
                return self.binary(_name, a, b, **kw)

            for suffix in ("Tensor", "Scalar", "out"):
                aten.impl(name + "." + suffix, binary)

            def inplace(a, b, _name=name, **kw):
                return self.binary(_name, a, b, out=a, **kw)

            for suffix in ("Tensor", "Scalar"):
                aten.impl(name + "_." + suffix, inplace)
        aten.impl("neg", lambda t: self.unary(t, 0))
        aten.impl("relu", lambda t: self.unary(t, 1))
        aten.impl("mm", self.mm)
        aten.impl("mm.out", self.mm)
        aten.impl("sum", self.reduce)
        aten.impl("sum.dim_IntList", self.reduce)
        aten.impl("mean", lambda t, **kw: self.reduce(t, mean=True, **kw))
        aten.impl(
            "mean.dim",
            lambda t, dim, keepdim=False, **kw: self.reduce(
                t, dim, keepdim, mean=True, **kw
            ),
        )
        self.libraries.extend((fallback, aten))


def start():
    from .. import _extension, _started, info
    from ... import LupineError

    if _started:
        if not getattr(_started.get("backend"), "native", False):
            raise LupineError("the worker backend is already active")
        return info()
    C = _extension()  # CUDA hooks must precede importing native torch.
    import torch

    backend = Backend(C)
    try:
        backend.install()
        C.register(1, False, backend.release, backend.synchronize)
        _started.update(
            address="native",
            dual=False,
            info=dict(device_count=1, torch=torch.__version__, native=True),
            backend=backend,
        )
        from .. import device

        torch.utils.rename_privateuse1_backend("lupine")
        torch._register_device_module("lupine", device)
        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
    except BaseException:
        C.unregister()
        backend.close()
        raise
    atexit.register(backend.close)
    atexit.register(C.unregister)
    return info()
