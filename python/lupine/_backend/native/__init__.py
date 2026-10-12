"""Experimental PyTorch operators backed by the native Mac CUDA dylibs.

Reuses the historical backend's device registration and metadata-only
storages. Each storage handle owns a real CUDA allocation here; every
supported tensor operation launches a GPU kernel. Unsupported operators
raise, rather than silently running arithmetic on the CPU.
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
import math

from .runtime import Runtime, ct, I, P
from .aten import Parameters


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
        self.seed = 0
        self.counter = 0

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
            pointer = self.runtime.operators.malloc(size)
            if existing is not None:
                try:
                    self.runtime.check(
                        self.runtime.copy(pointer, existing[0], existing[1], 3),
                        "cudaMemcpy(grow storage)",
                    )
                except BaseException:
                    self.runtime.operators.free(pointer)
                    raise
                self.runtime.operators.free(existing[0])
            self.allocations[handle] = (pointer, size)
        return P(
            self.allocations[handle][0]
            + tensor.storage_offset() * tensor.element_size()
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
            # Contiguous CPU transfers can use the tensor's final allocation.
            # cudaMemcpy waits for queued default-stream work before returning.
            if (
                destination.shape == source.shape
                and destination.dtype == source.dtype
                and destination.is_contiguous()
                and source.is_contiguous()
            ):
                kind = None
                if self.ours(destination) and source.device.type == "cpu":
                    target, origin = self.pointer(destination), source.data_ptr()
                    kind = 1
                elif destination.device.type == "cpu" and self.ours(source):
                    target, origin = destination.data_ptr(), self.pointer(source)
                    kind = 2
                if kind is not None:
                    self.runtime.check(
                        self.runtime.copy(
                            target,
                            origin,
                            destination.numel() * destination.element_size(),
                            kind,
                        ),
                        "cudaMemcpy(contiguous transfer)",
                    )
                    return destination
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
                source_pointer = self.runtime.operators.malloc(
                    host.numel() * host.element_size()
                )
                stack.callback(self.runtime.operators.free, source_pointer)
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
            self.runtime.operators.call(
                self, 0, destination, source, source_pointer=source_pointer
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
        with self.runtime.active():
            self.runtime.operators.call(
                self,
                1,
                tensor,
                parameters=Parameters(integer=integer, first=float(value)),
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
            self.runtime.operators.call(
                self,
                2,
                result,
                parameters=Parameters(
                    integer=int(start),
                    step=int(step),
                    first=float(start),
                    second=float(step),
                ),
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
        with self.runtime.active():
            self.runtime.operators.call(
                self,
                3,
                result,
                a,
                None if scalar else b,
                parameters=Parameters(
                    variant=("add", "sub", "mul", "div", "pow").index(name),
                    first=float(value),
                    alpha=alpha,
                ),
            )
        if out is not None:
            if tuple(out.shape) != tuple(shape):
                out.resize_(shape)
            return self.copy_(out, result)
        return result

    def unary(self, tensor, op):
        self.floats(tensor)
        result = self.empty(tensor.shape)
        with self.runtime.active():
            self.runtime.operators.call(
                self, 4, result, tensor, parameters=Parameters(variant=op)
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
            if m and n:
                if not k:
                    self.fill(result, 0)
                else:
                    left, right = a.contiguous(), b.contiguous()
                    self.runtime.matmul(
                        self.pointer(result),
                        self.pointer(left),
                        self.pointer(right),
                        m,
                        n,
                        k,
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

        with self.runtime.active():
            self.runtime.operators.call(
                self,
                5,
                result,
                tensor,
                parameters=Parameters(
                    variant=mean,
                    axis_count=len(dimensions),
                    axes=(ct.c_int64 * 8)(*dimensions),
                ),
            )
        return self.copy_(out, result) if out is not None else result

    def manual_seed(self, seed):
        if not -(2**63) <= seed < 2**64:
            raise ValueError("seed is outside the uint64/int64 range")
        with self.runtime.active():
            self.seed = seed % (2**64)
            self.counter = 0

    def random_(self, tensor, first=0.0, second=1.0, *, normal=False, generator=None):
        self.floats(tensor)
        if generator is not None:
            raise NotImplementedError("explicit generators are not supported yet")
        if (normal and second < 0) or (not normal and first > second):
            raise ValueError("invalid random distribution parameters")
        if any(
            size > 1 and stride == 0
            for size, stride in zip(tensor.shape, tensor.stride())
        ):
            raise ValueError("cannot write random values to overlapping elements")
        with self.runtime.active():
            parameters = Parameters(
                variant=normal,
                seed=self.seed,
                offset=self.counter,
                first=first,
                second=second,
            )
            self.runtime.operators.call(self, 6, tensor, parameters=parameters)
            self.counter = parameters.offset
        return tensor

    def activation_backward(self, grad, value, op, threshold=0):
        self.floats(grad, value)
        if grad.shape != value.shape:
            raise ValueError("activation backward requires matching shapes")
        result = self.empty(value.shape)
        with self.runtime.active():
            self.runtime.operators.call(
                self,
                7,
                result,
                grad,
                value,
                parameters=Parameters(variant=op, first=threshold),
            )
        return result

    def softmax(
        self, tensor, dim, half_to_float=False, *, logarithmic=False, safe=False
    ):
        self.floats(tensor)
        if half_to_float:
            raise NotImplementedError("half softmax is not supported yet")
        if dim < -tensor.ndim or dim >= tensor.ndim:
            raise IndexError("softmax dimension out of range")
        dim %= tensor.ndim
        if not tensor.numel():
            return self.empty(tensor.shape)
        with self.runtime.active():
            if safe:
                result = self.empty(tensor.shape)
                self.runtime.operators.call(
                    self, 8, result, tensor, parameters=Parameters(integer=dim)
                )
                return result
            packed = tensor.movedim(dim, -1).contiguous()
            rows = self.empty(packed.shape)
            shape = (packed.numel() // packed.shape[-1], packed.shape[-1], 1, 1)
            self.runtime.cudnn.softmax(
                shape,
                self.pointer(packed),
                self.pointer(rows),
                logarithmic=logarithmic,
            )
            return rows.movedim(-1, dim).contiguous()

    def safe_softmax(self, tensor, dim, dtype=None):
        if dtype not in (None, self.torch.float32):
            raise NotImplementedError("safe softmax currently requires float32")
        return self.softmax(tensor, dim, safe=True)

    def softmax_backward(self, grad, value, dim, input_dtype, *, logarithmic=False):
        self.floats(grad, value)
        if input_dtype != self.torch.float32 or grad.shape != value.shape:
            raise NotImplementedError(
                "softmax backward requires matching float32 tensors"
            )
        if dim < -value.ndim or dim >= value.ndim:
            raise IndexError("softmax dimension out of range")
        dim %= value.ndim
        if not value.numel():
            return self.empty(value.shape)
        with self.runtime.active():
            packed = value.movedim(dim, -1).contiguous()
            gradient = grad.movedim(dim, -1).contiguous()
            rows = self.empty(packed.shape)
            shape = (packed.numel() // packed.shape[-1], packed.shape[-1], 1, 1)
            self.runtime.cudnn.softmax(
                shape,
                self.pointer(packed),
                self.pointer(rows),
                logarithmic=logarithmic,
                grad=self.pointer(gradient),
            )
            return rows.movedim(-1, dim).contiguous()

    def bmm(self, a, b, *, out=None):
        self.floats(a, b)
        if (
            a.ndim != 3
            or b.ndim != 3
            or a.shape[0] != b.shape[0]
            or a.shape[2] != b.shape[1]
        ):
            raise ValueError("bmm requires compatible batches of matrices")
        result = self.empty((a.shape[0], a.shape[1], b.shape[2]))
        for index in range(a.shape[0]):
            self.copy_(result[index], self.mm(a[index], b[index]))
        return self.copy_(out, result) if out is not None else result

    def addmm(self, bias, a, b, *, beta=1, alpha=1, out=None):
        result = self.mm(a, b)
        if alpha != 1:
            result = self.binary("mul", result, alpha)
        if beta != 0:
            addend = bias if beta == 1 else self.binary("mul", bias, beta)
            result = self.binary("add", result, addend)
        return self.copy_(out, result) if out is not None else result

    def cat(self, tensors, dim=0, *, out=None):
        if not tensors:
            raise ValueError("cat needs at least one tensor")
        ndim = tensors[0].ndim
        if dim < -ndim or dim >= ndim:
            raise IndexError("cat dimension out of range")
        dim %= ndim
        shape = list(tensors[0].shape)
        shape[dim] = 0
        for tensor in tensors:
            if (
                tensor.dtype != tensors[0].dtype
                or tensor.ndim != ndim
                or any(
                    tensor.shape[i] != tensors[0].shape[i]
                    for i in range(ndim)
                    if i != dim
                )
            ):
                raise NotImplementedError(
                    "cat currently needs matching ranks, dtypes and non-cat sizes"
                )
            shape[dim] += tensor.shape[dim]
        result = self.empty(shape, tensors[0].dtype)
        offset = 0
        for tensor in tensors:
            self.copy_(result.narrow(dim, offset, tensor.shape[dim]), tensor)
            offset += tensor.shape[dim]
        return self.copy_(out, result) if out is not None else result

    def convolution_shape(
        self,
        input,
        weight,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        self.floats(input, weight)
        if transposed or any(output_padding) or input.ndim != 4 or weight.ndim != 4:
            raise NotImplementedError(
                "native convolution supports non-transposed 2D tensors"
            )
        if not (len(stride) == len(padding) == len(dilation) == 2):
            raise ValueError("2D convolution requires two spatial parameters")
        batch, ci, h, w = input.shape
        co, wc, kh, kw = weight.shape
        if groups <= 0 or ci % groups or co % groups or wc != ci // groups:
            raise ValueError("convolution channels do not match groups")
        if min(ci, co, kh, kw, *stride, *dilation) <= 0 or min(padding) < 0:
            raise ValueError("invalid convolution dimensions")
        oh, ow = (
            (size + 2 * pad - dil * (kernel - 1) - 1) // step + 1
            for size, pad, dil, kernel, step in zip(
                (h, w), padding, dilation, (kh, kw), stride
            )
        )
        if min(oh, ow) <= 0:
            raise ValueError("convolution kernel exceeds padded input")
        return (batch, co, oh, ow)

    def convolution(
        self,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        shape = self.convolution_shape(
            input, weight, stride, padding, dilation, transposed, output_padding, groups
        )
        if bias is not None:
            self.floats(bias)
            if tuple(bias.shape) != (shape[1],):
                raise ValueError("convolution bias must match output channels")
        result = self.empty(shape)
        if result.numel():
            with self.runtime.active():
                x, w = input.contiguous(), weight.contiguous()
                b = None if bias is None else bias.contiguous()
                self.runtime.cudnn.convolution(
                    tuple(x.shape),
                    tuple(w.shape),
                    shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    input=self.pointer(x),
                    weight=self.pointer(w),
                    output=self.pointer(result),
                    bias=None if b is None else self.pointer(b),
                )
        return result

    def convolution_backward(
        self,
        grad,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        shape = self.convolution_shape(
            input, weight, stride, padding, dilation, transposed, output_padding, groups
        )
        self.floats(grad)
        if tuple(grad.shape) != shape:
            raise ValueError("convolution gradient shape does not match output")
        dx = self.empty(input.shape) if output_mask[0] else None
        dw = self.empty(weight.shape) if output_mask[1] else None
        db = self.empty((weight.shape[0],)) if output_mask[2] else None
        if not grad.numel():
            for tensor in (dx, dw, db):
                if tensor is not None:
                    self.fill(tensor, 0)
        elif any(output_mask):
            with self.runtime.active():
                x, w, g = input.contiguous(), weight.contiguous(), grad.contiguous()
                self.runtime.cudnn.convolution(
                    tuple(x.shape),
                    tuple(w.shape),
                    shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    input=self.pointer(x),
                    weight=self.pointer(w),
                    grad=self.pointer(g),
                    dx=None if dx is None else self.pointer(dx),
                    dw=None if dw is None else self.pointer(dw),
                    db=None if db is None else self.pointer(db),
                )
        return dx, dw, db

    def layer_norm(self, tensor, normalized_shape, weight=None, bias=None, eps=1e-5):
        self.floats(tensor)
        shape = tuple(normalized_shape)
        if not shape or tuple(tensor.shape[-len(shape) :]) != shape:
            raise ValueError("normalized shape must match trailing input dimensions")
        dims = tuple(range(tensor.ndim - len(shape), tensor.ndim))
        mean = self.reduce(tensor, dims, True, mean=True)
        centered = self.binary("sub", tensor, mean)
        variance = self.reduce(
            self.binary("mul", centered, centered), dims, True, mean=True
        )
        rstd = self.unary(self.binary("add", variance, eps), 5)
        result = self.binary("mul", centered, rstd)
        if weight is not None:
            result = self.binary("mul", result, weight)
        if bias is not None:
            result = self.binary("add", result, bias)
        return result, mean, rstd

    def layer_norm_backward(
        self, grad, tensor, normalized_shape, mean, rstd, weight, bias, output_mask
    ):
        shape = tuple(normalized_shape)
        dims = tuple(range(tensor.ndim - len(shape), tensor.ndim))
        outer = tuple(range(tensor.ndim - len(shape)))
        normalized = self.binary("mul", self.binary("sub", tensor, mean), rstd)
        weighted = grad if weight is None else self.binary("mul", grad, weight)
        dx = dw = db = None
        if output_mask[0]:
            mean_grad = self.reduce(weighted, dims, True, mean=True)
            projection = self.reduce(
                self.binary("mul", weighted, normalized), dims, True, mean=True
            )
            correction = self.binary(
                "add", mean_grad, self.binary("mul", normalized, projection)
            )
            dx = self.binary("mul", self.binary("sub", weighted, correction), rstd)
        if output_mask[1] and weight is not None:
            dw = self.binary("mul", grad, normalized)
            if outer:
                dw = self.reduce(dw, outer)
        if output_mask[2] and bias is not None:
            db = self.reduce(grad, outer) if outer else grad
        return dx, dw, db

    def fallback(self, op, *args, **kwargs):
        raise NotImplementedError(
            f"Lupine native GPU kernel is not implemented for {op}"
        )

    def release(self, handle):
        with self.runtime.active():
            allocation = self.allocations.pop(handle, None)
            if allocation is not None:
                self.runtime.operators.free(allocation[0])

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
            # Only live tensor storage is counted here; operator scratch and
            # cached allocator reservations are not included.
            if name.startswith("max_"):
                raise NotImplementedError(
                    "Lupine native does not track peak memory yet"
                )
            with self.runtime.active():
                return sum(size for pointer, size in self.allocations.values())
        if name == "empty_cache":
            with self.runtime.active():
                if self.runtime._operators is not None:
                    self.runtime._operators.close()
            return None
        raise NotImplementedError(f"Lupine native does not implement torch.cuda.{name}")

    def close(self):
        if self.runtime.closed:
            return
        with self.runtime.active():
            self.runtime.check(self.runtime.sync(), "cudaDeviceSynchronize")
            for pointer, size in self.allocations.values():
                self.runtime.operators.free(pointer)
            self.allocations.clear()
            if self.runtime._cudnn is not None:
                self.runtime._cudnn.close()
                self.runtime._cudnn = None
            if self.runtime._blas is not None:
                status = self.runtime.blas_destroy(self.runtime._blas)
                if status:
                    raise RuntimeError(
                        f"cublasDestroy_v2 failed with cuBLAS status {status}"
                    )
                self.runtime._blas = None
            if self.runtime._operators is not None:
                self.runtime._operators.close()
                self.runtime._operators = None
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
        for number, name in enumerate(
            (
                "neg",
                "relu",
                "exp",
                "log",
                "sqrt",
                "rsqrt",
                "sin",
                "cos",
                "tanh",
                "sigmoid",
                "abs",
                "reciprocal",
            )
        ):
            aten.impl(name, lambda t, _op=number: self.unary(t, _op))
        aten.impl(
            "threshold_backward",
            lambda grad, value, threshold: self.activation_backward(
                grad, value, 0, threshold
            ),
        )
        aten.impl(
            "sigmoid_backward",
            lambda grad, value: self.activation_backward(grad, value, 1),
        )
        aten.impl(
            "tanh_backward",
            lambda grad, value: self.activation_backward(grad, value, 2),
        )
        aten.impl("uniform_", self.random_)
        aten.impl(
            "normal_",
            lambda t, mean=0, std=1, **kw: self.random_(
                t, mean, std, normal=True, **kw
            ),
        )
        aten.impl("_softmax", self.softmax)
        aten.impl("_safe_softmax", self.safe_softmax)
        aten.impl("_log_softmax", lambda *args: self.softmax(*args, logarithmic=True))
        aten.impl("_softmax_backward_data", self.softmax_backward)
        aten.impl(
            "_log_softmax_backward_data",
            lambda *args: self.softmax_backward(*args, logarithmic=True),
        )
        aten.impl("pow.Tensor_Scalar", lambda a, b: self.binary("pow", a, b))
        aten.impl("pow.Tensor_Tensor", lambda a, b: self.binary("pow", a, b))
        aten.impl("convolution", self.convolution)
        aten.impl("convolution_backward", self.convolution_backward)
        aten.impl("native_layer_norm", self.layer_norm)
        aten.impl("native_layer_norm_backward", self.layer_norm_backward)
        aten.impl("_fused_sdp_choice", lambda *args, **kwargs: 0)
        aten.impl("addmm", self.addmm)
        aten.impl("addmm.out", self.addmm)
        aten.impl("bmm", self.bmm)
        aten.impl("bmm.out", self.bmm)
        aten.impl("cat", self.cat)
        aten.impl("cat.out", self.cat)
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
        if _started["backend"].runtime.closed:
            raise LupineError("the native backend is closed; start a new process")
        return info()
    C = _extension()
    import torch

    backend = Backend(C)
    try:
        backend.install()
        C.register(1, backend.release, backend.synchronize)
        _started.update(
            address="native",
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
