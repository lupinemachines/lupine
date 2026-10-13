"""Local dylib calls; no torch worker or operator RPC."""

from __future__ import annotations

import ctypes as ct
from contextlib import contextmanager
import struct
import threading

from ... import _native

I = ct.c_int
U = ct.c_uint
Q = ct.c_int64
P = ct.c_void_p
S = ct.c_size_t
F = ct.c_float


class LinuxInt64(ct.c_int64):
    @classmethod
    def from_param(cls, value):
        return ct.c_int64.from_param(getattr(value, "value", value))


class LinuxUInt64(ct.c_uint64):
    @classmethod
    def from_param(cls, value):
        return ct.c_uint64.from_param(getattr(value, "value", value))


class LinuxFloatSlot(ct.c_double):
    @classmethod
    def from_param(cls, value):
        # Keep the float in the FP argument bank, but reserve an eight-byte
        # spill slot. A double carries the float's bits, not its numeric value.
        value = ct.c_float(getattr(value, "value", value)).value
        bits = struct.pack("<f", value) + b"\0" * 4
        return ct.c_double.from_param(struct.unpack("<d", bits)[0])


class LinuxBoolSlot(ct.c_uint64):
    @classmethod
    def from_param(cls, value):
        return ct.c_uint64.from_param(bool(getattr(value, "value", value)))


def linux_argtypes(types):
    """Give spilled Linux scalar arguments eight-byte AAPCS64 stack slots.

    Apple ARM64 packs narrow stack arguments. Register arguments and the
    separate floating-point register bank retain their declared types.
    """
    general, floating = 0, 0
    result = []
    signed = (ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong)
    unsigned = (
        ct.c_ubyte,
        ct.c_ushort,
        ct.c_uint,
        ct.c_ulong,
        ct.c_ulonglong,
        ct.c_bool,
    )
    for argument in types:
        if argument in (ct.c_float, ct.c_double):
            if floating >= 8 and argument == ct.c_float:
                argument = LinuxFloatSlot
            floating += 1
        else:
            if general >= 8 and ct.sizeof(argument) < 8:
                if argument is ct.c_bool:
                    argument = LinuxBoolSlot
                elif argument in signed:
                    argument = LinuxInt64
                elif argument in unsigned:
                    argument = LinuxUInt64
                else:
                    raise NotImplementedError("unsupported Linux stack argument type")
            general += 1
        result.append(argument)
    return tuple(result)


class Runtime:
    def __init__(self):
        loaded = _native.load(missing_ok=False)
        if "libcudart.dylib" not in loaded:
            raise RuntimeError(
                "native torch requires the ARM64 macOS wheel with packaged runtime dylibs"
            )
        self.runtime = ct.CDLL(loaded["libcudart.dylib"], mode=ct.RTLD_LOCAL)
        self.export_abi = getattr(self.runtime, "machso_native_export_abi", None)
        if self.export_abi is not None:
            self.export_abi.argtypes = [ct.c_char_p]
            self.export_abi.restype = I
        self.driver = ct.CDLL(loaded["libcuda.dylib"])
        self.lock = threading.RLock()
        self.launches = 0
        self.closed = False
        self._cudnn = None
        self._operators = None
        self.get_count = self.bind(self.runtime, "cudaGetDeviceCount", ct.POINTER(I))
        self.allocate = self.bind(self.runtime, "cudaMalloc", ct.POINTER(P), S)
        self.free = self.bind(self.runtime, "cudaFree", P)
        self.copy = self.bind(self.runtime, "cudaMemcpy", P, P, S, I)
        self.sync = self.bind(self.runtime, "cudaDeviceSynchronize")
        self.attribute = self.bind(
            self.runtime, "cudaDeviceGetAttribute", ct.POINTER(I), I, I
        )
        self.mem_info = self.bind(
            self.runtime, "cudaMemGetInfo", ct.POINTER(S), ct.POINTER(S)
        )
        self.set_context = self.bind(self.driver, "cuCtxSetCurrent", P)
        count = I()
        self.check(self.get_count(ct.byref(count)), "cudaGetDeviceCount")
        # One default-stream context is intentional for this first backend.
        if count.value != 1:
            raise RuntimeError(
                f"native torch currently requires exactly one GPU, found {count.value}"
            )
        self.check(self.free(None), "cudaFree(0)")
        self.context = P()
        get_context = self.bind(self.driver, "cuCtxGetCurrent", ct.POINTER(P))
        self.check(get_context(ct.byref(self.context)), "cuCtxGetCurrent")
        self._blas = None

    @property
    def blas(self):
        if self.closed:
            return None
        if self._blas is None and hasattr(self.runtime, "cublasCreate_v2"):
            handle = P()
            create = self.bind(self.runtime, "cublasCreate_v2", ct.POINTER(P))
            self.blas_destroy = self.bind(self.runtime, "cublasDestroy_v2", P)
            self.blas_gemm = self.bind(
                self.runtime,
                "cublasSgemm_v2",
                P,
                I,
                I,
                I,
                I,
                I,
                P,
                P,
                I,
                P,
                I,
                P,
                P,
                I,
            )
            status = create(ct.byref(handle))
            if status:
                raise RuntimeError(
                    f"cublasCreate_v2 failed with cuBLAS status {status}"
                )
            self._blas = handle
        return self._blas

    def matmul(self, output, a, b, m, n, k):
        """Row-major C=A@B via column-major C^T=B^T@A^T."""
        if self.blas is None:
            raise NotImplementedError("native matmul requires a bundle with cuBLAS")
        alpha, beta = F(1), F(0)
        status = self.blas_gemm(
            self.blas,
            0,
            0,
            n,
            m,
            k,
            ct.byref(alpha),
            b,
            n,
            a,
            k,
            ct.byref(beta),
            output,
            n,
        )
        if status:
            raise RuntimeError(f"cublasSgemm_v2 failed with cuBLAS status {status}")
        self.launches += 1

    @property
    def cudnn(self):
        if self._cudnn is None:
            if not hasattr(self.runtime, "cudnnCreate"):
                raise NotImplementedError(
                    "native tensor operations require a bundle with cuDNN"
                )
            from .cudnn import Cudnn

            self._cudnn = Cudnn(self)
        return self._cudnn

    @property
    def operators(self):
        if self._operators is None:
            loaded = _native.load(missing_ok=False)
            path = loaded.get("liblupine_aten.dylib")
            if path is None:
                raise NotImplementedError(
                    "native tensor operations require packaged CUDA operators"
                )
            from .aten import Operators

            self._operators = Operators(self, path)
        return self._operators

    def bind(self, library, name, *types):
        function = getattr(library, name)
        native = library is not self.runtime or (
            self.export_abi is not None and self.export_abi(name.encode())
        )
        function.argtypes = types if native else linux_argtypes(types)
        function.restype = I
        return function

    @staticmethod
    def check(status, name):
        if status:
            raise RuntimeError(f"Lupine native {name} failed with CUDA status {status}")

    @contextmanager
    def active(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Lupine native runtime is closed")
            self.check(self.set_context(self.context), "cuCtxSetCurrent")
            yield

    def malloc(self, size):
        result = P()
        self.check(self.allocate(ct.byref(result), max(size, 1)), "cudaMalloc")
        return result.value

    @contextmanager
    def allocated(self, size):
        operators = self._operators
        pointer = P(operators.malloc(size) if operators is not None else self.malloc(size))
        try:
            yield pointer
        finally:
            if operators is not None:
                operators.free(pointer)
            else:
                self.check(self.free(pointer), "cudaFree(temporary)")

    def synchronize(self):
        with self.active():
            self.check(self.sync(), "cudaDeviceSynchronize")
