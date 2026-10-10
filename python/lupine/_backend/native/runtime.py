"""Local dylib calls and GPU kernels; no torch worker or operator RPC."""

from __future__ import annotations

import ctypes as ct
from contextlib import contextmanager
from pathlib import Path
import threading

from ... import _native

I = ct.c_int
U = ct.c_uint
Q = ct.c_int64
P = ct.c_void_p
S = ct.c_size_t
F = ct.c_float


class Runtime:
    def __init__(self):
        loaded = _native.load(missing_ok=False)
        if "libcudart.dylib" not in loaded:
            raise RuntimeError(
                "native torch requires an ARM64 macOS bundle with libcudart.dylib"
            )
        self.runtime = ct.CDLL(loaded["libcudart.dylib"], mode=ct.RTLD_LOCAL)
        self.driver = ct.CDLL(loaded["libcuda.dylib"])
        self.lock = threading.RLock()
        self.launches = 0
        self.functions = {}
        self.closed = False
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
        self.launch = self.bind(
            self.driver,
            "cuLaunchKernel",
            P,
            U,
            U,
            U,
            U,
            U,
            U,
            U,
            P,
            ct.POINTER(P),
            ct.POINTER(P),
        )
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
        self.module = P()
        load = self.bind(self.driver, "cuModuleLoadData", ct.POINTER(P), P)
        ptx = ct.create_string_buffer(
            Path(__file__).with_name("kernels.ptx").read_bytes()
        )
        self.check(load(ct.byref(self.module), ptx), "cuModuleLoadData")
        self.get_function = self.bind(
            self.driver, "cuModuleGetFunction", ct.POINTER(P), P, ct.c_char_p
        )
        self.unload = self.bind(self.driver, "cuModuleUnload", P)

    @staticmethod
    def bind(library, name, *types):
        function = getattr(library, name)
        function.argtypes = types
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
    def uploaded(self, values):
        data = (Q * len(values))(*values)
        pointer = self.malloc(ct.sizeof(data))
        try:
            self.check(
                self.copy(pointer, data, ct.sizeof(data), 1),
                "cudaMemcpy(HtoD metadata)",
            )
            yield P(pointer)
        finally:
            self.check(self.free(pointer), "cudaFree(metadata)")

    def kernel(self, name, count, *args):
        if count == 0:
            return
        if name not in self.functions:
            function = P()
            self.check(
                self.get_function(ct.byref(function), self.module, name.encode()),
                "cuModuleGetFunction",
            )
            self.functions[name] = function
        parameters = (P * len(args))(*(ct.addressof(arg) for arg in args))
        self.check(
            self.launch(
                self.functions[name],
                (count + 127) // 128,
                1,
                1,
                128,
                1,
                1,
                0,
                None,
                parameters,
                None,
            ),
            "cuLaunchKernel",
        )
        self.launches += 1
        # Synchronous initial backend: descriptors and storage can be freed
        # immediately, and host guard events really are complete on return.
        self.check(self.sync(), "cudaDeviceSynchronize")

    def synchronize(self):
        with self.active():
            self.check(self.sync(), "cudaDeviceSynchronize")
