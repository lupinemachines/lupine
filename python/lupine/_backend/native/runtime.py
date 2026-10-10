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


class LinuxInt64(ct.c_int64):
    @classmethod
    def from_param(cls, value):
        return ct.c_int64.from_param(getattr(value, "value", value))


class LinuxUInt64(ct.c_uint64):
    @classmethod
    def from_param(cls, value):
        return ct.c_uint64.from_param(getattr(value, "value", value))


def linux_argtypes(types):
    """Give spilled Linux integer arguments eight-byte AAPCS64 stack slots.

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
                raise NotImplementedError(
                    "Linux float stack arguments need a typed ABI trampoline"
                )
            floating += 1
        else:
            if general >= 8 and ct.sizeof(argument) < 8:
                if argument in signed:
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
        self.compiler = (
            "nvrtc" if hasattr(self.runtime, "nvrtcCreateProgram") else "ptx"
        )
        code = (
            self.compile_kernels()
            if self.compiler == "nvrtc"
            else Path(__file__).with_name("kernels.ptx").read_bytes()
        )
        ptx = ct.create_string_buffer(code)
        self.check(load(ct.byref(self.module), ptx), "cuModuleLoadData")
        self.get_function = self.bind(
            self.driver, "cuModuleGetFunction", ct.POINTER(P), P, ct.c_char_p
        )
        self.unload = self.bind(self.driver, "cuModuleUnload", P)
        self.blas = None
        if hasattr(self.runtime, "cublasCreate_v2"):
            self.blas = P()
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
            status = create(ct.byref(self.blas))
            if status:
                self.unload(self.module)
                raise RuntimeError(
                    f"cublasCreate_v2 failed with cuBLAS status {status}"
                )

    def matmul(self, output, a, b, m, n, k):
        """Row-major C=A@B via column-major C^T=B^T@A^T."""
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
        self.check(self.sync(), "cudaDeviceSynchronize(cuBLAS)")
        self.launches += 1

    def compile_kernels(self):
        """Compile the shipped CUDA source locally with the converted NVRTC."""
        create = self.bind(
            self.runtime,
            "nvrtcCreateProgram",
            ct.POINTER(P),
            ct.c_char_p,
            ct.c_char_p,
            I,
            ct.POINTER(ct.c_char_p),
            ct.POINTER(ct.c_char_p),
        )
        compile_ = self.bind(
            self.runtime, "nvrtcCompileProgram", P, I, ct.POINTER(ct.c_char_p)
        )
        destroy = self.bind(self.runtime, "nvrtcDestroyProgram", ct.POINTER(P))
        program = P()
        # NVRTC provides CUDA/math builtins, but has no host SDK on macOS.
        # Supply the two fixed-width integer definitions used by our kernels.
        headers = (ct.c_char_p * 1)(
            b"typedef signed long long int64_t; typedef unsigned long long uint64_t;"
        )
        names = (ct.c_char_p * 1)(b"stdint.h")
        status = create(
            ct.byref(program),
            Path(__file__).with_name("kernels.cu").read_bytes(),
            b"kernels.cu",
            1,
            headers,
            names,
        )
        if status:
            raise RuntimeError(f"nvrtcCreateProgram failed with NVRTC status {status}")
        try:
            options = (ct.c_char_p * 3)(
                b"--gpu-architecture=compute_75", b"--std=c++17", b"--fmad=false"
            )
            status = compile_(program, len(options), options)

            def output(size_name, data_name):
                size = S()
                get_size = self.bind(self.runtime, size_name, P, ct.POINTER(S))
                get_data = self.bind(self.runtime, data_name, P, P)
                result = get_size(program, ct.byref(size))
                if result:
                    raise RuntimeError(f"{size_name} failed with NVRTC status {result}")
                data = ct.create_string_buffer(size.value)
                result = get_data(program, data)
                if result:
                    raise RuntimeError(f"{data_name} failed with NVRTC status {result}")
                return data.value

            if status:
                log = output("nvrtcGetProgramLogSize", "nvrtcGetProgramLog").decode(
                    errors="replace"
                )
                raise RuntimeError(
                    f"NVRTC kernel compilation failed ({status}):\n{log}"
                )
            return output("nvrtcGetPTXSize", "nvrtcGetPTX")
        finally:
            destroy(ct.byref(program))

    def bind(self, library, name, *types):
        function = getattr(library, name)
        function.argtypes = linux_argtypes(types) if library is self.runtime else types
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
