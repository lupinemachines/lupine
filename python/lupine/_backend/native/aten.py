"""Host bindings to the converted, packaged PyTorch CUDA operators."""

import ctypes as ct

from .runtime import I, Q, P


class TensorDesc(ct.Structure):
    _fields_ = [
        ("data", P),
        ("dims", Q),
        ("dtype", Q),
        ("size", Q * 8),
        ("stride", Q * 8),
    ]


class Parameters(ct.Structure):
    _fields_ = [
        ("variant", Q),
        ("axis_count", Q),
        ("axes", Q * 8),
        ("integer", Q),
        ("step", Q),
        ("seed", ct.c_uint64),
        ("offset", ct.c_uint64),
        ("first", ct.c_double),
        ("second", ct.c_double),
        ("alpha", ct.c_double),
    ]


class Operators:
    def __init__(self, runtime, path):
        self.runtime = runtime
        self.library = ct.CDLL(path, mode=ct.RTLD_LOCAL)
        self.run = self.library.lupine_aten_run
        self.run.argtypes = [I, P, P, P, P]
        self.run.restype = I
        self.error = self.library.lupine_aten_error
        self.error.argtypes, self.error.restype = [], ct.c_char_p
        self.destroy = self.library.lupine_aten_close
        self.destroy.argtypes, self.destroy.restype = [], I

    def check(self, status):
        if status:
            raise RuntimeError(f"packaged CUDA operator: {self.error().decode()}")

    def descriptor(self, backend, tensor, pointer=None):
        if tensor.ndim > 8:
            raise NotImplementedError("Lupine native supports at most eight dimensions")
        return TensorDesc(
            backend.pointer(tensor) if pointer is None else pointer,
            tensor.ndim,
            backend.dtype(tensor),
            (Q * 8)(*tensor.shape),
            (Q * 8)(*tensor.stride()),
        )

    def call(
        self,
        backend,
        operation,
        out,
        a=None,
        b=None,
        *,
        parameters=None,
        source_pointer=None,
    ):
        if not out.numel():
            return parameters
        destination = self.descriptor(backend, out)
        source = None if a is None else self.descriptor(backend, a, source_pointer)
        other = None if b is None else self.descriptor(backend, b)
        parameters = parameters or Parameters()
        self.check(
            self.run(
                operation,
                ct.byref(destination),
                None if source is None else ct.byref(source),
                None if other is None else ct.byref(other),
                ct.byref(parameters),
            )
        )
        self.runtime.check(self.runtime.sync(), "cudaDeviceSynchronize(ATen)")
        self.runtime.launches += 1
        return parameters

    def close(self):
        self.check(self.destroy())
