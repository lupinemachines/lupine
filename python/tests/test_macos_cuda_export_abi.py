"""GPU checks for exported APIs with packed Darwin scalar stack arguments."""
from contextlib import ExitStack
import ctypes as ct
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("LUPINE_TEST_MACOS_CUDA_EXPORT_ABI") != "1",
    reason="requires native export bridges and a remote GPU",
)


def run_probe():
    from lupine._backend.native.runtime import Runtime, P, I, F

    runtime = Runtime()
    gemm = runtime.bind(runtime.runtime, "cublasGemmEx",
                        P, I, I, I, I, I, P, P, I, I, P, I, I, P, P, I, I, I, I)
    entry = runtime.bind(runtime.runtime, "cudaGetDriverEntryPointByVersion", ct.c_char_p,
                         ct.POINTER(P), ct.c_uint, ct.c_ulonglong, ct.POINTER(I))
    with runtime.active(), ExitStack() as resources:
        resources.callback(runtime.unload, runtime.module)
        assert runtime.blas is not None
        resources.callback(runtime.blas_destroy, runtime.blas)

        def upload(host):
            pointer = P(runtime.malloc(ct.sizeof(host)))
            resources.callback(runtime.free, pointer)
            runtime.check(runtime.copy(pointer, host, ct.sizeof(host), 1), "upload")
            return pointer

        # Column-major 2x3 and 3x2 matrices, encoded as IEEE binary16.
        a = upload((ct.c_uint16 * 6).from_buffer_copy(struct.pack("<6e", 1, 4, 2, 5, 3, 6)))
        b = upload((ct.c_uint16 * 6).from_buffer_copy(struct.pack("<6e", 7, 9, 11, 8, 10, 12)))
        result = (F * 4)()
        c = upload(result)
        alpha, beta = F(1), F(0)
        runtime.check(gemm(runtime.blas, 0, 0, 2, 2, 3, ct.byref(alpha), a, 2, 2,
                           b, 2, 3, ct.byref(beta), c, 0, 2, 68, -1), "cublasGemmEx")
        runtime.check(runtime.sync(), "GEMM synchronize")
        runtime.check(runtime.copy(result, c, ct.sizeof(result), 2), "GEMM download")
        assert list(result) == [58, 139, 64, 154], list(result)
        print("Native export FP16 GEMM passed", flush=True)

        pointer, status = P(), I(-1)
        runtime.check(entry(b"cuLaunchKernel", ct.byref(pointer), 13000, 0, ct.byref(status)),
                      "cudaGetDriverEntryPointByVersion")
        assert status.value == 0
        assert pointer.value == ct.cast(runtime.driver.cuLaunchKernel, P).value
        print("CUDART returned native driver pointer passed", flush=True)


def test_half_gemm_native_pointer_and_teardown():
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native export FP16 GEMM passed" in result.stdout
    assert "CUDART returned native driver pointer passed" in result.stdout


if __name__ == "__main__":
    run_probe()
