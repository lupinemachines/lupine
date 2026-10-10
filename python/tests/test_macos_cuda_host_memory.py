"""Mapped host memory through native Darwin and converted Linux CUDA APIs."""

import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or os.environ.get("LUPINE_TEST_MACOS_CUDA_HOST_MEMORY") != "1",
    reason="requires an ARM64 Mac, converted runtime, and remote GPU",
)


def run_probe():
    from lupine._backend.native.runtime import Runtime, P, S, U

    runtime = Runtime()
    with runtime.active():
        try:
            for library, allocate_name, free_name, pointer_name in (
                (runtime.driver, "cuMemHostAlloc", "cuMemFreeHost", "cuMemHostGetDevicePointer_v2"),
                (runtime.runtime, "cudaHostAlloc", "cudaFreeHost", "cudaHostGetDevicePointer"),
            ):
                allocate = runtime.bind(library, allocate_name, ct.POINTER(P), S, U)
                free = runtime.bind(library, free_name, P)
                device_pointer = runtime.bind(library, pointer_name, ct.POINTER(P), P, U)
                for size in (4, 32772):
                    host, device = P(), P()
                    runtime.check(allocate(ct.byref(host), size, 2), allocate_name)
                    try:
                        runtime.check(device_pointer(ct.byref(device), host, 0), pointer_name)
                        values = (ct.c_ubyte * size).from_address(host.value)
                        values[0], values[-1] = 17, 29
                        downloaded = (ct.c_ubyte * size)()
                        runtime.check(runtime.copy(downloaded, device, size, 2), "mapped download")
                        assert (downloaded[0], downloaded[-1]) == (17, 29)
                        source = (ct.c_ubyte * size)()
                        source[0], source[-1] = 41, 53
                        runtime.check(runtime.copy(device, source, size, 1), "mapped upload")
                        runtime.check(runtime.sync(), "mapped synchronize")
                        assert (values[0], values[-1]) == (41, 53)
                        print(allocate_name, size, "mapped round trip passed", flush=True)
                    finally:
                        runtime.check(free(host), free_name)
        finally:
            if runtime.blas is not None:
                runtime.blas_destroy(runtime.blas)
            runtime.unload(runtime.module)


def test_mapped_host_memory_and_teardown():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("mapped round trip passed") == 4


if __name__ == "__main__":
    run_probe()
