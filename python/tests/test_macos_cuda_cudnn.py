"""Execute converted cuDNN on a remote GPU and require clean process teardown.

Opt in with LUPINE_TEST_MACOS_CUDA_CUDNN=1 and a bundle prepared with the
nvidia-cudnn-cu13 package, including every library from its wheel.
"""

from contextlib import ExitStack
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or os.environ.get("LUPINE_TEST_MACOS_CUDA_CUDNN") != "1",
    reason="requires converted cuDNN on an ARM64 Mac and a remote GPU",
)


def run_probe():
    from lupine._backend.native.runtime import Runtime, I, P, F, S

    runtime = Runtime()

    def bind(name, *arguments):
        return runtime.bind(runtime.runtime, name, *arguments)

    create = bind("cudnnCreate", ct.POINTER(P))
    destroy = bind("cudnnDestroy", P)
    tensor_create = bind("cudnnCreateTensorDescriptor", ct.POINTER(P))
    tensor_destroy = bind("cudnnDestroyTensorDescriptor", P)
    tensor_set = bind("cudnnSetTensor4dDescriptor", P, I, I, I, I, I, I)
    filter_create = bind("cudnnCreateFilterDescriptor", ct.POINTER(P))
    filter_destroy = bind("cudnnDestroyFilterDescriptor", P)
    filter_set = bind("cudnnSetFilter4dDescriptor", P, I, I, I, I, I, I)
    conv_create = bind("cudnnCreateConvolutionDescriptor", ct.POINTER(P))
    conv_destroy = bind("cudnnDestroyConvolutionDescriptor", P)
    conv_set = bind("cudnnSetConvolution2dDescriptor", P, I, I, I, I, I, I, I, I)
    error = runtime.runtime.cudnnGetErrorString
    error.argtypes, error.restype = [I], ct.c_char_p

    def check(status, operation):
        if status:
            raise RuntimeError(f"{operation}: cuDNN {status}: {error(status).decode()}")

    with runtime.active(), ExitStack() as cleanup:
        cleanup.callback(runtime.unload, runtime.module)
        if runtime.blas is not None:
            cleanup.callback(runtime.blas_destroy, runtime.blas)
        handle = P()
        check(create(ct.byref(handle)), "create")
        cleanup.callback(destroy, handle)

        def descriptor(create, destroy):
            value = P()
            check(create(ct.byref(value)), "descriptor")
            cleanup.callback(destroy, value)
            return value

        xdesc = descriptor(tensor_create, tensor_destroy)
        ydesc = descriptor(tensor_create, tensor_destroy)
        wdesc = descriptor(filter_create, filter_destroy)
        cdesc = descriptor(conv_create, conv_destroy)
        check(tensor_set(xdesc, 0, 0, 1, 1, 3, 3), "input descriptor")
        check(tensor_set(ydesc, 0, 0, 1, 1, 2, 2), "output descriptor")
        check(filter_set(wdesc, 0, 0, 1, 1, 2, 2), "filter descriptor")
        check(conv_set(cdesc, 0, 0, 1, 1, 1, 1, 1, 0), "convolution descriptor")

        def allocate(size):
            value = P(runtime.malloc(max(1, size)))
            cleanup.callback(runtime.free, value)
            return value

        def upload(values):
            host = (F * len(values))(*values)
            value = allocate(ct.sizeof(host))
            runtime.check(runtime.copy(value, host, ct.sizeof(host), 1), "upload")
            return value

        x, w, dy = upload(range(1, 10)), upload([1, 2, 3, 4]), upload([1, 2, 3, 4])
        cases = (
            ("Forward", xdesc, x, wdesc, w, ydesc, [37, 47, 67, 77]),
            ("BackwardData", wdesc, w, ydesc, dy, xdesc, [1, 4, 4, 6, 20, 16, 9, 24, 16]),
            ("BackwardFilter", xdesc, x, ydesc, dy, wdesc, [37, 47, 67, 77]),
        )
        for name, adesc, a, bdesc, b, outdesc, expected in cases:
            workspace_size = bind(
                f"cudnnGetConvolution{name}WorkspaceSize", P, P, P, P, P, I, ct.POINTER(S)
            )
            execute = bind(f"cudnnConvolution{name}", P, P, P, P, P, P, P, I, P, S, P, P, P)
            size = S()
            check(workspace_size(handle, adesc, bdesc, cdesc, outdesc, 0, ct.byref(size)), "workspace")
            workspace, output = allocate(size.value), allocate(4 * len(expected))
            alpha, beta = F(1), F(0)
            check(execute(handle, ct.byref(alpha), adesc, a, bdesc, b, cdesc, 0,
                          workspace, size, ct.byref(beta), outdesc, output), name)
            runtime.check(runtime.sync(), "synchronize")
            result = (F * len(expected))()
            runtime.check(runtime.copy(result, output, ct.sizeof(result), 2), "download")
            assert list(result) == expected, (name, list(result), expected)
            print("Converted cuDNN", name, "GPU result:", list(result), flush=True)


def test_convolution_gradients_and_process_teardown():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for operation in ("Forward", "BackwardData", "BackwardFilter"):
        assert f"Converted cuDNN {operation} GPU result:" in result.stdout


if __name__ == "__main__":
    run_probe()
