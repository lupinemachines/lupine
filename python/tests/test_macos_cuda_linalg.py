"""Real GPU checks for a converted cuSOLVER/cuSPARSE dependency closure."""

from contextlib import ExitStack
import ctypes as ct
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("LUPINE_TEST_MACOS_CUDA_LINALG") != "1",
    reason="requires converted cuSOLVER/cuSPARSE on an ARM64 Mac and a GPU server",
)


def run_probe():
    from lupine._backend.native.runtime import Runtime, F, I, P, Q, S

    r = Runtime()
    def bind(name, *args):
        return r.bind(r.runtime, name, *args)

    solver_create = bind("cusolverDnCreate", ct.POINTER(P))
    solver_destroy = bind("cusolverDnDestroy", P)
    potrf_size = bind("cusolverDnSpotrf_bufferSize", P, I, I, P, I, ct.POINTER(I))
    potrf = bind("cusolverDnSpotrf", P, I, I, P, I, P, I, P)
    sparse_create = bind("cusparseCreate", ct.POINTER(P))
    sparse_destroy = bind("cusparseDestroy", P)
    csr_create = bind("cusparseCreateCsr", ct.POINTER(P), Q, Q, Q, P, P, P, I, I, I, I)
    csr_destroy = bind("cusparseDestroySpMat", P)
    vector_create = bind("cusparseCreateDnVec", ct.POINTER(P), Q, P, I)
    vector_destroy = bind("cusparseDestroyDnVec", P)
    spmv_size = bind(
        "cusparseSpMV_bufferSize", P, I, P, P, P, P, P, I, I, ct.POINTER(S)
    )
    spmv = bind("cusparseSpMV", P, I, P, P, P, P, P, I, I, P)

    with r.active(), ExitStack() as resources:
        if r.blas is not None:
            resources.callback(r.blas_destroy, r.blas)

        def allocate(size):
            pointer = P(r.malloc(max(1, size)))
            resources.callback(r.free, pointer)
            return pointer

        def upload(host):
            pointer = allocate(ct.sizeof(host))
            r.check(r.copy(pointer, host, ct.sizeof(host), 1), "upload")
            return pointer

        solver = P()
        r.check(solver_create(ct.byref(solver)), "cusolverDnCreate")
        resources.callback(solver_destroy, solver)
        # Column-major SPD matrix; its lower Cholesky factor is known exactly.
        matrix = (F * 9)(4, 12, -16, 12, 37, -43, -16, -43, 98)
        data = upload(matrix)
        info = allocate(ct.sizeof(I))
        size = I()
        r.check(potrf_size(solver, 0, 3, data, 3, ct.byref(size)), "potrf workspace")
        workspace = allocate(size.value * ct.sizeof(F))
        r.check(potrf(solver, 0, 3, data, 3, workspace, size, info), "cusolverDnSpotrf")
        r.check(r.sync(), "Cholesky synchronize")
        status = I()
        r.check(r.copy(ct.byref(status), info, ct.sizeof(status), 2), "Cholesky info")
        assert status.value == 0, status.value
        r.check(r.copy(matrix, data, ct.sizeof(matrix), 2), "Cholesky download")
        for index, expected in ((0, 2), (1, 6), (2, -8), (4, 1), (5, 5), (8, 3)):
            assert abs(matrix[index] - expected) < 1e-5, (index, matrix[index])
        print("Converted cuSOLVER GPU Cholesky passed", flush=True)

        sparse = P()
        r.check(sparse_create(ct.byref(sparse)), "cusparseCreate")
        resources.callback(sparse_destroy, sparse)
        rows = upload((I * 4)(0, 2, 3, 5))
        columns = upload((I * 5)(0, 2, 1, 0, 2))
        values = upload((F * 5)(1, 2, 3, 4, 5))
        x = upload((F * 3)(2, 3, 4))
        y = upload((F * 3)(0, 0, 0))
        matrix_descriptor, x_descriptor, y_descriptor = P(), P(), P()
        r.check(
            csr_create(
                ct.byref(matrix_descriptor), 3, 3, 5, rows, columns, values, 2, 2, 0, 0
            ),
            "cusparseCreateCsr",
        )
        resources.callback(csr_destroy, matrix_descriptor)
        for descriptor, pointer in ((x_descriptor, x), (y_descriptor, y)):
            r.check(
                vector_create(ct.byref(descriptor), 3, pointer, 0),
                "cusparseCreateDnVec",
            )
            resources.callback(vector_destroy, descriptor)
        alpha, beta, size = F(1), F(0), S()
        args = (
            sparse,
            0,
            ct.byref(alpha),
            matrix_descriptor,
            x_descriptor,
            ct.byref(beta),
            y_descriptor,
            0,
            0,
        )
        r.check(spmv_size(*args, ct.byref(size)), "SpMV workspace")
        workspace = allocate(size.value)
        r.check(spmv(*args, workspace), "cusparseSpMV")
        r.check(r.sync(), "SpMV synchronize")
        result = (F * 3)()
        r.check(r.copy(result, y, ct.sizeof(result), 2), "SpMV download")
        assert list(result) == [10, 9, 28], list(result)
        print("Converted cuSPARSE GPU SpMV passed", flush=True)


def test_cholesky_sparse_matvec_and_process_teardown():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Converted cuSOLVER GPU Cholesky passed" in result.stdout
    assert "Converted cuSPARSE GPU SpMV passed" in result.stdout


if __name__ == "__main__":
    run_probe()
