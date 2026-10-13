"""Converted CUDA library correctness and late driver calls during process exit.

Opt in on ARM64 macOS with LUPINE_TEST_MACOS_CUDA_LIBRARIES=1 and a bundle
prepared with the cuFFT, cuRAND, and nvJitLink packages.
"""

import ctypes as ct
import cmath
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or os.environ.get("LUPINE_TEST_MACOS_CUDA_LIBRARIES") != "1",
    reason="requires converted cuFFT/cuRAND libraries on an ARM64 Mac and a GPU server",
)


def run_probe():
    from lupine._backend.native.runtime import Runtime, P, I, S

    r = Runtime()
    version = I()
    get_version = r.bind(r.runtime, "cufftGetVersion", ct.POINTER(I))
    plan_1d = r.bind(r.runtime, "cufftPlan1d", ct.POINTER(I), I, I, I)
    execute = r.bind(r.runtime, "cufftExecC2C", I, P, P, I)
    destroy_plan = r.bind(r.runtime, "cufftDestroy", I)
    create_rng = r.bind(r.runtime, "curandCreateGenerator", ct.POINTER(P), I)
    seed_rng = r.bind(
        r.runtime, "curandSetPseudoRandomGeneratorSeed", P, ct.c_ulonglong
    )
    generate = r.bind(r.runtime, "curandGenerateUniform", P, P, S)
    destroy_rng = r.bind(r.runtime, "curandDestroyGenerator", P)

    with r.active():
        r.check(get_version(ct.byref(version)), "cufftGetVersion")
        print("cuFFT version:", version.value, flush=True)
        for n in (8, 13, 257, 16384):
            plan = I()
            status = plan_1d(ct.byref(plan), n, 0x29, 1)
            print("cufftPlan1d size/status:", n, status, flush=True)
            r.check(status, "cufftPlan1d")
            values = (
                [float(i % 7) - 3 for i in range(n)]
                if n < 1024
                else [1.0] + [0.0] * (n - 1)
            )
            source = (ct.c_float * (2 * n))(
                *[value for x in values for value in (x, 0.0)]
            )
            result = (ct.c_float * (2 * n))()
            data = P(r.malloc(ct.sizeof(source)))
            try:
                r.check(r.copy(data, source, ct.sizeof(source), 1), "FFT upload")
                r.check(execute(plan, data, data, -1), "cufftExecC2C")
                r.check(r.sync(), "FFT synchronize")
                r.check(r.copy(result, data, ct.sizeof(result), 2), "FFT download")
                expected = (
                    [
                        sum(
                            x * cmath.exp(-2j * math.pi * i * k / n)
                            for i, x in enumerate(values)
                        )
                        for k in range(n)
                    ]
                    if n < 1024
                    else [1 + 0j] * n
                )
                actual = [complex(result[2 * k], result[2 * k + 1]) for k in range(n)]
                error = max(abs(a - b) for a, b in zip(actual, expected))
                assert error < 2e-3, (n, error)
                r.check(execute(plan, data, data, 1), "inverse cufftExecC2C")
                r.check(r.sync(), "inverse FFT synchronize")
                r.check(
                    r.copy(result, data, ct.sizeof(result), 2), "inverse FFT download"
                )
                error = max(
                    abs(complex(result[2 * i], result[2 * i + 1]) - n * x)
                    for i, x in enumerate(values)
                )
                assert error < max(2e-3, n * 2e-5), (n, error)
                print("Converted cuFFT GPU forward/inverse size:", n, flush=True)
            finally:
                r.free(data)
                destroy_plan(plan)
        generator = P()
        r.check(create_rng(ct.byref(generator), 100), "curandCreateGenerator")
        data = P(r.malloc(1024 * 4))
        result = (ct.c_float * 1024)()
        try:
            r.check(seed_rng(generator, 42), "curandSetSeed")
            r.check(generate(generator, data, 1024), "curandGenerateUniform")
            r.check(r.sync(), "RNG synchronize")
            r.check(r.copy(result, data, ct.sizeof(result), 2), "RNG download")
            assert all(0 < x <= 1 for x in result)
            assert abs(sum(result) / len(result) - 0.5) < 0.05
            print(
                "Converted cuRAND GPU uniform mean:",
                sum(result) / len(result),
                flush=True,
            )
        finally:
            r.free(data)
            destroy_rng(generator)
            if r.blas is not None:
                r.blas_destroy(r.blas)


def test_fft_rng_and_process_teardown():
    # A successful calculation alone misses static destructor ordering bugs.
    # Run the real libraries in a child and require clean interpreter teardown.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for size in (8, 13, 257, 16384):
        assert f"Converted cuFFT GPU forward/inverse size: {size}" in result.stdout
    assert "Converted cuRAND GPU uniform mean:" in result.stdout


if __name__ == "__main__":
    run_probe()
