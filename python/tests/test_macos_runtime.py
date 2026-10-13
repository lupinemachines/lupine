"""GPU integration checks for the converted runtime in an ARM64 macOS bundle."""
import ctypes
import os
import sys

import pytest

from lupine import _native

pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin' or os.environ.get('LUPINE_TEST_MACOS_RUNTIME') != '1',
    reason='requires an ARM64 macOS client bundle and a configured GPU server',
)


@pytest.fixture(scope='module')
def runtime():
    loaded = _native.load(missing_ok=False)
    return ctypes.CDLL(loaded['libcudart.dylib'], mode=ctypes.RTLD_LOCAL)


def function(runtime, name, *arguments):
    result = getattr(runtime, name)
    result.argtypes = arguments
    result.restype = ctypes.c_int
    return result


def test_device_discovery(runtime):
    count = ctypes.c_int()
    get_count = function(runtime, 'cudaGetDeviceCount', ctypes.POINTER(ctypes.c_int))
    assert get_count(ctypes.byref(count)) == 0
    assert count.value > 0
    get_attribute = function(runtime, 'cudaDeviceGetAttribute',
                             ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int)
    major, multiprocessors = ctypes.c_int(), ctypes.c_int()
    assert get_attribute(ctypes.byref(major), 75, 0) == 0
    assert get_attribute(ctypes.byref(multiprocessors), 16, 0) == 0
    assert major.value > 0 and multiprocessors.value > 0


def test_device_memory_round_trip(runtime):
    allocate = function(runtime, 'cudaMalloc', ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
    copy = function(runtime, 'cudaMemcpy', ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_size_t, ctypes.c_int)
    memset = function(runtime, 'cudaMemset', ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
    free = function(runtime, 'cudaFree', ctypes.c_void_p)
    source = (ctypes.c_ubyte * 256)(*(i ^ 0xa5 for i in range(256)))
    result = (ctypes.c_ubyte * 256)()
    device = ctypes.c_void_p()
    assert allocate(ctypes.byref(device), len(source)) == 0
    try:
        assert copy(device, source, len(source), 1) == 0
        assert copy(result, device, len(result), 2) == 0
        assert bytes(result) == bytes(source)
        assert memset(device, 0x3c, len(source)) == 0
        assert copy(result, device, len(result), 2) == 0
        assert bytes(result) == b'\x3c' * len(result)
    finally:
        assert free(device) == 0


def test_native_torch_coexists_with_runtime(runtime):
    # The runtime embeds Linux libc; native torch must retain its own libc
    # bindings. This checks coexistence, not CUDA dispatch in a CPU-only build.
    torch = pytest.importorskip('torch')
    left = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    right = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    assert (left @ right).tolist() == [[28., 34.], [76., 98.], [124., 162.]]
