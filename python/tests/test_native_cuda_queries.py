"""CPU checks using the actual native CUDA driver's query API exports."""
import ctypes as ct
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("LUPINE_TEST_NATIVE_CUDA_QUERY_DYLIB"),
    reason="requires the converted caller and native CUDA driver on ARM64 macOS",
)


@pytest.mark.parametrize("case", ["legacy", "modern", "recursive", "unsupported_legacy", "unsupported_modern"])
def test_native_cuda_query_returns_caller_abi_pointer(case):
    library = ct.CDLL(str(Path(os.environ["LUPINE_TEST_NATIVE_CUDA_QUERY_DYLIB"]).resolve()))
    function = getattr(library, f"check_{case}")
    function.argtypes, function.restype = [], ct.c_int
    assert function() == 1
