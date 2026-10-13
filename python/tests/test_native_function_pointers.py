"""CPU checks for native query APIs returning converted caller ABI pointers."""
import ctypes as ct
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("LUPINE_TEST_NATIVE_QUERY_DYLIB"),
    reason="requires the converted function pointer fixture on ARM64 macOS",
)


@pytest.fixture(scope="module")
def library():
    return ct.CDLL(str(Path(os.environ["LUPINE_TEST_NATIVE_QUERY_DYLIB"]).resolve()))


@pytest.mark.parametrize("case,expected", [("integers", 77), ("mixed", 1983), ("narrow", 78), ("stack", 77)])
def test_query_returns_caller_abi_pointer(library, case, expected):
    function = getattr(library, f"check_query_{case}")
    function.argtypes, function.restype = [], ct.c_double
    assert function() == expected


@pytest.mark.parametrize("case", ["unsupported", "error", "null", "unknown", "wide_error"])
def test_query_status_and_output(library, case):
    function = getattr(library, f"check_query_{case}")
    function.argtypes, function.restype = [], ct.c_int
    assert function() == 1
