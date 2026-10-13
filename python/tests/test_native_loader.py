"""Native macOS checks for converted ELF loader scopes and provider ABIs."""

import ctypes as ct
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("LUPINE_TEST_NATIVE_LOOKUP_DYLIB"),
    reason="requires the converted native-provider loader fixture on ARM64 macOS",
)


@pytest.fixture(scope="module")
def library():
    return ct.CDLL(str(Path(os.environ["LUPINE_TEST_NATIVE_LOOKUP_DYLIB"]).resolve()))


@pytest.mark.parametrize("owner", ["caller", "provider"])
@pytest.mark.parametrize("case,expected", [("integers", 77), ("mixed", 1983), ("narrow", 78)])
def test_late_native_function_pointer_abi(library, owner, case, expected):
    function = getattr(library, f"check_{owner}_{case}")
    function.argtypes, function.restype = [], ct.c_double
    assert function() == expected


def test_scopes_and_loader_errors(library):
    library.check_scope.argtypes = []
    library.check_scope.restype = ct.c_int
    assert library.check_scope() == 4299


def test_native_data_and_tls(library):
    library.check_native_data.argtypes = []
    library.check_native_data.restype = ct.c_int
    assert library.check_native_data() == 1
