"""Native callers, Linux callers, and returned pointers share declared ABI metadata."""
import ctypes as ct
import json
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not os.environ.get("LUPINE_TEST_NATIVE_EXPORTS"),
    reason="requires the exported ABI fixtures on ARM64 macOS",
)
TYPES = {"i8": ct.c_int8, "u8": ct.c_uint8, "i16": ct.c_int16, "u16": ct.c_uint16,
         "i32": ct.c_int32, "u32": ct.c_uint32, "u64": ct.c_uint64,
         "ptr": ct.c_void_p, "f32": ct.c_float, "f64": ct.c_double}


@pytest.fixture(scope="module")
def fixtures():
    directory = Path(os.environ["LUPINE_TEST_NATIVE_EXPORTS"]).resolve()
    interface = json.loads((directory / "provider.json").read_text())
    # A conflicting native image must not replace this bundle's Linux ABI scope.
    provider = ct.CDLL(str(directory / "libscalar-provider.dylib"), mode=ct.RTLD_GLOBAL)
    return directory, interface, provider


@pytest.mark.parametrize("case,expected", [("integers", 77), ("mixed", 1983), ("narrow", 78)])
@pytest.mark.parametrize("query", [False, True])
def test_native_scalar_calls_and_returned_pointers(fixtures, case, expected, query):
    directory, interface, _ = fixtures
    library = ct.CDLL(str(directory / "exports.dylib"))
    types = interface["signatures"]["native_" + case]["arguments"]
    values = [-(i + 1) if kind.startswith("i") else i + 1 for i, kind in enumerate(types)]
    if query:
        function = library.native_query
        function.argtypes, function.restype = [ct.c_int, ct.POINTER(ct.c_void_p)], ct.c_int
        pointer = ct.c_void_p()
        assert function(["integers", "mixed", "narrow"].index(case), ct.byref(pointer)) == 0
        function = ct.CFUNCTYPE(ct.c_double, *[TYPES[kind] for kind in types])(pointer.value)
    else:
        function = getattr(library, "native_" + case)
        function.argtypes, function.restype = [TYPES[kind] for kind in types], ct.c_double
    assert function(*values) == expected


@pytest.mark.parametrize("case,expected", [("integers", 77), ("mixed", 1983), ("narrow", 78), ("stack", 77)])
def test_internal_linux_calls(fixtures, case, expected):
    directory, _, _ = fixtures
    library = ct.CDLL(str(directory / "exports.dylib"))
    function = getattr(library, "check_query_" + case)
    function.argtypes, function.restype = [], ct.c_double
    assert function() == expected


@pytest.mark.parametrize("case", ["integers", "mixed", "narrow"])
def test_native_dependency_pointers_are_unwrapped(fixtures, case):
    directory, _, provider = fixtures
    library = ct.CDLL(str(directory / "forward.dylib"))
    function = library.forward_query
    function.argtypes, function.restype = [ct.c_int, ct.POINTER(ct.c_void_p)], ct.c_int
    pointer = ct.c_void_p()
    assert function(["integers", "mixed", "narrow"].index(case), ct.byref(pointer)) == 0
    assert pointer.value == ct.cast(getattr(provider, "native_" + case), ct.c_void_p).value


def test_default_scope_preserves_linux_abi(fixtures):
    directory, _, _ = fixtures
    library = ct.CDLL(str(directory / "scope.dylib"), mode=ct.RTLD_GLOBAL)
    function = library.check_default_scope
    function.argtypes, function.restype = [], ct.c_double
    assert function() == 77


@pytest.mark.parametrize("name,expected", [(b"native_integers", 1), (b"check_query_integers", 0), (b"missing", 0), (None, 0)])
def test_export_abi_metadata(fixtures, name, expected):
    directory, _, _ = fixtures
    library = ct.CDLL(str(directory / "exports.dylib"))
    function = library.machso_native_export_abi
    function.argtypes, function.restype = [ct.c_char_p], ct.c_int
    assert function(name) == expected
