"""The converted ELF boundary uses Linux stack slots, not Apple's packing."""

import ctypes as ct
import os
import struct
import sys
from types import SimpleNamespace

import pytest

from lupine._backend.native.runtime import (
    LinuxInt64,
    LinuxUInt64,
    LinuxFloatSlot,
    LinuxBoolSlot,
    Runtime,
    linux_argtypes,
)


def test_narrow_spills_use_linux_slots_after_eight_general_registers():
    types = (ct.c_void_p,) * 8 + (ct.c_int, ct.c_uint, ct.c_int)
    result = linux_argtypes(types)
    assert result[:8] == types[:8]
    assert result[8:] == (LinuxInt64, LinuxUInt64, LinuxInt64)
    assert all(ct.sizeof(argument) == 8 for argument in result[8:])


def test_float_arguments_have_a_separate_register_bank():
    types = (ct.c_float, ct.c_int) * 8 + (ct.c_uint,)
    assert linux_argtypes(types)[:-1] == types[:-1]
    assert linux_argtypes(types)[-1] is LinuxUInt64


def test_promoted_parameters_accept_existing_ctypes_values():
    signed = ct.CFUNCTYPE(ct.c_int64, LinuxInt64)(lambda value: value.value)
    unsigned = ct.CFUNCTYPE(ct.c_uint64, LinuxUInt64)(lambda value: value.value)
    assert signed(ct.c_int(-17)) == -17
    assert unsigned(ct.c_uint(2**32 - 1)) == 2**32 - 1


def test_spilled_floats_preserve_bits_in_eight_byte_slots():
    def read_bits(value):
        return struct.unpack("<Q", struct.pack("<d", value.value))[0]

    read = ct.CFUNCTYPE(ct.c_uint64, LinuxFloatSlot)(read_bits)
    for value in (0.0, -0.0, 1.25, -37.5, float("inf"), float("nan"), 1e40):
        original = ct.c_float(value)
        expected = struct.unpack("<I", struct.pack("<f", original.value))[0]
        assert read(original) == expected
    types = (ct.c_float, ct.c_int) * 11
    assert linux_argtypes(types)[:16] == types[:16]
    assert linux_argtypes(types)[16:] == (LinuxFloatSlot, LinuxInt64) * 3


def test_spilled_bool_values_are_canonical():
    read = ct.CFUNCTYPE(ct.c_uint64, LinuxBoolSlot)(lambda value: value.value)
    assert read(ct.c_bool(False)) == 0
    assert read(-19) == 1
    assert read(None) == 0
    assert linux_argtypes((ct.c_void_p,) * 8 + (ct.c_bool,))[-1] is LinuxBoolSlot


def test_native_driver_functions_keep_their_original_abi():
    runtime = Runtime.__new__(Runtime)
    runtime.export_abi = None
    runtime.runtime = SimpleNamespace(function=SimpleNamespace())
    driver = SimpleNamespace(function=SimpleNamespace())
    types = (ct.c_int,) * 11
    assert runtime.bind(driver, "function", *types).argtypes == types
    assert runtime.bind(runtime.runtime, "function", *types).argtypes[-1] is LinuxInt64
    runtime.export_abi = lambda name: name == b"function"
    assert runtime.bind(runtime.runtime, "function", *types).argtypes == types


@pytest.mark.parametrize("cached", [False, True])
def test_temporary_allocation_is_freed_when_body_fails(cached):
    runtime = Runtime.__new__(Runtime)
    runtime.malloc = lambda size: 4096
    released = []
    runtime.free = lambda pointer: released.append(pointer.value) or 0
    runtime._operators = (
        SimpleNamespace(malloc=runtime.malloc, free=runtime.free) if cached else None
    )
    if cached:
        runtime.malloc = lambda size: pytest.fail("must use the operator allocator")
        runtime.free = lambda pointer: pytest.fail("must return memory to its allocator")
    with pytest.raises(ValueError, match="operation failed"):
        with runtime.allocated(128) as pointer:
            assert pointer.value == 4096
            raise ValueError("operation failed")
    assert released == [4096]


def test_failed_temporary_allocation_does_not_free_a_pointer():
    runtime = Runtime.__new__(Runtime)
    runtime._operators = None
    runtime.allocate = lambda pointer, size: 2
    released = []
    runtime.free = lambda pointer: released.append(pointer.value) or 0
    with pytest.raises(RuntimeError, match="cudaMalloc.*status 2"):
        with runtime.allocated(128):
            pytest.fail("failed allocation must not enter the body")
    assert released == []


@pytest.fixture
def converted_fixture():
    path = os.environ.get("LUPINE_TEST_NATIVE_ABI_DYLIB")
    if sys.platform != "darwin" or not path:
        pytest.skip("requires the converted ARM64 ABI fixture on macOS")
    runtime = Runtime.__new__(Runtime)
    runtime.runtime = ct.CDLL(path)
    runtime.export_abi = None
    return runtime


def test_converted_mixed_register_and_stack_arguments(converted_fixture):
    function = converted_fixture.bind(
        converted_fixture.runtime, "abi_scalar_mix", *((ct.c_float, ct.c_int) * 11)
    )
    for sign in (-1, 1):
        arguments = []
        for i in range(11):
            arguments.extend(
                (ct.c_float(sign * (i + 0.25)), ct.c_int(sign * (100 + i)))
            )
        assert function(*arguments) == sign * 1210


def test_converted_tls_and_guest_x18_preserve_host_x18(converted_fixture):
    path = os.environ.get("LUPINE_TEST_NATIVE_ABI_HOST_DYLIB")
    assert path, "native host-register check library is required"
    host = ct.CDLL(path)
    check = host.abi_preserves_host_x18
    check.argtypes, check.restype = (ct.c_void_p, ct.c_void_p), ct.c_int
    tls = converted_fixture.runtime.abi_tls_x0
    guest_tls = converted_fixture.runtime.abi_tls_x18
    guest_value = converted_fixture.runtime.abi_value_x18
    for function in (tls, guest_tls, guest_value):
        function.argtypes, function.restype = (), ct.c_void_p
    for _ in range(100):
        pointer = tls()
        assert pointer
        assert check(guest_tls, pointer) == 1
        assert check(guest_value, 42) == 1
