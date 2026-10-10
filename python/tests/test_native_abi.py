"""The converted ELF boundary uses Linux stack slots, not Apple's packing."""

import ctypes as ct
from types import SimpleNamespace

from lupine._backend.native.runtime import (
    LinuxInt64,
    LinuxUInt64,
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


def test_native_driver_functions_keep_their_original_abi():
    runtime = Runtime.__new__(Runtime)
    runtime.runtime = SimpleNamespace(function=SimpleNamespace())
    driver = SimpleNamespace(function=SimpleNamespace())
    types = (ct.c_int,) * 11
    assert runtime.bind(driver, "function", *types).argtypes == types
    assert runtime.bind(runtime.runtime, "function", *types).argtypes[-1] is LinuxInt64
