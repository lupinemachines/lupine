"""GPU regression: internal snapshots stay silent; application errors still log."""

import ctypes as C
import sys

library = C.CDLL(sys.argv[1])
P, I, U = C.c_void_p, C.c_int, C.c_uint


def bind(name, arguments):
    fn = getattr(library, name)
    fn.argtypes = arguments
    fn.restype = I
    return fn


def check(result):
    assert result == 0, result


check(bind("cuInit", [U])(0))
context = P()
check(bind("cuDevicePrimaryCtxRetain", [C.POINTER(P), I])(C.byref(context), 0))
check(bind("cuCtxSetCurrent", [P])(context))
messages = []
Callback = C.CFUNCTYPE(None, P, I, P, C.c_size_t)


@Callback
def callback(user, level, message, length):
    messages.append(C.string_at(message, length).decode(errors="replace"))


handle = P()
check(
    bind("cuLogsRegisterCallback", [Callback, P, C.POINTER(P)])(
        callback, None, C.byref(handle)
    )
)
ptx = C.create_string_buffer(
    b".version 6.4\n.target sm_75\n.address_size 64\n"
    b".visible .entry probe(.param .u64 value) { ret; }\n"
)
gpu_library = P()
check(
    bind("cuLibraryLoadData", [C.POINTER(P), P, P, P, U, P, P, U])(
        C.byref(gpu_library), ptx, None, None, 0, None, None, 0
    )
)
internal = [
    m
    for m in messages
    if "cuKernelGetParamInfo" in m
    or "cuFuncGetAttribute" in m
    or "cuKernelGetAttribute" in m
]
assert not internal, internal
before = len(messages)
value = I()
result = bind("cuDeviceGetAttribute", [C.POINTER(I), I, I])(C.byref(value), 999999, 0)
assert result != 0, result
assert len(messages) > before, "application CUDA error callback was lost"
check(bind("cuLibraryUnload", [P])(gpu_library))
check(bind("cuLogsUnregisterCallback", [P])(handle))
check(bind("cuDevicePrimaryCtxRelease_v2", [I])(0))
print("RESULT: PASS internal probes silent; application error callback preserved")
