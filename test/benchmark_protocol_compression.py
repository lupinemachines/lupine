"""Linux CUDA protocol benchmark; select the client library with LD_LIBRARY_PATH.

Requires NumPy. LUPINE_SERVER selects the remote GPU through the normal shim.
For same-host GPU servers, isolate the real driver as the WAN benchmark does;
otherwise a local GPU can silently bypass the remote transport.
"""
import argparse
import ctypes as C
import json
import resource
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--rpc-count", type=int, default=2000)
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 4 or args.iterations <= 0 or args.rpc_count <= 0:
        parser.error("positive counts and a byte size divisible by four are required")
    cuda = C.CDLL("libcuda.so.1")

    def call(name, *arguments):
        result = getattr(cuda, name)(*arguments)
        if result:
            raise RuntimeError(f"{name}: CUDA error {result}")

    def measure(label, operation, count, size=0):
        start = time.perf_counter()
        cpu = time.process_time()
        for _ in range(count):
            operation()
        print(json.dumps({
            "label": label,
            "wall_s": time.perf_counter() - start,
            "cpu_s": time.process_time() - cpu,
            "calls": count,
            "logical_bytes": count * size,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }), flush=True)

    call("cuInit", 0)
    context = C.c_void_p()
    call("cuDevicePrimaryCtxRetain", C.byref(context), 0)
    call("cuCtxSetCurrent", context)
    pointer = C.c_uint64()
    call("cuMemAlloc_v2", C.byref(pointer), C.c_size_t(args.bytes))
    free, total = C.c_size_t(), C.c_size_t()
    measure("small_sync_rpc", lambda: call("cuMemGetInfo_v2", C.byref(free), C.byref(total)), args.rpc_count)
    random = np.random.default_rng(20260927)
    for kind in ("zeros", "random", "float32"):
        if kind == "zeros":
            data = np.zeros(args.bytes, dtype=np.uint8)
        elif kind == "random":
            data = random.integers(0, 256, args.bytes, dtype=np.uint8)
        else:
            data = random.standard_normal(args.bytes // 4, dtype=np.float32).view(np.uint8)
        output = np.empty_like(data)
        measure(kind + "_htod", lambda: call("cuMemcpyHtoD_v2", pointer, C.c_void_p(data.ctypes.data), C.c_size_t(args.bytes)), args.iterations, args.bytes)
        measure(kind + "_dtoh", lambda: call("cuMemcpyDtoH_v2", C.c_void_p(output.ctypes.data), pointer, C.c_size_t(args.bytes)), args.iterations, args.bytes)
        if not np.array_equal(data, output):
            raise RuntimeError(f"{kind}: round-trip mismatch")
    call("cuMemFree_v2", pointer)
    call("cuCtxSetCurrent", C.c_void_p())
    call("cuDevicePrimaryCtxRelease_v2", 0)
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
