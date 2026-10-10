"""CPU-only regression using real RPC end to end and a fake native NVML."""
import concurrent.futures
import ctypes as C
import os
import socket
import subprocess
import sys
import tempfile
import time


class Sample(C.Structure):
    _fields_ = [("pid", C.c_uint), ("timeStamp", C.c_ulonglong),
                ("smUtil", C.c_uint), ("memUtil", C.c_uint),
                ("encUtil", C.c_uint), ("decUtil", C.c_uint)]


def client_checks(path):
    lib = C.CDLL(path)
    query = lib.nvmlDeviceGetProcessUtilization
    query.argtypes = [C.c_void_p, C.POINTER(Sample), C.POINTER(C.c_uint), C.c_ulonglong]
    query.restype = C.c_int
    get_handle = lib.nvmlDeviceGetHandleByIndex_v2
    get_handle.argtypes = [C.c_uint, C.POINTER(C.c_void_p)]
    count = C.c_uint(0)
    assert query(None, None, C.byref(count), 0) == 1  # UNINITIALIZED
    assert lib.nvmlInit_v2() == 0
    assert lib.nvmlDeviceGetCount_v2(C.byref(count)) == 0 and count.value == 3
    devices = []
    for index in range(3):
        device = C.c_void_p()
        assert get_handle(index, C.byref(device)) == 0
        devices.append(device)
    assert query(devices[0], None, None, 0) == 2  # INVALID_ARGUMENT
    assert query(None, None, C.byref(count), 0) == 2

    for index, device in enumerate(devices[:2]):
        count.value = 123
        assert query(device, None, C.byref(count), 0) == 7 and count.value == 2
        samples = (Sample * 4)()
        C.memset(samples, 0xA5, C.sizeof(samples))
        original = bytes(samples)
        for capacity in (0, 1):
            count.value = capacity
            assert query(device, samples, C.byref(count), 0) == 7
            assert count.value == 2 and bytes(samples) == original
        count.value = 4
        assert query(device, samples, C.byref(count), 0) == 0 and count.value == 2
        assert [(s.pid, s.timeStamp, s.smUtil, s.memUtil, s.encUtil, s.decUtil)
                for s in samples[:2]] == [
                    (42000 + index * 100, 100, 50, 20, 3, 4),
                    (42001 + index * 100, 200, 51, 21, 3, 4)]
        assert bytes(samples)[2 * C.sizeof(Sample):] == original[2 * C.sizeof(Sample):]
        count.value = 4
        assert query(device, samples, C.byref(count), 100) == 0 and count.value == 1
        assert samples[0].timeStamp == 200
        for timestamp, expected in ((200, 6), (10, 3), (20, 15), (30, 999), (40, 7)):
            before = bytes(samples)
            count.value = 4
            assert query(device, samples, C.byref(count), timestamp) == expected
            assert bytes(samples) == before
        # Advertised capacity need not become an unbounded server allocation.
        count.value = 0xffffffff
        assert query(device, samples, C.byref(count), 0) == 0 and count.value == 2
    count.value = 4
    before = bytes(samples)
    assert query(devices[2], samples, C.byref(count), 0) == 13  # FUNCTION_NOT_FOUND
    assert bytes(samples) == before

    def thread_checks(index):
        for _ in range(16):
            buf = (Sample * 2)()
            n = C.c_uint(2)
            assert query(devices[index % 2], buf, C.byref(n), 0) == 0
            assert n.value == 2 and buf[0].pid == 42000 + (index % 2) * 100
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(thread_checks, range(8)))
    assert lib.nvmlShutdown() == 0
    assert query(devices[0], None, C.byref(count), 0) == 1
    print("PASS: NVML export, routing, count query, zero/small/large buffers, "
          "timestamps, unchanged PIDs, error codes, missing native API, "
          "buffer guards and 128 concurrent RPC queries", flush=True)


def run(server, client, mock, missing):
    processes = []
    logs = []
    endpoints = []
    try:
        for index, directory in enumerate((mock, mock, missing)):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = dict(os.environ, LUPINE_PORT=str(port), MOCK_NVML_ID=str(index),
                       LD_LIBRARY_PATH=directory)
            log = tempfile.TemporaryFile()
            logs.append(log)
            process = subprocess.Popen([server], env=env, stdout=log, stderr=log)
            processes.append(process)
            deadline = time.monotonic() + 10
            while True:
                assert process.poll() is None, "test server exited"
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("test server did not listen")
                    time.sleep(0.05)
            endpoints.append(f"127.0.0.1:{port}")
        env = dict(os.environ, LUPINE_SERVER=",".join(endpoints))
        subprocess.run([sys.executable, __file__, "--client", client], env=env,
                       check=True, timeout=30)
    except Exception:
        for log in logs:
            log.seek(0)
            print(log.read().decode(errors="replace"), file=sys.stderr)
        raise
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    if sys.argv[1] == "--client":
        client_checks(sys.argv[2])
    else:
        run(*sys.argv[1:])
