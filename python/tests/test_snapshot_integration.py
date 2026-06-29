"""Snapshot read/write integration tests, exercised against a REAL server.

Unlike the mocked unit tests in ``test_lupine_adapter.py``, these drive PyTorch
through the LUPINE client shim against a live server that has CRIU installed, so
they actually verify save/restore behavior end to end.

Skipped unless ``LUPINE_SNAPSHOT_IT=1``. Required environment:

  - ``LUPINE_SERVER``         the remote GPU host (e.g. ``host:14833``)
  - ``LUPINE_LIBCUDA``        path to the LUPINE ``libcuda.so.1`` client shim
  - ``LD_PRELOAD``            the same shim (so PyTorch routes through it)
  - ``CUDA_VISIBLE_DEVICES=""`` so device 0 is the remote GPU, not a local one
  - the server must run with ``LUPINE_SNAPSHOT_DIR`` set (CRIU available)

Each scenario runs in its own subprocess: ``connect()`` must be entered before
PyTorch initializes CUDA, and the contexts share one process so the
connection-scoped reconnect/restore path is exercised (a tensor created in a
``"w"`` context stays usable in a later ``"r"`` context).
"""

import os
import secrets
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LUPINE_SNAPSHOT_IT") != "1",
    reason="set LUPINE_SNAPSHOT_IT=1 (+ LUPINE_SERVER/LUPINE_LIBCUDA) to run",
)

N = 4096  # float32 elements per tensor


def _run(scenario: str, snapshot_id: str, *args: str) -> str:
    """Run one scenario phase in a fresh subprocess; assert it succeeds."""
    proc = subprocess.run(
        [sys.executable, "-u", __file__, scenario, snapshot_id, *args],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"scenario {scenario!r} failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    return proc.stdout


def test_write_then_read_reuses_same_pointer():
    """w creates a snapshot; a later r restores it at the same device pointer,
    with contents intact and the memory still writable."""
    _run("write_read", secrets.token_hex(16))


def test_read_only_is_reusable_and_does_not_persist():
    """r never saves: a mutation in one r session does not survive into the
    next r restore, and the snapshot can be restored repeatedly."""
    _run("read_reuse", secrets.token_hex(16))


def test_rw_persists_mutations():
    """rw restores and saves: a mutation made under rw is visible to a later r."""
    _run("rw_persist", secrets.token_hex(16))


def test_write_overwrites_existing_snapshot():
    """A second w starts fresh and replaces the snapshot (it does not restore);
    a later r sees the newer contents. This is inherently cross-process: a fresh
    "w" needs a fresh CUDA context, so each phase runs in its own process."""
    sid = secrets.token_hex(16)
    _run("make_value", sid, "1.0")
    out = _run("make_value", sid, "2.0")
    ptr = next(l.split("=", 1)[1] for l in out.splitlines() if l.startswith("PTR="))
    _run("check_value", sid, ptr, "2.0")


# --------------------------------------------------------------------------
# Scenarios (executed in the subprocess). The first CUDA op inside a restoring
# context is always a read, which triggers the reconnect+restore cleanly.
# --------------------------------------------------------------------------


def _connect(snapshot_id: str, snapshot_type: str):
    import lupine

    return lupine.connect(
        host=os.environ["LUPINE_SERVER"],
        snapshot_id=snapshot_id,
        snapshot_type=snapshot_type,
    )


def scenario_write_read(snapshot_id: str) -> None:
    import torch

    with _connect(snapshot_id, "w"):
        t = torch.arange(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        ptr = t.data_ptr()

    with _connect(snapshot_id, "r"):
        assert t.data_ptr() == ptr, "device pointer changed across restore"
        assert torch.equal(
            t.cpu(), torch.arange(N, dtype=torch.float32)
        ), "tensor contents were not restored"
        t[0] = 123.0
        torch.cuda.synchronize()
        assert t[0].item() == 123.0, "restored memory is not writable"


def scenario_read_reuse(snapshot_id: str) -> None:
    import torch

    with _connect(snapshot_id, "w"):
        t = torch.zeros(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()

    # First read-only session: mutate, but r must not save it.
    with _connect(snapshot_id, "r"):
        assert t[0].item() == 0.0
        t[0] = 777.0
        torch.cuda.synchronize()

    # Second read-only session restores the original snapshot again; the
    # mutation above must be gone (proves r is non-destructive and reusable).
    with _connect(snapshot_id, "r"):
        assert t[0].item() == 0.0, "a read-only session persisted a change"


def scenario_rw_persist(snapshot_id: str) -> None:
    import torch

    with _connect(snapshot_id, "w"):
        t = torch.zeros(N, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()

    with _connect(snapshot_id, "rw"):
        assert t[0].item() == 0.0  # restored
        t[0] = 555.0  # saved on exit (rw writes)
        torch.cuda.synchronize()

    with _connect(snapshot_id, "r"):
        assert t[0].item() == 555.0, "rw did not persist the mutation"


def scenario_make_value(snapshot_id: str, value: str) -> None:
    """w: fill a tensor with `value` and snapshot it; print its device pointer."""
    import torch

    with _connect(snapshot_id, "w"):
        t = torch.full((N,), float(value), dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        print(f"PTR={t.data_ptr()}")


def scenario_check_value(snapshot_id: str, ptr: str, value: str) -> None:
    """r: restore and read the raw bytes at `ptr` (a fresh process can't rebuild
    a torch tensor at a restored address, so read via the driver API)."""
    import ctypes

    address, expected = int(ptr), float(value)
    libcuda = ctypes.CDLL(os.environ["LUPINE_LIBCUDA"])
    dtoh = libcuda.cuMemcpyDtoH_v2
    dtoh.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]

    with _connect(snapshot_id, "r"):
        assert libcuda.cuInit(0) == 0  # first call connects + triggers restore
        dev = ctypes.c_int()
        libcuda.cuDeviceGet(ctypes.byref(dev), 0)
        buf = (ctypes.c_float * N)()
        assert dtoh(buf, address, N * 4) == 0, "device read failed"
        assert all(x == expected for x in buf), f"snapshot did not hold {expected}"


if __name__ == "__main__":
    scenarios = {
        "write_read": scenario_write_read,
        "read_reuse": scenario_read_reuse,
        "rw_persist": scenario_rw_persist,
        "make_value": scenario_make_value,
        "check_value": scenario_check_value,
    }
    scenarios[sys.argv[1]](sys.argv[2], *sys.argv[3:])
    print(f"{sys.argv[1]}=OK")
