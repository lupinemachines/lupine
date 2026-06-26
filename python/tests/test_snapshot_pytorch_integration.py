"""PyTorch snapshot read/write integration test.

Verifies the documented contract: a Python variable that owns GPU tensors keeps
working across a snapshot boundary. A model moved to CUDA under a ``"w"`` context
is still usable under a later ``"r"`` context in the same process — its tensors
keep the same device pointer, their contents survive, and they remain writable.

Needs a real LUPINE setup and is skipped unless ``LUPINE_SNAPSHOT_IT=1``:

  - ``LUPINE_SERVER``   the remote GPU host (e.g. ``host:14833``)
  - ``LUPINE_LIBCUDA``  path to the LUPINE ``libcuda.so.1`` client shim
  - the server must run with ``LUPINE_SNAPSHOT_DIR`` set (root, CRIU installed)
  - the client must target the REMOTE device (e.g. ``CUDA_VISIBLE_DEVICES=""``)

The body runs in a subprocess so CUDA initialization is isolated from the rest
of the suite and the two contexts share one process (as real usage would).
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LUPINE_SNAPSHOT_IT") != "1",
    reason="set LUPINE_SNAPSHOT_IT=1 (+ LUPINE_SERVER/LUPINE_LIBCUDA) to run",
)

N = 4096  # float32 elements


def _parse(stdout, key):
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"missing {key} in:\n{stdout}")


def test_tensor_survives_snapshot_in_same_process():
    proc = subprocess.run(
        [sys.executable, __file__],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (rc={proc.returncode})\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out = proc.stdout
    assert _parse(out, "SAME_PTR") == "OK"   # restored at the same device address
    assert _parse(out, "VERIFY") == "OK"     # contents survived the snapshot
    assert _parse(out, "WRITABLE") == "OK"   # restored memory is still writable


def _main():
    import secrets

    import lupine
    import torch

    snapshot_id = secrets.token_hex(16)
    host = os.environ["LUPINE_SERVER"]
    expected = torch.arange(N, dtype=torch.float32)

    # "w": move the tensor onto the GPU and snapshot that state on exit.
    with lupine.connect(host=host, snapshot_id=snapshot_id, snapshot_type="w"):
        t = expected.to("cuda")
        torch.cuda.synchronize()
        ptr = t.data_ptr()

    # "r": restore read-only. The same `t` must still be usable: same device
    # pointer, same contents, and writable — no reload or re-transfer.
    with lupine.connect(host=host, snapshot_id=snapshot_id, snapshot_type="r"):
        print("SAME_PTR=" + ("OK" if t.data_ptr() == ptr else "FAIL"))
        print("VERIFY=" + ("OK" if torch.equal(t.cpu(), expected) else "FAIL"))
        t[0] = 999.0
        torch.cuda.synchronize()
        print("WRITABLE=" + ("OK" if t[0].item() == 999.0 else "FAIL"))


if __name__ == "__main__":
    _main()
