"""The worker's request loop, on the CPU device (``LUPINE_WORKER_DEVICE=cpu``,
no shims, no torch CUDA needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import lupine._worker as worker  # noqa: E402


class _FakeServer:
    def __init__(self):
        self.responses = []

    def respond(self, ticket, status, payload):
        self.responses.append((ticket, status, payload))


def _sync_frame(executor):
    meta = executor.wire.pack_meta([], [], None, 0)
    return (1, executor.wire.OP_SYNC, meta, b"", None)


def test_sync_error_drains_a_stale_fire_and_forget_error(monkeypatch):
    monkeypatch.setenv("LUPINE_WORKER_DEVICE", "cpu")
    executor = worker.Executor(_FakeServer())
    server = executor.server
    wire = executor.wire

    # An earlier fire-and-forget request (e.g. an OP_PLAN replay) failed.
    executor.record("stale async error")

    # This sync call fails independently of that earlier error.
    executor.run = lambda meta, body: (_ for _ in ()).throw(RuntimeError("this call failed"))
    executor.handle(_sync_frame(executor))
    _, status1, _ = server.responses[0]
    assert status1 == wire.ERROR

    # The retry: an ordinary successful sync call must not see the drained
    # error again (accelerate's find_executable_batch_size retries on
    # exactly this pattern).
    executor.run = lambda meta, body: (None, [], "noop")
    executor.report = lambda value, inputs: b"ok"
    executor.handle(_sync_frame(executor))
    _, status2, payload2 = server.responses[1]
    assert status2 == wire.OK, payload2
