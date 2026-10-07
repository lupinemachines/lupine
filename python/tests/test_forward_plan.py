"""``Backend.run_sync``'s plan construction, isolated from the compiled
extension and the worker: a fake ``call`` stands in for the RPC."""

from __future__ import annotations

import pickle

import torch

from lupine._backend import forward


def _fake_backend():
    backend = forward.Backend.__new__(forward.Backend)
    backend.frees = []
    backend.numbered = 0
    backend.calls = []

    def fake_call(kind, meta, body):
        backend.calls.append((kind, meta, body))
        return pickle.dumps(([], [], pickle.dumps(None)))

    backend.call = fake_call
    backend.result_device = lambda table, args: torch.device("cpu")
    return backend


def test_no_return_op_plan_still_sends_rpc_on_replay():
    backend = _fake_backend()
    st = forward._Op(torch.ops.aten._assert_async.default)
    assert st.returns == 0
    assert not st.mutates

    value, plan = forward.Backend.run_sync(backend, st, [], [], b"body", (), True)

    assert value is None
    assert plan.needs_rpc
