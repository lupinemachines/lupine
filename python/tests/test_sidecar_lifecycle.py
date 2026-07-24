import gc
import io
import importlib
import json
import subprocess
import sys
import weakref

import pytest

torch = pytest.importorskip("torch")

sidecar = importlib.import_module("lupine.sidecar")


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def make_tensor(session, handle):
    return session._decode(
        {
            "type": "tensor",
            "handle": handle,
            "shape": [2],
            "dtype": "float32",
        },
        [],
    )


def test_tensor_close_is_idempotent_and_prevents_use_after_release(monkeypatch):
    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    requests = []
    monkeypatch.setattr(
        session, "_request", lambda payload: requests.append(payload) or {}
    )

    tensor = make_tensor(session, 1)
    assert session.live_handle_count == 1

    tensor.close()
    tensor.close()

    assert requests == [{"op": "release", "handles": [1]}]
    assert tensor.closed
    assert session.live_handle_count == 0
    with pytest.raises(sidecar.SidecarError, match="handle 1 is closed"):
        session._encode(tensor, [])


def test_tensor_finalizer_releases_owned_handle(monkeypatch):
    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    requests = []
    monkeypatch.setattr(
        session, "_request", lambda payload: requests.append(payload) or {}
    )

    tensor = make_tensor(session, 2)
    tensor_ref = weakref.ref(tensor)
    del tensor
    gc.collect()

    assert tensor_ref() is None
    assert requests == [{"op": "release", "handles": [2]}]
    assert session.live_handle_count == 0


def test_session_close_releases_all_handles_once(monkeypatch):
    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = FakeProcess()
    requests = []
    monkeypatch.setattr(
        session, "_request", lambda payload: requests.append(payload) or {}
    )

    first = make_tensor(session, 3)
    second = make_tensor(session, 4)
    session.close()
    session.close()

    assert requests == [{"op": "release", "handles": [3, 4]}]
    assert session.closed
    assert session.live_handle_count == 0
    assert first.closed and second.closed
    with pytest.raises(sidecar.SidecarError, match="closed"):
        session._encode(first, [])


def test_tensor_release_failure_closes_session(monkeypatch):
    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    proc = FakeProcess()
    session._proc = proc
    tensor = make_tensor(session, 7)
    requests = []

    def fail_release(payload):
        requests.append(payload)
        raise sidecar.SidecarError("release failed")

    monkeypatch.setattr(session, "_request", fail_release)

    with pytest.raises(sidecar.SidecarError, match="release failed"):
        tensor.close()
    tensor.close()

    assert requests == [{"op": "release", "handles": [7]}]
    assert proc.terminated
    assert session.closed
    assert session.live_handle_count == 0


def test_partial_decode_releases_every_result_handle(monkeypatch):
    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    result = {
        "type": "tuple",
        "items": [
            {"type": "tensor", "handle": 5, "shape": [1], "dtype": "float32"},
            {"type": "unsupported"},
        ],
    }
    responses = [
        {"ok": True, "result": result, "tensor_streams": []},
        {
            "ok": True,
            "result": {"released": 1, "live_handles": 0},
            "tensor_streams": [],
        },
    ]
    proc = FakeProcess()
    proc.stdin = io.BytesIO()
    proc.stdout = io.BytesIO(
        b"".join(json.dumps(response).encode() + b"\n" for response in responses)
    )
    proc.stderr = io.BytesIO()
    session._proc = proc

    with pytest.raises(sidecar.SidecarError, match="unsupported result"):
        session._request({"op": "call"}, decode_result=True)

    assert b'"op": "release", "handles": [5]' in proc.stdin.getvalue()
    assert session.live_handle_count == 0


def test_cancellation_after_response_releases_every_result_handle(monkeypatch):
    session = sidecar.SidecarSession(server="host-a:14833")
    result = {
        "type": "tensor",
        "handle": 8,
        "shape": [1],
        "dtype": "float32",
    }
    responses = [
        {"ok": True, "result": result, "tensor_streams": []},
        {
            "ok": True,
            "result": {"released": 1, "live_handles": 0},
            "tensor_streams": [],
        },
    ]
    proc = FakeProcess()
    proc.stdin = io.BytesIO()
    proc.stdout = io.BytesIO(
        b"".join(json.dumps(response).encode() + b"\n" for response in responses)
    )
    proc.stderr = io.BytesIO()
    session._proc = proc

    def cancel_decode(value, tensors):
        raise KeyboardInterrupt

    monkeypatch.setattr(session, "_decode", cancel_decode)

    with pytest.raises(KeyboardInterrupt):
        session._request({"op": "call"}, decode_result=True)

    assert b'"op": "release", "handles": [8]' in proc.stdin.getvalue()
    assert session.live_handle_count == 0


class BrokenPipe:
    def write(self, value):
        raise BrokenPipeError("test transport failure")

    def flush(self):
        pass

    def readline(self):
        return ""


def test_transport_failure_terminates_worker_and_drops_handle_ownership():
    session = sidecar.SidecarSession(server="host-a:14833")
    proc = FakeProcess()
    proc.stdin = BrokenPipe()
    proc.stdout = BrokenPipe()
    proc.stderr = None
    session._proc = proc
    session._live_handles.add(6)

    with pytest.raises(sidecar.SidecarError, match="transport failed"):
        session._request({"op": "ping"})

    assert proc.terminated
    assert session.closed
    assert session.live_handle_count == 0


_FAKE_TORCH = r"""
import sys
import types


class Device:
    def __init__(self, kind):
        self.type = kind.split(":", 1)[0]


class DType:
    def __str__(self):
        return "torch.float32"


class Tensor:
    def __init__(self):
        self.device = Device("cuda")
        self.shape = (2,)
        self.dtype = DType()

    def tolist(self):
        return [0.0, 0.0]


class Size(tuple):
    pass


class Cuda:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def get_device_name(self, index):
        return "fake-gpu"

    def memory_allocated(self):
        return len(getattr(sys.modules["lupine.worker"], "objects", {})) * 1024

    def memory_reserved(self):
        return self.memory_allocated()


def make():
    return Tensor()


def consume(tensor):
    return Tensor()


def partial():
    return [Tensor(), object()]


torch = types.ModuleType("torch")
torch.__version__ = "test"
torch.Tensor = Tensor
torch.Size = Size
torch.float32 = DType()
torch.device = Device
torch.cuda = Cuda()
torch.ops = types.SimpleNamespace(
    aten=types.SimpleNamespace(
        make=types.SimpleNamespace(default=make),
        consume=types.SimpleNamespace(default=consume),
        partial=types.SimpleNamespace(default=partial),
    )
)
torch.utils = types.SimpleNamespace(
    _python_dispatch=types.SimpleNamespace(TorchDispatchMode=object),
)
sys.modules["torch"] = torch
"""


def test_worker_handle_count_stays_stable_and_failure_rolls_back():
    script = _FAKE_TORCH + "\n" + sidecar._worker_source()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def request(payload):
        payload["tensor_streams"] = []
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    try:
        for _ in range(50):
            response = request(
                {
                    "op": "call",
                    "packet": "make",
                    "overload": "default",
                }
            )
            assert response["ok"]
            handle = response["result"]["handle"]
            assert request({"op": "stats"})["result"] == {
                "live_handles": 1,
                "cuda_memory_allocated": 1024,
                "cuda_memory_reserved": 1024,
            }
            released = request({"op": "release", "handles": [handle]})
            assert released["result"] == {"released": 1, "live_handles": 0}

        duplicate = request({"op": "release", "handles": [handle]})
        assert duplicate["result"] == {"released": 0, "live_handles": 0}

        partial = request({"op": "call", "packet": "partial", "overload": "default"})
        assert not partial["ok"]
        assert request({"op": "stats"})["result"] == {
            "live_handles": 0,
            "cuda_memory_allocated": 0,
            "cuda_memory_reserved": 0,
        }

        stale = request(
            {
                "op": "call",
                "packet": "consume",
                "overload": "default",
                "args": [{"__sidecar_tensor__": handle}],
            }
        )
        assert not stale["ok"]
        assert "released or unknown" in stale["error"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)
