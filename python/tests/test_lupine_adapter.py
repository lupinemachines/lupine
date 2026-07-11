import importlib
import os
import subprocess
import sys
import threading
import types
from io import BytesIO

import pytest


class FakeDevice:
    def __init__(self, kind, index=None):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return isinstance(other, FakeDevice) and (
            self.type,
            self.index,
        ) == (other.type, other.index)

    def __repr__(self):
        return f"{self.type}:{self.index}"


class FakeCuda:
    def __init__(self):
        self.initialized = False
        self.count = 2

    def is_initialized(self):
        return self.initialized

    def device_count(self):
        return self.count

class FakeTorch(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.cuda = FakeCuda()
        self.version = types.SimpleNamespace(cuda="fake-cuda")

    def device(self, kind, index=None):
        return FakeDevice(kind, index)


@pytest.fixture
def lupine_module(monkeypatch):
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("LUPINE_SERVER", raising=False)
    monkeypatch.setattr("ctypes.CDLL", lambda *args, **kwargs: None)
    import lupine

    yield importlib.reload(lupine), fake_torch
    importlib.reload(lupine)


def test_connect_sets_env_and_returns_devices(lupine_module):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.count = 1

    with lupine.connect(host="host-a") as session:
        assert os.environ["LUPINE_SERVER"] == "host-a:14833"
        assert session.devices() == [FakeDevice("cuda", 0)]


def test_connect_loads_explicit_libcuda(lupine_module, monkeypatch, tmp_path):
    lupine, _ = lupine_module
    loaded = []
    libcuda = tmp_path / "libcuda.so.1"
    libcuda.write_bytes(b"")
    monkeypatch.setattr(lupine.ctypes, "CDLL", lambda *args, **kwargs: loaded.append(args))

    with lupine.connect(host="host-a", libcuda=libcuda):
        pass

    assert loaded == [(str(libcuda),)]


def test_connect_accepts_multiple_hosts_in_order(lupine_module):
    lupine, _ = lupine_module

    with lupine.connect(host=["host-a:15000", "host-b:16000"]) as session:
        assert session.servers == ("host-a:15000", "host-b:16000")


def test_session_devices_use_native_topology_for_multi_gpu_server(lupine_module):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.count = 4

    with lupine.connect(host="four-gpu-server") as session:
        assert session.devices() == [
            FakeDevice("cuda", 0),
            FakeDevice("cuda", 1),
            FakeDevice("cuda", 2),
            FakeDevice("cuda", 3),
        ]


def test_connect_uses_sidecar_when_torch_has_no_cuda_backend(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    fake_torch.version.cuda = None
    sentinel = object()
    calls = []

    monkeypatch.setattr(lupine.sys, "platform", "darwin")
    monkeypatch.setattr(
        lupine,
        "sidecar",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert lupine.connect(host="host-a") is sentinel
    assert calls == [{"server": "host-a:14833"}]


def test_connect_sidecar_fallback_rejects_multiple_hosts(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    fake_torch.version.cuda = None
    monkeypatch.setattr(lupine.sys, "platform", "darwin")

    with pytest.raises(lupine.LupineError, match="supports one host"):
        lupine.connect(host=["host-a:14833", "host-b:14833"])


def test_connect_sidecar_fallback_rejects_libcuda(lupine_module, monkeypatch, tmp_path):
    lupine, fake_torch = lupine_module
    fake_torch.version.cuda = None
    monkeypatch.setattr(lupine.sys, "platform", "darwin")

    with pytest.raises(lupine.LupineError, match="libcuda"):
        lupine.connect(host="host-a", libcuda=tmp_path / "libcuda.so.1")


def test_connect_requires_cuda_backend_off_macos(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    fake_torch.version.cuda = None
    monkeypatch.setattr(lupine.sys, "platform", "linux")

    with pytest.raises(lupine.LupineError, match="automatic LUPINE sidecar"):
        lupine.connect(host="host-a")


def test_connect_restores_env_when_cuda_was_not_initialized(lupine_module, monkeypatch):
    lupine, _ = lupine_module

    with lupine.connect(host="host-a"):
        assert os.environ["LUPINE_SERVER"] == "host-a:14833"

    assert "LUPINE_SERVER" not in os.environ


def test_connect_restores_env_when_cuda_initialized_inside_context(lupine_module):
    lupine, fake_torch = lupine_module

    with lupine.connect(host="host-a"):
        fake_torch.cuda.initialized = True

    assert "LUPINE_SERVER" not in os.environ


def test_connect_restores_env_after_topology_query(lupine_module):
    lupine, _ = lupine_module

    with lupine.connect(host="host-a") as session:
        session.devices()

    assert "LUPINE_SERVER" not in os.environ


def test_connect_accepts_matching_preconfigured_env(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.setenv("LUPINE_SERVER", "host-a:14833")

    with lupine.connect(host="host-a:14833"):
        assert os.environ["LUPINE_SERVER"] == "host-a:14833"

    assert os.environ["LUPINE_SERVER"] == "host-a:14833"


def test_connect_rejects_different_preconfigured_env(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.setenv("LUPINE_SERVER", "other:14833")

    with pytest.raises(lupine.LupineError, match="already configured differently"):
        with lupine.connect(host="host-a:14833"):
            pass


def test_connect_refuses_after_cuda_init(lupine_module):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.initialized = True

    with pytest.raises(lupine.LupineError, match="before PyTorch initializes CUDA"):
        with lupine.connect(host="host-a"):
            pass


def test_devices_use_native_topology_without_configured_servers(lupine_module):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.count = 3

    assert lupine.devices() == [
        FakeDevice("cuda", 0),
        FakeDevice("cuda", 1),
        FakeDevice("cuda", 2),
    ]


def test_session_no_visible_devices(lupine_module):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.count = 0

    with lupine.connect(host="host-a") as session:
        assert session.devices() == []


def test_connect_accepts_empty_hosts(lupine_module):
    lupine, _ = lupine_module

    with lupine.connect(host=[]) as session:
        assert session.servers == ()
        assert session.devices() == []
        assert "LUPINE_SERVER" not in os.environ


def test_duplicate_hosts_are_preserved(lupine_module):
    lupine, _ = lupine_module

    with lupine.connect(host=["host-a:14833", "host-a"]) as session:
        assert session.servers == ("host-a:14833", "host-a:14833")


def test_sidecar_container_runtime_defaults_to_arm64(monkeypatch):
    pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    monkeypatch.setattr(sidecar.shutil, "which", lambda name: "/usr/bin/container")
    monkeypatch.setattr(sidecar.sys, "platform", "darwin")

    cmd = sidecar.ContainerRuntime(server="host-a:14833").command("print(1)")

    assert cmd[:8] == [
        "/usr/bin/container",
        "run",
        "--rm",
        "--interactive",
        "--progress",
        "none",
        "--platform",
        "linux/arm64",
    ]
    assert "--rosetta" not in cmd
    assert "LUPINE_SERVER=host-a:14833" in cmd


def test_sidecar_container_runtime_is_macos_only(monkeypatch):
    pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    monkeypatch.setattr(sidecar.sys, "platform", "linux")

    with pytest.raises(sidecar.SidecarError, match="only supported on macOS"):
        sidecar.ContainerRuntime(server="host-a:14833").command("print(1)")


def test_sidecar_container_runtime_requires_cli(monkeypatch):
    pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    monkeypatch.setattr(sidecar.shutil, "which", lambda name: None)
    monkeypatch.setattr(sidecar.sys, "platform", "darwin")

    with pytest.raises(sidecar.SidecarError, match="brew install --cask container"):
        sidecar.ContainerRuntime(server="host-a:14833").command("print(1)")


def test_sidecar_container_runtime_starts_services_and_pulls_missing_image(monkeypatch):
    pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["system", "status", "--format"]:
            return sidecar.subprocess.CompletedProcess(args, 0, '{"status":"stopped"}', "")
        return sidecar.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sidecar.shutil, "which", lambda name: "/usr/bin/container")
    monkeypatch.setattr(sidecar.sys, "platform", "darwin")
    monkeypatch.setattr(sidecar.subprocess, "run", fake_run)

    sidecar.ContainerRuntime(server="host-a:14833").prepare()

    assert calls == [
        ["/usr/bin/container", "system", "status", "--format", "json"],
        ["/usr/bin/container", "system", "start"],
        ["/usr/bin/container", "image", "inspect", sidecar.DEFAULT_IMAGE],
    ]


def test_sidecar_container_runtime_pulls_missing_image(monkeypatch):
    pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["image", "inspect"]:
            return sidecar.subprocess.CompletedProcess(args, 1, "", "missing")
        return sidecar.subprocess.CompletedProcess(args, 0, '{"status":"running"}', "")

    monkeypatch.setattr(sidecar.shutil, "which", lambda name: "/usr/bin/container")
    monkeypatch.setattr(sidecar.sys, "platform", "darwin")
    monkeypatch.setattr(sidecar.subprocess, "run", fake_run)

    sidecar.ContainerRuntime(server="host-a:14833").prepare()

    assert calls[-1] == [
        "/usr/bin/container",
        "image",
        "pull",
        "--progress",
        "none",
        "--platform",
        "linux/arm64",
        sidecar.DEFAULT_IMAGE,
    ]


def test_sidecar_dispatch_mode_forwards_factory_ops(monkeypatch):
    pytest.importorskip("torch")
    import torch
    import lupine.sidecar as sidecar

    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    calls = []

    def fake_request(payload, input_tensors=(), *, decode_result=False):
        calls.append(payload)
        result = {
            "type": "tensor",
            "handle": 1,
            "shape": [2, 3],
            "dtype": "float32",
        }
        return session._decode(result, []) if decode_result else result

    monkeypatch.setattr(session, "_request", fake_request)

    with sidecar.SidecarDispatchMode(session):
        tensor = torch.zeros((2, 3), device=session.device(), dtype=torch.float32)

    assert isinstance(tensor, sidecar.SidecarTensor)
    assert calls[0]["op"] == "call"
    assert calls[0]["packet"] == "zeros"
    assert calls[0]["kwargs"]["device"] == {"__device__": "cuda:0"}
    assert calls[0]["kwargs"]["dtype"] == {"__dtype__": "float32"}


def test_sidecar_dispatch_mode_forwards_tensor_ops(monkeypatch):
    pytest.importorskip("torch")
    import torch
    import lupine.sidecar as sidecar

    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    calls = []

    def fake_request(payload, input_tensors=(), *, decode_result=False):
        calls.append(payload)
        result = {"type": "tensor", "handle": 2, "shape": [2], "dtype": "float32"}
        return session._decode(result, []) if decode_result else result

    monkeypatch.setattr(session, "_request", fake_request)
    tensor = sidecar.SidecarTensor(
        session=session,
        handle=1,
        shape=(2,),
        dtype=torch.float32,
        device=session.device(),
    )

    with sidecar.SidecarDispatchMode(session):
        result = tensor + 3

    assert isinstance(result, sidecar.SidecarTensor)
    assert calls[0]["packet"] == "add"
    assert calls[0]["overload"] == "Tensor"
    assert calls[0]["args"]["__tuple__"][0] == {"__sidecar_tensor__": 1}


def test_sidecar_dispatch_mode_keeps_cpu_ops_local(monkeypatch):
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    session = sidecar.SidecarSession(server="host-a:14833")
    requests = []
    monkeypatch.setattr(
        session, "_request", lambda *args, **kwargs: requests.append(args)
    )
    tensor = torch.arange(8)

    with sidecar.SidecarDispatchMode(session):
        result = tensor + 1

    assert torch.equal(result, torch.arange(1, 9))
    assert requests == []


def test_sidecar_to_copy_uses_direct_cpu_upload(monkeypatch):
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    source = torch.arange(8, dtype=torch.float32)
    sentinel = object()
    calls = []

    def upload(tensor, dtype):
        calls.append((tensor, dtype))
        return sentinel

    monkeypatch.setattr(session, "_upload_cpu_tensor", upload)

    result = session.forward(
        torch.ops.aten._to_copy.default,
        (source,),
        {"device": session.device(), "dtype": torch.float64},
    )

    assert result is sentinel
    assert calls == [(source, torch.float64)]


def test_sidecar_to_copy_uses_direct_gpu_download(monkeypatch):
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    source = sidecar.SidecarTensor(
        session=session,
        handle=1,
        shape=(8,),
        dtype=torch.float32,
        device=session.device(),
    )
    sentinel = object()
    calls = []

    def download(tensor, dtype):
        calls.append((tensor, dtype))
        return sentinel

    monkeypatch.setattr(session, "_download_tensor", download)

    result = session.forward(
        torch.ops.aten._to_copy.default,
        (source,),
        {"device": torch.device("cpu"), "dtype": torch.float64},
    )

    assert result is sentinel
    assert calls == [(source, torch.float64)]


def test_sidecar_copy_uses_direct_cpu_stream(monkeypatch):
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    sidecar._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    destination = sidecar.SidecarTensor(
        session=session,
        handle=1,
        shape=(8,),
        dtype=torch.float32,
        device=session.device(),
    )
    source = torch.arange(8, dtype=torch.float32)
    calls = []
    monkeypatch.setattr(
        session,
        "_copy_from_cpu",
        lambda dst, src: calls.append((dst, src)),
    )

    result = session.forward(
        torch.ops.aten.copy_.default,
        (destination, source),
        {},
    )

    assert result is destination
    assert calls == [(destination, source)]


def test_sidecar_cpu_tensor_rejects_noncontiguous_input():
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    session = sidecar.SidecarSession(server="host-a:14833")
    tensor = torch.arange(12).reshape(3, 4).transpose(0, 1)

    with pytest.raises(sidecar.SidecarError, match="must be contiguous"):
        session._encode(tensor, [])


@pytest.mark.parametrize(
    "tensor, error",
    [
        pytest.param(
            lambda torch: torch.tensor([1 + 2j]).conj(),
            "resolve_conj",
            id="conjugate",
        ),
        pytest.param(
            lambda torch: torch.tensor([1.0])._neg_view(),
            "resolve_neg",
            id="negative",
        ),
    ],
)
def test_sidecar_cpu_tensor_rejects_lazy_views(tensor, error):
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    with pytest.raises(sidecar.SidecarError, match=error):
        sidecar._cpu_tensor_metadata(tensor(torch))


class _ShortWriter:
    def __init__(self, stream, limit):
        self._stream = stream
        self._limit = limit
        self.requests = []

    def write(self, data):
        view = memoryview(data)
        self.requests.append(len(view))
        return self._stream.write(view[: self._limit])


class _ShortReader:
    def __init__(self, stream, limit):
        self._stream = stream
        self._limit = limit
        self.requests = []

    def readinto(self, buffer):
        view = memoryview(buffer)
        self.requests.append(len(view))
        return self._stream.readinto(view[: self._limit])


def test_sidecar_cpu_tensor_transport_uses_bounded_binary_chunks():
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    tensor = torch.arange(2 * sidecar._TENSOR_CHUNK_BYTES + 37, dtype=torch.uint8)
    stream = BytesIO()
    writer = _ShortWriter(stream, limit=997)

    sidecar._write_tensor(writer, tensor)
    stream.seek(0)
    reader = _ShortReader(stream, limit=991)
    result = sidecar._decode_cpu_tensor(sidecar._cpu_tensor_metadata(tensor), reader)

    assert torch.equal(result, tensor)
    assert max(writer.requests) <= sidecar._TENSOR_CHUNK_BYTES
    assert max(reader.requests) <= sidecar._TENSOR_CHUNK_BYTES
    assert len(writer.requests) > 3
    assert len(reader.requests) > 3


@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(lambda torch: torch.tensor([1.25, -2.5]), id="float32"),
        pytest.param(lambda torch: torch.tensor([1, -2, 2**40]), id="int64"),
        pytest.param(lambda torch: torch.tensor([True, False]), id="bool"),
        pytest.param(
            lambda torch: torch.tensor([1 + 2j, -3 + 0.5j]), id="complex64"
        ),
        pytest.param(
            lambda torch: torch.tensor([1.5, -2.25], dtype=torch.bfloat16),
            id="bfloat16",
        ),
        pytest.param(lambda torch: torch.empty((0, 3)), id="empty"),
        pytest.param(lambda torch: torch.tensor(7.5), id="scalar"),
    ],
)
def test_sidecar_cpu_tensor_binary_round_trip(tensor):
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    expected = tensor(torch)
    stream = BytesIO()
    sidecar._write_tensor(stream, expected)
    stream.seek(0)

    result = sidecar._decode_cpu_tensor(
        sidecar._cpu_tensor_metadata(expected), stream
    )

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert torch.equal(result, expected)


def test_sidecar_worker_consumes_tensor_stream_before_operation_error():
    torch = pytest.importorskip("torch")
    import lupine.sidecar as sidecar

    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", sidecar._worker_source()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = proc
    session._lock = threading.Lock()
    tensor = torch.arange(2 * sidecar._TENSOR_CHUNK_BYTES + 37, dtype=torch.uint8)
    streams = [(tensor, sidecar._cpu_tensor_metadata(tensor))]
    request = {
        "op": "call",
        "packet": "missing_operation",
        "overload": "default",
        "args": {"__tuple__": [{"__cpu_tensor__": 0}]},
        "kwargs": {},
    }
    try:
        with pytest.raises(sidecar.SidecarError):
            session._request(request, streams, decode_result=True)

        assert "torch" in session._request({"op": "ping"})
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


def test_sidecar_worker_streams_cpu_cuda_transfers_in_chunks():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    import lupine.sidecar as sidecar

    sidecar._ensure_registered()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", sidecar._worker_source()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = proc
    session._lock = threading.Lock()
    source = torch.arange(
        sidecar._TENSOR_CHUNK_BYTES // 4 + 37,
        dtype=torch.float32,
    )
    try:
        uploaded = session._upload_cpu_tensor(source, torch.float64)
        downloaded = session._download_tensor(uploaded, torch.float64)
        assert torch.equal(downloaded, source.to(torch.float64))

        destination = session._upload_cpu_tensor(torch.zeros_like(source), source.dtype)
        session._copy_from_cpu(destination, source)
        copied = session._download_tensor(destination, source.dtype)
        assert torch.equal(copied, source)

        empty = torch.empty((0, 3), dtype=torch.float32)
        remote_empty = session._upload_cpu_tensor(empty, empty.dtype)
        downloaded_empty = session._download_tensor(remote_empty, empty.dtype)
        assert downloaded_empty.shape == empty.shape

        broadcast_destination = session._upload_cpu_tensor(
            torch.zeros(4, dtype=torch.float32), torch.float32
        )
        session._copy_from_cpu(broadcast_destination, torch.tensor(3.0))
        broadcast = session._download_tensor(broadcast_destination, torch.float32)
        assert torch.equal(broadcast, torch.full((4,), 3.0))

        with pytest.raises(sidecar.SidecarError):
            session._copy_from_cpu(broadcast_destination, torch.zeros(3))
        assert "torch" in session._request({"op": "ping"})

        matrix = session._upload_cpu_tensor(torch.arange(6).reshape(2, 3), torch.int64)
        transposed = session.forward(torch.ops.aten.t.default, (matrix,), {})
        with pytest.raises(sidecar.SidecarError, match="must be contiguous"):
            session._download_tensor(transposed, transposed.dtype)
        assert "torch" in session._request({"op": "ping"})
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)
