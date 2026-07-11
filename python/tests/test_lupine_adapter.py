import importlib
import os
import sys
import types

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
