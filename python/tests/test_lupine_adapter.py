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
    lupine = importlib.import_module("lupine")

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
    monkeypatch.setattr(
        lupine.ctypes, "CDLL", lambda *args, **kwargs: loaded.append(args)
    )

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


def test_session_device_returns_one_native_device(lupine_module):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.count = 2

    with lupine.connect(host="host-a") as session:
        assert session.device() == FakeDevice("cuda", 0)
        assert session.device(1) == FakeDevice("cuda", 1)
        assert session.device(-1) == FakeDevice("cuda", 1)
        with pytest.raises(IndexError):
            session.device(2)


def test_connect_auto_uses_sidecar_on_macos_without_cuda(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    fake_torch.version.cuda = None
    sentinel = object()
    calls = []

    monkeypatch.setattr(lupine.sys, "platform", "darwin")
    monkeypatch.setattr(
        lupine,
        "_create_sidecar",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert lupine.connect(host="host-a") is sentinel
    assert calls == [{"server": "host-a:14833"}]


def test_connect_sidecar_true_forces_sidecar(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    sentinel = object()
    calls = []

    monkeypatch.setattr(lupine.sys, "platform", "linux")
    monkeypatch.setattr(
        lupine,
        "_create_sidecar",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert lupine.connect(host="host-a", sidecar=True) is sentinel
    assert calls == [{"server": "host-a:14833"}]


def test_connect_sidecar_false_disables_auto_detection(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    fake_torch.version.cuda = None
    monkeypatch.setattr(lupine.sys, "platform", "darwin")
    monkeypatch.setattr(
        lupine,
        "_create_sidecar",
        lambda **kwargs: pytest.fail("sidecar was unexpectedly selected"),
    )

    with pytest.raises(lupine.LupineError, match="sidecar=False"):
        lupine.connect(host="host-a", sidecar=False)


def test_connect_auto_prefers_native_cuda_when_available(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.setattr(lupine.sys, "platform", "darwin")
    monkeypatch.setattr(
        lupine,
        "_create_sidecar",
        lambda **kwargs: pytest.fail("sidecar was unexpectedly selected"),
    )

    assert isinstance(lupine.connect(host="host-a"), lupine.Session)


def test_connect_sidecar_rejects_multiple_hosts(lupine_module):
    lupine, _ = lupine_module

    with pytest.raises(lupine.LupineError, match="exactly one host"):
        lupine.connect(
            host=["host-a:14833", "host-b:14833"],
            sidecar=True,
        )


def test_connect_sidecar_rejects_libcuda(lupine_module, tmp_path):
    lupine, _ = lupine_module

    with pytest.raises(lupine.LupineError, match="libcuda"):
        lupine.connect(
            host="host-a",
            libcuda=tmp_path / "libcuda.so.1",
            sidecar=True,
        )


def test_connect_sidecar_requires_host(lupine_module):
    lupine, _ = lupine_module

    with pytest.raises(lupine.LupineError, match="exactly one host"):
        lupine.connect(host=[], sidecar=True)


def test_connect_rejects_invalid_sidecar_option(lupine_module):
    lupine, _ = lupine_module

    with pytest.raises(TypeError, match="True, False, or None"):
        lupine.connect(host="host-a", sidecar="auto")


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


def test_bare_ipv6_host_gets_bracketed(lupine_module):
    lupine, _ = lupine_module

    assert lupine._normalize_server("::1") == "[::1]:14833"
    assert lupine._normalize_server("::1", 14833) == "[::1]:14833"
    assert lupine._normalize_server("2001:db8::1") == "[2001:db8::1]:14833"
    assert lupine._normalize_server("[::1]") == "[::1]:14833"
    assert lupine._normalize_server("[::1]:14833") == "[::1]:14833"


def test_explicit_port_replaces_the_host_port(lupine_module):
    lupine, _ = lupine_module

    assert lupine._normalize_server("host-a", 9443) == "host-a:9443"
    assert lupine._normalize_server("host-a:14833", 9443) == "host-a:9443"
    assert lupine._normalize_server("[::1]:14833", 9443) == "[::1]:9443"

    with lupine.connect(host="host-a:14833", port=9443) as session:
        assert session.servers == ("host-a:9443",)


def test_url_server_endpoint_is_preserved(lupine_module):
    lupine, _ = lupine_module

    assert lupine._normalize_server("https://host-a:9443") == "https://host-a:9443"
    assert lupine._normalize_server("https://host-a", 9443) == "https://host-a:9443"
    assert lupine._normalize_server("https://host-a") == "https://host-a"
    assert lupine._normalize_server("http://host-a") == "http://host-a:14833"


def test_connect_defaults_host_to_server_env(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    fake_torch.cuda.count = 1
    monkeypatch.setenv("LUPINE_SERVER", "host-a:14833")

    with lupine.connect() as session:
        assert session.servers == ("host-a:14833",)


def test_connect_defaults_to_every_host_in_server_env(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.setenv("LUPINE_SERVER", "host-a:14833,host-b:14833")

    with lupine.connect() as session:
        assert session.servers == ("host-a:14833", "host-b:14833")


def test_connect_without_host_requires_server_env(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.delenv("LUPINE_SERVER", raising=False)

    with pytest.raises(lupine.LupineError, match="pass host=... or set LUPINE_SERVER"):
        lupine.connect()
