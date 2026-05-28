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
        self.current = 1
        self.synchronized = None

    def is_initialized(self):
        return self.initialized

    def is_available(self):
        return self.count > 0

    def device_count(self):
        return self.count

    def current_device(self):
        return self.current

    def synchronize(self, selected):
        self.synchronized = selected


class FakeTorch(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.cuda = FakeCuda()

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
    lupine, _ = lupine_module

    with lupine.connect(host="host-a") as session:
        assert os.environ["LUPINE_SERVER"] == "host-a:14833"
        assert session.devices() == [FakeDevice("cuda", 0)]
        assert session.device() == FakeDevice("cuda", 0)


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
        assert session.devices() == [FakeDevice("cuda", 0), FakeDevice("cuda", 1)]
        assert session.device(1) == FakeDevice("cuda", 1)


def test_connect_restores_env_when_cuda_was_not_initialized(lupine_module, monkeypatch):
    lupine, _ = lupine_module

    with lupine.connect(host="host-a"):
        assert os.environ["LUPINE_SERVER"] == "host-a:14833"

    assert "LUPINE_SERVER" not in os.environ


def test_connect_leaves_env_when_cuda_initialized_inside_context(lupine_module):
    lupine, fake_torch = lupine_module

    with lupine.connect(host="host-a"):
        fake_torch.cuda.initialized = True

    assert os.environ["LUPINE_SERVER"] == "host-a:14833"


def test_connect_accepts_matching_preconfigured_env(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.setenv("LUPINE_SERVER", "host-a:14833")

    with lupine.connect(host="host-a:14833") as session:
        assert session.devices() == [FakeDevice("cuda", 0)]


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


def test_devices_use_current_env(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    monkeypatch.setenv("LUPINE_SERVER", "host-a:14833,host-b:14833")

    assert lupine.devices() == [FakeDevice("cuda", 0), FakeDevice("cuda", 1)]
    assert lupine.device(1) == FakeDevice("cuda", 1)


def test_device_bounds_check(lupine_module):
    lupine, _ = lupine_module

    with lupine.connect(host="host-a") as session:
        with pytest.raises(lupine.LupineError, match="out of range"):
            session.device(1)


def test_duplicate_hosts_are_rejected(lupine_module):
    lupine, _ = lupine_module

    with pytest.raises(lupine.LupineError, match="unique"):
        lupine.connect(host=["host-a:14833", "host-a"])


def test_remote_container_runtime_defaults_to_arm64(monkeypatch):
    import lupine.remote as remote

    monkeypatch.setattr(remote.shutil, "which", lambda name: "/usr/bin/container")
    monkeypatch.setattr(remote.sys, "platform", "darwin")

    cmd = remote.ContainerRuntime(server="host-a:14833").command("print(1)")

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


def test_remote_container_runtime_is_macos_only(monkeypatch):
    import lupine.remote as remote

    monkeypatch.setattr(remote.sys, "platform", "linux")

    with pytest.raises(remote.RemoteError, match="only supported on macOS"):
        remote.ContainerRuntime(server="host-a:14833").command("print(1)")


def test_remote_container_runtime_requires_cli(monkeypatch):
    import lupine.remote as remote

    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    monkeypatch.setattr(remote.sys, "platform", "darwin")

    with pytest.raises(remote.RemoteError, match="brew install --cask container"):
        remote.ContainerRuntime(server="host-a:14833").command("print(1)")


def test_remote_container_runtime_starts_services_and_pulls_missing_image(monkeypatch):
    import lupine.remote as remote

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["system", "status", "--format"]:
            return remote.subprocess.CompletedProcess(args, 0, '{"status":"stopped"}', "")
        return remote.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(remote.shutil, "which", lambda name: "/usr/bin/container")
    monkeypatch.setattr(remote.sys, "platform", "darwin")
    monkeypatch.setattr(remote.subprocess, "run", fake_run)

    remote.ContainerRuntime(server="host-a:14833").prepare()

    assert calls == [
        ["/usr/bin/container", "system", "status", "--format", "json"],
        ["/usr/bin/container", "system", "start"],
        ["/usr/bin/container", "image", "inspect", remote.DEFAULT_IMAGE],
    ]


def test_remote_container_runtime_pulls_missing_image(monkeypatch):
    import lupine.remote as remote

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:3] == ["image", "inspect"]:
            return remote.subprocess.CompletedProcess(args, 1, "", "missing")
        return remote.subprocess.CompletedProcess(args, 0, '{"status":"running"}', "")

    monkeypatch.setattr(remote.shutil, "which", lambda name: "/usr/bin/container")
    monkeypatch.setattr(remote.sys, "platform", "darwin")
    monkeypatch.setattr(remote.subprocess, "run", fake_run)

    remote.ContainerRuntime(server="host-a:14833").prepare()

    assert calls[-1] == [
        "/usr/bin/container",
        "image",
        "pull",
        "--progress",
        "none",
        "--platform",
        "linux/arm64",
        remote.DEFAULT_IMAGE,
    ]


def test_remote_dispatch_mode_forwards_factory_ops(monkeypatch):
    import torch
    import lupine.remote as remote

    remote._ensure_registered()
    session = remote.RemoteSession(server="host-a:14833")
    calls = []

    def fake_request(payload):
        calls.append(payload)
        return {"type": "tensor", "handle": 1, "shape": [2, 3], "dtype": "float32"}

    monkeypatch.setattr(session, "_request", fake_request)

    with remote.RemoteDispatchMode(session):
        tensor = torch.zeros((2, 3), device=session.device(), dtype=torch.float32)

    assert isinstance(tensor, remote.RemoteTensor)
    assert calls[0]["op"] == "call"
    assert calls[0]["packet"] == "zeros"
    assert calls[0]["kwargs"]["device"] == {"__device__": "cuda:0"}
    assert calls[0]["kwargs"]["dtype"] == {"__dtype__": "float32"}


def test_remote_dispatch_mode_forwards_tensor_ops(monkeypatch):
    import torch
    import lupine.remote as remote

    remote._ensure_registered()
    session = remote.RemoteSession(server="host-a:14833")
    calls = []

    def fake_request(payload):
        calls.append(payload)
        return {"type": "tensor", "handle": 2, "shape": [2], "dtype": "float32"}

    monkeypatch.setattr(session, "_request", fake_request)
    tensor = remote.RemoteTensor(
        session=session,
        handle=1,
        shape=(2,),
        dtype=torch.float32,
        device=session.device(),
    )

    with remote.RemoteDispatchMode(session):
        result = tensor + 3

    assert isinstance(result, remote.RemoteTensor)
    assert calls[0]["packet"] == "add"
    assert calls[0]["overload"] == "Tensor"
    assert calls[0]["args"]["__tuple__"][0] == {"__remote_tensor__": 1}
