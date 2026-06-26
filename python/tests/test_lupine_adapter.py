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
        self.synchronize_calls = 0

    def is_initialized(self):
        return self.initialized

    def is_available(self):
        return self.count > 0

    def device_count(self):
        return self.count

    def current_device(self):
        return self.current

    def synchronize(self, selected=None):
        self.synchronize_calls += 1
        self.synchronized = selected


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


def test_connect_leaves_env_when_cuda_initialized_inside_context(lupine_module):
    lupine, fake_torch = lupine_module

    with lupine.connect(host="host-a"):
        fake_torch.cuda.initialized = True

    assert os.environ["LUPINE_SERVER"] == "host-a:14833"


def test_connect_loads_snapshot_on_enter(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    snapshot_id = "0123456789abcdef0123456789abcdef"
    loaded = []
    saved = []
    cleared = []
    monkeypatch.setattr(lupine, "load_snapshot", lambda value: loaded.append(value))
    monkeypatch.setattr(lupine, "save_snapshot_and_exit", lambda value: saved.append(value))
    monkeypatch.setattr(lupine, "_clear_snapshot_load", lambda: cleared.append(True))

    with lupine.connect(host="host-a", snapshot_id=snapshot_id):
        assert loaded == [snapshot_id]
        assert "LUPINE_SNAPSHOT_ID" not in os.environ

    assert "LUPINE_SNAPSHOT_ID" not in os.environ
    assert saved == [snapshot_id]
    assert cleared == [True]


def test_connect_snapshot_exit_saves(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    snapshot_id = "0123456789abcdef0123456789abcdef"
    saved = []
    monkeypatch.setattr(lupine, "load_snapshot", lambda value: None)
    monkeypatch.setattr(lupine, "save_snapshot_and_exit", lambda value: saved.append(value))
    monkeypatch.setattr(lupine, "_clear_snapshot_load", lambda: None)

    with lupine.connect(host="host-a", snapshot_id=snapshot_id):
        fake_torch.cuda.initialized = True

    assert saved == [snapshot_id]
    assert "LUPINE_SNAPSHOT_ID" not in os.environ


def test_connect_rejects_invalid_snapshot_type(lupine_module):
    lupine, _ = lupine_module
    snapshot_id = "0123456789abcdef0123456789abcdef"

    with pytest.raises(lupine.LupineError, match="snapshot_type"):
        lupine.connect(host="host-a", snapshot_id=snapshot_id, snapshot_type="x")


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


class FakeSnapshotFunc:
    def __init__(self, fn):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.fn(*args)


class FakeSnapshotLib:
    def __init__(self):
        self.loaded = []
        self.lupine_snapshot_save_and_exit = FakeSnapshotFunc(self._save)
        self.lupine_snapshot_load = FakeSnapshotFunc(self._load)

    def _load(self, snapshot_id):
        self.loaded.append(None if snapshot_id is None else snapshot_id.decode())
        return 0

    def _save(self, snapshot_id):
        self.saved = snapshot_id.decode()
        return 0


def test_load_snapshot_calls_c_export(lupine_module, monkeypatch):
    lupine, _ = lupine_module
    lib = FakeSnapshotLib()
    monkeypatch.setattr(lupine, "_snapshot_lib", lambda: lib)
    snapshot_id = "0123456789abcdef0123456789abcdef"

    lupine.load_snapshot(snapshot_id)

    assert lib.loaded == [snapshot_id]


def test_save_snapshot_and_exit_calls_c_export(lupine_module, monkeypatch):
    lupine, fake_torch = lupine_module
    lib = FakeSnapshotLib()
    monkeypatch.setattr(lupine, "_snapshot_lib", lambda: lib)
    snapshot_id = "0123456789abcdef0123456789abcdef"

    lupine.save_snapshot_and_exit(snapshot_id)

    assert fake_torch.cuda.synchronize_calls == 1
    assert fake_torch.cuda.synchronized is None
    assert lib.saved == snapshot_id


def test_snapshot_id_allows_arbitrary_strings(lupine_module, monkeypatch):
    # Any non-empty, bounded string is accepted; the server hashes it into the
    # on-disk path, so values like "../bad" or "my model:v2" are fine.
    lupine, _ = lupine_module
    lib = FakeSnapshotLib()
    monkeypatch.setattr(lupine, "_snapshot_lib", lambda: lib)

    for ok_id in ("../bad", "my model:v2", "x"):
        lupine.load_snapshot(ok_id)
        assert lib.loaded[-1] == ok_id


def test_snapshot_id_rejects_empty_and_too_long(lupine_module):
    lupine, _ = lupine_module

    for bad_id in ("", "a" * 256):
        with pytest.raises(lupine.LupineError, match="snapshot id"):
            lupine.load_snapshot(bad_id)


def test_device_bounds_check(lupine_module):
    lupine, _ = lupine_module

    with lupine.connect(host="host-a") as session:
        with pytest.raises(lupine.LupineError, match="out of range"):
            session.device(1)


def test_duplicate_hosts_are_rejected(lupine_module):
    lupine, _ = lupine_module

    with pytest.raises(lupine.LupineError, match="unique"):
        lupine.connect(host=["host-a:14833", "host-a"])


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

    def fake_request(payload):
        calls.append(payload)
        return {"type": "tensor", "handle": 1, "shape": [2, 3], "dtype": "float32"}

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

    def fake_request(payload):
        calls.append(payload)
        return {"type": "tensor", "handle": 2, "shape": [2], "dtype": "float32"}

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
