import subprocess
import sys
import threading

import pytest

torch = pytest.importorskip("torch")
sidecar = pytest.importorskip("lupine.sidecar")
tensor_support = pytest.importorskip("lupine.tensor")


def test_sidecar_container_runtime_defaults_to_arm64(monkeypatch):
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
    monkeypatch.setattr(sidecar.sys, "platform", "linux")

    with pytest.raises(sidecar.SidecarError, match="only supported on macOS"):
        sidecar.ContainerRuntime(server="host-a:14833").command("print(1)")


def test_sidecar_container_runtime_requires_cli(monkeypatch):
    monkeypatch.setattr(sidecar.shutil, "which", lambda name: None)
    monkeypatch.setattr(sidecar.sys, "platform", "darwin")

    with pytest.raises(sidecar.SidecarError, match="brew install --cask container"):
        sidecar.ContainerRuntime(server="host-a:14833").command("print(1)")


def test_sidecar_container_runtime_starts_services_and_pulls_missing_image(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1:4] == ["system", "status", "--format"]:
            return sidecar.subprocess.CompletedProcess(
                args, 0, '{"status":"stopped"}', ""
            )
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
    tensor_support._ensure_registered()
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

    with tensor_support.SidecarDispatchMode(session):
        tensor = torch.zeros((2, 3), device=session.device(), dtype=torch.float32)

    assert sidecar.SidecarTensor is tensor_support.SidecarTensor
    assert isinstance(tensor, tensor_support.SidecarTensor)
    assert calls[0]["op"] == "call"
    assert calls[0]["packet"] == "zeros"
    assert calls[0]["kwargs"]["device"] == {"__device__": "cuda:0"}
    assert calls[0]["kwargs"]["dtype"] == {"__dtype__": "float32"}


def test_sidecar_dispatch_mode_forwards_tensor_ops(monkeypatch):
    tensor_support._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    calls = []

    def fake_request(payload, input_tensors=(), *, decode_result=False):
        calls.append(payload)
        result = {"type": "tensor", "handle": 2, "shape": [2], "dtype": "float32"}
        return session._decode(result, []) if decode_result else result

    monkeypatch.setattr(session, "_request", fake_request)
    tensor = tensor_support.SidecarTensor(
        session=session,
        handle=1,
        shape=(2,),
        dtype=torch.float32,
        device=session.device(),
    )

    with tensor_support.SidecarDispatchMode(session):
        result = tensor + 3

    assert isinstance(result, tensor_support.SidecarTensor)
    assert calls[0]["packet"] == "add"
    assert calls[0]["overload"] == "Tensor"
    assert calls[0]["args"]["__tuple__"][0] == {"__sidecar_tensor__": 1}


def test_sidecar_dispatch_mode_keeps_cpu_ops_local(monkeypatch):
    session = sidecar.SidecarSession(server="host-a:14833")
    requests = []
    monkeypatch.setattr(
        session, "_request", lambda *args, **kwargs: requests.append(args)
    )
    tensor = torch.arange(8)

    with tensor_support.SidecarDispatchMode(session):
        result = tensor + 1

    assert torch.equal(result, torch.arange(1, 9))
    assert requests == []


def test_sidecar_to_copy_uses_direct_cpu_upload(monkeypatch):
    tensor_support._ensure_registered()
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
    tensor_support._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    source = tensor_support.SidecarTensor(
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
    tensor_support._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    destination = tensor_support.SidecarTensor(
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


def test_sidecar_worker_consumes_tensor_stream_before_operation_error(tmp_path):
    worker_source = sidecar._worker_source()
    isolated_source = (
        "import importlib.abc\n"
        "import sys\n\n"
        "class RejectLupineImports(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'lupine' or fullname.startswith('lupine.'):\n"
        "            raise ImportError(f'unexpected external import: {fullname}')\n"
        "        return None\n\n"
        "sys.meta_path.insert(0, RejectLupineImports())\n"
        f"exec(compile({worker_source!r}, '<worker-bootstrap>', 'exec'))\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", isolated_source],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
    )
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = proc
    session._lock = threading.Lock()
    tensor = torch.arange(
        2 * tensor_support._TENSOR_CHUNK_BYTES + 37, dtype=torch.uint8
    )
    streams = [(tensor, tensor_support._cpu_tensor_metadata(tensor))]
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
