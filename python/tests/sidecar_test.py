import gc
import subprocess
import sys
import threading

import pytest
from packaging.version import Version

torch = pytest.importorskip("torch")
sidecar = pytest.importorskip("lupine.sidecar")
tensor_support = pytest.importorskip("lupine.tensor")


def _mock_httpx_client(monkeypatch, headers):
    calls = {}

    class FakeResponse:
        def __init__(self):
            self.headers = sidecar.httpx.Headers(headers)

        @staticmethod
        def raise_for_status():
            return None

    class FakeClient:
        def __init__(self, **options):
            calls["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def head(self, url, headers=None):
            calls["url"] = url
            calls["headers"] = dict(headers or {})
            return FakeResponse()

    monkeypatch.setattr(sidecar.httpx, "Client", FakeClient)
    return calls


def test_sidecar_queries_plaintext_server_with_http2_prior_knowledge(monkeypatch):
    calls = _mock_httpx_client(monkeypatch, {"x-lupine-cuda-version": "12.9.86"})

    version = sidecar._server_cuda_version("host-a:14833")

    assert version == Version("12.9.86")
    assert calls["url"] == "http://host-a:14833/"
    assert calls["options"]["http1"] is False
    assert calls["options"]["http2"] is True
    assert calls["options"]["timeout"].connect == 5
    assert calls["options"]["timeout"].read == 10


def test_sidecar_queries_https_server_with_http2(monkeypatch):
    calls = _mock_httpx_client(monkeypatch, {"X-Lupine-Cuda-Version": "13.1"})

    assert sidecar._server_cuda_version("https://host-a:14833") == Version("13.1")
    assert calls["url"] == "https://host-a:14833/"


def test_sidecar_requires_server_cuda_version_header(monkeypatch):
    _mock_httpx_client(monkeypatch, {})

    with pytest.raises(sidecar.SidecarError, match="pass image= explicitly"):
        sidecar._server_cuda_version("host-a:14833")


def test_sidecar_queries_registry_with_httpx(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"token": "anonymous-token"}

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(sidecar.httpx, "get", fake_get)
    assert sidecar._registry_json("/token") == {"token": "anonymous-token"}
    assert calls == [
        (
            "https://ghcr.io/token",
            {
                "headers": None,
                "timeout": 10,
            },
        )
    ]


def test_sidecar_discovers_cuda_worker_images_from_registry(monkeypatch):
    calls = []

    def fake_registry_json(path, headers=None):
        calls.append((path, headers))
        if path.startswith("/token?"):
            return {"token": "anonymous-token"}
        return {
            "tags": [
                "cuda-12.8.1",
                "cuda-13.1.0",
                "cuda-13.1.0-amd64",
                "cuda-13.0",
                "latest",
            ]
        }

    monkeypatch.setattr(sidecar, "_registry_json", fake_registry_json)

    assert sidecar._worker_images() == (
        (
            Version("13.1.0"),
            "ghcr.io/lupinemachines/lupine-pytorch-worker:cuda-13.1.0",
        ),
        (
            Version("12.8.1"),
            "ghcr.io/lupinemachines/lupine-pytorch-worker:cuda-12.8.1",
        ),
    )
    assert calls[0][0].startswith("/token?")
    assert calls[1] == (
        "/v2/lupinemachines/lupine-pytorch-worker/tags/list?n=1000",
        {"Authorization": "Bearer anonymous-token"},
    )


@pytest.mark.parametrize(
    ("server_version", "image"),
    [
        ("13.1.80", "cuda-13.1.0"),
        ("13.0.96", "cuda-13.0.2"),
        ("12.9.86", "cuda-12.9.1"),
        ("12.8.93", "cuda-12.8.1"),
        ("12.7.0", "cuda-12.6.2"),
    ],
)
def test_sidecar_selects_newest_compatible_worker(monkeypatch, server_version, image):
    monkeypatch.setattr(
        sidecar, "_server_cuda_version", lambda server: Version(server_version)
    )
    monkeypatch.setattr(
        sidecar,
        "_worker_images",
        lambda: tuple(
            (
                Version(version),
                f"ghcr.io/lupinemachines/lupine-pytorch-worker:cuda-{version}",
            )
            for version in ("13.1.0", "13.0.2", "12.9.1", "12.8.1", "12.6.2")
        ),
    )

    selected = sidecar._worker_image_for_server("host-a:14833")

    assert selected.endswith(image)


def test_sidecar_rejects_server_older_than_published_workers(monkeypatch):
    monkeypatch.setattr(
        sidecar, "_server_cuda_version", lambda server: Version("12.5.82")
    )
    monkeypatch.setattr(
        sidecar,
        "_worker_images",
        lambda: (
            (
                Version("12.6.2"),
                "ghcr.io/lupinemachines/lupine-pytorch-worker:cuda-12.6.2",
            ),
        ),
    )

    with pytest.raises(sidecar.SidecarError, match="pass image= explicitly"):
        sidecar._worker_image_for_server("host-a:14833")


def test_sidecar_explicit_image_skips_server_probe(monkeypatch):
    session = sidecar.SidecarSession(
        server="host-a:14833",
        image="registry.example/worker:custom",
    )
    monkeypatch.setattr(
        sidecar,
        "_worker_image_for_server",
        lambda server: pytest.fail("explicit image unexpectedly probed server"),
    )

    assert session._worker_image() == "registry.example/worker:custom"


def test_sidecar_stops_the_worker_when_startup_fails(monkeypatch):
    tensor_support._ensure_registered()
    session = sidecar.SidecarSession(
        server="host-a:14833",
        image="registry.example/worker:custom",
    )
    workers = []

    class Launcher:
        @staticmethod
        def command(script):
            return [sys.executable, "-c", "import sys; sys.stdin.read()"]

    def without_cuda(self, payload, input_tensors=(), *, decode_result=False):
        workers.append(self._proc)
        return {"cuda_available": False}

    monkeypatch.setattr(sidecar, "prepare_runtime", lambda *args, **kwargs: Launcher())
    monkeypatch.setattr(sidecar.SidecarSession, "_request", without_cuda)

    with pytest.raises(sidecar.SidecarError, match="no CUDA device"):
        with session:
            pytest.fail("session unexpectedly started without CUDA")

    assert workers[0].poll() is not None
    assert session._proc is None
    assert tensor_support._get_active_session() is None


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


def test_sidecar_releases_handles_of_collected_tensors(monkeypatch):
    tensor_support._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = object()
    sent = []

    def fake_send(payload, input_tensors=(), *, decode_result=False):
        sent.append(payload)
        return {"torch": "fake"}

    monkeypatch.setattr(session, "_send", fake_send)
    tensor = tensor_support.SidecarTensor(
        session=session,
        handle=7,
        shape=(2,),
        dtype=torch.float32,
        device=session.device(),
    )

    del tensor
    gc.collect()
    session._request({"op": "ping"})

    assert sent == [
        {"op": "release", "value": [{"__sidecar_tensor__": 7}]},
        {"op": "ping"},
    ]
    assert session._pending_release == []


def test_sidecar_keeps_handles_of_live_tensors(monkeypatch):
    tensor_support._ensure_registered()
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = object()
    sent = []
    monkeypatch.setattr(
        session,
        "_send",
        lambda payload, *args, **kwargs: sent.append(payload),
    )
    tensor = tensor_support.SidecarTensor(
        session=session,
        handle=7,
        shape=(2,),
        dtype=torch.float32,
        device=session.device(),
    )

    gc.collect()
    session._request({"op": "ping"})

    assert sent == [{"op": "ping"}]
    assert tensor._lupine_handle == 7


def test_sidecar_keeps_worker_after_unserializable_argument():
    tensor_support._ensure_registered()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", sidecar._worker_source()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = proc
    session._lock = threading.Lock()
    try:
        with pytest.raises(sidecar.SidecarError, match="cannot send this operation"):
            session.forward(torch.ops.aten.full.default, ((2,), 1 + 2j), {})

        assert "torch" in session._request({"op": "ping"})
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


def test_sidecar_worker_survives_an_unserializable_result():
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", sidecar._worker_source()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    session = sidecar.SidecarSession(server="host-a:14833")
    session._proc = proc
    session._lock = threading.Lock()
    try:
        with pytest.raises(sidecar.SidecarError, match="JSON serializable"):
            session.forward(torch.ops.aten.item.default, (torch.tensor(1 + 2j),), {})

        assert "torch" in session._request({"op": "ping"})
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5)


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


def test_sidecar_probe_sends_lease_header(monkeypatch):
    monkeypatch.setenv("LUPINE_SESSION", "lease-123")
    calls = _mock_httpx_client(monkeypatch, {"x-lupine-cuda-version": "13.1.80"})

    assert sidecar._server_cuda_version("https://gw:9443") == Version("13.1.80")
    assert calls["headers"] == {"x-lupine-session": "lease-123"}


def test_sidecar_probe_omits_lease_header_when_unset(monkeypatch):
    monkeypatch.delenv("LUPINE_SESSION", raising=False)
    calls = _mock_httpx_client(monkeypatch, {"x-lupine-cuda-version": "13.1.80"})

    sidecar._server_cuda_version("https://gw:9443")
    assert calls["headers"] == {}
