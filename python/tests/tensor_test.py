import subprocess
import sys
import threading
from io import BytesIO

import pytest

torch = pytest.importorskip("torch")
sidecar = pytest.importorskip("lupine.sidecar")
tensor_support = pytest.importorskip("lupine.tensor")


def test_cpu_tensor_rejects_noncontiguous_input():
    value = torch.arange(12).reshape(3, 4).transpose(0, 1)

    with pytest.raises(tensor_support.SidecarError, match="must be contiguous"):
        tensor_support._cpu_tensor_metadata(value)


@pytest.mark.parametrize(
    "value, error",
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
def test_cpu_tensor_rejects_lazy_views(value, error):
    with pytest.raises(tensor_support.SidecarError, match=error):
        tensor_support._cpu_tensor_metadata(value(torch))


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


def test_cpu_tensor_transport_uses_bounded_binary_chunks():
    value = torch.arange(2 * tensor_support._TENSOR_CHUNK_BYTES + 37, dtype=torch.uint8)
    stream = BytesIO()
    writer = _ShortWriter(stream, limit=997)

    tensor_support._write_tensor(writer, value)
    stream.seek(0)
    reader = _ShortReader(stream, limit=991)
    result = tensor_support._decode_cpu_tensor(
        tensor_support._cpu_tensor_metadata(value), reader
    )

    assert torch.equal(result, value)
    assert max(writer.requests) <= tensor_support._TENSOR_CHUNK_BYTES
    assert max(reader.requests) <= tensor_support._TENSOR_CHUNK_BYTES
    assert len(writer.requests) > 3
    assert len(reader.requests) > 3


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(lambda torch: torch.tensor([1.25, -2.5]), id="float32"),
        pytest.param(lambda torch: torch.tensor([1, -2, 2**40]), id="int64"),
        pytest.param(lambda torch: torch.tensor([True, False]), id="bool"),
        pytest.param(lambda torch: torch.tensor([1 + 2j, -3 + 0.5j]), id="complex64"),
        pytest.param(
            lambda torch: torch.tensor([1.5, -2.25], dtype=torch.bfloat16),
            id="bfloat16",
        ),
        pytest.param(lambda torch: torch.empty((0, 3)), id="empty"),
        pytest.param(lambda torch: torch.tensor(7.5), id="scalar"),
    ],
)
def test_cpu_tensor_binary_round_trip(value):
    expected = value(torch)
    stream = BytesIO()
    tensor_support._write_tensor(stream, expected)
    stream.seek(0)

    result = tensor_support._decode_cpu_tensor(
        tensor_support._cpu_tensor_metadata(expected), stream
    )

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert torch.equal(result, expected)


def test_worker_streams_cpu_cuda_transfers_in_chunks():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
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
    source = torch.arange(
        tensor_support._TENSOR_CHUNK_BYTES // 4 + 37,
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
