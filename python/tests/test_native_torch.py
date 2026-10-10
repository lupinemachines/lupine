"""Real native PyTorch/GPU tests, opt-in on the ARM64 Mac with an L4 server.

Run this file separately so Lupine can install its CUDA hooks before torch
is imported. The normal unit suite skips it and needs no GPU/compiler.
"""

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("LUPINE_TEST_NATIVE_TORCH") != "1",
    reason="requires an ARM64 Mac, native runtime bundle, and a GPU server",
)


@pytest.fixture(scope="module")
def backend():
    import lupine
    from lupine import _backend

    assert os.environ.get("LUPINE_TORCH_BACKEND") == "native"
    session = lupine.connect()
    import torch

    assert torch.version.cuda is None  # native macOS torch, not a converted CUDA wheel
    assert session.device() == torch.device("lupine:0")
    native = _backend._started["backend"]
    assert native.native and _backend._started["address"] == "native"
    yield torch, native
    torch.lupine.synchronize()


def test_arange_and_scalar_arithmetic(backend):
    torch, native = backend
    before = native.runtime.launches
    x = torch.arange(8, device="lupine", dtype=torch.float32)
    torch.testing.assert_close((x * 2).cpu(), torch.arange(8, dtype=torch.float32) * 2)
    assert native.runtime.launches >= before + 2
    assert torch.lupine.get_device_name() == "NVIDIA L4"
    assert torch.lupine.get_device_capability() == (8, 9)


def test_strided_broadcast_and_inplace(backend):
    torch, native = backend
    host = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    x = host.to("lupine")
    y = torch.tensor([1.0, 2.0, 3.0], device="lupine")
    torch.testing.assert_close((x.T + y).cpu(), host.T + torch.tensor([1.0, 2.0, 3.0]))
    view = x[:, ::2]
    view.mul_(3)
    host[:, ::2].mul_(3)
    torch.testing.assert_close(x.cpu(), host)
    torch.testing.assert_close(view.clone().cpu(), host[:, ::2])


def test_int64_transfer_and_arange(backend):
    torch, native = backend
    host = torch.tensor([2**60, -(2**60), 7], dtype=torch.int64)
    torch.testing.assert_close(host.to("lupine").cpu(), host)
    torch.testing.assert_close(
        torch.arange(9, 1, -2, device="lupine").cpu(), torch.arange(9, 1, -2)
    )


def test_matmul_and_autograd(backend):
    torch, native = backend
    a_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    b_cpu = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
    a = a_cpu.detach().to("lupine").requires_grad_()
    b = b_cpu.detach().to("lupine").requires_grad_()
    before = native.runtime.launches
    output = a @ b
    loss = (output * output).sum()
    loss.backward()
    expected = a_cpu @ b_cpu
    (expected * expected).sum().backward()
    torch.testing.assert_close(output.cpu(), expected.detach())
    torch.testing.assert_close(loss.cpu(), (expected * expected).sum().detach())
    torch.testing.assert_close(a.grad.cpu(), a_cpu.grad)
    torch.testing.assert_close(b.grad.cpu(), b_cpu.grad)
    assert native.runtime.launches > before


def test_reductions_over_strided_views(backend):
    torch, native = backend
    cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 2)
    gpu = cpu.to("lupine")
    torch.testing.assert_close(
        gpu.sum(dim=(0, 2), keepdim=True).cpu(), cpu.sum(dim=(0, 2), keepdim=True)
    )
    torch.testing.assert_close(gpu.mean(dim=1).cpu(), cpu.mean(dim=1))
    assert gpu.sum().item() == cpu.sum().item()


def test_unsupported_operator_is_explicit(backend):
    torch, native = backend
    x = torch.ones(4, device="lupine")
    with pytest.raises(NotImplementedError, match="not implemented.*sin"):
        torch.sin(x)


def test_storage_lifetime_tracks_views(backend):
    import gc

    torch, native = backend
    gc.collect()
    baseline = torch.lupine.memory_allocated()
    x = torch.ones(8, device="lupine")
    view = x[::2]
    assert torch.lupine.memory_allocated() == baseline + 32
    del x
    gc.collect()
    torch.testing.assert_close(view.cpu(), torch.ones(4))
    del view
    gc.collect()
    assert torch.lupine.memory_allocated() == baseline


def test_overlapping_writes_are_rejected(backend):
    torch, native = backend
    x = torch.ones(8, device="lupine")
    with pytest.raises(NotImplementedError, match="aliased views"):
        x[1:].copy_(x[:-1])
    with pytest.raises(ValueError, match="overlapping"):
        x[:1].expand(4).copy_(torch.ones(4))


def test_empty_and_nonfinite_fill(backend):
    torch, native = backend
    result = torch.full((2,), float("nan"), device="lupine")
    torch.testing.assert_close(
        result.cpu(), torch.full((2,), float("nan")), equal_nan=True
    )
    empty = torch.arange(0, device="lupine", dtype=torch.float32)
    assert empty.numel() == 0 and empty.cpu().numel() == 0
    assert empty.sum().item() == 0
    with pytest.raises(ValueError, match="inconsistent"):
        torch.arange(1, 4, -1, device="lupine")
