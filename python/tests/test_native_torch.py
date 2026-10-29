"""Real native PyTorch/GPU tests, opt-in on the ARM64 Mac with an L4 server.

Run this file separately so Lupine can install its CUDA hooks before torch
is imported. The normal unit suite skips it and needs no GPU/compiler.
"""

import os
import math
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("LUPINE_TEST_NATIVE_TORCH") != "1",
    reason="requires an ARM64 Mac, native runtime bundle, and a GPU server",
)


@pytest.fixture(scope="module")
def backend():
    import lupine._backend

    session = lupine.connect()
    import torch

    assert torch.version.cuda is None  # native macOS torch, not a converted CUDA wheel
    assert session.device() == torch.device("lupine:0")
    native = lupine._backend._started["backend"]
    assert native.native and lupine._backend._started["address"] == "native"
    if os.environ.get("LUPINE_TEST_NATIVE_LIBRARIES") == "1":
        assert native.runtime.operators.library.lupine_aten_run is not None
        assert native.runtime.blas is not None
    yield torch, native
    torch.lupine.synchronize()
    lupine._backend.close()
    assert native.runtime.closed and native.runtime._cudnn is None
    assert native.runtime._operators is None


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
    with pytest.raises(NotImplementedError, match="not implemented.*sort"):
        torch.sort(x)


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


def test_unary_functions_and_activation_gradients(backend):
    torch, native = backend
    cpu = torch.linspace(0.1, 0.9, 12).reshape(3, 4)
    gpu = cpu.to("lupine")
    for op in (torch.exp, torch.log, torch.sqrt, torch.sin, torch.cos, torch.abs):
        torch.testing.assert_close(op(gpu).cpu(), op(cpu), rtol=2e-5, atol=2e-6)
    for op in (torch.relu, torch.sigmoid, torch.tanh):
        reference = (cpu - 0.4).detach().requires_grad_()
        value = reference.detach().to("lupine").requires_grad_()
        op(value).sum().backward()
        op(reference).sum().backward()
        torch.testing.assert_close(
            value.grad.cpu(), reference.grad, rtol=2e-5, atol=2e-6
        )


def test_seeded_random_values_are_generated_on_gpu(backend):
    torch, native = backend
    torch.manual_seed(42)
    first = torch.rand(1024, device="lupine")
    second = torch.rand(1024, device="lupine")
    torch.manual_seed(42)
    repeat = torch.rand(1024, device="lupine")
    torch.testing.assert_close(first.cpu(), repeat.cpu())
    assert not torch.equal(first.cpu(), second.cpu())
    values = first.cpu()
    assert (values >= 0).all() and (values < 1).all()
    assert abs(values.mean().item() - 0.5) < 0.05
    normal = torch.randn(2048, device="lupine").cpu()
    assert abs(normal.mean().item()) < 0.1
    assert abs(normal.std().item() - 1) < 0.1


@pytest.mark.parametrize("logarithmic", [False, True])
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_softmax_forward_and_backward(backend, logarithmic, dim):
    torch, native = backend
    cpu = (
        torch.arange(24, dtype=torch.float32)
        .reshape(2, 3, 4)
        .transpose(0, 1)
        .requires_grad_()
    )
    value = cpu.detach().to("lupine").requires_grad_()
    op = torch.log_softmax if logarithmic else torch.softmax
    output = op(value, dim=dim)
    expected = op(cpu, dim=dim)
    (output * output).sum().backward()
    (expected * expected).sum().backward()
    torch.testing.assert_close(output.cpu(), expected.detach(), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(value.grad.cpu(), cpu.grad, rtol=2e-5, atol=2e-6)
    assert {"cudnnSoftmaxForward", "cudnnSoftmaxBackward"}.issubset(
        native.runtime._cudnn.functions
    )


def test_linear_training_step_and_batched_matmul(backend):
    torch, native = backend
    torch.manual_seed(123)
    host = torch.nn.Linear(4, 2)
    model = torch.nn.Linear(4, 2).to("lupine")
    model.load_state_dict(host.state_dict())
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
    expected = host(x)
    actual = model(x.to("lupine"))
    (actual**2).mean().backward()
    (expected**2).mean().backward()
    torch.testing.assert_close(actual.cpu(), expected.detach())
    torch.testing.assert_close(model.weight.grad.cpu(), host.weight.grad)
    torch.testing.assert_close(model.bias.grad.cpu(), host.bias.grad)
    with torch.no_grad():
        model.weight.add_(model.weight.grad, alpha=-0.01)
        host.weight.add_(host.weight.grad, alpha=-0.01)
    torch.testing.assert_close(model.weight.cpu(), host.weight.detach())
    a = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    b = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    torch.testing.assert_close(
        torch.bmm(a.to("lupine"), b.to("lupine")).cpu(), torch.bmm(a, b)
    )
    torch.testing.assert_close(
        torch.cat([a.to("lupine"), a.to("lupine")], 1).cpu(), torch.cat([a, a], 1)
    )


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 4)])
def test_layer_norm_forward_and_backward(backend, shape):
    torch, native = backend
    cpu = (
        torch.arange(math.prod(shape), dtype=torch.float32)
        .reshape(shape)
        .requires_grad_()
    )
    x = cpu.detach().to("lupine").requires_grad_()
    weight_cpu = torch.linspace(0.5, 1.0, 12).reshape(3, 4).requires_grad_()
    bias_cpu = torch.linspace(-0.2, 0.2, 12).reshape(3, 4).requires_grad_()
    weight = weight_cpu.detach().to("lupine").requires_grad_()
    bias = bias_cpu.detach().to("lupine").requires_grad_()
    expected = torch.nn.functional.layer_norm(cpu, (3, 4), weight_cpu, bias_cpu)
    output = torch.nn.functional.layer_norm(x, (3, 4), weight, bias)
    (output**2).sum().backward()
    (expected**2).sum().backward()
    torch.testing.assert_close(output.cpu(), expected.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(x.grad.cpu(), cpu.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(weight.grad.cpu(), weight_cpu.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(bias.grad.cpu(), bias_cpu.grad, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "groups,stride,dilation", [(1, (1, 1), (1, 1)), (2, (2, 1), (1, 2))]
)
def test_convolution_forward_and_backward(backend, groups, stride, dilation):
    torch, native = backend
    x_cpu = (
        torch.arange(2 * 4 * 5 * 6, dtype=torch.float32)
        .reshape(2, 4, 5, 6)
        .transpose(2, 3)
        / 100
    ).requires_grad_()
    w_cpu = (
        torch.arange(6 * (4 // groups) * 2 * 3, dtype=torch.float32).reshape(
            6, 4 // groups, 2, 3
        )
        / 100
    ).requires_grad_()
    b_cpu = torch.linspace(-0.1, 0.1, 6).requires_grad_()
    x = x_cpu.detach().to("lupine").requires_grad_()
    w = w_cpu.detach().to("lupine").requires_grad_()
    b = b_cpu.detach().to("lupine").requires_grad_()
    expected = torch.nn.functional.conv2d(
        x_cpu, w_cpu, b_cpu, stride, (1, 2), dilation, groups
    )
    output = torch.nn.functional.conv2d(x, w, b, stride, (1, 2), dilation, groups)
    (output**2).mean().backward()
    (expected**2).mean().backward()
    for actual, reference in (
        (output, expected),
        (x.grad, x_cpu.grad),
        (w.grad, w_cpu.grad),
        (b.grad, b_cpu.grad),
    ):
        torch.testing.assert_close(
            actual.cpu(), reference.detach(), rtol=2e-4, atol=2e-5
        )
    assert native.runtime._cudnn is not None
    assert {
        "cudnnConvolutionForward",
        "cudnnConvolutionBackwardData",
        "cudnnConvolutionBackwardFilter",
        "cudnnConvolutionBackwardBias",
    }.issubset(native.runtime._cudnn.functions)


@pytest.mark.parametrize("gradient", ["input", "weight", "bias"])
def test_convolution_selective_gradients(backend, gradient):
    torch, native = backend
    x_cpu = torch.linspace(-1, 1, 32).reshape(1, 2, 4, 4)
    w_cpu = torch.linspace(-0.5, 0.5, 24).reshape(3, 2, 2, 2)
    b_cpu = torch.linspace(-0.1, 0.1, 3) if gradient == "bias" else None
    x, w = x_cpu.to("lupine"), w_cpu.to("lupine")
    b = None if b_cpu is None else b_cpu.to("lupine")
    cpu = {"input": x_cpu, "weight": w_cpu, "bias": b_cpu}[gradient].requires_grad_()
    gpu = {"input": x, "weight": w, "bias": b}[gradient].requires_grad_()
    expected = torch.nn.functional.conv2d(x_cpu, w_cpu, b_cpu)
    actual = torch.nn.functional.conv2d(x, w, b)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(gpu.grad.cpu(), cpu.grad, rtol=2e-4, atol=2e-5)


def test_convolution_empty_batch_gradients(backend):
    torch, native = backend
    x = torch.empty((0, 2, 4, 4), device="lupine", requires_grad=True)
    w = torch.ones((3, 2, 2, 2), device="lupine", requires_grad=True)
    b = torch.ones(3, device="lupine", requires_grad=True)
    out = torch.nn.functional.conv2d(x, w, b)
    assert tuple(out.shape) == (0, 3, 3, 3)
    out.sum().backward()
    assert tuple(x.grad.shape) == tuple(x.shape)
    torch.testing.assert_close(w.grad.cpu(), torch.zeros_like(w.cpu()))
    torch.testing.assert_close(b.grad.cpu(), torch.zeros(3))


def test_matmul_transposed_views_and_empty_inner_dimension(backend):
    torch, native = backend
    a = torch.arange(12, dtype=torch.float32).reshape(4, 3).T
    b = torch.arange(8, dtype=torch.float32).reshape(2, 4).T
    torch.testing.assert_close((a.to("lupine") @ b.to("lupine")).cpu(), a @ b)
    empty = torch.empty((3, 0), device="lupine") @ torch.empty((0, 2), device="lupine")
    torch.testing.assert_close(empty.cpu(), torch.zeros((3, 2)))


def test_math_attention_and_fully_masked_rows(backend):
    torch, native = backend
    q_cpu = (
        torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 24
    ).requires_grad_()
    k_cpu = (q_cpu.detach().flip(-1) / 2).requires_grad_()
    v_cpu = (q_cpu.detach() + 0.5).requires_grad_()
    q, k, v = (t.detach().to("lupine").requires_grad_() for t in (q_cpu, k_cpu, v_cpu))
    mask = torch.tensor([[0.0, -float("inf"), 0.0], [-float("inf")] * 3, [0.0] * 3])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_cpu, k_cpu, v_cpu, mask
    )
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, mask.to("lupine")
    )
    (output**2).sum().backward()
    (expected**2).sum().backward()
    for actual, reference in (
        (output, expected),
        (q.grad, q_cpu.grad),
        (k.grad, k_cpu.grad),
        (v.grad, v_cpu.grad),
    ):
        torch.testing.assert_close(
            actual.cpu(), reference.detach(), rtol=2e-4, atol=2e-5
        )


def test_cpu_scalar_left_operand(backend):
    torch, native = backend
    cpu = torch.arange(4, dtype=torch.float32)
    x = cpu.to("lupine")
    torch.testing.assert_close((1 - x).cpu(), 1 - cpu)
    torch.testing.assert_close(
        torch.rsub(x, 2, alpha=3).cpu(), torch.rsub(cpu, 2, alpha=3)
    )
    torch.testing.assert_close((torch.tensor(0.5) * x).cpu(), 0.5 * cpu)
    torch.testing.assert_close((torch.tensor(3.0) / (x + 1)).cpu(), 3 / (cpu + 1))


def test_dtype_converting_transfers(backend):
    import numpy

    torch, native = backend
    values = numpy.linspace(-1, 1, 6).reshape(2, 3)
    x = torch.from_numpy(values).to("lupine", torch.float32)
    assert x.dtype == torch.float32
    torch.testing.assert_close(x.cpu(), torch.from_numpy(values).float())
    torch.testing.assert_close(
        x.to("cpu", torch.float64), torch.from_numpy(values).float().double()
    )
    torch.testing.assert_close(
        x.T.to("cpu", torch.float64), torch.from_numpy(values).float().double().T
    )
    ints = torch.arange(5).to("lupine", torch.float32)
    torch.testing.assert_close(ints.cpu(), torch.arange(5, dtype=torch.float32))
