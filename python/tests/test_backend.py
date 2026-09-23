"""The torch backend end to end: the host torch of this interpreter against a
worker subprocess of the same interpreter running the operators on its CPU
(``LUPINE_WORKER_DEVICE=cpu``), or on a GPU through the driver shims when
``LUPINE_SERVER`` and ``LUPINE_WORKER_PYTHON`` name one. Every result is
checked against the same computation on the CPU.

Opt in with ``LUPINE_TEST_TORCH_BACKEND=1``: the extension is JIT-built for
the installed torch on first use, which needs a C++ compiler and ninja.
"""

from __future__ import annotations

import copy
import os
import sys
import threading

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LUPINE_TEST_TORCH_BACKEND"),
    reason="set LUPINE_TEST_TORCH_BACKEND=1 to run the torch backend against a worker",
)

GPU = bool(os.environ.get("LUPINE_SERVER")) and os.environ.get("LUPINE_WORKER_DEVICE", "cuda") == "cuda"


@pytest.fixture(scope="module")
def backend():
    os.environ["LUPINE_TORCH_BACKEND"] = "1"
    os.environ.setdefault("LUPINE_WORKER_PYTHON", sys.executable)
    if not GPU:
        os.environ["LUPINE_WORKER_DEVICE"] = "cpu"
    import lupine

    session = lupine.connect() if GPU else lupine.connect(host="127.0.0.1:1")
    session.__enter__()
    import torch

    yield torch, session.device()
    session.__exit__(None, None, None)


def same(torch, got, want, **kw):
    assert got.device.type == "cpu"
    assert got.shape == want.shape, (got.shape, want.shape)
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    torch.testing.assert_close(got, want, **kw)


def test_elementwise_plans_and_replays(backend):
    torch, dev = backend
    x = torch.arange(12.0).reshape(3, 4)
    for _ in range(3):
        got = ((x.to(dev) * 2 + 1).sin() / 3).cpu()
        same(torch, got, (x * 2 + 1).sin() / 3)


def test_inplace_and_out(backend):
    torch, dev = backend
    a = torch.arange(6.0).to(dev)
    a.add_(1.0).mul_(2)
    same(torch, a.cpu(), (torch.arange(6.0) + 1) * 2)
    out = torch.empty(0, device=dev)
    torch.arange(5, out=out)
    same(torch, out.cpu(), torch.arange(5.0))
    torch.arange(9, out=out)
    same(torch, out.cpu(), torch.arange(9.0))


def test_views_at_offsets(backend):
    torch, dev = backend
    x = torch.arange(20.0).reshape(4, 5).to(dev)
    view = x[1:3, 2:]
    same(torch, (view * 10).cpu(), torch.arange(20.0).reshape(4, 5)[1:3, 2:] * 10)
    view.add_(100)
    want = torch.arange(20.0).reshape(4, 5)
    want[1:3, 2:] += 100
    same(torch, x.cpu(), want)
    same(torch, x.t().contiguous().cpu(), want.t().contiguous())
    same(torch, x.view(-1)[3:9].cpu(), want.view(-1)[3:9])


def test_data_dependent_ops_are_synchronous(backend):
    torch, dev = backend
    x = torch.tensor([3, 0, 5, 0, 7, 3, 3])
    d = x.to(dev)
    same(torch, torch.nonzero(d).cpu(), torch.nonzero(x))
    same(torch, torch.masked_select(d, d > 2).cpu(), torch.masked_select(x, x > 2))
    same(torch, d[d > 2].cpu(), x[x > 2])
    same(torch, d[torch.tensor([0, 2, 4])].cpu(), x[torch.tensor([0, 2, 4])])
    same(torch, torch.unique(d).cpu(), torch.unique(x))
    same(torch, torch.bincount(d).cpu(), torch.bincount(x))
    same(torch, torch.repeat_interleave(d[:3], d[:3]).cpu(), torch.repeat_interleave(x[:3], x[:3]))
    assert torch.equal(d, d.clone())
    assert not torch.equal(d, d + 1)
    big = torch.arange(200.0).to(dev)
    index = torch.randperm(200)[:100]
    same(torch, big[index].cpu(), torch.arange(200.0)[index])


def test_item(backend):
    torch, dev = backend
    x = torch.arange(6.0).reshape(2, 3).to(dev)
    assert x[1, 2].item() == 5.0
    assert x[1:2, 2].item() == 5.0
    assert x.sum().item() == 15.0
    assert int(torch.tensor(7).to(dev)) == 7
    assert bool(torch.tensor(True).to(dev))
    p = torch.tensor([0.0, 1.0]).to(dev)
    assert torch.multinomial(p, 1).item() == 1


def test_split_cat_stack(backend):
    torch, dev = backend
    x = torch.arange(12.0).reshape(4, 3)
    d = x.to(dev)
    parts = torch.split(d, [1, 3])
    same(torch, torch.cat([parts[1], parts[0]]).cpu(), torch.cat([x[1:], x[:1]]))
    same(torch, torch.stack(list(d.chunk(2))).cpu(), torch.stack(list(x.chunk(2))))
    for n in (2, 3, 5):
        same(torch, torch.cat([d] * n).cpu(), torch.cat([x] * n))


def test_inplace_replay_with_another_scalar(backend):
    # An in-place op's plan is keyed on its tensors alone: the second call
    # replays the plan with a call that differs from the template's.
    torch, dev = backend
    a = torch.ones(4).to(dev)
    b = torch.ones(2).to(dev)
    a.mul_(2.0)
    a.mul_(3.0)
    a.mul_(2.0)
    same(torch, a.cpu(), torch.full((4,), 12.0))
    torch._foreach_div_([a, b], [2.0, 4.0])
    torch._foreach_div_([a, b], [3.0, 0.5])
    same(torch, a.cpu(), torch.full((4,), 2.0))
    same(torch, b.cpu(), torch.full((2,), 0.5))


def test_replay_keeps_slots_apart_when_views_coincide(backend):
    # The miss, whose call is the template with two slots over one view,
    # then a replay whose slots differ.
    torch, dev = backend
    data = torch.arange(20.0).to(dev)

    def batch(ix):
        return torch.cat([data[i : i + 3] for i in ix]).cpu().tolist()

    assert batch([2, 2]) == [2, 3, 4, 2, 3, 4]
    assert batch([5, 5]) == [5, 6, 7, 5, 6, 7]
    assert batch([1, 9]) == [1, 2, 3, 9, 10, 11]
    assert batch([9, 1]) == [9, 10, 11, 1, 2, 3]


def test_autograd_and_optimizer(backend):
    torch, dev = backend
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    x = torch.randn(16, 4)
    y = torch.randn(16, 1)
    ref = copy.deepcopy(model)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    for _ in range(3):
        loss = torch.nn.functional.mse_loss(model(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()
        opt.zero_grad()
        ref_loss = torch.nn.functional.mse_loss(ref(x), y)
        ref_loss.backward()
        ref_opt.step()
        ref_opt.zero_grad()
        torch.testing.assert_close(loss.item(), ref_loss.item(), rtol=1e-4, atol=1e-5)
    for p, q in zip(model.parameters(), ref.parameters()):
        same(torch, p.detach().cpu(), q.detach(), rtol=1e-4, atol=1e-5)


def test_reductions_and_shapes(backend):
    torch, dev = backend
    x = torch.arange(24.0).reshape(2, 3, 4)
    d = x.to(dev)
    values, indices = d.max(dim=1)
    same(torch, values.cpu(), x.max(dim=1).values)
    same(torch, indices.cpu(), x.max(dim=1).indices)
    same(torch, d.softmax(-1).cpu(), x.softmax(-1))
    same(torch, torch.nn.functional.layer_norm(d, (4,)).cpu(), torch.nn.functional.layer_norm(x, (4,)))
    same(torch, d.mean().cpu(), x.mean())
    same(torch, (d % 5).cpu(), x % 5)
    same(torch, torch.where(d > 10, d, 0.0).cpu(), torch.where(x > 10, x, 0.0))
    same(torch, (d.half() * 2).float().cpu(), (x.half() * 2).float())
    same(torch, d.to(torch.int64).cpu(), x.to(torch.int64))
    same(torch, d.view(torch.int32).cpu(), x.view(torch.int32))


def test_sdpa_and_conv(backend):
    torch, dev = backend
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    sdpa = torch.nn.functional.scaled_dot_product_attention
    same(torch, sdpa(q.to(dev), k.to(dev), v.to(dev), is_causal=True).cpu(), sdpa(q, k, v, is_causal=True), rtol=1e-3, atol=1e-4)
    image = torch.randn(1, 2, 6, 6)
    weight = torch.randn(3, 2, 3, 3)
    conv = torch.nn.functional.conv2d
    same(torch, conv(image.to(dev), weight.to(dev), padding=1).cpu(), conv(image, weight, padding=1), rtol=1e-3, atol=1e-4)


def test_foreach(backend):
    torch, dev = backend
    a = torch.ones(3).to(dev)
    b = torch.ones(2).to(dev)
    torch._foreach_add_([a, b], 1.5)
    same(torch, a.cpu(), torch.full((3,), 2.5))
    same(torch, b.cpu(), torch.full((2,), 2.5))
    c, d = torch._foreach_mul([a, b], 2)
    same(torch, c.cpu(), torch.full((3,), 5.0))
    same(torch, d.cpu(), torch.full((2,), 5.0))


def test_copies(backend):
    torch, dev = backend
    x = torch.arange(6.0).reshape(2, 3)
    d = x.to(dev)
    same(torch, d.t().cpu(), x.t())
    target = torch.empty(3, 2)
    target.copy_(d.t())
    same(torch, target, x.t())
    d2 = torch.empty(2, 3, device=dev)
    d2.copy_(d)
    same(torch, d2.cpu(), x)
    d3 = torch.empty(4, 2, 3, device=dev)
    d3.copy_(x)
    same(torch, d3.cpu(), x.expand(4, 2, 3))
    d4 = torch.empty(2, 3, dtype=torch.float64, device=dev)
    d4.copy_(x)
    same(torch, d4.cpu(), x.double())
    empty = torch.empty(0, device=dev)
    assert empty.cpu().numel() == 0
    assert (empty + 1).cpu().numel() == 0


def test_storage_growth_and_set(backend):
    torch, dev = backend
    t = torch.empty(0, device=dev)
    t.resize_(5)
    t.fill_(3.0)
    same(torch, t.cpu(), torch.full((5,), 3.0))
    a = torch.arange(4.0).to(dev)
    b = torch.empty(0, device=dev)
    b.set_(a)
    same(torch, b.cpu(), torch.arange(4.0))
    b.set_(a.untyped_storage(), 1, (3,), (1,))
    same(torch, b.cpu(), torch.arange(1.0, 4.0))


def test_worker_errors_carry_the_kernel_message(backend):
    torch, dev = backend
    with pytest.raises(RuntimeError, match="singular"):
        torch.linalg.inv(torch.zeros(2, 2, device=dev))


def test_backward_then_read_loop(backend):
    """The cross-lane race of the first revision: a synchronous read must
    not overtake the fire-and-forget launches that produce the gradient."""

    torch, dev = backend
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for _ in range(15):
        x = a.to(dev).requires_grad_()
        (x * x).sum().backward()
        same(torch, x.grad.cpu(), 2 * a)


def test_ops_from_another_thread(backend):
    torch, dev = backend
    results = []

    def work():
        x = torch.arange(8.0).to(dev)
        results.append((x * 3).sum().item())

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [84.0] * 4


def test_seeded_randomness_is_reproducible(backend):
    torch, dev = backend
    torch.manual_seed(1337)
    first = torch.randn(4, device=dev).cpu()
    torch.manual_seed(1337)
    same(torch, torch.randn(4, device=dev).cpu(), first)
    if not GPU:
        return
    state = torch.cuda.get_rng_state()
    a = torch.rand(3, device=dev).cpu()
    torch.cuda.set_rng_state(state)
    same(torch, torch.rand(3, device=dev).cpu(), a)


@pytest.mark.skipif(not GPU, reason="device queries need the worker's CUDA torch")
def test_device_module(backend):
    torch, dev = backend
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
    assert torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 0 and "CudaDeviceProperties" in repr(props)
    assert torch.cuda.get_device_capability(0)[0] >= 5
    free, total = torch.cuda.mem_get_info()
    assert 0 < free <= total
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = torch.ones(4, 4, device=dev) @ torch.ones(4, 4, device=dev)
    assert y.dtype == torch.bfloat16


@pytest.mark.skipif(not GPU, reason="CUDA graphs are captured by the worker")
def test_cuda_graph(backend):
    torch, dev = backend
    static = torch.ones(4, device=dev)
    out = torch.zeros(4, device=dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(static * 2 + 1)
    static.fill_(3.0)
    graph.replay()
    same(torch, out.cpu(), torch.full((4,), 7.0))
