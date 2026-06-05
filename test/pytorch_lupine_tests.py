#!/usr/bin/env python3
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_cuda():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
    assert torch.cuda.device_count() >= 1, "no CUDA devices visible"
    torch.cuda.set_device(0)


def test_discover():
    require_cuda()
    props = torch.cuda.get_device_properties(0)
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={props.name} cc={props.major}.{props.minor} sms={props.multi_processor_count}")


def test_tensor_ops():
    require_cuda()
    x = torch.arange(4096, device="cuda", dtype=torch.float32)
    y = torch.sin(x) + torch.cos(x * 0.25)
    torch.cuda.synchronize()
    got = float(y[:128].sum().cpu())
    ref = float((torch.sin(torch.arange(4096, dtype=torch.float32)) +
                 torch.cos(torch.arange(4096, dtype=torch.float32) * 0.25))[:128].sum())
    assert math.isclose(got, ref, rel_tol=1e-5, abs_tol=1e-4), (got, ref)


def test_matmul():
    require_cuda()
    torch.manual_seed(123)
    a = torch.randn((512, 384), device="cuda")
    b = torch.randn((384, 256), device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert c.shape == (512, 256)
    assert torch.isfinite(c).all().item()


def test_fft():
    require_cuda()
    x = torch.randn(4096, device="cuda", dtype=torch.complex64)
    y = torch.fft.fft(x)
    z = torch.fft.ifft(y)
    torch.cuda.synchronize()
    err = (z - x).abs().max().cpu().item()
    assert err < 1e-4, err


def test_cudnn_conv():
    require_cuda()
    torch.backends.cudnn.enabled = True
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda()
    x = torch.randn((4, 3, 32, 32), device="cuda")
    y = conv(x).relu()
    torch.cuda.synchronize()
    assert y.shape == (4, 8, 32, 32)
    assert torch.isfinite(y).all().item()


def test_sparse_mm():
    require_cuda()
    indices = torch.tensor([[0, 1, 1, 2], [2, 0, 2, 3]], device="cuda")
    values = torch.tensor([3.0, 4.0, 5.0, 6.0], device="cuda")
    sparse = torch.sparse_coo_tensor(indices, values, (4, 4), device="cuda").coalesce()
    dense = torch.randn((4, 3), device="cuda")
    out = torch.sparse.mm(sparse, dense)
    torch.cuda.synchronize()
    assert out.shape == (4, 3)
    assert torch.isfinite(out).all().item()


def test_linalg_solve():
    require_cuda()
    torch.manual_seed(7)
    a = torch.randn((64, 64), device="cuda")
    a = a @ a.T + torch.eye(64, device="cuda") * 0.1
    b = torch.randn((64, 8), device="cuda")
    x = torch.linalg.solve(a, b)
    residual = (a @ x - b).norm() / b.norm()
    torch.cuda.synchronize()
    assert float(residual.cpu()) < 1e-4


def test_autograd_step():
    require_cuda()
    torch.manual_seed(5)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn((128, 32), device="cuda")
    target = torch.randn((128, 4), device="cuda")
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).item()


def test_compile_elementwise():
    require_cuda()

    @torch.compile
    def fn(x):
        return torch.tanh(x * 1.25 + 0.5)

    x = torch.randn((2048,), device="cuda")
    y = fn(x)
    torch.cuda.synchronize()
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(~self.mask[:, :, :time, :time], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, vocab_size=64, block_size=64, n_layer=3, n_head=4, n_embd=128):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch, time = idx.shape
        pos = torch.arange(0, time, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def test_microgpt_train():
    require_cuda()
    torch.manual_seed(42)
    device = torch.device("cuda")
    vocab_size = 64
    block_size = 64
    batch_size = 32
    steps = 24

    model = MicroGPT(vocab_size=vocab_size, block_size=block_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, fused=True)
    base = torch.arange(block_size + 1, device=device).unsqueeze(0)
    offsets = torch.arange(batch_size, device=device).unsqueeze(1) * 7

    first_loss = None
    last_loss = None
    for step in range(steps):
        noise = (step * 11 + offsets) % vocab_size
        seq = (base + noise) % vocab_size
        x = seq[:, :-1].contiguous()
        y = seq[:, 1:].contiguous()
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_value = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value

    torch.cuda.synchronize()
    print(f"microgpt first_loss={first_loss:.4f} last_loss={last_loss:.4f}")
    assert last_loss is not None and math.isfinite(last_loss)
    assert last_loss < first_loss * 0.75, (first_loss, last_loss)


TESTS = {
    "discover": test_discover,
    "tensor_ops": test_tensor_ops,
    "matmul": test_matmul,
    "fft": test_fft,
    "cudnn_conv": test_cudnn_conv,
    "sparse_mm": test_sparse_mm,
    "linalg_solve": test_linalg_solve,
    "autograd_step": test_autograd_step,
    "compile_elementwise": test_compile_elementwise,
    "microgpt_train": test_microgpt_train,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test", choices=sorted(TESTS))
    args = parser.parse_args()
    TESTS[args.test]()
    print(f"{args.test}: PASS")


if __name__ == "__main__":
    main()
