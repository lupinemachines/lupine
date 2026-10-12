"""Compare native macOS and Linux CUDA clients against the same remote GPU.

Set LUPINE_SERVER/LUPINE_LIBDIR, then run with --device lupine or --device cuda.
Compare runs with matching median_sync_seconds; raw WAN timings are not comparable.
"""

import argparse
import json
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "lupine"), required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument(
        "--workload", choices=("arithmetic", "matmul", "training", "layer_norm", "all"), default="all"
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.size < 1:
        parser.error("iterations and size must be positive")

    start = time.perf_counter()
    import lupine

    lupine.connect()
    import torch

    setup_start = start
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    device = args.device
    x_cpu = torch.randn(32, args.size, dtype=torch.float32) / args.size**0.5
    w_cpu = torch.randn(args.size, args.size, dtype=torch.float32) / args.size**0.5
    x, w = x_cpu.to(device), w_cpu.to(device).requires_grad_()
    syncs = []
    for index in range(5):
        start = time.perf_counter()
        getattr(torch, device).synchronize()
        syncs.append(time.perf_counter() - start)
        if index == 0:
            setup = time.perf_counter() - setup_start
    workloads = (
        ("arithmetic", "matmul", "training", "layer_norm")
        if args.workload == "all"
        else (args.workload,)
    )
    results = []
    for name in workloads:
        if name == "arithmetic":
            reference = (x_cpu.exp() + x_cpu * 2).sum()
            run = lambda: (x.exp() + x * 2).sum().cpu()
        elif name == "matmul":
            reference = x_cpu @ w_cpu
            run = lambda: (x @ w).cpu()
        elif name == "layer_norm":
            reference = torch.nn.functional.layer_norm(x_cpu, (args.size,))
            run = lambda: torch.nn.functional.layer_norm(x, (args.size,)).cpu()
        else:
            reference_w = w_cpu.clone().requires_grad_()
            reference = ((x_cpu @ reference_w) ** 2).mean()
            reference.backward()
            reference = reference.detach()

            def run():
                output = x @ w
                loss = (output * output).mean()
                loss.backward()
                return loss.cpu()

        timings = []
        for _ in range(args.iterations + 1):
            w.grad = None
            start = time.perf_counter()
            value = run()
            timings.append(time.perf_counter() - start)
            torch.testing.assert_close(value, reference, rtol=2e-4, atol=2e-5)
            if name == "training":
                torch.testing.assert_close(w.grad.cpu(), reference_w.grad, rtol=2e-4, atol=2e-5)
        results.append(
            {
                "workload": name,
                "cold_seconds": timings[0],
                "warm_seconds": timings[1:],
                "median_warm_seconds": statistics.median(timings[1:]),
            }
        )
    print(
        json.dumps(
            {
                "device": device,
                "torch": torch.__version__,
                "size": args.size,
                "setup_seconds": setup,
                "median_sync_seconds": statistics.median(syncs),
                "results": results,
            }
        ),
        flush=True,
    )
    from lupine import _backend

    _backend.close()


if __name__ == "__main__":
    main()
