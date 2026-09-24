# bicgstab's CUDA 11.8 hang (#836)

The intermittent timeout occurs inside NVIDIA's cuSPARSE library and also
reproduces without Lupine loaded. The sample has not reached its first
residual calculation when it hangs. Increasing its timeout cannot resolve it.

[NVIDIA/CUDALibrarySamples#370][upstream] adds a CUDA 11 compatibility path
using `cusparseDcsrsv2` with `CUSPARSE_SOLVE_POLICY_USE_LEVEL`. Its analysis
avoids the affected coloring kernel while preserving the matrix, ILU
preconditioner, and BiCGStab iteration. CUDA 12+ retains SpSV, with its scalar
initialized to one. The upstream patch is stored verbatim in
`patches/bicgstab-cuda11-analysis.patch`, following #920's patch workflow.

The runner applies it idempotently and rebuilds a cached bicgstab executable
when its source is newer. The sample runs with the ordinary timeout on all
CI toolkits, including CUDA 11.8. CUDA 12.0–12.2 are not validated: NVIDIA's
[CUDA 12.3 release notes][release-notes] describe a matching intermittent
SpSV analysis hang, so the sample README recommends 12.3+ for that path.

## Diagnosis

Investigation on 2026-09-24 used an NVIDIA L4, driver 580.173.02, Ubuntu
24.04, and CUDA 11.8 libraries. The original sample was built from pinned
CUDALibrarySamples commit `34377293fb148e90aad16fd762fa2445f941be55`, with
`-O3 -DNDEBUG`. Driver-only runs used Lupine at `67554bfc`; native controls
loaded NVIDIA's driver directly, with no shim or `LD_PRELOAD`.

Both native and driver-only captures showed the same active kernel and
instruction offset in CUDA-GDB:

```text
cusparse::find_colors_ker<int, 128, false>
grid = (58, 1, 1), block = (128, 1, 1)
+4560: LDG.E.STRONG.SYS R10[R8.64]
+4576: YIELD
+4592: ISETP.NE.AND P0,PT,R10,c[0x0][0xc],PT
+4608: @P0 BRA ...+4560
```

All 58 blocks were resident. Their first threads polled a grid barrier while
other threads waited at warp/block barriers. In the native capture, the
counter was 59 while the comparison expected 58, and the coloring iteration
register had reached 490002 for a matrix with 490000 rows. This locates the
failure inside the library; its exact internal race is not established.

The driver-only client was in `cusparseSpSV_analysis`, waiting for a
four-byte device-to-host copy. The server was inside NVIDIA's
`cuMemcpyDtoHAsync_v2`, waiting for preceding GPU work. Other processes
continued completing normally. CUPTI tracing confirmed that the CUDA 11
compatibility path does not launch `find_colors_ker`.

## Validation

Stress batches used six concurrent processes and repeated driver-only
MgSyevd load. Hangs timed out after 20 seconds, before `Initial Residual`.

| CUDA 11.8 variant | Native runs | Driver-only runs |
| --- | --- | --- |
| Original sample | 252 passed, 4 hung | Hang captured separately |
| Initialized SpSV scalar | 253 passed, 3 hung | — |
| Initialized scalar and zeroed workspace | 252 passed, 4 hung | — |
| Upstream compatibility patch | 256 passed, 0 hung | 256 passed, 0 hung |

Every successful CUDA 11.8 run converged in 13 iterations with final error
norm `1.522334e-08`. The patched runs took at most 10.6 seconds natively and
16.8 seconds through the shim while sharing the GPU with concurrent batches.

The patched CUDA 12+ path also passed a native smoke test and 64/64
concurrent driver-only runs on each of CUDA 12.4 (L4) and CUDA 13.3
(RTX 4090, driver 590.48.01). Every run converged in 13 iterations with final
error norm `1.522333e-08`; the slowest took 5.7 and 2.2 seconds respectively.

The actual runner was tested against a clean pinned sample checkout with an
old cached CUDA 13.3 build. `BUILD_SAMPLES=auto` applied the patch, rebuilt
bicgstab, and reported PASS. A second run applied no additional changes,
reused the executable, and passed. Patch reverse checks, shell syntax, and
whitespace checks also passed.

## Repeating the check

Build `cuSPARSE/bicgstab` and `cuSOLVER/MgSyevd` against the selected toolkit.
Run a repeated MgSyevd workload in another terminal, then run bicgstab in
six concurrent processes, capturing each process's output and exit status:

```bash
export BICGSTAB=/path/to/build/cuSPARSE/bicgstab/bicgstab_example
export LD_LIBRARY_PATH=/path/to/toolkit/lib64
unset LD_PRELOAD LUPINE_SERVER LUPINE_REAL_LIBCUDA
mkdir -p /tmp/bicgstab-native
seq 1 256 | xargs -P 6 -I{} sh -c '
  timeout --kill-after=5s 20s stdbuf -oL -eL "$BICGSTAB" \
    >"/tmp/bicgstab-native/{}.log" 2>&1
  printf "%s %s\n" "{}" "$?"
'
```

For a driver-only comparison, put the shim first in `LD_LIBRARY_PATH` and
set `LUPINE_SERVER`. A loopback client on a GPU host also needs
`LUPINE_REAL_LIBCUDA` pointed at the toolkit's stub `libcuda.so` so native
GPU discovery does not bypass the remote route. After capturing a hang,
terminate its server child to release the GPU context; the integration
runner already performs this cleanup.

[upstream]: https://github.com/NVIDIA/CUDALibrarySamples/pull/370
[release-notes]: https://docs.nvidia.com/cuda/archive/12.3.0/cuda-toolkit-release-notes/index.html#cusparse-release-12-3
