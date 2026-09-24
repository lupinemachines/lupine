# bicgstab's CUDA 11.8 hang (#836)

The intermittent timeout is a GPU hang inside NVIDIA's cuSPARSE library. It
also occurs without Lupine loaded. It is not an unusually long BiCGStab solve:
the sample has not reached its first residual calculation when it hangs.

The runner excludes this sample only on toolkits older than CUDA 12.3.
NVIDIA's [CUDA 12.3 cuSPARSE release notes][release-notes] report a fix for an
intermittent `cusparseSpSV_analysis()` hang in a multithreaded environment.
That description matches the failure below; NVIDIA does not publish the
kernel's implementation or the precise race fixed in that release.

Increasing `SAMPLE_TIMEOUT` cannot resolve this failure. Newer toolkits run
the sample with the ordinary timeout. `cuSOLVER/MgSyevd` is a separate, slow
workload that already uses `LONG_SAMPLE_TIMEOUT` (600 seconds by default).

## Reproduction and diagnosis

Investigation on 2026-09-24 used:

- An NVIDIA L4, driver 580.173.02, Ubuntu 24.04, matching the GPU integration
  image family.
- Unmodified CUDALibrarySamples commit
  `34377293fb148e90aad16fd762fa2445f941be55`, built with `-O3 -DNDEBUG`.
- CUDA 11.8 runtime, cuBLAS, and cuSPARSE; Lupine at `67554bfc` for the
  driver-only runs. No replacement CUDA runtime was used.
- Six concurrent bicgstab processes, with repeated driver-only MgSyevd runs
  in the background. This increases the chance of seeing the intermittent
  failure; it does not fail on every repetition.

Two driver-only runs hung before printing `Initial Residual`. A subsequent
batch of 256 native runs also hung once, on run 96. The native executable
loaded NVIDIA's driver directly, with no Lupine library or `LD_PRELOAD`.
Other processes continued completing normally while the affected process
remained stuck. Passing runs converged in 13 iterations, with a final error
norm around `1.522334e-08`.

Attach `cuda-gdb` to the affected server child for a driver-only run, or to
the sample process for a native run. Both showed exactly the same active
kernel and instruction offset:

```text
cusparse::find_colors_ker<int, 128, false>
grid = (58, 1, 1), block = (128, 1, 1)
+4560: LDG.E.STRONG.SYS R10[R8.64]
+4576: YIELD
+4592: ISETP.NE.AND P0,PT,R10,c[0x0][0xc],PT
+4608: @P0 BRA ...+4560
```

All 58 blocks were resident. Their first threads kept polling the grid
barrier, while the other threads waited at warp/block barriers. In the native
capture, the counter was 59 although the comparison expected 58. The coloring
iteration register had reached 490002 for a matrix with 490000 rows.
The corresponding driver-only capture had the same runaway loop and barrier
wait. This is an internal library kernel failure, not a missing RPC reply.

On the CPU, the driver-only client was in `cusparseSpSV_analysis()` waiting
for a four-byte device-to-host copy. The server was inside NVIDIA's
`cuMemcpyDtoHAsync_v2`, waiting for the preceding GPU work.

## Validation

- CUDA 12.4.0 libraries: 256/256 driver-only runs passed on the L4, six at a
  time with repeated MgSyevd load. The slowest took 3.4 seconds. This used the
  same sample and workload as the CUDA 11.8 reproduction.
- CUDA 13.3.0 libraries: 64/64 driver-only runs passed on an RTX 4090 with
  driver 590.48.01, six at a time. The slowest took 2.3 seconds.
- The runner was exercised with temporary sample/SSH stand-ins: CUDA 11.8
  and 12.2 skipped before starting a server; CUDA 12.3, 12.4, 13.3, and an
  unknown toolkit version reached execution. Explicit user skips still won.
- Shell syntax and whitespace checks passed. These checks validate the
  runner policy; the GPU runs validate the newly enabled library path.

## Repeating the check

Build the upstream `cuSPARSE/bicgstab` and `cuSOLVER/MgSyevd` samples against
the toolkit being tested. On the GPU host, start a repeated MgSyevd workload
in another terminal and run a batch of bicgstab processes. For a native
control, remove all shim paths and `LD_PRELOAD`; merely clearing
`LUPINE_SERVER` is insufficient if the shim remains in the loader path.

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

For a driver-only comparison, use the same libraries and sample, put the
driver shim first in `LD_LIBRARY_PATH`, and set `LUPINE_SERVER`. A loopback
client on a GPU host also needs `LUPINE_REAL_LIBCUDA` pointed at the toolkit's
stub `libcuda.so`, so native GPU discovery does not bypass the remote route.
Terminate the affected server child after capturing a hang; otherwise its
kernel keeps consuming the GPU. The integration runner's server cleanup
already does this.

[release-notes]: https://docs.nvidia.com/cuda/archive/12.3.0/cuda-toolkit-release-notes/index.html#cusparse-release-12-3
