# Timeout investigation: #899 and #836

Measured on 2026-09-22 from Lupine `45c2290f` (latest `origin/main` at the
start of the investigation), with CUDALibrarySamples `3437729`, CUDA 13.3.73,
a single RTX 4090, and driver 590.48.01. These are **not L4 measurements**.
The Lupine build is RelWithDebInfo; samples are Release, SM 89. Native and
loopback clients run on `inferable-node-006`; the LAN client runs on the CPU
build host. All times are wall-clock seconds including startup and validation.
Loopback uses TCP through the driver shim, with local CUDA disabled using
`LUPINE_REAL_LIBCUDA=/usr/local/cuda/lib64/stubs/libcuda.so` so the same physical
GPU is not also enumerated as a local device.

| Sample | Native | Lupine, loopback | Lupine, LAN |
| --- | ---: | ---: | ---: |
| `blocked_potrf`, upstream reference | 6.22 | 17.65 | 110.15 |
| `blocked_potrf`, batched reference readback | 5.31 | 9.28 | 23.38 |
| `bicgstab`, CUDA 13.3 | 0.53 | 0.94 | 1.81 |
| `MgSyevd` example 1 | 0.71 | 2.16 | 11.91 |
| `MgSyevd` example 2 | 0.70 | 2.18 | 14.81 |
| `MgSyevd` example 3 | 0.72 | 3.19 | 23.78 |
| `dgemm_emulation` | 55.40 | 59.97 | not measured |
| `dgemm_emulation_pipeline` | 40.82 | 45.13 | not measured |

These are individual successful runs, not latency percentiles. A separate
profiled LAN run of `MgSyevd` example 1 took 12.27 s. Check the numerical output,
not just the exit code: `MgSyevd` prints its eigenvalue error even on exit 0.
All three examples above reported `|D - lambda|_inf = 2.664535E-15`.

## #899: small synchronous readbacks dominate

`MathDx/cuSolverDx/common/cusolver_reference_cholesky.hpp` reads the result of
its batched cuSolver reference one triangle row at a time. For `blocked_potrf`,
400 matrices of 512 rows mean **204,800 device-to-host API calls**, copying
8–4096 bytes each. Despite the `cudaMemcpyAsync` spelling, the destination is
a pageable `std::vector`. Lupine completes each readback before returning;
deferring it would break callers that immediately consume pageable results.
See NVIDIA's [API synchronization behavior](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-driver-api/api-sync-behavior.html).

The embedded runner patch reads one complete matrix into a reusable temporary
buffer, synchronizes, then copies only the desired triangle on the CPU. The
other triangle and any leading-dimension padding in the reference output stay
unchanged. Matrix sizes, batch counts, kernel timings, precision, repetitions,
and the numerical comparison remain the same. Temporary host storage is one
matrix (2 MiB here). The tradeoff is reading the unused triangle too: 800 MiB
instead of about 401 MiB, in exchange for 512 times fewer readback calls.

`LUPINE_RPC_STATS` confirms 204,803 ordinary DtoH requests before and 403 after
(the extra three are metadata/info copies; the large output copy uses bulk
lanes). Total requests fall from 205,119 to 1,119, including the new per-matrix
synchronizations. The LAN speedup is 4.7x. Relative forward error remains about
`1.1e-16`. The shared reference also passes `potrf_batched_thread`,
`posv_batched_thread`, and `posv_batched_block`, both natively and through Lupine.

Remaining overhead includes several 800 MiB input/reset transfers and the
result readbacks. In the patched LAN profile, five HtoDAsync requests spend
9.41 s waiting for responses in total. The GPU potrf timings are tens of
milliseconds, so the old timeout was primarily validation/transport overhead,
not a slow factorization. The patched sample uses the ordinary sample timeout.

## #836: distinguish the solver workload from the intermittent hang

`MgSyevd` is fast natively on one GPU, so multi-GPU emulation on a single GPU is
not by itself the explanation. The first example sends 85,693 RPC requests,
including 16,104 `cuStreamSynchronize` calls. Those synchronizations account for
6.20 s of response wait in the 12.27 s LAN profile. Another 2,572
`cuStreamWaitEvent` calls cost 0.93 s; 1,896 `cuStreamGetGreenCtx` queries cost
0.69 s. Copies, event creation/destruction, and other driver traffic contribute
the remainder. These calls originate inside NVIDIA's cuSolverMg implementation.
The suite re-enables all three examples with `LONG_SAMPLE_TIMEOUT` to retain
coverage and allow for the shared CI GPU's load. This is a deliberate budget
choice, not a claimed solver speedup.

For CUDA 11.8 `bicgstab`, using the CUDA 11.8.0 container's libraries on the same
GPU, 25 consecutive native runs took 0.45–0.51 s (median 0.46), and 25 loopback
runs took 0.70–0.73 s (median 0.71). Each reported the expected final residual,
`1.522334e-08`. Three additional rounds of eight concurrent processes also passed
in both modes (24 native runs at 2.08–2.66 s and 24 loopback runs at 1.04–2.27 s).
This does not reproduce or resolve the L4 intermittent hang.
The original failed run `35308348283` has an empty sample log. The runner now
line-buffers stdout/stderr so a future timeout retains the last completed step.
The known-failure entry remains; increasing its timeout is not justified by
these measurements.

The two cuBLASDx DGEMM examples also tracked under #836 have substantial native
cost. They initialize/check large host matrices and repeatedly time an
8192-by-8192 native FP64 cuBLAS reference. Its measured kernel time is about
890 ms per multiplication, versus about 94–100 ms for the emulation path.
Loopback adds roughly 4–5 s to the whole program. Their known-failure entries
remain: the native benchmark itself has little margin under a loaded L4 lane's
120 s budget, and this patch does not reduce their workload.

## Reproducing the measurements

Build the driver client/server from the same revision, and build the samples
with `BUILD_ONLY=1` and `LIBRARY_SAMPLES_ARCH=89`. Keep an upstream binary before
applying the embedded patch if comparing before/after. Use the same executable
and CUDA library versions for native and shim runs, and run them sequentially
on an idle GPU. On the GPU host, with a separate server listening on 25997:

```sh
# Native: no shim in LD_LIBRARY_PATH or LD_PRELOAD.
/usr/bin/time -f 'elapsed_seconds=%e' env -u LD_PRELOAD \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64 ./sample

# The server uses the real driver; only the client gets the stub override.
env LD_LIBRARY_PATH=/usr/local/cuda/lib64 LUPINE_PORT=25997 ./lupine_driver_server

# In another shell, shim_dir contains libcuda.so.1 from this revision.
/usr/bin/time -f 'elapsed_seconds=%e' env \
  LD_LIBRARY_PATH="$shim_dir:/usr/local/cuda/lib64" \
  LD_PRELOAD="$shim_dir/libcuda.so.1" \
  LUPINE_REAL_LIBCUDA=/usr/local/cuda/lib64/stubs/libcuda.so \
  LUPINE_SERVER=127.0.0.1:25997 LUPINE_RPC_STATS=sample-rpc.tsv \
  stdbuf -oL -eL ./sample
```

For LAN measurements run the client on the CPU host and point `LUPINE_SERVER`
at the GPU host. `LUPINE_RPC_STATS` is a TSV of opcode, request count, and
nanoseconds spent in `rpc_wait_for_response`; map opcodes using
`codegen/gen_rpc_ids.h` and `rpc.h`. Wait totals exclude paths that call
`rpc_read_start` directly, including ordinary pageable DtoH, so zero recorded
wait for those copies does not mean they are asynchronous or free.
