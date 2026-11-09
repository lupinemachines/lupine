# HTTP/2 client writer spin profile

This isolates the client-only 5 → 20 µs spin change from
[lupinemachines/cloud#481](https://github.com/lupinemachines/cloud/pull/481).
It changes no CUDA copy behavior or wire format.

`benchmark_h2_writer.cpp` is a Linux-only microbenchmark using the real HTTP/2
transport, zstd encoding, and loopback TCP with the production socket options.
A caller submits 64-byte messages while a server thread drains and timestamps
them. Between submissions the caller busy-waits for the requested gap **after
its previous write returns**; it does not sleep or wait for a reply. Thus wall
throughput is mostly determined by the requested gap. CPU measurements include
that busy-wait on both variants.

The output columns are:

```
gap_us,count,wall_s,process_cpu_s,caller_cpu_s,voluntary_cs,involuntary_cs,enqueue_p50_us,enqueue_p99_us,delivery_p50_us,delivery_p99_us
```

Enqueue latency covers the transport write call and creation of its cursor;
delivery latency runs from just before that call until the server reads the
complete message. Context switches and CPU time come from `getrusage`, across
all process threads unless explicitly labeled caller. Measurements exclude
connection setup and teardown. Each process starts a new connection.

## Reproduce

From the repository root, build the transport without a GPU SDK:

```sh
cmake -S . -B /tmp/h2-profile-build -DCMAKE_BUILD_TYPE=Release \
  -DLUPINE_BUILD_CUDA=OFF -DLUPINE_BUILD_NVML=OFF \
  -DLUPINE_BUILD_HIP=OFF -DLUPINE_BUILD_SERVER=OFF
cmake --build /tmp/h2-profile-build --target h2_test transport_test -j8
ctest --test-dir /tmp/h2-profile-build -R '^(h2_test|transport_test)$' --output-on-failure
c++ -O3 -DNDEBUG -std=c++17 -DLUPINE_TLS_OPENSSL -I. \
  -Ithird_party/zstd/lib -c test/benchmark_h2_writer.cpp -o /tmp/h2-bench.o
```

For an exact reproduction of the measured baseline, extract `h2.cpp` from
`089d68891a9ecce4100257eb90f19d17d75afeb5`. Compile that source and the patched
source separately, then link each with the same transport archive. The explicit
object supplies the HTTP/2 symbols, so the archive's HTTP/2 object is not loaded.

```sh
git show 089d68891a9ecce4100257eb90f19d17d75afeb5:h2.cpp > /tmp/h2-baseline.cpp
c++ -O3 -DNDEBUG -std=c++17 -DLUPINE_TLS_OPENSSL -I. \
  -Ithird_party/zstd/lib -c /tmp/h2-baseline.cpp -o /tmp/h2-5.o
c++ -O3 -DNDEBUG -std=c++17 -DLUPINE_TLS_OPENSSL -I. \
  -Ithird_party/zstd/lib -c h2.cpp -o /tmp/h2-20.o
for spin in 5 20; do
  c++ /tmp/h2-bench.o /tmp/h2-$spin.o \
    /tmp/h2-profile-build/liblupine_rpc_core.a \
    /tmp/h2-profile-build/liblupine_zstd.a \
    -lnghttp2 -lssl -lcrypto -pthread -o /tmp/h2-bench-$spin
done
/tmp/h2-bench-5 30 20000
/tmp/h2-bench-20 30 20000
```

Repeat five times, alternating variant order per repetition, for gaps of 0, 10,
30, 100, and 1000 µs. Use 20,000 messages per run, or 2,000 for the 1000 µs gap.
Compare the median of each metric across runs.

## Scope

This measures unencrypted loopback transport on an ARM Neoverse-V2 host with
64 logical CPUs, Linux 6.8, GCC 11.4, Release (`-O3 -DNDEBUG`). Processes and
threads are not pinned. Both variants use the same fresh transport build;
only the HTTP/2 writer's spin deadline differs. No build or test runs overlap
the final measurements.

This does not establish a vLLM or GPU workload speedup. TLS, network latency,
CPU architecture, thread scheduling, and actual launch cadence can change the
tradeoff. The cloud PR's reported application results combine this change with
an async copy change and cannot be attributed to this change alone. `perf` was
unavailable for the host kernel, so no futex call counts or sampled stack
attribution are claimed; voluntary context switches are a scheduling metric,
not a direct count of writer wakeups.

## Measured results (2026-11-09)

Five repetitions per variant and gap; entries are medians across runs. The
baseline is main at `089d68891a9ecce4100257eb90f19d17d75afeb5`.
Raw runs are in [benchmark_h2_writer_results.csv](benchmark_h2_writer_results.csv).

| Gap (µs) | Enqueue p50, 5 → 20 µs spin (µs) | Delivery p50 (µs) | Process CPU (s) | Voluntary context switches |
|---|---|---|---|---|
| 0 | 0.160 → 0.192 | 24.512 → 25.856 | 0.030 → 0.030 | 2361 → 2341 |
| 10 | 0.544 → 0.448 | 17.824 → 17.760 | 0.574 → 0.569 | 22889 → 23519 |
| 30 | 2.144 → 0.576 | 24.192 → 20.800 | 1.188 → 1.418 | 59936 → 49506 |
| 100 | 2.208 → 2.240 | 24.768 → 24.928 | 2.599 → 2.902 | 60002 → 60006 |
| 1000 | 2.784 → 2.816 | 47.073 → 47.745 | 2.086 → 2.119 | 6000 → 6000 |

At the 30 µs gap, median enqueue latency decreases by
73.1%, median delivery latency by
14.0%, and voluntary context switches by
17.4%. Process CPU **increases by
19.4%**. Enqueue p99 changes from
2.560 to 2.720 µs; delivery p99 changes from
31.200 to 29.088 µs. This supports a median latency
benefit at that cadence, with a CPU cost, rather than a universal improvement.
At wider gaps the writer still sleeps and the longer spin mostly spends more CPU.
Small differences at the other gaps should be treated as noise-sensitive.

Both `h2_test` and `transport_test` pass against the fresh patched build.
