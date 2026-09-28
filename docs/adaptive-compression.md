# Automatic RPC compression

The choice is internal to each connection; applications do not select a codec
or protocol. The initial HTTP/2 POST remains `content-encoding: lz4` and offers
`x-lupine-accept-encoding: lupine-block-v1`. A new server advertises the same
capability and may use the block encoding for its response. After that existing
handshake, the client can use it on subsequent request lanes. Either direction
keeps legacy LZ4 if its peer does not advertise support. No discovery RPC or
additional round trip is required. This is compression compatibility, not a
promise that unrelated CUDA RPC schema changes interoperate across versions.

## Wire format and resource limits

A `lupine-block-v1` body contains a sequence of eight-byte headers followed by
encoded bytes. Header fields are little-endian:

| Field | Meaning |
| --- | --- |
| uint32 | `(encoded_length << 2) | codec` |
| uint32 | Decoded length |

Codecs are 0 (raw bytes), 1 (linked LZ4 frame fragment), 2 (complete Zstd frame),
and 3 (body terminator). Decoded length is at most 1 MiB; encoded length is at
most 1 MiB + 64 bytes. Bounds are checked before allocating. Raw and Zstd blocks
cannot exceed their decoded length. Raw lengths must match exactly. Zstd must
contain exactly one frame with a matching content size and decoded length.

LZ4 fragments share a frame/dictionary within their lane, including across
intervening raw or Zstd blocks. LZ4's own blocks are at most 256 KiB; larger frame
windows and external dictionary IDs are rejected. A zero-decoded-length LZ4
fragment carries the frame footer when that codec was used. The final codec-3
header has both lengths zero, requires that footer, and forbids trailing bytes.
HTTP/2 END_STREAM without the body terminator is a truncation error.

Each lane allocates codec contexts lazily and frees them when it closes. The
existing outbound queue limits and flow control apply. A large source is fed
one bounded block at a time, so its whole payload need not be compressed before
transmission starts. Raw blocks can be delivered directly from received data.
This change does not add compression worker threads.

## Selection

Small flushes (under 4 KiB) keep linked LZ4. Independent compression frames for
every tiny RPC lose useful history and add overhead; the bulk policy must not
make that regression the default.

Bulk starts with Zstd level 1. A cheap, stratified byte-frequency sample places
the block in one of four histories, separating mostly-zero, structured,
floating-point-like and high-entropy data. The first few blocks in each history
compare LZ4 and Zstd encoding size and elapsed encoding time. Steady-state
comparisons happen about once per 32 MiB in that history. Significant changes
in measured delivery rate bring the next comparison forward to within 4 MiB.
Other blocks encode once, or bypass compression when the LZ4 history indicates
negligible benefit. The raw decision is periodically sampled too, allowing
compression to recover when the data changes.

The cost model adds encoding time to predicted transmission time. Zstd also
gets a conservative extra decode allowance of 1 ns per input byte. This is a
heuristic, not a measurement of the remote CPU. A change needs two successive
comparisons predicting at least 20% improvement. The wire remains decodable
regardless of the prediction's quality.

## Delivery measurements

Socket-write time can measure kernel buffering rather than network capacity.
The transport instead queues a tagged HTTP/2 PING after at least 64 KiB of
outgoing compressed DATA bytes (excluding control frames), with at most eight such markers outstanding. ACK spacing
measures progress through the peer's HTTP/2 decoder. The fixed RTT cancels
between markers. Samples need at least 10 ms of ACK spacing and are smoothed.
A gap over 20 ms between sends starts a new measurement burst, excluding
application/GPU think time. Unknown speed starts conservatively at 10 Mb/s.
The existing shutdown and heartbeat PINGs have different tags and are unchanged.

This is effective delivery rate, not a physical NIC speed measurement: receiver
CPU, congestion, flow control and ACK batching can affect it. Codec comparisons
and hysteresis bound the consequences of a noisy estimate. No probe payloads
or synthetic speed tests are sent.

## Validation and remaining review

Unit tests cover both switching directions, hysteresis, sampling limits,
RTT-independent delivery measurements, idle bursts, bounded outstanding probes,
byte-fragmented codec transitions, truncation, invalid lengths, malformed
compressed data and oversized LZ4 windows. Transport tests cover negotiated
lanes in both directions and independent legacy peers omitting capability
headers. Existing HTTP/2 flow-control, large-message and shutdown tests remain.
Real GPU tests exercise both mixed-version directions too.

The policy constants need workload and CPU-diversity review. Particularly useful
follow-ups are many concurrent lanes, weaker/ARM clients and a connection whose
bandwidth changes during a long transfer. The 1 ns decode allowance and coarse
payload classes are deliberate approximations; they are not claims of optimal
codec selection for every workload. The PR remains draft while cross-platform
CI and policy/performance review run.

Zstandard 1.5.7 is statically vendored, like LZ4. Build only common/compression/
decompression sources, with no worker pool, legacy codecs or new system package
requirement. Diagnostics use the existing `LUPINE_DEBUG` logging switch; there
is no codec-selection environment variable.

## Measurements (2026-09-27)

RTX 4090 / Ryzen 7950X on node006. Baseline is main `fe7b6df1`, including
256 KiB linked LZ4 blocks; neither variant includes #971 filtering or #972 event
batching. WAN shaping uses a dedicated veth/network namespace with offloads
disabled, 75 ms delay in each direction, and symmetric rate limits. A private
mount namespace hides native CUDA from the client to ensure remote execution.

The unchanged `console.lupine.sh/gpt_bench.py` workload (with phase markers and
a seed) completed as follows. One full run per configuration; small differences
are provisional. Times exclude process teardown. Byte counts include network
overhead. CUDA training is not bitwise deterministic across runs.

| 150 ms RTT | LZ4 total | Adaptive total | LZ4 client TX | Adaptive client TX |
| --- | ---: | ---: | ---: | ---: |
| 10 Mb/s | 91.77 s | 80.54 s | 76.63 MB | 63.13 MB |
| 100 Mb/s | 37.59 s | 36.41 s | 76.47 MB | 63.99 MB |

Training stays about 4.12–4.14 s. Compression addresses bytes, not the remaining
synchronous CUDA round trips. The adaptive format trades some ratio for switching
flexibility: the earlier fixed, continuous Zstd-1 prototype took 78.80 s /
60.70 MB at 10 Mb/s, but had a substantial fast-link float32 regression.

Fast-link figures below are medians of three alternating processes per variant
on a same-host veth (not a physical NIC measurement). Each bulk phase transfers
16 × 16 MiB and checks the return bytes exactly. Small-RPC latency uses 2,000
synchronous `cuMemGetInfo_v2` calls. Client CPU includes the whole process's
copying/protocol overhead.

| Payload/direction | LZ4 Gb/s | Adaptive Gb/s | LZ4 client CPU s | Adaptive client CPU s |
| --- | ---: | ---: | ---: | ---: |
| zeros_htod | 17.24 | 17.28 | 0.031 | 0.050 |
| zeros_dtoh | 17.91 | 18.48 | 0.076 | 0.059 |
| random_htod | 6.85 | 6.69 | 0.257 | 0.279 |
| random_dtoh | 10.06 | 9.31 | 0.314 | 0.310 |
| float32_htod | 7.26 | 7.34 | 0.215 | 0.207 |
| float32_dtoh | 10.96 | 10.84 | 0.282 | 0.290 |

Small synchronous RPCs: 50.8 → 51.4 µs/call.

Float32 throughput stays within roughly 1% of baseline in this sample, instead
of the fixed-Zstd regression. Random DtoH is about 7% slower in the three-run
median and should be investigated/tuned before promoting this draft. These
results do not establish a universal win or CPU bound on other hardware.

`test/benchmark_protocol_compression.py` contains the payload benchmark; run it
with the normal `LUPINE_SERVER` and shim library path. For short WAN runs use
`--bytes 1048576 --iterations 4 --rpc-count 10`. The implementation has no codec
selection flag. Optional `LUPINE_DEBUG=1` logs sampled costs and choices.
