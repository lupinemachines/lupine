# Scatter memcpy

Server builds consume a required precompiled operations directory rather than
invoking an accelerator compiler during the server build. Each SDK writes its
artifacts below a backend directory; the current CUDA artifact is
`cuda/smemcpy.fatbin`, with HIP and additional operations able to share the
same top-level path. Build the CUDA operations with:

```sh
cmake -DLUPINE_PRECOMPILED_OPS="$PWD/build/precompiled-ops" \
  -P ops/precompile.cmake
```

`smemcpy.cu` copies a packed, device-visible source fragment into a pitched 2D
or 3D destination. It is intended for HtoD ring slots mapped into the GPU
address space: the network side fills one contiguous slot and this operation
restores the destination's row and slice discontinuities.

`logical_offset` is the fragment's byte offset in the packed logical region.
The caller owns both extents and must keep the fragment within them. There is
no pointer, width, or stride alignment requirement for correctness.

## Dispatch

Dispatch is shared between two front ends. `lupine_smemcpy_prepare_launch`
provides the standalone CUDA Runtime API, while Lupine's server embeds a
multi-architecture fatbin and uses the Driver API-only helpers in
`smemcpy_module.h`. Both call the pure descriptor selection in
`smemcpy_dispatch.h`, so they select the same operation:

- native CUDA memcpy when the entire fragment is physically contiguous;
- pitched 16-, 8-, 4-, and 2-byte copies when no vector can cross a row;
- a rebased 2D kernel when the fragment remains in one slice;
- architecture- and product-tuned, padding-preserving atomic paths for
  one- to three-byte rows;
- latency-hiding narrow-row kernels for widths of one to three bytes; and
- a general byte-addressed 3D fallback.

The vector paths peel at most 127 initial bytes so full mapped-host transactions
begin on a 128-byte boundary. Packed rows and slices bypass the kernels and use
CUDA's native copy engine.

The embedded fatbin contains native code for every architecture supported by
the build's CUDA toolkit and PTX for its oldest architecture as a
forward-compatible fallback. Keeping module loading and launches on the Driver
API avoids introducing a CUDA Runtime dependency or a second context into the
server.

## Alignment for maximum throughput

Correctness never depends on alignment. Contiguous fragments use native CUDA
memcpy regardless of alignment. For the fastest 16-byte pitched kernel path,
the source and the destination coordinate of the fragment must have the same
low four address bits:

```text
(source ^ destination_at(logical_offset)) & 15 == 0
```

For the pitched 16-byte path, all of the following are also required:

- `destination`, `width`, `destination_row_stride`, and
  `destination_slice_stride` are multiples of 16; and
- the source and first destination coordinate are congruent modulo 16.

The dispatcher automatically falls back through 8, 4, 2, and 1-byte accesses.
CUDA allocations are strongly aligned, but an application-provided destination
offset can still break congruence.

### Narrow rows and atomic padding updates

A one-byte row with a 256-byte pitch is a pathological ordinary store pattern:
each useful byte occupies a different 32-byte sector. On selected GPUs, the
dispatcher instead updates the aligned 32-bit word containing that byte with
`atomicCAS`. This preserves the other three bytes while routing the update
through the L2 atomic path rather than serialized L1TEX store wavefronts.
For two- and three-byte rows, one CAS updates the entire useful row and preserves
the remaining bytes in the word.

The selection is deliberately architecture and product specific. The measured
power-of-two pitch ranges and simultaneous CAS lane counts are:

| GPU target | Width 1 | Width 2 | Width 3 |
| --- | --- | --- | --- |
| Pascal GP100 (SM 6.0) | 128–512: 4 | 64: 8; 128–512: 4 | 64: 8; 128–512: 4 |
| Volta (SM 7.0) | 128–512: 4 | 64: 16; 128: 8; 256–512: 4 | 64–128: 16; 256–512: 4 |
| Turing (SM 7.5) | 64: 8; 128–512: 4 | 128–256: 4 | 64: 16; 128–512: 4 |
| Ampere A100-class (SM 8.0) | 256–512: 8 | ordinary stores | ordinary stores |
| NVIDIA L4 (SM 8.9) | 64: 16; 128–512: 8 | 64–512: 16 | 64–512: 16 |

An entry such as `128–512: 4` means pitches 128, 256, and 512 bytes with four
active CAS lanes per warp group. Other pitches and widths use ordinary stores.

Ada is not enabled solely by compute capability: the full-link L4 benefits,
whereas the tested RTX 4090 is already limited by its PCIe x2 mapped-source
bandwidth and loses throughput with CAS. Unmeasured SM variants, including
Hopper, retain ordinary byte stores until measured.

There is no destination pointer alignment requirement. Four kernel variants
encode the destination's byte offset within its naturally aligned 32-bit word,
the first row uses scalar stores when aligning down would precede the supplied
destination pointer, and the final copied row is always scalar. Thus no source
or destination headroom is needed beyond the normal pitched extent. Widths two
and three take this path only for whole-row fragments whose useful bytes fit
within one aligned 32-bit word; a row that crosses a word boundary falls back
automatically. Interior atomic updates preserve padding values but do read and
atomically rewrite their containing words. Do not concurrently modify those
destination words from another stream, the host, or another device.

For fragments crossing slice boundaries, the atomic path additionally requires
`destination_slice_stride == rows * destination_row_stride`. Slice layouts with
extra gaps and non-power-of-two row pitches use the ordinary byte-store kernel;
their Turing L2 partition behavior made the atomic path slower in profiling.
These restrictions are performance dispatch rules, not pointer-alignment
requirements.

## Mitigating an unknown destination shape

Use a 128-byte-aligned ring base and round each slot stride to a multiple of
128. If the destination is not known when the ring is allocated, reserve 15
bytes of source headroom per slot. Once the copy descriptor is known, start the
payload at:

```text
source = slot_base + (destination_at(logical_offset) & 15)
```

This makes the source and first destination coordinate congruent without
moving the payload. Round the total slot size, including headroom, back to 128
bytes. The kernel's short scalar prefix then reaches the next 128-byte source
boundary before entering the vector loop.

When source chunking is under Lupine's control, prefer 128-byte-multiple chunk
sizes and begin a new chunk at a destination coordinate aligned to 16 bytes.
That keeps every ring slot 128-byte aligned and avoids even the short prefix.
The dispatcher evaluates the actual destination descriptor at launch time, so
an unknown pitch only changes which vector width is chosen; it does not require
a separate precompiled shape. Preserve the packed representation until that
descriptor is available, then cache the prepared launch by width, strides,
relative alignment, and GPU architecture.

If destination padding is owned by the caller and may be overwritten, another
option is to receive directly into a source slot with the same row and slice
gaps, then copy the entire physical span. Use this only when padding inflation
is small: it increases ring traffic and is not valid for CUDA memcpy semantics
when padding must remain untouched. For normal application-owned destinations,
the packed source plus scatter kernel is the safe path.

For extremely sparse destinations, the largest win is to avoid materializing
the gaps. Copy the packed bytes into a compact device allocation and either
teach the consumer to use a compact view or fuse the scatter address calculation
into the first consuming kernel. A separate compact HtoD copy followed by an
unfused scatter usually loses because it adds another full pass; fusion removes
the pathological standalone store entirely.

If destination allocation is under partial control, bucket narrow shapes onto
the power-of-two pitches in the architecture table. Prefer 128 or 256 bytes
when the eventual GPU is also unknown: 128 activates GP100, Volta, Turing, and
L4, while 256 additionally activates A100. If padding may be overwritten and
the physical/logical size ratio is modest, receiving a shape-expanded source
and copying the entire physical span can convert the operation back into a
contiguous transfer. Do not use that strategy for a one-byte/256-byte-pitch
tensor: its 256x traffic inflation is worse than the scatter.

If receive placement cannot be changed after the destination becomes known,
do not realign an already received large payload with another host copy. The
128-byte prefix peel keeps the scalar fallback close to peak: on the test RTX
4090, arbitrary relative alignment reached about 3.13 GB/s versus 3.18 GB/s for
the vector path. Splitting a wide transfer into one launch per row is also
usually counterproductive; native 2D/3D setup overhead can dominate narrow or
deeply sliced shapes.

For small fragments, kernel launch cost dominates. Batch adjacent packed
fragments into one ring slot/launch when ordering permits. On the measured
systems the scatter path approaches link bandwidth by roughly 256 KiB to 1
MiB, while transfers of only a few KiB are primarily launch-bound.

## Correctness testing

`smemcpy_test` initializes two identical device allocations, writes one with a
native CUDA copy and the other with scatter memcpy, then compares every byte of
the physical allocations. Padding begins with a sentinel value, so the test
also detects writes outside the logical region.

The suite contains 156 comparisons plus invalid-parameter checks:

- 21 packed and fragmented `cudaMemcpyAsync` 1D references;
- 34 full-plane `cudaMemcpy2DAsync` references;
- 16 full-volume `cudaMemcpy3DAsync` references with regular and gapped slice
  strides; and
- 85 targeted and deterministic-random fragmented references assembled from
  native 1D row copies.

Sizes around every vector boundary, mapped-source and destination offsets,
power-of-two and irregular pitches, one- to three-byte atomic rows, tight
allocation tails, partial rows, and cross-slice fragments are covered. Build
and run it directly or through CTest:

```sh
cmake --build build --target smemcpy_test
ctest --test-dir build -R smemcpy_test --output-on-failure
```

GPU-less test hosts return CTest skip code 77.

## Performance rationale

Nsight Compute on the SM 7.5 one-byte-row case reports that the ordinary kernel
had no eligible warp for about 97.6% of scheduler cycles: each warp's byte store
touched 32 destination sectors. The atomic kernel reduced an 8 MiB copy to
2.09 ms (about 4.0 GB/s under the profiler), reduced no-eligible cycles to
88.8%, and reached 88.9% of measured DRAM throughput. The useful-byte rate is
still far below packed HtoD because each byte necessarily causes a full-sector
destination transaction; without changing the destination representation,
that traffic amplification is the remaining bound.

On A100, Nsight Compute measured 448 GB/s of memory-system throughput for the
ordinary pitch-256 kernel versus 908 GB/s for the eight-lane CAS kernel. Warp
cycles per issued instruction fell from about 1531 to 279. This agrees with
Nsight Compute's model: a scattered warp store is decomposed into serialized
L1TEX wavefronts, while atomics are passed through to the L2 atomic hardware.

The implementation follows NVIDIA's guidance that mapped pinned memory should
be accessed once and with coalesced operations:

- [CUDA C++ Best Practices Guide: Zero Copy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy)
- [CUDA C++ Programming Guide: Device Memory Accesses](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#device-memory-accesses)
- [Nsight Compute Profiling Guide: Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-workload-analysis)
- [Pascal Tuning Guide: Atomic Memory Operations](https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#atomic-memory-operations)
- [Volta Tuning Guide: Memory Throughput](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html#memory-throughput)
- [Ampere Tuning Guide: Memory System](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#memory-system)
- [Hopper Tuning Guide: Memory System](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#memory-system)

# ROCm scatter memcpy

`hip_smemcpy.hip` is the same operation for AMD GPUs. It is a separate kernel,
not a portability layer over `smemcpy.cu`: the two hardware families reward
different code, and keeping them apart lets each be tuned without regressing
the other. The contract is identical — a packed, device-visible source
fragment scattered into a pitched 2D or 3D destination — and the parameter
struct differs only in that addresses are plain unsigned integers, because
`hipDeviceptr_t` is `void *` and cannot carry the operation's arithmetic.

## What differs from the CUDA operation

**No atomic paths.** The CUDA dispatcher routes one- to three-byte rows through
`atomicCAS` on selected NVIDIA parts, because a scattered warp store there is
decomposed into serialized L1TEX wavefronts while atomics pass through to the
L2 atomic hardware. That is an NVIDIA memory-system property. The whole CAS
apparatus — the per-product table, the four destination-byte-offset kernel
variants, the whole-row and word-containment rules — is absent here.

**No device query.** With no per-product table, dispatch depends only on the
descriptor. `lupine_hip_smemcpy_prepare_launch` is a pure function; there is no
device property lookup and no cache.

**No wave-width dependency.** The CUDA narrow kernels stagger stores across
lane groups within a warp. On gfx1100 that measured as noise, winning and
losing across repeats of the same shape, because these kernels are bound by
mapped-read request rate rather than by their stores. Leaving it out means
nothing here depends on the wave being 32 or 64 lanes wide, so the same source
is correct on RDNA and CDNA without an architecture table.

**Wide loads everywhere.** This is the change that matters on AMD.

## Wide loads

A one- to three-byte row sends every useful byte to a different destination
sector, so no store can be widened. The load can be. Each thread pulls one
aligned 8-byte word through the host link and unpacks it into that many
scattered byte stores, rather than issuing the same bytes as separate loads.
The scalar prefix that reaches a 128-byte source boundary is what makes the
wide load aligned, and the bulk loop never reads past the fragment.

Measured on an RX 7900 XTX (gfx1100), 8 MiB per copy, GB/s:

| Shape | Byte loads | 8-byte loads |
| --- | --- | --- |
| width 1, pitch 64 | 2.49 | 6.42 |
| width 2, pitch 256 | 2.45 | 6.19 |
| width 3, pitch 256 | 2.86 | 6.44 |
| width 5, pitch 13 | 2.28 | 6.43 |
| width 33, pitch 67 | 2.30 | 6.44 |
| width 1023, pitch 2047 | 2.29 | 6.45 |

The same treatment applies to the general `width > 3` fallback, which is why
the last three rows move as much as the narrow ones. Eight bytes beat both four
and sixteen across widths and pitches.

## Throughput

End to end through the dispatcher on the same part, 8 MiB per copy:

| Shape | GB/s |
| --- | --- |
| packed 4096 | 6.38 |
| pitched vector, width 64, pitch 128 | 6.43 |
| narrow width 1, pitch 64 | 6.42 |
| narrow width 1, pitch 256 | 5.45 |
| narrow width 1, pitch 512 | 3.74 |
| narrow width 3, pitch 256 | 6.44 |
| general width 127, pitch 255 | 6.44 |
| general width 1023, pitch 2047 | 6.45 |

A packed contiguous copy reaches 6.38 GB/s on this link, so almost every
scattered shape now runs at the same rate as one. The exception is a one-byte
row on a 512-byte pitch, where each useful byte necessarily causes its own
destination sector transaction; that shape stays store-bound at 3.74 GB/s.
Widening the destination representation is the only remaining lever there, as
on NVIDIA.

## Build

CMake enables the HIP operation when a HIP compiler is found. The probe is a
real try-compile, so the header-only container build — which copies HIP headers
but no compiler — skips it. The CUDA and HIP operations are independent and
either may be built alone.

The build defaults to a fat binary over the supported CDNA and RDNA targets.
`hipcc` alone emits `gfx906`, which loads but never runs on a newer card, and
clang's `native` needs `/dev/kfd`, so it fails on GPU-less builders and under
WSL. Override with `CMAKE_HIP_ARCHITECTURES`.

## Correctness testing

`hip_smemcpy_test` mirrors `smemcpy_test` case for case, comparing against
native `hipMemcpy`, `hipMemcpy2DAsync`, and `hipMemcpy3DAsync` references
rather than CUDA ones. The 156 comparisons plus the invalid-parameter checks
pass on an RX 7900 XTX. Sizes around every load and vector boundary matter more
here than on NVIDIA, since the unpacked wide load has its own prefix, bulk, and
tail boundaries.

```sh
cmake --build build --target hip_smemcpy_test
ctest --test-dir build -R hip_smemcpy_test --output-on-failure
```

GPU-less test hosts return CTest skip code 77.
