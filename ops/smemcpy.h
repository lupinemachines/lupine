#ifndef LUPINE_OPS_SMEMCPY_H
#define LUPINE_OPS_SMEMCPY_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Describes a packed fragment of a logically 3D byte region. The source is a
// device-visible address, normally a mapped pinned-host ring slot. Destination
// points at logical coordinate (0, 0, 0); its row and slice strides retain the
// discontinuities that were removed while packing the source.
//
// There is no alignment requirement for correctness. A 128-byte-aligned ring
// base and slot stride minimize the scalar prefix. Matching the low four bits
// of source and the first destination coordinate enables 16-byte accesses when
// the destination width and strides also permit them. Each pitched row must be
// fully readable and writable through destination_row_stride bytes; the
// architecture-selected narrow-row fast path preserves padding but atomically
// accesses its containing 4-byte word. Width-2/3 atomic rows must be whole and
// must not cross that word. See ops/README.md.
typedef struct lupine_smemcpy_params {
  CUdeviceptr destination;
  CUdeviceptr source;
  size_t logical_offset;
  size_t bytes;
  size_t width;
  size_t rows;
  size_t destination_row_stride;
  size_t destination_slice_stride;
} lupine_smemcpy_params;

// Fully describes the kernel launch selected for a set of parameters. Pass the
// address of params as the kernel's sole argument. The prepared parameters may
// be rebased relative to the input.
typedef struct lupine_smemcpy_launch {
  lupine_smemcpy_params params;
  const void *kernel;
  unsigned int blocks;
  unsigned int threads;
} lupine_smemcpy_launch;

// Selects the best safe implementation and launch geometry for the current
// CUDA device. This is useful to graph code constructing or updating an
// equivalent kernel node.
cudaError_t lupine_smemcpy_prepare_launch(const lupine_smemcpy_params *params,
                                          lupine_smemcpy_launch *launch);

// Enqueues one scatter copy. Source byte i is written to the pitched
// destination coordinate represented by logical_offset + i.
cudaError_t lupine_smemcpy_async(const lupine_smemcpy_params *params,
                                 cudaStream_t stream);

// Returns the fully general scalar kernel. Prefer prepare_launch: this symbol
// is retained as a compatibility fallback for graph code that controls its own
// launch geometry.
const void *lupine_smemcpy_kernel(void);

#ifdef __cplusplus
}
#endif

#endif
