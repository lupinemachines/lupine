#ifndef LUPINE_OPS_SMEMCPY_DISPATCH_H
#define LUPINE_OPS_SMEMCPY_DISPATCH_H

#include <cuda.h>

#include <stdbool.h>
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

typedef struct lupine_smemcpy_device {
  int major;
  int minor;
  bool is_l4;
} lupine_smemcpy_device;

enum lupine_smemcpy_kernel_kind {
  LUPINE_SMEMCPY_KERNEL_3D,
  LUPINE_SMEMCPY_KERNEL_2D,
  LUPINE_SMEMCPY_KERNEL_PITCHED,
  LUPINE_SMEMCPY_KERNEL_NARROW_2D,
  LUPINE_SMEMCPY_KERNEL_NARROW_3D,
  LUPINE_SMEMCPY_KERNEL_ATOMIC_NARROW_2D,
  LUPINE_SMEMCPY_KERNEL_ATOMIC_ROWS,
};

// A runtime-independent launch decision. The variant fields describe the
// template specialization selected by the dispatcher.
typedef struct lupine_smemcpy_launch_descriptor {
  lupine_smemcpy_params params;
  bool use_cuda_memcpy;
  enum lupine_smemcpy_kernel_kind kernel;
  unsigned int blocks;
  unsigned int threads;
  unsigned int word_bytes;
  unsigned int width;
  unsigned int items;
  unsigned int byte_offset;
  unsigned int group_threads;
} lupine_smemcpy_launch_descriptor;

// Selects a launch without touching CUDA runtime state. Device properties are
// needed only by noncontiguous one- to three-byte rows; null selects the safe
// architecture-neutral path.
CUresult
lupine_smemcpy_prepare_descriptor(const lupine_smemcpy_params *params,
                                  const lupine_smemcpy_device *device,
                                  lupine_smemcpy_launch_descriptor *launch);

// Stable C-linkage name exported by the precompiled CUDA fatbin.
const char *
lupine_smemcpy_kernel_name(const lupine_smemcpy_launch_descriptor *launch);

#ifdef __cplusplus
}
#endif

#endif
