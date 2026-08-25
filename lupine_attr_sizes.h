#ifndef LUPINE_ATTR_SIZES_H
#define LUPINE_ATTR_SIZES_H

#include <cstddef>
#include <cuda.h>

inline bool lupine_mem_pool_attribute_size(CUmemPool_attribute attr,
                                           size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
    *size = sizeof(int);
    return true;
  case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
  case CU_MEMPOOL_ATTR_USED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
    *size = sizeof(cuuint64_t);
    return true;
  default:
    return false;
  }
}

inline bool lupine_pointer_attribute_size(CUpointer_attribute attr,
                                          size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_POINTER_ATTRIBUTE_CONTEXT:
    *size = sizeof(CUcontext);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    *size = sizeof(void *);
    return true;
  case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
    *size = sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS);
    return true;
  case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
  case CU_POINTER_ATTRIBUTE_IS_MANAGED:
  case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
  case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
  case CU_POINTER_ATTRIBUTE_MAPPED:
  case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
  case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
#if CUDA_VERSION >= 13000
  case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
#endif
    *size = sizeof(int);
    return true;
  case CU_POINTER_ATTRIBUTE_BUFFER_ID:
  case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
    *size = sizeof(unsigned long long);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
  case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
  case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
    *size = sizeof(size_t);
    return true;
  case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
    *size = sizeof(CUmemoryPool);
    return true;
  default:
    return false;
  }
}

// Size of the buffer cuPointerSetAttribute reads through its `value` argument.
// Unlike cuPointerGetAttribute, only a subset of the pointer attributes can be
// written; as of CUDA 13.1 that subset is exactly
// CU_POINTER_ATTRIBUTE_SYNC_MEMOPS ("Set attributes on a previously allocated
// memory region" in cuda.h). Returning false for everything else keeps the
// remoting layer from reading a length the caller never provided, and lets both
// ends reject the call cleanly. Sizes come from the table above so the set and
// get paths can never disagree.
inline bool lupine_settable_pointer_attribute_size(CUpointer_attribute attr,
                                                   size_t *size) {
  switch (attr) {
  case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    return lupine_pointer_attribute_size(attr, size);
  default:
    return false;
  }
}

#endif
