// Synthetic server encoder for exercising the RPC cache on pre-Hopper GPUs.
// This deliberately does not produce a usable hardware TMA descriptor.
#include <cstdint>
#include <cstring>
#include <cuda.h>

extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap *map, CUtensorMapDataType type, cuuint32_t rank, void *address,
    const cuuint64_t *dims, const cuuint64_t *strides, const cuuint32_t *box,
    const cuuint32_t *elements, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promotion,
    CUtensorMapFloatOOBfill fill) {
  if (int(swizzle) == 9999)
    return CUDA_ERROR_INVALID_VALUE;
  uint64_t hash = 14695981039346656037ull;
  auto bytes = [&](const void *data, size_t size) {
    auto p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i)
      hash = (hash ^ p[i]) * 1099511628211ull;
  };
  bytes(&type, sizeof(type));
  bytes(&rank, sizeof(rank));
  bytes(&address, sizeof(address));
  bytes(dims, rank * sizeof(*dims));
  if (rank > 1)
    bytes(strides, (rank - 1) * sizeof(*strides));
  bytes(box, rank * sizeof(*box));
  bytes(elements, rank * sizeof(*elements));
  bytes(&interleave, sizeof(interleave));
  bytes(&swizzle, sizeof(swizzle));
  bytes(&promotion, sizeof(promotion));
  bytes(&fill, sizeof(fill));
  for (size_t i = 0; i < sizeof(*map); i += sizeof(hash))
    std::memcpy(reinterpret_cast<unsigned char *>(map) + i, &hash,
                sizeof(hash));
  return CUDA_SUCCESS;
}
