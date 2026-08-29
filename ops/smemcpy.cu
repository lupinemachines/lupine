#include "ops/smemcpy.h"

#include <cstdint>
#include <cstring>

namespace {

constexpr size_t kSourceTransactionAlignment = 128;

template <size_t Bytes> struct word_type;
template <> struct word_type<1> {
  using type = unsigned char;
};
template <> struct word_type<2> {
  using type = unsigned short;
};
template <> struct word_type<4> {
  using type = unsigned int;
};
template <> struct word_type<8> {
  using type = unsigned long long;
};
template <> struct word_type<16> {
  using type = uint4;
};

__host__ __device__ __forceinline__ size_t source_prefix(CUdeviceptr source,
                                                         size_t bytes) {
  size_t prefix =
      (-static_cast<size_t>(source)) & (kSourceTransactionAlignment - 1);
  return prefix < bytes ? prefix : bytes;
}

__device__ __forceinline__ void
scatter_byte(const lupine_smemcpy_params &params,
             const unsigned char *__restrict__ source,
             unsigned char *__restrict__ destination, size_t index) {
  size_t logical = params.logical_offset + index;
  size_t row_index = logical / params.width;
  size_t x = logical - row_index * params.width;
  size_t slice = row_index / params.rows;
  size_t row = row_index - slice * params.rows;
  destination[slice * params.destination_slice_stride +
              row * params.destination_row_stride + x] = source[index];
}

__device__ __forceinline__ void
scatter_2d_byte(const lupine_smemcpy_params &params,
                const unsigned char *__restrict__ source,
                unsigned char *__restrict__ destination, size_t index) {
  size_t logical = params.logical_offset + index;
  size_t row = logical / params.width;
  size_t x = logical - row * params.width;
  destination[row * params.destination_row_stride + x] = source[index];
}

template <size_t Width>
__device__ __forceinline__ void
scatter_narrow_store(const lupine_smemcpy_params &params,
                     unsigned char *__restrict__ destination, size_t index,
                     unsigned char value) {
  size_t logical = params.logical_offset + index;
  size_t row = logical / Width;
  size_t x = logical - row * Width;
  destination[row * params.destination_row_stride + x] = value;
}

template <size_t Width>
__device__ __forceinline__ void
scatter_narrow_3d_store(const lupine_smemcpy_params &params,
                        unsigned char *__restrict__ destination, size_t index,
                        unsigned char value) {
  size_t logical = params.logical_offset + index;
  size_t row_index = logical / Width;
  size_t x = logical - row_index * Width;
  size_t slice = row_index / params.rows;
  size_t row = row_index - slice * params.rows;
  destination[slice * params.destination_slice_stride +
              row * params.destination_row_stride + x] = value;
}

template <unsigned int ByteOffset>
__device__ __forceinline__ void
atomic_store_byte(unsigned char *__restrict__ destination,
                  const unsigned char *__restrict__ accessible_begin,
                  unsigned char value) {
  static_assert(ByteOffset < sizeof(unsigned int));
  if constexpr (ByteOffset != 0) {
    if (destination < accessible_begin + ByteOffset) {
      *destination = value;
      return;
    }
  }
  auto *word = reinterpret_cast<unsigned int *>(destination - ByteOffset);
  constexpr unsigned int shift = ByteOffset * 8;
  constexpr unsigned int value_mask = 0xffU << shift;
  unsigned int old = *word;
  while (true) {
    unsigned int replacement =
        (old & ~value_mask) | (static_cast<unsigned int>(value) << shift);
    unsigned int observed = atomicCAS(word, old, replacement);
    if (observed == old) {
      return;
    }
    old = observed;
  }
}

template <size_t Width, unsigned int ByteOffset>
__device__ __forceinline__ void
atomic_store_row(unsigned char *__restrict__ destination,
                 const unsigned char *__restrict__ accessible_begin,
                 unsigned int value) {
  static_assert(Width == 2 || Width == 3);
  static_assert(ByteOffset + Width <= sizeof(unsigned int));
  if constexpr (ByteOffset != 0) {
    if (destination < accessible_begin + ByteOffset) {
#pragma unroll
      for (size_t x = 0; x < Width; ++x) {
        destination[x] = static_cast<unsigned char>(value >> (x * 8));
      }
      return;
    }
  }
  auto *word = reinterpret_cast<unsigned int *>(destination - ByteOffset);
  constexpr unsigned int shift = ByteOffset * 8;
  constexpr unsigned int value_mask = ((1U << (Width * 8)) - 1) << shift;
  value <<= shift;
  unsigned int old = *word;
  while (true) {
    unsigned int replacement = (old & ~value_mask) | value;
    unsigned int observed = atomicCAS(word, old, replacement);
    if (observed == old) {
      return;
    }
    old = observed;
  }
}

template <unsigned int ByteOffset>
__device__ __forceinline__ void
scatter_atomic_2d_store(const lupine_smemcpy_params &params,
                        unsigned char *__restrict__ destination, size_t index,
                        unsigned char value) {
  size_t row = params.logical_offset + index;
  atomic_store_byte<ByteOffset>(
      destination + row * params.destination_row_stride, destination, value);
}

template <size_t WordBytes>
__device__ __forceinline__ void
smemcpy_pitched_vector_body(lupine_smemcpy_params params) {
  using word = typename word_type<WordBytes>::type;
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_byte(params, source, destination, thread);
  }

  size_t bulk_bytes = (params.bytes - prefix) & ~(WordBytes - 1);
  for (size_t index = prefix + thread * WordBytes; index < prefix + bulk_bytes;
       index += step * WordBytes) {
    size_t logical = params.logical_offset + index;
    size_t row_index = logical / params.width;
    size_t x = logical - row_index * params.width;
    size_t slice = row_index / params.rows;
    size_t row = row_index - slice * params.rows;
    auto *output = destination + slice * params.destination_slice_stride +
                   row * params.destination_row_stride + x;
    *reinterpret_cast<word *>(output) =
        *reinterpret_cast<const word *>(source + index);
  }

  size_t tail = params.bytes - prefix - bulk_bytes;
  if (thread < tail) {
    scatter_byte(params, source, destination, prefix + bulk_bytes + thread);
  }
}

__device__ __forceinline__ void smemcpy_2d_body(lupine_smemcpy_params params) {
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_2d_byte(params, source, destination, thread);
  }
  for (size_t index = prefix + thread; index < params.bytes; index += step) {
    scatter_2d_byte(params, source, destination, index);
  }
}

template <size_t Width, size_t Items>
__device__ __forceinline__ void
smemcpy_narrow_2d_body(lupine_smemcpy_params params) {
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_narrow_store<Width>(params, destination, thread, source[thread]);
  }

  // Mapped host loads have PCIe-scale latency. Issue several independent,
  // coalesced loads before consuming any of them so narrow, sector-scattered
  // stores do not leave every resident warp waiting on the same dependency.
  unsigned char values[Items];
#pragma unroll
  for (size_t item = 0; item < Items; ++item) {
    size_t index = prefix + thread + item * step;
    if (index < params.bytes) {
      values[item] = source[index];
    }
  }
#pragma unroll
  for (size_t item = 0; item < Items; ++item) {
    size_t index = prefix + thread + item * step;
    if constexpr (Items == 1 || Width == 3) {
      if (index < params.bytes) {
        scatter_narrow_store<Width>(params, destination, index, values[item]);
      }
    } else {
      constexpr unsigned int group_threads = Width == 1 ? 4 : 8;
      unsigned int lane = threadIdx.x & (warpSize - 1);
#pragma unroll
      for (unsigned int group = 0; group < warpSize; group += group_threads) {
        if (lane >= group && lane < group + group_threads &&
            index < params.bytes) {
          scatter_narrow_store<Width>(params, destination, index, values[item]);
        }
      }
    }
  }
}

template <size_t Width, size_t Items>
__device__ __forceinline__ void
smemcpy_narrow_3d_body(lupine_smemcpy_params params) {
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_narrow_3d_store<Width>(params, destination, thread, source[thread]);
  }
  unsigned char values[Items];
#pragma unroll
  for (size_t item = 0; item < Items; ++item) {
    size_t index = prefix + thread + item * step;
    if (index < params.bytes) {
      values[item] = source[index];
    }
  }
#pragma unroll
  for (size_t item = 0; item < Items; ++item) {
    size_t index = prefix + thread + item * step;
    if constexpr (Items == 1 || Width == 3) {
      if (index < params.bytes) {
        scatter_narrow_3d_store<Width>(params, destination, index,
                                       values[item]);
      }
    } else {
      constexpr unsigned int group_threads = Width == 1 ? 4 : 8;
      unsigned int lane = threadIdx.x & (warpSize - 1);
#pragma unroll
      for (unsigned int group = 0; group < warpSize; group += group_threads) {
        if (lane >= group && lane < group + group_threads &&
            index < params.bytes) {
          scatter_narrow_3d_store<Width>(params, destination, index,
                                         values[item]);
        }
      }
    }
  }
}

template <unsigned int ByteOffset, unsigned int GroupThreads>
__device__ __forceinline__ void
smemcpy_atomic_narrow_2d_body(lupine_smemcpy_params params) {
  static_assert(GroupThreads == 4 || GroupThreads == 8 || GroupThreads == 16);
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    if (thread + 1 == params.bytes) {
      scatter_narrow_store<1>(params, destination, thread, source[thread]);
    } else {
      scatter_atomic_2d_store<ByteOffset>(params, destination, thread,
                                          source[thread]);
    }
  }
  for (size_t index = prefix + thread; index < params.bytes; index += step) {
    unsigned char value = source[index];
    if (index + 1 == params.bytes) {
      scatter_narrow_store<1>(params, destination, index, value);
      continue;
    }
    unsigned int lane = threadIdx.x & (warpSize - 1);
#pragma unroll
    for (unsigned int group = 0; group < warpSize; group += GroupThreads) {
      if (lane >= group && lane < group + GroupThreads) {
        scatter_atomic_2d_store<ByteOffset>(params, destination, index, value);
      }
    }
  }
}

template <size_t Width, unsigned int ByteOffset, unsigned int GroupThreads>
__device__ __forceinline__ void
smemcpy_atomic_rows_body(lupine_smemcpy_params params) {
  static_assert(Width == 2 || Width == 3);
  static_assert(GroupThreads == 4 || GroupThreads == 8 || GroupThreads == 16);
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t copy_rows = params.bytes / Width;
  unsigned int lane = threadIdx.x & (warpSize - 1);
  for (size_t copy_row = thread; copy_row < copy_rows; copy_row += step) {
    size_t source_index = copy_row * Width;
    unsigned int value = 0;
#pragma unroll
    for (size_t x = 0; x < Width; ++x) {
      value |= static_cast<unsigned int>(source[source_index + x]) << (x * 8);
    }

    size_t destination_row = params.logical_offset / Width + copy_row;
    unsigned char *output =
        destination + destination_row * params.destination_row_stride;
    if (copy_row + 1 == copy_rows) {
#pragma unroll
      for (size_t x = 0; x < Width; ++x) {
        output[x] = static_cast<unsigned char>(value >> (x * 8));
      }
      continue;
    }
#pragma unroll
    for (unsigned int group = 0; group < warpSize; group += GroupThreads) {
      if (lane >= group && lane < group + GroupThreads) {
        atomic_store_row<Width, ByteOffset>(output, destination, value);
      }
    }
  }
}

__device__ __forceinline__ void smemcpy_3d_body(lupine_smemcpy_params params) {
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_byte(params, source, destination, thread);
  }
  for (size_t index = prefix + thread; index < params.bytes; index += step) {
    scatter_byte(params, source, destination, index);
  }
}

} // namespace

#define LUPINE_SMEMCPY_KERNEL(name, ...)                                       \
  extern "C" __global__ void name(lupine_smemcpy_params params) { __VA_ARGS__; }

#define LUPINE_SMEMCPY_PITCHED(bytes)                                          \
  LUPINE_SMEMCPY_KERNEL(lupine_smemcpy_pitched_##bytes,                        \
                        smemcpy_pitched_vector_body<bytes>(params))
LUPINE_SMEMCPY_PITCHED(2)
LUPINE_SMEMCPY_PITCHED(4)
LUPINE_SMEMCPY_PITCHED(8)
LUPINE_SMEMCPY_PITCHED(16)

LUPINE_SMEMCPY_KERNEL(lupine_smemcpy_2d, smemcpy_2d_body(params))
LUPINE_SMEMCPY_KERNEL(lupine_smemcpy_3d, smemcpy_3d_body(params))

#define LUPINE_SMEMCPY_NARROW(dimension, width, items)                         \
  LUPINE_SMEMCPY_KERNEL(                                                       \
      lupine_smemcpy_narrow_##dimension##_##width##_##items,                   \
      smemcpy_narrow_##dimension##_body<width, items>(params))
#define LUPINE_SMEMCPY_NARROW_WIDTH(dimension, width)                          \
  LUPINE_SMEMCPY_NARROW(dimension, width, 1)                                   \
  LUPINE_SMEMCPY_NARROW(dimension, width, 4)
#define LUPINE_SMEMCPY_NARROW_DIMENSION(dimension)                             \
  LUPINE_SMEMCPY_NARROW_WIDTH(dimension, 1)                                    \
  LUPINE_SMEMCPY_NARROW_WIDTH(dimension, 2)                                    \
  LUPINE_SMEMCPY_NARROW_WIDTH(dimension, 3)
LUPINE_SMEMCPY_NARROW_DIMENSION(2d)
LUPINE_SMEMCPY_NARROW_DIMENSION(3d)

#define LUPINE_SMEMCPY_ATOMIC_2D(offset, group)                                \
  LUPINE_SMEMCPY_KERNEL(lupine_smemcpy_atomic_2d_##offset##_##group,           \
                        smemcpy_atomic_narrow_2d_body<offset, group>(params))
#define LUPINE_SMEMCPY_ATOMIC_2D_GROUP(group)                                  \
  LUPINE_SMEMCPY_ATOMIC_2D(0, group)                                           \
  LUPINE_SMEMCPY_ATOMIC_2D(1, group)                                           \
  LUPINE_SMEMCPY_ATOMIC_2D(2, group)                                           \
  LUPINE_SMEMCPY_ATOMIC_2D(3, group)
LUPINE_SMEMCPY_ATOMIC_2D_GROUP(4)
LUPINE_SMEMCPY_ATOMIC_2D_GROUP(8)
LUPINE_SMEMCPY_ATOMIC_2D_GROUP(16)

#define LUPINE_SMEMCPY_ATOMIC_ROWS(width, offset, group)                       \
  LUPINE_SMEMCPY_KERNEL(                                                       \
      lupine_smemcpy_atomic_rows_##width##_##offset##_##group,                 \
      smemcpy_atomic_rows_body<width, offset, group>(params))
#define LUPINE_SMEMCPY_ATOMIC_ROWS_WIDTH_GROUP(width, group)                   \
  LUPINE_SMEMCPY_ATOMIC_ROWS(width, 0, group)                                  \
  LUPINE_SMEMCPY_ATOMIC_ROWS(width, 1, group)
#define LUPINE_SMEMCPY_ATOMIC_ROWS_GROUP(group)                                \
  LUPINE_SMEMCPY_ATOMIC_ROWS_WIDTH_GROUP(2, group)                             \
  LUPINE_SMEMCPY_ATOMIC_ROWS(2, 2, group)                                      \
  LUPINE_SMEMCPY_ATOMIC_ROWS_WIDTH_GROUP(3, group)
LUPINE_SMEMCPY_ATOMIC_ROWS_GROUP(4)
LUPINE_SMEMCPY_ATOMIC_ROWS_GROUP(8)
LUPINE_SMEMCPY_ATOMIC_ROWS_GROUP(16)

#undef LUPINE_SMEMCPY_ATOMIC_ROWS_GROUP
#undef LUPINE_SMEMCPY_ATOMIC_ROWS_WIDTH_GROUP
#undef LUPINE_SMEMCPY_ATOMIC_ROWS
#undef LUPINE_SMEMCPY_ATOMIC_2D_GROUP
#undef LUPINE_SMEMCPY_ATOMIC_2D
#undef LUPINE_SMEMCPY_NARROW_DIMENSION
#undef LUPINE_SMEMCPY_NARROW_WIDTH
#undef LUPINE_SMEMCPY_NARROW
#undef LUPINE_SMEMCPY_PITCHED
#undef LUPINE_SMEMCPY_KERNEL

namespace {

struct kernel_entry {
  const char *name;
  const void *function;
};

#define LUPINE_SMEMCPY_ENTRY(name)                                             \
  { #name, reinterpret_cast<const void *>(name) }
#define LUPINE_SMEMCPY_PITCHED_ENTRY(bytes)                                    \
  LUPINE_SMEMCPY_ENTRY(lupine_smemcpy_pitched_##bytes)
#define LUPINE_SMEMCPY_NARROW_ENTRY(dimension, width, items)                   \
  LUPINE_SMEMCPY_ENTRY(lupine_smemcpy_narrow_##dimension##_##width##_##items)
#define LUPINE_SMEMCPY_ATOMIC_2D_ENTRY(offset, group)                          \
  LUPINE_SMEMCPY_ENTRY(lupine_smemcpy_atomic_2d_##offset##_##group)
#define LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY(width, offset, group)                 \
  LUPINE_SMEMCPY_ENTRY(lupine_smemcpy_atomic_rows_##width##_##offset##_##group)

const kernel_entry kKernels[] = {
    LUPINE_SMEMCPY_ENTRY(lupine_smemcpy_3d),
    LUPINE_SMEMCPY_ENTRY(lupine_smemcpy_2d),
    LUPINE_SMEMCPY_PITCHED_ENTRY(2),
    LUPINE_SMEMCPY_PITCHED_ENTRY(4),
    LUPINE_SMEMCPY_PITCHED_ENTRY(8),
    LUPINE_SMEMCPY_PITCHED_ENTRY(16),
    LUPINE_SMEMCPY_NARROW_ENTRY(2d, 1, 1),
    LUPINE_SMEMCPY_NARROW_ENTRY(2d, 1, 4),
    LUPINE_SMEMCPY_NARROW_ENTRY(2d, 2, 1),
    LUPINE_SMEMCPY_NARROW_ENTRY(2d, 2, 4),
    LUPINE_SMEMCPY_NARROW_ENTRY(2d, 3, 1),
    LUPINE_SMEMCPY_NARROW_ENTRY(2d, 3, 4),
    LUPINE_SMEMCPY_NARROW_ENTRY(3d, 1, 1),
    LUPINE_SMEMCPY_NARROW_ENTRY(3d, 1, 4),
    LUPINE_SMEMCPY_NARROW_ENTRY(3d, 2, 1),
    LUPINE_SMEMCPY_NARROW_ENTRY(3d, 2, 4),
    LUPINE_SMEMCPY_NARROW_ENTRY(3d, 3, 1),
    LUPINE_SMEMCPY_NARROW_ENTRY(3d, 3, 4),
#define LUPINE_SMEMCPY_ATOMIC_2D_OFFSET(group)                                 \
  LUPINE_SMEMCPY_ATOMIC_2D_ENTRY(0, group),                                    \
      LUPINE_SMEMCPY_ATOMIC_2D_ENTRY(1, group),                                \
      LUPINE_SMEMCPY_ATOMIC_2D_ENTRY(2, group),                                \
      LUPINE_SMEMCPY_ATOMIC_2D_ENTRY(3, group)
    LUPINE_SMEMCPY_ATOMIC_2D_OFFSET(4),
    LUPINE_SMEMCPY_ATOMIC_2D_OFFSET(8),
    LUPINE_SMEMCPY_ATOMIC_2D_OFFSET(16),
#undef LUPINE_SMEMCPY_ATOMIC_2D_OFFSET
#define LUPINE_SMEMCPY_ATOMIC_ROWS_OFFSET(group)                               \
  LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY(2, 0, group),                               \
      LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY(2, 1, group),                           \
      LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY(2, 2, group),                           \
      LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY(3, 0, group),                           \
      LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY(3, 1, group)
    LUPINE_SMEMCPY_ATOMIC_ROWS_OFFSET(4),
    LUPINE_SMEMCPY_ATOMIC_ROWS_OFFSET(8),
    LUPINE_SMEMCPY_ATOMIC_ROWS_OFFSET(16),
#undef LUPINE_SMEMCPY_ATOMIC_ROWS_OFFSET
};

#undef LUPINE_SMEMCPY_ATOMIC_ROWS_ENTRY
#undef LUPINE_SMEMCPY_ATOMIC_2D_ENTRY
#undef LUPINE_SMEMCPY_NARROW_ENTRY
#undef LUPINE_SMEMCPY_PITCHED_ENTRY
#undef LUPINE_SMEMCPY_ENTRY

const void *runtime_kernel(const lupine_smemcpy_launch_descriptor &descriptor) {
  const char *name = lupine_smemcpy_kernel_name(&descriptor);
  if (name == nullptr) {
    return nullptr;
  }
  for (const auto &entry : kKernels) {
    if (std::strcmp(name, entry.name) == 0) {
      return entry.function;
    }
  }
  return nullptr;
}

} // namespace

extern "C" const void *lupine_smemcpy_kernel(void) {
  return reinterpret_cast<const void *>(lupine_smemcpy_3d);
}

extern "C" cudaError_t
lupine_smemcpy_prepare_launch(const lupine_smemcpy_params *params,
                              lupine_smemcpy_launch *launch) {
  if (launch == nullptr) {
    return cudaErrorInvalidValue;
  }
  lupine_smemcpy_launch_descriptor descriptor = {};
  CUresult driver_result =
      lupine_smemcpy_prepare_descriptor(params, nullptr, &descriptor);
  if (driver_result != CUDA_SUCCESS) {
    return static_cast<cudaError_t>(driver_result);
  }
  if (params->bytes != 0 && params->width <= 3 && !descriptor.use_cuda_memcpy) {
    int device = 0;
    cudaError_t result = cudaGetDevice(&device);
    if (result != cudaSuccess) {
      return result;
    }
    cudaDeviceProp properties = {};
    result = cudaGetDeviceProperties(&properties, device);
    if (result != cudaSuccess) {
      return result;
    }
    lupine_smemcpy_device features = {
        properties.major, properties.minor,
        properties.major == 8 && properties.minor == 9 &&
            std::strcmp(properties.name, "NVIDIA L4") == 0};
    driver_result =
        lupine_smemcpy_prepare_descriptor(params, &features, &descriptor);
    if (driver_result != CUDA_SUCCESS) {
      return static_cast<cudaError_t>(driver_result);
    }
  }

  launch->params = descriptor.params;
  launch->use_cuda_memcpy = descriptor.use_cuda_memcpy;
  launch->kernel = runtime_kernel(descriptor);
  launch->blocks = descriptor.blocks;
  launch->threads = descriptor.threads;
  return descriptor.blocks != 0 && !descriptor.use_cuda_memcpy &&
                 launch->kernel == nullptr
             ? cudaErrorInvalidDeviceFunction
             : cudaSuccess;
}

extern "C" cudaError_t lupine_smemcpy_async(const lupine_smemcpy_params *params,
                                            cudaStream_t stream) {
  lupine_smemcpy_launch launch = {};
  cudaError_t result = lupine_smemcpy_prepare_launch(params, &launch);
  if (result != cudaSuccess) {
    return result;
  }

  if (launch.use_cuda_memcpy) {
    return cudaMemcpyAsync(reinterpret_cast<void *>(launch.params.destination),
                           reinterpret_cast<const void *>(launch.params.source),
                           launch.params.bytes, cudaMemcpyDefault, stream);
  }
  if (launch.blocks == 0) {
    return cudaSuccess;
  }

  void *arguments[] = {&launch.params};
  return cudaLaunchKernel(launch.kernel, dim3(launch.blocks),
                          dim3(launch.threads), arguments, 0, stream);
}
