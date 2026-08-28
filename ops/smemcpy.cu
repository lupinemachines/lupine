#include "ops/smemcpy.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr size_t kMaximumBlocks = 65535;
constexpr size_t kSourceTransactionAlignment = 128;
constexpr size_t kNarrowItemsPerThread = 4;

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
__global__ void smemcpy_pitched_vector_kernel(lupine_smemcpy_params params) {
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

__global__ void smemcpy_2d_kernel(lupine_smemcpy_params params) {
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
__global__ void smemcpy_narrow_2d_kernel(lupine_smemcpy_params params) {
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
__global__ void smemcpy_narrow_3d_kernel(lupine_smemcpy_params params) {
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
__global__ void smemcpy_atomic_narrow_2d_kernel(lupine_smemcpy_params params) {
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
__global__ void smemcpy_atomic_rows_kernel(lupine_smemcpy_params params) {
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

__global__ void smemcpy_3d_kernel(lupine_smemcpy_params params) {
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

size_t blocks_for(size_t bytes, size_t bytes_per_thread) {
  size_t block_bytes = kThreadsPerBlock * bytes_per_thread;
  size_t blocks = bytes / block_bytes + (bytes % block_bytes != 0);
  return std::min(std::max<size_t>(blocks, 1), kMaximumBlocks);
}

CUdeviceptr destination_at(const lupine_smemcpy_params &params,
                           size_t logical) {
  size_t row_index = logical / params.width;
  size_t x = logical - row_index * params.width;
  size_t slice = row_index / params.rows;
  size_t row = row_index - slice * params.rows;
  return params.destination + slice * params.destination_slice_stride +
         row * params.destination_row_stride + x;
}

bool contiguous_destination(const lupine_smemcpy_params &params,
                            size_t slice_bytes) {
  size_t first_row = params.logical_offset / params.width;
  size_t last_row = (params.logical_offset + params.bytes - 1) / params.width;
  if (first_row == last_row) {
    return true;
  }
  if (params.destination_row_stride != params.width) {
    return false;
  }
  return first_row / params.rows == last_row / params.rows ||
         params.destination_slice_stride == slice_bytes;
}

template <size_t WordBytes>
bool can_vectorize_pitched(const lupine_smemcpy_params &params,
                           CUdeviceptr first_destination) {
  constexpr size_t mask = WordBytes - 1;
  return (params.width & mask) == 0 && (params.destination & mask) == 0 &&
         (params.destination_row_stride & mask) == 0 &&
         (params.destination_slice_stride & mask) == 0 &&
         ((params.source ^ first_destination) & mask) == 0;
}

template <size_t WordBytes>
void set_pitched_vector_launch(lupine_smemcpy_launch *launch) {
  launch->kernel =
      reinterpret_cast<const void *>(smemcpy_pitched_vector_kernel<WordBytes>);
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, WordBytes));
}

bool select_pitched_vector_launch(lupine_smemcpy_launch *launch,
                                  CUdeviceptr first_destination) {
  if (can_vectorize_pitched<16>(launch->params, first_destination)) {
    set_pitched_vector_launch<16>(launch);
  } else if (can_vectorize_pitched<8>(launch->params, first_destination)) {
    set_pitched_vector_launch<8>(launch);
  } else if (can_vectorize_pitched<4>(launch->params, first_destination)) {
    set_pitched_vector_launch<4>(launch);
  } else if (can_vectorize_pitched<2>(launch->params, first_destination)) {
    set_pitched_vector_launch<2>(launch);
  } else {
    return false;
  }
  return true;
}

template <size_t Items>
void set_narrow_2d_launch(lupine_smemcpy_launch *launch, size_t width) {
  switch (width) {
  case 1:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_2d_kernel<1, Items>);
    break;
  case 2:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_2d_kernel<2, Items>);
    break;
  default:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_2d_kernel<3, Items>);
    break;
  }
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, Items));
}

template <size_t Items>
void set_narrow_3d_launch(lupine_smemcpy_launch *launch, size_t width) {
  switch (width) {
  case 1:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_3d_kernel<1, Items>);
    break;
  case 2:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_3d_kernel<2, Items>);
    break;
  default:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_3d_kernel<3, Items>);
    break;
  }
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, Items));
}

template <size_t Width, unsigned int ByteOffset, unsigned int GroupThreads>
void set_atomic_narrow_launch(lupine_smemcpy_launch *launch) {
  if constexpr (Width == 1) {
    launch->kernel = reinterpret_cast<const void *>(
        smemcpy_atomic_narrow_2d_kernel<ByteOffset, GroupThreads>);
  } else {
    launch->kernel = reinterpret_cast<const void *>(
        smemcpy_atomic_rows_kernel<Width, ByteOffset, GroupThreads>);
  }
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, Width));
}

template <size_t Width, unsigned int GroupThreads>
void select_atomic_narrow_launch_for_width(lupine_smemcpy_launch *launch) {
  constexpr size_t mask = sizeof(unsigned int) - 1;
  switch (launch->params.destination & mask) {
  case 0:
    set_atomic_narrow_launch<Width, 0, GroupThreads>(launch);
    break;
  case 1:
    set_atomic_narrow_launch<Width, 1, GroupThreads>(launch);
    break;
  case 2:
    if constexpr (Width <= 2) {
      set_atomic_narrow_launch<Width, 2, GroupThreads>(launch);
    }
    break;
  default:
    if constexpr (Width == 1) {
      set_atomic_narrow_launch<Width, 3, GroupThreads>(launch);
    }
    break;
  }
}

template <unsigned int GroupThreads>
void select_atomic_narrow_launch_for_group(lupine_smemcpy_launch *launch,
                                           size_t width) {
  switch (width) {
  case 1:
    select_atomic_narrow_launch_for_width<1, GroupThreads>(launch);
    break;
  case 2:
    select_atomic_narrow_launch_for_width<2, GroupThreads>(launch);
    break;
  default:
    select_atomic_narrow_launch_for_width<3, GroupThreads>(launch);
    break;
  }
}

void select_atomic_narrow_launch(lupine_smemcpy_launch *launch,
                                 unsigned int group_threads, size_t width) {
  if (group_threads == 16) {
    select_atomic_narrow_launch_for_group<16>(launch, width);
  } else if (group_threads == 8) {
    select_atomic_narrow_launch_for_group<8>(launch, width);
  } else {
    select_atomic_narrow_launch_for_group<4>(launch, width);
  }
}

enum class atomic_narrow_target {
  none,
  pascal,
  volta,
  turing,
  ampere,
  l4,
};

cudaError_t narrow_device_features(bool *latency_hiding,
                                   atomic_narrow_target *atomic_target) {
  int device = 0;
  cudaError_t result = cudaGetDevice(&device);
  if (result != cudaSuccess) {
    return result;
  }
  static thread_local int cached_device = -1;
  static thread_local int cached_major = 0;
  static thread_local atomic_narrow_target cached_atomic_target =
      atomic_narrow_target::none;
  if (cached_device != device) {
    cudaDeviceProp properties = {};
    result = cudaGetDeviceProperties(&properties, device);
    if (result != cudaSuccess) {
      return result;
    }
    cached_major = properties.major;
    cached_atomic_target = atomic_narrow_target::none;
    if (properties.major == 6 && properties.minor == 0) {
      cached_atomic_target = atomic_narrow_target::pascal;
    } else if (properties.major == 7 && properties.minor == 0) {
      cached_atomic_target = atomic_narrow_target::volta;
    } else if (properties.major == 7 && properties.minor == 5) {
      cached_atomic_target = atomic_narrow_target::turing;
    } else if (properties.major == 8 && properties.minor == 0) {
      cached_atomic_target = atomic_narrow_target::ampere;
    } else if (properties.major == 8 && properties.minor == 9 &&
               std::strcmp(properties.name, "NVIDIA L4") == 0) {
      cached_atomic_target = atomic_narrow_target::l4;
    }
    cached_device = device;
  }
  // Pre-Ampere devices benefit from several independent mapped reads. Newer
  // devices saturate the tested host links with one item per thread.
  *latency_hiding = cached_major < 8;
  *atomic_target = cached_atomic_target;
  return cudaSuccess;
}

unsigned int atomic_group_threads(atomic_narrow_target target, size_t width,
                                  size_t stride) {
  // This is an empirical policy, not a compute-capability extrapolation.
  if (stride < 64 || stride > 512 || (stride & (stride - 1)) != 0) {
    return 0;
  }
  switch (target) {
  case atomic_narrow_target::pascal:
    if (width == 1) {
      return stride >= 128 ? 4 : 0;
    }
    return stride == 64 ? 8 : 4;
  case atomic_narrow_target::volta:
    if (width == 1) {
      return stride >= 128 ? 4 : 0;
    }
    if (stride == 64 || (width == 3 && stride == 128)) {
      return 16;
    }
    return stride == 128 ? 8 : 4;
  case atomic_narrow_target::turing:
    if (width == 1) {
      return stride == 64 ? 8 : 4;
    }
    if (width == 2) {
      return stride == 128 || stride == 256 ? 4 : 0;
    }
    return stride == 64 ? 16 : 4;
  case atomic_narrow_target::ampere:
    return width == 1 && stride >= 256 ? 8 : 0;
  case atomic_narrow_target::l4:
    if (width == 1) {
      return stride == 64 ? 16 : 8;
    }
    return 16;
  case atomic_narrow_target::none:
    return 0;
  }
  return 0;
}

bool rows_fit_atomic_word(const lupine_smemcpy_params &params,
                          CUdeviceptr first_destination) {
  bool owns_complete_rows = params.logical_offset % params.width == 0 &&
                            params.bytes % params.width == 0;
  size_t byte_offset = first_destination & (sizeof(unsigned int) - 1);
  return params.width <= 3 && owns_complete_rows &&
         byte_offset + params.width <= sizeof(unsigned int);
}

} // namespace

extern "C" const void *lupine_smemcpy_kernel(void) {
  return reinterpret_cast<const void *>(smemcpy_3d_kernel);
}

extern "C" cudaError_t
lupine_smemcpy_prepare_launch(const lupine_smemcpy_params *params,
                              lupine_smemcpy_launch *launch) {
  if (params == nullptr || launch == nullptr || params->width == 0 ||
      params->rows == 0 ||
      (params->bytes != 0 &&
       (params->source == 0 || params->destination == 0)) ||
      params->width > std::numeric_limits<size_t>::max() / params->rows ||
      params->bytes >
          std::numeric_limits<size_t>::max() - params->logical_offset) {
    return cudaErrorInvalidValue;
  }

  launch->params = *params;
  launch->use_cuda_memcpy = false;
  launch->kernel = reinterpret_cast<const void *>(smemcpy_3d_kernel);
  launch->blocks = 0;
  launch->threads = kThreadsPerBlock;
  if (params->bytes == 0) {
    return cudaSuccess;
  }

  size_t slice_bytes = params->width * params->rows;
  CUdeviceptr first_destination =
      destination_at(*params, params->logical_offset);
  if (contiguous_destination(*params, slice_bytes)) {
    launch->params.destination = first_destination;
    launch->params.logical_offset = 0;
    launch->use_cuda_memcpy = true;
    launch->kernel = nullptr;
    return cudaSuccess;
  }

  size_t first_row_index = params->logical_offset / params->width;
  size_t first_slice = first_row_index / params->rows;
  size_t last_logical = params->logical_offset + params->bytes - 1;
  size_t last_row_index = last_logical / params->width;
  size_t last_slice = last_row_index / params->rows;
  bool latency_hiding_narrow = false;
  atomic_narrow_target atomic_target = atomic_narrow_target::none;
  unsigned int atomic_threads = 0;
  if (params->width <= 3) {
    cudaError_t result =
        narrow_device_features(&latency_hiding_narrow, &atomic_target);
    if (result != cudaSuccess) {
      return result;
    }
    atomic_threads = atomic_group_threads(atomic_target, params->width,
                                          params->destination_row_stride);
    if (atomic_threads != 0 &&
        !rows_fit_atomic_word(*params, first_destination)) {
      atomic_threads = 0;
    }
  }

  bool regular_rows_across_slices =
      params->destination_row_stride <=
          std::numeric_limits<size_t>::max() / params->rows &&
      params->destination_slice_stride ==
          params->destination_row_stride * params->rows;
  if (atomic_threads != 0 && first_slice != last_slice &&
      !regular_rows_across_slices) {
    atomic_threads = 0;
  }

  if (atomic_threads == 0 && !latency_hiding_narrow &&
      select_pitched_vector_launch(launch, first_destination)) {
    return cudaSuccess;
  }

  if (first_slice == last_slice) {
    size_t first_row = first_row_index - first_slice * params->rows;
    size_t first_x = params->logical_offset - first_row_index * params->width;
    launch->params.destination =
        params->destination + first_slice * params->destination_slice_stride +
        first_row * params->destination_row_stride;
    launch->params.logical_offset = first_x;
    if (params->width <= 3) {
      if (atomic_threads != 0) {
        select_atomic_narrow_launch(launch, atomic_threads, params->width);
      } else if (latency_hiding_narrow) {
        set_narrow_2d_launch<kNarrowItemsPerThread>(launch, params->width);
      } else {
        set_narrow_2d_launch<1>(launch, params->width);
      }
    } else {
      launch->kernel = reinterpret_cast<const void *>(smemcpy_2d_kernel);
      launch->blocks = static_cast<unsigned int>(blocks_for(params->bytes, 1));
    }
    return cudaSuccess;
  }

  if (first_slice != 0) {
    launch->params.destination +=
        first_slice * params->destination_slice_stride;
    launch->params.logical_offset -= first_slice * slice_bytes;
  }
  if (params->width <= 3) {
    if (atomic_threads != 0) {
      select_atomic_narrow_launch(launch, atomic_threads, params->width);
    } else if (latency_hiding_narrow) {
      set_narrow_3d_launch<kNarrowItemsPerThread>(launch, params->width);
    } else {
      set_narrow_3d_launch<1>(launch, params->width);
    }
  } else {
    launch->kernel = reinterpret_cast<const void *>(smemcpy_3d_kernel);
    launch->blocks = static_cast<unsigned int>(blocks_for(params->bytes, 1));
  }
  return cudaSuccess;
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
