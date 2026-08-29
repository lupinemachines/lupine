#include "ops/hip_smemcpy.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr size_t kMaximumBlocks = 65535;
constexpr size_t kSourceTransactionAlignment = 128;
// Every scatter kernel below pulls the source through one wide load per
// thread. Measured on gfx1100, 8 bytes held ~6.4 GB/s across narrow and
// general shapes alike, ahead of both 4- and 16-byte loads.
constexpr size_t kScatterLoadBytes = 8;

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

__host__ __device__ __forceinline__ size_t
source_prefix(unsigned long long source, size_t bytes) {
  size_t prefix =
      (-static_cast<size_t>(source)) & (kSourceTransactionAlignment - 1);
  return prefix < bytes ? prefix : bytes;
}

__device__ __forceinline__ void
scatter_byte(const lupine_hip_smemcpy_params &params,
             unsigned char *__restrict__ destination, size_t index,
             unsigned char value) {
  size_t logical = params.logical_offset + index;
  size_t row_index = logical / params.width;
  size_t x = logical - row_index * params.width;
  size_t slice = row_index / params.rows;
  size_t row = row_index - slice * params.rows;
  destination[slice * params.destination_slice_stride +
              row * params.destination_row_stride + x] = value;
}

__device__ __forceinline__ void
scatter_2d_byte(const lupine_hip_smemcpy_params &params,
                unsigned char *__restrict__ destination, size_t index,
                unsigned char value) {
  size_t logical = params.logical_offset + index;
  size_t row = logical / params.width;
  size_t x = logical - row * params.width;
  destination[row * params.destination_row_stride + x] = value;
}

template <size_t Width>
__device__ __forceinline__ void
scatter_narrow_store(const lupine_hip_smemcpy_params &params,
                     unsigned char *__restrict__ destination, size_t index,
                     unsigned char value) {
  size_t logical = params.logical_offset + index;
  size_t row = logical / Width;
  size_t x = logical - row * Width;
  destination[row * params.destination_row_stride + x] = value;
}

template <size_t Width>
__device__ __forceinline__ void
scatter_narrow_3d_store(const lupine_hip_smemcpy_params &params,
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

template <size_t WordBytes>
__global__ void smemcpy_linear_kernel(lupine_hip_smemcpy_params params) {
  using word = typename word_type<WordBytes>::type;
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination) +
      params.logical_offset;
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    destination[thread] = source[thread];
  }

  size_t bulk_bytes = (params.bytes - prefix) & ~(WordBytes - 1);
  for (size_t index = prefix + thread * WordBytes; index < prefix + bulk_bytes;
       index += step * WordBytes) {
    *reinterpret_cast<word *>(destination + index) =
        *reinterpret_cast<const word *>(source + index);
  }

  size_t tail = params.bytes - prefix - bulk_bytes;
  if (thread < tail) {
    size_t index = prefix + bulk_bytes + thread;
    destination[index] = source[index];
  }
}

template <size_t WordBytes>
__global__ void
smemcpy_pitched_vector_kernel(lupine_hip_smemcpy_params params) {
  using word = typename word_type<WordBytes>::type;
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_byte(params, destination, thread, source[thread]);
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
    size_t index = prefix + bulk_bytes + thread;
    scatter_byte(params, destination, index, source[index]);
  }
}

template <size_t LoadBytes>
__global__ void smemcpy_2d_kernel(lupine_hip_smemcpy_params params) {
  using word = typename word_type<LoadBytes>::type;
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_2d_byte(params, destination, thread, source[thread]);
  }

  size_t bulk_bytes = (params.bytes - prefix) & ~(LoadBytes - 1);
  for (size_t index = prefix + thread * LoadBytes; index < prefix + bulk_bytes;
       index += step * LoadBytes) {
    word packed = *reinterpret_cast<const word *>(source + index);
    const auto *unpacked = reinterpret_cast<const unsigned char *>(&packed);
#pragma unroll
    for (size_t byte = 0; byte < LoadBytes; ++byte) {
      scatter_2d_byte(params, destination, index + byte, unpacked[byte]);
    }
  }

  size_t tail = params.bytes - prefix - bulk_bytes;
  if (thread < tail) {
    size_t index = prefix + bulk_bytes + thread;
    scatter_2d_byte(params, destination, index, source[index]);
  }
}

template <size_t LoadBytes>
__global__ void smemcpy_3d_kernel(lupine_hip_smemcpy_params params) {
  using word = typename word_type<LoadBytes>::type;
  const auto *__restrict__ source =
      reinterpret_cast<const unsigned char *>(params.source);
  auto *__restrict__ destination =
      reinterpret_cast<unsigned char *>(params.destination);
  size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
  size_t prefix = source_prefix(params.source, params.bytes);

  if (thread < prefix) {
    scatter_byte(params, destination, thread, source[thread]);
  }

  size_t bulk_bytes = (params.bytes - prefix) & ~(LoadBytes - 1);
  for (size_t index = prefix + thread * LoadBytes; index < prefix + bulk_bytes;
       index += step * LoadBytes) {
    word packed = *reinterpret_cast<const word *>(source + index);
    const auto *unpacked = reinterpret_cast<const unsigned char *>(&packed);
#pragma unroll
    for (size_t byte = 0; byte < LoadBytes; ++byte) {
      scatter_byte(params, destination, index + byte, unpacked[byte]);
    }
  }

  size_t tail = params.bytes - prefix - bulk_bytes;
  if (thread < tail) {
    size_t index = prefix + bulk_bytes + thread;
    scatter_byte(params, destination, index, source[index]);
  }
}

// A row one to three bytes wide sends every useful byte to a different
// destination sector, so no store can be widened. The load can be: these
// kernels are bound by how many mapped-read requests they issue, not by the
// stores. Each thread pulls one aligned kScatterLoadBytes-wide word through the
// host link and unpacks it into that many scattered byte stores, which
// measured 2.5x the throughput of issuing the same bytes as separate loads.
// The scalar prefix reaching a 128-byte source boundary is what makes the wide
// load aligned.
//
// Unlike the CUDA operation there is no lane-group staggering around the
// stores. That mitigates NVIDIA's serialized L1TEX store wavefronts; on
// gfx1100 it measured as noise, winning and losing across repeats of the same
// shape. Leaving it out keeps the operation free of any wave-width dependency,
// so it is correct on wave32 and wave64 alike without an architecture table.
template <size_t Width, size_t LoadBytes>
__global__ void smemcpy_narrow_2d_kernel(lupine_hip_smemcpy_params params) {
  using word = typename word_type<LoadBytes>::type;
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

  size_t bulk_bytes = (params.bytes - prefix) & ~(LoadBytes - 1);
  for (size_t index = prefix + thread * LoadBytes; index < prefix + bulk_bytes;
       index += step * LoadBytes) {
    word packed = *reinterpret_cast<const word *>(source + index);
    const auto *unpacked = reinterpret_cast<const unsigned char *>(&packed);
#pragma unroll
    for (size_t byte = 0; byte < LoadBytes; ++byte) {
      scatter_narrow_store<Width>(params, destination, index + byte,
                                  unpacked[byte]);
    }
  }

  size_t tail = params.bytes - prefix - bulk_bytes;
  if (thread < tail) {
    size_t index = prefix + bulk_bytes + thread;
    scatter_narrow_store<Width>(params, destination, index, source[index]);
  }
}

template <size_t Width, size_t LoadBytes>
__global__ void smemcpy_narrow_3d_kernel(lupine_hip_smemcpy_params params) {
  using word = typename word_type<LoadBytes>::type;
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

  size_t bulk_bytes = (params.bytes - prefix) & ~(LoadBytes - 1);
  for (size_t index = prefix + thread * LoadBytes; index < prefix + bulk_bytes;
       index += step * LoadBytes) {
    word packed = *reinterpret_cast<const word *>(source + index);
    const auto *unpacked = reinterpret_cast<const unsigned char *>(&packed);
#pragma unroll
    for (size_t byte = 0; byte < LoadBytes; ++byte) {
      scatter_narrow_3d_store<Width>(params, destination, index + byte,
                                     unpacked[byte]);
    }
  }

  size_t tail = params.bytes - prefix - bulk_bytes;
  if (thread < tail) {
    size_t index = prefix + bulk_bytes + thread;
    scatter_narrow_3d_store<Width>(params, destination, index, source[index]);
  }
}

size_t blocks_for(size_t bytes, size_t bytes_per_thread) {
  size_t block_bytes = kThreadsPerBlock * bytes_per_thread;
  size_t blocks = bytes / block_bytes + (bytes % block_bytes != 0);
  return std::min(std::max<size_t>(blocks, 1), kMaximumBlocks);
}

bool packed_destination(const lupine_hip_smemcpy_params &params,
                        size_t slice_bytes) {
  return params.destination_row_stride == params.width &&
         params.destination_slice_stride == slice_bytes;
}

unsigned long long destination_at(const lupine_hip_smemcpy_params &params,
                                  size_t logical) {
  size_t row_index = logical / params.width;
  size_t x = logical - row_index * params.width;
  size_t slice = row_index / params.rows;
  size_t row = row_index - slice * params.rows;
  return params.destination + slice * params.destination_slice_stride +
         row * params.destination_row_stride + x;
}

template <size_t WordBytes>
bool can_vectorize_pitched(const lupine_hip_smemcpy_params &params,
                           unsigned long long first_destination) {
  constexpr size_t mask = WordBytes - 1;
  return (params.width & mask) == 0 && (params.destination & mask) == 0 &&
         (params.destination_row_stride & mask) == 0 &&
         (params.destination_slice_stride & mask) == 0 &&
         ((params.source ^ first_destination) & mask) == 0;
}

template <size_t WordBytes>
void set_linear_launch(lupine_hip_smemcpy_launch *launch) {
  launch->kernel =
      reinterpret_cast<const void *>(smemcpy_linear_kernel<WordBytes>);
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, WordBytes));
}

template <size_t WordBytes>
void set_pitched_vector_launch(lupine_hip_smemcpy_launch *launch) {
  launch->kernel =
      reinterpret_cast<const void *>(smemcpy_pitched_vector_kernel<WordBytes>);
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, WordBytes));
}

void select_linear_launch(lupine_hip_smemcpy_launch *launch,
                          unsigned long long first_destination) {
  uintptr_t different_alignment =
      static_cast<uintptr_t>(launch->params.source ^ first_destination);
  if ((different_alignment & 15) == 0) {
    set_linear_launch<16>(launch);
  } else if ((different_alignment & 7) == 0) {
    set_linear_launch<8>(launch);
  } else if ((different_alignment & 3) == 0) {
    set_linear_launch<4>(launch);
  } else if ((different_alignment & 1) == 0) {
    set_linear_launch<2>(launch);
  } else {
    set_linear_launch<1>(launch);
  }
}

bool select_pitched_vector_launch(lupine_hip_smemcpy_launch *launch,
                                  unsigned long long first_destination) {
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

void set_narrow_2d_launch(lupine_hip_smemcpy_launch *launch, size_t width) {
  constexpr size_t load = kScatterLoadBytes;
  switch (width) {
  case 1:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_2d_kernel<1, load>);
    break;
  case 2:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_2d_kernel<2, load>);
    break;
  default:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_2d_kernel<3, load>);
    break;
  }
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, load));
}

void set_narrow_3d_launch(lupine_hip_smemcpy_launch *launch, size_t width) {
  constexpr size_t load = kScatterLoadBytes;
  switch (width) {
  case 1:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_3d_kernel<1, load>);
    break;
  case 2:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_3d_kernel<2, load>);
    break;
  default:
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_narrow_3d_kernel<3, load>);
    break;
  }
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, load));
}

} // namespace

extern "C" const void *lupine_hip_smemcpy_kernel(void) {
  return reinterpret_cast<const void *>(smemcpy_3d_kernel<kScatterLoadBytes>);
}

extern "C" hipError_t
lupine_hip_smemcpy_prepare_launch(const lupine_hip_smemcpy_params *params,
                                  lupine_hip_smemcpy_launch *launch) {
  if (params == nullptr || launch == nullptr || params->width == 0 ||
      params->rows == 0 ||
      (params->bytes != 0 &&
       (params->source == 0 || params->destination == 0)) ||
      params->width > std::numeric_limits<size_t>::max() / params->rows ||
      params->bytes >
          std::numeric_limits<size_t>::max() - params->logical_offset) {
    return hipErrorInvalidValue;
  }

  launch->params = *params;
  launch->kernel =
      reinterpret_cast<const void *>(smemcpy_3d_kernel<kScatterLoadBytes>);
  launch->blocks = 0;
  launch->threads = kThreadsPerBlock;
  if (params->bytes == 0) {
    return hipSuccess;
  }

  size_t slice_bytes = params->width * params->rows;
  unsigned long long first_destination =
      destination_at(*params, params->logical_offset);
  if (packed_destination(*params, slice_bytes)) {
    select_linear_launch(launch, first_destination);
    return hipSuccess;
  }

  // Always prefer a vector store where the shape admits one. A width that
  // vectorizes cleanly gains more from wide stores than a narrow kernel gains
  // from batching its mapped reads.
  if (select_pitched_vector_launch(launch, first_destination)) {
    return hipSuccess;
  }

  size_t first_row_index = params->logical_offset / params->width;
  size_t first_slice = first_row_index / params->rows;
  size_t last_logical = params->logical_offset + params->bytes - 1;
  size_t last_row_index = last_logical / params->width;
  size_t last_slice = last_row_index / params->rows;

  if (first_slice == last_slice) {
    size_t first_row = first_row_index - first_slice * params->rows;
    size_t first_x = params->logical_offset - first_row_index * params->width;
    launch->params.destination =
        params->destination + first_slice * params->destination_slice_stride +
        first_row * params->destination_row_stride;
    launch->params.logical_offset = first_x;
    if (params->width <= 3) {
      set_narrow_2d_launch(launch, params->width);
    } else {
      launch->kernel =
          reinterpret_cast<const void *>(smemcpy_2d_kernel<kScatterLoadBytes>);
      launch->blocks = static_cast<unsigned int>(
          blocks_for(params->bytes, kScatterLoadBytes));
    }
    return hipSuccess;
  }

  if (first_slice != 0) {
    launch->params.destination +=
        first_slice * params->destination_slice_stride;
    launch->params.logical_offset -= first_slice * slice_bytes;
  }
  if (params->width <= 3) {
    set_narrow_3d_launch(launch, params->width);
  } else {
    launch->kernel =
        reinterpret_cast<const void *>(smemcpy_3d_kernel<kScatterLoadBytes>);
    launch->blocks =
        static_cast<unsigned int>(blocks_for(params->bytes, kScatterLoadBytes));
  }
  return hipSuccess;
}

extern "C" hipError_t
lupine_hip_smemcpy_async(const lupine_hip_smemcpy_params *params,
                         hipStream_t stream) {
  lupine_hip_smemcpy_launch launch = {};
  hipError_t result = lupine_hip_smemcpy_prepare_launch(params, &launch);
  if (result != hipSuccess || launch.blocks == 0) {
    return result;
  }

  void *arguments[] = {&launch.params};
  return hipLaunchKernel(launch.kernel, dim3(launch.blocks),
                         dim3(launch.threads), arguments, 0, stream);
}
