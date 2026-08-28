#include "ops/smemcpy_dispatch.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr size_t kMaximumBlocks = 65535;
constexpr size_t kNarrowItemsPerThread = 4;

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
void set_pitched_vector_launch(lupine_smemcpy_launch_descriptor *launch) {
  launch->kernel = LUPINE_SMEMCPY_KERNEL_PITCHED;
  launch->word_bytes = WordBytes;
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, WordBytes));
}

bool select_pitched_vector_launch(lupine_smemcpy_launch_descriptor *launch,
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

void set_narrow_launch(lupine_smemcpy_launch_descriptor *launch, bool is_3d,
                       size_t width, size_t items) {
  launch->kernel =
      is_3d ? LUPINE_SMEMCPY_KERNEL_NARROW_3D : LUPINE_SMEMCPY_KERNEL_NARROW_2D;
  launch->width = static_cast<unsigned int>(width);
  launch->items = static_cast<unsigned int>(items);
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, items));
}

void set_atomic_launch(lupine_smemcpy_launch_descriptor *launch, size_t width,
                       unsigned int byte_offset, unsigned int group_threads) {
  launch->kernel = width == 1 ? LUPINE_SMEMCPY_KERNEL_ATOMIC_NARROW_2D
                              : LUPINE_SMEMCPY_KERNEL_ATOMIC_ROWS;
  launch->width = static_cast<unsigned int>(width);
  launch->byte_offset = byte_offset;
  launch->group_threads = group_threads;
  launch->blocks =
      static_cast<unsigned int>(blocks_for(launch->params.bytes, width));
}

enum atomic_narrow_target {
  atomic_none,
  atomic_pascal,
  atomic_volta,
  atomic_turing,
  atomic_ampere,
  atomic_l4,
};

atomic_narrow_target atomic_target_for(const lupine_smemcpy_device *device) {
  if (device == nullptr) {
    return atomic_none;
  }
  if (device->major == 6 && device->minor == 0) {
    return atomic_pascal;
  }
  if (device->major == 7 && device->minor == 0) {
    return atomic_volta;
  }
  if (device->major == 7 && device->minor == 5) {
    return atomic_turing;
  }
  if (device->major == 8 && device->minor == 0) {
    return atomic_ampere;
  }
  if (device->major == 8 && device->minor == 9 && device->is_l4) {
    return atomic_l4;
  }
  return atomic_none;
}

unsigned int atomic_group_threads(atomic_narrow_target target, size_t width,
                                  size_t stride) {
  if (stride < 64 || stride > 512 || (stride & (stride - 1)) != 0) {
    return 0;
  }
  switch (target) {
  case atomic_pascal:
    if (width == 1) {
      return stride >= 128 ? 4 : 0;
    }
    return stride == 64 ? 8 : 4;
  case atomic_volta:
    if (width == 1) {
      return stride >= 128 ? 4 : 0;
    }
    if (stride == 64 || (width == 3 && stride == 128)) {
      return 16;
    }
    return stride == 128 ? 8 : 4;
  case atomic_turing:
    if (width == 1) {
      return stride == 64 ? 8 : 4;
    }
    if (width == 2) {
      return stride == 128 || stride == 256 ? 4 : 0;
    }
    return stride == 64 ? 16 : 4;
  case atomic_ampere:
    return width == 1 && stride >= 256 ? 8 : 0;
  case atomic_l4:
    if (width == 1) {
      return stride == 64 ? 16 : 8;
    }
    return 16;
  case atomic_none:
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

extern "C" CUresult
lupine_smemcpy_prepare_descriptor(const lupine_smemcpy_params *params,
                                  const lupine_smemcpy_device *device,
                                  lupine_smemcpy_launch_descriptor *launch) {
  if (params == nullptr || launch == nullptr || params->width == 0 ||
      params->rows == 0 ||
      (params->bytes != 0 &&
       (params->source == 0 || params->destination == 0)) ||
      params->width > std::numeric_limits<size_t>::max() / params->rows ||
      params->bytes >
          std::numeric_limits<size_t>::max() - params->logical_offset) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  *launch = {};
  launch->params = *params;
  launch->kernel = LUPINE_SMEMCPY_KERNEL_3D;
  launch->threads = kThreadsPerBlock;
  if (params->bytes == 0) {
    return CUDA_SUCCESS;
  }

  size_t slice_bytes = params->width * params->rows;
  CUdeviceptr first_destination =
      destination_at(*params, params->logical_offset);
  if (contiguous_destination(*params, slice_bytes)) {
    launch->params.destination = first_destination;
    launch->params.logical_offset = 0;
    launch->use_cuda_memcpy = true;
    return CUDA_SUCCESS;
  }

  size_t first_row_index = params->logical_offset / params->width;
  size_t first_slice = first_row_index / params->rows;
  size_t last_logical = params->logical_offset + params->bytes - 1;
  size_t last_row_index = last_logical / params->width;
  size_t last_slice = last_row_index / params->rows;
  bool latency_hiding_narrow =
      params->width <= 3 && device != nullptr && device->major < 8;
  unsigned int atomic_threads = 0;
  if (params->width <= 3) {
    atomic_threads =
        atomic_group_threads(atomic_target_for(device), params->width,
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
    return CUDA_SUCCESS;
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
        set_atomic_launch(launch, params->width,
                          static_cast<unsigned int>(first_destination & 3),
                          atomic_threads);
      } else {
        set_narrow_launch(launch, false, params->width,
                          latency_hiding_narrow ? kNarrowItemsPerThread : 1);
      }
    } else {
      launch->kernel = LUPINE_SMEMCPY_KERNEL_2D;
      launch->blocks = static_cast<unsigned int>(blocks_for(params->bytes, 1));
    }
    return CUDA_SUCCESS;
  }

  if (first_slice != 0) {
    launch->params.destination +=
        first_slice * params->destination_slice_stride;
    launch->params.logical_offset -= first_slice * slice_bytes;
  }
  if (params->width <= 3) {
    if (atomic_threads != 0) {
      set_atomic_launch(launch, params->width,
                        static_cast<unsigned int>(first_destination & 3),
                        atomic_threads);
    } else {
      set_narrow_launch(launch, true, params->width,
                        latency_hiding_narrow ? kNarrowItemsPerThread : 1);
    }
  } else {
    launch->kernel = LUPINE_SMEMCPY_KERNEL_3D;
    launch->blocks = static_cast<unsigned int>(blocks_for(params->bytes, 1));
  }
  return CUDA_SUCCESS;
}

extern "C" const char *
lupine_smemcpy_kernel_name(const lupine_smemcpy_launch_descriptor *launch) {
  static thread_local char name[64];
  if (launch == nullptr || launch->use_cuda_memcpy || launch->blocks == 0) {
    return nullptr;
  }
  switch (launch->kernel) {
  case LUPINE_SMEMCPY_KERNEL_3D:
    return "lupine_smemcpy_3d";
  case LUPINE_SMEMCPY_KERNEL_2D:
    return "lupine_smemcpy_2d";
  case LUPINE_SMEMCPY_KERNEL_PITCHED:
    std::snprintf(name, sizeof(name), "lupine_smemcpy_pitched_%u",
                  launch->word_bytes);
    break;
  case LUPINE_SMEMCPY_KERNEL_NARROW_2D:
    std::snprintf(name, sizeof(name), "lupine_smemcpy_narrow_2d_%u_%u",
                  launch->width, launch->items);
    break;
  case LUPINE_SMEMCPY_KERNEL_NARROW_3D:
    std::snprintf(name, sizeof(name), "lupine_smemcpy_narrow_3d_%u_%u",
                  launch->width, launch->items);
    break;
  case LUPINE_SMEMCPY_KERNEL_ATOMIC_NARROW_2D:
    std::snprintf(name, sizeof(name), "lupine_smemcpy_atomic_2d_%u_%u",
                  launch->byte_offset, launch->group_threads);
    break;
  case LUPINE_SMEMCPY_KERNEL_ATOMIC_ROWS:
    std::snprintf(name, sizeof(name), "lupine_smemcpy_atomic_rows_%u_%u_%u",
                  launch->width, launch->byte_offset, launch->group_threads);
    break;
  }
  return name;
}
