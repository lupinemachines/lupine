#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// Server-originated requests are handled by the client's dedicated dispatch
// thread. CUDA host functions use this channel for side effects that have to
// occur when stream work executes rather than when the API call is captured.
enum class lupine_side_effect_op : int {
  host_function = 1,
  stream_callback = 2,
  read_host_memory = 3,
};

enum class lupine_htod_source_location : uint8_t {
  client = 0,
  server = 1,
};

enum class lupine_copy_direction : uint8_t {
  host_to_host = 0,
  host_to_device = 1,
  device_to_host = 2,
  device_to_device = 3,
};

// Dimensional memcpy RPCs carry this before the CUDA descriptor so the server
// can choose the direction handler and, for HtoD, whether the source is already
// authoritative on the server.
struct lupine_memcpy_wire_flags {
  lupine_copy_direction direction = lupine_copy_direction::device_to_device;
  lupine_htod_source_location htod_source =
      lupine_htod_source_location::client;
};

static_assert(sizeof(lupine_memcpy_wire_flags) == 2);

// A possibly pitched client host-memory region. The response contains the
// selected bytes packed as slices of adjacent rows; padding never travels.
struct lupine_host_memory_view {
  const void *data = nullptr;
  size_t width = 0;
  size_t rows = 0;
  size_t row_stride = 0;
  size_t slices = 0;
  size_t slice_stride = 0;
};

static inline bool
lupine_host_memory_view_sizes(const lupine_host_memory_view &view,
                              size_t *packed_bytes, size_t *span_bytes) {
  if (packed_bytes == nullptr || span_bytes == nullptr) {
    return false;
  }
  if (view.width == 0 || view.rows == 0 || view.slices == 0) {
    *packed_bytes = 0;
    *span_bytes = 0;
    return true;
  }
  constexpr size_t maximum = std::numeric_limits<size_t>::max();
  if (view.rows > maximum / view.slices ||
      view.width > maximum / (view.rows * view.slices) ||
      (view.rows > 1 && view.row_stride == 0) ||
      (view.slices > 1 && view.slice_stride == 0)) {
    return false;
  }
  size_t packed = view.width * view.rows * view.slices;
  size_t span = view.width;
  if (view.rows - 1 > (maximum - span) / view.row_stride) {
    return false;
  }
  span += (view.rows - 1) * view.row_stride;
  if (view.slices - 1 > (maximum - span) / view.slice_stride) {
    return false;
  }
  span += (view.slices - 1) * view.slice_stride;
  *packed_bytes = packed;
  *span_bytes = span;
  return true;
}
