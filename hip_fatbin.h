#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

// Clang's uncompressed offload bundle: magic, bundle count, then a table of
// (offset, size, target-ID length, target-ID bytes). Copy the whole bundle so
// the server's runtime selects the code object for its own GPU.
constexpr size_t lupine_hip_max_image_size = 256 * 1024 * 1024;

inline size_t
lupine_hip_bundle_size(const void *image,
                       size_t available = std::numeric_limits<size_t>::max()) {
  static constexpr std::string_view magic = "__CLANG_OFFLOAD_BUNDLE__";
  constexpr size_t header_size = magic.size() + sizeof(uint64_t);
  if (image == nullptr || available < header_size ||
      memcmp(image, magic.data(), magic.size()) != 0) {
    return 0;
  }
  const auto *bytes = static_cast<const unsigned char *>(image);
  uint64_t count = 0;
  memcpy(&count, bytes + magic.size(), sizeof(count));
  if (count == 0 || count > 64) {
    return 0;
  }
  size_t cursor = header_size;
  size_t extent = header_size;
  uint64_t first_payload = lupine_hip_max_image_size;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t fields[3];
    if (cursor > available || available - cursor < sizeof(fields) ||
        cursor + sizeof(fields) > first_payload) {
      return 0;
    }
    memcpy(fields, bytes + cursor, sizeof(fields));
    const auto offset = fields[0], size = fields[1], id_size = fields[2];
    if (offset > lupine_hip_max_image_size ||
        size > lupine_hip_max_image_size - offset || id_size > 4096) {
      return 0;
    }
    cursor += sizeof(fields);
    if (id_size > available - cursor) {
      return 0;
    }
    cursor += static_cast<size_t>(id_size);
    first_payload = std::min(first_payload, offset);
    extent = std::max(extent, static_cast<size_t>(offset + size));
  }
  if (cursor > first_payload || extent > available) {
    return 0;
  }
  return extent;
}
