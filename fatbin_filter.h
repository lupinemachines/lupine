#ifndef LUPINE_FATBIN_FILTER_H
#define LUPINE_FATBIN_FILTER_H

#include "lupine_fatbin.h"

#include <algorithm>
#include <cstring>
#include <vector>

// Keep every member that might serve any device on the destination server.
// The driver still selects among retained entries, including architecture and
// family qualifiers. Unknown formats, incomplete device inventories and a
// selection with no usable member are forwarded unchanged.
inline std::vector<unsigned char>
lupine_filter_fatbin(const unsigned char *image, size_t size,
                     const std::vector<unsigned> &devices) {
  if (image == nullptr || size < sizeof(lupine_fatbin_header) ||
      devices.empty() ||
      std::find(devices.begin(), devices.end(), 0) != devices.end()) {
    return {};
  }
  lupine_fatbin_header header;
  std::memcpy(&header, image, sizeof(header));
  if (header.magic != LUPINE_FATBIN_MAGIC ||
      header.version != LUPINE_FATBIN_VERSION ||
      header.header_size != sizeof(header) ||
      header.files_size != size - sizeof(header)) {
    return {};
  }

  struct span {
    size_t offset;
    size_t size;
  };
  std::vector<span> kept;
  size_t selected_size = sizeof(header);
  for (size_t at = sizeof(header); at < size;) {
    if (size - at < sizeof(lupine_fatbin_entry)) {
      return {};
    }
    lupine_fatbin_entry entry;
    std::memcpy(&entry, image + at, sizeof(entry));
    if (entry.version != LUPINE_FATBIN_ENTRY_VERSION ||
        entry.header_size < sizeof(entry) || entry.header_size > size - at ||
        entry.payload_size > size - at - entry.header_size || entry.arch == 0 ||
        (entry.kind != LUPINE_FATBIN_ENTRY_CUBIN &&
         entry.kind != LUPINE_FATBIN_ENTRY_PTX)) {
      return {};
    }
    const size_t length = entry.header_size + entry.payload_size;
    const bool keep =
        std::any_of(devices.begin(), devices.end(), [&entry](unsigned device) {
          if (entry.kind == LUPINE_FATBIN_ENTRY_PTX) {
            return entry.arch <= device;
          }
          return entry.arch / 10 == device / 10 && entry.arch <= device;
        });
    if (keep) {
      kept.push_back({at, length});
      selected_size += length;
    }
    at += length;
  }
  if (kept.empty() || selected_size == size) {
    return {};
  }

  std::vector<unsigned char> selected(selected_size);
  header.files_size = selected_size - sizeof(header);
  std::memcpy(selected.data(), &header, sizeof(header));
  size_t at = sizeof(header);
  for (const auto &entry : kept) {
    std::memcpy(selected.data() + at, image + entry.offset, entry.size);
    at += entry.size;
  }
  return selected;
}

#endif
