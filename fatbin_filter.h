#ifndef LUPINE_FATBIN_FILTER_H
#define LUPINE_FATBIN_FILTER_H

#include "lupine_fatbin.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

struct lupine_fatbin_span {
  uint64_t offset;
  uint64_t size;
};

struct lupine_filtered_fatbin {
  static constexpr size_t inline_capacity = 16;

  lupine_fatbin_header header = {};
  uint64_t size = 0;
  bool valid = false;
  size_t span_count = 0;
  lupine_fatbin_span inline_spans[inline_capacity] = {};
  std::vector<lupine_fatbin_span> overflow_spans;

  const lupine_fatbin_span *spans() const {
    return overflow_spans.empty() ? inline_spans : overflow_spans.data();
  }

  void append(uint64_t offset, uint64_t len) {
    lupine_fatbin_span *last =
        span_count == 0
            ? nullptr
            : (overflow_spans.empty() ? &inline_spans[span_count - 1]
                                      : &overflow_spans.back());
    if (last != nullptr && last->offset + last->size == offset) {
      last->size += len;
      return;
    }
    if (overflow_spans.empty() && span_count < inline_capacity) {
      inline_spans[span_count++] = {offset, len};
      return;
    }
    if (overflow_spans.empty()) {
      overflow_spans.reserve(inline_capacity * 2);
      overflow_spans.assign(inline_spans, inline_spans + inline_capacity);
    }
    overflow_spans.push_back({offset, len});
    ++span_count;
  }
};

// Keep every member that might serve any device on the destination server.
// The driver still selects among retained entries, including architecture and
// family qualifiers. Unknown formats, incomplete device inventories and a
// selection with no usable member are forwarded unchanged.
inline lupine_filtered_fatbin
lupine_filter_fatbin_spans(const unsigned char *image, size_t size,
                           const std::vector<unsigned> &devices) {
  if (image == nullptr || size < sizeof(lupine_fatbin_header) ||
      devices.empty() ||
      std::find(devices.begin(), devices.end(), 0) != devices.end()) {
    return {};
  }
  lupine_filtered_fatbin result;
  std::memcpy(&result.header, image, sizeof(result.header));
  if (result.header.magic != LUPINE_FATBIN_MAGIC ||
      result.header.version != LUPINE_FATBIN_VERSION ||
      result.header.header_size != sizeof(result.header) ||
      result.header.files_size != size - sizeof(result.header)) {
    return {};
  }

  size_t selected_size = sizeof(result.header);
  for (size_t at = sizeof(result.header); at < size;) {
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
    if (std::any_of(devices.begin(), devices.end(), [&entry](unsigned device) {
          return entry.kind == LUPINE_FATBIN_ENTRY_PTX
                     ? entry.arch <= device
                     : (entry.arch / 10 == device / 10 && entry.arch <= device);
        })) {
      result.append(at, length);
      selected_size += length;
    }
    at += length;
  }
  if (result.span_count == 0 || selected_size == size) {
    return {};
  }
  result.header.files_size = selected_size - sizeof(result.header);
  result.size = selected_size;
  result.valid = true;
  return result;
}

inline std::vector<unsigned char>
lupine_filter_fatbin(const unsigned char *image, size_t size,
                     const std::vector<unsigned> &devices) {
  lupine_filtered_fatbin filtered =
      lupine_filter_fatbin_spans(image, size, devices);
  if (!filtered.valid) {
    return {};
  }
  std::vector<unsigned char> selected(static_cast<size_t>(filtered.size));
  std::memcpy(selected.data(), &filtered.header, sizeof(filtered.header));
  size_t at = sizeof(filtered.header);
  const lupine_fatbin_span *spans = filtered.spans();
  for (size_t i = 0; i < filtered.span_count; ++i) {
    std::memcpy(selected.data() + at, image + spans[i].offset,
                static_cast<size_t>(spans[i].size));
    at += static_cast<size_t>(spans[i].size);
  }
  return selected;
}

#endif
