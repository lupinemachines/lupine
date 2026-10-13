#ifndef LUPINE_MACOS_LIBRARY_PROFILE_H
#define LUPINE_MACOS_LIBRARY_PROFILE_H

#if defined(__APPLE__)
#include <crt_externs.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach/vm_prot.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace lupine_macos_profile {

struct range {
  uintptr_t start;
  uint64_t size;
};
struct segment {
  uintptr_t start;
  uint64_t offset, size;
  std::vector<range> zero_fill;
};
struct object {
  const mach_header *header;
  std::string uuid;
  std::vector<segment> segments;
};
struct registry {
  std::mutex mutex;
  std::once_flag initialized;
  std::vector<object> objects;
};

inline registry &images() {
  // dyld removal callbacks also run during process teardown.
  static auto *state = new registry;
  return *state;
}

inline void add_image(const mach_header *header, intptr_t slide) {
  if (header->magic != MH_MAGIC_64) {
    return;
  }
  object entry{header, {}, {}};
  const auto *h = reinterpret_cast<const mach_header_64 *>(header);
  const auto *commands = reinterpret_cast<const unsigned char *>(h + 1);
  const unsigned char *end = commands + h->sizeofcmds;
  for (const unsigned char *p = commands; p + sizeof(load_command) <= end;) {
    const auto *command = reinterpret_cast<const load_command *>(p);
    if (command->cmdsize < sizeof(*command) || command->cmdsize > end - p) {
      return;
    }
    if (command->cmd == LC_UUID && command->cmdsize >= sizeof(uuid_command)) {
      const auto *u = reinterpret_cast<const uuid_command *>(p);
      entry.uuid.assign(reinterpret_cast<const char *>(u->uuid),
                        sizeof(u->uuid));
    }
    if (command->cmd == LC_SEGMENT_64 &&
        command->cmdsize >= sizeof(segment_command_64)) {
      const auto *seg = reinterpret_cast<const segment_command_64 *>(p);
      if (seg->nsects >
          (command->cmdsize - sizeof(*seg)) / sizeof(section_64)) {
        return;
      }
      if ((seg->initprot & VM_PROT_READ) && seg->filesize) {
        segment mapped{seg->vmaddr + slide,
                       seg->fileoff,
                       std::min(seg->filesize, seg->vmsize),
                       {}};
        const auto *sections = reinterpret_cast<const section_64 *>(seg + 1);
        for (uint32_t j = 0; j < seg->nsects; ++j) {
          unsigned type = sections[j].flags & SECTION_TYPE;
          if (type == S_ZEROFILL || type == S_GB_ZEROFILL ||
              type == S_THREAD_LOCAL_ZEROFILL) {
            mapped.zero_fill.push_back(
                {sections[j].addr + slide, sections[j].size});
          }
        }
        entry.segments.push_back(std::move(mapped));
      }
    }
    p += command->cmdsize;
  }
  if (!entry.uuid.empty()) {
    auto &state = images();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.objects.push_back(std::move(entry));
  }
}

inline void remove_image(const mach_header *header, intptr_t) {
  auto &state = images();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto &objects = state.objects;
  objects.erase(
      std::remove_if(objects.begin(), objects.end(),
                     [&](const object &o) { return o.header == header; }),
      objects.end());
}

// Cache metadata through dyld's loader callbacks: enumerating image indices
// races concurrent dlopen/dlclose. No dyld calls occur while holding our mutex.
// UUID plus file offset survives ASLR, including machso's embedded ELF
// sections.
inline const unsigned char *image(const unsigned char *data, uint64_t size,
                                  std::string &id, uint64_t &offset) {
  auto &state = images();
  std::call_once(state.initialized, [] {
    _dyld_register_func_for_add_image(add_image);
    _dyld_register_func_for_remove_image(remove_image);
  });
  std::lock_guard<std::mutex> lock(state.mutex);
  for (const auto &o : state.objects) {
    if (data == nullptr && o.uuid != id) {
      continue;
    }
    for (const auto &seg : o.segments) {
      uint64_t at = data != nullptr
                        ? reinterpret_cast<uintptr_t>(data) - seg.start
                        : offset - seg.offset;
      if (at >= seg.size || size > seg.size - at) {
        continue;
      }
      // Segment file padding can overlap a zero-fill section. Exclude it even
      // if dyld reports that page inside the file-backed segment size.
      uintptr_t address = seg.start + at;
      bool zero_fill = std::any_of(
          seg.zero_fill.begin(), seg.zero_fill.end(), [&](const range &r) {
            return address >= r.start ? address - r.start < r.size
                                      : r.start - address < size;
          });
      if (zero_fill) {
        continue;
      }
      if (data != nullptr) {
        id = o.uuid;
        offset = seg.offset + at;
      }
      return reinterpret_cast<const unsigned char *>(address);
    }
  }
  return nullptr;
}

inline std::string command_line() {
  std::string result;
  char **argv = *_NSGetArgv();
  for (int i = 0; i < *_NSGetArgc(); ++i) {
    result.append(argv[i]);
    result.push_back('\0');
  }
  return result;
}

} // namespace lupine_macos_profile
#endif
#endif
