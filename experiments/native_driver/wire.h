#pragma once
// Trusted, loopback-only experiment. Both ends use the Linux x86-64 native ABI.
// This protocol is not the production Lupine RPC protocol or a checkpoint
// format.
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>
enum Kind : uint32_t {
  OPEN = 1,
  CLOSE,
  IOCTL,
  PAGE,
  CHANGE,
  DONE,
  MAP,
  UNMAP,
  SYNC,
  POLL,
  REGISTER
};
struct Mapping {
  uint32_t fd, prot, flags, reserved;
  uint64_t offset;
};
struct Message {
  uint32_t kind, count;
  uint64_t a, b;
};
struct Range {
  uint64_t begin, end;
};

inline void subtract_range(std::vector<Range> &ranges, Range excluded) {
  std::vector<Range> result;
  for (Range range : ranges) {
    if (excluded.end <= range.begin || excluded.begin >= range.end) {
      result.push_back(range);
      continue;
    }
    if (range.begin < excluded.begin)
      result.push_back({range.begin, excluded.begin});
    if (excluded.end < range.end)
      result.push_back({excluded.end, range.end});
  }
  ranges = std::move(result);
}

// The caller snapshots live GPU-writable memory before comparing. Otherwise
// a timer changing between comparisons can leave the run scan stuck in place.
template <class Emit>
void changed_bytes(const unsigned char *before, const unsigned char *after,
                   size_t size, Emit emit) {
  for (size_t page = 0; page < size; page += 4096) {
    size_t end = std::min(page + 4096, size);
    if (!std::memcmp(before + page, after + page, end - page))
      continue;
    for (size_t i = page; i < end;) {
      if (before[i] == after[i]) {
        ++i;
        continue;
      }
      size_t begin = i;
      while (i < end && before[i] != after[i])
        ++i;
      emit(begin, i - begin);
    }
  }
}
inline void transfer(int fd, void *data, size_t size, bool sending) {
  auto *p = static_cast<char *>(data);
  while (size) {
    ssize_t n = sending ? write(fd, p, size) : read(fd, p, size);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      std::perror("transport");
      std::exit(2);
    }
    p += n;
    size -= n;
  }
}
inline void send_message(int fd, Message message) {
  transfer(fd, &message, sizeof(message), true);
}
inline Message receive_message(int fd) {
  Message m{};
  transfer(fd, &m, sizeof(m), false);
  return m;
}
