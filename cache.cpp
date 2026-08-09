#include "cache.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr size_t kLaneContextCacheSlots = 32;

struct current_context_device_cache_entry {
  uint64_t epoch = 0;
  CUcontext context = nullptr;
  CUdevice device = -1;
};

struct lane_context_cache_entry {
  uint64_t epoch = 0;
  int route_id = -2;
  CUcontext context = nullptr;
};

std::atomic<uint64_t> &current_context_device_cache_epoch() {
  static std::atomic<uint64_t> epoch{1};
  return epoch;
}

current_context_device_cache_entry &current_context_device_cache() {
  static thread_local current_context_device_cache_entry cache;
  return cache;
}

std::atomic<uint64_t> &lane_context_cache_epoch() {
  static std::atomic<uint64_t> epoch{1};
  return epoch;
}

std::array<lane_context_cache_entry, kLaneContextCacheSlots> &
lane_context_cache() {
  static thread_local std::array<lane_context_cache_entry,
                                 kLaneContextCacheSlots>
      cache;
  return cache;
}

lane_context_cache_entry *lane_context_cache_entry_for(int route_id) {
  if (route_id < -1) {
    return nullptr;
  }
  static_assert((kLaneContextCacheSlots & (kLaneContextCacheSlots - 1)) == 0,
                "lane context cache size must be a power of two");
  size_t slot = static_cast<size_t>(route_id) & (kLaneContextCacheSlots - 1);
  return &lane_context_cache()[slot];
}

constexpr char kKernelTableMagic[8] = {'L', 'U', 'P', 'K', 'T', 'A', 'B', '1'};

struct sha256_state {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint64_t length = 0;
  unsigned char block[64] = {};
  size_t filled = 0;
};

uint32_t rotate_right(uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

void sha256_compress(sha256_state &state, const unsigned char *block) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t w[64];
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           static_cast<uint32_t>(block[i * 4 + 3]);
  }
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = rotate_right(w[i - 15], 7) ^ rotate_right(w[i - 15], 18) ^
                  (w[i - 15] >> 3);
    uint32_t s1 = rotate_right(w[i - 2], 17) ^ rotate_right(w[i - 2], 19) ^
                  (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t v[8];
  memcpy(v, state.h, sizeof(v));
  for (int i = 0; i < 64; ++i) {
    uint32_t s1 =
        rotate_right(v[4], 6) ^ rotate_right(v[4], 11) ^ rotate_right(v[4], 25);
    uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
    uint32_t temp1 = v[7] + s1 + ch + k[i] + w[i];
    uint32_t s0 =
        rotate_right(v[0], 2) ^ rotate_right(v[0], 13) ^ rotate_right(v[0], 22);
    uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
    uint32_t temp2 = s0 + maj;
    v[7] = v[6];
    v[6] = v[5];
    v[5] = v[4];
    v[4] = v[3] + temp1;
    v[3] = v[2];
    v[2] = v[1];
    v[1] = v[0];
    v[0] = temp1 + temp2;
  }
  for (int i = 0; i < 8; ++i) {
    state.h[i] += v[i];
  }
}

void sha256_update(sha256_state &state, const unsigned char *data,
                   size_t size) {
  state.length += size;
  while (size != 0) {
    size_t take = sizeof(state.block) - state.filled;
    take = take < size ? take : size;
    memcpy(state.block + state.filled, data, take);
    state.filled += take;
    data += take;
    size -= take;
    if (state.filled == sizeof(state.block)) {
      sha256_compress(state, state.block);
      state.filled = 0;
    }
  }
}

std::string sha256_hex(const unsigned char *data, size_t size) {
  sha256_state state;
  sha256_update(state, data, size);
  uint64_t bits = state.length * 8;
  unsigned char padding[72] = {0x80};
  size_t pad = state.filled < 56 ? 56 - state.filled : 120 - state.filled;
  sha256_update(state, padding, pad);
  unsigned char tail[8];
  for (int i = 0; i < 8; ++i) {
    tail[i] = static_cast<unsigned char>(bits >> (56 - i * 8));
  }
  sha256_update(state, tail, sizeof(tail));

  static const char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64);
  for (uint32_t word : state.h) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      hex.push_back(digits[(word >> shift) & 0xF]);
    }
  }
  return hex;
}

std::string kernel_table_cache_dir() {
  const char *configured = getenv("LUPINE_CACHE_DIR");
  std::string root;
  if (configured != nullptr && configured[0] != '\0') {
    root = configured;
  } else {
    const char *xdg = getenv("XDG_CACHE_HOME");
    if (xdg != nullptr && xdg[0] != '\0') {
      root = std::string(xdg) + "/lupine";
    } else {
      const char *home = getenv("HOME");
      if (home == nullptr || home[0] == '\0') {
        return std::string();
      }
      root = std::string(home) + "/.cache/lupine";
    }
  }
  return root + "/kernels";
}

bool make_directories(const std::string &path) {
  for (size_t i = 1; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/') {
      continue;
    }
    std::string prefix = path.substr(0, i);
    if (mkdir(prefix.c_str(), 0755) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

std::string kernel_table_cache_path(const unsigned char *image, size_t size,
                                    uint32_t driver_version) {
  std::string dir = kernel_table_cache_dir();
  if (dir.empty() || image == nullptr || size == 0) {
    return std::string();
  }
  return dir + "/" + sha256_hex(image, size) + "-" +
         std::to_string(driver_version) + ".ktab";
}

template <typename T> bool read_pod(std::istream &stream, T *value) {
  return stream.read(reinterpret_cast<char *>(value), sizeof(T)).good();
}

template <typename T> void write_pod(std::ostream &stream, const T &value) {
  stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

} // namespace

bool lupine_kernel_table_cache_load(
    const unsigned char *image, size_t size, uint32_t driver_version,
    std::vector<lupine_kernel_table_entry> *out) {
  if (out == nullptr) {
    return false;
  }
  out->clear();
  std::string path = kernel_table_cache_path(image, size, driver_version);
  if (path.empty()) {
    return false;
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  char magic[sizeof(kKernelTableMagic)] = {};
  uint32_t count = 0;
  if (!file.read(magic, sizeof(magic)).good() ||
      memcmp(magic, kKernelTableMagic, sizeof(magic)) != 0 ||
      !read_pod(file, &count) || count > 1u << 20) {
    return false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t name_length = 0;
    uint32_t param_count = 0;
    lupine_kernel_table_entry entry;
    if (!read_pod(file, &name_length) || name_length == 0 ||
        name_length > 64 * 1024) {
      out->clear();
      return false;
    }
    entry.name.resize(name_length);
    if (!file.read(entry.name.data(), name_length).good() ||
        !read_pod(file, &param_count) || param_count > 4096) {
      out->clear();
      return false;
    }
    entry.params.resize(static_cast<size_t>(param_count) * 2);
    if (param_count != 0 &&
        !file.read(reinterpret_cast<char *>(entry.params.data()),
                   static_cast<std::streamsize>(entry.params.size() *
                                                sizeof(uint64_t)))
             .good()) {
      out->clear();
      return false;
    }
    out->push_back(std::move(entry));
  }
  return !out->empty();
}

void lupine_kernel_table_cache_store(
    const unsigned char *image, size_t size, uint32_t driver_version,
    const std::vector<lupine_kernel_table_entry> &table) {
  std::string path = kernel_table_cache_path(image, size, driver_version);
  if (path.empty() || table.empty() ||
      !make_directories(kernel_table_cache_dir())) {
    return;
  }
  std::string temporary =
      path + ".tmp." + std::to_string(static_cast<long>(getpid()));
  {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    if (!file) {
      return;
    }
    file.write(kKernelTableMagic, sizeof(kKernelTableMagic));
    write_pod(file, static_cast<uint32_t>(table.size()));
    for (const auto &entry : table) {
      write_pod(file, static_cast<uint32_t>(entry.name.size()));
      file.write(entry.name.data(),
                 static_cast<std::streamsize>(entry.name.size()));
      write_pod(file, static_cast<uint32_t>(entry.params.size() / 2));
      file.write(
          reinterpret_cast<const char *>(entry.params.data()),
          static_cast<std::streamsize>(entry.params.size() * sizeof(uint64_t)));
    }
    file.flush();
    if (!file) {
      file.close();
      remove(temporary.c_str());
      return;
    }
  }
  if (rename(temporary.c_str(), path.c_str()) != 0) {
    remove(temporary.c_str());
  }
}

bool lupine_current_context_device_cache_lookup(CUcontext context,
                                                CUdevice *device) {
  uint64_t current_epoch =
      current_context_device_cache_epoch().load(std::memory_order_acquire);
  auto &entry = current_context_device_cache();
  if (context == nullptr || device == nullptr || entry.epoch != current_epoch ||
      entry.context != context) {
    return false;
  }
  *device = entry.device;
  return true;
}

void lupine_current_context_device_cache_insert(CUcontext context,
                                                CUdevice device) {
  if (context == nullptr) {
    return;
  }
  uint64_t epoch =
      current_context_device_cache_epoch().load(std::memory_order_acquire);
  auto &entry = current_context_device_cache();
  entry.context = context;
  entry.device = device;
  entry.epoch = epoch;
}

void lupine_current_context_device_cache_invalidate() {
  current_context_device_cache_epoch().fetch_add(1, std::memory_order_acq_rel);
}

uint64_t lupine_lane_context_cache_epoch() {
  return lane_context_cache_epoch().load(std::memory_order_acquire);
}

bool lupine_lane_context_cache_matches(int route_id, CUcontext context) {
  auto *entry = lane_context_cache_entry_for(route_id);
  return entry != nullptr && entry->route_id == route_id &&
         entry->context == context &&
         entry->epoch ==
             lane_context_cache_epoch().load(std::memory_order_acquire);
}

void lupine_lane_context_cache_update(int route_id, CUcontext context,
                                      uint64_t epoch, bool succeeded) {
  auto *entry = lane_context_cache_entry_for(route_id);
  if (entry == nullptr) {
    return;
  }
  if (!succeeded ||
      lane_context_cache_epoch().load(std::memory_order_acquire) != epoch) {
    if (entry->route_id == route_id) {
      entry->epoch = 0;
    }
    return;
  }
  entry->route_id = route_id;
  entry->context = context;
  entry->epoch = epoch;
}

void lupine_lane_context_cache_store(int route_id, CUcontext context) {
  lupine_lane_context_cache_update(route_id, context,
                                   lupine_lane_context_cache_epoch(), true);
}

extern "C" void lupine_invalidate_current_context_cache() {
  lupine_current_context_device_cache_invalidate();
  lane_context_cache_epoch().fetch_add(1, std::memory_order_acq_rel);
}
