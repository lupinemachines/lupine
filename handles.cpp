#include "handles.h"

#include <condition_variable>
#include <mutex>
#include <unordered_map>

namespace {

struct lupine_handle_entry {
  uintptr_t real = 0;
  bool ready = false;
};

struct lupine_handle_table {
  std::mutex mutex;
  std::condition_variable ready;
  uint64_t counter = 0;
  std::unordered_map<uintptr_t, lupine_handle_entry>
      entries[LUPINE_HANDLE_FAMILY_COUNT];
};

lupine_handle_table &table() {
  static lupine_handle_table instance;
  return instance;
}

bool valid_family(lupine_handle_family family) {
  return family < LUPINE_HANDLE_FAMILY_COUNT;
}

} // namespace

extern "C" bool lupine_handle_is_synthetic(uintptr_t value) {
  return (static_cast<uint64_t>(value) & LUPINE_HANDLE_TAG_MASK) ==
         (LUPINE_HANDLE_TAG << LUPINE_HANDLE_TAG_SHIFT);
}

extern "C" uintptr_t lupine_handle_mint(lupine_handle_family family) {
  if (!valid_family(family)) {
    return 0;
  }
  lupine_handle_table &state = table();
  std::lock_guard<std::mutex> lock(state.mutex);
  uintptr_t synthetic =
      static_cast<uintptr_t>((LUPINE_HANDLE_TAG << LUPINE_HANDLE_TAG_SHIFT) |
                             ((static_cast<uint64_t>(family) & 0xFFull) << 40) |
                             ((++state.counter) & 0xFFFFFFFFFFull));
  state.entries[family][synthetic] = lupine_handle_entry{};
  return synthetic;
}

extern "C" void lupine_handle_fulfill(lupine_handle_family family,
                                      uintptr_t synthetic, uintptr_t real) {
  if (!valid_family(family)) {
    return;
  }
  lupine_handle_table &state = table();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto entry = state.entries[family].find(synthetic);
    if (entry == state.entries[family].end()) {
      return;
    }
    entry->second.real = real;
    entry->second.ready = true;
  }
  state.ready.notify_all();
}

extern "C" bool lupine_handle_try_resolve(lupine_handle_family family,
                                          uintptr_t synthetic,
                                          uintptr_t *real) {
  if (real == nullptr) {
    return false;
  }
  if (!lupine_handle_is_synthetic(synthetic)) {
    *real = synthetic;
    return true;
  }
  if (!valid_family(family)) {
    *real = 0;
    return true;
  }
  lupine_handle_table &state = table();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto entry = state.entries[family].find(synthetic);
  if (entry == state.entries[family].end()) {
    *real = 0;
    return true;
  }
  if (!entry->second.ready) {
    return false;
  }
  *real = entry->second.real;
  return true;
}

extern "C" uintptr_t lupine_handle_resolve(lupine_handle_family family,
                                           uintptr_t synthetic) {
  if (!lupine_handle_is_synthetic(synthetic) || !valid_family(family)) {
    return lupine_handle_is_synthetic(synthetic) ? 0 : synthetic;
  }
  lupine_handle_table &state = table();
  std::unique_lock<std::mutex> lock(state.mutex);
  auto &entries = state.entries[family];
  state.ready.wait(lock, [&] {
    auto entry = entries.find(synthetic);
    return entry == entries.end() || entry->second.ready;
  });
  auto entry = entries.find(synthetic);
  return entry == entries.end() ? 0 : entry->second.real;
}

extern "C" void lupine_handle_forget(lupine_handle_family family,
                                     uintptr_t synthetic) {
  if (!lupine_handle_is_synthetic(synthetic) || !valid_family(family)) {
    return;
  }
  lupine_handle_table &state = table();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.entries[family].erase(synthetic);
  }
  state.ready.notify_all();
}
