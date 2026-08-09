#include "handles.h"

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct lupine_handle_entry {
  uintptr_t real = 0;
  bool ready = false;
  void *payload = nullptr;
  bool applying = false;
};

struct lupine_handle_table {
  std::mutex mutex;
  std::condition_variable ready;
  uint64_t counter = 0;
  std::unordered_map<uintptr_t, lupine_handle_entry>
      entries[LUPINE_HANDLE_FAMILY_COUNT];
  std::unordered_map<uintptr_t, std::unordered_map<std::string, uintptr_t>>
      named[LUPINE_HANDLE_FAMILY_COUNT];
};

lupine_handle_table &table() {
  static lupine_handle_table instance;
  return instance;
}

bool valid_family(lupine_handle_family family) {
  return family < LUPINE_HANDLE_FAMILY_COUNT;
}

uintptr_t mint_locked(lupine_handle_table &state, lupine_handle_family family) {
  uintptr_t synthetic =
      static_cast<uintptr_t>((LUPINE_HANDLE_TAG << LUPINE_HANDLE_TAG_SHIFT) |
                             ((static_cast<uint64_t>(family) & 0xFFull) << 40) |
                             ((++state.counter) & 0xFFFFFFFFFFull));
  state.entries[family][synthetic] = lupine_handle_entry{};
  return synthetic;
}

} // namespace

extern "C" bool lupine_handle_is_synthetic(uintptr_t value) {
  return (static_cast<uint64_t>(value) & LUPINE_HANDLE_TAG_MASK) ==
         (LUPINE_HANDLE_TAG << LUPINE_HANDLE_TAG_SHIFT);
}

extern "C" lupine_handle_family lupine_handle_family_of(uintptr_t value) {
  if (!lupine_handle_is_synthetic(value)) {
    return LUPINE_HANDLE_FAMILY_COUNT;
  }
  auto family = static_cast<lupine_handle_family>(
      (static_cast<uint64_t>(value) >> 40) & 0xFFull);
  return valid_family(family) ? family : LUPINE_HANDLE_FAMILY_COUNT;
}

extern "C" uintptr_t lupine_handle_mint(lupine_handle_family family) {
  if (!valid_family(family)) {
    return 0;
  }
  lupine_handle_table &state = table();
  std::lock_guard<std::mutex> lock(state.mutex);
  return mint_locked(state, family);
}

extern "C" uintptr_t lupine_handle_mint_named(lupine_handle_family family,
                                              uintptr_t owner,
                                              const char *name) {
  if (!valid_family(family) || name == nullptr) {
    return 0;
  }
  lupine_handle_table &state = table();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto &owned = state.named[family][owner];
  auto existing = owned.find(name);
  if (existing != owned.end()) {
    return existing->second;
  }
  uintptr_t synthetic = mint_locked(state, family);
  owned.emplace(name, synthetic);
  return synthetic;
}

extern "C" void lupine_handle_fulfill(lupine_handle_family family,
                                      uintptr_t synthetic, uintptr_t real) {
  lupine_handle_fulfill_payload(family, synthetic, real, nullptr);
}

extern "C" bool lupine_handle_fulfill_payload(lupine_handle_family family,
                                              uintptr_t synthetic,
                                              uintptr_t real, void *payload) {
  if (!valid_family(family)) {
    return false;
  }
  lupine_handle_table &state = table();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto entry = state.entries[family].find(synthetic);
    if (entry == state.entries[family].end()) {
      return false;
    }
    entry->second.real = real;
    entry->second.payload = payload;
    entry->second.ready = true;
  }
  state.ready.notify_all();
  return true;
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
  return lupine_handle_resolve_applied(family, synthetic, nullptr, nullptr);
}

extern "C" uintptr_t lupine_handle_resolve_applied(lupine_handle_family family,
                                                   uintptr_t synthetic,
                                                   lupine_handle_apply_fn apply,
                                                   void *context) {
  if (!lupine_handle_is_synthetic(synthetic) || !valid_family(family)) {
    return lupine_handle_is_synthetic(synthetic) ? 0 : synthetic;
  }
  lupine_handle_table &state = table();
  std::unique_lock<std::mutex> lock(state.mutex);
  auto &entries = state.entries[family];
  state.ready.wait(lock, [&] {
    auto entry = entries.find(synthetic);
    return entry == entries.end() ||
           (entry->second.ready && !entry->second.applying);
  });
  auto entry = entries.find(synthetic);
  if (entry == entries.end()) {
    return 0;
  }
  if (entry->second.payload == nullptr || apply == nullptr) {
    return entry->second.real;
  }
  void *payload = entry->second.payload;
  uintptr_t real = entry->second.real;
  entry->second.payload = nullptr;
  entry->second.applying = true;
  lock.unlock();
  apply(synthetic, real, payload, context);
  lock.lock();
  entry = entries.find(synthetic);
  if (entry != entries.end()) {
    entry->second.applying = false;
  }
  lock.unlock();
  state.ready.notify_all();
  return real;
}

extern "C" void lupine_handle_forget(lupine_handle_family family,
                                     uintptr_t synthetic) {
  lupine_handle_forget_payload(family, synthetic);
}

extern "C" void *lupine_handle_forget_payload(lupine_handle_family family,
                                              uintptr_t synthetic) {
  if (!lupine_handle_is_synthetic(synthetic) || !valid_family(family)) {
    return nullptr;
  }
  lupine_handle_table &state = table();
  void *payload = nullptr;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto entry = state.entries[family].find(synthetic);
    if (entry != state.entries[family].end()) {
      payload = entry->second.payload;
      state.entries[family].erase(entry);
    }
  }
  state.ready.notify_all();
  return payload;
}

extern "C" void lupine_handle_forget_owned(lupine_handle_family family,
                                           uintptr_t owner) {
  if (!valid_family(family)) {
    return;
  }
  lupine_handle_table &state = table();
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto owned = state.named[family].find(owner);
    if (owned == state.named[family].end()) {
      return;
    }
    for (const auto &entry : owned->second) {
      state.entries[family].erase(entry.second);
    }
    state.named[family].erase(owned);
  }
  state.ready.notify_all();
}
