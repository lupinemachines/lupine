#ifndef LUPINE_SERVER_REGISTRY_H
#define LUPINE_SERVER_REGISTRY_H

#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

#include <atomic>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

// Concurrent handle metadata for values with process-stable lifetimes.
//
// libcuckoo owns the concurrent hash table. Each mapped value lives in a
// separate allocation so table resizing cannot invalidate atomic slots held by
// callers. Removing a key only clears its pointer value; a later store reuses
// the same slot. The caller owns the pointed-to value and must keep it alive
// while readers may still hold snapshots.
template <typename Key, typename Value, typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class lupine_registry {
  static_assert(std::is_pointer<Value>::value,
                "registry values must be pointers");
  static_assert(ATOMIC_POINTER_LOCK_FREE == 2,
                "registry requires always-lock-free atomic pointers");

  struct entry {
    std::atomic<Value> value{nullptr};
  };

  using map_type =
      libcuckoo::cuckoohash_map<Key, std::unique_ptr<entry>, Hash, KeyEqual>;

public:
  lupine_registry() = default;
  lupine_registry(const lupine_registry &) = delete;
  lupine_registry &operator=(const lupine_registry &) = delete;

  void store(const Key &key, Value value) {
    if (value == nullptr) {
      remove(key);
      return;
    }
    find_or_insert(key)->value.store(value, std::memory_order_release);
  }

  bool lookup(const Key &key, Value *value) const {
    entry *found = find(key);
    if (found == nullptr) {
      return false;
    }
    Value snapshot = found->value.load(std::memory_order_acquire);
    if (snapshot == nullptr) {
      return false;
    }
    if (value != nullptr) {
      *value = snapshot;
    }
    return true;
  }

  // Installs candidate only when the key currently has no value and returns
  // the installed value. The caller may destroy candidate when another value
  // wins.
  Value get_or_store(const Key &key, Value candidate) {
    if (candidate == nullptr) {
      Value found = nullptr;
      lookup(key, &found);
      return found;
    }
    entry *found = find_or_insert(key);
    Value expected = nullptr;
    if (found->value.compare_exchange_strong(expected, candidate,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
      return candidate;
    }
    return expected;
  }

  bool remove(const Key &key, Value *value = nullptr) {
    entry *found = find(key);
    if (found == nullptr) {
      return false;
    }
    Value removed = found->value.exchange(nullptr, std::memory_order_acq_rel);
    if (removed == nullptr) {
      return false;
    }
    if (value != nullptr) {
      *value = removed;
    }
    return true;
  }

  // Exposes a stable atomic slot for structures such as per-key stacks.
  std::atomic<Value> &slot(const Key &key) {
    return find_or_insert(key)->value;
  }

  // libcuckoo requires an exclusive table view for iteration. Keyed map
  // operations are only blocked for the duration of this full traversal.
  template <typename Visitor> void for_each_slot(Visitor visitor) {
    auto table = entries_.lock_table();
    for (auto &item : table) {
      visitor(item.first, item.second->value);
    }
  }

private:
  entry *find(const Key &key) const {
    entry *found = nullptr;
    entries_.find_fn(key, [&found](const std::unique_ptr<entry> &item) {
      found = item.get();
    });
    return found;
  }

  entry *find_or_insert(const Key &key) {
    if (entry *found = find(key)) {
      return found;
    }

    auto candidate = std::make_unique<entry>();
    entry *found = nullptr;
    entries_.upsert(
        key,
        [&found](std::unique_ptr<entry> &item, libcuckoo::UpsertContext) {
          found = item.get();
        },
        std::move(candidate));
    return found;
  }

  map_type entries_;
};

#endif
