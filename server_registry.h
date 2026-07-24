#ifndef LUPINE_SERVER_REGISTRY_H
#define LUPINE_SERVER_REGISTRY_H

#include <cstddef>
#include <functional>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Thread-safe storage for server-side handles. Values are copied out while the
// mutex is held, so callers never retain a reference or iterator that a
// concurrent insertion or removal could invalidate. Registry values therefore
// need copy semantics; shared_ptr should be used for state with shared
// lifetime.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class lupine_handle_registry {
public:
  static_assert(std::is_default_constructible<Value>::value,
                "registry values must be default constructible");
  static_assert(std::is_copy_constructible<Value>::value,
                "registry values must support snapshot copies");
  static_assert(std::is_copy_assignable<Value>::value,
                "registry values must support snapshot copies");

  void store(Key key, Value value) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found == entries_.end()) {
      entries_.emplace(std::move(key), std::move(value));
    } else {
      std::swap(found->second, value);
    }
  }

  bool lookup(const Key &key, Value *value) const {
    Value snapshot{};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = entries_.find(key);
      if (found == entries_.end()) {
        return false;
      }
      snapshot = found->second;
    }
    if (value != nullptr) {
      *value = std::move(snapshot);
    }
    return true;
  }

  template <typename Factory>
  Value get_or_create(const Key &key, Factory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      return found->second;
    }
    Value value = factory();
    auto inserted = entries_.emplace(key, std::move(value));
    return inserted.first->second;
  }

  bool remove(const Key &key, Value *value = nullptr) {
    Value removed{};
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = entries_.find(key);
      if (found == entries_.end()) {
        return false;
      }
      removed = std::move(found->second);
      entries_.erase(found);
    }
    if (value != nullptr) {
      *value = std::move(removed);
    }
    return true;
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<Key, Value, Hash> entries_;
};

// Some CUDA graph resources must outlive destroyed handles because queued GPU
// work and callbacks may still reference their host allocations. This list is
// intentionally process-lifetime storage; synchronization makes concurrent
// handle retirement safe without changing that lifetime policy.
template <typename Value> class lupine_retirement_list {
public:
  void retire(Value value) {
    std::lock_guard<std::mutex> lock(mutex_);
    values_.push_back(std::move(value));
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return values_.size();
  }

private:
  mutable std::mutex mutex_;
  std::vector<Value> values_;
};

#endif
