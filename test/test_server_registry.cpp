#include "server_registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kThreadCount = 8;
constexpr std::size_t kIterations = 5000;

bool stress_concurrent_insert_lookup_remove() {
  lupine_handle_registry<std::uint64_t, std::uint64_t> registry;
  lupine_retirement_list<std::uint64_t> retired;
  std::atomic<bool> start{false};
  std::atomic<bool> failed{false};
  std::vector<std::thread> threads;

  for (std::size_t thread = 0; thread < kThreadCount; ++thread) {
    threads.emplace_back([&, thread]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (std::size_t iteration = 0; iteration < kIterations; ++iteration) {
        const std::uint64_t key = thread * kIterations + iteration;
        registry.store(key, key + 1);

        std::uint64_t value = 0;
        if (!registry.lookup(key, &value) || value != key + 1) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }

        if ((iteration & 1U) == 0) {
          std::uint64_t removed = 0;
          if (!registry.remove(key, &removed) || removed != key + 1) {
            failed.store(true, std::memory_order_relaxed);
            return;
          }
        }
        retired.retire(key);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  const std::size_t expected_entries = kThreadCount * (kIterations / 2);
  return !failed.load(std::memory_order_relaxed) &&
         registry.size() == expected_entries &&
         retired.size() == kThreadCount * kIterations;
}

bool stress_get_or_create() {
  using shared_value = std::shared_ptr<std::uint64_t>;
  lupine_handle_registry<std::uint64_t, shared_value> registry;
  std::atomic<bool> start{false};
  std::atomic<unsigned int> factory_calls{0};
  std::vector<shared_value> values(kThreadCount);
  std::vector<std::thread> threads;

  for (std::size_t thread = 0; thread < kThreadCount; ++thread) {
    threads.emplace_back([&, thread]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      values[thread] = registry.get_or_create(1, [&]() {
        factory_calls.fetch_add(1, std::memory_order_relaxed);
        return std::make_shared<std::uint64_t>(42);
      });
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  for (const auto &value : values) {
    if (value != values.front() || value == nullptr || *value != 42) {
      return false;
    }
  }
  return factory_calls.load(std::memory_order_relaxed) == 1;
}

bool snapshot_survives_concurrent_remove() {
  using shared_value = std::shared_ptr<std::uint64_t>;
  lupine_handle_registry<std::uint64_t, shared_value> registry;
  auto original = std::make_shared<std::uint64_t>(42);
  registry.store(1, original);

  std::atomic<bool> snapshot_ready{false};
  std::atomic<bool> removed{false};
  std::atomic<bool> failed{false};
  std::thread reader([&]() {
    shared_value snapshot;
    const bool found = registry.lookup(1, &snapshot);
    if (!found) {
      failed.store(true, std::memory_order_relaxed);
    }
    snapshot_ready.store(true, std::memory_order_release);
    if (!found) {
      return;
    }
    while (!removed.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (snapshot == nullptr || *snapshot != 42) {
      failed.store(true, std::memory_order_relaxed);
    }
  });

  while (!snapshot_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  shared_value retired;
  if (!registry.remove(1, &retired)) {
    failed.store(true, std::memory_order_relaxed);
  }
  original.reset();
  retired.reset();
  removed.store(true, std::memory_order_release);
  reader.join();

  return !failed.load(std::memory_order_relaxed) && registry.size() == 0;
}

} // namespace

int main() {
  if (!stress_concurrent_insert_lookup_remove()) {
    std::cerr << "concurrent insert/lookup/remove stress failed\n";
    return 1;
  }
  if (!stress_get_or_create()) {
    std::cerr << "concurrent get-or-create stress failed\n";
    return 1;
  }
  if (!snapshot_survives_concurrent_remove()) {
    std::cerr << "snapshot lifetime stress failed\n";
    return 1;
  }
  return 0;
}
