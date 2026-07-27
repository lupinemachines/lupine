#include "server_registry.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kThreadCount = 8;
constexpr std::size_t kIterations = 5000;

bool stress_concurrent_insert_lookup_remove() {
  using value_type = const std::uint64_t *;
  lupine_registry<std::uint64_t, value_type> registry;
  std::vector<std::uint64_t> values(kThreadCount * kIterations);
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
        values[key] = key + 1;
        registry.store(key, &values[key]);

        value_type value = nullptr;
        if (!registry.lookup(key, &value) || value == nullptr ||
            *value != key + 1) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }

        if ((iteration & 1U) == 0) {
          value_type removed = nullptr;
          if (!registry.remove(key, &removed) || removed != &values[key]) {
            failed.store(true, std::memory_order_relaxed);
            return;
          }
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  for (std::size_t key = 0; key < values.size(); ++key) {
    value_type value = nullptr;
    const bool found = registry.lookup(key, &value);
    const bool expected = ((key % kIterations) & 1U) != 0;
    if (found != expected || (found && value != &values[key])) {
      return false;
    }
  }
  return !failed.load(std::memory_order_relaxed);
}

bool stress_get_or_store() {
  using value_type = const std::uint64_t *;
  lupine_registry<std::uint64_t, value_type> registry;
  std::vector<std::uint64_t> candidates(kThreadCount, 42);
  std::vector<value_type> values(kThreadCount);
  std::atomic<bool> start{false};
  std::vector<std::thread> threads;

  for (std::size_t thread = 0; thread < kThreadCount; ++thread) {
    threads.emplace_back([&, thread]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      values[thread] = registry.get_or_store(1, &candidates[thread]);
    });
  }

  start.store(true, std::memory_order_release);
  for (auto &thread : threads) {
    thread.join();
  }

  for (const auto value : values) {
    if (value != values.front() || value == nullptr || *value != 42) {
      return false;
    }
  }
  value_type stored = nullptr;
  return registry.lookup(1, &stored) && stored == values.front();
}

bool snapshot_survives_concurrent_remove() {
  using value_type = const std::uint64_t *;
  lupine_registry<std::uint64_t, value_type> registry;
  std::uint64_t original = 42;
  registry.store(1, &original);

  std::atomic<bool> snapshot_ready{false};
  std::atomic<bool> removed{false};
  std::atomic<bool> failed{false};
  std::thread reader([&]() {
    value_type snapshot = nullptr;
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
  value_type retired = nullptr;
  if (!registry.remove(1, &retired) || retired != &original) {
    failed.store(true, std::memory_order_relaxed);
  }
  removed.store(true, std::memory_order_release);
  reader.join();

  return !failed.load(std::memory_order_relaxed) &&
         !registry.lookup(1, nullptr);
}

bool slot_survives_table_growth() {
  using value_type = const std::uint64_t *;
  lupine_registry<std::uint64_t, value_type> registry;
  std::vector<std::uint64_t> values(kIterations);

  auto *original_slot = &registry.slot(0);
  for (std::size_t key = 0; key < values.size(); ++key) {
    values[key] = key + 1;
    registry.store(key, &values[key]);
  }

  std::size_t slot_count = 0;
  registry.for_each_slot(
      [&slot_count](std::uint64_t, std::atomic<value_type> &) {
        ++slot_count;
      });

  value_type value = nullptr;
  return &registry.slot(0) == original_slot && registry.lookup(0, &value) &&
         value == &values[0] && slot_count == values.size();
}

} // namespace

int main() {
  if (!stress_concurrent_insert_lookup_remove()) {
    std::cerr << "concurrent insert/lookup/remove stress failed\n";
    return 1;
  }
  if (!stress_get_or_store()) {
    std::cerr << "concurrent get-or-store stress failed\n";
    return 1;
  }
  if (!snapshot_survives_concurrent_remove()) {
    std::cerr << "snapshot lifetime stress failed\n";
    return 1;
  }
  if (!slot_survives_table_growth()) {
    std::cerr << "slot stability stress failed\n";
    return 1;
  }
  return 0;
}
