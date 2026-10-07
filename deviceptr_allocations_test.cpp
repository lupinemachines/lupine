#include "cache.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>

static void expect(bool condition, const char *description) {
  if (!condition) {
    fprintf(stderr, "FAILED: %s\n", description);
    std::exit(1);
  }
}

static CUcontext context(int id) {
  return reinterpret_cast<CUcontext>(static_cast<uintptr_t>(id));
}

static void expect_owner(CUdeviceptr ptr, int route, const char *description) {
  const auto *allocation = lupine_deviceptr_allocation_cache_lookup(ptr);
  expect(allocation != nullptr && allocation->route_id == route &&
             allocation->context == context(route + 2),
         description);
}

static void test_boundaries_and_replacement() {
  expect(lupine_deviceptr_allocation_cache_lookup(100) == nullptr,
         "empty index");
  lupine_deviceptr_allocation_cache_insert(100, 20, 0, context(2));
  lupine_deviceptr_allocation_cache_insert(120, 10, 1, context(3));
  expect(lupine_deviceptr_allocation_cache_lookup(99) == nullptr,
         "before first allocation");
  expect_owner(100, 0, "allocation base");
  expect_owner(119, 0, "allocation last byte");
  expect_owner(120, 1, "touching allocations do not overlap");
  expect(lupine_deviceptr_allocation_cache_lookup(130) == nullptr,
         "exclusive endpoint");
  lupine_deviceptr_allocation_cache_insert(100, 5, 2, context(4));
  expect_owner(104, 2, "replacement changes route and context");
  expect(lupine_deviceptr_allocation_cache_lookup(105) == nullptr,
         "replacement shrinks range");
  lupine_deviceptr_allocation_cache_insert(100, 0, 3, context(5));
  expect(lupine_deviceptr_allocation_cache_lookup(100) == nullptr,
         "unknown size removes range");
  lupine_deviceptr_allocation_cache_erase(120);
  lupine_deviceptr_allocation_cache_erase(120);
  expect(lupine_deviceptr_allocation_cache_lookup(125) == nullptr,
         "erasing an allocation is idempotent");
  lupine_deviceptr_allocation_cache_insert(0, 100, 0, context(2));
  expect(lupine_deviceptr_allocation_cache_lookup(1) == nullptr,
         "null base is not an allocation");
}

static void test_overlapping_ranges() {
  lupine_deviceptr_allocation_cache_insert(100, 100, 0, context(2));
  lupine_deviceptr_allocation_cache_insert(120, 10, 1, context(3));
  lupine_deviceptr_allocation_cache_insert(140, 10, 2, context(4));
  expect_owner(125, 1, "nested allocation has precedence");
  expect_owner(135, 0, "outer allocation behind shorter predecessor");
  expect_owner(190, 0, "outer allocation behind several predecessors");
  lupine_deviceptr_allocation_cache_erase(120);
  expect_owner(190, 0, "deletion reconnects overlapping neighbors");
  lupine_deviceptr_allocation_cache_insert(100, 30, 3, context(5));
  expect(lupine_deviceptr_allocation_cache_lookup(135) == nullptr,
         "shrinking outer allocation opens gap");
  lupine_deviceptr_allocation_cache_insert(120, 100, 4, context(6));
  expect_owner(210, 4, "partially overlapping allocation");
  lupine_deviceptr_allocation_cache_insert(120, 0, 4, context(6));
  expect(lupine_deviceptr_allocation_cache_lookup(210) == nullptr,
         "zero size removes overlapping allocation");
  lupine_deviceptr_allocation_cache_insert(100, 100, 5, context(7));
  expect_owner(190, 5, "replacement restores enclosing allocation");
  lupine_deviceptr_allocation_cache_erase(100);
  expect(lupine_deviceptr_allocation_cache_lookup(190) == nullptr,
         "removing outer preserves only inner");
  expect_owner(145, 2, "inner survives outer deletion");
  lupine_deviceptr_allocation_cache_erase(140);
}

static void test_overflow() {
  const CUdeviceptr maximum = std::numeric_limits<CUdeviceptr>::max();
  lupine_deviceptr_allocation_cache_insert(maximum - 99, 200, 0, context(2));
  lupine_deviceptr_allocation_cache_insert(maximum - 49, 10, 1, context(3));
  expect_owner(maximum - 45, 1, "high nested allocation");
  expect_owner(maximum, 0, "range endpoint does not wrap");
  expect(lupine_deviceptr_allocation_cache_lookup(5) == nullptr,
         "wrapped low address is not covered");
  lupine_deviceptr_allocation_cache_erase(maximum - 99);
  expect(lupine_deviceptr_allocation_cache_lookup(maximum) == nullptr,
         "high enclosing allocation removed");
  lupine_deviceptr_allocation_cache_erase(maximum - 49);
}

static void test_against_linear_lookup() {
  std::map<CUdeviceptr, lupine_deviceptr_allocation_record> reference;
  std::mt19937 random(314159);
  for (int step = 0; step < 2000; ++step) {
    CUdeviceptr base = 1 + random() % 256;
    size_t size = random() % 128;
    int route = random() % 8;
    if (random() % 4 == 0 || size == 0) {
      lupine_deviceptr_allocation_cache_erase(base);
      reference.erase(base);
    } else {
      lupine_deviceptr_allocation_cache_insert(base, size, route,
                                               context(route + 2));
      reference[base] = {size, route, context(route + 2)};
    }
    for (CUdeviceptr ptr = 0; ptr < 400; ++ptr) {
      const lupine_deviceptr_allocation_record *expected = nullptr;
      for (const auto &entry : reference) {
        if (ptr >= entry.first && ptr - entry.first < entry.second.size) {
          expected = &entry.second;
        }
      }
      const auto *actual = lupine_deviceptr_allocation_cache_lookup(ptr);
      expect((actual == nullptr) == (expected == nullptr),
             "random mutations preserve coverage");
      if (expected != nullptr) {
        expect(actual->route_id == expected->route_id &&
                   actual->context == expected->context &&
                   actual->size == expected->size,
               "random mutations preserve containing allocation ownership");
      }
    }
  }
  for (const auto &entry : reference) {
    lupine_deviceptr_allocation_cache_erase(entry.first);
  }
}

int main() {
  test_boundaries_and_replacement();
  test_overlapping_ranges();
  test_overflow();
  test_against_linear_lookup();
  printf("deviceptr_allocations_test passed\n");
}
