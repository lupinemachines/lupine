#include "deviceptr_allocations.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

static void expect_owner(const lupine_deviceptr_allocation_index &index,
                         CUdeviceptr ptr, int route, const char *description) {
  const auto *allocation = index.find(ptr);
  expect(allocation != nullptr && allocation->route_id == route &&
             allocation->context == context(route + 2),
         description);
}

static void test_boundaries_and_replacement() {
  lupine_deviceptr_allocation_index index;
  expect(index.find(100) == nullptr, "empty index");
  index.insert_or_assign(100, 20, 0, context(2));
  index.insert_or_assign(120, 10, 1, context(3));
  expect(index.find(99) == nullptr, "before first allocation");
  expect_owner(index, 100, 0, "allocation base");
  expect_owner(index, 119, 0, "allocation last byte");
  expect_owner(index, 120, 1, "touching allocations do not overlap");
  expect(index.find(130) == nullptr, "exclusive endpoint");
  index.insert_or_assign(100, 5, 2, context(4));
  expect_owner(index, 104, 2, "replacement changes route and context");
  expect(index.find(105) == nullptr, "replacement shrinks range");
  index.insert_or_assign(100, 0, 3, context(5));
  expect(index.find(100) == nullptr, "unknown size removes range");
  index.erase(120);
  index.erase(120);
  expect(index.find(125) == nullptr, "erasing an allocation is idempotent");
  index.insert_or_assign(0, 100, 0, context(2));
  expect(index.find(1) == nullptr, "null base is not an allocation");
}

static void test_overlapping_ranges() {
  lupine_deviceptr_allocation_index index;
  index.insert_or_assign(100, 100, 0, context(2));
  index.insert_or_assign(120, 10, 1, context(3));
  index.insert_or_assign(140, 10, 2, context(4));
  expect_owner(index, 125, 1, "nested allocation has precedence");
  expect_owner(index, 135, 0, "outer allocation behind shorter predecessor");
  expect_owner(index, 190, 0, "outer allocation behind several predecessors");
  index.erase(120);
  expect_owner(index, 190, 0, "deletion reconnects overlapping neighbors");
  index.insert_or_assign(100, 30, 3, context(5));
  expect(index.find(135) == nullptr, "shrinking outer allocation opens gap");
  index.insert_or_assign(120, 100, 4, context(6));
  expect_owner(index, 210, 4, "partially overlapping allocation");
  index.insert_or_assign(120, 0, 4, context(6));
  expect(index.find(210) == nullptr,
         "zero size removes overlapping allocation");
  index.insert_or_assign(100, 100, 5, context(7));
  expect_owner(index, 190, 5, "replacement restores enclosing allocation");
  index.erase(100);
  expect(index.find(190) == nullptr, "removing outer preserves only inner");
  expect_owner(index, 145, 2, "inner survives outer deletion");
}

static void test_overflow() {
  const CUdeviceptr maximum = std::numeric_limits<CUdeviceptr>::max();
  lupine_deviceptr_allocation_index index;
  index.insert_or_assign(maximum - 99, 200, 0, context(2));
  index.insert_or_assign(maximum - 49, 10, 1, context(3));
  expect_owner(index, maximum - 45, 1, "high nested allocation");
  expect_owner(index, maximum, 0, "range endpoint does not wrap");
  expect(index.find(5) == nullptr, "wrapped low address is not covered");
  index.erase(maximum - 99);
  expect(index.find(maximum) == nullptr, "high enclosing allocation removed");
}

static void test_against_linear_lookup() {
  lupine_deviceptr_allocation_index index;
  std::map<CUdeviceptr, lupine_deviceptr_allocation_record> reference;
  std::mt19937 random(314159);
  for (int step = 0; step < 2000; ++step) {
    CUdeviceptr base = 1 + random() % 256;
    size_t size = random() % 128;
    int route = random() % 8;
    if (random() % 4 == 0 || size == 0) {
      index.erase(base);
      reference.erase(base);
    } else {
      index.insert_or_assign(base, size, route, context(route + 2));
      reference[base] = {size, route, context(route + 2)};
    }
    for (CUdeviceptr ptr = 0; ptr < 400; ++ptr) {
      const lupine_deviceptr_allocation_record *expected = nullptr;
      for (const auto &entry : reference) {
        if (ptr >= entry.first && ptr - entry.first < entry.second.size) {
          expected = &entry.second;
        }
      }
      const auto *actual = index.find(ptr);
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
}

int main() {
  test_boundaries_and_replacement();
  test_overlapping_ranges();
  test_overflow();
  test_against_linear_lookup();
  printf("deviceptr_allocations_test passed\n");
}
