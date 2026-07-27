#include "test_user_exceptions.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <stdexcept>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

using libcuckoo::UnitTestInternalAccess;

void maybeThrow(bool throwException) {
  if (throwException) {
    throw std::runtime_error("exception");
  }
}

bool constructorThrow, moveThrow, hashThrow, equalityThrow;

class ExceptionInt {
public:
  ExceptionInt() {
    maybeThrow(constructorThrow);
    val = 0;
  }

  ExceptionInt(size_t x) {
    maybeThrow(constructorThrow);
    val = x;
  }

  ExceptionInt(const ExceptionInt &i) {
    maybeThrow(constructorThrow);
    val = static_cast<size_t>(i);
  }

  ExceptionInt(ExceptionInt &&i) {
    maybeThrow(constructorThrow || moveThrow);
    val = static_cast<size_t>(i);
  }

  ExceptionInt &operator=(const ExceptionInt &i) {
    maybeThrow(constructorThrow);
    val = static_cast<size_t>(i);
    return *this;
  }

  ExceptionInt &operator=(ExceptionInt &&i) {
    maybeThrow(constructorThrow || moveThrow);
    val = static_cast<size_t>(i);
    return *this;
  }

  operator size_t() const { return val; }

private:
  size_t val;
};

namespace std {
template <> struct hash<ExceptionInt> {
  size_t operator()(const ExceptionInt &x) const {
    maybeThrow(hashThrow);
    return x;
  }
};

template <> struct equal_to<ExceptionInt> {
  bool operator()(const ExceptionInt &lhs, const ExceptionInt &rhs) const {
    maybeThrow(equalityThrow);
    return static_cast<size_t>(lhs) == static_cast<size_t>(rhs);
  }
};
} // namespace std

typedef libcuckoo::cuckoohash_map<ExceptionInt, size_t, std::hash<ExceptionInt>,
                                  std::equal_to<ExceptionInt>>
    exceptionTable;

void checkIterTable(exceptionTable &tbl, size_t expectedSize) {
  auto lockedTable = tbl.lock_table();
  size_t actualSize = 0;
  for (auto it = lockedTable.begin(); it != lockedTable.end(); ++it) {
    ++actualSize;
  }
  TEST_CHECK(actualSize == expectedSize);
}

void test_user_exceptions() {
  constructorThrow = hashThrow = equalityThrow = moveThrow = false;

  // "find/contains"
  {
    exceptionTable tbl;
    tbl.insert(1, 1);
    tbl.insert(2, 2);
    tbl.insert(3, 3);
    hashThrow = true;
    TEST_EXCEPTION(tbl.find(3), std::runtime_error);
    TEST_EXCEPTION(tbl.contains(3), std::runtime_error);
    hashThrow = false;
    equalityThrow = true;
    TEST_EXCEPTION(tbl.find(3), std::runtime_error);
    TEST_EXCEPTION(tbl.contains(3), std::runtime_error);
    equalityThrow = false;
    TEST_CHECK(tbl.find(3) == 3);
    TEST_CHECK(tbl.contains(3));
    checkIterTable(tbl, 3);
  }

  // "insert"
  {
    exceptionTable tbl;
    constructorThrow = true;
    TEST_EXCEPTION(tbl.insert(100, 100), std::runtime_error);
    constructorThrow = false;
    TEST_CHECK(tbl.insert(100, 100));
    checkIterTable(tbl, 1);
  }

  // "erase"
  {
    exceptionTable tbl;
    for (int i = 0; i < 10; ++i) {
      tbl.insert(i, i);
    }
    hashThrow = true;
    TEST_EXCEPTION(tbl.erase(5), std::runtime_error);
    hashThrow = false;
    equalityThrow = true;
    TEST_EXCEPTION(tbl.erase(5), std::runtime_error);
    equalityThrow = false;
    TEST_CHECK(tbl.erase(5));
    checkIterTable(tbl, 9);
  }

  // "update"
  {
    exceptionTable tbl;
    tbl.insert(9, 9);
    tbl.insert(10, 10);
    hashThrow = true;
    TEST_EXCEPTION(tbl.update(9, 10), std::runtime_error);
    hashThrow = false;
    equalityThrow = true;
    TEST_EXCEPTION(tbl.update(9, 10), std::runtime_error);
    equalityThrow = false;
    TEST_CHECK(tbl.update(9, 10));
    checkIterTable(tbl, 2);
  }

  // "update_fn"
  {
    exceptionTable tbl;
    tbl.insert(9, 9);
    tbl.insert(10, 10);
    auto updater = [](size_t &val) { val++; };
    hashThrow = true;
    TEST_EXCEPTION(tbl.update_fn(9, updater), std::runtime_error);
    hashThrow = false;
    equalityThrow = true;
    TEST_EXCEPTION(tbl.update_fn(9, updater), std::runtime_error);
    equalityThrow = false;
    TEST_CHECK(tbl.update_fn(9, updater));
    checkIterTable(tbl, 2);
  }

  // "upsert"
  {
    exceptionTable tbl;
    tbl.insert(9, 9);
    auto updater = [](size_t &val) { val++; };
    hashThrow = true;
    TEST_EXCEPTION(tbl.upsert(9, updater, 10), std::runtime_error);
    hashThrow = false;
    equalityThrow = true;
    TEST_EXCEPTION(tbl.upsert(9, updater, 10), std::runtime_error);
    equalityThrow = false;
    tbl.upsert(9, updater, 10);
    constructorThrow = true;
    TEST_EXCEPTION(tbl.upsert(10, updater, 10), std::runtime_error);
    constructorThrow = false;
    tbl.upsert(10, updater, 10);
    checkIterTable(tbl, 2);
  }

  // rehash
  {
    exceptionTable tbl;
    for (int i = 0; i < 10; ++i) {
      tbl.insert(i, i);
    }
    size_t original_hashpower = tbl.hashpower();
    size_t next_hashpower = original_hashpower + 1;
    constructorThrow = true;
    TEST_EXCEPTION(tbl.rehash(next_hashpower), std::runtime_error);
    constructorThrow = false;
    TEST_CHECK(tbl.hashpower() == original_hashpower);
    hashThrow = true;
    TEST_EXCEPTION(tbl.rehash(next_hashpower), std::runtime_error);
    hashThrow = false;
    TEST_CHECK(tbl.hashpower() == original_hashpower);
    // This shouldn't throw, because the partial keys are different between
    // the different hash values, which means they shouldn't be compared for
    // actual equality.
    equalityThrow = true;
    TEST_CHECK(tbl.rehash(next_hashpower));
    TEST_CHECK(tbl.hashpower() == next_hashpower);
    equalityThrow = false;
    checkIterTable(tbl, 10);
  }

  // "reserve"
  {
    exceptionTable tbl;
    for (int i = 0; i < 10; ++i) {
      tbl.insert(i, i);
    }
    size_t original_hashpower = tbl.hashpower();
    size_t next_hashpower = original_hashpower + 1;
    size_t next_reserve = (1UL << next_hashpower) * tbl.slot_per_bucket();
    constructorThrow = true;
    TEST_EXCEPTION(tbl.reserve(next_reserve), std::runtime_error);
    constructorThrow = false;
    TEST_CHECK(tbl.hashpower() == original_hashpower);
    hashThrow = true;
    TEST_EXCEPTION(tbl.reserve(next_reserve), std::runtime_error);
    hashThrow = false;
    TEST_CHECK(tbl.hashpower() == original_hashpower);
    // This shouldn't throw, because the partial keys are different between
    // the different hash values, which means they shouldn't be compared for
    // actual equality.
    equalityThrow = true;
    TEST_CHECK(tbl.reserve(next_reserve));
    TEST_CHECK(tbl.hashpower() == next_hashpower);
    checkIterTable(tbl, 10);
  }

  // "insert resize"
  {
    exceptionTable tbl(1000);
    TEST_CHECK(tbl.rehash(1));
    // Fill up the entire table
    for (size_t i = 0; i < exceptionTable::slot_per_bucket() * 2; ++i) {
      tbl.insert(i * 2, 0);
    }
    // Only throw on move, which should be triggered when we do a resize.
    moveThrow = true;
    TEST_EXCEPTION(tbl.insert((exceptionTable::slot_per_bucket() * 2) * 2, 0),
                   std::runtime_error);
    moveThrow = false;
    TEST_CHECK(tbl.insert((exceptionTable::slot_per_bucket() * 2) * 2, 0));
    checkIterTable(tbl, exceptionTable::slot_per_bucket() * 2 + 1);
  }

  // "insert cuckoohash" -- broken?
  // {
  //     exceptionTable tbl(0);
  //     TEST_CHECK(tbl.rehash(2));
  //     size_t cuckooKey = 0;
  //     size_t cuckooKeyHash = std::hash<ExceptionInt>()(cuckooKey);
  //     size_t cuckooKeyIndex = UnitTestInternalAccess::index_hash<
  //         exceptionTable>(tbl.hashpower(), cuckooKeyHash);
  //     size_t cuckooKeyPartial = UnitTestInternalAccess::partial_key<
  //         exceptionTable>(tbl.hashpower(), cuckooKeyHash);
  //     size_t cuckooKeyAltIndex = UnitTestInternalAccess::alt_index<
  //         exceptionTable>(tbl.hashpower(), cuckooKeyPartial, cuckooKeyIndex);

  //     if (cuckooKeyIndex == cuckooKeyAltIndex) {
  //         // Fill up one bucket with elements that have the same value as
  //         // cuckooKeyIndex mod tbl.hashpower()
  //         for (size_t i = 0; i < exceptionTable::slot_per_bucket; ++i) {
  //             tbl.insert(tbl.hashpower() * (i + 1) + cuckooKeyIndex, 0);
  //         }
  //     } else {
  //         // Fill up one bucket on index cuckooKeyIndex, and another bucket
  //         on
  //         // cuckooKeyAlt
  //         for (size_t i = 0; i < exceptionTable::slot_per_bucket; ++i) {
  //             tbl.insert(tbl.hashpower() * (i + 1) + cuckooKeyIndex, 0);
  //             tbl.insert(tbl.hashpower() * (i + 1) + cuckooKeyAltIndex, 0);
  //         }
  //     }

  //     // Now inserting cuckooKey should trigger a cuckoo hash, which moves
  //     // elements around
  //     moveThrow = true;
  //     TEST_EXCEPTION(tbl.insert(cuckooKey, 0), std::runtime_error);
  //     moveThrow = false;
  //     TEST_CHECK(tbl.insert(cuckooKey, 0));
  //     checkIterTable(tbl,
  //                    exceptionTable::slot_per_bucket * (
  //                        cuckooKeyIndex == cuckooKeyAltIndex ? 1 : 2) + 1);
  // }
}
