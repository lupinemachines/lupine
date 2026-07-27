#include "test_hash_properties.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_map.hh>

using libcuckoo::UnitTestInternalAccess;

// Checks that the alt index function returns a different bucket, and can
// recover the old bucket when called with the alternate bucket as the index.
template <class CuckoohashMap>
void check_key(size_t hashpower, const typename CuckoohashMap::key_type &key) {
  auto hashfn = typename CuckoohashMap::hasher();
  size_t hv = hashfn(key);
  auto partial = UnitTestInternalAccess::partial_key<CuckoohashMap>(hv);
  size_t bucket =
      UnitTestInternalAccess::index_hash<CuckoohashMap>(hashpower, hv);
  size_t alt_bucket = UnitTestInternalAccess::alt_index<CuckoohashMap>(
      hashpower, partial, bucket);
  size_t orig_bucket = UnitTestInternalAccess::alt_index<CuckoohashMap>(
      hashpower, partial, alt_bucket);

  TEST_CHECK(bucket != alt_bucket);
  TEST_CHECK(bucket == orig_bucket);
}

void test_hash_properties_int_alt_index_works_correctly() {
  for (size_t hashpower = 10; hashpower < 15; ++hashpower) {
    for (int key = 0; key < 10000; ++key) {
      check_key<IntIntTable>(hashpower, key);
    }
  }
}

void test_hash_properties_string_alt_index_works_correctly() {
  for (size_t hashpower = 10; hashpower < 15; ++hashpower) {
    for (int key = 0; key < 10000; ++key) {
      check_key<StringIntTable>(hashpower, std::to_string(key));
    }
  }
}

void test_hash_properties_hash_with_larger_hashpower_only_adds_top_bits() {
  std::string key = "abc";
  size_t hv = StringIntTable::hasher()(key);
  for (size_t hashpower = 1; hashpower < 30; ++hashpower) {
    auto partial = UnitTestInternalAccess::partial_key<StringIntTable>(hv);
    size_t index_bucket1 =
        UnitTestInternalAccess::index_hash<StringIntTable>(hashpower, hv);
    size_t index_bucket2 =
        UnitTestInternalAccess::index_hash<StringIntTable>(hashpower + 1, hv);
    TEST_CHECK((index_bucket2 & ~(1L << hashpower)) == index_bucket1);

    size_t alt_bucket1 = UnitTestInternalAccess::alt_index<StringIntTable>(
        hashpower, partial, index_bucket1);
    size_t alt_bucket2 = UnitTestInternalAccess::alt_index<StringIntTable>(
        hashpower, partial, index_bucket2);

    TEST_CHECK((alt_bucket2 & ~(1L << hashpower)) == alt_bucket1);
  }
}
