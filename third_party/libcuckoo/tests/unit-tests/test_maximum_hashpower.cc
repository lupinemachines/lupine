#include "test_maximum_hashpower.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <iostream>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_config.hh>
#include <libcuckoo/cuckoohash_map.hh>

void test_maximum_hashpower_initialized_to_default() {
  IntIntTable tbl;
  TEST_CHECK(tbl.maximum_hashpower() == libcuckoo::NO_MAXIMUM_HASHPOWER);
}

void test_maximum_hashpower_caps_any_expansion() {
  IntIntTable tbl(1);
  tbl.maximum_hashpower(1);
  for (size_t i = 0; i < 2 * tbl.slot_per_bucket(); ++i) {
    tbl.insert(i, i);
  }

  TEST_CHECK(tbl.hashpower() == 1);
  TEST_EXCEPTION(tbl.insert(2 * tbl.slot_per_bucket(), 0),
                 libcuckoo::maximum_hashpower_exceeded);
  TEST_EXCEPTION(tbl.rehash(2), libcuckoo::maximum_hashpower_exceeded);
  TEST_EXCEPTION(tbl.reserve(4 * tbl.slot_per_bucket()),
                 libcuckoo::maximum_hashpower_exceeded);
}

void test_maximum_hash_power_none() {
  // It's difficult to check that we actually don't ever set a maximum hash
  // power, but if we explicitly unset it, we should be able to expand beyond
  // the limit that we had previously set.
  IntIntTable tbl(1);
  tbl.maximum_hashpower(1);
  TEST_EXCEPTION(tbl.rehash(2), libcuckoo::maximum_hashpower_exceeded);

  tbl.maximum_hashpower(2);
  tbl.rehash(2);
  TEST_CHECK(tbl.hashpower() == 2);
  TEST_EXCEPTION(tbl.rehash(3), libcuckoo::maximum_hashpower_exceeded);

  tbl.maximum_hashpower(libcuckoo::NO_MAXIMUM_HASHPOWER);
  tbl.rehash(10);
  TEST_CHECK(tbl.hashpower() == 10);
}
