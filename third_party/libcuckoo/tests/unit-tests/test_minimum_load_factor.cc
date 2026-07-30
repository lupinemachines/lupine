#include "test_minimum_load_factor.h"

#define TEST_NO_MAIN
#include "acutest.h"

#include <iostream>

#include "unit_test_util.hh"
#include <libcuckoo/cuckoohash_config.hh>
#include <libcuckoo/cuckoohash_map.hh>

void test_minimum_load_factor_initialized_to_default() {
  IntIntTable tbl;
  TEST_CHECK(tbl.minimum_load_factor() ==
             libcuckoo::DEFAULT_MINIMUM_LOAD_FACTOR);
}

class BadHashFunction {
public:
  size_t operator()(int) { return 0; }
};

void test_minimum_load_factor_caps_automatic_expansion() {
  const size_t slot_per_bucket = 4;
  libcuckoo::cuckoohash_map<int, int, BadHashFunction, std::equal_to<int>,
                            std::allocator<std::pair<const int, int>>,
                            slot_per_bucket>
      tbl(16);
  tbl.minimum_load_factor(0.6);

  for (size_t i = 0; i < 2 * slot_per_bucket; ++i) {
    tbl.insert(i, i);
  }

  TEST_EXCEPTION(tbl.insert(2 * slot_per_bucket, 0),
                 libcuckoo::load_factor_too_low);
}

void test_minimum_load_factor_invalid() {
  IntIntTable tbl;
  TEST_EXCEPTION(tbl.minimum_load_factor(-0.01), std::invalid_argument);
  TEST_EXCEPTION(tbl.minimum_load_factor(1.01), std::invalid_argument);
}
