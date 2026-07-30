#include "unit_test_util.hh"

std::atomic<int64_t> &get_unfreed_bytes() {
  static std::atomic<int64_t> unfreed_bytes(0L);
  return unfreed_bytes;
}
