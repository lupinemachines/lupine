#include "lupine_platform.h"

#include <cstdint>
#include <iostream>

namespace {

bool test_normal_multiply_is_unchanged() {
  return lupine_checked_mul_size(4, 16) == 64 &&
         lupine_checked_mul_size(0, 16) == 0 &&
         lupine_checked_mul_size(4, 0) == 0;
}

bool test_overflow_saturates_to_size_max() {
  // The issue's repro: numDependencies * sizeof(CUgraphNode) wraps to 16.
  size_t huge = 0x1000000000000001ULL;
  return lupine_checked_mul_size(huge, 16) == SIZE_MAX &&
         lupine_checked_mul_size(SIZE_MAX, 2) == SIZE_MAX;
}

} // namespace

int main() {
  if (!test_normal_multiply_is_unchanged() ||
      !test_overflow_saturates_to_size_max()) {
    std::cerr << "FAIL\n";
    return 1;
  }
  std::cout << "lupine_checked_mul_size tests passed\n";
  return 0;
}
