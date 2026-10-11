#include "internal_cuda_query.h"

#include <cstdlib>
#include <iostream>
#include <thread>

static void check(bool passed, const char *message) {
  if (!passed) {
    std::cerr << message << '\n';
    std::abort();
  }
}

int main() {
  check(!lupine_is_internal_cuda_query(), "application calls remain visible");
  try {
    lupine_internal_cuda_query outer;
    check(lupine_is_internal_cuda_query(), "internal query is marked");
    std::thread([] {
      check(!lupine_is_internal_cuda_query(),
            "other application threads are visible");
      lupine_internal_cuda_query inner;
      check(lupine_is_internal_cuda_query(),
            "other internal queries are marked");
    }).join();
    {
      lupine_internal_cuda_query inner;
      check(lupine_is_internal_cuda_query(), "nested query is marked");
    }
    check(lupine_is_internal_cuda_query(), "inner scope preserves outer scope");
    throw 1;
  } catch (int) {
    check(!lupine_is_internal_cuda_query(),
          "unwinding restores application logs");
  }
}
