#include "async.h"

#include <cassert>
#include <iostream>

static void test_ff_window() {
  lupine_conn_async_state state;
  // Strict mode always wants a response.
  assert(lupine_async_ff_wants_response(&state, 16, true));
  // Oversized copies always want a response and do not charge the window.
  assert(lupine_async_ff_wants_response(&state, LUPINE_FF_HTOD_MAX_BYTES + 1,
                                        false));
  // Small copies are fire-and-forget until the window fills.
  size_t chunk = LUPINE_FF_HTOD_MAX_BYTES;
  size_t sent = 0;
  while (sent + chunk <= LUPINE_FF_HTOD_WINDOW_BYTES) {
    assert(!lupine_async_ff_wants_response(&state, chunk, false));
    sent += chunk;
  }
  assert(lupine_async_ff_wants_response(&state, chunk, false));
  // A response resets the window.
  lupine_async_ff_acknowledged(&state);
  assert(!lupine_async_ff_wants_response(&state, chunk, false));
  // A null state degrades to always-blocking.
  assert(lupine_async_ff_wants_response(nullptr, 16, false));
}

int main() {
  test_ff_window();
  std::cout << "PASS: async latency-hiding state" << std::endl;
  return 0;
}
