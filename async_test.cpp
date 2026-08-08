#include "async.h"

#include <cassert>
#include <iostream>

static CUstream stream_sentinel(uintptr_t value) {
  return reinterpret_cast<CUstream>(value);
}

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

static void test_pending_dtoh() {
  lupine_conn_async_state state;
  assert(lupine_async_can_elide_sync(&state, nullptr, false));
  lupine_async_note_pending_dtoh(&state);
  lupine_async_note_pending_dtoh(&state);
  assert(!lupine_async_can_elide_sync(&state, nullptr, false));
  lupine_async_note_dtoh_delivered(&state, 1);
  assert(!lupine_async_can_elide_sync(&state, nullptr, false));
  lupine_async_note_dtoh_delivered(&state, 1);
  assert(lupine_async_can_elide_sync(&state, nullptr, false));
  // Over-delivery (e.g. graph copies in the same response) floors at zero.
  lupine_async_note_dtoh_delivered(&state, 5);
  assert(lupine_async_can_elide_sync(&state, nullptr, false));
  lupine_async_note_pending_dtoh(&state);
  assert(!lupine_async_can_elide_sync(&state, nullptr, false));
}

static void test_elision_gates() {
  lupine_conn_async_state state;
  // Default-stream sentinels are elidable; named streams are not.
  assert(lupine_async_can_elide_sync(&state, nullptr, false));
  assert(lupine_async_can_elide_sync(&state, stream_sentinel(1), false));
  assert(lupine_async_can_elide_sync(&state, stream_sentinel(2), false));
  assert(!lupine_async_can_elide_sync(&state, stream_sentinel(0x7f00), false));
  // Strict mode and a null state disable elision.
  assert(!lupine_async_can_elide_sync(&state, nullptr, true));
  assert(!lupine_async_can_elide_sync(nullptr, nullptr, false));
  // Host callbacks / graph execs disable elision permanently.
  lupine_async_disable_elision(&state);
  assert(!lupine_async_can_elide_sync(&state, nullptr, false));
}

int main() {
  test_ff_window();
  test_pending_dtoh();
  test_elision_gates();
  std::cout << "PASS: async latency-hiding state" << std::endl;
  return 0;
}
