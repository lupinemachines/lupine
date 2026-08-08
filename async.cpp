#include "async.h"

#include <cstdlib>
#include <memory>

#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

// Returns true when a fire-and-forget copy of `bytes` should instead request
// a response: large copies, or the unacknowledged window filling up, so the
// server's staging cannot grow unboundedly.
bool lupine_async_ff_wants_response(lupine_conn_async_state *state,
                                    size_t bytes, bool strict) {
  if (strict || state == nullptr || bytes > LUPINE_FF_HTOD_MAX_BYTES) {
    return true;
  }
  unsigned long long outstanding =
      state->ff_outstanding_bytes.fetch_add(bytes, std::memory_order_acq_rel);
  return outstanding + bytes > LUPINE_FF_HTOD_WINDOW_BYTES;
}

void lupine_async_ff_acknowledged(lupine_conn_async_state *state) {
  if (state != nullptr) {
    state->ff_outstanding_bytes.store(0, std::memory_order_release);
  }
}

static bool lupine_async_strict() {
  static const bool strict = getenv("LUPINE_STRICT_SYNC") != nullptr;
  return strict;
}

static libcuckoo::cuckoohash_map<conn_t *,
                                 std::shared_ptr<lupine_conn_async_state>> &
lupine_conn_async_states() {
  static auto *states =
      new libcuckoo::cuckoohash_map<conn_t *,
                                    std::shared_ptr<lupine_conn_async_state>>();
  return *states;
}

static lupine_conn_async_state *lupine_conn_async_state_for(conn_t *conn) {
  if (conn == nullptr) {
    return nullptr;
  }
  std::shared_ptr<lupine_conn_async_state> state;
  if (!lupine_conn_async_states().find(conn, state)) {
    state = std::make_shared<lupine_conn_async_state>();
    lupine_conn_async_states().insert(conn, state);
    lupine_conn_async_states().find(conn, state);
  }
  return state.get();
}

extern "C" int lupine_ff_htod_wants_response(conn_t *conn, size_t bytes) {
  return lupine_async_ff_wants_response(lupine_conn_async_state_for(conn),
                                        bytes, lupine_async_strict())
             ? 1
             : 0;
}

extern "C" void lupine_ff_htod_acknowledged(conn_t *conn) {
  lupine_async_ff_acknowledged(lupine_conn_async_state_for(conn));
}
