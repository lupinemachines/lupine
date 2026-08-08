#include "async.h"

#include <cstdlib>
#include <memory>

#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

void lupine_async_note_pending_dtoh(lupine_conn_async_state *state) {
  if (state != nullptr) {
    state->pending_dtoh.fetch_add(1, std::memory_order_acq_rel);
  }
}

void lupine_async_note_dtoh_delivered(lupine_conn_async_state *state,
                                      long count) {
  if (state == nullptr || count <= 0) {
    return;
  }
  long previous =
      state->pending_dtoh.fetch_sub(count, std::memory_order_acq_rel);
  if (previous < count) {
    state->pending_dtoh.store(0, std::memory_order_release);
  }
}

void lupine_async_disable_elision(lupine_conn_async_state *state) {
  if (state != nullptr) {
    state->elision_disabled.store(true, std::memory_order_release);
  }
}

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

// Only the default-stream sentinels are elidable: a synchronous fetch of
// mapped memory is ordered after legacy-stream work, so elision stays
// coherent there; named streams keep the full round trip.
bool lupine_async_can_elide_sync(lupine_conn_async_state *state,
                                 CUstream hStream, bool strict) {
  if (strict || state == nullptr) {
    return false;
  }
  if (hStream != nullptr &&
      hStream != reinterpret_cast<CUstream>(uintptr_t{1}) &&
      hStream != reinterpret_cast<CUstream>(uintptr_t{2})) {
    return false;
  }
  return !state->elision_disabled.load(std::memory_order_acquire) &&
         state->pending_dtoh.load(std::memory_order_acquire) == 0;
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

extern "C" void lupine_note_pending_dtoh(conn_t *conn) {
  lupine_async_note_pending_dtoh(lupine_conn_async_state_for(conn));
}

extern "C" void lupine_note_dtoh_delivered(conn_t *conn, long count) {
  lupine_async_note_dtoh_delivered(lupine_conn_async_state_for(conn), count);
}

extern "C" void lupine_disable_sync_elision(conn_t *conn) {
  lupine_async_disable_elision(lupine_conn_async_state_for(conn));
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

extern "C" bool lupine_can_elide_stream_sync(conn_t *conn, CUstream hStream) {
  return lupine_async_can_elide_sync(lupine_conn_async_state_for(conn),
                                     hStream, lupine_async_strict());
}
