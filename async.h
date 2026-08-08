#ifndef LUPINE_ASYNC_H
#define LUPINE_ASYNC_H

// Client-side latency-hiding state for fire-and-forget submissions. The
// lupine_async_* functions operate on a bare state record so the policy is
// unit-testable without a connection; the lupine_* wrappers key the same logic
// by conn_t for the client wrappers.

#include <atomic>
#include <cstddef>

#include "cuda_compat.h"
#include "rpc.h"

// ff_outstanding_bytes bounds how far fire-and-forget host-to-device copies
// may run ahead of the server before one copy requests a response again.
struct lupine_conn_async_state {
  std::atomic<unsigned long long> ff_outstanding_bytes{0};
};

bool lupine_async_ff_wants_response(lupine_conn_async_state *state,
                                    size_t bytes, bool strict);
void lupine_async_ff_acknowledged(lupine_conn_async_state *state);

extern "C" int lupine_ff_htod_wants_response(conn_t *conn, size_t bytes);
extern "C" void lupine_ff_htod_acknowledged(conn_t *conn);

#endif
