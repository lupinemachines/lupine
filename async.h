#ifndef LUPINE_ASYNC_H
#define LUPINE_ASYNC_H

// Client-side latency-hiding state for fire-and-forget submissions and
// synchronize elision. The lupine_async_* functions operate on a bare state
// record so the policy is unit-testable without a connection; the lupine_*
// wrappers key the same logic by conn_t for the client wrappers.

#include <atomic>
#include <cstddef>

#include "cuda_compat.h"
#include "rpc.h"

// pending_dtoh counts async DtoH copies whose data the server delivers with a
// later synchronize response; while any are outstanding a stream synchronize
// must go to the server. elision_disabled is set permanently once the
// application uses graph execs or host callbacks, whose host-visible effects
// are delivered at synchronization points the client cannot account for.
// ff_outstanding_bytes bounds how far fire-and-forget host-to-device copies
// may run ahead of the server before one copy requests a response again.
struct lupine_conn_async_state {
  std::atomic<long> pending_dtoh{0};
  std::atomic<bool> elision_disabled{false};
  std::atomic<unsigned long long> ff_outstanding_bytes{0};
};

void lupine_async_note_pending_dtoh(lupine_conn_async_state *state);
void lupine_async_note_dtoh_delivered(lupine_conn_async_state *state,
                                      long count);
void lupine_async_disable_elision(lupine_conn_async_state *state);
bool lupine_async_ff_wants_response(lupine_conn_async_state *state,
                                    size_t bytes, bool strict);
void lupine_async_ff_acknowledged(lupine_conn_async_state *state);
bool lupine_async_can_elide_sync(lupine_conn_async_state *state,
                                 CUstream hStream, bool strict);

extern "C" void lupine_note_pending_dtoh(conn_t *conn);
extern "C" void lupine_note_dtoh_delivered(conn_t *conn, long count);
extern "C" void lupine_disable_sync_elision(conn_t *conn);
extern "C" int lupine_ff_htod_wants_response(conn_t *conn, size_t bytes);
extern "C" void lupine_ff_htod_acknowledged(conn_t *conn);
extern "C" bool lupine_can_elide_stream_sync(conn_t *conn, CUstream hStream);

#endif
