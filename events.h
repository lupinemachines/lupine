#pragma once

// Client-side event completion tracking behind the batched cuEventQuery. The
// lupine_event_* functions operate on a bare table and take the
// connection resolver as a parameter, so the batching policy is unit-testable
// without routing; the lupine_note_*/lupine_collect_* wrappers bind the
// process-wide table and the real resolver for the client wrappers.

#include <atomic>
#include <cstdint>
#include <mutex>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include "rpc.h"

// Completion is monotonic between records: once the server reports an event
// complete for a given record, that stays true until the event is recorded
// again. A query therefore also asks about every other event the client has
// outstanding, so a poller walking several events pays one round trip instead
// of one per event. Deferred device-to-host copies only reach the client on a
// query that actually reaches the server, so a locally answered query is legal
// only while no async copy is waiting to be drained.
constexpr uint32_t kLupineEventQueryBatch = LUPINE_EVENT_QUERY_BATCH_MAX;

struct lupine_event_slot {
  CUevent event;
  uint64_t recorded;
  uint64_t completed;
};

struct lupine_event_table {
  std::mutex mutex;
  lupine_event_slot slots[kLupineEventQueryBatch] = {};
  std::atomic<uint64_t> record_seq{0};
  std::atomic<uint64_t> dtoh_issued{0};
  std::atomic<uint64_t> dtoh_drained{0};
};

typedef conn_t *(*lupine_event_conn_resolver)(CUevent event);

void lupine_event_note_recorded(lupine_event_table *table, CUevent event);
uint32_t lupine_event_collect_query_batch(lupine_event_table *table,
                                          CUevent primary, conn_t *conn,
                                          lupine_event_conn_resolver resolver,
                                          CUevent *events, uint64_t *recorded);
void lupine_event_note_query_results(lupine_event_table *table,
                                     const CUevent *events,
                                     const uint64_t *recorded,
                                     const CUresult *results, uint32_t count);
void lupine_event_note_async_dtoh(lupine_event_table *table);
uint64_t lupine_event_dtoh_issued(lupine_event_table *table);
void lupine_event_note_dtoh_drained(lupine_event_table *table,
                                    uint64_t drained);

void lupine_note_event_recorded(CUevent event);
uint32_t lupine_collect_event_query_batch(CUevent primary, conn_t *conn,
                                          CUevent *events, uint64_t *recorded);
void lupine_note_event_query_results(const CUevent *events,
                                     const uint64_t *recorded,
                                     const CUresult *results, uint32_t count);
void lupine_note_async_dtoh_copy();
uint64_t lupine_async_dtoh_issued_count();
void lupine_note_async_dtoh_drained(uint64_t drained);
