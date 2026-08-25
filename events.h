#pragma once

// Client-side cache of event completion, and the policy for when cuEventQuery
// may be answered from it instead of costing a round trip.
//
// Two things are tracked. Per event: the stamp of its current record, and
// whether that record is known complete. Completion is monotonic between
// records, so once the server reports an event complete the client can answer
// later queries for that same record itself. Process-wide: how many deferred
// device-to-host copies the server still owes, because those ride back on
// responses and a locally answered query never collects one.
//
// Every operation has a form taking an explicit table so the policy is
// testable without routing; the short forms bind the process-wide table.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include "rpc.h"

constexpr uint32_t kLupineEventQueryBatch = 16;

struct lupine_event_slot {
  CUevent event;
  // Monotonic stamp for this event's current record. It is also the eviction
  // key, and the token that ties an in-flight query to the record it asked
  // about: a re-record while that query travels bumps the stamp, so the answer
  // no longer matches and cannot be cached against the newer record.
  uint64_t record;
  bool complete;
};

struct lupine_event_table {
  std::mutex mutex;
  lupine_event_slot slots[kLupineEventQueryBatch] = {};
  std::atomic<uint64_t> record_seq{0};
  // Held by every deferred device-to-host copy the server has not handed back
  // yet. While it is nonzero every cached completion is unusable, because a
  // query answered locally is a query that never collects those copies. It
  // counts rather than flags: a response carries only the copies for the
  // stream it synchronized, so "some arrived" never means "none are left".
  std::atomic<uint64_t> stale_semaphore{0};
};

typedef conn_t *(*lupine_event_conn_resolver)(CUevent event);

// A fresh record voids the cached completion, and starts tracking the event if
// it was not tracked before.
void lupine_event_invalidate_completion(lupine_event_table *table,
                                        CUevent event);
void lupine_event_invalidate_completion(CUevent event);

// Drops the event, so a handle the driver reuses cannot inherit its answer.
void lupine_event_forget(lupine_event_table *table, CUevent event);
void lupine_event_forget(CUevent event);

// True when the query has to reach the server. Writes the record stamp to hand
// back to lupine_event_cache_completions, or zero if the event is not tracked.
bool lupine_event_query_needed(lupine_event_table *table, CUevent event,
                               uint64_t *record);
bool lupine_event_query_needed(CUevent event, uint64_t *record);

// Fills events/records with other outstanding events on the same connection,
// to warm the cache alongside a query that is already going out.
uint32_t lupine_event_collect_query_batch(lupine_event_table *table,
                                          CUevent exclude, conn_t *conn,
                                          lupine_event_conn_resolver resolver,
                                          CUevent *events, uint64_t *records);
uint32_t lupine_event_collect_query_batch(CUevent exclude, conn_t *conn,
                                          CUevent *events, uint64_t *records);

// Caches the successful answers whose record stamp still matches.
void lupine_event_cache_completions(lupine_event_table *table,
                                    const CUevent *events,
                                    const uint64_t *records,
                                    const CUresult *results, uint32_t count);
void lupine_event_cache_completions(const CUevent *events,
                                    const uint64_t *records,
                                    const CUresult *results, uint32_t count);

// delta is +1 when a copy is deferred and -N when N of them arrive.
void lupine_event_stale_semaphore_add(lupine_event_table *table, int64_t delta);
void lupine_event_stale_semaphore_add(int64_t delta);

// Held shared across a query and exclusively across a destroy, so an event
// cannot be destroyed out from under a query in flight.
std::shared_mutex &lupine_event_lifecycle_mutex();
