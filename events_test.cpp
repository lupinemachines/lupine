#include "events.h"

#include <iostream>

// events.cpp's conn-aware wrapper binds the routing resolver; the policy tests
// below inject their own, so a stub satisfies the link without routing.
extern "C" conn_t *lupine_rpc_conn_for_event(CUevent) { return nullptr; }

namespace {

conn_t *const kConnA = reinterpret_cast<conn_t *>(0x1000);
conn_t *const kConnB = reinterpret_cast<conn_t *>(0x2000);

CUevent fake_event(uintptr_t index) {
  return reinterpret_cast<CUevent>(0x100 + index * 8);
}

conn_t *resolve_all_to_a(CUevent) { return kConnA; }

// Sends the last recorded event of a 16-event run to a different connection.
conn_t *resolve_last_to_b(CUevent event) {
  return event == fake_event(15) ? kConnB : kConnA;
}

bool batch_contains(const CUevent *events, uint32_t count, CUevent event) {
  for (uint32_t i = 0; i < count; ++i) {
    if (events[i] == event) {
      return true;
    }
  }
  return false;
}

void complete(lupine_event_table *table, const CUevent *events,
              const uint64_t *recorded, uint32_t count) {
  CUresult results[kLupineEventQueryBatch];
  for (uint32_t i = 0; i < count; ++i) {
    results[i] = CUDA_SUCCESS;
  }
  lupine_event_note_query_results(table, events, recorded, results, count);
}

bool test_completed_event_answers_locally() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  lupine_event_note_recorded(&table, fake_event(0));
  uint32_t count = lupine_event_collect_query_batch(
      &table, fake_event(0), kConnA, resolve_all_to_a, events, recorded);
  if (count != 1 || events[0] != fake_event(0) || recorded[0] != 1) {
    std::cerr << "FAIL: first query did not batch the recorded event\n";
    return false;
  }
  complete(&table, events, recorded, count);
  if (lupine_event_collect_query_batch(&table, fake_event(0), kConnA,
                                       resolve_all_to_a, events,
                                       recorded) != 0) {
    std::cerr << "FAIL: completed event did not answer locally\n";
    return false;
  }
  return true;
}

bool test_rerecord_forces_fresh_batch() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  lupine_event_note_recorded(&table, fake_event(0));
  uint32_t count = lupine_event_collect_query_batch(
      &table, fake_event(0), kConnA, resolve_all_to_a, events, recorded);
  complete(&table, events, recorded, count);
  lupine_event_note_recorded(&table, fake_event(0));
  count = lupine_event_collect_query_batch(&table, fake_event(0), kConnA,
                                           resolve_all_to_a, events, recorded);
  if (count != 1 || recorded[0] != 2) {
    std::cerr << "FAIL: re-recorded event did not raise the record sequence\n";
    return false;
  }
  // A result carrying the stale sequence must not mark the new record done.
  uint64_t stale[kLupineEventQueryBatch] = {1};
  complete(&table, events, stale, 1);
  if (lupine_event_collect_query_batch(&table, fake_event(0), kConnA,
                                       resolve_all_to_a, events,
                                       recorded) == 0) {
    std::cerr << "FAIL: stale completion answered a newer record locally\n";
    return false;
  }
  return true;
}

bool test_eviction_is_oldest_recorded() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  for (uintptr_t i = 0; i < kLupineEventQueryBatch + 1; ++i) {
    lupine_event_note_recorded(&table, fake_event(i));
  }
  uint32_t count = lupine_event_collect_query_batch(
      &table, fake_event(kLupineEventQueryBatch), kConnA, resolve_all_to_a,
      events, recorded);
  if (recorded[0] != kLupineEventQueryBatch + 1) {
    std::cerr << "FAIL: newest event lost its slot\n";
    return false;
  }
  if (batch_contains(events, count, fake_event(0))) {
    std::cerr << "FAIL: oldest recorded event was not evicted\n";
    return false;
  }
  count = lupine_event_collect_query_batch(&table, fake_event(0), kConnA,
                                           resolve_all_to_a, events, recorded);
  if (count == 0 || events[0] != fake_event(0) || recorded[0] != 0) {
    std::cerr << "FAIL: untracked event did not force a remote batch\n";
    return false;
  }
  return true;
}

bool test_batch_caps_and_filters() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  for (uintptr_t i = 0; i < kLupineEventQueryBatch; ++i) {
    lupine_event_note_recorded(&table, fake_event(i));
  }
  CUevent untracked = fake_event(99);
  uint32_t count = lupine_event_collect_query_batch(
      &table, untracked, kConnA, resolve_all_to_a, events, recorded);
  if (count != kLupineEventQueryBatch || events[0] != untracked ||
      recorded[0] != 0) {
    std::cerr << "FAIL: batch was not capped with the primary first\n";
    return false;
  }
  // One event on another connection and one already completed both drop out,
  // so a full table now leaves room under the cap.
  CUevent done[1] = {fake_event(3)};
  uint64_t done_recorded[1] = {4};
  complete(&table, done, done_recorded, 1);
  count = lupine_event_collect_query_batch(&table, untracked, kConnA,
                                           resolve_last_to_b, events, recorded);
  if (count != kLupineEventQueryBatch - 1) {
    std::cerr << "FAIL: completed and foreign-connection events were batched\n";
    return false;
  }
  if (batch_contains(events, count, fake_event(3)) ||
      batch_contains(events, count, fake_event(15))) {
    std::cerr << "FAIL: excluded event appeared in the batch\n";
    return false;
  }
  return true;
}

bool test_pending_dtoh_suppresses_local_answer() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  lupine_event_note_recorded(&table, fake_event(0));
  uint32_t count = lupine_event_collect_query_batch(
      &table, fake_event(0), kConnA, resolve_all_to_a, events, recorded);
  complete(&table, events, recorded, count);
  lupine_event_note_async_dtoh(&table);
  if (lupine_event_collect_query_batch(&table, fake_event(0), kConnA,
                                       resolve_all_to_a, events,
                                       recorded) == 0) {
    std::cerr << "FAIL: pending async copy was answered locally\n";
    return false;
  }
  lupine_event_note_dtoh_drained(&table, lupine_event_dtoh_issued(&table));
  if (lupine_event_collect_query_batch(&table, fake_event(0), kConnA,
                                       resolve_all_to_a, events,
                                       recorded) != 0) {
    std::cerr << "FAIL: drained copy still forced a round trip\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!test_completed_event_answers_locally() ||
      !test_rerecord_forces_fresh_batch() ||
      !test_eviction_is_oldest_recorded() || !test_batch_caps_and_filters() ||
      !test_pending_dtoh_suppresses_local_answer()) {
    return 1;
  }
  std::cout << "event completion tracking tests passed\n";
  return 0;
}
