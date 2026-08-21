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
  CUevent event = fake_event(0);
  uint64_t recorded = 0;
  lupine_event_note_recorded(&table, fake_event(0));
  if (!lupine_event_query_needed(&table, event, &recorded) || recorded != 1) {
    std::cerr << "FAIL: first query did not require the recorded event\n";
    return false;
  }
  CUresult result = CUDA_SUCCESS;
  lupine_event_note_query_results(&table, &event, &recorded, &result, 1);
  if (lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: completed event did not answer locally\n";
    return false;
  }
  return true;
}

bool test_rerecord_forces_fresh_query() {
  lupine_event_table table;
  CUevent event = fake_event(0);
  uint64_t recorded = 0;
  lupine_event_note_recorded(&table, event);
  if (!lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: first record did not require a query\n";
    return false;
  }
  CUresult result = CUDA_SUCCESS;
  lupine_event_note_query_results(&table, &event, &recorded, &result, 1);
  lupine_event_note_recorded(&table, event);
  if (!lupine_event_query_needed(&table, event, &recorded) || recorded != 2) {
    std::cerr << "FAIL: re-recorded event did not raise the record sequence\n";
    return false;
  }
  // A result carrying the stale sequence must not mark the new record done.
  uint64_t stale = 1;
  lupine_event_note_query_results(&table, &event, &stale, &result, 1);
  if (!lupine_event_query_needed(&table, event, &recorded)) {
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
  uint64_t newest_recorded = 0;
  if (!lupine_event_query_needed(&table, fake_event(kLupineEventQueryBatch),
                                 &newest_recorded) ||
      newest_recorded != kLupineEventQueryBatch + 1) {
    std::cerr << "FAIL: newest event lost its slot\n";
    return false;
  }
  uint32_t count = lupine_event_collect_query_prefetch(
      &table, fake_event(kLupineEventQueryBatch), kConnA, resolve_all_to_a,
      events, recorded);
  if (batch_contains(events, count, fake_event(0))) {
    std::cerr << "FAIL: oldest recorded event was not evicted\n";
    return false;
  }
  uint64_t untracked_recorded = 1;
  if (!lupine_event_query_needed(&table, fake_event(0), &untracked_recorded) ||
      untracked_recorded != 0) {
    std::cerr << "FAIL: untracked event did not force a remote query\n";
    return false;
  }
  return true;
}

bool test_prefetch_batch_excludes_and_filters() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  for (uintptr_t i = 0; i < kLupineEventQueryBatch; ++i) {
    lupine_event_note_recorded(&table, fake_event(i));
  }
  CUevent untracked = fake_event(99);
  uint32_t count = lupine_event_collect_query_prefetch(
      &table, fake_event(0), kConnA, resolve_all_to_a, events, recorded);
  if (count != kLupineEventQueryBatch - 1 ||
      batch_contains(events, count, fake_event(0))) {
    std::cerr << "FAIL: requested event appeared in the prefetch batch\n";
    return false;
  }
  // One event on another connection and one already completed both drop out.
  CUevent done[1] = {fake_event(3)};
  uint64_t done_recorded[1] = {4};
  complete(&table, done, done_recorded, 1);
  count = lupine_event_collect_query_prefetch(
      &table, untracked, kConnA, resolve_last_to_b, events, recorded);
  if (count != kLupineEventQueryBatch - 2) {
    std::cerr << "FAIL: prefetch did not filter completed and foreign events\n";
    return false;
  }
  if (batch_contains(events, count, fake_event(3)) ||
      batch_contains(events, count, fake_event(15))) {
    std::cerr << "FAIL: excluded event appeared in the batch\n";
    return false;
  }
  return true;
}

bool test_query_results_cache_only_successes() {
  lupine_event_table table;
  CUevent events[2] = {fake_event(0), fake_event(1)};
  uint64_t recorded[2] = {1, 2};
  CUresult results[2] = {CUDA_SUCCESS, CUDA_ERROR_NOT_READY};
  lupine_event_note_recorded(&table, events[0]);
  lupine_event_note_recorded(&table, events[1]);
  lupine_event_note_query_results(&table, events, recorded, results, 2);

  uint64_t query_recorded = 0;
  if (lupine_event_query_needed(&table, events[0], &query_recorded)) {
    std::cerr << "FAIL: successful prefetch result was not cached\n";
    return false;
  }
  if (!lupine_event_query_needed(&table, events[1], &query_recorded)) {
    std::cerr << "FAIL: incomplete prefetch result was cached\n";
    return false;
  }
  return true;
}

bool test_destroyed_event_is_removed_from_prefetch() {
  lupine_event_table table;
  CUevent events[kLupineEventQueryBatch];
  uint64_t recorded[kLupineEventQueryBatch];
  CUevent primary = fake_event(0);
  CUevent destroyed = fake_event(1);
  lupine_event_note_recorded(&table, primary);
  lupine_event_note_recorded(&table, destroyed);
  lupine_event_note_destroyed(&table, destroyed);

  uint32_t count = lupine_event_collect_query_prefetch(
      &table, nullptr, kConnA, resolve_all_to_a, events, recorded);
  if (count != 1 || events[0] != primary ||
      batch_contains(events, count, destroyed)) {
    std::cerr << "FAIL: destroyed event remained in the prefetch batch\n";
    return false;
  }
  return true;
}

bool test_pending_dtoh_suppresses_local_answer() {
  lupine_event_table table;
  CUevent event = fake_event(0);
  uint64_t recorded = 0;
  lupine_event_note_recorded(&table, event);
  if (!lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: first record did not require a query\n";
    return false;
  }
  CUresult result = CUDA_SUCCESS;
  lupine_event_note_query_results(&table, &event, &recorded, &result, 1);
  lupine_event_note_async_dtoh(&table);
  if (!lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: pending async copy was answered locally\n";
    return false;
  }
  lupine_event_note_dtoh_drained(&table, 1);
  if (lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: drained copy still forced a round trip\n";
    return false;
  }
  return true;
}

bool test_partially_drained_dtoh_stays_pending() {
  lupine_event_table table;
  CUevent event = fake_event(0);
  uint64_t recorded = 0;
  CUresult result = CUDA_SUCCESS;
  lupine_event_note_recorded(&table, event);
  if (!lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: first record did not require a query\n";
    return false;
  }
  lupine_event_note_query_results(&table, &event, &recorded, &result, 1);
  lupine_event_note_async_dtoh(&table);
  lupine_event_note_async_dtoh(&table);
  lupine_event_note_dtoh_drained(&table, 1);
  if (!lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: partially drained copies were treated as complete\n";
    return false;
  }
  lupine_event_note_dtoh_drained(&table, 1);
  if (lupine_event_query_needed(&table, event, &recorded)) {
    std::cerr << "FAIL: fully drained copies still forced a round trip\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!test_completed_event_answers_locally() ||
      !test_rerecord_forces_fresh_query() ||
      !test_eviction_is_oldest_recorded() ||
      !test_prefetch_batch_excludes_and_filters() ||
      !test_query_results_cache_only_successes() ||
      !test_destroyed_event_is_removed_from_prefetch() ||
      !test_pending_dtoh_suppresses_local_answer() ||
      !test_partially_drained_dtoh_stays_pending()) {
    return 1;
  }
  std::cout << "event completion tracking tests passed\n";
  return 0;
}
