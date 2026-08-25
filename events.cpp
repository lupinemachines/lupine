#include "events.h"

#include "client_routing.h"

static lupine_event_table &lupine_event_table_instance() {
  static lupine_event_table table;
  return table;
}

std::shared_mutex &lupine_event_lifecycle_mutex() {
  static std::shared_mutex mutex;
  return mutex;
}

void lupine_event_invalidate_completion(lupine_event_table *table,
                                        CUevent event) {
  uint64_t record = table->record_seq.fetch_add(1) + 1;
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  lupine_event_slot *victim = &slots[0];
  for (uint32_t i = 0; i < kLupineEventQueryBatch; ++i) {
    if (slots[i].event == event) {
      slots[i].record = record;
      slots[i].complete = false;
      return;
    }
    if (slots[i].record < victim->record) {
      victim = &slots[i];
    }
  }
  *victim = {event, record, false};
}

void lupine_event_forget(lupine_event_table *table, CUevent event) {
  std::lock_guard<std::mutex> lock(table->mutex);
  for (auto &slot : table->slots) {
    if (slot.event == event) {
      slot = {};
    }
  }
}

bool lupine_event_query_needed(lupine_event_table *table, CUevent event,
                               uint64_t *record) {
  bool copies_owed =
      table->stale_semaphore.load(std::memory_order_relaxed) != 0;
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  *record = 0;
  for (uint32_t i = 0; i < kLupineEventQueryBatch; ++i) {
    if (slots[i].event != event) {
      continue;
    }
    *record = slots[i].record;
    return copies_owed || !slots[i].complete;
  }
  return true;
}

uint32_t lupine_event_collect_query_batch(lupine_event_table *table,
                                          CUevent exclude, conn_t *conn,
                                          lupine_event_conn_resolver resolver,
                                          CUevent *events, uint64_t *records) {
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  uint32_t count = 0;
  for (uint32_t i = 0; i < kLupineEventQueryBatch; ++i) {
    if (slots[i].event == nullptr || slots[i].event == exclude ||
        slots[i].complete || resolver(slots[i].event) != conn) {
      continue;
    }
    events[count] = slots[i].event;
    records[count] = slots[i].record;
    if (++count == kLupineEventQueryBatch) {
      break;
    }
  }
  return count;
}

void lupine_event_cache_completions(lupine_event_table *table,
                                    const CUevent *events,
                                    const uint64_t *records,
                                    const CUresult *results, uint32_t count) {
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  for (uint32_t i = 0; i < count; ++i) {
    if (results[i] != CUDA_SUCCESS || records[i] == 0) {
      continue;
    }
    for (uint32_t j = 0; j < kLupineEventQueryBatch; ++j) {
      if (slots[j].event == events[i] && slots[j].record == records[i]) {
        slots[j].complete = true;
        break;
      }
    }
  }
}

void lupine_event_stale_semaphore_add(lupine_event_table *table,
                                      int64_t delta) {
  if (delta >= 0) {
    table->stale_semaphore.fetch_add(static_cast<uint64_t>(delta),
                                     std::memory_order_relaxed);
    return;
  }
  // A graph replay hands back host copies this client never counted, so the
  // subtraction saturates at zero rather than wrapping.
  uint64_t delivered = static_cast<uint64_t>(-delta);
  uint64_t owed = table->stale_semaphore.load(std::memory_order_relaxed);
  while (owed != 0 && !table->stale_semaphore.compare_exchange_weak(
                          owed, delivered >= owed ? 0 : owed - delivered,
                          std::memory_order_relaxed)) {
  }
}

void lupine_event_invalidate_completion(CUevent event) {
  lupine_event_invalidate_completion(&lupine_event_table_instance(), event);
}

void lupine_event_forget(CUevent event) {
  lupine_event_forget(&lupine_event_table_instance(), event);
}

bool lupine_event_query_needed(CUevent event, uint64_t *record) {
  return lupine_event_query_needed(&lupine_event_table_instance(), event,
                                   record);
}

uint32_t lupine_event_collect_query_batch(CUevent exclude, conn_t *conn,
                                          CUevent *events, uint64_t *records) {
  return lupine_event_collect_query_batch(
      &lupine_event_table_instance(), exclude, conn, lupine_rpc_conn_for_event,
      events, records);
}

void lupine_event_cache_completions(const CUevent *events,
                                    const uint64_t *records,
                                    const CUresult *results, uint32_t count) {
  lupine_event_cache_completions(&lupine_event_table_instance(), events,
                                 records, results, count);
}

void lupine_event_stale_semaphore_add(int64_t delta) {
  lupine_event_stale_semaphore_add(&lupine_event_table_instance(), delta);
}
