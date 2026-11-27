#include "events.h"

#include <algorithm>

lupine_event_table lupine_events;

std::shared_mutex &lupine_event_lifecycle_mutex() {
  static std::shared_mutex mutex;
  return mutex;
}

uint64_t lupine_event_invalidate_completion(lupine_event_table *table,
                                            CUevent event, bool push) {
  uint64_t record = table->record_seq.fetch_add(1) + 1;
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  lupine_event_slot *victim = &slots[0];
  for (uint32_t i = 0; i < kLupineEventSlots; ++i) {
    if (slots[i].event == event) {
      slots[i].record = record;
      slots[i].complete = false;
      slots[i].pushing = push && slots[i].watched;
      slots[i].local_answers = 0;
      return slots[i].pushing ? record : 0;
    }
    if (slots[i].record < victim->record) {
      victim = &slots[i];
    }
  }
  *victim = {event, record, false, false, false, 0};
  return 0;
}

void lupine_event_watch(lupine_event_table *table, CUevent event) {
  std::lock_guard<std::mutex> lock(table->mutex);
  for (auto &slot : table->slots) {
    if (slot.event == event) {
      slot.watched = true;
    }
  }
}

void lupine_event_invalidate_all_completions(lupine_event_table *table) {
  std::lock_guard<std::mutex> lock(table->mutex);
  for (auto &slot : table->slots) {
    if (slot.event != nullptr) {
      // A new stamp also voids an answer already in flight.
      slot.record = table->record_seq.fetch_add(1) + 1;
      slot.complete = false;
      slot.pushing = false;
    }
  }
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
                               uint64_t *record, bool *not_ready) {
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  *record = 0;
  for (uint32_t i = 0; i < kLupineEventSlots; ++i) {
    if (slots[i].event != event) {
      continue;
    }
    *record = slots[i].record;
    if (slots[i].complete) {
      return false;
    }
    if (not_ready == nullptr || !slots[i].pushing ||
        table->observed >= slots[i].record) {
      return true;
    }
    if (slots[i].local_answers == kLupineEventLocalAnswers) {
      slots[i].local_answers = 0;
      return true;
    }
    ++slots[i].local_answers;
    *not_ready = true;
    return false;
  }
  return true;
}

void lupine_event_observe_synchronize(lupine_event_table *table) {
  uint64_t record = table->record_seq.load();
  std::lock_guard<std::mutex> lock(table->mutex);
  table->observed = std::max(table->observed, record);
}

uint32_t lupine_event_collect_query_batch(lupine_event_table *table,
                                          CUevent exclude, conn_t *conn,
                                          lupine_event_conn_resolver resolver,
                                          CUevent *events, uint64_t *records) {
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  uint32_t count = 0;
  for (uint32_t i = 0; i < kLupineEventSlots; ++i) {
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
    for (uint32_t j = 0; j < kLupineEventSlots; ++j) {
      if (slots[j].event == events[i] && slots[j].record == records[i]) {
        slots[j].complete = true;
        table->observed = std::max(table->observed, records[i]);
        break;
      }
    }
  }
}
