#include "events.h"

#include "client_routing.h"

void lupine_event_note_recorded(lupine_event_table *table, CUevent event) {
  uint64_t seq = table->record_seq.fetch_add(1) + 1;
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  lupine_event_slot *victim = &slots[0];
  for (uint32_t i = 0; i < kLupineEventQueryBatch; ++i) {
    if (slots[i].event == event) {
      slots[i].recorded = seq;
      return;
    }
    if (slots[i].recorded < victim->recorded) {
      victim = &slots[i];
    }
  }
  *victim = {event, seq, 0};
}

bool lupine_event_query_needed(lupine_event_table *table, CUevent event,
                               uint64_t *recorded) {
  bool copies_pending = table->dtoh_issued.load(std::memory_order_relaxed) !=
                        table->dtoh_drained.load(std::memory_order_relaxed);
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  *recorded = 0;
  for (uint32_t i = 0; i < kLupineEventQueryBatch; ++i) {
    if (slots[i].event != event) {
      continue;
    }
    *recorded = slots[i].recorded;
    return copies_pending || slots[i].recorded != slots[i].completed;
  }
  return true;
}

uint32_t lupine_event_collect_query_prefetch(
    lupine_event_table *table, CUevent exclude, conn_t *conn,
    lupine_event_conn_resolver resolver, CUevent *events, uint64_t *recorded) {
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  uint32_t count = 0;
  for (uint32_t i = 0; i < kLupineEventQueryBatch; ++i) {
    if (slots[i].event == nullptr || slots[i].event == exclude ||
        slots[i].recorded == slots[i].completed ||
        resolver(slots[i].event) != conn) {
      continue;
    }
    events[count] = slots[i].event;
    recorded[count] = slots[i].recorded;
    if (++count == kLupineEventQueryBatch) {
      break;
    }
  }
  return count;
}

void lupine_event_note_query_results(lupine_event_table *table,
                                     const CUevent *events,
                                     const uint64_t *recorded,
                                     const CUresult *results, uint32_t count) {
  lupine_event_slot *slots = table->slots;
  std::lock_guard<std::mutex> lock(table->mutex);
  for (uint32_t i = 0; i < count; ++i) {
    if (results[i] != CUDA_SUCCESS || recorded[i] == 0) {
      continue;
    }
    for (uint32_t j = 0; j < kLupineEventQueryBatch; ++j) {
      if (slots[j].event == events[i] && slots[j].recorded == recorded[i]) {
        slots[j].completed = recorded[i];
        break;
      }
    }
  }
}

void lupine_event_note_async_dtoh(lupine_event_table *table) {
  table->dtoh_issued.fetch_add(1, std::memory_order_relaxed);
}

uint64_t lupine_event_dtoh_issued(lupine_event_table *table) {
  return table->dtoh_issued.load(std::memory_order_relaxed);
}

void lupine_event_note_dtoh_drained(lupine_event_table *table,
                                    uint64_t drained) {
  table->dtoh_drained.store(drained, std::memory_order_relaxed);
}

static lupine_event_table &lupine_event_table_instance() {
  static lupine_event_table table;
  return table;
}

void lupine_note_event_recorded(CUevent event) {
  lupine_event_note_recorded(&lupine_event_table_instance(), event);
}

bool lupine_event_query_needed(CUevent event, uint64_t *recorded) {
  return lupine_event_query_needed(&lupine_event_table_instance(), event,
                                   recorded);
}

uint32_t lupine_collect_event_query_prefetch(CUevent exclude, conn_t *conn,
                                             CUevent *events,
                                             uint64_t *recorded) {
  return lupine_event_collect_query_prefetch(
      &lupine_event_table_instance(), exclude, conn, lupine_rpc_conn_for_event,
      events, recorded);
}

void lupine_note_event_query_results(const CUevent *events,
                                     const uint64_t *recorded,
                                     const CUresult *results, uint32_t count) {
  lupine_event_note_query_results(&lupine_event_table_instance(), events,
                                  recorded, results, count);
}

void lupine_note_async_dtoh_copy() {
  lupine_event_note_async_dtoh(&lupine_event_table_instance());
}

uint64_t lupine_async_dtoh_issued_count() {
  return lupine_event_dtoh_issued(&lupine_event_table_instance());
}

void lupine_note_async_dtoh_drained(uint64_t drained) {
  lupine_event_note_dtoh_drained(&lupine_event_table_instance(), drained);
}
