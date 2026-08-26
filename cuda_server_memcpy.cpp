// Server-side copy machinery: the staging pools the device copies stage
// through, the lifetime rules that tie those pools to a context, and the
// chunked host-to-device and device-to-host loops. The wire framing and
// compression these drive live in rpc.cpp.
#include <cuda.h>

#include "codegen/gen_rpc_ids.h"
#include "cuda_server.h"
#include "cuda_server_memcpy.h"
#include "rpc.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lupine_log.h"
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *context,
                                           unsigned int flags, CUdevice device);

static constexpr size_t LUPINE_HTOD_CHUNK_BYTES = 64 * 1024 * 1024;
static constexpr size_t LUPINE_SYNC_HTOD_SLOT_BYTES = 8 * 1024 * 1024;
static constexpr size_t LUPINE_SYNC_HTOD_SLOT_COUNT = 2;
static constexpr size_t LUPINE_ASYNC_HTOD_SLOT_COUNT = 4;
static constexpr size_t LUPINE_ASYNC_HTOD_SLOT_BYTES = 8 * 1024 * 1024;
static constexpr auto LUPINE_ASYNC_HTOD_POLL_BUDGET =
    std::chrono::milliseconds(2);
static constexpr size_t LUPINE_STAGING_RETAIN_BYTES = 8 * 1024 * 1024;

static_assert(LUPINE_SYNC_HTOD_SLOT_BYTES % LUPINE_COMPRESS_BLOCK_BYTES == 0,
              "HtoD staging slots must preserve LZ4 block alignment");

struct lupine_staging {
  void *ptr = nullptr;
  bool owned = false; // true => caller must release (a per-call allocation)
  bool pinned = false;
};

struct lupine_retained_staging {
  void *ptr = nullptr;
  size_t size = 0;
  CUcontext allocation_context = nullptr;
};

struct lupine_sync_htod_pool {
  std::array<void *, LUPINE_SYNC_HTOD_SLOT_COUNT> slots = {};
  CUcontext context = nullptr;
  bool disabled = false;
};

// Returns a host buffer of at least `bytes`. On success ptr != nullptr; when
// owned is true the caller must release it via lupine_release_staging, when
// false it borrows the retained buffer and must not free it.
static lupine_staging
lupine_acquire_staging(size_t bytes, lupine_retained_staging &retained) {
  lupine_staging out;
  if (bytes == 0) {
    return out;
  }
#ifdef _WIN32
  (void)retained;
  if (cuMemAllocHost(&out.ptr, bytes) == CUDA_SUCCESS) {
    out.owned = true;
    out.pinned = true;
  } else if ((out.ptr = malloc(bytes)) != nullptr) {
    out.owned = true;
  }
  return out;
#endif
  if (bytes > LUPINE_STAGING_RETAIN_BYTES) {
    if (cuMemAllocHost(&out.ptr, bytes) == CUDA_SUCCESS) {
      out.owned = true;
      out.pinned = true;
    } else if ((out.ptr = malloc(bytes)) != nullptr) {
      out.owned = true;
    }
    return out;
  }
  CUcontext current = nullptr;
  if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current == nullptr) {
    return out;
  }
  if (retained.allocation_context != current && retained.ptr != nullptr) {
    // Synchronous staging is idle when this function returns, so it can be
    // retired immediately before switching its allocation owner.
    CUresult switch_result = cuCtxSetCurrent(retained.allocation_context);
    if (switch_result == CUDA_SUCCESS) {
      cuMemFreeHost(retained.ptr);
      cuCtxSetCurrent(current);
    }
    retained = {};
  }
  if (retained.size < bytes) {
    void *grown = nullptr;
    if (cuMemAllocHost(&grown, bytes) != CUDA_SUCCESS) {
      return out;
    }
    if (retained.ptr != nullptr) {
      cuMemFreeHost(retained.ptr);
    }
    retained.ptr = grown;
    retained.size = bytes;
    retained.allocation_context = current;
  }
  out.ptr = retained.ptr;
  out.pinned = true;
  return out;
}

static void lupine_release_staging(const lupine_staging &s) {
  if (s.ptr != nullptr && s.owned) {
    if (s.pinned) {
      cuMemFreeHost(s.ptr);
    } else {
      free(s.ptr);
    }
  }
}

enum class lupine_async_htod_state { available, in_flight, quarantined };

struct lupine_async_htod_slot {
  void *ptr = nullptr;
  size_t size = 0;
  CUcontext allocation_context = nullptr;
  CUevent completion = nullptr;
  CUcontext event_context = nullptr;
  lupine_async_htod_state state = lupine_async_htod_state::available;
  rpc_http2_window_credit held_window_credit;
};

struct lupine_async_htod_spill {
  void *ptr = nullptr;
  CUcontext allocation_context = nullptr;
  CUevent completion = nullptr;
  CUcontext event_context = nullptr;
  bool completion_recorded = false;
  bool work_queued = false;
  rpc_http2_window_credit held_window_credit;
};

struct lupine_staging_state {
  conn_t *conn = nullptr;
  std::mutex lifecycle_mutex;
  std::mutex mutex;
  std::condition_variable condition;
  bool staging_operation_active = false;
  lupine_retained_staging sync_staging;
  lupine_sync_htod_pool sync_htod;
  std::array<lupine_async_htod_slot, LUPINE_ASYNC_HTOD_SLOT_COUNT> slots;
  std::vector<lupine_async_htod_spill> spills;
  std::unordered_map<CUdevice, CUcontext> primary_contexts;
  std::unordered_set<CUcontext> created_contexts;
  std::unordered_set<CUcontext> teardown_contexts;
  std::unordered_set<CUdevice> teardown_devices;
};

using lupine_staging_registry =
    libcuckoo::cuckoohash_map<conn_t *, std::unique_ptr<lupine_staging_state>>;

static lupine_staging_registry &lupine_staging_states() {
  static auto *registry = new lupine_staging_registry();
  return *registry;
}

static lupine_staging_state *lupine_staging_state_for(conn_t *conn) {
  if (conn == nullptr) {
    return nullptr;
  }
  lupine_staging_state *state = nullptr;
  lupine_staging_states().find_fn(
      conn, [&state](const std::unique_ptr<lupine_staging_state> &stored) {
        state = stored.get();
      });
  return state;
}

// Called with state.mutex held. Runtime-created primary contexts
// normally pass through cuDevicePrimaryCtxRetain, but some clients establish
// them through other entry points. Explicitly created contexts are excluded.
static void lupine_note_inferred_primary_context(lupine_staging_state &state,
                                                 CUcontext context,
                                                 CUdevice device) {
  if (context == nullptr || state.created_contexts.count(context) != 0) {
    return;
  }
  try {
    state.primary_contexts[device] = context;
  } catch (...) {
    LUPINE_LOG_ERROR("Failed to infer primary CUDA context for staging");
  }
}

class lupine_staging_operation {
public:
  lupine_staging_operation(lupine_staging_state *state, CUcontext context,
                           CUdevice device)
      : state_(state) {
    if (state_ == nullptr) {
      return;
    }
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->condition.wait(lock, [&] {
      return !state_->staging_operation_active ||
             state_->teardown_contexts.count(context) != 0 ||
             state_->teardown_devices.count(device) != 0;
    });
    if (state_->teardown_contexts.count(context) != 0 ||
        state_->teardown_devices.count(device) != 0) {
      return;
    }
    state_->staging_operation_active = true;
    acquired_ = true;
    lupine_note_inferred_primary_context(*state_, context, device);
  }

  ~lupine_staging_operation() {
    if (!acquired_) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(state_->mutex);
      state_->staging_operation_active = false;
    }
    state_->condition.notify_all();
  }

  bool acquired() const { return acquired_; }

private:
  lupine_staging_state *state_ = nullptr;
  bool acquired_ = false;
};

class lupine_scoped_context {
public:
  explicit lupine_scoped_context(CUcontext target) {
    status_ = cuCtxGetCurrent(&previous_);
    if (status_ == CUDA_SUCCESS && previous_ != target) {
      status_ = cuCtxSetCurrent(target);
      changed_ = status_ == CUDA_SUCCESS;
    }
  }

  ~lupine_scoped_context() {
    if (changed_) {
      CUresult result = cuCtxSetCurrent(previous_);
      if (result != CUDA_SUCCESS) {
        LUPINE_LOG_ERROR("Failed to restore CUDA context after staging "
                         "cleanup: "
                         << result);
      }
    }
  }

  CUresult status() const { return status_; }

private:
  CUcontext previous_ = nullptr;
  CUresult status_ = CUDA_ERROR_INVALID_CONTEXT;
  bool changed_ = false;
};

static void lupine_sync_htod_forget(lupine_sync_htod_pool &pool) {
  pool.slots = {};
  pool.context = nullptr;
  pool.disabled = false;
}

static void lupine_sync_htod_retire(lupine_sync_htod_pool &pool) {
  if (pool.context == nullptr || pool.disabled) {
    // An error may leave a submitted DMA without a completion event. Let CUDA
    // context teardown own quarantined allocations rather than freeing them.
    lupine_sync_htod_forget(pool);
    return;
  }

  lupine_scoped_context current(pool.context);
  if (current.status() != CUDA_SUCCESS) {
    // Context teardown owns allocations whose CUDA context is already gone.
    lupine_sync_htod_forget(pool);
    return;
  }
  for (void *slot : pool.slots) {
    if (slot != nullptr) {
      (void)cuMemFreeHost(slot);
    }
  }
  lupine_sync_htod_forget(pool);
}

static bool lupine_sync_htod_prepare(lupine_sync_htod_pool &pool,
                                     CUcontext context) {
  if (pool.disabled || context == nullptr) {
    return false;
  }
  if (pool.context != context) {
    lupine_sync_htod_retire(pool);
    pool.context = context;
  }

  for (void *slot : pool.slots) {
    if (slot == nullptr) {
      continue;
    }
    unsigned int flags = 0;
    if (cuMemHostGetFlags(&flags, slot) != CUDA_SUCCESS) {
      // Primary-context reset may recycle the CUcontext handle while
      // invalidating allocations from its previous incarnation.
      lupine_sync_htod_forget(pool);
      pool.context = context;
      break;
    }
  }
  for (void *&slot : pool.slots) {
    if (slot == nullptr &&
        cuMemHostAlloc(&slot, LUPINE_SYNC_HTOD_SLOT_BYTES,
                       CU_MEMHOSTALLOC_PORTABLE) != CUDA_SUCCESS) {
      return false;
    }
  }
  return true;
}

// Credits a staging buffer's payload bytes back to the client's send window.
// Every path that stops tracking a buffer -- retired, quarantined, forgotten --
// runs through here, so an unprovable DMA costs pinned memory but never leaves
// window charged to a buffer nothing will ever retire.
static void lupine_release_staging_window(lupine_staging_state &state,
                                          rpc_http2_window_credit &held) {
  rpc_http2_window_credit credit = held;
  held = {};
  rpc_http2_window_release(state.conn, credit);
}

// Reads a payload into staging with the connection's window held, charging the
// received bytes to the staging buffer rather than crediting them immediately.
static int lupine_read_staged_payload(conn_t *conn, void *host, size_t bytes,
                                      rpc_http2_window_credit &held) {
  rpc_http2_window_hold_begin(conn);
  int result = rpc_read_payload_part(conn, host, bytes);
  rpc_http2_window_credit credit = rpc_http2_window_hold_end(conn);
  if (held.bytes != 0 && held.stream_id != credit.stream_id) {
    rpc_http2_window_release(conn, credit);
    return -1;
  }
  held.stream_id = credit.stream_id;
  held.bytes += credit.bytes;
  return result;
}

static CUresult lupine_async_htod_destroy_event(CUevent *event,
                                                CUcontext context) {
  if (event == nullptr || *event == nullptr) {
    return CUDA_SUCCESS;
  }
  lupine_scoped_context current(context);
  CUresult result = current.status();
  if (result == CUDA_SUCCESS) {
    result = cuEventDestroy(*event);
  }
  if (result != CUDA_SUCCESS) {
    LUPINE_LOG_ERROR(
        "Failed to destroy async HtoD completion event: " << result);
  }
  *event = nullptr;
  return result;
}

static CUresult lupine_async_htod_free_host(void **ptr, CUcontext context) {
  if (ptr == nullptr || *ptr == nullptr) {
    return CUDA_SUCCESS;
  }
  lupine_scoped_context current(context);
  CUresult result = current.status();
  if (result == CUDA_SUCCESS) {
    result = cuMemFreeHost(*ptr);
  }
  if (result != CUDA_SUCCESS) {
    LUPINE_LOG_ERROR("Failed to free async HtoD pinned staging: " << result);
  }
  *ptr = nullptr;
  return result;
}

static CUresult lupine_async_htod_query(CUevent event, CUcontext context) {
  if (event == nullptr) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  lupine_scoped_context current(context);
  if (current.status() != CUDA_SUCCESS) {
    return current.status();
  }
  return cuEventQuery(event);
}

static void lupine_async_htod_reclaim(lupine_staging_state &state,
                                      CUcontext context) {
  for (auto &slot : state.slots) {
    if (slot.state != lupine_async_htod_state::in_flight ||
        slot.event_context != context || slot.completion == nullptr) {
      continue;
    }
    CUresult result = cuEventQuery(slot.completion);
    if (result == CUDA_SUCCESS) {
      slot.state = lupine_async_htod_state::available;
      lupine_release_staging_window(state, slot.held_window_credit);
    } else if (result != CUDA_ERROR_NOT_READY) {
      // An error cannot prove the DMA is finished. Never make this allocation
      // reusable until its context is explicitly retired.
      slot.state = lupine_async_htod_state::quarantined;
      lupine_release_staging_window(state, slot.held_window_credit);
      LUPINE_LOG_ERROR("Async HtoD slot event query failed: " << result);
    }
  }

  for (auto spill = state.spills.begin(); spill != state.spills.end();) {
    if (!spill->completion_recorded || spill->event_context != context) {
      ++spill;
      continue;
    }
    CUresult result = cuEventQuery(spill->completion);
    if (result == CUDA_ERROR_NOT_READY) {
      ++spill;
      continue;
    }
    if (result != CUDA_SUCCESS) {
      // As with a ring slot, retain the allocation when completion is unknown.
      spill->completion_recorded = false;
      lupine_release_staging_window(state, spill->held_window_credit);
      LUPINE_LOG_ERROR("Async HtoD spill event query failed: " << result);
      ++spill;
      continue;
    }
    lupine_release_staging_window(state, spill->held_window_credit);
    lupine_async_htod_destroy_event(&spill->completion, spill->event_context);
    lupine_async_htod_free_host(&spill->ptr, spill->allocation_context);
    spill = state.spills.erase(spill);
  }
}

static void
lupine_async_htod_reset_available_slot(lupine_async_htod_slot *slot) {
  if (slot == nullptr || slot->state != lupine_async_htod_state::available) {
    return;
  }
  lupine_async_htod_destroy_event(&slot->completion, slot->event_context);
  lupine_async_htod_free_host(&slot->ptr, slot->allocation_context);
  *slot = {};
}

static bool lupine_async_htod_prepare_slot(lupine_staging_state &state,
                                           lupine_async_htod_slot *slot,
                                           CUcontext context,
                                           size_t slot_bytes) {
  if (slot == nullptr || slot->state != lupine_async_htod_state::available) {
    return false;
  }
  if (slot->ptr != nullptr &&
      (slot->allocation_context != context || slot->size < slot_bytes)) {
    lupine_async_htod_reset_available_slot(slot);
  }
  if (slot->ptr == nullptr) {
    CUresult result =
        cuMemHostAlloc(&slot->ptr, slot_bytes, CU_MEMHOSTALLOC_PORTABLE);
    if (result != CUDA_SUCCESS) {
      slot->ptr = nullptr;
      return false;
    }
    slot->size = slot_bytes;
    slot->allocation_context = context;
  }
  if (slot->completion != nullptr && slot->event_context != context) {
    lupine_async_htod_destroy_event(&slot->completion, slot->event_context);
  }
  if (slot->completion == nullptr) {
    CUresult result = cuEventCreate(&slot->completion, CU_EVENT_DISABLE_TIMING);
    if (result != CUDA_SUCCESS) {
      slot->completion = nullptr;
      return false;
    }
    slot->event_context = context;
  }
  return true;
}

static lupine_async_htod_slot *lupine_async_htod_acquire_slot(
    lupine_staging_state &state, CUcontext context, size_t slot_bytes,
    std::chrono::steady_clock::duration *poll_budget) {
  bool polling = false;
  auto poll_started = std::chrono::steady_clock::time_point{};
  for (;;) {
    if (polling) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed = now - poll_started;
      if (elapsed >= *poll_budget) {
        *poll_budget = std::chrono::steady_clock::duration::zero();
        return nullptr;
      }
      *poll_budget -= elapsed;
      poll_started = now;
    }

    lupine_async_htod_reclaim(state, context);

    // Prefer an allocation already owned by this context, then an empty slot,
    // and only then retire an idle allocation left by another live context.
    for (int pass = 0; pass != 3; ++pass) {
      auto &slots = state.slots;
      for (size_t index = 0; index != LUPINE_ASYNC_HTOD_SLOT_COUNT; ++index) {
        auto &slot = slots[index];
        if (slot.state != lupine_async_htod_state::available) {
          continue;
        }
        bool candidate = false;
        if (pass == 0) {
          candidate = slot.ptr != nullptr && slot.allocation_context == context;
        } else if (pass == 1) {
          candidate = slot.ptr == nullptr;
        } else {
          candidate = slot.ptr != nullptr;
        }
        if (candidate &&
            lupine_async_htod_prepare_slot(state, &slot, context, slot_bytes)) {
          return &slot;
        }
      }
    }

    if (*poll_budget <= std::chrono::steady_clock::duration::zero()) {
      return nullptr;
    }
    if (!polling) {
      polling = true;
      poll_started = std::chrono::steady_clock::now();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

static CUresult lupine_async_htod_publish_slot(lupine_staging_state &state,
                                               lupine_async_htod_slot *slot,
                                               CUstream stream) {
  CUresult result = cuEventRecord(slot->completion, stream);
  slot->state = result == CUDA_SUCCESS ? lupine_async_htod_state::in_flight
                                       : lupine_async_htod_state::quarantined;
  if (result != CUDA_SUCCESS) {
    lupine_release_staging_window(state, slot->held_window_credit);
  }
  return result;
}

static CUresult lupine_async_htod_publish_spill(lupine_staging_state &state,
                                                lupine_async_htod_spill &spill,
                                                CUstream stream) {
  CUresult result = cuEventRecord(spill.completion, stream);
  spill.completion_recorded = result == CUDA_SUCCESS;
  if (result != CUDA_SUCCESS) {
    lupine_release_staging_window(state, spill.held_window_credit);
  }
  return result;
}

static void lupine_async_htod_discard_spill(lupine_staging_state &state,
                                            lupine_async_htod_spill &spill) {
  lupine_release_staging_window(state, spill.held_window_credit);
  lupine_async_htod_destroy_event(&spill.completion, spill.event_context);
  lupine_async_htod_free_host(&spill.ptr, spill.allocation_context);
}

static CUresult lupine_async_htod_enqueue_spill(
    lupine_staging_state &state, conn_t *conn, CUdeviceptr destination,
    size_t bytes, CUstream stream, CUcontext context, bool *payload_consumed,
    bool *connection_failed) {
  if (payload_consumed != nullptr) {
    *payload_consumed = false;
  }
  if (connection_failed != nullptr) {
    *connection_failed = false;
  }
  try {
    state.spills.emplace_back();
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  auto &spill = state.spills.back();
  spill.allocation_context = context;
  spill.event_context = context;

  CUresult result = cuMemHostAlloc(&spill.ptr, bytes, CU_MEMHOSTALLOC_PORTABLE);
  if (result == CUDA_SUCCESS) {
    result = cuEventCreate(&spill.completion, CU_EVENT_DISABLE_TIMING);
  }
  if (result != CUDA_SUCCESS) {
    lupine_async_htod_discard_spill(state, spill);
    state.spills.pop_back();
    return result;
  }
  size_t offset = 0;
  while (offset < bytes) {
    size_t chunk = std::min(LUPINE_HTOD_CHUNK_BYTES, bytes - offset);
    auto *chunk_host = static_cast<unsigned char *>(spill.ptr) + offset;
    if (lupine_read_staged_payload(conn, chunk_host, chunk,
                                   spill.held_window_credit) < 0) {
      if (spill.work_queued) {
        (void)lupine_async_htod_publish_spill(state, spill, stream);
      } else {
        lupine_async_htod_discard_spill(state, spill);
        state.spills.pop_back();
      }
      if (connection_failed != nullptr) {
        *connection_failed = true;
      }
      return CUDA_ERROR_UNKNOWN;
    }

    result =
        cuMemcpyHtoDAsync_v2(destination + offset, chunk_host, chunk, stream);
    // A non-success result may report deferred work from an earlier launch;
    // it does not prove this submission was rejected. Publish an event for
    // every attempted copy before deciding whether its staging can be freed.
    spill.work_queued = true;
    offset += chunk;
    if (result != CUDA_SUCCESS) {
      (void)lupine_async_htod_publish_spill(state, spill, stream);
      if (rpc_drain_payload(conn, bytes - offset) < 0) {
        if (connection_failed != nullptr) {
          *connection_failed = true;
        }
      } else if (payload_consumed != nullptr) {
        *payload_consumed = true;
      }
      return result;
    }
  }

  if (payload_consumed != nullptr) {
    *payload_consumed = true;
  }
  return lupine_async_htod_publish_spill(state, spill, stream);
}

static void lupine_async_htod_retire_context(lupine_staging_state &state,
                                             CUcontext context) {
  if (context == nullptr) {
    return;
  }

  for (auto &slot : state.slots) {
    bool allocation_owned = slot.allocation_context == context;
    bool event_owned = slot.event_context == context;
    if (!allocation_owned && !event_owned) {
      continue;
    }
    if (slot.state == lupine_async_htod_state::in_flight) {
      CUresult result =
          lupine_async_htod_query(slot.completion, slot.event_context);
      if (result == CUDA_ERROR_NOT_READY) {
        continue;
      }
      if (result != CUDA_SUCCESS) {
        slot.state = lupine_async_htod_state::quarantined;
        lupine_release_staging_window(state, slot.held_window_credit);
        LUPINE_LOG_ERROR(
            "Could not prove async HtoD slot completion; quarantining it: "
            << result);
        continue;
      }
      slot.state = lupine_async_htod_state::available;
    } else if (slot.state == lupine_async_htod_state::quarantined) {
      continue;
    }
    lupine_release_staging_window(state, slot.held_window_credit);
    lupine_async_htod_destroy_event(&slot.completion, slot.event_context);
    if (allocation_owned) {
      lupine_async_htod_free_host(&slot.ptr, slot.allocation_context);
      slot = {};
    } else {
      slot.event_context = nullptr;
    }
  }

  for (auto spill = state.spills.begin(); spill != state.spills.end();) {
    bool allocation_owned = spill->allocation_context == context;
    bool event_owned = spill->event_context == context;
    if (!allocation_owned && !event_owned) {
      ++spill;
      continue;
    }
    CUresult result = CUDA_SUCCESS;
    if (spill->work_queued) {
      if (spill->completion_recorded) {
        result =
            lupine_async_htod_query(spill->completion, spill->event_context);
      } else {
        ++spill;
        continue;
      }
    }
    if (result == CUDA_ERROR_NOT_READY) {
      ++spill;
      continue;
    }
    if (result != CUDA_SUCCESS) {
      spill->completion_recorded = false;
      lupine_release_staging_window(state, spill->held_window_credit);
      LUPINE_LOG_ERROR(
          "Could not prove async HtoD spill completion; quarantining it: "
          << result);
      ++spill;
      continue;
    }
    lupine_release_staging_window(state, spill->held_window_credit);
    lupine_async_htod_destroy_event(&spill->completion, spill->event_context);
    lupine_async_htod_free_host(&spill->ptr, spill->allocation_context);
    spill = state.spills.erase(spill);
  }

  auto &sync_staging = state.sync_staging;
  if (sync_staging.allocation_context == context) {
    lupine_async_htod_free_host(&sync_staging.ptr,
                                sync_staging.allocation_context);
    sync_staging = {};
  }
  if (state.sync_htod.context == context) {
    lupine_sync_htod_retire(state.sync_htod);
  }
}

// Forget CUDA handles only after CUDA has confirmed that their context was
// destroyed/reset. In particular, this function deliberately does not call
// cuMemFreeHost: an unproven DMA completion must never become a host-memory
// use-after-free. The CUDA context teardown owns those orphaned resources.
static void lupine_async_htod_forget_context(lupine_staging_state &state,
                                             CUcontext context) {
  for (auto &slot : state.slots) {
    if (slot.allocation_context == context || slot.event_context == context) {
      lupine_release_staging_window(state, slot.held_window_credit);
      slot = {};
    }
  }

  state.spills.erase(std::remove_if(state.spills.begin(), state.spills.end(),
                                    [&state, context](auto &spill) {
                                      if (spill.allocation_context != context &&
                                          spill.event_context != context) {
                                        return false;
                                      }
                                      lupine_release_staging_window(
                                          state, spill.held_window_credit);
                                      return true;
                                    }),
                     state.spills.end());

  auto &sync_staging = state.sync_staging;
  if (sync_staging.allocation_context == context) {
    sync_staging = {};
  }
  if (state.sync_htod.context == context) {
    lupine_sync_htod_forget(state.sync_htod);
  }
}

static void lupine_server_forget_context_metadata(lupine_staging_state &state,
                                                  CUcontext context) {
  lupine_async_htod_forget_context(state, context);
  state.created_contexts.erase(context);
  for (auto it = state.primary_contexts.begin();
       it != state.primary_contexts.end();) {
    if (it->second == context) {
      it = state.primary_contexts.erase(it);
    } else {
      ++it;
    }
  }
}

bool lupine_server_initialize_connection(conn_t *conn) {
  if (conn == nullptr) {
    return false;
  }
  std::unique_ptr<lupine_staging_state> state(new (std::nothrow)
                                                  lupine_staging_state());
  if (state == nullptr) {
    return false;
  }
  state->conn = conn;
  return lupine_staging_states().insert(conn, std::move(state));
}

void lupine_server_begin_lifecycle_transaction(conn_t *conn) {
  auto *state = lupine_staging_state_for(conn);
  if (state != nullptr) {
    state->lifecycle_mutex.lock();
  }
}

void lupine_server_end_lifecycle_transaction(conn_t *conn) {
  auto *state = lupine_staging_state_for(conn);
  if (state != nullptr) {
    state->lifecycle_mutex.unlock();
  }
}

void lupine_server_note_primary_context(conn_t *conn, CUdevice device,
                                        CUcontext context, CUresult result) {
  if (result != CUDA_SUCCESS || context == nullptr) {
    return;
  }
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  try {
    state->primary_contexts[device] = context;
  } catch (...) {
    LUPINE_LOG_ERROR("Failed to remember primary CUDA context for staging");
  }
}

void lupine_server_note_created_context(conn_t *conn, CUcontext context,
                                        CUresult result) {
  if (result != CUDA_SUCCESS || context == nullptr) {
    return;
  }
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  try {
    state->created_contexts.insert(context);
  } catch (...) {
    LUPINE_LOG_ERROR("Failed to remember created CUDA context for staging");
  }
}

void lupine_server_prepare_primary_context(conn_t *conn, CUdevice device) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  state->teardown_devices.insert(device);
  state->condition.wait(lock, [&] { return !state->staging_operation_active; });
  auto it = state->primary_contexts.find(device);
  if (it != state->primary_contexts.end()) {
    state->teardown_contexts.insert(it->second);
    lupine_async_htod_retire_context(*state, it->second);
  }
}

void lupine_server_finish_primary_context(conn_t *conn, CUdevice device,
                                          bool reset, CUresult result) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  CUcontext context = nullptr;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto it = state->primary_contexts.find(device);
    if (it != state->primary_contexts.end()) {
      context = it->second;
    }
  }
  unsigned int flags = 0;
  int active = 0;
  CUresult state_result = cuDevicePrimaryCtxGetState(device, &flags, &active);

  std::unique_lock<std::mutex> lock(state->mutex);
  bool forget =
      context != nullptr && (reset || result != CUDA_SUCCESS ||
                             state_result != CUDA_SUCCESS || active == 0);
  if (forget) {
    state->condition.wait(lock,
                          [&] { return !state->staging_operation_active; });
    lupine_server_forget_context_metadata(*state, context);
  }
  state->teardown_devices.erase(device);
  state->teardown_contexts.erase(context);
}

void lupine_server_prepare_context_destroy(conn_t *conn, CUcontext context) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  state->teardown_contexts.insert(context);
  state->condition.wait(lock, [&] { return !state->staging_operation_active; });
  lupine_async_htod_retire_context(*state, context);
}

void lupine_server_finish_context_destroy(conn_t *conn, CUcontext context,
                                          CUresult result) {
  (void)result;
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  state->condition.wait(lock, [&] { return !state->staging_operation_active; });
  // Destructive APIs can return a deferred error after taking effect. Detach
  // every old handle on any result; leaking is safer than stale-handle reuse.
  lupine_server_forget_context_metadata(*state, context);
  state->teardown_contexts.erase(context);
}

void lupine_server_finish_context_detach(conn_t *conn, CUcontext context,
                                         CUresult result) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  unsigned int version = 0;
  CUresult query_result = cuCtxGetApiVersion(context, &version);
  std::unique_lock<std::mutex> lock(state->mutex);
  if (result != CUDA_SUCCESS || query_result != CUDA_SUCCESS) {
    state->condition.wait(lock,
                          [&] { return !state->staging_operation_active; });
    lupine_server_forget_context_metadata(*state, context);
  }
  state->teardown_contexts.erase(context);
}

void lupine_server_cleanup_connection(conn_t *conn) {
  std::unique_ptr<lupine_staging_state> owned_state;
  if (!lupine_staging_states().erase_fn(
          conn, [&owned_state](std::unique_ptr<lupine_staging_state> &state) {
            owned_state = std::move(state);
            return true;
          })) {
    return;
  }
  auto *state = owned_state.get();

  // No lane workers remain. Reclaim only resources whose completion can be
  // proven without blocking; unresolved DMA ownership is deliberately
  // detached and left to CUDA process/context teardown.
  for (auto &slot : state->slots) {
    bool complete = slot.state == lupine_async_htod_state::available;
    if (slot.state == lupine_async_htod_state::in_flight) {
      complete = lupine_async_htod_query(slot.completion, slot.event_context) ==
                 CUDA_SUCCESS;
    }
    if (complete) {
      lupine_async_htod_destroy_event(&slot.completion, slot.event_context);
      lupine_async_htod_free_host(&slot.ptr, slot.allocation_context);
    }
  }

  for (auto &spill : state->spills) {
    bool complete = !spill.work_queued;
    if (spill.completion_recorded) {
      complete = lupine_async_htod_query(spill.completion,
                                         spill.event_context) == CUDA_SUCCESS;
    }
    if (complete) {
      lupine_async_htod_destroy_event(&spill.completion, spill.event_context);
      lupine_async_htod_free_host(&spill.ptr, spill.allocation_context);
    }
  }

  if (state->sync_staging.ptr != nullptr) {
    lupine_async_htod_free_host(&state->sync_staging.ptr,
                                state->sync_staging.allocation_context);
  }
  lupine_sync_htod_retire(state->sync_htod);
}

int lupine_write_lifecycle_response(conn_t *conn, int request_id,
                                    CUresult result) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int lupine_copy_htod_serial(conn_t *conn, CUdeviceptr destination, size_t bytes,
                            lupine_staging_state &state, CUresult *result) {
  size_t chunk_bytes = std::min(LUPINE_HTOD_CHUNK_BYTES, bytes);
  lupine_staging staging =
      lupine_acquire_staging(chunk_bytes, state.sync_staging);
  if (chunk_bytes != 0 && staging.ptr == nullptr) {
    *result = CUDA_ERROR_OUT_OF_MEMORY;
    return rpc_drain_payload(conn, bytes) < 0 ? -1 : 0;
  }

  size_t offset = 0;
  while (*result == CUDA_SUCCESS && offset < bytes) {
    size_t chunk = std::min(chunk_bytes, bytes - offset);
    if (rpc_read_payload_part(conn, staging.ptr, chunk) < 0) {
      lupine_release_staging(staging);
      return -1;
    }
    *result = cuMemcpyHtoD_v2(destination + offset, staging.ptr, chunk);
    offset += chunk;
    if (*result != CUDA_SUCCESS &&
        rpc_drain_payload(conn, bytes - offset) < 0) {
      lupine_release_staging(staging);
      return -1;
    }
  }
  lupine_release_staging(staging);
  return 0;
}

static void lupine_destroy_sync_htod_events(
    std::array<CUevent, LUPINE_SYNC_HTOD_SLOT_COUNT> &events) {
  for (CUevent event : events) {
    if (event != nullptr) {
      (void)cuEventDestroy(event);
    }
  }
}

static CUresult lupine_wait_sync_htod_events(
    const std::array<CUevent, LUPINE_SYNC_HTOD_SLOT_COUNT> &events,
    std::array<bool, LUPINE_SYNC_HTOD_SLOT_COUNT> &in_flight) {
  CUresult result = CUDA_SUCCESS;
  for (size_t index = 0; index < in_flight.size(); ++index) {
    if (!in_flight[index]) {
      continue;
    }
    CUresult wait_result = cuEventSynchronize(events[index]);
    if (result == CUDA_SUCCESS && wait_result != CUDA_SUCCESS) {
      result = wait_result;
    }
    in_flight[index] = false;
  }
  return result;
}

// Returns 1 after consuming the payload, 0 when the serial path should handle
// it, and -1 on a transport failure. The legacy stream preserves synchronous
// memcpy ordering while two pinned slots overlap network receipt with DMA.
static int lupine_copy_htod_pipelined(conn_t *conn, CUdeviceptr destination,
                                      size_t bytes, CUcontext context,
                                      lupine_staging_state &state,
                                      CUresult *result) {
#ifdef _WIN32
  (void)conn;
  (void)destination;
  (void)bytes;
  (void)context;
  (void)state;
  (void)result;
  return 0;
#else
  if (bytes <= LUPINE_SYNC_HTOD_SLOT_BYTES ||
      !lupine_sync_htod_prepare(state.sync_htod, context)) {
    return 0;
  }

  std::array<CUevent, LUPINE_SYNC_HTOD_SLOT_COUNT> events = {};
  for (CUevent &event : events) {
    if (cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
      lupine_destroy_sync_htod_events(events);
      return 0;
    }
  }

  std::array<bool, LUPINE_SYNC_HTOD_SLOT_COUNT> in_flight = {};
  size_t offset = 0;
  size_t slot = 0;
  while (offset < bytes) {
    if (in_flight[slot]) {
      CUresult wait_result = cuEventSynchronize(events[slot]);
      in_flight[slot] = false;
      if (wait_result != CUDA_SUCCESS) {
        *result = wait_result;
        state.sync_htod.disabled = true;
        if (rpc_drain_payload(conn, bytes - offset) < 0) {
          lupine_destroy_sync_htod_events(events);
          return -1;
        }
        (void)lupine_wait_sync_htod_events(events, in_flight);
        lupine_destroy_sync_htod_events(events);
        return 1;
      }
    }

    size_t chunk = std::min(LUPINE_SYNC_HTOD_SLOT_BYTES, bytes - offset);
    void *host = state.sync_htod.slots[slot];
    if (rpc_read_payload_part(conn, host, chunk) < 0) {
      state.sync_htod.disabled = true;
      (void)cuStreamSynchronize(CU_STREAM_LEGACY);
      lupine_destroy_sync_htod_events(events);
      return -1;
    }

    CUresult copy_result = cuMemcpyHtoDAsync_v2(destination + offset, host,
                                                chunk, CU_STREAM_LEGACY);
    offset += chunk;
    if (copy_result != CUDA_SUCCESS) {
      *result = copy_result;
      state.sync_htod.disabled = true;
      if (rpc_drain_payload(conn, bytes - offset) < 0) {
        lupine_destroy_sync_htod_events(events);
        return -1;
      }
      (void)lupine_wait_sync_htod_events(events, in_flight);
      lupine_destroy_sync_htod_events(events);
      return 1;
    }

    CUresult record_result = cuEventRecord(events[slot], CU_STREAM_LEGACY);
    if (record_result != CUDA_SUCCESS) {
      CUresult synchronize_result = cuStreamSynchronize(CU_STREAM_LEGACY);
      if (synchronize_result != CUDA_SUCCESS) {
        *result = synchronize_result;
        state.sync_htod.disabled = true;
        if (rpc_drain_payload(conn, bytes - offset) < 0) {
          lupine_destroy_sync_htod_events(events);
          return -1;
        }
      } else if (lupine_copy_htod_serial(conn, destination + offset,
                                         bytes - offset, state, result) < 0) {
        lupine_destroy_sync_htod_events(events);
        return -1;
      }
      lupine_destroy_sync_htod_events(events);
      return 1;
    }

    in_flight[slot] = true;
    slot = (slot + 1) % LUPINE_SYNC_HTOD_SLOT_COUNT;
  }

  *result = lupine_wait_sync_htod_events(events, in_flight);
  if (*result != CUDA_SUCCESS) {
    state.sync_htod.disabled = true;
  }
  lupine_destroy_sync_htod_events(events);
  return 1;
#endif
}

static int lupine_write_dtoh_chunk_response(conn_t *conn, int request_id,
                                            CUresult result, const void *data,
                                            size_t bytes) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (result == CUDA_SUCCESS && rpc_write_payload(conn, data, bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int lupine_copy_dtoh_serial(conn_t *conn, int request_id,
                                   CUdeviceptr source, size_t bytes,
                                   size_t offset, CUstream stream) {
  if (offset > bytes) {
    return -1;
  }

  size_t staging_size =
      std::min(bytes - offset, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
  std::vector<unsigned char> host;
  try {
    host.resize(staging_size);
  } catch (...) {
    return lupine_write_dtoh_chunk_response(
        conn, request_id, CUDA_ERROR_OUT_OF_MEMORY, nullptr, 0);
  }

  do {
    size_t chunk = std::min(bytes - offset, staging_size);
    void *destination = chunk == 0 ? nullptr : host.data();
    // host is pageable, so this returns only once the chunk has landed,
    // ordered behind whatever was already queued on the caller's stream.
    CUresult result =
        cuMemcpyDtoHAsync_v2(destination, source + offset, chunk, stream);
    if (lupine_write_dtoh_chunk_response(conn, request_id, result, host.data(),
                                         chunk) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      return 0;
    }
    offset += chunk;
  } while (offset < bytes);
  return 0;
}

static constexpr size_t LUPINE_DTOH_PIPELINE_SLOT_BYTES =
    LUPINE_COMPRESS_BLOCK_BYTES;
static constexpr size_t LUPINE_DTOH_PIPELINE_SLOT_COUNT = 2;
// Setup costs dominate small transfers, so reserve the pipeline for copies
// large enough to amortize page-locking two slots and creating their events.
static constexpr size_t LUPINE_DTOH_PIPELINE_MIN_BYTES =
    8 * LUPINE_DTOH_PIPELINE_SLOT_BYTES;

struct lupine_dtoh_pipeline_slot {
  unsigned char *data = nullptr;
  CUevent completion = nullptr;
  size_t offset = 0;
  size_t bytes = 0;
  bool in_flight = false;
  bool event_recorded = false;
};

struct lupine_dtoh_pipeline {
  void *storage = nullptr;
  std::array<lupine_dtoh_pipeline_slot, LUPINE_DTOH_PIPELINE_SLOT_COUNT> slots;
};

static CUresult lupine_cleanup_dtoh_pipeline(lupine_dtoh_pipeline &pipeline,
                                             CUstream stream,
                                             bool synchronize_stream) {
  CUresult completion_result = CUDA_SUCCESS;
  bool completion_confirmed = true;
  for (const auto &slot : pipeline.slots) {
    if (slot.in_flight && !slot.event_recorded) {
      synchronize_stream = true;
    }
  }
  if (synchronize_stream) {
    completion_result = cuStreamSynchronize(stream);
    completion_confirmed = completion_result == CUDA_SUCCESS;
  } else {
    for (const auto &slot : pipeline.slots) {
      if (!slot.in_flight || !slot.event_recorded) {
        continue;
      }
      CUresult result = cuEventSynchronize(slot.completion);
      if (result != CUDA_SUCCESS) {
        completion_result = result;
        CUresult stream_result = cuStreamSynchronize(stream);
        completion_confirmed = stream_result == CUDA_SUCCESS;
        break;
      }
    }
  }
  for (auto &slot : pipeline.slots) {
    if (slot.completion != nullptr) {
      (void)cuEventDestroy(slot.completion);
      slot.completion = nullptr;
    }
  }
  if (pipeline.storage != nullptr && completion_confirmed) {
    (void)cuMemFreeHost(pipeline.storage);
  }
  // If completion cannot be established, quarantine the pinned allocation
  // until CUDA tears down the context rather than risk a DMA use-after-free.
  pipeline.storage = nullptr;
  return completion_result;
}

// Returns 0 after writing the complete response, -1 on a transport error, and
// 1 when setup failed before consuming fallback_offset and serial should
// resume.
static int lupine_copy_dtoh_pipelined(conn_t *conn, int request_id,
                                      CUdeviceptr source, size_t bytes,
                                      size_t *fallback_offset,
                                      CUstream stream) {
  lupine_dtoh_pipeline pipeline;
  if (cuMemAllocHost(&pipeline.storage, LUPINE_DTOH_PIPELINE_SLOT_COUNT *
                                            LUPINE_DTOH_PIPELINE_SLOT_BYTES) !=
      CUDA_SUCCESS) {
    *fallback_offset = 0;
    return 1;
  }

  auto *storage = static_cast<unsigned char *>(pipeline.storage);
  for (size_t index = 0; index < pipeline.slots.size(); ++index) {
    auto &slot = pipeline.slots[index];
    slot.data = storage + index * LUPINE_DTOH_PIPELINE_SLOT_BYTES;
    if (cuEventCreate(&slot.completion, CU_EVENT_DISABLE_TIMING) !=
        CUDA_SUCCESS) {
      (void)lupine_cleanup_dtoh_pipeline(pipeline, stream, false);
      *fallback_offset = 0;
      return 1;
    }
  }

  bool event_record_failed = false;
  auto submit = [&](lupine_dtoh_pipeline_slot &slot,
                    size_t offset) -> CUresult {
    slot.offset = offset;
    slot.bytes = std::min(LUPINE_DTOH_PIPELINE_SLOT_BYTES, bytes - offset);
    // CUDA may report an error from earlier asynchronous work without proving
    // that this DMA was rejected. Treat every attempted submission as
    // unproven until an event or stream synchronization establishes completion.
    slot.in_flight = true;
    slot.event_recorded = false;
    CUresult result =
        cuMemcpyDtoHAsync_v2(slot.data, source + offset, slot.bytes, stream);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (cuEventRecord(slot.completion, stream) != CUDA_SUCCESS) {
      event_record_failed = true;
    } else {
      slot.event_recorded = true;
    }
    return CUDA_SUCCESS;
  };

  size_t sent_offset = 0;
  size_t submitted_offset = 0;
  bool terminal_error_pending = false;
  size_t terminal_error_offset = 0;
  CUresult terminal_error = CUDA_SUCCESS;

  for (auto &slot : pipeline.slots) {
    if (submitted_offset == bytes) {
      break;
    }
    CUresult result = submit(slot, submitted_offset);
    if (event_record_failed) {
      CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, stream, true);
      if (cleanup != CUDA_SUCCESS) {
        return lupine_write_dtoh_chunk_response(conn, request_id, cleanup,
                                                nullptr, 0);
      }
      *fallback_offset = sent_offset;
      return 1;
    }
    if (result != CUDA_SUCCESS) {
      terminal_error_pending = true;
      terminal_error_offset = submitted_offset;
      terminal_error = result;
      break;
    }
    submitted_offset += slot.bytes;
  }

  while (sent_offset < bytes) {
    if (terminal_error_pending && terminal_error_offset == sent_offset) {
      int write_result = lupine_write_dtoh_chunk_response(
          conn, request_id, terminal_error, nullptr, 0);
      (void)lupine_cleanup_dtoh_pipeline(pipeline, stream, false);
      return write_result;
    }

    size_t index = (sent_offset / LUPINE_DTOH_PIPELINE_SLOT_BYTES) %
                   LUPINE_DTOH_PIPELINE_SLOT_COUNT;
    auto &slot = pipeline.slots[index];
    if (!slot.in_flight || !slot.event_recorded || slot.offset != sent_offset) {
      CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, stream, true);
      if (cleanup != CUDA_SUCCESS) {
        return lupine_write_dtoh_chunk_response(conn, request_id, cleanup,
                                                nullptr, 0);
      }
      *fallback_offset = sent_offset;
      return 1;
    }

    CUresult result = cuEventSynchronize(slot.completion);
    if (result != CUDA_SUCCESS) {
      int write_result = lupine_write_dtoh_chunk_response(conn, request_id,
                                                          result, nullptr, 0);
      (void)lupine_cleanup_dtoh_pipeline(pipeline, stream, true);
      return write_result;
    }
    if (lupine_write_dtoh_chunk_response(conn, request_id, CUDA_SUCCESS,
                                         slot.data, slot.bytes) < 0) {
      (void)lupine_cleanup_dtoh_pipeline(pipeline, stream, false);
      return -1;
    }

    sent_offset += slot.bytes;
    slot.in_flight = false;
    slot.event_recorded = false;
    if (submitted_offset < bytes && !terminal_error_pending) {
      result = submit(slot, submitted_offset);
      if (event_record_failed) {
        CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, stream, true);
        if (cleanup != CUDA_SUCCESS) {
          return lupine_write_dtoh_chunk_response(conn, request_id, cleanup,
                                                  nullptr, 0);
        }
        *fallback_offset = sent_offset;
        return 1;
      }
      if (result != CUDA_SUCCESS) {
        terminal_error_pending = true;
        terminal_error_offset = submitted_offset;
        terminal_error = result;
      } else {
        submitted_offset += slot.bytes;
      }
    }
  }

  (void)lupine_cleanup_dtoh_pipeline(pipeline, stream, false);
  return 0;
}

int handle_cuMemcpyDtoH_v2(conn_t *conn) {
  CUdeviceptr source = 0;
  size_t bytes = 0;
  // The stream the caller's copy was queued on. A synchronous cuMemcpyDtoH
  // sends the legacy stream; an async copy into pageable memory sends its own,
  // so the chunks stay ordered behind that stream's prior work.
  CUstream stream = CU_STREAM_LEGACY;
  if (rpc_read(conn, &source, sizeof(source)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t fallback_offset = 0;
  if (bytes >= LUPINE_DTOH_PIPELINE_MIN_BYTES) {
    int result = lupine_copy_dtoh_pipelined(conn, request_id, source, bytes,
                                            &fallback_offset, stream);
    if (result <= 0) {
      return result;
    }
  }
  return lupine_copy_dtoh_serial(conn, request_id, source, bytes,
                                 fallback_offset, stream);
}

int lupine_server_copy_htod_async(conn_t *conn, CUdeviceptr dstDevice,
                                  size_t byteCount, CUstream stream,
                                  CUresult &result) {
  auto *state = lupine_staging_state_for(conn);
  CUcontext context = nullptr;
  CUdevice device = 0;
  result =
      state == nullptr ? CUDA_ERROR_OUT_OF_MEMORY : cuCtxGetCurrent(&context);
  if (result == CUDA_SUCCESS && context == nullptr) {
    result = CUDA_ERROR_INVALID_CONTEXT;
  }
  if (result == CUDA_SUCCESS) {
    result = cuCtxGetDevice(&device);
  }
  lupine_staging_operation operation(result == CUDA_SUCCESS ? state : nullptr,
                                     context, device);
  if (result == CUDA_SUCCESS && !operation.acquired()) {
    result = CUDA_ERROR_INVALID_CONTEXT;
  }

  // A single cumulative budget covers every ring-full wait in this API call.
  // Once it is consumed, the unconsumed suffix is staged in one event-owned
  // spill rather than granting every chunk a fresh polling interval.
  auto poll_budget =
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          LUPINE_ASYNC_HTOD_POLL_BUDGET);
  size_t slot_bytes = LUPINE_ASYNC_HTOD_SLOT_BYTES;
  size_t offset = 0;
  while (result == CUDA_SUCCESS && offset < byteCount) {
    auto *slot = lupine_async_htod_acquire_slot(*state, context, slot_bytes,
                                                &poll_budget);
    if (slot == nullptr) {
      break;
    }
    size_t chunk = std::min(slot->size, byteCount - offset);
    if (lupine_read_staged_payload(conn, slot->ptr, chunk,
                                   slot->held_window_credit) < 0) {
      return -1;
    }

    CUresult copy_result =
        cuMemcpyHtoDAsync_v2(dstDevice + offset, slot->ptr, chunk, stream);
    offset += chunk;
    if (copy_result != CUDA_SUCCESS) {
      (void)lupine_async_htod_publish_slot(*state, slot, stream);
      result = copy_result;
      if (rpc_drain_payload(conn, byteCount - offset) < 0) {
        return -1;
      }
      break;
    }
    result = lupine_async_htod_publish_slot(*state, slot, stream);
    if (result != CUDA_SUCCESS) {
      // The copy was accepted but its completion could not be published.
      // Keep the slot quarantined until context teardown and report the CUDA
      // error without ever synchronizing the caller's stream.
      if (rpc_drain_payload(conn, byteCount - offset) < 0) {
        return -1;
      }
      break;
    }
  }

  if (result == CUDA_SUCCESS && offset < byteCount) {
    bool payload_consumed = false;
    bool connection_failed = false;
    size_t remaining = byteCount - offset;
    result = lupine_async_htod_enqueue_spill(
        *state, conn, dstDevice + offset, remaining, stream, context,
        &payload_consumed, &connection_failed);
    if (connection_failed) {
      return -1;
    }
    if (!payload_consumed && rpc_drain_payload(conn, remaining) < 0) {
      return -1;
    }
  } else if (result != CUDA_SUCCESS && offset == 0 &&
             rpc_drain_payload(conn, byteCount) < 0) {
    return -1;
  }
  return 0;
}

// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
int handle_cuMemcpy3DPeer(conn_t *conn) {
  CUDA_MEMCPY3D_PEER copy = {};
  if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction =
      copy.srcMemoryType == CU_MEMORYTYPE_HOST
          ? lupine_copy_direction::host_to_device
          : (copy.dstMemoryType == CU_MEMORYTYPE_HOST
                 ? lupine_copy_direction::device_to_host
                 : lupine_copy_direction::device_to_device);
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    size_t slice = copy.srcHeight * copy.srcPitch;
    size_t offset =
        copy.srcZ * slice + copy.srcY * copy.srcPitch + copy.srcXInBytes;
    std::vector<unsigned char> host((copy.Depth - 1) * slice +
                                    (copy.Height - 1) * copy.srcPitch + offset +
                                    copy.WidthInBytes);
    copy.srcHost = host.data();
    if (rpc_read_pitched(conn, host.data() + offset, copy.WidthInBytes,
                         copy.Height, copy.srcPitch, copy.Depth, slice) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy3DPeer(&copy);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case lupine_copy_direction::device_to_host: {
    size_t slice = copy.dstHeight * copy.dstPitch;
    size_t offset =
        copy.dstZ * slice + copy.dstY * copy.dstPitch + copy.dstXInBytes;
    std::vector<unsigned char> host((copy.Depth - 1) * slice +
                                    (copy.Height - 1) * copy.dstPitch + offset +
                                    copy.WidthInBytes);
    copy.dstHost = host.data();
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy3DPeer(&copy);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_write_pitched(conn, host.data() + offset, copy.WidthInBytes,
                           copy.Height, copy.dstPitch, copy.Depth,
                           slice) < 0) ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  default: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy3DPeer(&copy);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  }
}

// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
int handle_cuMemcpy3DPeerAsync(conn_t *conn) {
  CUDA_MEMCPY3D_PEER copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction =
      copy.srcMemoryType == CU_MEMORYTYPE_HOST
          ? lupine_copy_direction::host_to_device
          : (copy.dstMemoryType == CU_MEMORYTYPE_HOST
                 ? lupine_copy_direction::device_to_host
                 : lupine_copy_direction::device_to_device);
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    size_t slice = copy.srcHeight * copy.srcPitch;
    size_t offset =
        copy.srcZ * slice + copy.srcY * copy.srcPitch + copy.srcXInBytes;
    std::vector<unsigned char> host((copy.Depth - 1) * slice +
                                    (copy.Height - 1) * copy.srcPitch + offset +
                                    copy.WidthInBytes);
    copy.srcHost = host.data();
    if (rpc_read_pitched(conn, host.data() + offset, copy.WidthInBytes,
                         copy.Height, copy.srcPitch, copy.Depth, slice) < 0 ||
        rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy3DPeerAsync(&copy, stream);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case lupine_copy_direction::device_to_host: {
    size_t slice = copy.dstHeight * copy.dstPitch;
    size_t offset =
        copy.dstZ * slice + copy.dstY * copy.dstPitch + copy.dstXInBytes;
    std::vector<unsigned char> host((copy.Depth - 1) * slice +
                                    (copy.Height - 1) * copy.dstPitch + offset +
                                    copy.WidthInBytes);
    copy.dstHost = host.data();
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy3DPeerAsync(&copy, stream);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_write_pitched(conn, host.data() + offset, copy.WidthInBytes,
                           copy.Height, copy.dstPitch, copy.Depth,
                           slice) < 0) ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  default: {
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy3DPeerAsync(&copy, stream);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  }
}

// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
int handle_cuMemcpyAtoH_v2(conn_t *conn) {
  CUarray srcArray = nullptr;
  size_t srcOffset = 0;
  size_t byteCount = 0;
  int request_id = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned char> dstHost;

  if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t staging_size =
      std::min(byteCount, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
  if (staging_size != 0) {
    try {
      dstHost.resize(staging_size);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &result, sizeof(result)) < 0 ||
          rpc_write_end(conn) < 0) {
        return -1;
      }
      return 0;
    }
  }

  size_t offset = 0;
  do {
    size_t chunk = std::min(byteCount - offset, staging_size);
    void *chunk_dst = chunk == 0 ? nullptr : dstHost.data();
    result = cuMemcpyAtoH_v2(chunk_dst, srcArray, srcOffset + offset, chunk);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_write(conn, dstHost.data(), chunk) < 0) ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      return 0;
    }
    offset += chunk;
  } while (offset < byteCount);

  return 0;
}

int handle_cuMemcpyDtoHAsync_v2(conn_t *conn) {
  void *dstHost = nullptr;
  CUdeviceptr srcDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  if (rpc_read_end(conn) < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }

  void *host = nullptr;
  CUresult alloc_result = CUDA_ERROR_INVALID_VALUE;
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto *resources = lupine_get_stream_resources(stream);
    host = lupine_alloc_capture_scratch(resources, byteCount);
    if (host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS) {
        lupine_graph_note_dtoh_copy(resources, dstHost, host, byteCount);
      }
      host = nullptr;
    }
  } else {
    alloc_result = cuMemAllocHost(&host, byteCount);
    if (alloc_result != CUDA_SUCCESS) {
      host = byteCount == 0 ? nullptr : malloc(byteCount);
    }
    if (byteCount != 0 && host == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS && byteCount != 0) {
        lupine_pending_dtoh_item copy{nullptr, dstHost, host, byteCount,
                                      alloc_result == CUDA_SUCCESS};
        lupine_pending_dtoh_copies().upsert(
            conn,
            [stream, &copy](lupine_pending_dtoh_streams &streams,
                            libcuckoo::UpsertContext) {
              streams[stream].push_back(copy);
            },
            lupine_pending_dtoh_streams{});
        host = nullptr;
      }
    }
  }

  // A fire-and-forget copy drops an immediate validation error, matching launch
  // semantics: an execution failure poisons the context and the driver reports
  // it from the client's next synchronize.
  if (alloc_result == CUDA_SUCCESS && host != nullptr) {
    cuMemFreeHost(host);
  } else if (host != nullptr) {
    free(host);
  }
  return 0;
}

// One staging ring per lane. A lane serves its requests one at a time, so a
// thread-local ring is never contended and needs no lock, and the buffers are
// page-locked once for the lane's life instead of on every copy -- allocating
// them per call cost more than the overlap they buy.
//
// The ring is tied to the context it was allocated under. If the lane moves to
// a different context the old buffers are abandoned rather than freed, because
// a context teardown has already reclaimed them and freeing again would be a
// use-after-free.
namespace {
constexpr size_t kLupineCopySlots = 3;

struct lupine_copy_ring {
  void *host[kLupineCopySlots] = {};
  CUevent event[kLupineCopySlots] = {};
  size_t capacity = 0;
  CUcontext context = nullptr;
};

lupine_copy_ring &lupine_lane_copy_ring() {
  static thread_local lupine_copy_ring ring;
  return ring;
}

// Ensures the lane's ring can stage `bytes` of pinned memory per slot.
bool lupine_copy_ring_reserve(lupine_copy_ring &ring, size_t bytes) {
  CUcontext current = nullptr;
  cuCtxGetCurrent(&current);
  if (ring.context != current) {
    // Abandon rather than free: the previous context owns those pages.
    for (size_t i = 0; i < kLupineCopySlots; ++i) {
      ring.host[i] = nullptr;
      ring.event[i] = nullptr;
    }
    ring.capacity = 0;
    ring.context = current;
  }
  if (ring.capacity >= bytes) {
    return true;
  }
  for (size_t i = 0; i < kLupineCopySlots; ++i) {
    if (ring.host[i] != nullptr) {
      cuMemFreeHost(ring.host[i]);
      ring.host[i] = nullptr;
    }
    if (ring.event[i] != nullptr) {
      cuEventDestroy(ring.event[i]);
      ring.event[i] = nullptr;
    }
  }
  ring.capacity = 0;
  for (size_t i = 0; i < kLupineCopySlots; ++i) {
    if (cuMemAllocHost(&ring.host[i], bytes) != CUDA_SUCCESS ||
        cuEventCreate(&ring.event[i], CU_EVENT_DISABLE_TIMING) !=
            CUDA_SUCCESS) {
      for (size_t j = 0; j <= i; ++j) {
        if (ring.event[j] != nullptr) {
          cuEventDestroy(ring.event[j]);
          ring.event[j] = nullptr;
        }
        if (ring.host[j] != nullptr) {
          cuMemFreeHost(ring.host[j]);
          ring.host[j] = nullptr;
        }
      }
      return false;
    }
  }
  ring.capacity = bytes;
  return true;
}
} // namespace

static int lupine_server_memcpy_dtod(conn_t *conn, const CUDA_MEMCPY3D &copy,
                                     CUstream stream, bool blocking,
                                     bool server_source) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuMemcpy3DAsync_v2(&copy, stream);
  if (server_source && result != CUDA_SUCCESS) {
    return -1;
  }
  if (server_source && blocking) {
    // The caller may refill a synchronous HtoD source as soon as this answer
    // arrives, so its persistent server mirror must no longer be in use.
    result = cuStreamSynchronize(stream);
    if (result != CUDA_SUCCESS) {
      return -1;
    }
  }
  if (server_source && !blocking) {
    return 0;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// Host transfers share bounded staging and band construction; the directional
// entry points below decide whether bytes enter or leave that staging ring.
static int lupine_server_memcpy_host(conn_t *conn, const CUDA_MEMCPY3D &request,
                                     CUstream stream, bool blocking,
                                     CUdeviceptr mirrored_destination) {
  CUDA_MEMCPY3D copy = request;
  const bool host_source = copy.srcMemoryType == CU_MEMORYTYPE_HOST;
  const size_t run = copy.WidthInBytes;
  const size_t rows = copy.Height;
  const size_t slices = copy.Depth;
  const size_t total = run * rows * slices;

  const size_t slot = LUPINE_COMPRESS_BLOCK_BYTES;
  // A run shorter than a slot is carried whole rows at a time; a run longer
  // than a slot is cut inside one row. Either shape is one band.
  auto chunk_bytes = [&](size_t offset) {
    size_t column = offset % run;
    // A band that starts mid-run can only reach that run's end: the staging
    // block boundary does not have to land on a run boundary, and a band that
    // ran past it would address the next row's bytes as if they were this
    // row's.
    if (column != 0 || run > slot) {
      return std::min(slot, run - column);
    }
    size_t rows_left = rows - (offset / run) % rows;
    return std::min(std::min((slot / run) * run, rows_left * run),
                    total - offset);
  };

  // The staging side of a band is packed; the device side keeps the caller's
  // geometry, advanced to this chunk's column, row and slice.
  auto band_for = [&](void *staging, size_t offset, size_t bytes) {
    size_t index = offset / run;
    size_t column = offset % run;
    bool whole_runs = column == 0 && bytes >= run;
    CUDA_MEMCPY3D band = copy;
    band.WidthInBytes = whole_runs ? run : bytes;
    band.Height = whole_runs ? bytes / run : 1;
    band.Depth = 1;
    if (host_source) {
      band.srcHost = staging;
      band.srcPitch = band.WidthInBytes;
      band.srcHeight = band.Height;
      band.srcXInBytes = band.srcY = band.srcZ = 0;
      band.dstXInBytes = copy.dstXInBytes + column;
      band.dstY = copy.dstY + index % rows;
      band.dstZ = copy.dstZ + index / rows;
    } else {
      band.dstHost = staging;
      band.dstPitch = band.WidthInBytes;
      band.dstHeight = band.Height;
      band.dstXInBytes = band.dstY = band.dstZ = 0;
      band.srcXInBytes = copy.srcXInBytes + column;
      band.srcY = copy.srcY + index % rows;
      band.srcZ = copy.srcZ + index / rows;
    }
    return band;
  };

  // Three staging buffers in rotation: while one is being filled from the wire
  // the others are still being DMA'd, so the link and the copy engine overlap.
  // Pinned staging is what makes that possible and also what makes it
  // necessary: a slot cannot be touched again until its event fires.
  constexpr size_t kSlotCount = kLupineCopySlots;
  struct {
    void *host = nullptr;
    CUevent event = nullptr;
    bool in_flight = false;
    size_t offset = 0;
    size_t bytes = 0;
  } slots[kSlotCount];

  const size_t slot_bytes = std::min(total, slot);
  lupine_copy_ring &ring = lupine_lane_copy_ring();
  if (!lupine_copy_ring_reserve(ring, slot_bytes)) {
    return -1;
  }
  for (size_t i = 0; i < kSlotCount; ++i) {
    slots[i].host = ring.host[i];
    slots[i].event = ring.event[i];
  }
  // The ring outlives the call, so only the in-flight copies are settled here.
  auto release_slots = [&]() {
    for (auto &entry : slots) {
      if (entry.in_flight) {
        (void)cuEventSynchronize(entry.event);
        entry.in_flight = false;
      }
    }
  };
  auto reclaim = [&](size_t index) {
    if (!slots[index].in_flight) {
      return CUDA_SUCCESS;
    }
    CUresult waited = cuEventSynchronize(slots[index].event);
    slots[index].in_flight = false;
    return waited;
  };
  // A slot holds one compression block, which the run-boundary rule may cut
  // into several bands: a slice smaller than a block gives one band per slice.
  // They are all issued from the one staging buffer before its event is
  // recorded, so the slot stays owned until the last of them completes.
  auto submit = [&](size_t index, size_t offset, size_t bytes) {
    for (size_t placed = 0; placed < bytes;) {
      size_t span = std::min(chunk_bytes(offset + placed), bytes - placed);
      // A band carries either whole runs or a piece of one. Trimming the
      // partial run off the end keeps that true when the staging block ends
      // mid-run; the remainder opens the next band.
      if (span > run) {
        span -= span % run;
      }
      CUDA_MEMCPY3D band =
          band_for(static_cast<char *>(slots[index].host) + placed,
                   offset + placed, span);
      CUresult issued = cuMemcpy3DAsync_v2(&band, stream);
      if (issued != CUDA_SUCCESS) {
        return issued;
      }
      placed += span;
    }
    CUresult recorded = cuEventRecord(slots[index].event, stream);
    if (recorded == CUDA_SUCCESS) {
      slots[index].in_flight = true;
    }
    return recorded;
  };

  if (host_source) {
    size_t index = 0;
    for (size_t offset = 0; offset < total;) {
      size_t bytes = std::min(slot_bytes, total - offset);
      if (reclaim(index) != CUDA_SUCCESS ||
          rpc_read_payload_part(conn, slots[index].host, bytes) < 0 ||
          submit(index, offset, bytes) != CUDA_SUCCESS) {
        release_slots();
        return -1;
      }
      offset += bytes;
      index = (index + 1) % kSlotCount;
    }
    for (size_t i = 0; i < kSlotCount; ++i) {
      if (reclaim(i) != CUDA_SUCCESS) {
        release_slots();
        return -1;
      }
    }
    release_slots();
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    // A non-blocking upload is answered by the caller's next synchronize, not
    // here, matching the driver: the call has already returned.
    if (blocking == 0) {
      return 0;
    }
    CUresult result = CUDA_SUCCESS;
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }

  // Host destination: copies for the next slots are already running while the
  // current one is written out, and each chunk states its own size so the two
  // ends never have to agree on how the transfer was split.
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    release_slots();
    return -1;
  }
  CUresult result = CUDA_SUCCESS;
  const size_t dst_slice_pitch = copy.dstHeight * copy.dstPitch;
  auto update_mirror = [&](size_t offset, const void *source, size_t bytes) {
    auto *destination = reinterpret_cast<unsigned char *>(mirrored_destination);
    auto *packed = static_cast<const unsigned char *>(source);
    for (size_t filled = 0; filled < bytes;) {
      size_t position = offset + filled;
      size_t row_index = position / run;
      size_t column = position % run;
      size_t span = std::min(run - column, bytes - filled);
      memcpy(destination + (row_index / rows) * dst_slice_pitch +
                 (row_index % rows) * copy.dstPitch + column,
             packed + filled, span);
      filled += span;
    }
  };
  size_t submitted = 0;
  for (size_t i = 0; i < kSlotCount && submitted < total; ++i) {
    size_t bytes = std::min(slot_bytes, total - submitted);
    slots[i].offset = submitted;
    slots[i].bytes = bytes;
    CUresult submitted_result = submit(i, submitted, bytes);
    if (result == CUDA_SUCCESS) {
      result = submitted_result;
    }
    submitted += bytes;
  }
  size_t index = 0;
  for (size_t sent = 0; sent < total;) {
    CUresult reclaimed = reclaim(index);
    if (result == CUDA_SUCCESS) {
      result = reclaimed;
    }
    size_t bytes = slots[index].bytes;
    uint64_t carried = result == CUDA_SUCCESS ? bytes : 0;
    if (carried != 0 && mirrored_destination != 0) {
      update_mirror(slots[index].offset, slots[index].host, bytes);
    }
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write(conn, &carried, sizeof(carried)) < 0 ||
        (carried != 0 &&
         rpc_write_payload(conn, slots[index].host, bytes) < 0) ||
        rpc_write_end(conn) < 0) {
      release_slots();
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      release_slots();
      return 0;
    }
    sent += bytes;
    // The slot is free again only now that its payload has reached the wire.
    if (submitted < total) {
      size_t next = std::min(slot_bytes, total - submitted);
      slots[index].offset = submitted;
      slots[index].bytes = next;
      CUresult submitted_result = submit(index, submitted, next);
      if (result == CUDA_SUCCESS) {
        result = submitted_result;
      }
      submitted += next;
    }
    index = (index + 1) % kSlotCount;
  }
  release_slots();
  return 0;
}

static int lupine_server_memcpy_htod(conn_t *conn, const CUDA_MEMCPY3D &copy,
                                     CUstream stream, bool blocking) {
  return lupine_server_memcpy_host(conn, copy, stream, blocking, 0);
}

static int lupine_server_memcpy_dtoh(conn_t *conn, const CUDA_MEMCPY3D &copy,
                                     CUstream stream, bool blocking,
                                     bool server_destination) {
  CUdeviceptr mirrored_destination = 0;
  if (server_destination &&
      rpc_read(conn, &mirrored_destination, sizeof(mirrored_destination)) < 0) {
    return -1;
  }
  return lupine_server_memcpy_host(conn, copy, stream, blocking,
                                   mirrored_destination);
}

// Every client-side cuMemcpy* arrives as flags, a widened CUDA_MEMCPY3D, and a
// stream. HtoH is completed entirely by the client and never reaches here.
int handle_lupineMemcpy(conn_t *conn) {
  uint8_t wire_flags = 0;
  CUDA_MEMCPY3D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  const bool blocking = (wire_flags & LUPINE_MEMCPY_BLOCKING) != 0;
  const bool server_source = (wire_flags & LUPINE_MEMCPY_SERVER_SOURCE) != 0;
  const bool server_destination =
      (wire_flags & LUPINE_MEMCPY_SERVER_DESTINATION) != 0;
  const bool host_source = copy.srcMemoryType == CU_MEMORYTYPE_HOST;
  const bool host_destination = copy.dstMemoryType == CU_MEMORYTYPE_HOST;
  if (host_source && host_destination) {
    return -1;
  }
  if (server_destination && !host_destination) {
    return -1;
  }
  if (host_source) {
    return lupine_server_memcpy_htod(conn, copy, stream, blocking);
  }
  if (host_destination) {
    return lupine_server_memcpy_dtoh(conn, copy, stream, blocking,
                                     server_destination);
  }
  return lupine_server_memcpy_dtod(conn, copy, stream, blocking, server_source);
}
