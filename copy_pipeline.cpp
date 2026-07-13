#include <algorithm>
#include <cstddef>
#include <cuda.h>

#include "codegen/gen_api.h"
#include "copy_pipeline.h"
#include "rpc.h"

#ifdef LUPINE_RPC_CLIENT

#include "client_routing.h"

extern "C" void lupine_prepare_host_range_write(void *host, size_t size);
extern "C" void lupine_mark_host_range_clean(void *host, size_t size);

extern "C" CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice,
                                    size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(srcDevice);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  using real_fn_t = CUresult (*)(void *, CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemcpyDtoH_v2",
                                                  &return_value, dstHost,
                                                  srcDevice, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoH_v2) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  int request_id = rpc_write_end(conn);
  if (request_id < 0 || rpc_read_start(conn, request_id) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }

  lupine_prepare_host_range_write(dstHost, ByteCount);
  auto *copy_dst = static_cast<unsigned char *>(dstHost);
  size_t offset = 0;
  do {
    size_t chunk =
        std::min(ByteCount - offset, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
    if (rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
        (return_value == CUDA_SUCCESS && chunk != 0 &&
         rpc_read_payload(conn, copy_dst + offset, chunk) < 0)) {
      rpc_read_end(conn);
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    bool final_chunk =
        return_value != CUDA_SUCCESS || offset + chunk == ByteCount;
    if (rpc_read_end(conn) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    if (return_value != CUDA_SUCCESS) {
      return return_value;
    }
    offset += chunk;
    if (!final_chunk && rpc_read_start(conn, request_id) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
  } while (offset < ByteCount);
  lupine_mark_host_range_clean(dstHost, ByteCount);
  return return_value;
}

#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
extern "C" CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                         const void *srcHost, size_t ByteCount,
                                         CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  if (lupine_route_is_local(route)) {
    using real_fn_t = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
    auto real = lupine_real_cuda_fn<real_fn_t>("cuMemcpyHtoDAsync_v2");
    return real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE
                           : real(dstDevice, srcHost, ByteCount, hStream);
  }
  if (ByteCount != 0 && srcHost == nullptr) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value = CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyHtoDAsync_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(ByteCount)) < 0 ||
      rpc_write(conn, &hStream, sizeof(hStream)) < 0 ||
      (ByteCount != 0 && rpc_write_payload(conn, srcHost, ByteCount) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(return_value)) < 0 ||
      rpc_read_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return return_value;
}

#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

#elif defined(LUPINE_RPC_SERVER)

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
};

struct lupine_async_htod_spill {
  void *ptr = nullptr;
  CUcontext allocation_context = nullptr;
  CUevent completion = nullptr;
  CUcontext event_context = nullptr;
  bool completion_recorded = false;
  bool work_queued = false;
};

struct lupine_staging_state {
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

struct lupine_staging_registry {
  std::mutex mutex;
  std::unordered_map<conn_t *, std::unique_ptr<lupine_staging_state>> states;
};

static lupine_staging_registry &lupine_staging_states() {
  static auto *registry = new lupine_staging_registry();
  return *registry;
}

static lupine_staging_state *lupine_staging_state_for(conn_t *conn) {
  if (conn == nullptr) {
    return nullptr;
  }
  auto &registry = lupine_staging_states();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto state = registry.states.find(conn);
  return state == registry.states.end() ? nullptr : state->second.get();
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
    } else if (result != CUDA_ERROR_NOT_READY) {
      // An error cannot prove the DMA is finished. Never make this allocation
      // reusable until its context is explicitly retired.
      slot.state = lupine_async_htod_state::quarantined;
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
      LUPINE_LOG_ERROR("Async HtoD spill event query failed: " << result);
      ++spill;
      continue;
    }
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
  return result;
}

static CUresult lupine_async_htod_publish_spill(lupine_async_htod_spill &spill,
                                                CUstream stream) {
  CUresult result = cuEventRecord(spill.completion, stream);
  spill.completion_recorded = result == CUDA_SUCCESS;
  return result;
}

static void lupine_async_htod_discard_spill(lupine_async_htod_spill &spill) {
  lupine_async_htod_destroy_event(&spill.completion, spill.event_context);
  lupine_async_htod_free_host(&spill.ptr, spill.allocation_context);
}

static CUresult lupine_async_htod_enqueue_spill(
    lupine_staging_state &state, conn_t *conn, int framed,
    CUdeviceptr destination, size_t bytes, CUstream stream, CUcontext context,
    bool *payload_consumed, bool *connection_failed) {
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
    lupine_async_htod_discard_spill(spill);
    state.spills.pop_back();
    return result;
  }
  size_t offset = 0;
  while (offset < bytes) {
    size_t chunk = std::min(LUPINE_HTOD_CHUNK_BYTES, bytes - offset);
    auto *chunk_host = static_cast<unsigned char *>(spill.ptr) + offset;
    if (rpc_read_payload_part(conn, framed, chunk_host, chunk) < 0) {
      if (spill.work_queued) {
        (void)lupine_async_htod_publish_spill(spill, stream);
      } else {
        lupine_async_htod_discard_spill(spill);
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
      (void)lupine_async_htod_publish_spill(spill, stream);
      if (rpc_drain_payload(conn, framed, bytes - offset) < 0) {
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
  return lupine_async_htod_publish_spill(spill, stream);
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
        LUPINE_LOG_ERROR(
            "Could not prove async HtoD slot completion; quarantining it: "
            << result);
        continue;
      }
      slot.state = lupine_async_htod_state::available;
    } else if (slot.state == lupine_async_htod_state::quarantined) {
      continue;
    }
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
      LUPINE_LOG_ERROR(
          "Could not prove async HtoD spill completion; quarantining it: "
          << result);
      ++spill;
      continue;
    }
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
      slot = {};
    }
  }

  state.spills.erase(std::remove_if(state.spills.begin(), state.spills.end(),
                                    [context](const auto &spill) {
                                      return spill.allocation_context ==
                                                 context ||
                                             spill.event_context == context;
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
  auto &registry = lupine_staging_states();
  std::lock_guard<std::mutex> lock(registry.mutex);
  return registry.states.emplace(conn, std::move(state)).second;
}

static void lupine_server_begin_lifecycle_transaction(conn_t *conn) {
  auto *state = lupine_staging_state_for(conn);
  if (state != nullptr) {
    state->lifecycle_mutex.lock();
  }
}

static void lupine_server_end_lifecycle_transaction(conn_t *conn) {
  auto *state = lupine_staging_state_for(conn);
  if (state != nullptr) {
    state->lifecycle_mutex.unlock();
  }
}

static void lupine_server_note_primary_context(conn_t *conn, CUdevice device,
                                               CUcontext context,
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
    state->primary_contexts[device] = context;
  } catch (...) {
    LUPINE_LOG_ERROR("Failed to remember primary CUDA context for staging");
  }
}

static void lupine_server_note_created_context(conn_t *conn, CUcontext context,
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

static void lupine_server_prepare_primary_context(conn_t *conn,
                                                  CUdevice device) {
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

static void lupine_server_finish_primary_context(conn_t *conn, CUdevice device,
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

static void lupine_server_prepare_context_destroy(conn_t *conn,
                                                  CUcontext context) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  state->teardown_contexts.insert(context);
  state->condition.wait(lock, [&] { return !state->staging_operation_active; });
  lupine_async_htod_retire_context(*state, context);
}

static void lupine_server_finish_context_destroy(conn_t *conn,
                                                 CUcontext context,
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

static void lupine_server_finish_context_detach(conn_t *conn, CUcontext context,
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
  {
    auto &registry = lupine_staging_states();
    std::lock_guard<std::mutex> lock(registry.mutex);
    auto state = registry.states.find(conn);
    if (state == registry.states.end()) {
      return;
    }
    owned_state = std::move(state->second);
    registry.states.erase(state);
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

static int lupine_write_lifecycle_response(conn_t *conn, int request_id,
                                           CUresult result) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuCtxCreate_v2(conn_t *conn) {
  unsigned int flags = 0;
  CUdevice device = 0;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &device, sizeof(device)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUcontext context = nullptr;
  lupine_server_begin_lifecycle_transaction(conn);
  CUresult result = cuCtxCreate_v2(&context, flags, device);
  lupine_server_note_created_context(conn, context, result);
  lupine_server_end_lifecycle_transaction(conn);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &context, sizeof(context)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDevicePrimaryCtxRetain(conn_t *conn) {
  CUdevice device = 0;
  if (rpc_read(conn, &device, sizeof(device)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUcontext context = nullptr;
  lupine_server_begin_lifecycle_transaction(conn);
  CUresult result = cuDevicePrimaryCtxRetain(&context, device);
  lupine_server_note_primary_context(conn, device, context, result);
  lupine_server_end_lifecycle_transaction(conn);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &context, sizeof(context)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDevicePrimaryCtxRelease_v2(conn_t *conn) {
  CUdevice device = 0;
  if (rpc_read(conn, &device, sizeof(device)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_server_begin_lifecycle_transaction(conn);
  lupine_server_prepare_primary_context(conn, device);
  CUresult result = cuDevicePrimaryCtxRelease_v2(device);
  lupine_server_finish_primary_context(conn, device, false, result);
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_manual_cuDevicePrimaryCtxReset_v2(conn_t *conn) {
  CUdevice device = 0;
  if (rpc_read(conn, &device, sizeof(device)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_server_begin_lifecycle_transaction(conn);
  lupine_server_prepare_primary_context(conn, device);
  CUresult result = cuDevicePrimaryCtxReset_v2(device);
  lupine_server_finish_primary_context(conn, device, true, result);
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_manual_cuCtxAttach(conn_t *conn) {
  unsigned int flags = 0;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUcontext context = nullptr;
  lupine_server_begin_lifecycle_transaction(conn);
  CUresult result = cuCtxAttach(&context, flags);
  lupine_server_end_lifecycle_transaction(conn);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &context, sizeof(context)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuCtxDestroy_v2(conn_t *conn) {
  CUcontext context = nullptr;
  if (rpc_read(conn, &context, sizeof(context)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_server_begin_lifecycle_transaction(conn);
  lupine_server_prepare_context_destroy(conn, context);
  CUresult result = cuCtxDestroy_v2(context);
  lupine_server_finish_context_destroy(conn, context, result);
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_manual_cuCtxDetach(conn_t *conn) {
  CUcontext context = nullptr;
  if (rpc_read(conn, &context, sizeof(context)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_server_begin_lifecycle_transaction(conn);
  lupine_server_prepare_context_destroy(conn, context);
  CUresult result = cuCtxDetach(context);
  lupine_server_finish_context_detach(conn, context, result);
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

static int lupine_copy_htod_serial(conn_t *conn, int framed,
                                   CUdeviceptr destination, size_t bytes,
                                   lupine_staging_state &state,
                                   CUresult *result) {
  size_t chunk_bytes = std::min(LUPINE_HTOD_CHUNK_BYTES, bytes);
  lupine_staging staging =
      lupine_acquire_staging(chunk_bytes, state.sync_staging);
  if (chunk_bytes != 0 && staging.ptr == nullptr) {
    *result = CUDA_ERROR_OUT_OF_MEMORY;
    return rpc_drain_payload(conn, framed, bytes) < 0 ? -1 : 0;
  }

  size_t offset = 0;
  while (*result == CUDA_SUCCESS && offset < bytes) {
    size_t chunk = std::min(chunk_bytes, bytes - offset);
    if (rpc_read_payload_part(conn, framed, staging.ptr, chunk) < 0) {
      lupine_release_staging(staging);
      return -1;
    }
    *result = cuMemcpyHtoD_v2(destination + offset, staging.ptr, chunk);
    offset += chunk;
    if (*result != CUDA_SUCCESS &&
        rpc_drain_payload(conn, framed, bytes - offset) < 0) {
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
static int lupine_copy_htod_pipelined(conn_t *conn, int framed,
                                      CUdeviceptr destination, size_t bytes,
                                      CUcontext context,
                                      lupine_staging_state &state,
                                      CUresult *result) {
#ifdef _WIN32
  (void)conn;
  (void)framed;
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
        if (rpc_drain_payload(conn, framed, bytes - offset) < 0) {
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
    if (rpc_read_payload_part(conn, framed, host, chunk) < 0) {
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
      if (rpc_drain_payload(conn, framed, bytes - offset) < 0) {
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
        if (rpc_drain_payload(conn, framed, bytes - offset) < 0) {
          lupine_destroy_sync_htod_events(events);
          return -1;
        }
      } else if (lupine_copy_htod_serial(conn, framed, destination + offset,
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

int handle_manual_cuMemcpyHtoD_v2(conn_t *conn) {
  CUdeviceptr destination = 0;
  size_t bytes = 0;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &destination, sizeof(destination)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
    return -1;
  }

  int framed = lupine_payload_framed(conn, bytes);
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

  int pipeline_result = 0;
  if (result == CUDA_SUCCESS) {
    pipeline_result = lupine_copy_htod_pipelined(
        conn, framed, destination, bytes, context, *state, &result);
  }
  if (pipeline_result < 0 ||
      (pipeline_result == 0 && result == CUDA_SUCCESS &&
       lupine_copy_htod_serial(conn, framed, destination, bytes, *state,
                               &result) < 0)) {
    return -1;
  }
  if (pipeline_result == 0 && result != CUDA_SUCCESS &&
      rpc_drain_payload(conn, framed, bytes) < 0) {
    return -1;
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int lupine_write_dtoh_chunk_response(conn_t *conn, int request_id,
                                            CUresult result, const void *data,
                                            size_t bytes) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (result == CUDA_SUCCESS && bytes != 0 &&
       rpc_write_payload(conn, data, bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int lupine_copy_dtoh_serial(conn_t *conn, int request_id,
                                   CUdeviceptr source, size_t bytes,
                                   size_t offset) {
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
    CUresult result = cuMemcpyDtoH_v2(destination, source + offset, chunk);
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
                                             bool synchronize_stream) {
  CUresult completion_result = CUDA_SUCCESS;
  bool completion_confirmed = true;
  for (const auto &slot : pipeline.slots) {
    if (slot.in_flight && !slot.event_recorded) {
      synchronize_stream = true;
    }
  }
  if (synchronize_stream) {
    completion_result = cuStreamSynchronize(CU_STREAM_LEGACY);
    completion_confirmed = completion_result == CUDA_SUCCESS;
  } else {
    for (const auto &slot : pipeline.slots) {
      if (!slot.in_flight || !slot.event_recorded) {
        continue;
      }
      CUresult result = cuEventSynchronize(slot.completion);
      if (result != CUDA_SUCCESS) {
        completion_result = result;
        CUresult stream_result = cuStreamSynchronize(CU_STREAM_LEGACY);
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
                                      size_t *fallback_offset) {
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
      (void)lupine_cleanup_dtoh_pipeline(pipeline, false);
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
    CUresult result = cuMemcpyDtoHAsync_v2(slot.data, source + offset,
                                           slot.bytes, CU_STREAM_LEGACY);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (cuEventRecord(slot.completion, CU_STREAM_LEGACY) != CUDA_SUCCESS) {
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
      CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, true);
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
      (void)lupine_cleanup_dtoh_pipeline(pipeline, false);
      return write_result;
    }

    size_t index = (sent_offset / LUPINE_DTOH_PIPELINE_SLOT_BYTES) %
                   LUPINE_DTOH_PIPELINE_SLOT_COUNT;
    auto &slot = pipeline.slots[index];
    if (!slot.in_flight || !slot.event_recorded || slot.offset != sent_offset) {
      CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, true);
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
      (void)lupine_cleanup_dtoh_pipeline(pipeline, true);
      return write_result;
    }
    if (lupine_write_dtoh_chunk_response(conn, request_id, CUDA_SUCCESS,
                                         slot.data, slot.bytes) < 0) {
      (void)lupine_cleanup_dtoh_pipeline(pipeline, false);
      return -1;
    }

    sent_offset += slot.bytes;
    slot.in_flight = false;
    slot.event_recorded = false;
    if (submitted_offset < bytes && !terminal_error_pending) {
      result = submit(slot, submitted_offset);
      if (event_record_failed) {
        CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, true);
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

  (void)lupine_cleanup_dtoh_pipeline(pipeline, false);
  return 0;
}

int handle_manual_cuMemcpyDtoH_v2(conn_t *conn) {
  CUdeviceptr source = 0;
  size_t bytes = 0;
  if (rpc_read(conn, &source, sizeof(source)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t fallback_offset = 0;
  if (bytes >= LUPINE_DTOH_PIPELINE_MIN_BYTES) {
    int result = lupine_copy_dtoh_pipelined(conn, request_id, source, bytes,
                                            &fallback_offset);
    if (result <= 0) {
      return result;
    }
  }
  return lupine_copy_dtoh_serial(conn, request_id, source, bytes,
                                 fallback_offset);
}

int lupine_server_copy_htod_async(conn_t *conn, int framed,
                                  CUdeviceptr dstDevice, size_t byteCount,
                                  CUstream stream, CUresult &result) {
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
    if (rpc_read_payload_part(conn, framed, slot->ptr, chunk) < 0) {
      return -1;
    }

    CUresult copy_result =
        cuMemcpyHtoDAsync_v2(dstDevice + offset, slot->ptr, chunk, stream);
    offset += chunk;
    if (copy_result != CUDA_SUCCESS) {
      (void)lupine_async_htod_publish_slot(*state, slot, stream);
      result = copy_result;
      if (rpc_drain_payload(conn, framed, byteCount - offset) < 0) {
        return -1;
      }
      break;
    }
    result = lupine_async_htod_publish_slot(*state, slot, stream);
    if (result != CUDA_SUCCESS) {
      // The copy was accepted but its completion could not be published.
      // Keep the slot quarantined until context teardown and report the CUDA
      // error without ever synchronizing the caller's stream.
      if (rpc_drain_payload(conn, framed, byteCount - offset) < 0) {
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
        *state, conn, framed, dstDevice + offset, remaining, stream, context,
        &payload_consumed, &connection_failed);
    if (connection_failed) {
      return -1;
    }
    if (!payload_consumed && rpc_drain_payload(conn, framed, remaining) < 0) {
      return -1;
    }
  } else if (result != CUDA_SUCCESS && offset == 0 &&
             rpc_drain_payload(conn, framed, byteCount) < 0) {
    return -1;
  }
  return 0;
}

#else
#error "memcpy.cpp must be built for the Lupine RPC client or server"
#endif
