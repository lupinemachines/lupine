// Server-side copy machinery: the staging pools the device copies stage
// through, the lifetime rules that tie those pools to a context, and the
// chunked host-to-device and device-to-host loops. HTTP content compression
// is handled transparently by the transport.
#include <cuda.h>

#include "codegen/gen_rpc_ids.h"
#include "cuda_server.h"
#include "cuda_server_memcpy.h"
#include "cuda_side_effect.h"
#include "rpc.h"

#include <atomic>
#include <array>
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

class lupine_htod_side_effect_ring;

struct lupine_htod_callback_data {
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  lupine_host_memory_view source;
  size_t bytes = 0;
  size_t slot = 0;
  bool persistent = false;
};

struct lupine_staging_state {
  conn_t *conn = nullptr;
  std::mutex lifecycle_mutex;
  std::mutex mutex;
  std::condition_variable condition;
  bool staging_operation_active = false;
  std::unordered_map<CUdevice, CUcontext> primary_contexts;
  std::unordered_set<CUcontext> created_contexts;
  std::unordered_set<CUcontext> teardown_contexts;
  std::unordered_set<CUdevice> teardown_devices;
  std::shared_ptr<lupine_htod_side_effect_ring> htod_ring;
  std::vector<std::unique_ptr<lupine_htod_callback_data>>
      captured_htod_callbacks;
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

static CUresult lupine_current_htod_context(conn_t *conn,
                                            lupine_staging_state **state,
                                            CUcontext *context,
                                            CUdevice *device) {
  *state = lupine_staging_state_for(conn);
  if (*state == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  CUresult result = cuCtxGetCurrent(context);
  if (result == CUDA_SUCCESS && *context == nullptr) {
    return CUDA_ERROR_INVALID_CONTEXT;
  }
  return result == CUDA_SUCCESS ? cuCtxGetDevice(device) : result;
}

enum class lupine_htod_copy_kind {
  linear,
  two_dimensional,
  three_dimensional
};

struct lupine_htod_chunk {
  lupine_host_memory_view source;
  size_t bytes = 0;
  lupine_htod_copy_kind kind = lupine_htod_copy_kind::linear;
  CUdeviceptr linear_destination = 0;
  CUDA_MEMCPY2D copy_2d = {};
  CUDA_MEMCPY3D copy_3d = {};
};

struct lupine_htod_plan {
  std::vector<lupine_htod_chunk> chunks;
};

// A host node reads one chunk into a pinned slot and the following CUDA node
// consumes it. A GPU write releases that slot after the DMA, so callbacks on
// other streams can share this fixed ring without a producer thread.
class lupine_htod_side_effect_ring {
public:
  static constexpr size_t slot_count = 2;
  static constexpr size_t slot_bytes = 8 * 1024 * 1024;
  static constexpr uint32_t slot_free = 0;
  static constexpr uint32_t slot_busy = 1;

  explicit lupine_htod_side_effect_ring(conn_t *conn) : conn_(conn) {}

  ~lupine_htod_side_effect_ring() {
    if (signals_ != nullptr) {
      (void)cuMemFreeHost(signals_);
    }
    if (storage_ != nullptr) {
      (void)cuMemFreeHost(storage_);
    }
  }

  CUresult prepare() {
    std::lock_guard<std::mutex> lock(prepare_mutex_);
    if (storage_ != nullptr) {
      return CUDA_SUCCESS;
    }

    void *storage = nullptr;
    CUresult result = cuMemHostAlloc(&storage, slot_count * slot_bytes,
                                     CU_MEMHOSTALLOC_PORTABLE);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    slot_signal *signals = nullptr;
    result = cuMemHostAlloc(reinterpret_cast<void **>(&signals),
                            slot_count * sizeof(slot_signal),
                            CU_MEMHOSTALLOC_PORTABLE |
                                CU_MEMHOSTALLOC_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
      (void)cuMemFreeHost(storage);
      return result;
    }
    CUdeviceptr device_signals = 0;
    result = cuMemHostGetDevicePointer_v2(&device_signals, signals, 0);
    if (result != CUDA_SUCCESS) {
      (void)cuMemFreeHost(signals);
      (void)cuMemFreeHost(storage);
      return result;
    }
    for (size_t slot = 0; slot < slot_count; ++slot) {
      __atomic_store_n(&signals[slot].value, slot_free, __ATOMIC_RELEASE);
    }
    storage_ = storage;
    signals_ = signals;
    device_signals_ = device_signals;
    return CUDA_SUCCESS;
  }

  size_t next_slot() {
    return next_slot_.fetch_add(1, std::memory_order_relaxed) % slot_count;
  }

  void *data(size_t slot) const {
    return static_cast<unsigned char *>(storage_) + slot * slot_bytes;
  }

  CUdeviceptr device_signal(size_t slot) const {
    return device_signals_ + slot * sizeof(slot_signal);
  }

  int fetch(size_t slot, const lupine_host_memory_view &source, size_t bytes) {
    uint32_t expected = slot_free;
    while (!__atomic_compare_exchange_n(&signals_[slot].value, &expected,
                                        slot_busy, false, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE)) {
      expected = slot_free;
      std::this_thread::yield();
    }
    uint64_t count = 1;
    if (rpc_write_start_request(
            conn_, static_cast<int>(lupine_side_effect_op::read_host_memory)) <
            0 ||
        rpc_write(conn_, &count, sizeof(count)) < 0 ||
        rpc_write(conn_, &source, sizeof(source)) < 0 ||
        rpc_wait_for_response(conn_) < 0 ||
        rpc_read(conn_, data(slot), bytes) < 0 ||
        rpc_read_end(conn_) < 0) {
      __atomic_store_n(&signals_[slot].value, slot_free, __ATOMIC_RELEASE);
      return -1;
    }
    return 0;
  }

private:
  struct alignas(64) slot_signal {
    uint32_t value = slot_free;
  };

  conn_t *conn_ = nullptr;
  std::mutex prepare_mutex_;
  void *storage_ = nullptr;
  slot_signal *signals_ = nullptr;
  CUdeviceptr device_signals_ = 0;
  std::atomic<size_t> next_slot_{0};
};

static std::shared_ptr<lupine_htod_side_effect_ring>
lupine_prepare_htod_side_effect_ring(lupine_staging_state &state,
                                     CUresult &result) {
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    ring = state.htod_ring;
    if (ring == nullptr) {
      try {
        ring = std::make_shared<lupine_htod_side_effect_ring>(state.conn);
        state.htod_ring = ring;
      } catch (...) {
        result = CUDA_ERROR_OUT_OF_MEMORY;
        return nullptr;
      }
    }
  }
  result = ring->prepare();
  return result == CUDA_SUCCESS ? ring : nullptr;
}

static void CUDA_CB lupine_htod_side_effect_callback(void *opaque) {
  auto *data = static_cast<lupine_htod_callback_data *>(opaque);
  if (data == nullptr) {
    return;
  }
  bool persistent = data->persistent;
  if (data->bytes != 0 &&
      data->ring->fetch(data->slot, data->source, data->bytes) < 0) {
    LUPINE_LOG_ERROR("HtoD side-effect transport failed");
  }
  if (!persistent) {
    delete data;
  }
}

static CUresult lupine_enqueue_htod_callback(
    lupine_staging_state &state,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    const lupine_host_memory_view &source, size_t bytes, size_t slot,
    CUstream stream, bool persistent) {
  auto callback = std::unique_ptr<lupine_htod_callback_data>(
      new (std::nothrow) lupine_htod_callback_data{ring, source, bytes, slot,
                                                   persistent});
  if (callback == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  auto *callback_ptr = callback.get();
  if (persistent) {
    try {
      state.captured_htod_callbacks.push_back(std::move(callback));
    } catch (...) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  CUresult result =
      cuLaunchHostFunc(stream, lupine_htod_side_effect_callback, callback_ptr);
  if (result != CUDA_SUCCESS) {
    if (persistent) {
      state.captured_htod_callbacks.pop_back();
    }
  } else if (!persistent) {
    (void)callback.release();
  }
  return result;
}

static CUresult lupine_enqueue_htod_chunk(const lupine_htod_chunk &chunk,
                                          void *source, CUstream stream) {
  switch (chunk.kind) {
  case lupine_htod_copy_kind::linear:
    return cuMemcpyHtoDAsync_v2(chunk.linear_destination, source, chunk.bytes,
                                stream);
  case lupine_htod_copy_kind::two_dimensional: {
    CUDA_MEMCPY2D copy = chunk.copy_2d;
    copy.srcHost = source;
    return cuMemcpy2DAsync_v2(&copy, stream);
  }
  case lupine_htod_copy_kind::three_dimensional: {
    CUDA_MEMCPY3D copy = chunk.copy_3d;
    copy.srcHost = source;
    return cuMemcpy3DAsync_v2(&copy, stream);
  }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

static CUresult lupine_enqueue_client_htod_plan(
    lupine_staging_state &state,
    const lupine_htod_plan &plan, CUstream stream, bool blocking) {
  if (plan.chunks.empty()) {
    return CUDA_SUCCESS;
  }

  CUresult result = CUDA_SUCCESS;
  auto ring = lupine_prepare_htod_side_effect_ring(state, result);
  if (ring == nullptr) {
    return result;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    result = cuStreamIsCapturing(stream, &capture_status);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  bool persistent = capture_status != CU_STREAM_CAPTURE_STATUS_NONE;
  for (const auto &chunk : plan.chunks) {
    size_t slot = ring->next_slot();
    result = lupine_enqueue_htod_callback(state, ring, chunk.source,
                                          chunk.bytes, slot, stream,
                                          persistent);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    result = lupine_enqueue_htod_chunk(chunk, ring->data(slot), stream);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    result = cuStreamWriteValue32_v2(stream, ring->device_signal(slot),
                                     lupine_htod_side_effect_ring::slot_free,
                                     0);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  if (!blocking) {
    return CUDA_SUCCESS;
  }
  return cuStreamSynchronize(stream);
}

template <typename MakePlan>
static CUresult lupine_copy_client_host_to_device(conn_t *conn,
                                                  CUstream stream,
                                                  bool blocking,
                                                  MakePlan make_plan) {
  lupine_staging_state *state = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  CUresult result =
      lupine_current_htod_context(conn, &state, &context, &device);
  lupine_staging_operation operation(result == CUDA_SUCCESS ? state : nullptr,
                                     context, device);
  if (result == CUDA_SUCCESS && !operation.acquired()) {
    result = CUDA_ERROR_INVALID_CONTEXT;
  }

  lupine_htod_plan plan;
  if (result == CUDA_SUCCESS) {
    try {
      plan = make_plan();
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  if (result == CUDA_SUCCESS) {
    result = lupine_enqueue_client_htod_plan(*state, plan, stream, blocking);
  }
  return result;
}

static lupine_htod_plan lupine_make_linear_htod_plan(CUdeviceptr destination,
                                                      const void *source,
                                                      size_t bytes) {
  lupine_htod_plan plan;
  plan.chunks.reserve(bytes / lupine_htod_side_effect_ring::slot_bytes +
                      (bytes % lupine_htod_side_effect_ring::slot_bytes != 0));
  size_t offset = 0;
  while (offset < bytes) {
    size_t chunk_bytes =
        std::min(bytes - offset, lupine_htod_side_effect_ring::slot_bytes);
    lupine_htod_chunk chunk;
    const auto *source_bytes = static_cast<const unsigned char *>(source);
    chunk.source = {source_bytes == nullptr ? nullptr : source_bytes + offset,
                    chunk_bytes,
                    1,
                    chunk_bytes,
                    1,
                    chunk_bytes};
    chunk.bytes = chunk_bytes;
    chunk.linear_destination = destination + offset;
    plan.chunks.push_back(chunk);
    offset += chunk_bytes;
  }
  return plan;
}

static lupine_htod_plan
lupine_make_2d_htod_plan(const CUDA_MEMCPY2D &original, bool unaligned) {
  lupine_htod_plan plan;
  const size_t width = original.WidthInBytes;
  const size_t height = original.Height;
  if (width == 0 || height == 0) {
    return plan;
  }

  const auto *source = static_cast<const unsigned char *>(original.srcHost) +
                       original.srcY * original.srcPitch + original.srcXInBytes;
  auto append = [&](const void *chunk_source, size_t chunk_width,
                    size_t chunk_rows, size_t source_stride,
                    size_t destination_x, size_t destination_y) {
    lupine_htod_chunk chunk;
    chunk.source = {chunk_source,  chunk_width, chunk_rows,
                    source_stride, 1,           chunk_width * chunk_rows};
    chunk.bytes = chunk_width * chunk_rows;
    chunk.kind = lupine_htod_copy_kind::two_dimensional;
    chunk.copy_2d = original;
    chunk.copy_2d.srcMemoryType = CU_MEMORYTYPE_HOST;
    chunk.copy_2d.srcHost = nullptr;
    chunk.copy_2d.srcPitch = chunk_width;
    chunk.copy_2d.srcXInBytes = 0;
    chunk.copy_2d.srcY = 0;
    chunk.copy_2d.dstXInBytes = original.dstXInBytes + destination_x;
    chunk.copy_2d.dstY = original.dstY + destination_y;
    chunk.copy_2d.WidthInBytes = chunk_width;
    chunk.copy_2d.Height = chunk_rows;
    plan.chunks.push_back(chunk);
  };

  if (width <= lupine_htod_side_effect_ring::slot_bytes) {
    // cuMemcpy2DUnaligned has no asynchronous entry point. Splitting it into
    // single-row cuMemcpy2DAsync calls preserves its relaxed source-pitch
    // behavior because each staged row is tightly packed.
    size_t rows_per_chunk =
        unaligned ? 1 : lupine_htod_side_effect_ring::slot_bytes / width;
    for (size_t row = 0; row < height; row += rows_per_chunk) {
      size_t rows = std::min(rows_per_chunk, height - row);
      append(source + row * original.srcPitch, width, rows, original.srcPitch,
             0, row);
    }
  } else {
    for (size_t row = 0; row < height; ++row) {
      for (size_t x = 0; x < width;
           x += lupine_htod_side_effect_ring::slot_bytes) {
        size_t run =
            std::min(lupine_htod_side_effect_ring::slot_bytes, width - x);
        append(source + row * original.srcPitch + x, run, 1, run, x, row);
      }
    }
  }
  return plan;
}

static lupine_htod_plan
lupine_make_3d_htod_plan(const CUDA_MEMCPY3D &original) {
  lupine_htod_plan plan;
  const size_t width = original.WidthInBytes;
  const size_t height = original.Height;
  const size_t depth = original.Depth;
  if (width == 0 || height == 0 || depth == 0) {
    return plan;
  }

  const size_t source_slice_stride = original.srcHeight * original.srcPitch;
  const auto *source = static_cast<const unsigned char *>(original.srcHost) +
                       original.srcZ * source_slice_stride +
                       original.srcY * original.srcPitch + original.srcXInBytes;
  auto append = [&](const void *chunk_source, size_t chunk_width,
                    size_t chunk_rows, size_t chunk_slices,
                    size_t source_row_stride, size_t source_slice_pitch,
                    size_t destination_x, size_t destination_y,
                    size_t destination_z) {
    lupine_htod_chunk chunk;
    chunk.source = {chunk_source,      chunk_width,  chunk_rows,
                    source_row_stride, chunk_slices, source_slice_pitch};
    chunk.bytes = chunk_width * chunk_rows * chunk_slices;
    chunk.kind = lupine_htod_copy_kind::three_dimensional;
    chunk.copy_3d = original;
    chunk.copy_3d.srcMemoryType = CU_MEMORYTYPE_HOST;
    chunk.copy_3d.srcHost = nullptr;
    chunk.copy_3d.srcPitch = chunk_width;
    chunk.copy_3d.srcHeight = chunk_rows;
    chunk.copy_3d.srcXInBytes = 0;
    chunk.copy_3d.srcY = 0;
    chunk.copy_3d.srcZ = 0;
    chunk.copy_3d.dstXInBytes = original.dstXInBytes + destination_x;
    chunk.copy_3d.dstY = original.dstY + destination_y;
    chunk.copy_3d.dstZ = original.dstZ + destination_z;
    chunk.copy_3d.WidthInBytes = chunk_width;
    chunk.copy_3d.Height = chunk_rows;
    chunk.copy_3d.Depth = chunk_slices;
    plan.chunks.push_back(chunk);
  };

  if (width > lupine_htod_side_effect_ring::slot_bytes) {
    for (size_t z = 0; z < depth; ++z) {
      for (size_t row = 0; row < height; ++row) {
        for (size_t x = 0; x < width;
             x += lupine_htod_side_effect_ring::slot_bytes) {
          size_t run =
              std::min(lupine_htod_side_effect_ring::slot_bytes, width - x);
          append(source + z * source_slice_stride + row * original.srcPitch + x,
                 run, 1, 1, run, run, x, row, z);
        }
      }
    }
    return plan;
  }

  const size_t rows_per_slot = lupine_htod_side_effect_ring::slot_bytes / width;
  if (rows_per_slot < height) {
    for (size_t z = 0; z < depth; ++z) {
      for (size_t row = 0; row < height; row += rows_per_slot) {
        size_t rows = std::min(rows_per_slot, height - row);
        append(source + z * source_slice_stride + row * original.srcPitch,
               width, rows, 1, original.srcPitch, width * rows, 0, row, z);
      }
    }
    return plan;
  }

  const size_t slices_per_slot = rows_per_slot / height;
  for (size_t z = 0; z < depth; z += slices_per_slot) {
    size_t slices = std::min(slices_per_slot, depth - z);
    append(source + z * source_slice_stride, width, height, slices,
           original.srcPitch, source_slice_stride, 0, 0, z);
  }
  return plan;
}

static void lupine_server_forget_context_metadata(lupine_staging_state &state,
                                                  CUcontext context) {
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

CUresult lupine_server_prepare_htod_capture(conn_t *conn) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  CUresult result = CUDA_SUCCESS;
  (void)lupine_prepare_htod_side_effect_ring(*state, result);
  return result;
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
    state->captured_htod_callbacks.clear();
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
  state->captured_htod_callbacks.clear();
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
}

int lupine_write_lifecycle_response(conn_t *conn, int request_id,
                                    CUresult result) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cuMemcpyHtoD_v2(conn_t *conn) {
  CUdeviceptr destination = 0;
  const void *source = nullptr;
  lupine_htod_source_location source_location =
      lupine_htod_source_location::client;
  size_t bytes = 0;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &source_location, sizeof(source_location)) < 0 ||
      rpc_read(conn, &destination, sizeof(destination)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_read(conn, &source, sizeof(source)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  bool source_on_server =
      source_location == lupine_htod_source_location::server;
  if (bytes != 0 && source_on_server) {
    result = cuMemcpyHtoD_v2(destination, source, bytes);
  } else if (bytes != 0) {
    result = lupine_copy_client_host_to_device(
        conn, CU_STREAM_LEGACY, true, [&] {
          return lupine_make_linear_htod_plan(destination, source, bytes);
        });
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
      (result == CUDA_SUCCESS && rpc_write(conn, data, bytes) < 0) ||
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
      std::min(bytes - offset, (size_t)LUPINE_RPC_TRANSFER_CHUNK_BYTES);
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
    LUPINE_RPC_TRANSFER_CHUNK_BYTES;
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

// The client resolves host-to-host locally and picks the direction, so at most
// one side is host here. That side's staging buffer reproduces the caller's
// pitch and offsets, so the descriptor reaches the driver exactly as written
// and only the copied rows travel.
int handle_cuMemcpy3D_v2(conn_t *conn) {
  lupine_memcpy_wire_flags wire_flags;
  CUDA_MEMCPY3D copy = {};
  if (rpc_read(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction = wire_flags.direction;
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        wire_flags.htod_source == lupine_htod_source_location::server
            ? cuMemcpy3D_v2(&copy)
            : lupine_copy_client_host_to_device(
                  conn, CU_STREAM_LEGACY, true, [&] {
                    return lupine_make_3d_htod_plan(copy);
                  });
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
    CUresult result = cuMemcpy3D_v2(&copy);
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
    CUresult result = cuMemcpy3D_v2(&copy);
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
int handle_cuMemcpy3DAsync_v2(conn_t *conn) {
  lupine_memcpy_wire_flags wire_flags;
  CUDA_MEMCPY3D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction = wire_flags.direction;
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        wire_flags.htod_source == lupine_htod_source_location::server
            ? cuMemcpy3DAsync_v2(&copy, stream)
            : lupine_copy_client_host_to_device(
                  conn, stream, false, [&] {
                    return lupine_make_3d_htod_plan(copy);
                  });
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
    CUresult result = cuMemcpy3DAsync_v2(&copy, stream);
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
    CUresult result = cuMemcpy3DAsync_v2(&copy, stream);
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
int handle_cuMemcpy2D_v2(conn_t *conn) {
  lupine_memcpy_wire_flags wire_flags;
  CUDA_MEMCPY2D copy = {};
  if (rpc_read(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction = wire_flags.direction;
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        wire_flags.htod_source == lupine_htod_source_location::server
            ? cuMemcpy2D_v2(&copy)
            : lupine_copy_client_host_to_device(
                  conn, CU_STREAM_LEGACY, true, [&] {
                    return lupine_make_2d_htod_plan(copy, false);
                  });
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case lupine_copy_direction::device_to_host: {
    size_t offset = copy.dstY * copy.dstPitch + copy.dstXInBytes;
    std::vector<unsigned char> host((copy.Height - 1) * copy.dstPitch + offset +
                                    copy.WidthInBytes);
    copy.dstHost = host.data();
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy2D_v2(&copy);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_write_pitched(conn, host.data() + offset, copy.WidthInBytes,
                           copy.Height, copy.dstPitch, 1, 0) < 0) ||
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
    CUresult result = cuMemcpy2D_v2(&copy);
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
int handle_cuMemcpy2DUnaligned_v2(conn_t *conn) {
  lupine_memcpy_wire_flags wire_flags;
  CUDA_MEMCPY2D copy = {};
  if (rpc_read(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction = wire_flags.direction;
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        wire_flags.htod_source == lupine_htod_source_location::server
            ? cuMemcpy2DUnaligned_v2(&copy)
            : lupine_copy_client_host_to_device(
                  conn, CU_STREAM_LEGACY, true, [&] {
                    return lupine_make_2d_htod_plan(copy, true);
                  });
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case lupine_copy_direction::device_to_host: {
    size_t offset = copy.dstY * copy.dstPitch + copy.dstXInBytes;
    std::vector<unsigned char> host((copy.Height - 1) * copy.dstPitch + offset +
                                    copy.WidthInBytes);
    copy.dstHost = host.data();
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy2DUnaligned_v2(&copy);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_write_pitched(conn, host.data() + offset, copy.WidthInBytes,
                           copy.Height, copy.dstPitch, 1, 0) < 0) ||
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
    CUresult result = cuMemcpy2DUnaligned_v2(&copy);
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
int handle_cuMemcpy2DAsync_v2(conn_t *conn) {
  lupine_memcpy_wire_flags wire_flags;
  CUDA_MEMCPY2D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  lupine_copy_direction direction = wire_flags.direction;
  switch (direction) {
  case lupine_copy_direction::host_to_device: {
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        wire_flags.htod_source == lupine_htod_source_location::server
            ? cuMemcpy2DAsync_v2(&copy, stream)
            : lupine_copy_client_host_to_device(
                  conn, stream, false, [&] {
                    return lupine_make_2d_htod_plan(copy, false);
                  });
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case lupine_copy_direction::device_to_host: {
    size_t offset = copy.dstY * copy.dstPitch + copy.dstXInBytes;
    std::vector<unsigned char> host((copy.Height - 1) * copy.dstPitch + offset +
                                    copy.WidthInBytes);
    copy.dstHost = host.data();
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = cuMemcpy2DAsync_v2(&copy, stream);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS &&
         rpc_write_pitched(conn, host.data() + offset, copy.WidthInBytes,
                           copy.Height, copy.dstPitch, 1, 0) < 0) ||
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
    CUresult result = cuMemcpy2DAsync_v2(&copy, stream);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  }
}

int handle_cuMemcpyHtoDAsync_v2(conn_t *conn) {
  CUdeviceptr dstDevice = 0;
  const void *srcHost = nullptr;
  lupine_htod_source_location source_location =
      lupine_htod_source_location::client;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &source_location, sizeof(source_location)) < 0 ||
      rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &srcHost, sizeof(srcHost)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  bool source_on_server =
      source_location == lupine_htod_source_location::server;
  if (byteCount != 0 && source_on_server) {
    result = cuMemcpyHtoDAsync_v2(dstDevice, srcHost, byteCount, stream);
  } else if (byteCount != 0) {
    result = lupine_copy_client_host_to_device(
        conn, stream, false, [&] {
          return lupine_make_linear_htod_plan(dstDevice, srcHost, byteCount);
        });
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

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
      std::min(byteCount, (size_t)LUPINE_RPC_TRANSFER_CHUNK_BYTES);
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
