// Server-side copy machinery: the staging pools the device copies stage
// through, the lifetime rules that tie those pools to a context, and the
// chunked host-to-device and device-to-host loops. HTTP content compression
// is handled transparently by the transport.
#include <cuda.h>

#include "codegen/gen_rpc_ids.h"
#include "cuda_server.h"
#include "cuda_server_memcpy.h"
#include "ops/smemcpy_module.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <atomic>
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
  std::unordered_map<CUcontext, std::shared_ptr<lupine_htod_side_effect_ring>>
      htod_rings;
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

struct lupine_graph_host_copy_node {
  explicit lupine_graph_host_copy_node(lupine_graph_host_copy node_copy)
      : copy(node_copy) {}

  lupine_graph_host_copy copy;
  lupine_graph_host_copy_node *next = nullptr;
};

struct lupine_graph_capture_scratch {
  lupine_graph_capture_scratch(void *scratch_ptr, size_t scratch_size)
      : ptr(scratch_ptr), size(scratch_size) {}

  void *ptr;
  size_t size;
  std::atomic<size_t> offset{0};
};

struct lupine_graph_resources {
  void add_dtoh_copy(lupine_graph_host_copy copy) {
    auto *node = new lupine_graph_host_copy_node(copy);
    node->next = dtoh_copies.load(std::memory_order_relaxed);
    while (!dtoh_copies.compare_exchange_weak(node->next, node,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
    }
  }

  std::vector<lupine_graph_host_copy> dtoh_copy_snapshot() const {
    std::vector<lupine_graph_host_copy> copies;
    for (auto *node = dtoh_copies.load(std::memory_order_acquire);
         node != nullptr; node = node->next) {
      copies.push_back(node->copy);
    }
    std::reverse(copies.begin(), copies.end());
    return copies;
  }

  bool has_capture_scratch() const {
    return capture_scratch.load(std::memory_order_acquire) != nullptr;
  }

  bool install_capture_scratch(void *scratch, size_t size) {
    if (scratch == nullptr) {
      return false;
    }
    auto *candidate = new lupine_graph_capture_scratch(scratch, size);
    lupine_graph_capture_scratch *expected = nullptr;
    if (!capture_scratch.compare_exchange_strong(expected, candidate,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
      delete candidate;
      return false;
    }
    return true;
  }

  void *allocate_capture_scratch(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    auto *scratch = capture_scratch.load(std::memory_order_acquire);
    if (scratch == nullptr) {
      return nullptr;
    }
    size_t current = scratch->offset.load(std::memory_order_relaxed);
    for (;;) {
      if (current > scratch->size || current > SIZE_MAX - 255) {
        return nullptr;
      }
      size_t aligned = (current + 255) & ~size_t(255);
      if (aligned > scratch->size || bytes > scratch->size - aligned) {
        return nullptr;
      }
      if (scratch->offset.compare_exchange_weak(current, aligned + bytes,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
        return static_cast<unsigned char *>(scratch->ptr) + aligned;
      }
    }
  }

  std::atomic<lupine_graph_host_copy_node *> dtoh_copies{nullptr};
  std::atomic<lupine_graph_capture_scratch *> capture_scratch{nullptr};
};

// Graph host buffers must remain valid for any queued launch or replay.
// Graph-resource objects intentionally have process lifetime; maps use stable
// raw pointers while each object retains its owned allocations.
static libcuckoo::cuckoohash_map<CUgraph, lupine_graph_resources *> &
lupine_graph_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraph, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUgraphExec, lupine_graph_resources *> &
lupine_graph_exec_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraphExec, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *> &
lupine_stream_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *> &
lupine_active_stream_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *> &
lupine_event_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *> &
lupine_active_event_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *>();
  return *resources;
}

template <typename Map>
static void lupine_erase_capture_resources(Map &map,
                                           lupine_graph_resources *resources) {
  auto table = map.lock_table();
  for (auto it = table.begin(); it != table.end();) {
    if (it->second == resources) {
      it = table.erase(it);
    } else {
      ++it;
    }
  }
}

lupine_graph_resources *lupine_get_graph_resources(CUgraph graph) {
  auto *candidate = new lupine_graph_resources();
  auto *resources = candidate;
  lupine_graph_resource_map().upsert(
      graph,
      [&resources](lupine_graph_resources *&existing,
                   libcuckoo::UpsertContext) { resources = existing; },
      candidate);
  if (resources != candidate) {
    delete candidate;
  }
  return resources;
}

lupine_graph_resources *lupine_get_stream_resources(CUstream stream) {
  auto *candidate = new lupine_graph_resources();
  auto *resources = candidate;
  lupine_stream_capture_resource_map().upsert(
      stream,
      [&resources](lupine_graph_resources *&existing,
                   libcuckoo::UpsertContext) { resources = existing; },
      candidate);
  if (resources != candidate) {
    delete candidate;
  }
  return resources;
}

lupine_graph_resources *lupine_begin_stream_capture_resources(CUstream stream) {
  // The map also tracks resources from the last graph launched on a stream.
  // A new capture needs independent callback ordering and owned storage.
  auto *resources = new lupine_graph_resources();
  lupine_stream_capture_resource_map().insert_or_assign(stream, resources);
  lupine_active_stream_capture_resource_map().insert_or_assign(stream,
                                                               resources);
  return resources;
}

lupine_graph_resources *lupine_captured_stream_resources(CUstream stream) {
  lupine_graph_resources *resources = nullptr;
  (void)lupine_active_stream_capture_resource_map().find(stream, resources);
  return resources;
}

void lupine_discard_stream_capture_resources(
    lupine_graph_resources *resources) {
  lupine_erase_capture_resources(lupine_active_stream_capture_resource_map(),
                                 resources);
  lupine_erase_capture_resources(lupine_stream_capture_resource_map(),
                                 resources);
}

void lupine_finish_stream_capture_resources(CUstream stream, CUgraph graph,
                                            bool success) {
  lupine_graph_resources *resources = nullptr;
  (void)lupine_active_stream_capture_resource_map().find(stream, resources);
  if (resources == nullptr) {
    return;
  }
  lupine_erase_capture_resources(lupine_active_stream_capture_resource_map(),
                                 resources);
  lupine_erase_capture_resources(lupine_active_event_capture_resource_map(),
                                 resources);
  lupine_erase_capture_resources(lupine_stream_capture_resource_map(),
                                 resources);
  if (success) {
    lupine_graph_resource_map().insert_or_assign(graph, resources);
  }
}

void lupine_record_event_capture_resources(CUevent event, CUstream stream) {
  lupine_graph_resources *resources = nullptr;
  (void)lupine_stream_capture_resource_map().find(stream, resources);
  if (resources == nullptr) {
    lupine_event_capture_resource_map().erase(event);
  } else {
    lupine_event_capture_resource_map().insert_or_assign(event, resources);
  }

  resources = nullptr;
  (void)lupine_active_stream_capture_resource_map().find(stream, resources);
  if (resources == nullptr) {
    lupine_active_event_capture_resource_map().erase(event);
  } else {
    lupine_active_event_capture_resource_map().insert_or_assign(event,
                                                                resources);
  }
}

void lupine_forget_event_capture_resources(CUevent event) {
  lupine_event_capture_resource_map().erase(event);
  lupine_active_event_capture_resource_map().erase(event);
}

void lupine_wait_event_capture_resources(CUstream stream, CUevent event) {
  lupine_graph_resources *resources = nullptr;
  if (lupine_event_capture_resource_map().find(event, resources)) {
    lupine_stream_capture_resource_map().insert(stream, resources);
  }
  if (lupine_active_event_capture_resource_map().find(event, resources)) {
    lupine_active_stream_capture_resource_map().insert(stream, resources);
  }
}

void lupine_clone_graph_resources(CUgraph clone, CUgraph original) {
  lupine_graph_resources *resources = nullptr;
  if (lupine_graph_resource_map().find(original, resources)) {
    lupine_graph_resource_map().insert_or_assign(clone, resources);
  }
}

void lupine_erase_graph_resources(CUgraph graph) {
  lupine_graph_resource_map().erase(graph);
}

void lupine_note_graph_launch(CUgraphExec exec, CUstream stream,
                              CUresult result) {
  lupine_graph_resources *resources = nullptr;
  (void)lupine_graph_exec_resource_map().find(exec, resources);
  if (result == CUDA_SUCCESS && resources != nullptr) {
    lupine_stream_capture_resource_map().insert_or_assign(stream, resources);
  }
}

bool lupine_graph_has_capture_scratch(lupine_graph_resources *resources) {
  return resources != nullptr && resources->has_capture_scratch();
}

bool lupine_graph_install_capture_scratch(lupine_graph_resources *resources,
                                          void *scratch, size_t size) {
  return resources != nullptr &&
         resources->install_capture_scratch(scratch, size);
}

std::vector<lupine_graph_host_copy>
lupine_graph_dtoh_copy_snapshot(lupine_graph_resources *resources) {
  return resources == nullptr ? std::vector<lupine_graph_host_copy>()
                              : resources->dtoh_copy_snapshot();
}

std::vector<lupine_graph_host_copy>
lupine_stream_dtoh_copy_snapshot(CUstream stream) {
  lupine_graph_resources *resources = nullptr;
  (void)lupine_stream_capture_resource_map().find(stream, resources);
  return lupine_graph_dtoh_copy_snapshot(resources);
}

void *lupine_alloc_capture_scratch(lupine_graph_resources *resources,
                                   size_t bytes) {
  return resources == nullptr ? nullptr
                              : resources->allocate_capture_scratch(bytes);
}

void lupine_graph_note_dtoh_copy(lupine_graph_resources *resources,
                                 void *client_dst, void *server_src,
                                 size_t bytes) {
  resources->add_dtoh_copy({client_dst, server_src, bytes});
}

struct lupine_htod_fragment {
  size_t logical_offset = 0;
  size_t bytes = 0;
  size_t x = 0;
  size_t row = 0;
  size_t slice = 0;
  size_t width = 0;
  size_t rows = 0;
  size_t slices = 0;
};

struct lupine_htod_copy {
  const void *source = nullptr;
  size_t source_row_stride = 0;
  size_t source_slice_stride = 0;
  size_t slices = 0;
  size_t bytes = 0;
  bool use_cuda_memcpy = false;
  CUarray destination_array = nullptr;
  size_t destination_lod = 0;
  size_t destination_x = 0;
  size_t destination_y = 0;
  size_t destination_z = 0;
  lupine_smemcpy_params smemcpy = {};

  lupine_htod_fragment fragment(size_t offset, size_t capacity) const {
    lupine_htod_fragment fragment = {};
    fragment.logical_offset = offset;
    size_t row_index = offset / smemcpy.width;
    fragment.x = offset - row_index * smemcpy.width;
    fragment.slice = row_index / smemcpy.rows;
    fragment.row = row_index - fragment.slice * smemcpy.rows;

    if (!use_cuda_memcpy) {
      fragment.bytes = std::min(capacity, bytes - offset);
      return fragment;
    }
    if (fragment.x != 0 || smemcpy.width > capacity) {
      fragment.bytes =
          std::min({capacity, smemcpy.width - fragment.x, bytes - offset});
      fragment.width = fragment.bytes;
      fragment.rows = 1;
      fragment.slices = 1;
      return fragment;
    }

    size_t rows_per_fragment = capacity / smemcpy.width;
    if (fragment.row != 0 || rows_per_fragment < smemcpy.rows) {
      fragment.width = smemcpy.width;
      fragment.rows = std::min(rows_per_fragment, smemcpy.rows - fragment.row);
      fragment.slices = 1;
      fragment.bytes = fragment.width * fragment.rows;
      return fragment;
    }

    fragment.width = smemcpy.width;
    fragment.rows = smemcpy.rows;
    size_t slice_bytes = fragment.width * fragment.rows;
    fragment.slices = std::min(capacity / slice_bytes, slices - fragment.slice);
    fragment.bytes = slice_bytes * fragment.slices;
    return fragment;
  }

  size_t fragment_count(size_t capacity) const {
    size_t count = 0;
    for (size_t offset = 0; offset < bytes; ++count) {
      offset += fragment(offset, capacity).bytes;
    }
    return count;
  }

  CUdeviceptr destination_at(size_t logical_offset) const {
    size_t row_index = logical_offset / smemcpy.width;
    size_t x = logical_offset - row_index * smemcpy.width;
    size_t slice = row_index / smemcpy.rows;
    size_t row = row_index - slice * smemcpy.rows;
    return smemcpy.destination + slice * smemcpy.destination_slice_stride +
           row * smemcpy.destination_row_stride + x;
  }
};

// The callback pumps a complete host transfer while this private CUDA stream
// consumes the fixed ring. The callback itself is the CPU execution context;
// no Lupine worker thread is needed. This stream is dedicated to client-host
// HtoD staging and must never carry application or general RPC work.
class lupine_htod_side_effect_ring {
public:
  static constexpr size_t slot_count = 2;
  static constexpr size_t slot_bytes = 8 * 1024 * 1024;
  static constexpr size_t slot_headroom = 15;
  static constexpr size_t slot_stride =
      (slot_bytes + slot_headroom + 127) & ~size_t(127);
  static constexpr uint32_t slot_free = 0;
  static constexpr uint32_t slot_ready = 1;

  lupine_htod_side_effect_ring(conn_t *conn, CUcontext context)
      : conn_(conn), context_(context) {}

  ~lupine_htod_side_effect_ring() {
    if (context_ != nullptr && cuCtxPushCurrent_v2(context_) == CUDA_SUCCESS) {
      if (transfer_stream_ != nullptr) {
        (void)cuStreamDestroy_v2(transfer_stream_);
      }
      if (smemcpy_module_ != nullptr) {
        (void)cuModuleUnload(smemcpy_module_);
      }
      CUcontext popped = nullptr;
      (void)cuCtxPopCurrent_v2(&popped);
    }
    if (signals_ != nullptr) {
      (void)cuMemFreeHost(signals_);
    }
    if (storage_ != nullptr) {
      (void)cuMemHostUnregister(storage_);
      free_storage(storage_);
    }
  }

  CUresult prepare() {
    std::lock_guard<std::mutex> lock(prepare_mutex_);
    if (storage_ != nullptr) {
      return CUDA_SUCCESS;
    }

    void *storage = allocate_storage();
    if (storage == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult result = cuMemHostRegister(storage, slot_count * slot_stride,
                                        CU_MEMHOSTREGISTER_PORTABLE |
                                            CU_MEMHOSTREGISTER_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
      free_storage(storage);
      return result;
    }
    CUdeviceptr device_storage = 0;
    result = cuMemHostGetDevicePointer_v2(&device_storage, storage, 0);
    if (result != CUDA_SUCCESS) {
      (void)cuMemHostUnregister(storage);
      free_storage(storage);
      return result;
    }
    slot_signal *signals = nullptr;
    result = cuMemHostAlloc(
        reinterpret_cast<void **>(&signals), slot_count * sizeof(slot_signal),
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
      (void)cuMemHostUnregister(storage);
      free_storage(storage);
      return result;
    }
    CUdeviceptr device_signals = 0;
    result = cuMemHostGetDevicePointer_v2(&device_signals, signals, 0);
    if (result != CUDA_SUCCESS) {
      (void)cuMemFreeHost(signals);
      (void)cuMemHostUnregister(storage);
      free_storage(storage);
      return result;
    }
    for (size_t slot = 0; slot < slot_count; ++slot) {
      new (&signals[slot]) slot_signal();
    }

    CUstream transfer_stream = nullptr;
    result = cuStreamCreate(&transfer_stream, CU_STREAM_NON_BLOCKING);
    if (result != CUDA_SUCCESS) {
      (void)cuMemFreeHost(signals);
      (void)cuMemHostUnregister(storage);
      free_storage(storage);
      return result;
    }
    // Load and JIT the fatbin before any wait-value nodes can reach this
    // stream. First-use module work may synchronize with earlier stream work,
    // which would deadlock if that work were waiting for the callback.
    CUmodule smemcpy_module = nullptr;
    result = lupine_smemcpy_module_load(&smemcpy_module);
    if (result != CUDA_SUCCESS) {
      (void)cuStreamDestroy_v2(transfer_stream);
      (void)cuMemFreeHost(signals);
      (void)cuMemHostUnregister(storage);
      free_storage(storage);
      return result;
    }
    storage_ = storage;
    device_storage_ = device_storage;
    signals_ = signals;
    device_signals_ = device_signals;
    transfer_stream_ = transfer_stream;
    smemcpy_module_ = smemcpy_module;
    return CUDA_SUCCESS;
  }

  void acquire_execution() {
    std::unique_lock<std::mutex> lock(execution_mutex_);
    execution_condition_.wait(lock, [&] { return !execution_active_; });
    execution_active_ = true;
  }

  void release_execution() {
    {
      std::lock_guard<std::mutex> lock(execution_mutex_);
      execution_active_ = false;
    }
    execution_condition_.notify_all();
  }

  void *data(size_t slot, size_t offset = 0) const {
    return static_cast<unsigned char *>(storage_) + slot * slot_stride + offset;
  }

  CUdeviceptr device_data(size_t slot, size_t offset = 0) const {
    return device_storage_ + slot * slot_stride + offset;
  }

  CUdeviceptr device_signal(size_t slot) const {
    return device_signals_ + slot * sizeof(slot_signal);
  }

  CUstream transfer_stream() const { return transfer_stream_; }

  conn_t *connection() const { return conn_; }

  CUcontext context() const { return context_; }

  bool data_offset(const void *pointer, size_t *offset) const {
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    uintptr_t begin = reinterpret_cast<uintptr_t>(storage_);
    size_t bytes = slot_count * slot_stride;
    if (address < begin || address - begin >= bytes) {
      return false;
    }
    *offset = address - begin;
    return true;
  }

  bool device_data_offset(CUdeviceptr pointer, size_t *offset) const {
    size_t bytes = slot_count * slot_stride;
    if (pointer < device_storage_ || pointer - device_storage_ >= bytes) {
      return false;
    }
    *offset = static_cast<size_t>(pointer - device_storage_);
    return true;
  }

  bool signal_offset(CUdeviceptr pointer, size_t *offset) const {
    size_t bytes = slot_count * sizeof(slot_signal);
    if (pointer < device_signals_ || pointer - device_signals_ >= bytes) {
      return false;
    }
    *offset = static_cast<size_t>(pointer - device_signals_);
    return true;
  }

  void *data_at(size_t offset) const {
    return static_cast<unsigned char *>(storage_) + offset;
  }

  CUdeviceptr device_data_at(size_t offset) const {
    return device_storage_ + offset;
  }

  CUdeviceptr signal_at(size_t offset) const {
    return device_signals_ + offset;
  }

  CUresult synchronize() const { return cuStreamSynchronize(transfer_stream_); }

  size_t ring_offset(const lupine_htod_copy &copy,
                     size_t logical_offset) const {
    return copy.destination_array == nullptr
               ? copy.destination_at(logical_offset) & 15
               : 0;
  }

  void note_smemcpy_function(CUfunction function) {
    std::lock_guard<std::mutex> lock(smemcpy_functions_mutex_);
    smemcpy_functions_.insert(function);
  }

  bool is_smemcpy_function(CUfunction function) {
    std::lock_guard<std::mutex> lock(smemcpy_functions_mutex_);
    return smemcpy_functions_.count(function) != 0;
  }

  CUresult smemcpy_function(const lupine_smemcpy_launch_descriptor &launch,
                            CUfunction *function) {
    CUresult result =
        lupine_smemcpy_module_function(smemcpy_module_, &launch, function);
    if (result == CUDA_SUCCESS) {
      note_smemcpy_function(*function);
    }
    return result;
  }

  int fetch(const lupine_htod_copy &copy) {
    if (rpc_write_start_request(conn_, LUPINE_SIDE_EFFECT_READ_HOST_MEMORY) <
            0 ||
        rpc_write(conn_, &copy.source, sizeof(copy.source)) < 0 ||
        rpc_write(conn_, &copy.smemcpy.width, sizeof(copy.smemcpy.width)) < 0 ||
        rpc_write(conn_, &copy.smemcpy.rows, sizeof(copy.smemcpy.rows)) < 0 ||
        rpc_write(conn_, &copy.source_row_stride,
                  sizeof(copy.source_row_stride)) < 0 ||
        rpc_write(conn_, &copy.slices, sizeof(copy.slices)) < 0 ||
        rpc_write(conn_, &copy.source_slice_stride,
                  sizeof(copy.source_slice_stride)) < 0) {
      return -1;
    }
    if (rpc_wait_for_response(conn_) < 0) {
      return -1;
    }

    size_t offset = 0;
    for (size_t index = 0; offset < copy.bytes; ++index) {
      auto fragment = copy.fragment(offset, slot_bytes);
      size_t slot = index % slot_count;
      while (signals_[slot].value.load(std::memory_order_acquire) !=
             slot_free) {
        std::this_thread::yield();
      }
      if (rpc_read(conn_, data(slot, ring_offset(copy, offset)),
                   fragment.bytes) < 0) {
        return -1;
      }
      signals_[slot].value.store(slot_ready, std::memory_order_release);
      offset += fragment.bytes;
    }
    if (rpc_read_end(conn_) < 0) {
      return -1;
    }
    for (size_t slot = 0; slot < slot_count; ++slot) {
      while (signals_[slot].value.load(std::memory_order_acquire) !=
             slot_free) {
        std::this_thread::yield();
      }
    }
    return 0;
  }

private:
  static constexpr size_t storage_alignment = 4096;

  static void *allocate_storage() noexcept {
    return ::operator new(slot_count * slot_stride,
                          std::align_val_t(storage_alignment), std::nothrow);
  }

  static void free_storage(void *storage) noexcept {
    ::operator delete(storage, std::align_val_t(storage_alignment));
  }

  struct alignas(64) slot_signal {
    std::atomic<uint32_t> value{slot_free};
  };

  static_assert(std::atomic<uint32_t>::is_always_lock_free);

  conn_t *conn_ = nullptr;
  CUcontext context_ = nullptr;
  std::mutex prepare_mutex_;
  std::mutex execution_mutex_;
  std::mutex smemcpy_functions_mutex_;
  std::condition_variable execution_condition_;
  bool execution_active_ = false;
  void *storage_ = nullptr;
  CUdeviceptr device_storage_ = 0;
  slot_signal *signals_ = nullptr;
  CUdeviceptr device_signals_ = 0;
  CUstream transfer_stream_ = nullptr;
  CUmodule smemcpy_module_ = nullptr;
  std::unordered_set<CUfunction> smemcpy_functions_;
};

struct lupine_htod_capture_events {
  explicit lupine_htod_capture_events(CUcontext event_context)
      : context(event_context) {}

  ~lupine_htod_capture_events() {
    if (context == nullptr || cuCtxPushCurrent_v2(context) != CUDA_SUCCESS) {
      return;
    }
    if (join != nullptr) {
      (void)cuEventDestroy_v2(join);
    }
    if (fork != nullptr) {
      (void)cuEventDestroy_v2(fork);
    }
    CUcontext popped = nullptr;
    (void)cuCtxPopCurrent_v2(&popped);
  }

  CUcontext context = nullptr;
  CUevent fork = nullptr;
  CUevent join = nullptr;
};

// Event creation does not enqueue context work, so it is valid during stream
// capture. Each HtoD owns a pair to keep its fork/join edges independent.
static CUresult lupine_make_htod_capture_events(
    CUcontext context, std::shared_ptr<lupine_htod_capture_events> &events) {
  try {
    events = std::make_shared<lupine_htod_capture_events>(context);
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  CUresult result = cuEventCreate(&events->fork, CU_EVENT_DISABLE_TIMING);
  if (result == CUDA_SUCCESS) {
    result = cuEventCreate(&events->join, CU_EVENT_DISABLE_TIMING);
  }
  if (result != CUDA_SUCCESS) {
    events.reset();
  }
  return result;
}

static std::shared_ptr<lupine_htod_side_effect_ring>
lupine_prepare_htod_side_effect_ring(lupine_staging_state &state,
                                     CUcontext context, CUresult &result) {
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    auto existing = state.htod_rings.find(context);
    if (existing != state.htod_rings.end()) {
      ring = existing->second;
    } else {
      try {
        ring =
            std::make_shared<lupine_htod_side_effect_ring>(state.conn, context);
        state.htod_rings.emplace(context, ring);
      } catch (...) {
        result = CUDA_ERROR_OUT_OF_MEMORY;
        return nullptr;
      }
    }
  }
  result = ring->prepare();
  return result == CUDA_SUCCESS ? ring : nullptr;
}

class lupine_htod_graph_execution {
public:
  explicit lupine_htod_graph_execution(
      std::shared_ptr<lupine_htod_side_effect_ring> ring,
      size_t callback_count = 0)
      : ring_(std::move(ring)), callback_count_(callback_count) {}

  size_t reserve_callback() {
    std::lock_guard<std::mutex> lock(mutex_);
    return callback_count_++;
  }

  void wait_for_turn(size_t callback) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return running_callback_ == callback; });
  }

  void finish_turn() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_callback_ = (running_callback_ + 1) % callback_count_;
    }
    condition_.notify_all();
  }

  const std::shared_ptr<lupine_htod_side_effect_ring> &ring() const {
    return ring_;
  }

  size_t callback_count() const { return callback_count_; }

private:
  std::shared_ptr<lupine_htod_side_effect_ring> ring_;
  std::mutex mutex_;
  std::condition_variable condition_;
  size_t callback_count_ = 0;
  size_t running_callback_ = 0;
};

struct lupine_htod_callback_data {
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  lupine_htod_copy copy;
  std::shared_ptr<lupine_htod_graph_execution> graph_execution;
  std::shared_ptr<lupine_htod_capture_events> capture_events;
  size_t graph_callback = 0;
};

struct lupine_htod_graph_state {
  explicit lupine_htod_graph_state(
      const std::shared_ptr<lupine_htod_side_effect_ring> &ring)
      : execution(std::make_shared<lupine_htod_graph_execution>(ring)) {}

  std::shared_ptr<lupine_htod_graph_execution> execution;
  std::mutex callbacks_mutex;
  std::vector<std::shared_ptr<lupine_htod_callback_data>> callbacks;
};

using lupine_htod_graph_registry =
    libcuckoo::cuckoohash_map<lupine_graph_resources *,
                              std::shared_ptr<lupine_htod_graph_state>>;

static lupine_htod_graph_registry &lupine_htod_graph_states() {
  static auto *states = new lupine_htod_graph_registry();
  return *states;
}

static std::shared_ptr<lupine_htod_graph_state> lupine_htod_graph_state_for(
    lupine_graph_resources *resources,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring) {
  std::shared_ptr<lupine_htod_graph_state> state;
  if (lupine_htod_graph_states().find(resources, state)) {
    return state;
  }
  try {
    auto candidate = std::make_shared<lupine_htod_graph_state>(ring);
    state = candidate;
    lupine_htod_graph_states().upsert(
        resources,
        [&state](std::shared_ptr<lupine_htod_graph_state> &existing,
                 libcuckoo::UpsertContext) { state = existing; },
        std::move(candidate));
  } catch (...) {
    return nullptr;
  }
  return state;
}

struct lupine_htod_exec_resources {
  ~lupine_htod_exec_resources() {
    if (prepared == nullptr || prepared == original) {
      return;
    }
    if (context != nullptr && cuCtxPushCurrent_v2(context) == CUDA_SUCCESS) {
      (void)cuGraphDestroy(prepared);
      CUcontext popped = nullptr;
      (void)cuCtxPopCurrent_v2(&popped);
    }
  }

  std::vector<std::shared_ptr<lupine_htod_callback_data>> callbacks;
  // Exec-node APIs require a node from the graph used for instantiation. Keep
  // the private ring-rebound clone so client nodes can be translated to it.
  CUgraph original = nullptr;
  CUgraph prepared = nullptr;
  CUcontext context = nullptr;
};

static libcuckoo::cuckoohash_map<CUgraphExec,
                                 std::shared_ptr<lupine_htod_exec_resources>> &
lupine_htod_exec_resource_map() {
  static auto *resources = new libcuckoo::cuckoohash_map<
      CUgraphExec, std::shared_ptr<lupine_htod_exec_resources>>();
  return *resources;
}

static void CUDA_CB lupine_htod_side_effect_callback(void *opaque) {
  auto *data = static_cast<lupine_htod_callback_data *>(opaque);
  if (data == nullptr) {
    return;
  }
  if (data->graph_execution != nullptr) {
    data->graph_execution->wait_for_turn(data->graph_callback);
  }
  if (data->ring->fetch(data->copy) < 0) {
    LUPINE_LOG_ERROR("HtoD side-effect transport failed");
  }
  if (data->graph_execution != nullptr) {
    data->graph_execution->finish_turn();
  } else {
    data->ring->release_execution();
  }
  if (data->graph_execution == nullptr) {
    delete data;
  }
}

static CUresult
lupine_prepare_htod_graph_exec(CUgraph graph, lupine_graph_resources *resources,
                               lupine_htod_graph_binding *binding) {
  *binding = {};
  binding->original = graph;
  binding->prepared = graph;
  std::shared_ptr<lupine_htod_graph_state> captured_state;
  if (resources == nullptr ||
      !lupine_htod_graph_states().find(resources, captured_state)) {
    return CUDA_SUCCESS;
  }
  auto captured_execution = captured_state->execution;

  bool prepared_in_place = false;
  CUresult result = cuGraphClone(&binding->prepared, graph);
  // CUDA forbids cloning graphs that contain allocation or free nodes. Such
  // graphs also permit only one executable at a time, so rebinding the graph
  // itself still gives that executable an independent staging ring.
  if (result == CUDA_ERROR_NOT_SUPPORTED) {
    binding->prepared = graph;
    prepared_in_place = true;
    result = CUDA_SUCCESS;
  }
  if (result != CUDA_SUCCESS) {
    binding->prepared = graph;
    return result;
  }

  auto captured_ring = captured_execution->ring();
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  std::shared_ptr<lupine_htod_graph_execution> execution;
  try {
    ring = std::make_shared<lupine_htod_side_effect_ring>(
        captured_ring->connection(), captured_ring->context());
    execution = std::make_shared<lupine_htod_graph_execution>(
        ring, captured_execution->callback_count());
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  result = ring->prepare();
  if (result != CUDA_SUCCESS) {
    return result;
  }

  size_t node_count = 0;
  result = cuGraphGetNodes(binding->prepared, nullptr, &node_count);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  std::vector<CUgraphNode> nodes;
  try {
    nodes.resize(node_count);
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (node_count != 0) {
    result = cuGraphGetNodes(binding->prepared, nodes.data(), &node_count);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    nodes.resize(node_count);
  }

  std::vector<std::shared_ptr<lupine_htod_callback_data>> callbacks;
  std::unordered_map<CUevent, CUevent> events;
  size_t expected_copy_nodes = 0;
  size_t rebound_copy_nodes = 0;
  try {
    callbacks.reserve(captured_execution->callback_count());
    events.reserve(captured_execution->callback_count() * 2);
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // Rebind host-node data first; this also establishes the event mapping used
  // by the record/wait nodes in the second pass.
  for (CUgraphNode node : nodes) {
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    result = cuGraphNodeGetType(node, &type);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (type != CU_GRAPH_NODE_TYPE_HOST) {
      continue;
    }

    CUDA_HOST_NODE_PARAMS params = {};
    result = cuGraphHostNodeGetParams(node, &params);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (params.fn != lupine_htod_side_effect_callback ||
        params.userData == nullptr) {
      continue;
    }
    auto *captured = static_cast<lupine_htod_callback_data *>(params.userData);
    if (captured->graph_execution != captured_execution ||
        captured->capture_events == nullptr) {
      continue;
    }

    std::shared_ptr<lupine_htod_capture_events> callback_events;
    result = lupine_make_htod_capture_events(ring->context(), callback_events);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    std::shared_ptr<lupine_htod_callback_data> callback;
    try {
      callback = std::make_shared<lupine_htod_callback_data>(
          lupine_htod_callback_data{ring, captured->copy, execution,
                                    callback_events, captured->graph_callback});
      events.emplace(captured->capture_events->fork, callback_events->fork);
      events.emplace(captured->capture_events->join, callback_events->join);
      expected_copy_nodes += captured->copy.fragment_count(
          lupine_htod_side_effect_ring::slot_bytes);
      callbacks.push_back(callback);
    } catch (...) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    params.userData = callback.get();
    result = cuGraphHostNodeSetParams(node, &params);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  for (CUgraphNode node : nodes) {
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    result = cuGraphNodeGetType(node, &type);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS params = {};
      result = cuGraphKernelNodeGetParams(node, &params);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      if (params.func == nullptr ||
          !captured_ring->is_smemcpy_function(params.func)) {
        continue;
      }
      if (params.kernelParams == nullptr || params.kernelParams[0] == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
      }
      auto copy =
          *static_cast<const lupine_smemcpy_params *>(params.kernelParams[0]);
      size_t offset = 0;
      if (!captured_ring->device_data_offset(copy.source, &offset)) {
        continue;
      }
      copy.source = ring->device_data_at(offset);
      void *arguments[] = {&copy};
      params.kernelParams = arguments;
      params.extra = nullptr;
      result = cuGraphKernelNodeSetParams(node, &params);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      ++rebound_copy_nodes;
    } else if (type == CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D params = {};
      result = cuGraphMemcpyNodeGetParams(node, &params);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      size_t offset = 0;
      if (params.srcMemoryType == CU_MEMORYTYPE_HOST &&
          captured_ring->data_offset(params.srcHost, &offset)) {
        params.srcHost = ring->data_at(offset);
        result = cuGraphMemcpyNodeSetParams(node, &params);
        if (result != CUDA_SUCCESS) {
          return result;
        }
        ++rebound_copy_nodes;
      }
    } else if (type == CU_GRAPH_NODE_TYPE_BATCH_MEM_OP) {
      CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
      result = cuGraphBatchMemOpNodeGetParams(node, &params);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      std::vector<CUstreamBatchMemOpParams> operations;
      try {
        operations.assign(params.paramArray, params.paramArray + params.count);
      } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
      bool changed = false;
      for (auto &operation : operations) {
        CUdeviceptr *address = nullptr;
        if (operation.operation == CU_STREAM_MEM_OP_WAIT_VALUE_32) {
          address = &operation.waitValue.address;
        } else if (operation.operation == CU_STREAM_MEM_OP_WRITE_VALUE_32) {
          address = &operation.writeValue.address;
        }
        size_t offset = 0;
        if (address != nullptr &&
            captured_ring->signal_offset(*address, &offset)) {
          *address = ring->signal_at(offset);
          changed = true;
        }
      }
      if (changed) {
        params.ctx = ring->context();
        params.paramArray = operations.data();
        result = cuGraphBatchMemOpNodeSetParams(node, &params);
        if (result != CUDA_SUCCESS) {
          return result;
        }
      }
    } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD) {
      CUevent event = nullptr;
      result = cuGraphEventRecordNodeGetEvent(node, &event);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      auto replacement = events.find(event);
      if (replacement != events.end()) {
        result = cuGraphEventRecordNodeSetEvent(node, replacement->second);
        if (result != CUDA_SUCCESS) {
          return result;
        }
      }
    } else if (type == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
      CUevent event = nullptr;
      result = cuGraphEventWaitNodeGetEvent(node, &event);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      auto replacement = events.find(event);
      if (replacement != events.end()) {
        result = cuGraphEventWaitNodeSetEvent(node, replacement->second);
        if (result != CUDA_SUCCESS) {
          return result;
        }
      }
    }
  }

  if (callbacks.size() != captured_execution->callback_count() ||
      rebound_copy_nodes != expected_copy_nodes) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  try {
    auto resources = std::make_shared<lupine_htod_exec_resources>();
    resources->callbacks = callbacks;
    resources->original = binding->original;
    resources->prepared = binding->prepared;
    resources->context = ring->context();
    binding->resources = std::move(resources);
    if (prepared_in_place) {
      std::lock_guard<std::mutex> lock(captured_state->callbacks_mutex);
      captured_state->execution = execution;
      captured_state->callbacks = std::move(callbacks);
    }
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  return CUDA_SUCCESS;
}

static CUresult
lupine_commit_htod_graph_exec(CUgraphExec exec,
                              const lupine_htod_graph_binding &binding) {
  if (binding.resources == nullptr) {
    return CUDA_SUCCESS;
  }
  auto resources =
      std::static_pointer_cast<lupine_htod_exec_resources>(binding.resources);
  try {
    lupine_htod_exec_resource_map().insert_or_assign(exec,
                                                     std::move(resources));
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  return CUDA_SUCCESS;
}

CUresult
lupine_prepare_graph_exec_resources(CUgraph graph,
                                    lupine_graph_resources **resources,
                                    lupine_htod_graph_binding *binding) {
  *resources = nullptr;
  (void)lupine_graph_resource_map().find(graph, *resources);
  return lupine_prepare_htod_graph_exec(graph, *resources, binding);
}

CUresult lupine_associate_graph_exec_resources(
    CUgraphExec exec, lupine_graph_resources *resources,
    const lupine_htod_graph_binding &binding) {
  if (resources == nullptr) {
    return CUDA_SUCCESS;
  }
  CUresult result = lupine_commit_htod_graph_exec(exec, binding);
  if (result == CUDA_SUCCESS) {
    lupine_graph_exec_resource_map().insert_or_assign(exec, resources);
  }
  return result;
}

CUgraphNode
lupine_original_htod_graph_node(const lupine_htod_graph_binding &binding,
                                CUgraphNode node) {
  if (node == nullptr || binding.prepared == binding.original) {
    return node;
  }
  size_t node_count = 0;
  if (cuGraphGetNodes(binding.original, nullptr, &node_count) != CUDA_SUCCESS) {
    return node;
  }
  std::vector<CUgraphNode> nodes;
  try {
    nodes.resize(node_count);
  } catch (...) {
    return node;
  }
  if (node_count != 0 && cuGraphGetNodes(binding.original, nodes.data(),
                                         &node_count) != CUDA_SUCCESS) {
    return node;
  }
  for (CUgraphNode original : nodes) {
    CUgraphNode clone = nullptr;
    if (cuGraphNodeFindInClone(&clone, original, binding.prepared) ==
            CUDA_SUCCESS &&
        clone == node) {
      return original;
    }
  }
  return node;
}

CUgraphNode lupine_htod_graph_exec_node(CUgraphExec exec, CUgraphNode node) {
  std::shared_ptr<lupine_htod_exec_resources> resources;
  if (node == nullptr ||
      !lupine_htod_exec_resource_map().find(exec, resources) ||
      resources->prepared == resources->original) {
    return node;
  }
  CUgraphNode prepared = nullptr;
  return cuGraphNodeFindInClone(&prepared, node, resources->prepared) ==
                 CUDA_SUCCESS
             ? prepared
             : node;
}

void lupine_release_htod_graph_binding(lupine_htod_graph_binding *binding) {
  if (binding->resources == nullptr && binding->prepared != nullptr &&
      binding->prepared != binding->original) {
    (void)cuGraphDestroy(binding->prepared);
  }
  *binding = {};
}

static CUresult lupine_release_htod_graph_exec(CUgraphExec exec) {
  std::shared_ptr<lupine_htod_exec_resources> resources;
  if (!lupine_htod_exec_resource_map().find(exec, resources)) {
    return CUDA_SUCCESS;
  }

  // cuGraphExecDestroy leaves in-flight launches running. Keep callback data,
  // events, and the pinned ring alive until no launch can still reference
  // them. This synchronization is confined to destroying graph execs that
  // contain pageable HtoD side effects.
  CUresult result = cuCtxPushCurrent_v2(resources->context);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  result = cuCtxSynchronize();
  CUcontext popped = nullptr;
  CUresult pop_result = cuCtxPopCurrent_v2(&popped);
  if (result == CUDA_SUCCESS) {
    result = pop_result;
  }
  if (result == CUDA_SUCCESS) {
    lupine_htod_exec_resource_map().erase(exec);
  }
  return result;
}

CUresult lupine_release_graph_exec_resources(CUgraphExec exec) {
  CUresult result = lupine_release_htod_graph_exec(exec);
  if (result == CUDA_SUCCESS) {
    lupine_graph_exec_resource_map().erase(exec);
  }
  return result;
}

static CUresult lupine_enqueue_htod_callback(
    const std::shared_ptr<lupine_htod_graph_state> &graph_state,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    lupine_htod_copy copy,
    const std::shared_ptr<lupine_htod_capture_events> &capture_events,
    size_t graph_callback, CUstream stream) {
  if (graph_state != nullptr) {
    std::shared_ptr<lupine_htod_callback_data> callback;
    try {
      callback =
          std::make_shared<lupine_htod_callback_data>(lupine_htod_callback_data{
              ring, std::move(copy), graph_state->execution, capture_events,
              graph_callback});
    } catch (...) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    auto *callback_ptr = callback.get();
    try {
      std::lock_guard<std::mutex> lock(graph_state->callbacks_mutex);
      graph_state->callbacks.push_back(std::move(callback));
    } catch (...) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return cuLaunchHostFunc(stream, lupine_htod_side_effect_callback,
                            callback_ptr);
  }

  auto callback = std::unique_ptr<lupine_htod_callback_data>(
      new (std::nothrow) lupine_htod_callback_data{ring, std::move(copy),
                                                   nullptr, nullptr, 0});
  if (callback == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  auto *callback_ptr = callback.get();
  CUresult result =
      cuLaunchHostFunc(stream, lupine_htod_side_effect_callback, callback_ptr);
  if (result == CUDA_SUCCESS) {
    (void)callback.release();
  }
  return result;
}

static CUresult lupine_enqueue_htod_fragment(
    const lupine_htod_copy &copy, const lupine_htod_fragment &fragment,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring, size_t slot,
    CUstream stream) {
  size_t ring_offset = ring->ring_offset(copy, fragment.logical_offset);
  void *host_source = ring->data(slot, ring_offset);
  if (copy.use_cuda_memcpy) {
    if (copy.destination_array == nullptr) {
      return cuMemcpyHtoDAsync_v2(copy.destination_at(fragment.logical_offset),
                                  host_source, fragment.bytes, stream);
    }
    CUDA_MEMCPY3D params = {};
    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcHost = host_source;
    params.srcPitch = fragment.width;
    params.srcHeight = fragment.rows;
    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.dstArray = copy.destination_array;
    params.dstLOD = copy.destination_lod;
    params.dstXInBytes = copy.destination_x + fragment.x;
    params.dstY = copy.destination_y + fragment.row;
    params.dstZ = copy.destination_z + fragment.slice;
    params.WidthInBytes = fragment.width;
    params.Height = fragment.rows;
    params.Depth = fragment.slices;
    return cuMemcpy3DAsync_v2(&params, stream);
  }

  lupine_smemcpy_params params = copy.smemcpy;
  params.source = ring->device_data(slot, ring_offset);
  params.logical_offset = fragment.logical_offset;
  params.bytes = fragment.bytes;
  lupine_smemcpy_launch_descriptor launch = {};
  CUresult result = lupine_smemcpy_prepare_driver_launch(&params, &launch);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (launch.use_cuda_memcpy) {
    return cuMemcpyHtoDAsync_v2(launch.params.destination, host_source,
                                launch.params.bytes, stream);
  }

  CUfunction function = nullptr;
  result = ring->smemcpy_function(launch, &function);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  void *arguments[] = {&launch.params};
  return cuLaunchKernel(function, launch.blocks, 1, 1, launch.threads, 1, 1, 0,
                        stream, arguments, nullptr);
}

static CUresult lupine_prepare_htod_copy(
    const lupine_htod_copy &copy,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring) {
  if (copy.use_cuda_memcpy) {
    return CUDA_SUCCESS;
  }
  size_t offset = 0;
  for (size_t index = 0; offset < copy.bytes; ++index) {
    auto fragment =
        copy.fragment(offset, lupine_htod_side_effect_ring::slot_bytes);
    size_t slot = index % lupine_htod_side_effect_ring::slot_count;
    lupine_smemcpy_params params = copy.smemcpy;
    params.source = ring->device_data(
        slot, ring->ring_offset(copy, fragment.logical_offset));
    params.logical_offset = fragment.logical_offset;
    params.bytes = fragment.bytes;
    lupine_smemcpy_launch_descriptor launch = {};
    CUresult result = lupine_smemcpy_prepare_driver_launch(&params, &launch);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    if (!launch.use_cuda_memcpy) {
      CUfunction function = nullptr;
      result = ring->smemcpy_function(launch, &function);
      if (result != CUDA_SUCCESS) {
        return result;
      }
    }
    offset += fragment.bytes;
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_enqueue_client_htod_copy(lupine_staging_state &state,
                                                lupine_htod_copy copy,
                                                CUcontext context,
                                                CUstream stream,
                                                bool blocking) {
  if (copy.bytes == 0) {
    return CUDA_SUCCESS;
  }

  CUresult result = CUDA_SUCCESS;
  auto ring = lupine_prepare_htod_side_effect_ring(state, context, result);
  if (ring == nullptr) {
    return result;
  }
  result = lupine_prepare_htod_copy(copy, ring);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  auto *resources = lupine_captured_stream_resources(stream);
  std::shared_ptr<lupine_htod_graph_state> graph_state;
  std::shared_ptr<lupine_htod_capture_events> capture_events;
  size_t graph_callback = 0;
  if (resources != nullptr) {
    graph_state = lupine_htod_graph_state_for(resources, ring);
    if (graph_state == nullptr || graph_state->execution->ring() != ring) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = lupine_make_htod_capture_events(context, capture_events);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    graph_callback = graph_state->execution->reserve_callback();
    result = cuEventRecord(capture_events->fork, stream);
    if (result == CUDA_SUCCESS) {
      result =
          cuStreamWaitEvent(ring->transfer_stream(), capture_events->fork, 0);
    }
    if (result != CUDA_SUCCESS) {
      return result;
    }
  } else {
    ring->acquire_execution();
  }

  size_t offset = 0;
  for (size_t index = 0; offset < copy.bytes; ++index) {
    auto fragment =
        copy.fragment(offset, lupine_htod_side_effect_ring::slot_bytes);
    size_t slot = index % lupine_htod_side_effect_ring::slot_count;
    result = cuStreamWaitValue32_v2(
        ring->transfer_stream(), ring->device_signal(slot),
        lupine_htod_side_effect_ring::slot_ready, CU_STREAM_WAIT_VALUE_EQ);
    if (result == CUDA_SUCCESS) {
      result = lupine_enqueue_htod_fragment(copy, fragment, ring, slot,
                                            ring->transfer_stream());
    }
    if (result == CUDA_SUCCESS) {
      result = cuStreamWriteValue32_v2(
          ring->transfer_stream(), ring->device_signal(slot),
          lupine_htod_side_effect_ring::slot_free, 0);
    }
    if (result != CUDA_SUCCESS) {
      if (resources == nullptr) {
        ring->release_execution();
      }
      return result;
    }
    offset += fragment.bytes;
  }

  if (resources != nullptr) {
    result = cuEventRecord(capture_events->join, ring->transfer_stream());
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  result = lupine_enqueue_htod_callback(graph_state, ring, std::move(copy),
                                        capture_events, graph_callback, stream);
  if (result != CUDA_SUCCESS) {
    if (resources == nullptr) {
      ring->release_execution();
    }
    return result;
  }
  if (resources != nullptr) {
    result = cuStreamWaitEvent(stream, capture_events->join, 0);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  // CUDA may stage pageable HtoD input before an asynchronous API returns.
  // Preserve that source-lifetime guarantee: outside capture, the callback
  // must finish pulling the client buffer before the caller can reuse it.
  return blocking || resources == nullptr ? cuStreamSynchronize(stream)
                                          : CUDA_SUCCESS;
}

static CUresult lupine_copy_client_host_to_device(conn_t *conn, CUstream stream,
                                                  bool blocking,
                                                  lupine_htod_copy copy) {
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

  if (result == CUDA_SUCCESS) {
    result = lupine_enqueue_client_htod_copy(*state, std::move(copy), context,
                                             stream, blocking);
  }
  return result;
}

static lupine_htod_copy lupine_make_linear_htod_copy(CUdeviceptr destination,
                                                     const void *source,
                                                     size_t bytes) {
  lupine_htod_copy copy = {};
  copy.source = source;
  copy.source_row_stride = bytes;
  copy.source_slice_stride = bytes;
  copy.slices = 1;
  copy.bytes = bytes;
  copy.use_cuda_memcpy = true;
  copy.smemcpy.destination = destination;
  copy.smemcpy.width = bytes;
  copy.smemcpy.rows = 1;
  copy.smemcpy.destination_row_stride = bytes;
  copy.smemcpy.destination_slice_stride = bytes;
  return copy;
}

static lupine_htod_copy
lupine_make_2d_htod_copy(const CUDA_MEMCPY2D &original) {
  lupine_htod_copy copy = {};
  copy.bytes = original.WidthInBytes * original.Height;
  if (copy.bytes == 0) {
    return copy;
  }
  copy.source = static_cast<const unsigned char *>(original.srcHost) +
                original.srcY * original.srcPitch + original.srcXInBytes;
  copy.source_row_stride = original.srcPitch;
  copy.source_slice_stride = original.srcPitch * original.Height;
  copy.slices = 1;
  copy.smemcpy.width = original.WidthInBytes;
  copy.smemcpy.rows = original.Height;
  if (original.dstMemoryType == CU_MEMORYTYPE_ARRAY) {
    copy.use_cuda_memcpy = true;
    copy.destination_array = original.dstArray;
    copy.destination_x = original.dstXInBytes;
    copy.destination_y = original.dstY;
    copy.smemcpy.destination_row_stride = original.WidthInBytes;
    copy.smemcpy.destination_slice_stride = copy.bytes;
  } else {
    copy.smemcpy.destination = original.dstDevice +
                               original.dstY * original.dstPitch +
                               original.dstXInBytes;
    copy.smemcpy.destination_row_stride = original.dstPitch;
    copy.smemcpy.destination_slice_stride = original.dstPitch * original.Height;
  }
  return copy;
}

static lupine_htod_copy
lupine_make_3d_htod_copy(const CUDA_MEMCPY3D &original) {
  lupine_htod_copy copy = {};
  copy.bytes = original.WidthInBytes * original.Height * original.Depth;
  if (copy.bytes == 0) {
    return copy;
  }
  const size_t source_slice_stride = original.srcHeight * original.srcPitch;
  copy.source = static_cast<const unsigned char *>(original.srcHost) +
                original.srcZ * source_slice_stride +
                original.srcY * original.srcPitch + original.srcXInBytes;
  copy.source_row_stride = original.srcPitch;
  copy.source_slice_stride = source_slice_stride;
  copy.slices = original.Depth;
  copy.smemcpy.width = original.WidthInBytes;
  copy.smemcpy.rows = original.Height;
  if (original.dstMemoryType == CU_MEMORYTYPE_ARRAY) {
    copy.use_cuda_memcpy = true;
    copy.destination_array = original.dstArray;
    copy.destination_lod = original.dstLOD;
    copy.destination_x = original.dstXInBytes;
    copy.destination_y = original.dstY;
    copy.destination_z = original.dstZ;
    copy.smemcpy.destination_row_stride = original.WidthInBytes;
    copy.smemcpy.destination_slice_stride =
        original.WidthInBytes * original.Height;
  } else {
    copy.smemcpy.destination =
        original.dstDevice +
        original.dstZ * original.dstHeight * original.dstPitch +
        original.dstY * original.dstPitch + original.dstXInBytes;
    copy.smemcpy.destination_row_stride = original.dstPitch;
    copy.smemcpy.destination_slice_stride =
        original.dstPitch * original.dstHeight;
  }
  return copy;
}

static void lupine_server_forget_context_metadata(lupine_staging_state &state,
                                                  CUcontext context) {
  state.created_contexts.erase(context);
  state.htod_rings.erase(context);
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
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS || context == nullptr) {
    return result == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : result;
  }
  auto ring = lupine_prepare_htod_side_effect_ring(*state, context, result);
  return ring == nullptr ? result : ring->synchronize();
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
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->teardown_devices.insert(device);
    state->condition.wait(lock,
                          [&] { return !state->staging_operation_active; });
    auto context = state->primary_contexts.find(device);
    if (context != state->primary_contexts.end()) {
      state->teardown_contexts.insert(context->second);
      auto existing = state->htod_rings.find(context->second);
      if (existing != state->htod_rings.end()) {
        ring = std::move(existing->second);
        state->htod_rings.erase(existing);
      }
    }
  }
  ring.reset();
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
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->teardown_contexts.insert(context);
    state->condition.wait(lock,
                          [&] { return !state->staging_operation_active; });
    auto it = state->htod_rings.find(context);
    if (it != state->htod_rings.end()) {
      ring = std::move(it->second);
      state->htod_rings.erase(it);
    }
  }
  // Streams, modules, and mapped registrations belong to the live context.
  // Drop the registry's ownership while the caller still has it current.
  ring.reset();
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
  bool is_server_authoritative = false;
  size_t bytes = 0;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &destination, sizeof(destination)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_read(conn, &source, sizeof(source)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (bytes != 0 && is_server_authoritative) {
    result =
        cuMemcpy(destination, reinterpret_cast<CUdeviceptr>(source), bytes);
  } else if (bytes != 0) {
    result = lupine_copy_client_host_to_device(
        conn, CU_STREAM_LEGACY, true,
        lupine_make_linear_htod_copy(destination, source, bytes));
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  CUDA_MEMCPY3D copy = {};
  if (rpc_read(conn, &direction, sizeof(direction)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    bool is_server_authoritative = false;
    if (rpc_read(conn, &is_server_authoritative,
                 sizeof(is_server_authoritative)) < 0 ||
        rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy3D_v2(&copy)
            : lupine_copy_client_host_to_device(conn, CU_STREAM_LEGACY, true,
                                                lupine_make_3d_htod_copy(copy));
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  CUDA_MEMCPY3D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &direction, sizeof(direction)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    bool is_server_authoritative = false;
    if (rpc_read(conn, &is_server_authoritative,
                 sizeof(is_server_authoritative)) < 0 ||
        rpc_read(conn, &copy, sizeof(copy)) < 0 ||
        rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy3DAsync_v2(&copy, stream)
            : lupine_copy_client_host_to_device(conn, stream, false,
                                                lupine_make_3d_htod_copy(copy));
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
    if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
        rpc_read(conn, &stream, sizeof(stream)) < 0) {
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
  uint8_t direction = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                          ? LUPINE_COPY_DIRECTION_HTOD
                          : (copy.dstMemoryType == CU_MEMORYTYPE_HOST
                                 ? LUPINE_COPY_DIRECTION_DTOH
                                 : LUPINE_COPY_DIRECTION_DTOD);
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
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
  case LUPINE_COPY_DIRECTION_DTOH: {
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
  uint8_t direction = copy.srcMemoryType == CU_MEMORYTYPE_HOST
                          ? LUPINE_COPY_DIRECTION_HTOD
                          : (copy.dstMemoryType == CU_MEMORYTYPE_HOST
                                 ? LUPINE_COPY_DIRECTION_DTOH
                                 : LUPINE_COPY_DIRECTION_DTOD);
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
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
  case LUPINE_COPY_DIRECTION_DTOH: {
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  CUDA_MEMCPY2D copy = {};
  if (rpc_read(conn, &direction, sizeof(direction)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    bool is_server_authoritative = false;
    if (rpc_read(conn, &is_server_authoritative,
                 sizeof(is_server_authoritative)) < 0 ||
        rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy2D_v2(&copy)
            : lupine_copy_client_host_to_device(conn, CU_STREAM_LEGACY, true,
                                                lupine_make_2d_htod_copy(copy));
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  CUDA_MEMCPY2D copy = {};
  if (rpc_read(conn, &direction, sizeof(direction)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    bool is_server_authoritative = false;
    if (rpc_read(conn, &is_server_authoritative,
                 sizeof(is_server_authoritative)) < 0 ||
        rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy2DUnaligned_v2(&copy)
            : lupine_copy_client_host_to_device(conn, CU_STREAM_LEGACY, true,
                                                lupine_make_2d_htod_copy(copy));
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  CUDA_MEMCPY2D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &direction, sizeof(direction)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    bool is_server_authoritative = false;
    if (rpc_read(conn, &is_server_authoritative,
                 sizeof(is_server_authoritative)) < 0 ||
        rpc_read(conn, &copy, sizeof(copy)) < 0 ||
        rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy2DAsync_v2(&copy, stream)
            : lupine_copy_client_host_to_device(conn, stream, false,
                                                lupine_make_2d_htod_copy(copy));
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
    if (rpc_read(conn, &copy, sizeof(copy)) < 0) {
      return -1;
    }
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
    if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
        rpc_read(conn, &stream, sizeof(stream)) < 0) {
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
  bool is_server_authoritative = false;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
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

  if (byteCount != 0 && is_server_authoritative) {
    result = cuMemcpyAsync(dstDevice, reinterpret_cast<CUdeviceptr>(srcHost),
                           byteCount, stream);
  } else if (byteCount != 0) {
    result = lupine_copy_client_host_to_device(
        conn, stream, false,
        lupine_make_linear_htod_copy(dstDevice, srcHost, byteCount));
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
