// Server-side copy machinery: the staging pools the device copies stage
// through, the lifetime rules that tie those pools to a context, and the
// chunked host-to-device and device-to-host loops. HTTP content compression
// is handled transparently by the transport.
#include <cuda.h>

#include "codegen/gen_rpc_ids.h"
#include "cuda_server.h"
#include "cuda_server_memcpy.h"
#ifdef LUPINE_HAVE_SMEMCPY
#include "ops/smemcpy.h"
#endif
#include "rpc.h"

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

#ifdef LUPINE_HAVE_SMEMCPY
static CUresult lupine_runtime_result(cudaError_t result) {
  switch (result) {
  case cudaSuccess:
    return CUDA_SUCCESS;
  case cudaErrorInvalidConfiguration:
    return CUDA_ERROR_INVALID_VALUE;
  case cudaErrorInvalidDeviceFunction:
    return CUDA_ERROR_INVALID_IMAGE;
  default:
    return static_cast<CUresult>(result);
  }
}
#endif

enum class lupine_htod_copy_kind { linear, two_dimensional, three_dimensional };

struct lupine_htod_chunk {
  const void *source = nullptr;
  size_t width = 0;
  size_t rows = 0;
  size_t row_stride = 0;
  size_t slices = 0;
  size_t slice_stride = 0;
  size_t bytes = 0;
  size_t ring_offset = 0;
  lupine_htod_copy_kind kind = lupine_htod_copy_kind::linear;
#ifdef LUPINE_HAVE_SMEMCPY
  CUfunction smemcpy_function = nullptr;
  lupine_smemcpy_launch smemcpy = {};
#endif
  CUdeviceptr linear_destination = 0;
  CUDA_MEMCPY2D copy_2d = {};
  CUDA_MEMCPY3D copy_3d = {};
};

struct lupine_htod_plan {
  std::vector<lupine_htod_chunk> chunks;
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
      CUcontext popped = nullptr;
      (void)cuCtxPopCurrent_v2(&popped);
    }
    if (signals_ != nullptr) {
      (void)cuMemFreeHost(signals_);
    }
    if (storage_ != nullptr) {
      (void)cuMemHostUnregister(storage_);
      std::free(storage_);
    }
  }

  CUresult prepare() {
    std::lock_guard<std::mutex> lock(prepare_mutex_);
    if (storage_ != nullptr) {
      return CUDA_SUCCESS;
    }

    void *storage = nullptr;
    if (posix_memalign(&storage, 4096, slot_count * slot_stride) != 0) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult result = cuMemHostRegister(storage, slot_count * slot_stride,
                                        CU_MEMHOSTREGISTER_PORTABLE |
                                            CU_MEMHOSTREGISTER_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
      std::free(storage);
      return result;
    }
    CUdeviceptr device_storage = 0;
    result = cuMemHostGetDevicePointer_v2(&device_storage, storage, 0);
    if (result != CUDA_SUCCESS) {
      (void)cuMemHostUnregister(storage);
      std::free(storage);
      return result;
    }
    slot_signal *signals = nullptr;
    result = cuMemHostAlloc(
        reinterpret_cast<void **>(&signals), slot_count * sizeof(slot_signal),
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
      (void)cuMemHostUnregister(storage);
      std::free(storage);
      return result;
    }
    CUdeviceptr device_signals = 0;
    result = cuMemHostGetDevicePointer_v2(&device_signals, signals, 0);
    if (result != CUDA_SUCCESS) {
      (void)cuMemFreeHost(signals);
      (void)cuMemHostUnregister(storage);
      std::free(storage);
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
      std::free(storage);
      return result;
    }
#ifdef LUPINE_HAVE_SMEMCPY
    cudaFunction_t smemcpy_function = nullptr;
    // Load the smemcpy module before any wait-value nodes can reach this
    // stream. A first-use runtime launch may synchronize module loading with
    // earlier stream work, which would deadlock if that work is waiting for
    // the callback to publish a ring slot.
    cudaError_t runtime_result =
        cudaGetFuncBySymbol(&smemcpy_function, lupine_smemcpy_kernel());
    if (runtime_result != cudaSuccess) {
      (void)cuStreamDestroy_v2(transfer_stream);
      (void)cuMemFreeHost(signals);
      (void)cuMemHostUnregister(storage);
      std::free(storage);
      return lupine_runtime_result(runtime_result);
    }
#endif
    storage_ = storage;
    device_storage_ = device_storage;
    signals_ = signals;
    device_signals_ = device_signals;
    transfer_stream_ = transfer_stream;
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

  int fetch(const lupine_htod_plan &plan) {
    uint64_t count = plan.chunks.size();
    if (rpc_write_start_request(conn_, LUPINE_SIDE_EFFECT_READ_HOST_MEMORY) <
            0 ||
        rpc_write(conn_, &count, sizeof(count)) < 0) {
      return -1;
    }
    for (const auto &chunk : plan.chunks) {
      if (rpc_write(conn_, &chunk.source, sizeof(chunk.source)) < 0 ||
          rpc_write(conn_, &chunk.width, sizeof(chunk.width)) < 0 ||
          rpc_write(conn_, &chunk.rows, sizeof(chunk.rows)) < 0 ||
          rpc_write(conn_, &chunk.row_stride, sizeof(chunk.row_stride)) < 0 ||
          rpc_write(conn_, &chunk.slices, sizeof(chunk.slices)) < 0 ||
          rpc_write(conn_, &chunk.slice_stride, sizeof(chunk.slice_stride)) <
              0) {
        return -1;
      }
    }
    if (rpc_wait_for_response(conn_) < 0) {
      return -1;
    }

    for (size_t index = 0; index < plan.chunks.size(); ++index) {
      size_t slot = index % slot_count;
      while (signals_[slot].value.load(std::memory_order_acquire) !=
             slot_free) {
        std::this_thread::yield();
      }
      if (rpc_read(conn_, data(slot, plan.chunks[index].ring_offset),
                   plan.chunks[index].bytes) < 0) {
        return -1;
      }
      signals_[slot].value.store(slot_ready, std::memory_order_release);
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
  struct alignas(64) slot_signal {
    std::atomic<uint32_t> value{slot_free};
  };

  static_assert(std::atomic<uint32_t>::is_always_lock_free);

  conn_t *conn_ = nullptr;
  CUcontext context_ = nullptr;
  std::mutex prepare_mutex_;
  std::mutex execution_mutex_;
  std::condition_variable execution_condition_;
  bool execution_active_ = false;
  void *storage_ = nullptr;
  CUdeviceptr device_storage_ = 0;
  slot_signal *signals_ = nullptr;
  CUdeviceptr device_signals_ = 0;
  CUstream transfer_stream_ = nullptr;
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
      std::shared_ptr<lupine_htod_side_effect_ring> ring, size_t plan_count = 0)
      : ring_(std::move(ring)), plan_count_(plan_count) {}

  size_t reserve_plan() {
    std::lock_guard<std::mutex> lock(mutex_);
    return plan_count_++;
  }

  void wait_for_turn(size_t plan) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return running_plan_ == plan; });
  }

  void finish_turn() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_plan_ = (running_plan_ + 1) % plan_count_;
    }
    condition_.notify_all();
  }

  const std::shared_ptr<lupine_htod_side_effect_ring> &ring() const {
    return ring_;
  }

  size_t plan_count() const { return plan_count_; }

private:
  std::shared_ptr<lupine_htod_side_effect_ring> ring_;
  std::mutex mutex_;
  std::condition_variable condition_;
  size_t plan_count_ = 0;
  size_t running_plan_ = 0;
};

using lupine_htod_graph_registry =
    libcuckoo::cuckoohash_map<lupine_graph_resources *,
                              std::shared_ptr<lupine_htod_graph_execution>>;

static lupine_htod_graph_registry &lupine_htod_graph_executions() {
  static auto *executions = new lupine_htod_graph_registry();
  return *executions;
}

static std::shared_ptr<lupine_htod_graph_execution>
lupine_htod_graph_execution_for(
    lupine_graph_resources *resources,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring) {
  std::shared_ptr<lupine_htod_graph_execution> execution;
  if (lupine_htod_graph_executions().find(resources, execution)) {
    return execution;
  }
  try {
    auto candidate = std::make_shared<lupine_htod_graph_execution>(ring);
    execution = candidate;
    lupine_htod_graph_executions().upsert(
        resources,
        [&execution](std::shared_ptr<lupine_htod_graph_execution> &existing,
                     libcuckoo::UpsertContext) { execution = existing; },
        std::move(candidate));
  } catch (...) {
    return nullptr;
  }
  return execution;
}

struct lupine_htod_callback_data {
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  std::shared_ptr<lupine_htod_plan> plan;
  std::shared_ptr<lupine_htod_graph_execution> graph_execution;
  std::shared_ptr<lupine_htod_capture_events> capture_events;
  size_t graph_plan = 0;
};

using lupine_htod_exec_resources =
    std::vector<std::shared_ptr<lupine_htod_callback_data>>;

static libcuckoo::cuckoohash_map<CUgraphExec, lupine_htod_exec_resources> &
lupine_htod_exec_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraphExec, lupine_htod_exec_resources>();
  return *resources;
}

static void CUDA_CB lupine_htod_side_effect_callback(void *opaque) {
  auto *data = static_cast<lupine_htod_callback_data *>(opaque);
  if (data == nullptr) {
    return;
  }
  if (data->graph_execution != nullptr) {
    data->graph_execution->wait_for_turn(data->graph_plan);
  }
  if (data->ring->fetch(*data->plan) < 0) {
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

CUresult lupine_prepare_htod_graph_exec(CUgraph graph,
                                        lupine_graph_resources *resources,
                                        lupine_htod_graph_binding *binding) {
  *binding = {};
  binding->original = graph;
  binding->prepared = graph;
  std::shared_ptr<lupine_htod_graph_execution> captured_execution;
  if (resources == nullptr ||
      !lupine_htod_graph_executions().find(resources, captured_execution)) {
    return CUDA_SUCCESS;
  }

  CUresult result = cuGraphClone(&binding->prepared, graph);
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
        ring, captured_execution->plan_count());
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

  lupine_htod_exec_resources callbacks;
  std::unordered_map<CUevent, CUevent> events;
#ifdef LUPINE_HAVE_SMEMCPY
  std::unordered_set<CUfunction> smemcpy_functions;
  size_t expected_smemcpy_nodes = 0;
  size_t rebound_smemcpy_nodes = 0;
#endif
  try {
    callbacks.reserve(captured_execution->plan_count());
    events.reserve(captured_execution->plan_count() * 2);
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
          lupine_htod_callback_data{ring, captured->plan, execution,
                                    callback_events, captured->graph_plan});
      events.emplace(captured->capture_events->fork, callback_events->fork);
      events.emplace(captured->capture_events->join, callback_events->join);
#ifdef LUPINE_HAVE_SMEMCPY
      for (const auto &chunk : captured->plan->chunks) {
        if (chunk.smemcpy_function != nullptr) {
          smemcpy_functions.insert(chunk.smemcpy_function);
          ++expected_smemcpy_nodes;
        }
      }
#endif
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
#ifdef LUPINE_HAVE_SMEMCPY
      CUDA_KERNEL_NODE_PARAMS params = {};
      result = cuGraphKernelNodeGetParams(node, &params);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      if (params.func == nullptr || smemcpy_functions.count(params.func) == 0) {
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
      ++rebound_smemcpy_nodes;
#endif
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

  if (callbacks.size() != captured_execution->plan_count()
#ifdef LUPINE_HAVE_SMEMCPY
      || rebound_smemcpy_nodes != expected_smemcpy_nodes
#endif
  ) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  try {
    binding->resources =
        std::make_shared<lupine_htod_exec_resources>(std::move(callbacks));
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  return CUDA_SUCCESS;
}

CUresult
lupine_commit_htod_graph_exec(CUgraphExec exec,
                              const lupine_htod_graph_binding &binding) {
  if (binding.resources == nullptr) {
    return CUDA_SUCCESS;
  }
  auto callbacks =
      std::static_pointer_cast<lupine_htod_exec_resources>(binding.resources);
  try {
    lupine_htod_exec_resource_map().insert_or_assign(exec, *callbacks);
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  return CUDA_SUCCESS;
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

void lupine_release_htod_graph_binding(lupine_htod_graph_binding *binding) {
  if (binding->prepared != nullptr && binding->prepared != binding->original) {
    (void)cuGraphDestroy(binding->prepared);
  }
  *binding = {};
}

CUresult lupine_release_htod_graph_exec(CUgraphExec exec) {
  lupine_htod_exec_resources resources;
  if (!lupine_htod_exec_resource_map().find(exec, resources)) {
    return CUDA_SUCCESS;
  }

  // cuGraphExecDestroy leaves in-flight launches running. Keep callback data,
  // events, and the pinned ring alive until no launch can still reference
  // them. This synchronization is confined to destroying graph execs that
  // contain pageable HtoD side effects.
  CUresult result = cuCtxPushCurrent_v2(resources.front()->ring->context());
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

static CUresult lupine_enqueue_htod_callback(
    lupine_graph_resources *resources,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    std::shared_ptr<lupine_htod_plan> plan,
    const std::shared_ptr<lupine_htod_graph_execution> &graph_execution,
    const std::shared_ptr<lupine_htod_capture_events> &capture_events,
    size_t graph_plan, CUstream stream) {
  if (resources != nullptr) {
    std::shared_ptr<lupine_htod_callback_data> callback;
    try {
      callback = std::make_shared<lupine_htod_callback_data>(
          lupine_htod_callback_data{ring, std::move(plan), graph_execution,
                                    capture_events, graph_plan});
    } catch (...) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    auto *callback_ptr = callback.get();
    if (!lupine_graph_retain_resource(resources, std::move(callback))) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return cuLaunchHostFunc(stream, lupine_htod_side_effect_callback,
                            callback_ptr);
  }

  auto callback = std::unique_ptr<lupine_htod_callback_data>(
      new (std::nothrow) lupine_htod_callback_data{ring, std::move(plan),
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

#ifdef LUPINE_HAVE_SMEMCPY
static bool lupine_htod_smemcpy_params(const lupine_htod_chunk &chunk,
                                       CUdeviceptr source,
                                       lupine_smemcpy_params *params) {
  *params = {};
  params->source = source;
  params->bytes = chunk.bytes;
  switch (chunk.kind) {
  case lupine_htod_copy_kind::linear:
    params->destination = chunk.linear_destination;
    params->width = chunk.bytes;
    params->rows = 1;
    params->destination_row_stride = chunk.bytes;
    params->destination_slice_stride = chunk.bytes;
    return true;
  case lupine_htod_copy_kind::two_dimensional: {
    const auto &copy = chunk.copy_2d;
    if (copy.dstMemoryType != CU_MEMORYTYPE_DEVICE &&
        copy.dstMemoryType != CU_MEMORYTYPE_UNIFIED) {
      return false;
    }
    params->destination =
        copy.dstDevice + copy.dstY * copy.dstPitch + copy.dstXInBytes;
    params->width = copy.WidthInBytes;
    params->rows = copy.Height;
    params->destination_row_stride = copy.dstPitch;
    params->destination_slice_stride = copy.dstPitch * copy.Height;
    return true;
  }
  case lupine_htod_copy_kind::three_dimensional: {
    const auto &copy = chunk.copy_3d;
    if (copy.dstMemoryType != CU_MEMORYTYPE_DEVICE &&
        copy.dstMemoryType != CU_MEMORYTYPE_UNIFIED) {
      return false;
    }
    params->destination = copy.dstDevice +
                          copy.dstZ * copy.dstHeight * copy.dstPitch +
                          copy.dstY * copy.dstPitch + copy.dstXInBytes;
    params->width = copy.WidthInBytes;
    params->rows = copy.Height;
    params->destination_row_stride = copy.dstPitch;
    params->destination_slice_stride = copy.dstPitch * copy.dstHeight;
    return true;
  }
  }
  return false;
}

static CUresult lupine_prepare_htod_chunk(
    lupine_htod_chunk &chunk,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring, size_t slot) {
  lupine_smemcpy_params params = {};
  if (!lupine_htod_smemcpy_params(
          chunk, ring->device_data(slot, chunk.ring_offset), &params)) {
    return CUDA_SUCCESS;
  }
  cudaError_t result = lupine_smemcpy_prepare_launch(&params, &chunk.smemcpy);
  if (result != cudaSuccess) {
    return lupine_runtime_result(result);
  }
  cudaFunction_t function = nullptr;
  result = cudaGetFuncBySymbol(&function, chunk.smemcpy.kernel);
  if (result != cudaSuccess) {
    return lupine_runtime_result(result);
  }
  chunk.smemcpy_function = reinterpret_cast<CUfunction>(function);
  return CUDA_SUCCESS;
}

static CUresult lupine_enqueue_smemcpy(lupine_htod_chunk &chunk,
                                       CUstream stream) {
  void *arguments[] = {&chunk.smemcpy.params};
  cudaError_t result =
      cudaLaunchKernel(chunk.smemcpy.kernel, dim3(chunk.smemcpy.blocks),
                       dim3(chunk.smemcpy.threads), arguments, 0,
                       reinterpret_cast<cudaStream_t>(stream));
  return lupine_runtime_result(result);
}
#else
static CUresult
lupine_prepare_htod_chunk(lupine_htod_chunk &,
                          const std::shared_ptr<lupine_htod_side_effect_ring> &,
                          size_t) {
  return CUDA_SUCCESS;
}
#endif

static CUresult lupine_enqueue_htod_chunk(
    lupine_htod_chunk &chunk,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring, size_t slot,
    CUstream stream) {
  void *host_source = ring->data(slot, chunk.ring_offset);
#ifdef LUPINE_HAVE_SMEMCPY
  if (chunk.smemcpy.kernel != nullptr) {
    return lupine_enqueue_smemcpy(chunk, stream);
  }
#endif
  switch (chunk.kind) {
  case lupine_htod_copy_kind::linear:
    return cuMemcpyHtoDAsync_v2(chunk.linear_destination, host_source,
                                chunk.bytes, stream);
  case lupine_htod_copy_kind::two_dimensional: {
    CUDA_MEMCPY2D copy = chunk.copy_2d;
    copy.srcHost = host_source;
    return cuMemcpy2DAsync_v2(&copy, stream);
  }
  case lupine_htod_copy_kind::three_dimensional: {
    CUDA_MEMCPY3D copy = chunk.copy_3d;
    copy.srcHost = host_source;
    return cuMemcpy3DAsync_v2(&copy, stream);
  }
  }
  return CUDA_ERROR_INVALID_VALUE;
}

static CUresult lupine_enqueue_client_htod_plan(lupine_staging_state &state,
                                                lupine_htod_plan plan,
                                                CUcontext context,
                                                CUstream stream,
                                                bool blocking) {
  if (plan.chunks.empty()) {
    return CUDA_SUCCESS;
  }

  CUresult result = CUDA_SUCCESS;
  auto ring = lupine_prepare_htod_side_effect_ring(state, context, result);
  if (ring == nullptr) {
    return result;
  }

  auto *resources = lupine_captured_stream_resources(stream);
  std::shared_ptr<lupine_htod_plan> plan_resource;
  try {
    plan_resource = std::make_shared<lupine_htod_plan>(std::move(plan));
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  for (size_t index = 0; index < plan_resource->chunks.size(); ++index) {
    result = lupine_prepare_htod_chunk(
        plan_resource->chunks[index], ring,
        index % lupine_htod_side_effect_ring::slot_count);
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }

  std::shared_ptr<lupine_htod_graph_execution> graph_execution;
  std::shared_ptr<lupine_htod_capture_events> capture_events;
  size_t graph_plan = 0;
  if (resources != nullptr) {
    graph_execution = lupine_htod_graph_execution_for(resources, ring);
    if (graph_execution == nullptr || graph_execution->ring() != ring) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = lupine_make_htod_capture_events(context, capture_events);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    graph_plan = graph_execution->reserve_plan();
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

  for (size_t index = 0; index < plan_resource->chunks.size(); ++index) {
    size_t slot = index % lupine_htod_side_effect_ring::slot_count;
    result = cuStreamWaitValue32_v2(
        ring->transfer_stream(), ring->device_signal(slot),
        lupine_htod_side_effect_ring::slot_ready, CU_STREAM_WAIT_VALUE_EQ);
    if (result == CUDA_SUCCESS) {
      result = lupine_enqueue_htod_chunk(plan_resource->chunks[index], ring,
                                         slot, ring->transfer_stream());
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
  }

  if (resources != nullptr) {
    result = cuEventRecord(capture_events->join, ring->transfer_stream());
    if (result != CUDA_SUCCESS) {
      return result;
    }
  }
  result = lupine_enqueue_htod_callback(resources, ring, plan_resource,
                                        graph_execution, capture_events,
                                        graph_plan, stream);
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

  return blocking ? cuStreamSynchronize(stream) : CUDA_SUCCESS;
}

template <typename MakePlan>
static CUresult lupine_copy_client_host_to_device(conn_t *conn, CUstream stream,
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
    result = lupine_enqueue_client_htod_plan(*state, std::move(plan), context,
                                             stream, blocking);
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
    chunk.source = source_bytes == nullptr ? nullptr : source_bytes + offset;
    chunk.width = chunk_bytes;
    chunk.rows = 1;
    chunk.row_stride = chunk_bytes;
    chunk.slices = 1;
    chunk.slice_stride = chunk_bytes;
    chunk.bytes = chunk_bytes;
    chunk.linear_destination = destination + offset;
    chunk.ring_offset = chunk.linear_destination & 15;
    plan.chunks.push_back(chunk);
    offset += chunk_bytes;
  }
  return plan;
}

static lupine_htod_plan
lupine_make_2d_htod_plan(const CUDA_MEMCPY2D &original) {
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
    chunk.source = chunk_source;
    chunk.width = chunk_width;
    chunk.rows = chunk_rows;
    chunk.row_stride = source_stride;
    chunk.slices = 1;
    chunk.slice_stride = chunk_width * chunk_rows;
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
    if (chunk.copy_2d.dstMemoryType == CU_MEMORYTYPE_DEVICE ||
        chunk.copy_2d.dstMemoryType == CU_MEMORYTYPE_UNIFIED) {
      CUdeviceptr first_destination =
          chunk.copy_2d.dstDevice +
          chunk.copy_2d.dstY * chunk.copy_2d.dstPitch +
          chunk.copy_2d.dstXInBytes;
      chunk.ring_offset = first_destination & 15;
    }
    plan.chunks.push_back(chunk);
  };

  if (width <= lupine_htod_side_effect_ring::slot_bytes) {
    size_t rows_per_chunk = lupine_htod_side_effect_ring::slot_bytes / width;
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
    chunk.source = chunk_source;
    chunk.width = chunk_width;
    chunk.rows = chunk_rows;
    chunk.row_stride = source_row_stride;
    chunk.slices = chunk_slices;
    chunk.slice_stride = source_slice_pitch;
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
    if (chunk.copy_3d.dstMemoryType == CU_MEMORYTYPE_DEVICE ||
        chunk.copy_3d.dstMemoryType == CU_MEMORYTYPE_UNIFIED) {
      CUdeviceptr first_destination =
          chunk.copy_3d.dstDevice +
          chunk.copy_3d.dstZ * chunk.copy_3d.dstHeight *
              chunk.copy_3d.dstPitch +
          chunk.copy_3d.dstY * chunk.copy_3d.dstPitch +
          chunk.copy_3d.dstXInBytes;
      chunk.ring_offset = first_destination & 15;
    }
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
  std::unique_lock<std::mutex> lock(state->mutex);
  state->teardown_devices.insert(device);
  state->condition.wait(lock, [&] { return !state->staging_operation_active; });
  auto it = state->primary_contexts.find(device);
  if (it != state->primary_contexts.end()) {
    state->teardown_contexts.insert(it->second);
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
    result =
        lupine_copy_client_host_to_device(conn, CU_STREAM_LEGACY, true, [&] {
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  bool is_server_authoritative = false;
  CUDA_MEMCPY3D copy = {};
  if (rpc_read(conn, &direction, sizeof(direction)) < 0 ||
      rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = is_server_authoritative
                          ? cuMemcpy3D_v2(&copy)
                          : lupine_copy_client_host_to_device(
                                conn, CU_STREAM_LEGACY, true,
                                [&] { return lupine_make_3d_htod_plan(copy); });
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  bool is_server_authoritative = false;
  CUDA_MEMCPY3D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &direction, sizeof(direction)) < 0 ||
      rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy3DAsync_v2(&copy, stream)
            : lupine_copy_client_host_to_device(conn, stream, false, [&] {
                return lupine_make_3d_htod_plan(copy);
              });
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
  bool is_server_authoritative = false;
  CUDA_MEMCPY2D copy = {};
  if (rpc_read(conn, &direction, sizeof(direction)) < 0 ||
      rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = is_server_authoritative
                          ? cuMemcpy2D_v2(&copy)
                          : lupine_copy_client_host_to_device(
                                conn, CU_STREAM_LEGACY, true,
                                [&] { return lupine_make_2d_htod_plan(copy); });
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  bool is_server_authoritative = false;
  CUDA_MEMCPY2D copy = {};
  if (rpc_read(conn, &direction, sizeof(direction)) < 0 ||
      rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result = is_server_authoritative
                          ? cuMemcpy2DUnaligned_v2(&copy)
                          : lupine_copy_client_host_to_device(
                                conn, CU_STREAM_LEGACY, true,
                                [&] { return lupine_make_2d_htod_plan(copy); });
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
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
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  bool is_server_authoritative = false;
  CUDA_MEMCPY2D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &direction, sizeof(direction)) < 0 ||
      rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  switch (direction) {
  case LUPINE_COPY_DIRECTION_HTOD: {
    if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
      return -1;
    }
    int request_id = rpc_read_end(conn);
    if (request_id < 0) {
      return -1;
    }
    CUresult result =
        is_server_authoritative
            ? cuMemcpy2DAsync_v2(&copy, stream)
            : lupine_copy_client_host_to_device(conn, stream, false, [&] {
                return lupine_make_2d_htod_plan(copy);
              });
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }
  case LUPINE_COPY_DIRECTION_DTOH: {
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
    result = lupine_copy_client_host_to_device(conn, stream, false, [&] {
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
