// Server-side copy machinery: the staging pools the device copies stage
// through, the lifetime rules that tie those pools to a context, and the
// chunked host-to-device and device-to-host loops. HTTP content compression
// is handled transparently by the transport.
#include <cuda.h>

#include "codegen/gen_rpc_ids.h"
#include "cuda_compat.h"
#include "cuda_server.h"
#include "cuda_server_memcpy.h"
#include "ops/smemcpy_module.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lupine_log.h"

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *context,
                                           unsigned int flags, CUdevice device);

template <typename T, typename Tag> static T &lupine_leaked(Tag) {
  static T *instance = new T();
  return *instance;
}

struct lupine_raw_bytes {
  const void *data;
  size_t size;
};

template <typename T> static int lupine_write_one(conn_t *conn, const T &val) {
  return rpc_write(conn, &val, sizeof(val));
}

static int lupine_write_one(conn_t *conn, lupine_raw_bytes raw) {
  return raw.size == 0 ? 0 : rpc_write(conn, raw.data, raw.size);
}

template <typename... Args>
static bool lupine_read_all(conn_t *conn, Args &...args) {
  return ((rpc_read(conn, &args, sizeof(args)) >= 0) && ...);
}

template <typename... Args>
static int lupine_write_response(conn_t *conn, int request_id,
                                 const Args &...args) {
  return rpc_write_start_response(conn, request_id) < 0 ||
                 ((lupine_write_one(conn, args) < 0) || ...) ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

template <typename T, size_t N> struct lupine_inline_buf {
  T stack[N];
  std::unique_ptr<T[]> heap;
  T *ptr = stack;
  T *alloc(size_t n) {
    if (n <= N) {
      return ptr = stack;
    }
    if (n > SIZE_MAX / sizeof(T)) {
      return nullptr;
    }
    heap.reset(new (std::nothrow) T[n]);
    return ptr = heap.get();
  }
};

template <typename Fn>
static bool lupine_with_pushed_context(CUcontext context, Fn &&fn) {
  if (context == nullptr || cuCtxPushCurrent_v2(context) != CUDA_SUCCESS) {
    return false;
  }
  fn();
  CUcontext popped = nullptr;
  (void)cuCtxPopCurrent_v2(&popped);
  return true;
}

class lupine_htod_side_effect_ring;

struct lupine_fetch_stream {
  CUcontext context = nullptr;
  CUstream stream = nullptr;

  ~lupine_fetch_stream() {
    if (stream != nullptr) {
      lupine_with_pushed_context(context,
                                 [&] { (void)cuStreamDestroy_v2(stream); });
    }
  }
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
  std::unordered_map<CUcontext, std::shared_ptr<lupine_htod_side_effect_ring>>
      htod_rings;
  std::unordered_map<CUcontext, std::shared_ptr<lupine_fetch_stream>>
      fetch_streams;
};

struct lupine_staging_registry {
  std::atomic<lupine_staging_state *> primary{nullptr};
  std::mutex mutex;
  std::unordered_map<conn_t *, std::unique_ptr<lupine_staging_state>> states;
};

static lupine_staging_registry &lupine_staging_states() {
  return lupine_leaked<lupine_staging_registry>([] {});
}

static lupine_staging_state *lupine_staging_state_for(conn_t *conn) {
  if (conn == nullptr) {
    return nullptr;
  }
  auto &reg = lupine_staging_states();
  if (auto *s = reg.primary.load(std::memory_order_acquire);
      s != nullptr && s->conn == conn) {
    return s;
  }
  std::lock_guard<std::mutex> lock(reg.mutex);
  auto it = reg.states.find(conn);
  return it == reg.states.end() ? nullptr : it->second.get();
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

// What a capture's host-to-device side effects are keyed by: the graph it
// builds, and the streams and events joined to it while it runs. Monotonic
// non-zero IDs (0 = none) avoid per-capture heap allocations and leaks.
static lupine_capture_id lupine_next_capture_id() {
  static std::atomic<lupine_capture_id> next_id{1};
  return next_id.fetch_add(1, std::memory_order_relaxed);
}

template <typename K, typename V> struct lupine_tracked_map {
  std::atomic<size_t> count{0};
  std::mutex mutex;
  std::unordered_map<K, V> entries;

  bool find(const K &key, V &out) {
    if (count.load(std::memory_order_acquire) == 0)
      return false;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = entries.find(key);
    return it != entries.end() && (out = it->second, true);
  }
  void insert_or_assign(const K &key, V value) {
    V replaced;
    std::lock_guard<std::mutex> lock(mutex);
    auto &slot = entries[key];
    replaced = std::move(slot), slot = std::move(value);
    count.store(entries.size(), std::memory_order_release);
  }
  void erase(const K &key) {
    if (count.load(std::memory_order_acquire) == 0)
      return;
    V removed;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = entries.find(key); it != entries.end()) {
      removed = std::move(it->second), entries.erase(it);
      count.store(entries.size(), std::memory_order_release);
    }
  }
};

struct lupine_capture_tracker {
  struct members {
    std::vector<CUstream> streams;
    std::vector<CUevent> events;
  };
  std::atomic<size_t> active_stream_captures{0};
  std::atomic<size_t> active_event_captures{0};
  std::atomic<size_t> tracked_capture_graphs{0};
  std::mutex mutex;
  std::unordered_map<CUstream, lupine_capture_id> stream_to_capture;
  std::unordered_map<CUevent, lupine_capture_id> event_to_capture;
  std::unordered_map<lupine_capture_id, members> capture_members;
  std::unordered_map<CUgraph, lupine_capture_id> graph_to_capture;

  void sync_counts_locked() {
    active_stream_captures.store(stream_to_capture.size(),
                                 std::memory_order_release);
    active_event_captures.store(event_to_capture.size(),
                                std::memory_order_release);
  }
  void erase_capture_locked(lupine_capture_id id) {
    if (auto m = capture_members.find(id); m != capture_members.end()) {
      auto erase_owned = [id](auto &map, const auto &keys) {
        for (auto k : keys)
          if (auto it = map.find(k); it != map.end() && it->second == id)
            map.erase(it);
      };
      erase_owned(stream_to_capture, m->second.streams);
      erase_owned(event_to_capture, m->second.events);
      capture_members.erase(m);
    }
    sync_counts_locked();
  }
};

static lupine_capture_tracker &lupine_capture_resources() {
  return lupine_leaked<lupine_capture_tracker>([] {});
}

lupine_capture_id lupine_begin_stream_capture_resources(CUstream stream) {
  lupine_capture_id id = lupine_next_capture_id();
  auto &t = lupine_capture_resources();
  std::lock_guard<std::mutex> lock(t.mutex);
  t.stream_to_capture[stream] = id;
  t.capture_members[id].streams.push_back(stream);
  t.sync_counts_locked();
  return id;
}

static lupine_capture_id lupine_captured_stream_resources(CUstream stream) {
  auto &t = lupine_capture_resources();
  if (t.active_stream_captures.load(std::memory_order_acquire) == 0)
    return 0;
  std::lock_guard<std::mutex> lock(t.mutex);
  auto it = t.stream_to_capture.find(stream);
  return it == t.stream_to_capture.end() ? 0 : it->second;
}

void lupine_discard_stream_capture_resources(lupine_capture_id capture_id) {
  if (capture_id != 0) {
    auto &t = lupine_capture_resources();
    std::lock_guard<std::mutex> lock(t.mutex);
    t.erase_capture_locked(capture_id);
  }
}

void lupine_finish_stream_capture_resources(CUstream stream, CUgraph graph,
                                            bool success) {
  auto &t = lupine_capture_resources();
  if (t.active_stream_captures.load(std::memory_order_acquire) == 0)
    return;
  std::lock_guard<std::mutex> lock(t.mutex);
  auto it = t.stream_to_capture.find(stream);
  if (it == t.stream_to_capture.end() || it->second == 0)
    return;
  lupine_capture_id id = it->second;
  t.erase_capture_locked(id);
  if (success) {
    t.graph_to_capture[graph] = id;
    t.tracked_capture_graphs.store(t.graph_to_capture.size(),
                                   std::memory_order_release);
  }
}

void lupine_record_event_capture_resources(CUevent event, CUstream stream) {
  auto &t = lupine_capture_resources();
  if (t.active_stream_captures.load(std::memory_order_acquire) == 0 &&
      t.active_event_captures.load(std::memory_order_acquire) == 0)
    return;
  std::lock_guard<std::mutex> lock(t.mutex);
  auto sit = t.stream_to_capture.find(stream);
  if (sit == t.stream_to_capture.end()) {
    if (t.event_to_capture.erase(event) != 0)
      t.sync_counts_locked();
    return;
  }
  auto [eit, ins] = t.event_to_capture.emplace(event, sit->second);
  if (ins || eit->second != sit->second) {
    eit->second = sit->second;
    t.capture_members[sit->second].events.push_back(event);
    t.sync_counts_locked();
  }
}

void lupine_forget_event_capture_resources(CUevent event) {
  auto &t = lupine_capture_resources();
  if (t.active_event_captures.load(std::memory_order_acquire) == 0)
    return;
  std::lock_guard<std::mutex> lock(t.mutex);
  if (t.event_to_capture.erase(event) != 0)
    t.sync_counts_locked();
}

void lupine_wait_event_capture_resources(CUstream stream, CUevent event) {
  auto &t = lupine_capture_resources();
  if (t.active_event_captures.load(std::memory_order_acquire) == 0)
    return;
  std::lock_guard<std::mutex> lock(t.mutex);
  if (auto eit = t.event_to_capture.find(event);
      eit != t.event_to_capture.end() &&
      t.stream_to_capture.emplace(stream, eit->second).second) {
    t.capture_members[eit->second].streams.push_back(stream);
    t.sync_counts_locked();
  }
}

static lupine_capture_id lupine_graph_capture_id(CUgraph graph) {
  auto &t = lupine_capture_resources();
  if (t.tracked_capture_graphs.load(std::memory_order_acquire) == 0)
    return 0;
  std::lock_guard<std::mutex> lock(t.mutex);
  auto it = t.graph_to_capture.find(graph);
  return it == t.graph_to_capture.end() ? 0 : it->second;
}

static std::mutex &lupine_dtoh_graph_nodes_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static auto &lupine_dtoh_node_graph_map() {
  return lupine_leaked<std::unordered_map<CUgraphNode, CUgraph>>([] {});
}

static auto &lupine_dtoh_graph_push_nodes_map() {
  return lupine_leaked<std::unordered_map<CUgraph, std::vector<CUgraphNode>>>(
      [] {});
}

void lupine_clone_graph_resources(CUgraph clone, CUgraph original) {
  auto &tracker = lupine_capture_resources();
  if (tracker.tracked_capture_graphs.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> lock(tracker.mutex);
    if (auto it = tracker.graph_to_capture.find(original);
        it != tracker.graph_to_capture.end()) {
      tracker.graph_to_capture[clone] = it->second;
      tracker.tracked_capture_graphs.store(tracker.graph_to_capture.size(),
                                           std::memory_order_release);
    }
  }
  std::vector<CUgraphNode> original_nodes;
  {
    std::lock_guard<std::mutex> lock(lupine_dtoh_graph_nodes_mutex());
    auto it = lupine_dtoh_graph_push_nodes_map().find(original);
    if (it != lupine_dtoh_graph_push_nodes_map().end()) {
      original_nodes = it->second;
    }
  }
  if (!original_nodes.empty()) {
    std::vector<CUgraphNode> cloned_nodes;
    for (CUgraphNode orig_push : original_nodes) {
      CUgraphNode cloned_push = nullptr;
      if (cuGraphNodeFindInClone(&cloned_push, orig_push, clone) ==
              CUDA_SUCCESS &&
          cloned_push != nullptr) {
        cloned_nodes.push_back(cloned_push);
      }
    }
    if (!cloned_nodes.empty()) {
      std::lock_guard<std::mutex> lock(lupine_dtoh_graph_nodes_mutex());
      auto &slot = lupine_dtoh_graph_push_nodes_map()[clone];
      for (CUgraphNode cloned_push : cloned_nodes) {
        lupine_dtoh_node_graph_map()[cloned_push] = clone;
        slot.push_back(cloned_push);
      }
    }
  }
}

static void lupine_free_retired_dtoh_staging();

void lupine_erase_graph_resources(CUgraph graph) {
  auto &tracker = lupine_capture_resources();
  if (tracker.tracked_capture_graphs.load(std::memory_order_acquire) != 0) {
    std::lock_guard<std::mutex> lock(tracker.mutex);
    if (tracker.graph_to_capture.erase(graph) != 0) {
      tracker.tracked_capture_graphs.store(tracker.graph_to_capture.size(),
                                           std::memory_order_release);
    }
  }
  {
    std::lock_guard<std::mutex> lock(lupine_dtoh_graph_nodes_mutex());
    auto it = lupine_dtoh_graph_push_nodes_map().find(graph);
    if (it != lupine_dtoh_graph_push_nodes_map().end()) {
      for (CUgraphNode push : it->second) {
        lupine_dtoh_node_graph_map().erase(push);
      }
      lupine_dtoh_graph_push_nodes_map().erase(it);
    }
  }
  lupine_free_retired_dtoh_staging();
}

// Pinned staging a one-shot copy lands in, or a graph copy's scratch.
struct lupine_dtoh_staging {
  void *host = nullptr;
  // Zero for heap memory, which needs no pinning bookkeeping.
  size_t capacity = 0;
  unsigned long long buffer_id = 0;
};

// A delivery's copy id when it carries no bytes: the copy wrote the server's
// copy of memory the client refreshes on access, so the client marks it stale.
static constexpr uint64_t kLupineDtohInvalidate = ~uint64_t{0};

// A device-to-host copy on its way to the client.
struct lupine_dtoh_delivery {
  uint64_t number = 0;
  lupine_host_rows client;
  lupine_dtoh_staging bytes;
  bool invalidate = false;
  // Staging capacity this delivery holds against kLupineDtohPushLimitBytes.
  size_t reserved = 0;
};

// Staging held by queued copies, from submission until the bytes are sent.
// Handlers about to queue more device-to-host work wait while this is
// exceeded, so a slow client or a long kernel bounds server memory; a CUDA
// callback never waits.
static constexpr size_t kLupineDtohPushLimitBytes = 256ull << 20;
// Deliveries this large leave over a bulk connection, so they do not hold up
// answers on the session connection.
static constexpr size_t kLupineDtohBulkBytes = 8ull << 20;

struct lupine_dtoh_push {
  conn_t *conn = nullptr;
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<lupine_dtoh_delivery> queue;
  size_t reserved_bytes = 0;
  bool sending = false;
  std::atomic<uint64_t> queued{0};
  std::atomic<uint64_t> assigned{0};
  std::atomic<bool> graph_dtoh{false};
};

// Created by whichever comes first, a copy or the client's push lane, and never
// freed: callbacks still queued on a stream may outlive the connection.
static lupine_dtoh_push *lupine_dtoh_push_for(conn_t *conn) {
  return rpc_dtoh_state<lupine_dtoh_push>(
      conn, [](lupine_dtoh_push *push, conn_t *owner) { push->conn = owner; });
}

uint64_t lupine_dtoh_watermark(conn_t *conn) {
  return lupine_dtoh_push_for(conn)->assigned.load(std::memory_order_acquire);
}

void lupine_dtoh_sync_contexts(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  auto *push = static_cast<lupine_dtoh_push *>(
      __atomic_load_n(&conn->dtoh_push, __ATOMIC_ACQUIRE));
  if (push == nullptr || (!push->graph_dtoh.load(std::memory_order_acquire) &&
                          push->assigned.load(std::memory_order_acquire) ==
                              push->queued.load(std::memory_order_acquire))) {
    return;
  }
  std::vector<CUcontext> contexts;
  CUcontext current = nullptr;
  if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr) {
    contexts.push_back(current);
  }
  if (lupine_staging_state *state = lupine_staging_state_for(conn)) {
    std::lock_guard<std::mutex> lock(state->mutex);
    auto add_ctx = [&](CUcontext ctx, CUdevice dev, bool has_dev) {
      if (ctx == nullptr || state->teardown_contexts.count(ctx) != 0 ||
          (has_dev && state->teardown_devices.count(dev) != 0)) {
        return;
      }
      if (std::find(contexts.begin(), contexts.end(), ctx) == contexts.end()) {
        contexts.push_back(ctx);
      }
    };
    for (const auto &entry : state->primary_contexts) {
      add_ctx(entry.second, entry.first, true);
    }
    for (CUcontext ctx : state->created_contexts) {
      add_ctx(ctx, 0, false);
    }
  }
  for (CUcontext ctx : contexts) {
    lupine_with_pushed_context(ctx, [] { (void)cuCtxSynchronize(); });
  }
}

static unsigned long long lupine_pinned_buffer_id(void *host) {
  unsigned long long buffer_id = 0;
  return host != nullptr &&
                 cuPointerGetAttribute(
                     &buffer_id, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                     reinterpret_cast<CUdeviceptr>(host)) == CUDA_SUCCESS
             ? buffer_id
             : 0;
}

// Sent staging comes back to a pool, since pinning memory costs more than a
// small copy, and the sender could not free it anyway: freeing pinned memory
// can wait for the device, which may be waiting for a host callback, which may
// be waiting for a delivery. The pool evicts to make room, and what it does not
// keep, and graph scratch, is retired for a handler thread to free. The buffer
// id guards against a context teardown having freed a buffer and its address
// having come back.
static constexpr size_t kLupineDtohPoolBytes = 256ull << 20;
static constexpr size_t kLupineDtohMinStagingBytes = 4096;

static std::mutex &lupine_dtoh_staging_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static auto &lupine_dtoh_staging_pool() {
  return lupine_leaked<std::vector<lupine_dtoh_staging>>([] {});
}

static auto &lupine_retired_dtoh_staging() {
  return lupine_leaked<std::vector<lupine_dtoh_staging>>([] {});
}

static size_t lupine_dtoh_pooled_bytes = 0;

static size_t lupine_dtoh_staging_capacity(size_t bytes) {
  size_t capacity = kLupineDtohMinStagingBytes;
  while (capacity < bytes) {
    capacity *= 2;
  }
  return capacity;
}

static CUresult lupine_alloc_dtoh_host(conn_t *conn, CUstream stream,
                                       void **host, size_t bytes) {
  CUresult result = cuMemAllocHost(host, bytes);
  if (result != CUDA_ERROR_INVALID_CONTEXT) {
    return result;
  }
  CUcontext ctx = nullptr;
  if (stream != nullptr && stream != CU_STREAM_LEGACY &&
      stream != CU_STREAM_PER_THREAD &&
      cuStreamGetCtx(stream, &ctx) == CUDA_SUCCESS && ctx != nullptr &&
      cuCtxSetCurrent(ctx) == CUDA_SUCCESS) {
    return cuMemAllocHost(host, bytes);
  }
  if (lupine_staging_state *state = lupine_staging_state_for(conn)) {
    ctx = nullptr;
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      for (const auto &entry : state->primary_contexts) {
        if (entry.second != nullptr &&
            state->teardown_contexts.count(entry.second) == 0 &&
            state->teardown_devices.count(entry.first) == 0) {
          ctx = entry.second;
          break;
        }
      }
      if (ctx == nullptr) {
        for (CUcontext created : state->created_contexts) {
          if (created != nullptr &&
              state->teardown_contexts.count(created) == 0) {
            ctx = created;
            break;
          }
        }
      }
    }
    if (ctx != nullptr && cuCtxSetCurrent(ctx) == CUDA_SUCCESS) {
      return cuMemAllocHost(host, bytes);
    }
  }
  return result;
}

static lupine_dtoh_staging
lupine_take_dtoh_staging(conn_t *conn, CUstream stream, size_t bytes) {
  size_t capacity = lupine_dtoh_staging_capacity(bytes);
  lupine_dtoh_staging staging;
  {
    std::lock_guard<std::mutex> lock(lupine_dtoh_staging_mutex());
    auto &pool = lupine_dtoh_staging_pool();
    auto it = std::find_if(pool.begin(), pool.end(), [&](const auto &entry) {
      return entry.capacity == capacity;
    });
    if (it != pool.end()) {
      staging = *it;
      *it = pool.back();
      pool.pop_back();
      lupine_dtoh_pooled_bytes -= capacity;
    }
  }
  if (staging.host != nullptr &&
      lupine_pinned_buffer_id(staging.host) == staging.buffer_id) {
    return staging;
  }
  if (lupine_alloc_dtoh_host(conn, stream, &staging.host, capacity) ==
      CUDA_SUCCESS) {
    return {staging.host, capacity, lupine_pinned_buffer_id(staging.host)};
  }
  return {std::malloc(bytes), 0, 0};
}

static void lupine_give_back_dtoh_staging(const lupine_dtoh_staging &staging) {
  if (staging.buffer_id == 0) {
    std::free(staging.host);
    return;
  }
  std::lock_guard<std::mutex> lock(lupine_dtoh_staging_mutex());
  auto &pool = lupine_dtoh_staging_pool();
  auto &retired = lupine_retired_dtoh_staging();
  if (staging.capacity == 0 || staging.capacity > kLupineDtohPoolBytes) {
    retired.push_back(staging);
    return;
  }
  while (lupine_dtoh_pooled_bytes + staging.capacity > kLupineDtohPoolBytes) {
    lupine_dtoh_pooled_bytes -= pool.front().capacity;
    retired.push_back(pool.front());
    pool.erase(pool.begin());
  }
  pool.push_back(staging);
  lupine_dtoh_pooled_bytes += staging.capacity;
}

static void lupine_free_retired_dtoh_staging() {
  std::vector<lupine_dtoh_staging> retired;
  {
    std::lock_guard<std::mutex> lock(lupine_dtoh_staging_mutex());
    retired.swap(lupine_retired_dtoh_staging());
  }
  for (const auto &staging : retired) {
    if (lupine_pinned_buffer_id(staging.host) == staging.buffer_id) {
      (void)cuMemFreeHost(staging.host);
    }
  }
}

// What the host function queued behind a copy delivers. A one-shot copy hands
// over the staging it landed in; a graph copy's scratch is rewritten by the
// next replay, so each run snapshots it. With no staging at all, the copy
// wrote the server's copy of memory the client refreshes, which only has to be
// marked stale.
struct lupine_dtoh_push_data {
  lupine_dtoh_push *push = nullptr;
  lupine_host_rows client;
  lupine_dtoh_staging source;
  bool one_shot = true;
  size_t reserved = 0;
  CUgraph graph = nullptr;
};

// Numbers the copy and hands it to the push lane. It runs in stream order, so
// anything that observes the stream past this point -- a synchronize, an event,
// a host function on any stream that waited for it -- reads a watermark that
// covers it.
static void CUDA_CB lupine_dtoh_push_callback(void *opaque) {
  auto *data = static_cast<lupine_dtoh_push_data *>(opaque);
  bool invalidate = data->source.host == nullptr;
  lupine_dtoh_delivery delivery{0, data->client, data->source, invalidate,
                                data->reserved};
  if (!data->one_shot && !invalidate) {
    size_t bytes = delivery.client.bytes();
    delivery.bytes = {std::malloc(bytes), 0, 0};
    delivery.reserved = bytes;
    if (delivery.bytes.host == nullptr && bytes != 0) {
      // Still numbered, so no client waits for a number never sent, but the
      // next synchronize reports the lost copy.
      lupine_keep_async_error(data->push->conn, CUDA_ERROR_OUT_OF_MEMORY);
      delivery.client = {};
      delivery.reserved = 0;
    } else if (bytes != 0) {
      memcpy(delivery.bytes.host, data->source.host, bytes);
    }
  }
  lupine_dtoh_push *push = data->push;
  {
    std::lock_guard<std::mutex> lock(push->mutex);
    delivery.number = push->assigned.load(std::memory_order_relaxed) + 1;
    if (!data->one_shot) {
      push->reserved_bytes += delivery.reserved;
    }
    push->queue.push_back(delivery);
    push->assigned.store(delivery.number, std::memory_order_release);
  }
  push->changed.notify_all();
  if (data->one_shot) {
    delete data;
  }
}

static void lupine_serve_host_bulk_read(conn_t *conn, uint64_t copy_id,
                                        const void *host, size_t bytes);
static void lupine_open_push_bulk();
static void lupine_close_push_bulk();
static bool lupine_bulk_lost();
static void lupine_bulk_wake();

// The client opens one lane per connection for its deliveries, saying whether
// it reads large ones over a bulk connection, and sends nothing more on it.
// This handler keeps that lane until the connection closes, sending each
// delivery in number order as {number, copy id, rows}, then the bytes unless
// the bulk connection carries them or the copy id says the delivery has none.
int handle_lupineDtohPush(conn_t *conn) {
  uint64_t bulk_reader = 0;
  if (rpc_read(conn, &bulk_reader, sizeof(bulk_reader)) < 0 ||
      rpc_read_end(conn) < 0) {
    return -1;
  }
  if (bulk_reader != 0) {
    lupine_open_push_bulk();
  }
  int32_t stream_id = rpc_current_http2_stream(conn);
  lupine_dtoh_push *push = lupine_dtoh_push_for(conn);
  std::unique_lock<std::mutex> lock(push->mutex);
  for (;;) {
    push->changed.wait(lock,
                       [&] { return conn->closed || !push->queue.empty(); });
    if (conn->closed) {
      break;
    }
    lupine_dtoh_delivery delivery = push->queue.front();
    push->queue.pop_front();
    push->sending = true;
    lock.unlock();
    size_t bytes = delivery.client.bytes();
    bool bulk = bulk_reader != 0 && !lupine_bulk_lost() &&
                bytes >= kLupineDtohBulkBytes && delivery.client.height == 1 &&
                delivery.client.depth == 1;
    uint64_t copy_id = delivery.invalidate ? kLupineDtohInvalidate
                       : bulk ? (uint64_t{1} << 62) | delivery.number
                              : 0;
    rpc_write_cursor cursors[] = {
        rpc_write_cursor(&delivery.number, sizeof(delivery.number)),
        rpc_write_cursor(&copy_id, sizeof(copy_id)),
        rpc_write_cursor(&delivery.client, sizeof(delivery.client)),
        rpc_write_cursor(delivery.bytes.host,
                         bulk || delivery.invalidate ? 0 : bytes)};
    int sent =
        rpc_http2_write_stream(conn, stream_id, cursors, std::size(cursors));
    if (sent == 0 && bulk) {
      lupine_serve_host_bulk_read(conn, copy_id, delivery.bytes.host, bytes);
    }
    lupine_give_back_dtoh_staging(delivery.bytes);
    lock.lock();
    push->sending = false;
    push->reserved_bytes -= delivery.reserved;
    push->changed.notify_all();
    if (sent < 0) {
      break;
    }
  }
  // Nothing sends the rest any more; their staging is no longer in use.
  for (const auto &delivery : push->queue) {
    lupine_give_back_dtoh_staging(delivery.bytes);
    push->reserved_bytes -= delivery.reserved;
  }
  push->queue.clear();
  push->changed.notify_all();
  lock.unlock();
  if (bulk_reader != 0) {
    lupine_close_push_bulk();
  }
  return 0;
}

void lupine_dtoh_wake(conn_t *conn) {
  auto *push = static_cast<lupine_dtoh_push *>(
      __atomic_load_n(&conn->dtoh_push, __ATOMIC_ACQUIRE));
  if (push != nullptr) {
    std::lock_guard<std::mutex> lock(push->mutex);
    push->changed.notify_all();
  }
  lupine_bulk_wake();
}

// Admits `capacity` bytes of staging once the queue has room.
static void lupine_dtoh_reserve(lupine_dtoh_push *push, size_t capacity) {
  lupine_free_retired_dtoh_staging();
  std::unique_lock<std::mutex> lock(push->mutex);
  push->changed.wait(lock, [&] {
    return push->conn->closed || push->reserved_bytes == 0 ||
           push->reserved_bytes + capacity <= kLupineDtohPushLimitBytes;
  });
  push->reserved_bytes += capacity;
}

void lupine_dtoh_wait_for_room(conn_t *conn) {
  lupine_dtoh_push *push = lupine_dtoh_push_for(conn);
  lupine_dtoh_reserve(push, 0);
}

void lupine_dtoh_drain(conn_t *conn) {
  lupine_dtoh_push *push = lupine_dtoh_push_for(conn);
  std::unique_lock<std::mutex> lock(push->mutex);
  push->changed.wait(lock, [&] {
    return conn->closed || (push->queue.empty() && !push->sending);
  });
}

template <typename CopyFn>
static CUresult lupine_capture_dtoh(conn_t *conn, CUstream stream,
                                    const lupine_host_rows &client,
                                    CopyFn &&copy_to_host, void *mirror,
                                    bool refresh);

// Queues a device-to-host copy into client memory on `stream`, and the host
// function that numbers it once the stream has run it. The copy lands in
// staging of its own that is pushed to `client`; with `mirror`, the server's
// copy of the destination is updated from it on the stream first. With
// `refresh`, the copy lands in `mirror` alone and the client is told to mark
// `client` stale, to fetch it from there on access. While the stream captures,
// all of this is recorded into its graph.
template <typename CopyFn>
static CUresult lupine_push_dtoh_with(conn_t *conn, CUstream stream,
                                      const lupine_host_rows &client,
                                      CopyFn &&copy_to_host, void *mirror,
                                      bool refresh) {
  size_t bytes = client.bytes();
  CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    (void)cuStreamIsCapturing(stream, &capture);
  }
  if (capture != CU_STREAM_CAPTURE_STATUS_NONE) {
    return lupine_capture_dtoh(conn, stream, client,
                               std::forward<CopyFn>(copy_to_host), mirror,
                               refresh);
  }
  if (bytes == 0) {
    return CUDA_SUCCESS;
  }
  lupine_dtoh_push *push = lupine_dtoh_push_for(conn);
  lupine_dtoh_staging staging;
  CUresult result = CUDA_SUCCESS;
  if (refresh) {
    lupine_dtoh_reserve(push, 0);
    result = copy_to_host(mirror, stream);
  } else {
    lupine_dtoh_reserve(push, lupine_dtoh_staging_capacity(bytes));
    staging = lupine_take_dtoh_staging(conn, stream, bytes);
    result = staging.host == nullptr ? CUDA_ERROR_OUT_OF_MEMORY
                                     : copy_to_host(staging.host, stream);
    if (result == CUDA_SUCCESS && mirror != nullptr) {
      result = cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(mirror),
                             reinterpret_cast<CUdeviceptr>(staging.host), bytes,
                             stream);
    }
  }
  size_t reserved = refresh ? 0 : lupine_dtoh_staging_capacity(bytes);
  if (result == CUDA_SUCCESS) {
    auto *data =
        new lupine_dtoh_push_data{push, client, staging, true, reserved};
    push->queued.fetch_add(1, std::memory_order_acq_rel);
    result = cuLaunchHostFunc(stream, lupine_dtoh_push_callback, data);
    if (result == CUDA_SUCCESS) {
      return result;
    }
    push->queued.fetch_sub(1, std::memory_order_acq_rel);
    delete data;
    // The copy may still be running into the staging, which is therefore
    // left alone rather than reused.
    staging = {};
  }
  lupine_give_back_dtoh_staging(staging);
  std::lock_guard<std::mutex> lock(push->mutex);
  push->reserved_bytes -= reserved;
  push->changed.notify_all();
  return result;
}

static CUresult lupine_push_dtoh(conn_t *conn, CUstream stream,
                                 const lupine_host_rows &client,
                                 CUdeviceptr source, void *mirror,
                                 bool refresh) {
  size_t bytes = client.bytes();
  return lupine_push_dtoh_with(
      conn, stream, client,
      [&](void *dst, CUstream s) {
        return refresh ? cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                       source, bytes, s)
                       : cuMemcpyDtoHAsync_v2(dst, source, bytes, s);
      },
      mirror, refresh);
}

// A graph copy's staging and push data serve every replay, and live until the
// graph and everything instantiated, cloned or launched from it is gone,
// which CUDA tracks with a user object. Its destructor cannot call CUDA, so it
// retires the staging for a handler thread to free.
static void CUDA_CB lupine_release_graph_dtoh(void *opaque) {
  auto *data = static_cast<lupine_dtoh_push_data *>(opaque);
  lupine_give_back_dtoh_staging(data->source);
  delete data;
}

struct lupine_dtoh_exec_entries {
  conn_t *conn = nullptr;
  std::vector<lupine_dtoh_push_data *> entries;
};

static std::mutex &lupine_dtoh_exec_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static auto &lupine_dtoh_exec_map() {
  return lupine_leaked<
      std::unordered_map<CUgraphExec, lupine_dtoh_exec_entries>>([] {});
}

static lupine_dtoh_push_data *
lupine_own_graph_dtoh(conn_t *conn, CUgraph graph,
                      const lupine_host_rows &client, void *source) {
  lupine_dtoh_push *push = lupine_dtoh_push_for(conn);
  push->graph_dtoh.store(true, std::memory_order_release);
  auto *data = new lupine_dtoh_push_data{
      push,  client, {source, 0, lupine_pinned_buffer_id(source)},
      false, 0,      graph};
  CUuserObject object = nullptr;
  if (cuUserObjectCreate(&object, data, lupine_release_graph_dtoh, 1,
                         CU_USER_OBJECT_NO_DESTRUCTOR_SYNC) != CUDA_SUCCESS) {
    lupine_release_graph_dtoh(data);
    return nullptr;
  }
  if (cuGraphRetainUserObject(graph, object, 1, CU_GRAPH_USER_OBJECT_MOVE) !=
      CUDA_SUCCESS) {
    (void)cuUserObjectRelease(object, 1);
    return nullptr;
  }
  return data;
}

// A captured copy lands in staging the capture's graph owns, and the push
// behind it is captured too, so each replay delivers its own bytes. A copy the
// client refreshes from the server lands in the server's copy instead, and each
// replay tells the client to mark it stale.
template <typename CopyFn>
static CUresult lupine_capture_dtoh(conn_t *conn, CUstream stream,
                                    const lupine_host_rows &client,
                                    CopyFn &&copy_to_host, void *mirror,
                                    bool refresh) {
  size_t bytes = client.bytes();
  if (bytes == 0) {
    return CUDA_SUCCESS;
  }
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  CUgraph graph = nullptr;
  CUresult result = cuStreamGetCaptureInfo_v3(stream, &status, nullptr, &graph,
                                              nullptr, nullptr, nullptr);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  // Host allocation is an unsafe call under global or thread-local capture and
  // would invalidate the capture, so it runs in relaxed mode.
  void *host = nullptr;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_RELAXED;
  (void)cuThreadExchangeStreamCaptureMode(&mode);
  if (!refresh &&
      lupine_alloc_dtoh_host(conn, stream, &host, bytes) != CUDA_SUCCESS) {
    host = nullptr;
  }
  lupine_dtoh_push_data *data =
      host == nullptr && !refresh
          ? nullptr
          : lupine_own_graph_dtoh(conn, graph, client, host);
  (void)cuThreadExchangeStreamCaptureMode(&mode);
  if (data == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  result = copy_to_host(refresh ? mirror : host, stream);
  if (result == CUDA_SUCCESS && !refresh && mirror != nullptr) {
    result = cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(mirror),
                           reinterpret_cast<CUdeviceptr>(host), bytes, stream);
  }
  if (result == CUDA_SUCCESS) {
    result = cuLaunchHostFunc(stream, lupine_dtoh_push_callback, data);
  }
  return result;
}

CUresult lupine_graph_add_dtoh_push(conn_t *conn, CUgraph graph,
                                    CUgraphNode copy,
                                    const lupine_host_rows &client,
                                    void *server_src) {
  lupine_dtoh_push_data *data =
      lupine_own_graph_dtoh(conn, graph, client, server_src);
  if (data == nullptr || copy == nullptr) {
    return data == nullptr ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
  }
  CUDA_HOST_NODE_PARAMS params = {};
  params.fn = lupine_dtoh_push_callback;
  params.userData = data;
  CUgraphNode push = nullptr;
  CUresult result = cuGraphAddHostNode(&push, graph, &copy, 1, &params);
  if (result == CUDA_SUCCESS && push != nullptr) {
    std::lock_guard<std::mutex> lock(lupine_dtoh_graph_nodes_mutex());
    lupine_dtoh_node_graph_map()[push] = graph;
    lupine_dtoh_graph_push_nodes_map()[graph].push_back(push);
  }
  return result;
}

// The node pushing the bytes of explicit graph copy `copy`, if it has one.
static CUgraphNode lupine_graph_push_node(CUgraphNode copy) {
  CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
  size_t count = 0;
  if (cuGraphNodeGetType(copy, &type) != CUDA_SUCCESS ||
      type != CU_GRAPH_NODE_TYPE_MEMCPY ||
      cuGraphNodeGetDependentNodes_v2(copy, nullptr, nullptr, &count) !=
          CUDA_SUCCESS ||
      count == 0) {
    return nullptr;
  }
  lupine_inline_buf<CUgraphNode, 8> dep_buf;
  CUgraphNode *dependents = dep_buf.alloc(count);
  if (!dependents || cuGraphNodeGetDependentNodes_v2(copy, dependents, nullptr,
                                                     &count) != CUDA_SUCCESS) {
    return nullptr;
  }
  for (size_t i = 0; i < count; ++i) {
    CUgraphNode node = dependents[i];
    CUDA_HOST_NODE_PARAMS params = {};
    if (cuGraphNodeGetType(node, &type) == CUDA_SUCCESS &&
        type == CU_GRAPH_NODE_TYPE_HOST &&
        cuGraphHostNodeGetParams(node, &params) == CUDA_SUCCESS &&
        params.fn == lupine_dtoh_push_callback) {
      return node;
    }
  }
  return nullptr;
}

void lupine_graph_push_dependencies(CUgraphNode *deps, size_t count) {
  for (size_t i = 0; deps != nullptr && i < count; ++i) {
    if (CUgraphNode push = lupine_graph_push_node(deps[i])) {
      deps[i] = push;
    }
  }
}

// Retargets an explicit graph copy to client memory, and the node pushing its
// bytes: the copy lands in new staging, owned by the graph or executable so
// replays made before the update still keep their own scratch. With `exec`,
// only that exec changes. Turning such a copy into one that does not reach
// client memory, or the reverse, is not supported.
static CUresult lupine_set_graph_dtoh_params(conn_t *conn, CUgraphExec exec,
                                             CUgraphNode node,
                                             CUDA_MEMCPY3D params,
                                             CUcontext ctx) {
  CUgraphNode push = lupine_graph_push_node(node);
  CUgraphNode exec_node =
      exec == nullptr ? node : lupine_htod_graph_exec_node(exec, node);
  bool to_client = params.dstMemoryType == CU_MEMORYTYPE_HOST;
  if (!to_client || push == nullptr) {
    if (to_client || push != nullptr) {
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    return exec == nullptr
               ? cuGraphMemcpyNodeSetParams(node, &params)
               : cuGraphExecMemcpyNodeSetParams(exec, exec_node, &params, ctx);
  }
  CUDA_HOST_NODE_PARAMS old_push = {};
  if (cuGraphHostNodeGetParams(push, &old_push) != CUDA_SUCCESS ||
      old_push.userData == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto *old_data = static_cast<lupine_dtoh_push_data *>(old_push.userData);
  lupine_host_rows client = lupine_host_destination_rows(params);
  lupine_pack_host_destination(params);
  void *host = nullptr;
  if (client.bytes() != 0 &&
      lupine_alloc_dtoh_host(conn, nullptr, &host, client.bytes()) !=
          CUDA_SUCCESS &&
      (host = std::malloc(client.bytes())) == nullptr) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  params.dstHost = host;
  CUresult result =
      exec == nullptr
          ? cuGraphMemcpyNodeSetParams(node, &params)
          : cuGraphExecMemcpyNodeSetParams(exec, exec_node, &params, ctx);
  if (result != CUDA_SUCCESS) {
    lupine_give_back_dtoh_staging({host, 0, lupine_pinned_buffer_id(host)});
    return result;
  }
  lupine_dtoh_push *push_state = lupine_dtoh_push_for(conn);
  push_state->graph_dtoh.store(true, std::memory_order_release);
  CUgraph owning_graph = old_data->graph;
  {
    std::lock_guard<std::mutex> lock(lupine_dtoh_graph_nodes_mutex());
    auto it = lupine_dtoh_node_graph_map().find(push);
    if (it != lupine_dtoh_node_graph_map().end()) {
      owning_graph = it->second;
    }
  }
  lupine_dtoh_push_data *data = nullptr;
  if (exec == nullptr) {
    data = lupine_own_graph_dtoh(conn, owning_graph, client, host);
    if (data == nullptr) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
  } else {
    data = new lupine_dtoh_push_data{
        push_state, client, {host, 0, lupine_pinned_buffer_id(host)},
        false,      0,      owning_graph};
  }
  CUDA_HOST_NODE_PARAMS push_params = {};
  push_params.fn = lupine_dtoh_push_callback;
  push_params.userData = data;
  result = exec == nullptr ? cuGraphHostNodeSetParams(push, &push_params)
                           : cuGraphExecHostNodeSetParams(
                                 exec, lupine_htod_graph_exec_node(exec, push),
                                 &push_params);
  if (exec != nullptr) {
    if (result == CUDA_SUCCESS) {
      std::lock_guard<std::mutex> lock(lupine_dtoh_exec_mutex());
      auto &slot = lupine_dtoh_exec_map()[exec];
      slot.conn = conn;
      slot.entries.push_back(data);
    } else {
      lupine_release_graph_dtoh(data);
    }
  }
  return result;
}

static int handle_cuGraphMemcpyNodeSetParamsCommon(conn_t *conn,
                                                   bool exec_node) {
  CUgraphExec exec = nullptr;
  CUgraphNode node = nullptr;
  CUDA_MEMCPY3D params = {};
  CUcontext ctx = nullptr;
  if ((exec_node && rpc_read(conn, &exec, sizeof(exec)) < 0) ||
      rpc_read(conn, &node, sizeof(node)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0 ||
      (exec_node && rpc_read(conn, &ctx, sizeof(ctx)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = lupine_set_graph_dtoh_params(conn, exec, node, params, ctx);
  return lupine_unanswered(conn, result)
             ? 0
             : lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_cuGraphMemcpyNodeSetParams(conn_t *conn) {
  return handle_cuGraphMemcpyNodeSetParamsCommon(conn, false);
}

int handle_cuGraphExecMemcpyNodeSetParams(conn_t *conn) {
  return handle_cuGraphMemcpyNodeSetParamsCommon(conn, true);
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
  bool destination_is_htoa = false;
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

  CUdeviceptr destination_at(size_t logical_offset) const {
    size_t row_index = logical_offset / smemcpy.width;
    size_t x = logical_offset - row_index * smemcpy.width;
    size_t slice = row_index / smemcpy.rows;
    size_t row = row_index - slice * smemcpy.rows;
    return smemcpy.destination + slice * smemcpy.destination_slice_stride +
           row * smemcpy.destination_row_stride + x;
  }
};

class lupine_htod_graph_execution;
struct lupine_htod_capture_events;

struct lupine_htod_callback_data {
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  lupine_htod_copy copy;
  lupine_htod_fragment fragment;
  std::shared_ptr<lupine_htod_graph_execution> graph_execution;
  std::shared_ptr<lupine_htod_capture_events> capture_events;
};

// Each CUDA host callback fetches one bounded fragment into this ring and
// returns. The following copy runs on the same private stream, so slot reuse is
// stream ordered and the callback never waits for CUDA work.
class lupine_htod_side_effect_ring {
public:
  static constexpr size_t fragment_bytes = 8 * 1024 * 1024;
  static constexpr size_t storage_headroom = 15;
  static constexpr size_t storage_bytes =
      (fragment_bytes + storage_headroom + 127) & ~size_t(127);

  lupine_htod_side_effect_ring(conn_t *conn, CUcontext context)
      : conn_(conn), context_(context) {}

  ~lupine_htod_side_effect_ring() {
    lupine_with_pushed_context(context_, [&] {
      if (transfer_stream_ != nullptr) {
        (void)cuStreamDestroy_v2(transfer_stream_);
      }
      if (ordering_event_ != nullptr) {
        (void)cuEventDestroy_v2(ordering_event_);
      }
      lupine_smemcpy_cached_module_unload(&smemcpy_module_);
    });
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
    CUresult result = cuMemHostRegister(storage, storage_bytes,
                                        CU_MEMHOSTREGISTER_PORTABLE |
                                            CU_MEMHOSTREGISTER_DEVICEMAP);
    if (result != CUDA_SUCCESS) {
      free_storage(storage);
      return result;
    }
    CUdeviceptr device_storage = 0;
    CUstream transfer_stream = nullptr;
    CUevent ordering_event = nullptr;
    result = cuMemHostGetDevicePointer_v2(&device_storage, storage, 0);
    if (result == CUDA_SUCCESS) {
      result = cuStreamCreate(&transfer_stream, CU_STREAM_NON_BLOCKING);
    }
    if (result == CUDA_SUCCESS) {
      result = cuEventCreate(&ordering_event, CU_EVENT_DISABLE_TIMING);
    }
    // Load and JIT the fatbin before graph capture can enqueue a copy that
    // needs one of its kernels.
    if (result == CUDA_SUCCESS) {
      result = lupine_smemcpy_cached_module_load(&smemcpy_module_);
    }
    if (result != CUDA_SUCCESS) {
      if (ordering_event != nullptr) {
        (void)cuEventDestroy_v2(ordering_event);
      }
      if (transfer_stream != nullptr) {
        (void)cuStreamDestroy_v2(transfer_stream);
      }
      (void)cuMemHostUnregister(storage);
      free_storage(storage);
      return result;
    }
    storage_ = storage;
    device_storage_ = device_storage;
    transfer_stream_ = transfer_stream;
    ordering_event_ = ordering_event;
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

  void *data(size_t offset = 0) const {
    return static_cast<unsigned char *>(storage_) + offset;
  }

  CUdeviceptr device_data(size_t offset = 0) const {
    return device_storage_ + offset;
  }

  CUstream transfer_stream() const { return transfer_stream_; }

  CUevent ordering_event() const { return ordering_event_; }

  conn_t *connection() const { return conn_; }

  CUcontext context() const { return context_; }

  bool data_offset(const void *pointer, size_t *offset) const {
    uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    uintptr_t begin = reinterpret_cast<uintptr_t>(storage_);
    size_t bytes = storage_bytes;
    if (address < begin || address - begin >= bytes) {
      return false;
    }
    *offset = address - begin;
    return true;
  }

  bool device_data_offset(CUdeviceptr pointer, size_t *offset) const {
    size_t bytes = storage_bytes;
    if (pointer < device_storage_ || pointer - device_storage_ >= bytes) {
      return false;
    }
    *offset = static_cast<size_t>(pointer - device_storage_);
    return true;
  }

  CUresult synchronize() const { return cuStreamSynchronize(transfer_stream_); }

  size_t ring_offset(const lupine_htod_copy &copy,
                     size_t logical_offset) const {
    return copy.destination_array == nullptr
               ? copy.destination_at(logical_offset) & 15
               : 0;
  }

  bool is_smemcpy_function(CUfunction function) const {
    return lupine_smemcpy_cached_contains_function(&smemcpy_module_, function);
  }

  CUresult
  prepare_smemcpy_launch(const lupine_smemcpy_params &params,
                         lupine_smemcpy_launch_descriptor *launch) const {
    return lupine_smemcpy_cached_prepare_launch(&smemcpy_module_, &params,
                                                launch);
  }

  CUresult smemcpy_function(const lupine_smemcpy_launch_descriptor &launch,
                            CUfunction *function) {
    return lupine_smemcpy_cached_function(&smemcpy_module_, &launch, function);
  }

  int fetch(const lupine_htod_copy &copy,
            const lupine_htod_fragment &fragment) {
    auto *ring = static_cast<unsigned char *>(
        data(ring_offset(copy, fragment.logical_offset)));
    // A device-to-host copy into this source that the stream ran before it got
    // here is numbered by now; the client applies it before reading.
    uint64_t delivered = lupine_dtoh_watermark(conn_);
    if (rpc_write_start_request(conn_, LUPINE_SIDE_EFFECT_READ_HOST_MEMORY) <
            0 ||
        lupine_write_one(conn_, delivered) < 0 ||
        lupine_write_one(conn_, copy.source) < 0 ||
        lupine_write_one(conn_, copy.smemcpy.width) < 0 ||
        lupine_write_one(conn_, copy.smemcpy.rows) < 0 ||
        lupine_write_one(conn_, copy.source_row_stride) < 0 ||
        lupine_write_one(conn_, copy.slices) < 0 ||
        lupine_write_one(conn_, copy.source_slice_stride) < 0 ||
        lupine_write_one(conn_, fragment.logical_offset) < 0 ||
        lupine_write_one(conn_, fragment.bytes) < 0 ||
        rpc_wait_for_response(conn_) < 0 ||
        rpc_read(conn_, ring, fragment.bytes) < 0 || rpc_read_end(conn_) < 0) {
      return -1;
    }
    return 0;
  }

private:
  static constexpr size_t storage_alignment = 4096;

  static void *allocate_storage() noexcept {
    return ::operator new(storage_bytes, std::align_val_t(storage_alignment),
                          std::nothrow);
  }

  static void free_storage(void *storage) noexcept {
    ::operator delete(storage, std::align_val_t(storage_alignment));
  }

  conn_t *conn_ = nullptr;
  CUcontext context_ = nullptr;
  std::mutex prepare_mutex_;
  std::mutex execution_mutex_;
  std::condition_variable execution_condition_;
  bool execution_active_ = false;
  void *storage_ = nullptr;
  CUdeviceptr device_storage_ = 0;
  CUstream transfer_stream_ = nullptr;
  CUevent ordering_event_ = nullptr;
  lupine_smemcpy_cached_module smemcpy_module_;
};

struct lupine_htod_capture_events {
  explicit lupine_htod_capture_events(CUcontext event_context)
      : context(event_context) {}

  ~lupine_htod_capture_events() {
    lupine_with_pushed_context(context, [&] {
      if (join != nullptr) {
        (void)cuEventDestroy_v2(join);
      }
      if (fork != nullptr) {
        (void)cuEventDestroy_v2(fork);
      }
    });
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

  void note_callback() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++callback_count_;
  }

  const std::shared_ptr<lupine_htod_side_effect_ring> &ring() const {
    return ring_;
  }

  size_t callback_count() const { return callback_count_; }

private:
  std::shared_ptr<lupine_htod_side_effect_ring> ring_;
  std::mutex mutex_;
  size_t callback_count_ = 0;
};

struct lupine_htod_graph_state {
  explicit lupine_htod_graph_state(
      const std::shared_ptr<lupine_htod_side_effect_ring> &ring)
      : execution(std::make_shared<lupine_htod_graph_execution>(ring)) {}

  std::shared_ptr<lupine_htod_graph_execution> execution;
  std::mutex callbacks_mutex;
  std::vector<std::shared_ptr<lupine_htod_callback_data>> callbacks;
};

static auto &lupine_htod_graph_states() {
  return lupine_leaked<lupine_tracked_map<
      lupine_capture_id, std::shared_ptr<lupine_htod_graph_state>>>([] {});
}

static std::shared_ptr<lupine_htod_graph_state> lupine_htod_graph_state_for(
    lupine_capture_id capture_id,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring) {
  auto &reg = lupine_htod_graph_states();
  std::lock_guard<std::mutex> lock(reg.mutex);
  if (auto it = reg.entries.find(capture_id); it != reg.entries.end()) {
    return it->second;
  }
  try {
    auto state = std::make_shared<lupine_htod_graph_state>(ring);
    reg.entries.emplace(capture_id, state);
    reg.count.store(reg.entries.size(), std::memory_order_release);
    return state;
  } catch (...) {
    return nullptr;
  }
}

struct lupine_htod_exec_resources {
  ~lupine_htod_exec_resources() {
    if (prepared != nullptr && prepared != original) {
      lupine_with_pushed_context(context,
                                 [&] { (void)cuGraphDestroy(prepared); });
    }
  }

  std::vector<std::shared_ptr<lupine_htod_callback_data>> callbacks;
  // Exec-node APIs require a node from the graph used for instantiation. Keep
  // the private ring-rebound clone so client nodes can be translated to it.
  CUgraph original = nullptr;
  CUgraph prepared = nullptr;
  CUcontext context = nullptr;
};

static auto &lupine_htod_exec_resource_map() {
  return lupine_leaked<lupine_tracked_map<
      CUgraphExec, std::shared_ptr<lupine_htod_exec_resources>>>([] {});
}

static void CUDA_CB lupine_htod_side_effect_callback(void *opaque) {
  auto *data = static_cast<lupine_htod_callback_data *>(opaque);
  if (data == nullptr) {
    return;
  }
  if (data->ring->fetch(data->copy, data->fragment) < 0) {
    LUPINE_LOG_ERROR("HtoD side-effect transport failed");
  }
  if (data->graph_execution == nullptr) {
    delete data;
  }
}

template <size_t N>
static CUresult lupine_get_graph_nodes(CUgraph graph,
                                       lupine_inline_buf<CUgraphNode, N> &buf,
                                       size_t *count) {
  *count = 0;
  CUresult result = cuGraphGetNodes(graph, nullptr, count);
  if (result != CUDA_SUCCESS || *count == 0) {
    return result;
  }
  CUgraphNode *nodes = buf.alloc(*count);
  return nodes == nullptr ? CUDA_ERROR_OUT_OF_MEMORY
                          : cuGraphGetNodes(graph, nodes, count);
}

static CUresult
lupine_prepare_htod_graph_exec(CUgraph graph, lupine_capture_id capture_id,
                               lupine_htod_graph_binding *binding) {
  *binding = {};
  binding->original = graph;
  binding->prepared = graph;
  std::shared_ptr<lupine_htod_graph_state> captured_state;
  if (capture_id == 0 ||
      !lupine_htod_graph_states().find(capture_id, captured_state)) {
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

  lupine_inline_buf<CUgraphNode, 32> node_buf;
  size_t node_count = 0;
  result = lupine_get_graph_nodes(binding->prepared, node_buf, &node_count);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  CUgraphNode *nodes = node_buf.ptr;

  std::vector<std::shared_ptr<lupine_htod_callback_data>> callbacks;
  std::unordered_map<CUevent, CUevent> events;
  std::unordered_map<const lupine_htod_capture_events *,
                     std::shared_ptr<lupine_htod_capture_events>>
      callback_event_sets;
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
  for (size_t node_idx = 0; node_idx < node_count; ++node_idx) {
    CUgraphNode node = nodes[node_idx];
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
    auto existing_events =
        callback_event_sets.find(captured->capture_events.get());
    if (existing_events != callback_event_sets.end()) {
      callback_events = existing_events->second;
    } else {
      result =
          lupine_make_htod_capture_events(ring->context(), callback_events);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      try {
        callback_event_sets.emplace(captured->capture_events.get(),
                                    callback_events);
        events.emplace(captured->capture_events->fork, callback_events->fork);
        events.emplace(captured->capture_events->join, callback_events->join);
      } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
      }
    }
    std::shared_ptr<lupine_htod_callback_data> callback;
    try {
      callback = std::make_shared<lupine_htod_callback_data>(
          lupine_htod_callback_data{ring, captured->copy, captured->fragment,
                                    execution, callback_events});
      ++expected_copy_nodes;
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

  for (size_t node_idx = 0; node_idx < node_count; ++node_idx) {
    CUgraphNode node = nodes[node_idx];
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
      copy.source = ring->device_data(offset);
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
        params.srcHost = ring->data(offset);
        result = cuGraphMemcpyNodeSetParams(node, &params);
        if (result != CUDA_SUCCESS) {
          return result;
        }
        ++rebound_copy_nodes;
      }
    } else if (type == CU_GRAPH_NODE_TYPE_EVENT_RECORD ||
               type == CU_GRAPH_NODE_TYPE_WAIT_EVENT) {
      bool is_record = type == CU_GRAPH_NODE_TYPE_EVENT_RECORD;
      CUevent event = nullptr;
      result = is_record ? cuGraphEventRecordNodeGetEvent(node, &event)
                         : cuGraphEventWaitNodeGetEvent(node, &event);
      if (result != CUDA_SUCCESS) {
        return result;
      }
      auto replacement = events.find(event);
      if (replacement != events.end()) {
        result = is_record
                     ? cuGraphEventRecordNodeSetEvent(node, replacement->second)
                     : cuGraphEventWaitNodeSetEvent(node, replacement->second);
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
                                    lupine_htod_graph_binding *binding) {
  return lupine_prepare_htod_graph_exec(graph, lupine_graph_capture_id(graph),
                                        binding);
}

CUresult lupine_associate_graph_exec_resources(
    CUgraphExec exec, const lupine_htod_graph_binding &binding) {
  return lupine_commit_htod_graph_exec(exec, binding);
}

CUgraphNode
lupine_original_htod_graph_node(const lupine_htod_graph_binding &binding,
                                CUgraphNode node) {
  lupine_inline_buf<CUgraphNode, 32> node_buf;
  size_t node_count = 0;
  if (node == nullptr || binding.prepared == binding.original ||
      lupine_get_graph_nodes(binding.original, node_buf, &node_count) !=
          CUDA_SUCCESS) {
    return node;
  }
  for (size_t i = 0; i < node_count; ++i) {
    CUgraphNode clone = nullptr;
    if (cuGraphNodeFindInClone(&clone, node_buf.ptr[i], binding.prepared) ==
            CUDA_SUCCESS &&
        clone == node) {
      return node_buf.ptr[i];
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

// cuGraphExecDestroy and cuGraphExecUpdate leave in-flight launches running.
// Keep callback data, events, and the pinned ring alive until no launch can
// still reference them. This synchronization is confined to graph execs that
// contain pageable HtoD side effects.
static CUresult lupine_drain_htod_graph_exec(CUgraphExec exec) {
  std::shared_ptr<lupine_htod_exec_resources> resources;
  if (!lupine_htod_exec_resource_map().find(exec, resources) ||
      resources->callbacks.empty()) {
    return CUDA_SUCCESS;
  }
  CUresult result = cuCtxPushCurrent_v2(resources->context);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  result = cuCtxSynchronize();
  CUcontext popped = nullptr;
  CUresult pop_result = cuCtxPopCurrent_v2(&popped);
  return result == CUDA_SUCCESS ? pop_result : result;
}

static CUresult lupine_release_htod_graph_exec(CUgraphExec exec) {
  CUresult result = lupine_drain_htod_graph_exec(exec);
  if (result == CUDA_SUCCESS) {
    lupine_htod_exec_resource_map().erase(exec);
  }
  return result;
}

// An update copies the update graph's node parameters into the exec, so that
// graph is rebound to a private ring first, exactly as for instantiation.
CUresult lupine_prepare_graph_exec_update(CUgraphExec exec, CUgraph graph,
                                          lupine_htod_graph_binding *binding) {
  CUresult result = lupine_drain_htod_graph_exec(exec);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return lupine_prepare_graph_exec_resources(graph, binding);
}

static void
lupine_commit_htod_graph_exec_update(CUgraphExec exec,
                                     lupine_htod_graph_binding *binding) {
  auto resources =
      std::static_pointer_cast<lupine_htod_exec_resources>(binding->resources);
  // CUDA copied the parameters, so the update clone is released with the
  // binding. Exec-node APIs still take nodes of the instantiation graph, so
  // the instantiation clone stays the translation target.
  binding->resources.reset();
  std::shared_ptr<lupine_htod_exec_resources> previous;
  (void)lupine_htod_exec_resource_map().find(exec, previous);
  if (resources == nullptr) {
    if (previous != nullptr) {
      previous->callbacks.clear();
    }
    return;
  }
  resources->original = nullptr;
  resources->prepared = nullptr;
  if (previous != nullptr) {
    std::swap(resources->original, previous->original);
    std::swap(resources->prepared, previous->prepared);
  }
  lupine_htod_exec_resource_map().insert_or_assign(exec, std::move(resources));
}

void lupine_finish_graph_exec_update(CUgraphExec exec, CUgraph graph,
                                     lupine_htod_graph_binding *binding,
                                     CUresult result, CUgraphNode *error_node) {
  if (result == CUDA_SUCCESS) {
    lupine_commit_htod_graph_exec_update(exec, binding);
  } else if (error_node != nullptr) {
    *error_node = lupine_original_htod_graph_node(*binding, *error_node);
  }
  lupine_release_htod_graph_binding(binding);
}

void lupine_finish_graph_exec_update(CUgraphExec exec, CUgraph graph,
                                     lupine_htod_graph_binding *binding,
                                     CUresult result,
                                     CUgraphExecUpdateResultInfo *info) {
#if CUDA_VERSION >= 12000
  if (result != CUDA_SUCCESS) {
    info->errorFromNode =
        lupine_original_htod_graph_node(*binding, info->errorFromNode);
  }
#endif
  lupine_finish_graph_exec_update(exec, graph, binding, result,
                                  &info->errorNode);
}

CUresult lupine_release_graph_exec_resources(CUgraphExec exec) {
  CUresult result = lupine_release_htod_graph_exec(exec);
  lupine_dtoh_exec_entries dtoh;
  {
    std::lock_guard<std::mutex> lock(lupine_dtoh_exec_mutex());
    auto it = lupine_dtoh_exec_map().find(exec);
    if (it != lupine_dtoh_exec_map().end()) {
      dtoh = std::move(it->second);
      lupine_dtoh_exec_map().erase(it);
    }
  }
  if (!dtoh.entries.empty()) {
    lupine_dtoh_sync_contexts(dtoh.conn);
    for (lupine_dtoh_push_data *data : dtoh.entries) {
      lupine_release_graph_dtoh(data);
    }
  }
  lupine_free_retired_dtoh_staging();
  return result;
}

static CUresult lupine_enqueue_htod_callback(
    const std::shared_ptr<lupine_htod_graph_state> &graph_state,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    const lupine_htod_copy &copy, const lupine_htod_fragment &fragment,
    const std::shared_ptr<lupine_htod_capture_events> &capture_events,
    CUstream stream) {
  if (graph_state != nullptr) {
    graph_state->execution->note_callback();
    std::shared_ptr<lupine_htod_callback_data> callback;
    try {
      callback =
          std::make_shared<lupine_htod_callback_data>(lupine_htod_callback_data{
              ring, copy, fragment, graph_state->execution, capture_events});
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
      new (std::nothrow)
          lupine_htod_callback_data{ring, copy, fragment, nullptr, nullptr});
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

static CUresult lupine_prepare_fragment_smemcpy(
    const lupine_htod_copy &copy, const lupine_htod_fragment &fragment,
    size_t ring_offset,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    lupine_smemcpy_launch_descriptor *launch, CUfunction *function) {
  lupine_smemcpy_params params = copy.smemcpy;
  params.source = ring->device_data(ring_offset);
  params.logical_offset = fragment.logical_offset;
  params.bytes = fragment.bytes;
  CUresult result = ring->prepare_smemcpy_launch(params, launch);
  if (result == CUDA_SUCCESS && !launch->use_cuda_memcpy) {
    result = ring->smemcpy_function(*launch, function);
  }
  return result;
}

static CUresult lupine_enqueue_htod_fragment(
    const lupine_htod_copy &copy, const lupine_htod_fragment &fragment,
    const std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    CUstream stream) {
  size_t ring_offset = ring->ring_offset(copy, fragment.logical_offset);
  void *host_source = ring->data(ring_offset);
  if (copy.use_cuda_memcpy) {
    if (copy.destination_is_htoa) {
      return cuMemcpyHtoAAsync_v2(copy.destination_array,
                                  copy.destination_x + fragment.logical_offset,
                                  host_source, fragment.bytes, stream);
    }
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

  lupine_smemcpy_launch_descriptor launch = {};
  CUfunction function = nullptr;
  CUresult result = lupine_prepare_fragment_smemcpy(copy, fragment, ring_offset,
                                                    ring, &launch, &function);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (launch.use_cuda_memcpy) {
    return cuMemcpyHtoDAsync_v2(launch.params.destination, host_source,
                                launch.params.bytes, stream);
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
  for (size_t offset = 0; offset < copy.bytes;) {
    auto fragment =
        copy.fragment(offset, lupine_htod_side_effect_ring::fragment_bytes);
    lupine_smemcpy_launch_descriptor launch = {};
    CUfunction function = nullptr;
    CUresult result = lupine_prepare_fragment_smemcpy(
        copy, fragment, ring->ring_offset(copy, fragment.logical_offset), ring,
        &launch, &function);
    if (result != CUDA_SUCCESS) {
      return result;
    }
    offset += fragment.bytes;
  }
  return CUDA_SUCCESS;
}

static CUresult lupine_enqueue_client_htod_copy(
    conn_t *conn, lupine_staging_state &state, const lupine_htod_copy &copy,
    CUcontext context, CUstream stream, bool blocking, bool wait_for_source) {
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

  lupine_capture_id capture_id = lupine_captured_stream_resources(stream);
  std::shared_ptr<lupine_htod_graph_state> graph_state;
  std::shared_ptr<lupine_htod_capture_events> capture_events;
  if (capture_id != 0) {
    graph_state = lupine_htod_graph_state_for(capture_id, ring);
    if (graph_state == nullptr || graph_state->execution->ring() != ring) {
      return CUDA_ERROR_INVALID_CONTEXT;
    }
    result = lupine_make_htod_capture_events(context, capture_events);
    if (result != CUDA_SUCCESS) {
      return result;
    }
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
    result = cuEventRecord(ring->ordering_event(), stream);
    if (result == CUDA_SUCCESS) {
      result =
          cuStreamWaitEvent(ring->transfer_stream(), ring->ordering_event(), 0);
    }
    if (result != CUDA_SUCCESS) {
      ring->release_execution();
      return result;
    }
  }

  size_t offset = 0;
  while (offset < copy.bytes) {
    auto fragment =
        copy.fragment(offset, lupine_htod_side_effect_ring::fragment_bytes);
    // The callback and its copy share one private stream. The callback only
    // fetches this bounded fragment, so it always returns without waiting for
    // CUDA; stream order then makes the filled slot visible to the copy.
    result =
        lupine_enqueue_htod_callback(graph_state, ring, copy, fragment,
                                     capture_events, ring->transfer_stream());
    if (result == CUDA_SUCCESS) {
      result = lupine_enqueue_htod_fragment(copy, fragment, ring,
                                            ring->transfer_stream());
    }
    if (result != CUDA_SUCCESS) {
      if (capture_id == 0) {
        (void)ring->synchronize();
        ring->release_execution();
      }
      return result;
    }
    offset += fragment.bytes;
  }

  CUevent completion =
      capture_id == 0 ? ring->ordering_event() : capture_events->join;
  result = cuEventRecord(completion, ring->transfer_stream());
  if (result == CUDA_SUCCESS) {
    result = cuStreamWaitEvent(stream, completion, 0);
  }
  if (result != CUDA_SUCCESS) {
    if (capture_id == 0) {
      (void)ring->synchronize();
      ring->release_execution();
    }
    return result;
  }
  // The copy is queued. Tickets order submission, so a ticket the handler
  // claimed ends here rather than once the stream reaches the copy: a later
  // request from another lane must not wait behind work queued ahead of it.
  rpc_async_sequence_end(conn);

  // CUDA may stage pageable HtoD input before an asynchronous API returns.
  // Preserve that source-lifetime guarantee: outside capture, the callback
  // must finish pulling the client buffer before the caller can reuse it. A
  // page-locked source has no such guarantee natively, so its unanswered copy
  // returns once queued: blocking here would hold up every later request on
  // this lane, whose tickets other lanes wait for. The transfer stream keeps
  // the ring's slot in order; a writer from the CPU drains it first.
  if (capture_id == 0) {
    result = wait_for_source ? ring->synchronize() : CUDA_SUCCESS;
    ring->release_execution();
    return result;
  }
  return blocking ? cuStreamSynchronize(stream) : CUDA_SUCCESS;
}

static CUresult lupine_copy_client_host_to_device(conn_t *conn, CUstream stream,
                                                  bool blocking,
                                                  const lupine_htod_copy &copy,
                                                  bool wait_for_source = true) {
  // CUDA allows the copy stream to belong to a different context from the
  // caller's current one. The staging stream and its ordering events must be
  // created in that stream's context, then leave the caller's binding intact.
  CUcontext stream_context = nullptr;
  CUresult result = cuStreamGetCtx(stream, &stream_context);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  result = cuCtxPushCurrent_v2(stream_context);
  if (result != CUDA_SUCCESS) {
    return result;
  }

  lupine_staging_state *state = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  result = lupine_current_htod_context(conn, &state, &context, &device);
  lupine_staging_operation operation(result == CUDA_SUCCESS ? state : nullptr,
                                     context, device);
  if (result == CUDA_SUCCESS && !operation.acquired()) {
    result = CUDA_ERROR_INVALID_CONTEXT;
  }

  if (result == CUDA_SUCCESS) {
    result = lupine_enqueue_client_htod_copy(conn, *state, copy, context,
                                             stream, blocking, wait_for_source);
  }
  CUcontext popped = nullptr;
  CUresult restore = cuCtxPopCurrent_v2(&popped);
  return result == CUDA_SUCCESS ? restore : result;
}

// The pageable source travels in the request, so this thread reads it
// straight into the ring and never originates a server-to-client call: it
// may wait for the stream here without holding the call lock that a callback
// queued ahead of the copy could need. Page-locked sources still arrive by
// pull, read when the stream reaches the copy, because queued work may write
// them first.
static CUresult
lupine_copy_pushed_host_to_device(conn_t *conn, CUstream stream, bool blocking,
                                  const lupine_htod_copy &copy) {
  // A captured copy is replayed by the graph without the client, so it keeps
  // pulling at launch and the pushed bytes are dropped.
  if (lupine_captured_stream_resources(stream) != 0) {
    if (rpc_drain(conn, copy.bytes) < 0) {
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    return lupine_copy_client_host_to_device(conn, stream, blocking, copy);
  }

  CUcontext stream_context = nullptr;
  CUresult result = cuStreamGetCtx(stream, &stream_context);
  if (result == CUDA_SUCCESS) {
    result = cuCtxPushCurrent_v2(stream_context);
  }
  if (result != CUDA_SUCCESS) {
    return rpc_drain(conn, copy.bytes) < 0 ? CUDA_ERROR_DEVICE_UNAVAILABLE
                                           : result;
  }

  // The driver stages a small pageable source before an async copy returns,
  // without waiting for the stream, so a small linear copy goes to it
  // directly and this handler never waits on the stream.
  if (!blocking && copy.bytes <= LUPINE_INLINE_HTOD_BYTES &&
      copy.use_cuda_memcpy && copy.destination_array == nullptr &&
      copy.slices == 1 && copy.smemcpy.rows == 1) {
    unsigned char source[LUPINE_INLINE_HTOD_BYTES];
    result = rpc_read(conn, source, copy.bytes) < 0
                 ? CUDA_ERROR_DEVICE_UNAVAILABLE
                 : cuMemcpyHtoDAsync_v2(copy.smemcpy.destination, source,
                                        copy.bytes, stream);
    CUcontext popped = nullptr;
    CUresult restore = cuCtxPopCurrent_v2(&popped);
    return result == CUDA_SUCCESS ? restore : result;
  }

  lupine_staging_state *state = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  result = lupine_current_htod_context(conn, &state, &context, &device);
  lupine_staging_operation operation(result == CUDA_SUCCESS ? state : nullptr,
                                     context, device);
  if (result == CUDA_SUCCESS && !operation.acquired()) {
    result = CUDA_ERROR_INVALID_CONTEXT;
  }
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  if (result == CUDA_SUCCESS) {
    ring = lupine_prepare_htod_side_effect_ring(*state, context, result);
  }
  if (result == CUDA_SUCCESS) {
    result = lupine_prepare_htod_copy(copy, ring);
  }
  bool executing = false;
  if (result == CUDA_SUCCESS) {
    ring->acquire_execution();
    executing = true;
    // This copy writes the ring's slot from the CPU, so an unanswered pull
    // still queued on the transfer stream must leave it first.
    result = ring->synchronize();
  }
  if (result == CUDA_SUCCESS) {
    result = cuEventRecord(ring->ordering_event(), stream);
    if (result == CUDA_SUCCESS) {
      result =
          cuStreamWaitEvent(ring->transfer_stream(), ring->ordering_event(), 0);
    }
  }

  // The body is consumed whole even after a failure so the lane stays in
  // sync; fragments past the failure are read and dropped.
  size_t offset = 0;
  while (offset < copy.bytes) {
    auto fragment =
        copy.fragment(offset, lupine_htod_side_effect_ring::fragment_bytes);
    void *slot =
        ring == nullptr
            ? nullptr
            : ring->data(ring->ring_offset(copy, fragment.logical_offset));
    int read = slot == nullptr ? rpc_drain(conn, fragment.bytes)
                               : rpc_read(conn, slot, fragment.bytes);
    if (read < 0) {
      result = CUDA_ERROR_DEVICE_UNAVAILABLE;
      break;
    }
    offset += fragment.bytes;
    if (result == CUDA_SUCCESS) {
      result = lupine_enqueue_htod_fragment(copy, fragment, ring,
                                            ring->transfer_stream());
    }
    // The ring holds one fragment: the copy must leave it before the next one
    // lands.
    if (result == CUDA_SUCCESS) {
      result = ring->synchronize();
    }
  }
  if (executing) {
    (void)ring->synchronize();
    ring->release_execution();
  }
  CUcontext popped = nullptr;
  CUresult restore = cuCtxPopCurrent_v2(&popped);
  return result == CUDA_SUCCESS ? restore : result;
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

static lupine_htod_copy lupine_make_htoa_copy(CUarray destination,
                                              size_t destination_offset,
                                              const void *source,
                                              size_t bytes) {
  lupine_htod_copy copy = lupine_make_linear_htod_copy(0, source, bytes);
  copy.destination_is_htoa = true;
  copy.destination_array = destination;
  copy.destination_x = destination_offset;
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
  state.fetch_streams.erase(context);
  for (auto it = state.primary_contexts.begin();
       it != state.primary_contexts.end();) {
    it = it->second == context ? state.primary_contexts.erase(it)
                               : std::next(it);
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
  auto &reg = lupine_staging_states();
  std::lock_guard<std::mutex> lock(reg.mutex);
  auto [it, inserted] = reg.states.emplace(conn, std::move(state));
  if (inserted) {
    reg.primary.store(reg.states.size() == 1 ? it->second.get() : nullptr,
                      std::memory_order_release);
  }
  return inserted;
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

template <typename Fn>
static void lupine_server_note_context_impl(conn_t *conn, CUcontext context,
                                            CUresult result, const char *err,
                                            Fn record) {
  auto *state = (result == CUDA_SUCCESS && context != nullptr)
                    ? lupine_staging_state_for(conn)
                    : nullptr;
  if (state == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  try {
    record(*state);
  } catch (...) {
    LUPINE_LOG_ERROR(err);
  }
}

void lupine_server_note_primary_context(conn_t *conn, CUdevice device,
                                        CUcontext context, CUresult result) {
  lupine_server_note_context_impl(
      conn, context, result,
      "Failed to remember primary CUDA context for staging",
      [&](lupine_staging_state &s) { s.primary_contexts[device] = context; });
}

void lupine_server_note_created_context(conn_t *conn, CUcontext context,
                                        CUresult result) {
  lupine_server_note_context_impl(
      conn, context, result,
      "Failed to remember created CUDA context for staging",
      [&](lupine_staging_state &s) { s.created_contexts.insert(context); });
}

static void lupine_extract_context_staging_locked(
    lupine_staging_state &state, CUcontext context,
    std::shared_ptr<lupine_htod_side_effect_ring> &ring,
    std::shared_ptr<lupine_fetch_stream> &fetch_stream) {
  state.teardown_contexts.insert(context);
  if (auto it = state.htod_rings.find(context); it != state.htod_rings.end()) {
    ring = std::move(it->second);
    state.htod_rings.erase(it);
  }
  if (auto it = state.fetch_streams.find(context);
      it != state.fetch_streams.end()) {
    fetch_stream = std::move(it->second);
    state.fetch_streams.erase(it);
  }
}

void lupine_server_prepare_primary_context(conn_t *conn, CUdevice device) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  std::shared_ptr<lupine_fetch_stream> fetch_stream;
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    state->teardown_devices.insert(device);
    state->condition.wait(lock,
                          [&] { return !state->staging_operation_active; });
    auto it = state->primary_contexts.find(device);
    if (it != state->primary_contexts.end()) {
      lupine_extract_context_staging_locked(*state, it->second, ring,
                                            fetch_stream);
    }
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
    if (auto it = state->primary_contexts.find(device);
        it != state->primary_contexts.end()) {
      context = it->second;
    }
  }
  unsigned int flags = 0;
  int active = 0;
  CUresult state_result = cuDevicePrimaryCtxGetState(device, &flags, &active);

  std::unique_lock<std::mutex> lock(state->mutex);
  if (context != nullptr && (reset || result != CUDA_SUCCESS ||
                             state_result != CUDA_SUCCESS || active == 0)) {
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
  // Streams, modules, and mapped registrations belong to the live context.
  // Drop the registry's ownership while the caller still has it current.
  std::shared_ptr<lupine_htod_side_effect_ring> ring;
  std::shared_ptr<lupine_fetch_stream> fetch_stream;
  std::unique_lock<std::mutex> lock(state->mutex);
  state->teardown_contexts.insert(context);
  state->condition.wait(lock, [&] { return !state->staging_operation_active; });
  lupine_extract_context_staging_locked(*state, context, ring, fetch_stream);
}

static void lupine_server_finish_context_teardown(conn_t *conn,
                                                  CUcontext context,
                                                  bool forget) {
  auto *state = lupine_staging_state_for(conn);
  if (state == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> lock(state->mutex);
  if (forget) {
    state->condition.wait(lock,
                          [&] { return !state->staging_operation_active; });
    // Destructive APIs can return a deferred error after taking effect. Detach
    // every old handle on any result; leaking is safer than stale-handle reuse.
    lupine_server_forget_context_metadata(*state, context);
  }
  state->teardown_contexts.erase(context);
}

void lupine_server_finish_context_destroy(conn_t *conn, CUcontext context,
                                          CUresult) {
  lupine_server_finish_context_teardown(conn, context, true);
}

void lupine_server_finish_context_detach(conn_t *conn, CUcontext context,
                                         CUresult result) {
  unsigned int version = 0;
  bool forget = result != CUDA_SUCCESS ||
                cuCtxGetApiVersion(context, &version) != CUDA_SUCCESS;
  lupine_server_finish_context_teardown(conn, context, forget);
}

void lupine_server_cleanup_connection(conn_t *conn) {
  lupine_server_cleanup_log_callbacks(conn);
  std::unique_ptr<lupine_staging_state> owned_state;
  {
    auto &reg = lupine_staging_states();
    std::lock_guard<std::mutex> lock(reg.mutex);
    auto it = reg.states.find(conn);
    if (it == reg.states.end()) {
      return;
    }
    reg.primary.store(nullptr, std::memory_order_release);
    owned_state = std::move(it->second);
    reg.states.erase(it);
    if (reg.states.size() == 1) {
      reg.primary.store(reg.states.begin()->second.get(),
                        std::memory_order_release);
    }
  }
}

int lupine_write_lifecycle_response(conn_t *conn, int request_id,
                                    CUresult result) {
  return lupine_write_response(conn, request_id, result);
}

static int lupine_serve_memcpy_htod(conn_t *conn, bool async) {
  CUdeviceptr destination = 0;
  const void *source = nullptr;
  // The client waits for no answer to a copy it sends fire-and-forget.
  bool fire_and_forget = !rpc_request_awaited(conn);
  bool is_server_authoritative = false;
  uint64_t pushed_bytes = 0;
  size_t bytes = 0;
  CUstream stream = CU_STREAM_LEGACY;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &destination, sizeof(destination)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0) ||
      rpc_read(conn, &source, sizeof(source)) < 0 ||
      rpc_read(conn, &pushed_bytes, sizeof(pushed_bytes)) < 0) {
    return -1;
  }
  if (pushed_bytes != 0 && pushed_bytes != bytes) {
    return -1;
  }
  // Only a synchronous pageable copy, a page-locked async source, or a small
  // pushed async copy is sent fire-and-forget. It holds its ticket until the
  // copy is queued, not until the stream reaches it, and a failure reaches the
  // client from its next synchronize.
  if (fire_and_forget &&
      (!async ? pushed_bytes != bytes
              : (bytes == 0 || pushed_bytes > LUPINE_INLINE_HTOD_BYTES))) {
    return -1;
  }
  if (pushed_bytes != 0) {
    result = lupine_copy_pushed_host_to_device(
        conn, stream, !async && !fire_and_forget,
        lupine_make_linear_htod_copy(destination, source, bytes));
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (bytes != 0 && is_server_authoritative) {
    result =
        (async || fire_and_forget)
            ? cuMemcpyAsync(destination, reinterpret_cast<CUdeviceptr>(source),
                            bytes, stream)
            : cuMemcpy(destination, reinterpret_cast<CUdeviceptr>(source),
                       bytes);
  } else if (bytes != 0 && pushed_bytes == 0) {
    result = lupine_copy_client_host_to_device(
        conn, stream, !async,
        lupine_make_linear_htod_copy(destination, source, bytes),
        !fire_and_forget);
  }
  // An unanswered copy's failure reaches the client from its next
  // synchronize, as a launch's does.
  if (lupine_unanswered(conn, result)) {
    return 0;
  }
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_cuMemcpyHtoD_v2(conn_t *conn) {
  return lupine_serve_memcpy_htod(conn, false);
}

// Chunks of one striped copy arrive on the bulk connections in any order and
// only stage here; the device copy runs on the main lane, which keeps it FIFO
// behind the client's earlier work on that lane.
struct lupine_bulk_staging {
  unsigned char *data = nullptr;
  size_t total = 0;
  size_t landed = 0;
};

static std::mutex lupine_bulk_staging_mutex;
static std::condition_variable lupine_bulk_staging_progress;
static std::unordered_map<uint64_t, lupine_bulk_staging> lupine_bulk_stagings;
static bool lupine_bulk_connection_lost = false;

void lupine_server_bulk_connection_lost() {
  std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
  lupine_bulk_connection_lost = true;
  lupine_bulk_staging_progress.notify_all();
}

int handle_lupineBulkChunk(conn_t *conn) {
  uint64_t copy_id = 0;
  uint64_t total = 0;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  if (!lupine_read_all(conn, copy_id, total, offset, bytes)) {
    return -1;
  }
  unsigned char *destination = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
    lupine_bulk_staging &staging = lupine_bulk_stagings[copy_id];
    if (staging.data == nullptr) {
      staging.data = static_cast<unsigned char *>(malloc(total));
      if (staging.data == nullptr) {
        std::abort();
      }
      staging.total = total;
    }
    destination = staging.data + offset;
  }
  if (rpc_read(conn, destination, bytes) < 0 || rpc_read_end(conn) < 0) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
  lupine_bulk_staging &staging = lupine_bulk_stagings[copy_id];
  staging.landed += bytes;
  if (staging.landed == staging.total) {
    lupine_bulk_staging_progress.notify_all();
  }
  return 0;
}

int handle_lupineMemcpyHtoDBulk(conn_t *conn) {
  uint64_t copy_id = 0;
  uint64_t destination = 0;
  size_t bytes = 0;
  // A host destination is the server's copy of a client host allocation, the
  // same memory lupineManagedHostFlush writes.
  uint8_t to_host = 0;
  int request_id = -1;
  if (!lupine_read_all(conn, copy_id, destination, bytes, to_host) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = CUDA_ERROR_DEVICE_UNAVAILABLE;
  unsigned char *data = nullptr;
  {
    std::unique_lock<std::mutex> lock(lupine_bulk_staging_mutex);
    lupine_bulk_staging_progress.wait(lock, [&] {
      auto it = lupine_bulk_stagings.find(copy_id);
      return lupine_bulk_connection_lost ||
             (it != lupine_bulk_stagings.end() &&
              it->second.landed == it->second.total);
    });
    auto it = lupine_bulk_stagings.find(copy_id);
    if (it != lupine_bulk_stagings.end()) {
      if (it->second.landed == it->second.total) {
        data = it->second.data;
      } else {
        free(it->second.data);
      }
      lupine_bulk_stagings.erase(it);
    }
  }
  if (data != nullptr && to_host != 0) {
    memcpy(reinterpret_cast<void *>(destination), data, bytes);
    result = CUDA_SUCCESS;
  } else if (data != nullptr) {
    result =
        cuMemcpyHtoD_v2(static_cast<CUdeviceptr>(destination), data, bytes);
  }
  free(data);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

// A striped DtoH holds the caller's lane until every reader it announced has
// finished, so the copy stays FIFO there. Each reader, one per bulk connection,
// claims chunks from the shared counter, copies them on the caller's context
// and stream, and streams them back; a connection whose window has collapsed
// simply claims fewer.
struct lupine_bulk_read {
  bool ready = false;
  CUcontext context = nullptr;
  CUdeviceptr source = 0;
  // A pushed delivery's bytes, already on the host, instead of `source`.
  const unsigned char *host = nullptr;
  size_t total = 0;
  CUstream stream = CU_STREAM_LEGACY;
  size_t next = 0;
  unsigned int arrived = 0;
  unsigned int finished = 0;
  CUresult result = CUDA_SUCCESS;
};

static std::unordered_map<uint64_t, lupine_bulk_read> lupine_bulk_reads;
static uint64_t lupine_retired_push_bulk_number = 0;
static bool lupine_push_bulk_closed = false;

static int lupine_write_bulk_read_frame(conn_t *conn, int request_id,
                                        uint64_t offset, const void *data,
                                        uint64_t bytes) {
  return lupine_write_response(conn, request_id, offset, bytes,
                               lupine_raw_bytes{data, bytes});
}

int handle_lupineBulkRead(conn_t *conn) {
  uint64_t copy_id = 0;
  if (rpc_read(conn, &copy_id, sizeof(copy_id)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  constexpr uint64_t kPushBulkBit = uint64_t{1} << 62;
  bool push_bulk = (copy_id & kPushBulkBit) != 0;
  std::unique_ptr<unsigned char[]> host;
  if (!push_bulk) {
    host.reset(
        new (std::nothrow) unsigned char[LUPINE_RPC_TRANSFER_CHUNK_BYTES]);
  }
  int status = 0;
  std::unique_lock<std::mutex> lock(lupine_bulk_staging_mutex);
  if (push_bulk &&
      (lupine_push_bulk_closed ||
       (copy_id & ~kPushBulkBit) <= lupine_retired_push_bulk_number)) {
    lock.unlock();
    return lupine_write_bulk_read_frame(conn, request_id, 0, nullptr, 0);
  }
  lupine_bulk_read &read = lupine_bulk_reads[copy_id];
  ++read.arrived;
  lupine_bulk_staging_progress.wait(lock, [&] {
    return read.ready || lupine_bulk_connection_lost ||
           (push_bulk && lupine_push_bulk_closed);
  });
  if (read.ready && read.result == CUDA_SUCCESS && read.host == nullptr) {
    if (!host) {
      read.result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      CUcontext context = read.context;
      lock.unlock();
      CUresult result = cuCtxSetCurrent(context);
      lock.lock();
      if (result != CUDA_SUCCESS && read.result == CUDA_SUCCESS) {
        read.result = result;
      }
    }
  }
  while (read.ready && !lupine_bulk_connection_lost &&
         read.result == CUDA_SUCCESS && read.next < read.total) {
    size_t offset = read.next;
    size_t bytes =
        std::min<size_t>(LUPINE_RPC_TRANSFER_CHUNK_BYTES, read.total - offset);
    read.next += bytes;
    CUdeviceptr source = read.source + offset;
    const unsigned char *pushed = read.host;
    CUstream stream = read.stream;
    lock.unlock();
    // host is pageable, so this returns once the chunk has landed, ordered
    // behind whatever was already queued on the caller's stream.
    CUresult result =
        pushed != nullptr
            ? CUDA_SUCCESS
            : cuMemcpyDtoHAsync_v2(host.get(), source, bytes, stream);
    if (result == CUDA_SUCCESS) {
      status = lupine_write_bulk_read_frame(
          conn, request_id, offset,
          pushed != nullptr ? pushed + offset : host.get(), bytes);
    }
    lock.lock();
    if (result != CUDA_SUCCESS && read.result == CUDA_SUCCESS) {
      read.result = result;
    }
    if (status < 0) {
      break;
    }
  }
  ++read.finished;
  lupine_bulk_staging_progress.notify_all();
  lock.unlock();
  if (status < 0) {
    return -1;
  }
  return lupine_write_bulk_read_frame(conn, request_id, 0, nullptr, 0);
}

// The push lane hands a large delivery's bytes to the client's bulk reader.
// It gives up when the session or a bulk connection goes, once a reader that
// already started is done with the bytes.
static void lupine_serve_host_bulk_read(conn_t *conn, uint64_t copy_id,
                                        const void *host, size_t bytes) {
  constexpr uint64_t kPushBulkBit = uint64_t{1} << 62;
  std::unique_lock<std::mutex> lock(lupine_bulk_staging_mutex);
  lupine_bulk_read &read = lupine_bulk_reads[copy_id];
  read.ready = true;
  read.host = static_cast<const unsigned char *>(host);
  read.total = bytes;
  lupine_bulk_staging_progress.notify_all();
  lupine_bulk_staging_progress.wait(lock, [&] {
    return read.finished != 0 ||
           ((conn->closed || lupine_bulk_connection_lost) &&
            read.finished == read.arrived);
  });
  lupine_retired_push_bulk_number =
      std::max(lupine_retired_push_bulk_number, copy_id & ~kPushBulkBit);
  lupine_bulk_reads.erase(copy_id);
}

static void lupine_open_push_bulk() {
  std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
  lupine_push_bulk_closed = false;
  lupine_retired_push_bulk_number = 0;
}

static void lupine_close_push_bulk() {
  std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
  lupine_push_bulk_closed = true;
  lupine_bulk_staging_progress.notify_all();
}

static bool lupine_bulk_lost() {
  std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
  return lupine_bulk_connection_lost;
}

static void lupine_bulk_wake() {
  std::lock_guard<std::mutex> lock(lupine_bulk_staging_mutex);
  lupine_bulk_staging_progress.notify_all();
}

int handle_lupineMemcpyDtoHBulk(conn_t *conn) {
  uint64_t copy_id = 0;
  CUdeviceptr source = 0;
  size_t bytes = 0;
  CUstream stream = CU_STREAM_LEGACY;
  uint32_t readers = 0;
  int request_id = -1;
  if (!lupine_read_all(conn, copy_id, source, bytes, stream, readers) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  {
    std::unique_lock<std::mutex> lock(lupine_bulk_staging_mutex);
    lupine_bulk_read &read = lupine_bulk_reads[copy_id];
    read.ready = true;
    read.context = context;
    read.source = source;
    read.total = bytes;
    read.stream = stream;
    read.result = result;
    lupine_bulk_staging_progress.notify_all();
    lupine_bulk_staging_progress.wait(lock, [&] {
      return read.finished == readers ||
             (lupine_bulk_connection_lost && read.finished == read.arrived);
    });
    result = read.result;
    if (result == CUDA_SUCCESS && read.next < read.total) {
      result = CUDA_ERROR_DEVICE_UNAVAILABLE;
    }
    lupine_bulk_reads.erase(copy_id);
  }
  uint64_t delivered = lupine_dtoh_watermark(conn);
  return lupine_write_response(conn, request_id, delivered, result);
}

// Each chunk carries the deliveries numbered once it was copied: a readback
// waited for its stream, so pushed copies that stream ran land first.
static int lupine_write_dtoh_chunk_response(conn_t *conn, int request_id,
                                            CUresult result, const void *data,
                                            size_t bytes) {
  uint64_t delivered = lupine_dtoh_watermark(conn);
  return lupine_write_response(
      conn, request_id, delivered, result,
      lupine_raw_bytes{data, result == CUDA_SUCCESS ? bytes : 0});
}

// Sends [offset, bytes) as LUPINE_RPC_TRANSFER_CHUNK_BYTES chunks staged one at
// a time, each led by its CUresult; a failed chunk ends the response.
template <typename CopyChunk>
static int lupine_write_dtoh_chunks(conn_t *conn, int request_id, size_t bytes,
                                    size_t offset, CopyChunk copy_chunk) {
  if (offset > bytes) {
    return -1;
  }

  size_t staging_size =
      std::min(bytes - offset, (size_t)LUPINE_RPC_TRANSFER_CHUNK_BYTES);
  unsigned char inline_host[64 * 1024];
  struct reusable_staging_buffer {
    std::unique_ptr<unsigned char[]> data;
    size_t capacity = 0;
    bool in_use = false;
  };
  static thread_local reusable_staging_buffer tls_staging;
  struct staging_lease {
    reusable_staging_buffer *tls = nullptr;
    ~staging_lease() {
      if (tls != nullptr) {
        tls->in_use = false;
      }
    }
  } lease;
  std::unique_ptr<unsigned char[]> fallback_host;
  unsigned char *host = nullptr;
  if (staging_size != 0) {
    if (staging_size <= sizeof(inline_host)) {
      host = inline_host;
    } else if (!tls_staging.in_use) {
      if (tls_staging.capacity < staging_size) {
        std::unique_ptr<unsigned char[]> grown(
            new (std::nothrow) unsigned char[staging_size]);
        if (!grown) {
          return lupine_write_dtoh_chunk_response(
              conn, request_id, CUDA_ERROR_OUT_OF_MEMORY, nullptr, 0);
        }
        tls_staging.data = std::move(grown);
        tls_staging.capacity = staging_size;
      }
      tls_staging.in_use = true;
      lease.tls = &tls_staging;
      host = tls_staging.data.get();
    } else {
      fallback_host.reset(new (std::nothrow) unsigned char[staging_size]);
      if (!fallback_host) {
        return lupine_write_dtoh_chunk_response(
            conn, request_id, CUDA_ERROR_OUT_OF_MEMORY, nullptr, 0);
      }
      host = fallback_host.get();
    }
  }

  do {
    size_t chunk = std::min(bytes - offset, staging_size);
    CUresult result = copy_chunk(chunk == 0 ? nullptr : host, offset, chunk);
    if (lupine_write_dtoh_chunk_response(conn, request_id, result, host,
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

static int lupine_copy_dtoh_serial(conn_t *conn, int request_id,
                                   CUdeviceptr source, size_t bytes,
                                   size_t offset, CUstream stream) {
  return lupine_write_dtoh_chunks(
      conn, request_id, bytes, offset,
      [&](void *destination, size_t chunk_offset, size_t chunk) {
        // host is pageable, so this returns only once the chunk has landed,
        // ordered behind whatever was already queued on the caller's stream.
        return cuMemcpyDtoHAsync_v2(destination, source + chunk_offset, chunk,
                                    stream);
      });
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
  auto fallback = [&]() -> int {
    CUresult cleanup = lupine_cleanup_dtoh_pipeline(pipeline, stream, true);
    if (cleanup != CUDA_SUCCESS) {
      return lupine_write_dtoh_chunk_response(conn, request_id, cleanup,
                                              nullptr, 0);
    }
    *fallback_offset = sent_offset;
    return 1;
  };

  for (auto &slot : pipeline.slots) {
    if (submitted_offset == bytes) {
      break;
    }
    CUresult result = submit(slot, submitted_offset);
    if (event_record_failed) {
      return fallback();
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
      return fallback();
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
        return fallback();
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

static std::shared_ptr<lupine_fetch_stream>
lupine_prepare_fetch_stream(conn_t *conn, CUresult *out_result) {
  lupine_staging_state *state = nullptr;
  CUcontext context = nullptr;
  CUdevice device = 0;
  CUresult result =
      lupine_current_htod_context(conn, &state, &context, &device);
  if (result != CUDA_SUCCESS) {
    *out_result = result;
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->teardown_contexts.count(context) != 0 ||
        state->teardown_devices.count(device) != 0) {
      *out_result = CUDA_ERROR_INVALID_CONTEXT;
      return nullptr;
    }
    lupine_note_inferred_primary_context(*state, context, device);
    auto existing = state->fetch_streams.find(context);
    if (existing != state->fetch_streams.end()) {
      *out_result = CUDA_SUCCESS;
      return existing->second;
    }
  }

  CUstream raw_stream = nullptr;
  result = cuStreamCreate(&raw_stream, CU_STREAM_NON_BLOCKING);
  if (result != CUDA_SUCCESS) {
    *out_result = result;
    return nullptr;
  }
  std::shared_ptr<lupine_fetch_stream> created(
      new (std::nothrow) lupine_fetch_stream{context, raw_stream});
  if (created == nullptr) {
    (void)cuStreamDestroy_v2(raw_stream);
    *out_result = CUDA_ERROR_OUT_OF_MEMORY;
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(state->mutex);
  if (state->teardown_contexts.count(context) != 0 ||
      state->teardown_devices.count(device) != 0) {
    *out_result = CUDA_ERROR_INVALID_CONTEXT;
    return nullptr;
  }
  auto &slot = state->fetch_streams[context];
  if (slot == nullptr) {
    slot = created;
  }
  *out_result = CUDA_SUCCESS;
  return slot;
}

int handle_cuMemcpyDtoH_v2(conn_t *conn) {
  CUdeviceptr source = 0;
  size_t bytes = 0;
  // The stream the caller's copy was queued on. A synchronous cuMemcpyDtoH
  // sends the legacy stream; an async copy into pageable memory sends its own,
  // so the chunks stay ordered behind that stream's prior work.
  CUstream stream = CU_STREAM_LEGACY;
  int request_id = -1;
  if (!lupine_read_all(conn, source, bytes, stream) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  uintptr_t stream_bits = reinterpret_cast<uintptr_t>(stream);
  bool fetch_sentinel = stream_bits == LUPINE_STREAM_FETCH_READY_BITS ||
                        stream_bits == LUPINE_STREAM_FETCH_SYNC_ALL_BITS;
  struct relaxed_capture_guard {
    bool active = false;
    CUstreamCaptureMode previous = CU_STREAM_CAPTURE_MODE_RELAXED;
    explicit relaxed_capture_guard(bool enable) : active(enable) {
      if (active) {
        (void)cuThreadExchangeStreamCaptureMode(&previous);
      }
    }
    ~relaxed_capture_guard() {
      if (active) {
        (void)cuThreadExchangeStreamCaptureMode(&previous);
      }
    }
  } capture_guard(fetch_sentinel);

  std::shared_ptr<lupine_fetch_stream> fetch_stream;
  if (fetch_sentinel) {
    if (stream_bits == LUPINE_STREAM_FETCH_SYNC_ALL_BITS) {
      CUresult sync_result = cuCtxSynchronize();
      if (sync_result != CUDA_SUCCESS) {
        return lupine_write_dtoh_chunk_response(conn, request_id, sync_result,
                                                nullptr, 0);
      }
    }
    CUresult stream_result = CUDA_SUCCESS;
    fetch_stream = lupine_prepare_fetch_stream(conn, &stream_result);
    if (fetch_stream == nullptr) {
      return lupine_write_dtoh_chunk_response(conn, request_id, stream_result,
                                              nullptr, 0);
    }
    stream = fetch_stream->stream;
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

using lupine_native_memcpy3d = CUresult (*)(const CUDA_MEMCPY3D &copy,
                                            CUstream stream);
static int lupine_push_pitched_dtoh(conn_t *conn,
                                    lupine_native_memcpy3d native);

// Every 2D and 3D copy arrives promoted to the 3D descriptor. Host-to-host
// requests read completed pinned backing; other host sides are staged densely.
static int lupine_handle_memcpy3d(conn_t *conn, bool async, bool peer,
                                  lupine_native_memcpy3d native) {
  uint8_t direction = LUPINE_COPY_DIRECTION_DTOD;
  if (rpc_read(conn, &direction, sizeof(direction)) < 0) {
    return -1;
  }
  if (direction == LUPINE_COPY_DIRECTION_DTOH && async && !peer) {
    return lupine_push_pitched_dtoh(conn, native);
  }
  bool is_server_authoritative = false;
  CUDA_MEMCPY3D copy = {};
  CUstream stream = CU_STREAM_LEGACY;
  lupine_inline_buf<unsigned char, 64 * 1024> host_buf;
  unsigned char *host_data = nullptr;
  size_t host_size = 0;
  auto allocate_host = [&](size_t size) -> bool {
    host_size = size;
    host_data = host_buf.alloc(size);
    return host_data != nullptr;
  };
  if ((direction == LUPINE_COPY_DIRECTION_HTOD && !peer &&
       rpc_read(conn, &is_server_authoritative,
                sizeof(is_server_authoritative)) < 0) ||
      rpc_read(conn, &copy, sizeof(copy)) < 0) {
    return -1;
  }
  if (direction == LUPINE_COPY_DIRECTION_HTOD && peer) {
    if (!allocate_host(lupine_pack_host_source(copy)) ||
        (host_size != 0 && rpc_read(conn, host_data, host_size) < 0)) {
      return -1;
    }
    copy.srcHost = host_data;
  }
  if (direction == LUPINE_COPY_DIRECTION_DTOH) {
    if (!allocate_host(lupine_pack_host_destination(copy))) {
      return -1;
    }
    copy.dstHost = host_data;
  }
  int request_id = -1;
  if ((async && rpc_read(conn, &stream, sizeof(stream)) < 0) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  // Read completed CPU backing without entering CUDA, which may be locked by
  // cuMemFree while it waits for the host callback requesting these bytes.
  if (direction == LUPINE_COPY_DIRECTION_HTOH) {
    const auto *source = static_cast<const char *>(copy.srcHost) +
                         copy.srcZ * copy.srcHeight * copy.srcPitch +
                         copy.srcY * copy.srcPitch + copy.srcXInBytes;
    CUresult result = CUDA_SUCCESS;
    return rpc_write_start_response(conn, request_id) < 0 ||
                   rpc_write(conn, &result, sizeof(result)) < 0 ||
                   rpc_write_pitched(conn, source, copy.WidthInBytes,
                                     copy.Height, copy.srcPitch, copy.Depth,
                                     copy.srcHeight * copy.srcPitch) < 0 ||
                   rpc_write_end(conn) < 0
               ? -1
               : 0;
  }

  CUresult result =
      direction == LUPINE_COPY_DIRECTION_HTOD && !peer &&
              !is_server_authoritative
          ? lupine_copy_client_host_to_device(conn, stream, !async,
                                              lupine_make_3d_htod_copy(copy))
          : native(copy, stream);
  // A readback waited for its stream: pushed copies that stream ran land
  // first.
  uint64_t delivered = lupine_dtoh_watermark(conn);
  return rpc_write_start_response(conn, request_id) < 0 ||
                 (direction == LUPINE_COPY_DIRECTION_DTOH &&
                  rpc_write(conn, &delivered, sizeof(delivered)) < 0) ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 (direction == LUPINE_COPY_DIRECTION_DTOH &&
                  result == CUDA_SUCCESS &&
                  rpc_write(conn, host_data, host_size) < 0) ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

template <CUresult (*Fn)(const CUDA_MEMCPY2D *)>
static CUresult lupine_call_memcpy2d(const CUDA_MEMCPY3D &copy, CUstream) {
  CUDA_MEMCPY2D flat = lupine_memcpy2d_of(copy);
  return Fn(&flat);
}

int handle_cuMemcpy2D_v2(conn_t *conn) {
  return lupine_handle_memcpy3d(conn, false, false,
                                lupine_call_memcpy2d<cuMemcpy2D_v2>);
}

int handle_cuMemcpy2DUnaligned_v2(conn_t *conn) {
  return lupine_handle_memcpy3d(conn, false, false,
                                lupine_call_memcpy2d<cuMemcpy2DUnaligned_v2>);
}

int handle_cuMemcpy2DAsync_v2(conn_t *conn) {
  return lupine_handle_memcpy3d(conn, true, false,
                                [](const CUDA_MEMCPY3D &c, CUstream s) {
                                  CUDA_MEMCPY2D flat = lupine_memcpy2d_of(c);
                                  return cuMemcpy2DAsync_v2(&flat, s);
                                });
}

int handle_cuMemcpy3D_v2(conn_t *conn) {
  return lupine_handle_memcpy3d(
      conn, false, false,
      [](const CUDA_MEMCPY3D &c, CUstream) { return cuMemcpy3D_v2(&c); });
}

int handle_cuMemcpy3DAsync_v2(conn_t *conn) {
  return lupine_handle_memcpy3d(conn, true, false,
                                [](const CUDA_MEMCPY3D &c, CUstream s) {
                                  return cuMemcpy3DAsync_v2(&c, s);
                                });
}

int handle_cuMemcpy3DPeer(conn_t *conn) {
  return lupine_handle_memcpy3d(conn, false, true,
                                [](const CUDA_MEMCPY3D &c, CUstream) {
                                  auto p = lupine_memcpy3d_peer_of(c);
                                  return cuMemcpy3DPeer(&p);
                                });
}

int handle_cuMemcpy3DPeerAsync(conn_t *conn) {
  return lupine_handle_memcpy3d(conn, true, true,
                                [](const CUDA_MEMCPY3D &c, CUstream s) {
                                  auto p = lupine_memcpy3d_peer_of(c);
                                  return cuMemcpy3DPeerAsync(&p, s);
                                });
}

int handle_cuMemcpyHtoDAsync_v2(conn_t *conn) {
  return lupine_serve_memcpy_htod(conn, true);
}

static int lupine_serve_memcpy_htoa(conn_t *conn, bool async) {
  CUarray dstArray = nullptr;
  size_t dstOffset = 0;
  const void *srcHost = nullptr;
  bool is_server_authoritative = false;
  uint64_t pushed_bytes = 0;
  size_t byteCount = 0;
  CUstream stream = CU_STREAM_LEGACY;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &is_server_authoritative,
               sizeof(is_server_authoritative)) < 0 ||
      rpc_read(conn, &dstArray, sizeof(dstArray)) < 0 ||
      rpc_read(conn, &dstOffset, sizeof(dstOffset)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0) ||
      rpc_read(conn, &srcHost, sizeof(srcHost)) < 0 ||
      rpc_read(conn, &pushed_bytes, sizeof(pushed_bytes)) < 0) {
    return -1;
  }
  if (pushed_bytes != 0 && pushed_bytes != byteCount) {
    return -1;
  }
  if (pushed_bytes != 0) {
    result = lupine_copy_pushed_host_to_device(
        conn, stream, !async,
        lupine_make_htoa_copy(dstArray, dstOffset, srcHost, byteCount));
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (byteCount == 0 || dstArray == nullptr || is_server_authoritative) {
    result = async ? cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost,
                                          byteCount, stream)
                   : cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, byteCount);
  } else if (pushed_bytes == 0) {
    result = lupine_copy_client_host_to_device(
        conn, stream, !async,
        lupine_make_htoa_copy(dstArray, dstOffset, srcHost, byteCount));
  }
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_cuMemcpyHtoA_v2(conn_t *conn) {
  return lupine_serve_memcpy_htoa(conn, false);
}

int handle_cuMemcpyHtoAAsync_v2(conn_t *conn) {
  return lupine_serve_memcpy_htoa(conn, true);
}

static int lupine_serve_memcpy_atoh(conn_t *conn, bool async) {
  CUarray srcArray = nullptr;
  size_t srcOffset = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  return lupine_write_dtoh_chunks(
      conn, request_id, byteCount, 0,
      [&](void *destination, size_t offset, size_t chunk) {
        if (!async) {
          return cuMemcpyAtoH_v2(destination, srcArray, srcOffset + offset,
                                 chunk);
        }
        CUresult result = cuMemcpyAtoHAsync_v2(
            destination, srcArray, srcOffset + offset, chunk, stream);
        return result == CUDA_SUCCESS ? cuStreamSynchronize(stream) : result;
      });
}

int handle_cuMemcpyAtoH_v2(conn_t *conn) {
  return lupine_serve_memcpy_atoh(conn, false);
}

int handle_cuMemcpyAtoHAsync_v2(conn_t *conn) {
  return lupine_serve_memcpy_atoh(conn, true);
}

int handle_cuMemcpyDtoHAsync_v2(conn_t *conn) {
  void *dstHost = nullptr;
  void *server_host = nullptr;
  uint8_t refresh = 0;
  CUdeviceptr srcDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;

  if (!lupine_read_all(conn, dstHost, server_host, refresh, srcDevice,
                       byteCount, stream) ||
      rpc_read_end(conn) < 0) {
    return -1;
  }
  // Unanswered: a failure reaches the client from its next synchronize.
  // Memory the client refreshes on access sets `refresh`, writing only the
  // server's copy and telling the client to mark `dstHost` stale; other
  // page-locked memory pushes to its writable alias, updating `server_host` on
  // the stream when device code can read it.
  lupine_keep_async_error(
      conn, lupine_push_dtoh(conn, stream, {dstHost, byteCount}, srcDevice,
                             server_host, refresh != 0));
  return 0;
}

// A pitched device-to-host copy packs its rows directly into dense pinned host
// staging and leaves as a pushed linear copy; the client scatters the rows
// back to its own pitch.
static int lupine_push_pitched_dtoh(conn_t *conn,
                                    lupine_native_memcpy3d native) {
  CUDA_MEMCPY3D copy = {};
  CUstream stream = nullptr;
  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 || rpc_read_end(conn) < 0) {
    return -1;
  }
  lupine_host_rows client = lupine_host_destination_rows(copy);
  lupine_pack_host_destination(copy);
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = nullptr;
  if (client.bytes() == 0) {
    return 0;
  }
  CUresult result = lupine_push_dtoh_with(
      conn, stream, client,
      [&](void *dst, CUstream s) {
        copy.dstHost = dst;
        return native(copy, s);
      },
      nullptr, false);
  lupine_keep_async_error(conn, result);
  return 0;
}
