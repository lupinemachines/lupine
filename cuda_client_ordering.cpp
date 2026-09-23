#include "cuda_client_ordering.h"
#include "client_routing.h"
#include "rpc.h"

#include <atomic>
#include <map>
#include <mutex>

namespace {

enum domain_kind : uint64_t {
  stream_domain,
  context_domain,
  blocking_domain,
  event_domain,
  connection_domain
};

struct stream_ordering {
  rpc_ordering_domain *stream;
  rpc_ordering_domain *context;
  rpc_ordering_domain *blocking;
  rpc_ordering_domain *legacy;
  rpc_ordering_domain *connection;
  bool nonblocking;
  bool is_legacy;
};

struct ordering_state {
  std::mutex mutex;
  std::atomic<uint64_t> generation{1};
  std::map<std::pair<conn_t *, CUstream>, std::pair<CUcontext, unsigned int>>
      streams;
};

ordering_state &state() {
  static auto *value = new ordering_state;
  return *value;
}

uint64_t thread_identity() {
  static std::atomic<uint64_t> next{1};
  static thread_local uint64_t id =
      next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

stream_ordering stream_scope(conn_t *conn, CUstream stream) {
  auto &s = state();
  // Repeated launches on a thread's current stream need no map lookup.
  // Stream creation/reuse and connection cleanup invalidate this metadata.
  struct cached_scope {
    conn_t *conn = nullptr;
    CUstream stream = nullptr;
    CUcontext current = nullptr;
    uint64_t generation = 0;
    stream_ordering value = {};
  };
  static thread_local cached_scope cached;
  CUcontext current = lupine_current_context_hint();
  uint64_t generation = s.generation.load(std::memory_order_acquire);
  if (cached.conn == conn && cached.stream == stream &&
      cached.current == current && cached.generation == generation) {
    return cached.value;
  }
  std::lock_guard<std::mutex> lock(s.mutex);
  cached = {
      conn, stream, current, s.generation.load(std::memory_order_relaxed), {}};
  CUcontext context = current;
  unsigned int flags = 0;
  auto known = s.streams.find({conn, stream});
  if (known != s.streams.end()) {
    context = known->second.first;
    flags = known->second.second;
  }
  bool legacy = stream == nullptr || stream == CU_STREAM_LEGACY;
  if (legacy) {
    stream = CU_STREAM_LEGACY;
  }
  uint64_t lane = stream == CU_STREAM_PER_THREAD ? thread_identity() : 0;
  auto ctx = reinterpret_cast<uintptr_t>(context);
  stream_ordering value = {
      rpc_dependency_domain(conn, stream_domain,
                            reinterpret_cast<uintptr_t>(stream), ctx, lane),
      rpc_dependency_domain(conn, context_domain, ctx),
      rpc_dependency_domain(conn, blocking_domain, ctx),
      rpc_dependency_domain(conn, stream_domain,
                            reinterpret_cast<uintptr_t>(CU_STREAM_LEGACY), ctx),
      rpc_dependency_domain(conn, connection_domain, 0),
      (flags & CU_STREAM_NON_BLOCKING) != 0,
      legacy};
  return cached.value = value;
}

} // namespace

void lupine_cuda_stream_created(CUstream stream, conn_t *conn,
                                CUcontext context, unsigned int flags) {
  if (conn == nullptr || stream == nullptr) {
    return;
  }
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.streams[{conn, stream}] = {context, flags};
  // CUDA may reuse a destroyed stream's handle with different flags.
  s.generation.fetch_add(1, std::memory_order_release);
}

rpc_dependency_call lupine_cuda_stream_call(CUstream stream, CUevent event,
                                            bool record_event,
                                            conn_t *connection) {
  conn_t *conn =
      connection != nullptr ? connection : lupine_rpc_conn_for_stream(stream);
  if (conn == nullptr) {
    return rpc_dependency_call(nullptr, {});
  }
  auto scope = stream_scope(conn, stream);
  std::vector<rpc_ordering_domain *> required{scope.stream};
  std::vector<rpc_ordering_domain *> published{scope.stream, scope.context,
                                               scope.connection};
  if (!scope.nonblocking) {
    required.push_back(scope.is_legacy ? scope.blocking : scope.legacy);
    published.push_back(scope.blocking);
  }
  if (event != nullptr) {
    auto domain = rpc_dependency_domain(conn, event_domain,
                                        reinterpret_cast<uintptr_t>(event));
    required.push_back(domain);
    if (record_event) {
      published.push_back(domain);
    }
  }
  return rpc_dependency_call(conn, required, std::move(published));
}

rpc_dependency_call
lupine_cuda_context_call(CUcontext context, bool all_contexts,
                         conn_t *connection,
                         const std::vector<CUevent> &events) {
  if (context == nullptr) {
    context = lupine_current_context_hint();
  }
  conn_t *conn =
      connection != nullptr ? connection : lupine_rpc_conn_for_context(context);
  auto *domain =
      all_contexts
          ? rpc_dependency_domain(conn, connection_domain, 0)
          : rpc_dependency_domain(conn, context_domain,
                                  reinterpret_cast<uintptr_t>(context));
  std::vector<rpc_ordering_domain *> required{domain};
  for (CUevent event : events) {
    required.push_back(rpc_dependency_domain(
        conn, event_domain, reinterpret_cast<uintptr_t>(event)));
  }
  return rpc_dependency_call(conn, required);
}

rpc_dependency_call lupine_cuda_event_call(const std::vector<CUevent> &events,
                                           conn_t *connection) {
  conn_t *conn = connection;
  if (conn == nullptr && !events.empty()) {
    conn = lupine_rpc_conn_for_event(events.front());
  }
  std::vector<rpc_ordering_domain *> required;
  for (CUevent event : events) {
    required.push_back(rpc_dependency_domain(
        conn, event_domain, reinterpret_cast<uintptr_t>(event)));
  }
  return rpc_dependency_call(conn, required);
}

void lupine_cuda_ordering_forget_connection(conn_t *conn) {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  for (auto it = s.streams.begin(); it != s.streams.end();) {
    it = it->first.first == conn ? s.streams.erase(it) : std::next(it);
  }
  s.generation.fetch_add(1, std::memory_order_release);
}
