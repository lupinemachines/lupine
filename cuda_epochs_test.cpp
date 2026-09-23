#include "cuda_client_epochs.h"
#include "rpc.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
conn_t connection = {};
thread_local CUcontext current = reinterpret_cast<CUcontext>(1);

void require(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

void publish() {
  rpc_epoch_publish(&connection, rpc_epoch_prepare(&connection, true));
}

bool has_dependencies() {
  return !rpc_epoch_prepare(&connection, false).required.empty();
}
} // namespace

// Routing is independent of the ordering contract tested here.
extern "C" CUcontext lupine_current_context_hint() { return current; }
extern "C" conn_t *lupine_rpc_conn_for_stream(CUstream) { return &connection; }
extern "C" conn_t *lupine_rpc_conn_for_context(CUcontext) {
  return &connection;
}
extern "C" conn_t *lupine_rpc_conn_for_event(CUevent) { return &connection; }

int main() {
  require(rpc_conn_init(&connection, LUPINE_INVALID_SOCKET, 0) == 0, "init");
  auto blocking = reinterpret_cast<CUstream>(10);
  auto other_blocking = reinterpret_cast<CUstream>(11);
  auto nonblocking = reinterpret_cast<CUstream>(12);
  auto event = reinterpret_cast<CUevent>(20);
  lupine_cuda_stream_created(blocking, &connection, current, 0);
  lupine_cuda_stream_created(other_blocking, &connection, current, 0);
  lupine_cuda_stream_created(nonblocking, &connection, current,
                             CU_STREAM_NON_BLOCKING);
  {
    auto call = lupine_cuda_stream_call(blocking, event, true);
    require(!has_dependencies(), "first submission has prerequisites");
    publish();
  }
  {
    auto call = lupine_cuda_stream_call(other_blocking);
    require(!has_dependencies(),
            "ordinary blocking streams serialized each other");
  }
  {
    auto call = lupine_cuda_context_call();
    require(has_dependencies(), "context synchronization missed stream work");
  }
  {
    auto call = lupine_cuda_event_call({event});
    require(has_dependencies(), "event synchronization missed record");
  }
  {
    auto call = lupine_cuda_stream_call(nullptr);
    require(has_dependencies(), "legacy stream missed blocking stream work");
    publish();
  }
  {
    auto call = lupine_cuda_stream_call(other_blocking);
    require(has_dependencies(), "blocking stream missed legacy work");
  }
  lupine_cuda_stream_created(other_blocking, &connection, current,
                             CU_STREAM_NON_BLOCKING);
  {
    auto call = lupine_cuda_stream_call(other_blocking);
    require(!has_dependencies(), "reused stream kept cached blocking flags");
  }
  {
    auto call = lupine_cuda_stream_call(nonblocking);
    require(!has_dependencies(),
            "nonblocking stream acquired a legacy dependency");
    publish();
  }
  current = reinterpret_cast<CUcontext>(2);
  {
    auto call = lupine_cuda_context_call();
    require(!has_dependencies(), "context synchronization crossed contexts");
  }
  {
    auto call = lupine_cuda_context_call(current, false, &connection, {event});
    require(has_dependencies(),
            "context wait missed an event from another context");
  }
  {
    auto call = lupine_cuda_stream_call(nonblocking);
    require(has_dependencies(), "explicit stream lost its owning context");
  }
  {
    auto call = lupine_cuda_stream_call(CU_STREAM_PER_THREAD);
    require(!has_dependencies(), "new per-thread stream has prerequisites");
    publish();
  }
  std::thread other([&] {
    current = reinterpret_cast<CUcontext>(2);
    auto call = lupine_cuda_stream_call(CU_STREAM_PER_THREAD);
    require(!has_dependencies(), "per-thread streams shared an epoch");
  });
  other.join();
  {
    auto call = lupine_cuda_stream_call(CU_STREAM_PER_THREAD);
    require(has_dependencies(), "per-thread stream lost its own work");
  }
  lupine_cuda_epochs_forget_connection(&connection);
  rpc_conn_destroy(&connection);
  std::puts("cuda_epochs_test: PASS");
}
