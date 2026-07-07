#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <thread>
#include <unordered_map>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <unistd.h>
#endif

#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "lupine_log.h"
#include "manual_server.h"
#include "rpc.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

static void lupine_log_manual_handler_error(const char *name) {
  LUPINE_LOG_ERROR("Error handling manual " << name << " request.");
}

struct lupine_manual_handler {
  RequestHandler handler;
  const char *name;
};

// Manual handlers are looked up before the auto-generated handlers from
// get_handler(), so entries here take precedence for overlapping ops.
static const std::unordered_map<int, lupine_manual_handler> &
lupine_manual_handlers() {
  static const std::unordered_map<int, lupine_manual_handler> handlers = {
      {LUPINE_RPC_cuGetExportTableMetadata,
       {handle_manual_cuGetExportTableMetadata, "cuGetExportTable metadata"}},
      {LUPINE_RPC_cuPrivateGetModuleNode,
       {handle_manual_cuPrivateGetModuleNode, "private module node"}},
      {RPC_cuModuleLoad, {handle_manual_cuModuleLoad, "cuModuleLoad"}},
      {RPC_cuModuleLoadData,
       {handle_manual_cuModuleLoadData, "cuModuleLoadData"}},
      {RPC_cuLibraryLoadData,
       {handle_manual_cuLibraryLoadData, "cuLibraryLoadData"}},
      {RPC_cuCtxCreate_v2, {handle_manual_cuCtxCreate_v2, "cuCtxCreate_v2"}},
      {RPC_cuMemPoolSetAttribute,
       {handle_manual_cuMemPoolSetAttribute, "cuMemPoolSetAttribute"}},
      {RPC_cuMemPoolGetAttribute,
       {handle_manual_cuMemPoolGetAttribute, "cuMemPoolGetAttribute"}},
      {RPC_cuPointerGetAttribute,
       {handle_manual_cuPointerGetAttribute, "cuPointerGetAttribute"}},
      {RPC_cuPointerGetAttributes,
       {handle_manual_cuPointerGetAttributes, "cuPointerGetAttributes"}},
      {RPC_cuLinkCreate_v2, {handle_manual_cuLinkCreate_v2, "cuLinkCreate_v2"}},
      {RPC_cuLinkAddData_v2,
       {handle_manual_cuLinkAddData_v2, "cuLinkAddData_v2"}},
      {RPC_cuLinkAddFile_v2,
       {handle_manual_cuLinkAddFile_v2, "cuLinkAddFile_v2"}},
      {RPC_cuLinkComplete, {handle_manual_cuLinkComplete, "cuLinkComplete"}},
      {RPC_cuLinkDestroy, {handle_manual_cuLinkDestroy, "cuLinkDestroy"}},
      {RPC_cuMemcpy3D_v2, {handle_manual_cuMemcpy3D_v2, "cuMemcpy3D_v2"}},
      {RPC_cuMemcpy2D_v2, {handle_manual_cuMemcpy2D_v2, "cuMemcpy2D_v2"}},
      {RPC_cuMemcpy2DUnaligned_v2,
       {handle_manual_cuMemcpy2DUnaligned_v2, "cuMemcpy2DUnaligned_v2"}},
      {RPC_cuMemcpy2DAsync_v2,
       {handle_manual_cuMemcpy2DAsync_v2, "cuMemcpy2DAsync_v2"}},
      {RPC_cuMemcpyDtoH_v2, {handle_manual_cuMemcpyDtoH_v2, "cuMemcpyDtoH_v2"}},
      {RPC_cuMemcpyAtoH_v2, {handle_manual_cuMemcpyAtoH_v2, "cuMemcpyAtoH_v2"}},
      {RPC_cuDeviceGetGraphMemAttribute,
       {handle_manual_cuDeviceGetGraphMemAttribute,
        "cuDeviceGetGraphMemAttribute"}},
      {RPC_cuDeviceSetGraphMemAttribute,
       {handle_manual_cuDeviceSetGraphMemAttribute,
        "cuDeviceSetGraphMemAttribute"}},
      {RPC_cuLibraryGetModule,
       {handle_manual_cuLibraryGetModule, "cuLibraryGetModule"}},
      {RPC_cuLibraryUnload, {handle_manual_cuLibraryUnload, "cuLibraryUnload"}},
      {RPC_cuModuleGetGlobal_v2,
       {handle_manual_cuModuleGetGlobal_v2, "cuModuleGetGlobal_v2"}},
      {LUPINE_RPC_cuFuncGetParamLayout,
       {handle_manual_cuFuncGetParamLayout, "cuFuncGetParamLayout"}},
      {RPC_cuOccupancyMaxPotentialBlockSize,
       {[](conn_t *conn) {
          return handle_manual_cuOccupancyMaxPotentialBlockSize(conn, false);
        },
        "cuOccupancyMaxPotentialBlockSize"}},
      {RPC_cuOccupancyMaxPotentialBlockSizeWithFlags,
       {[](conn_t *conn) {
          return handle_manual_cuOccupancyMaxPotentialBlockSize(conn, true);
        },
        "cuOccupancyMaxPotentialBlockSizeWithFlags"}},
      {RPC_cuLaunchKernel, {handle_manual_cuLaunchKernel, "cuLaunchKernel"}},
      {RPC_cuLaunchCooperativeKernel,
       {handle_manual_cuLaunchCooperativeKernel, "cuLaunchCooperativeKernel"}},
      {RPC_cuGraphAddKernelNode_v2,
       {handle_manual_cuGraphAddKernelNode, "cuGraphAddKernelNode"}},
      {RPC_cuGraphKernelNodeGetParams_v2,
       {handle_manual_cuGraphKernelNodeGetParams,
        "cuGraphKernelNodeGetParams_v2"}},
      {RPC_cuGraphKernelNodeSetParams_v2,
       {handle_manual_cuGraphKernelNodeSetParams,
        "cuGraphKernelNodeSetParams_v2"}},
      {RPC_cuGraphAddMemcpyNode,
       {handle_manual_cuGraphAddMemcpyNode, "cuGraphAddMemcpyNode"}},
      {RPC_cuGraphAddMemsetNode,
       {handle_manual_cuGraphAddMemsetNode, "cuGraphAddMemsetNode"}},
      {RPC_cuGraphAddHostNode,
       {handle_manual_cuGraphAddHostNode, "cuGraphAddHostNode"}},
      {RPC_cuGraphExecKernelNodeSetParams_v2,
       {handle_manual_cuGraphExecKernelNodeSetParams,
        "cuGraphExecKernelNodeSetParams_v2"}},
      {LUPINE_RPC_cuGraphConditionalHandleCreate,
       {handle_manual_cuGraphConditionalHandleCreate,
        "cuGraphConditionalHandleCreate"}},
      {LUPINE_RPC_cuGraphAddNode_v2,
       {handle_manual_cuGraphAddNode, "cuGraphAddNode"}},
      {RPC_cuGraphLaunch, {handle_manual_cuGraphLaunch, "cuGraphLaunch"}},
      {RPC_cuGraphGetEdges_v2,
       {handle_manual_cuGraphGetEdges, "cuGraphGetEdges"}},
      {RPC_cuGraphNodeGetDependencies_v2,
       {handle_manual_cuGraphNodeGetDependencies,
        "cuGraphNodeGetDependencies"}},
      {RPC_cuGraphNodeGetDependentNodes_v2,
       {handle_manual_cuGraphNodeGetDependentNodes,
        "cuGraphNodeGetDependentNodes"}},
      {LUPINE_RPC_cuMemPrefetchAsync,
       {handle_manual_cuMemPrefetchAsync, "cuMemPrefetchAsync"}},
      {RPC_cuGraphHostNodeGetParams,
       {handle_manual_cuGraphHostNodeGetParams, "cuGraphHostNodeGetParams"}},
      {RPC_cuGraphHostNodeSetParams,
       {handle_manual_cuGraphHostNodeSetParams, "cuGraphHostNodeSetParams"}},
      {RPC_cuGraphExecHostNodeSetParams,
       {handle_manual_cuGraphExecHostNodeSetParams,
        "cuGraphExecHostNodeSetParams"}},
      {RPC_cuLaunchHostFunc,
       {handle_manual_cuLaunchHostFunc, "cuLaunchHostFunc"}},
      {RPC_cuStreamAddCallback,
       {handle_manual_cuStreamAddCallback, "cuStreamAddCallback"}},
      {RPC_cuEventRecord,
       {[](conn_t *conn) { return handle_manual_cuEventRecord(conn, false); },
        "cuEventRecord"}},
      {RPC_cuEventRecordWithFlags,
       {[](conn_t *conn) { return handle_manual_cuEventRecord(conn, true); },
        "cuEventRecordWithFlags"}},
      {RPC_cuEventQuery, {handle_manual_cuEventQuery, "cuEventQuery"}},
      {RPC_cuStreamWaitEvent,
       {handle_manual_cuStreamWaitEvent, "cuStreamWaitEvent"}},
      {LUPINE_RPC_cuStreamBeginCaptureToGraph,
       {handle_manual_cuStreamBeginCaptureToGraph,
        "cuStreamBeginCaptureToGraph"}},
      {RPC_cuStreamUpdateCaptureDependencies_v2,
       {handle_manual_cuStreamUpdateCaptureDependencies,
        "cuStreamUpdateCaptureDependencies"}},
      {LUPINE_RPC_cuStreamGetCaptureInfo_v3,
       {handle_manual_cuStreamGetCaptureInfo, "cuStreamGetCaptureInfo"}},
      {RPC_cuStreamBeginCapture_v2,
       {handle_manual_cuStreamBeginCapture, "cuStreamBeginCapture"}},
      {RPC_cuStreamEndCapture,
       {handle_manual_cuStreamEndCapture, "cuStreamEndCapture"}},
      {RPC_cuGraphClone, {handle_manual_cuGraphClone, "cuGraphClone"}},
      {RPC_cuGraphInstantiateWithFlags,
       {handle_manual_cuGraphInstantiateWithFlags,
        "cuGraphInstantiateWithFlags"}},
      {RPC_cuGraphInstantiateWithParams,
       {handle_manual_cuGraphInstantiateWithParams,
        "cuGraphInstantiateWithParams"}},
      {RPC_cuGraphExecDestroy,
       {handle_manual_cuGraphExecDestroy, "cuGraphExecDestroy"}},
      {RPC_cuGraphDestroy, {handle_manual_cuGraphDestroy, "cuGraphDestroy"}},
      {RPC_cuMemcpyHtoD_v2, {handle_manual_cuMemcpyHtoD_v2, "cuMemcpyHtoD_v2"}},
      {RPC_cuMemcpyHtoDAsync_v2,
       {handle_manual_cuMemcpyHtoDAsync_v2, "cuMemcpyHtoDAsync_v2"}},
      {RPC_cuMemcpyDtoHAsync_v2,
       {handle_manual_cuMemcpyDtoHAsync_v2, "cuMemcpyDtoHAsync_v2"}},
      {RPC_cuCtxSynchronize,
       {handle_manual_cuCtxSynchronize, "cuCtxSynchronize"}},
      {RPC_cuStreamSynchronize,
       {handle_manual_cuStreamSynchronize, "cuStreamSynchronize"}},
      {RPC_cuEventSynchronize,
       {handle_manual_cuEventSynchronize, "cuEventSynchronize"}},
  };
  return handlers;
}

static int lupine_handle_rpc_request(conn_t *conn, int op) {
  LUPINE_TRACE_LOG("LUPINE server handling op " << op);

  const auto &manual_handlers = lupine_manual_handlers();
  auto manual = manual_handlers.find(op);
  if (manual != manual_handlers.end()) {
    if (manual->second.handler(conn) < 0) {
      lupine_log_manual_handler_error(manual->second.name);
      return -1;
    }
    return 0;
  }

  auto opHandler = get_handler(op);
  if (opHandler == nullptr) {
    LUPINE_LOG_ERROR("No RPC handler for op " << op << "; closing client.");
    return -1;
  }
  if (opHandler(conn) < 0) {
    LUPINE_LOG_ERROR("Error handling request.");
    return -1;
  }
  return 0;
}

static size_t lupine_max_lanes() {
  const char *value = getenv("LUPINE_MAX_LANES");
  if (value == nullptr || value[0] == '\0') {
    return 64;
  }
  char *end = nullptr;
  long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed < 1) {
    return 64;
  }
  return static_cast<size_t>(parsed);
}

struct lupine_lane {
  uint32_t id = 0;
  std::mutex mutex;
  std::condition_variable cond;
  std::deque<rpc_frame> queue;
  bool stopping = false;
  bool finished = false;
  std::thread worker;
};

struct lupine_lane_map {
  conn_t *conn = nullptr;
  size_t max_lanes = 0;
  std::mutex mutex;
  std::unordered_map<uint32_t, std::shared_ptr<lupine_lane>> lanes;
};

static void lupine_run_lane(conn_t *conn, std::shared_ptr<lupine_lane> lane) {
  for (;;) {
    rpc_frame frame;
    {
      std::unique_lock<std::mutex> lock(lane->mutex);
      lane->cond.wait(lock,
                      [&]() { return lane->stopping || !lane->queue.empty(); });
      if (lane->queue.empty()) {
        break;
      }
      frame = std::move(lane->queue.front());
      lane->queue.pop_front();
    }

    int op = frame.op;
    if (op == LUPINE_RPC_RELEASE_LANE) {
      if (rpc_activate_frame(conn, std::move(frame)) == 0) {
        rpc_read_end(conn);
      }
      break;
    }
    if (rpc_activate_frame(conn, std::move(frame)) < 0 ||
        lupine_handle_rpc_request(conn, op) < 0) {
      conn->closed = 1;
      pthread_cond_broadcast(&conn->read_cond);
      break;
    }
  }

  std::lock_guard<std::mutex> lock(lane->mutex);
  lane->finished = true;
}

static void lupine_cleanup_finished_lanes(lupine_lane_map *lane_map) {
  for (auto it = lane_map->lanes.begin(); it != lane_map->lanes.end();) {
    std::shared_ptr<lupine_lane> lane = it->second;
    bool finished = false;
    {
      std::lock_guard<std::mutex> lock(lane->mutex);
      finished = lane->finished;
    }
    if (!finished) {
      ++it;
      continue;
    }
    if (lane->worker.joinable()) {
      lane->worker.join();
    }
    it = lane_map->lanes.erase(it);
  }
}

static int lupine_enqueue_lane(lupine_lane_map *lane_map, rpc_frame &&frame) {
  std::shared_ptr<lupine_lane> lane;
  {
    std::lock_guard<std::mutex> lock(lane_map->mutex);
    lupine_cleanup_finished_lanes(lane_map);
    auto it = lane_map->lanes.find(frame.lane_id);
    if (it == lane_map->lanes.end()) {
      if (lane_map->lanes.size() >= lane_map->max_lanes) {
        LUPINE_LOG_ERROR("Too many active RPC lanes; set LUPINE_MAX_LANES "
                         "higher if this workload needs more threads.");
        return -1;
      }
      lane = std::make_shared<lupine_lane>();
      lane->id = frame.lane_id;
      conn_t *conn = lane_map->conn;
      lane->worker =
          std::thread([conn, lane]() { lupine_run_lane(conn, lane); });
      lane_map->lanes.emplace(frame.lane_id, lane);
    } else {
      lane = it->second;
    }
  }

  {
    std::lock_guard<std::mutex> lock(lane->mutex);
    lane->queue.push_back(std::move(frame));
  }
  lane->cond.notify_one();
  return 0;
}

static void lupine_stop_lanes(lupine_lane_map *lane_map) {
  std::unordered_map<uint32_t, std::shared_ptr<lupine_lane>> lanes;
  {
    std::lock_guard<std::mutex> lock(lane_map->mutex);
    lanes.swap(lane_map->lanes);
  }
  for (auto &entry : lanes) {
    auto &lane = entry.second;
    {
      std::lock_guard<std::mutex> lock(lane->mutex);
      lane->stopping = true;
    }
    lane->cond.notify_one();
  }
  for (auto &entry : lanes) {
    auto &lane = entry.second;
    if (lane->worker.joinable()) {
      lane->worker.join();
    }
  }
}

void client_handler(lupine_socket_t connfd) {
  conn_t conn = {};
  conn.connfd = connfd;
  conn.request_id = 1;
  conn.local_request_parity = conn.request_id & 1;
  if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.write_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.call_mutex, NULL) < 0 ||
      pthread_cond_init(&conn.read_cond, NULL) < 0 ||
      rpc_connection_state_init(&conn) < 0 ||
      rpc_http2_server_init(&conn) < 0) {
    LUPINE_LOG_ERROR("Error initializing mutex.");
    return;
  }

  printf("Client connected.\n");

  lupine_lane_map lanes = {&conn, lupine_max_lanes()};
  while (!conn.closed) {
    rpc_frame frame;
    if (rpc_read_wire_frame(&conn, &frame) < 0) {
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }
    if (frame.op == -1) {
      if (rpc_deliver_response_frame(&conn, std::move(frame)) < 0) {
        break;
      }
      continue;
    }
    if (lupine_enqueue_lane(&lanes, std::move(frame)) < 0) {
      rpc_frame_ready(&conn);
      break;
    }
    if (rpc_frame_ready(&conn) < 0) {
      break;
    }
  }

  conn.closed = 1;
  pthread_cond_broadcast(&conn.read_cond);
  lupine_stop_lanes(&lanes);
  lupine_socket_close(connfd);
  if (conn.rpc_thread != 0) {
    pthread_join(conn.rpc_thread, nullptr);
    conn.rpc_thread = 0;
  }

  if (pthread_mutex_destroy(&conn.read_mutex) != 0 ||
      pthread_mutex_destroy(&conn.write_mutex) != 0 ||
      pthread_mutex_destroy(&conn.call_mutex) != 0 ||
      pthread_cond_destroy(&conn.read_cond) != 0)
    LUPINE_LOG_ERROR("Error destroying mutex.");

  rpc_write_queue_free(&conn);
  rpc_connection_state_free(&conn);
}

int main() {
  int port = DEFAULT_PORT;
  struct sockaddr_in servaddr, cli;
  if (lupine_socket_init() < 0) {
    printf("Socket initialization failed.\n");
    exit(EXIT_FAILURE);
  }

  lupine_socket_t sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == LUPINE_INVALID_SOCKET) {
    printf("Socket creation failed.\n");
    exit(EXIT_FAILURE);
  }

  char *p = getenv("LUPINE_PORT");

  if (p == NULL) {
    port = DEFAULT_PORT;
  } else {
    // Validate LUPINE_PORT so a typo (e.g. "14833x" or "") can't silently
    // fall back to atoi's 0, which the kernel would reinterpret as an
    // ephemeral port.
    char *end = nullptr;
    long parsed = strtol(p, &end, 10);
    if (end == p || *end != '\0' || parsed < 1 || parsed > 65535) {
      printf("Invalid LUPINE_PORT '%s'; expected 1-65535.\n", p);
      exit(EXIT_FAILURE);
    }
    port = static_cast<int>(parsed);
  }

  // Bind the socket
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(port);

  if (lupine_socket_set_reuseaddr(sockfd) < 0) {
    printf("Socket bind failed.\n");
    exit(EXIT_FAILURE);
  }

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
    printf("Socket bind failed.\n");
    exit(EXIT_FAILURE);
  }

  if (listen(sockfd, MAX_CLIENTS) != 0) {
    printf("Listen failed.\n");
    exit(EXIT_FAILURE);
  }

  printf("Server listening on port %d...\n", port);

#ifndef _WIN32
  // reap exited connection processes automatically
  signal(SIGCHLD, SIG_IGN);
#endif

  // Server loop
  while (1) {
    socklen_t len = sizeof(cli);
    lupine_socket_t connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

    if (connfd == LUPINE_INVALID_SOCKET) {
      LUPINE_LOG_ERROR("Server accept failed.");
      continue;
    }

#ifndef _WIN32
    int flag = 1;
    setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
#endif

#ifndef _WIN32
    // fork a process per connection so each client gets its own CUDA driver
    // state (primary context, allocations, modules). this matches local
    // semantics: a client resetting or corrupting its context cannot affect
    // other clients, and everything is released when the client disconnects.
    // the parent must never initialize CUDA; forked children cannot use a
    // parent's initialized driver.
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
      LUPINE_LOG_ERROR("Server fork failed.");
      lupine_socket_close(connfd);
      continue;
    }
    if (pid == 0) {
      lupine_socket_close(sockfd);
      client_handler(connfd);
      exit(0);
    }
    lupine_socket_close(connfd);
#else
    // Windows has no fork; connections share the server process.
    std::thread client_thread(client_handler, connfd);

    // detach the thread so it runs independently
    client_thread.detach();
#endif
  }

  lupine_socket_close(sockfd);
  return 0;
}
