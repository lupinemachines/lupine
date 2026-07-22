#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "checkpoint.h"
#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "copy_pipeline.h"
#include "lupine_log.h"
#include "manual_server.h"
#include "rpc.h"
#include "server_checkpoint.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10
#define MAX_LANES 256

#ifndef _WIN32
static volatile sig_atomic_t lupine_parent_termination_requested = 0;
static volatile sig_atomic_t lupine_parent_child_exited = 0;

static void lupine_parent_sigterm_handler(int) {
  lupine_parent_termination_requested = 1;
}

static void lupine_parent_sigchld_handler(int) {
  lupine_parent_child_exited = 1;
}

static bool lupine_install_parent_signal_handlers() {
  struct sigaction term_action = {};
  term_action.sa_handler = lupine_parent_sigterm_handler;
  sigemptyset(&term_action.sa_mask);

  struct sigaction child_action = {};
  child_action.sa_handler = lupine_parent_sigchld_handler;
  sigemptyset(&child_action.sa_mask);

  return sigaction(SIGTERM, &term_action, nullptr) == 0 &&
         sigaction(SIGCHLD, &child_action, nullptr) == 0;
}

static void
lupine_reap_connection_children(std::unordered_set<pid_t> &children) {
  sigset_t child_mask;
  sigset_t previous_mask;
  sigemptyset(&child_mask);
  sigaddset(&child_mask, SIGCHLD);
  bool signal_blocked =
      sigprocmask(SIG_BLOCK, &child_mask, &previous_mask) == 0;

  for (;;) {
    int status = 0;
    pid_t child = waitpid(-1, &status, WNOHANG);
    if (child <= 0) {
      break;
    }
    children.erase(child);
  }
  lupine_parent_child_exited = 0;
  if (signal_blocked) {
    (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
  }
}
#endif

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
      {RPC_cuDevicePrimaryCtxRetain,
       {handle_manual_cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"}},
      {RPC_cuDevicePrimaryCtxRelease_v2,
       {handle_manual_cuDevicePrimaryCtxRelease_v2,
        "cuDevicePrimaryCtxRelease_v2"}},
      {RPC_cuDevicePrimaryCtxReset_v2,
       {handle_manual_cuDevicePrimaryCtxReset_v2,
        "cuDevicePrimaryCtxReset_v2"}},
      {RPC_cuCtxAttach, {handle_manual_cuCtxAttach, "cuCtxAttach"}},
      {RPC_cuCtxDestroy_v2, {handle_manual_cuCtxDestroy_v2, "cuCtxDestroy_v2"}},
      {RPC_cuCtxDetach, {handle_manual_cuCtxDetach, "cuCtxDetach"}},
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
      {RPC_cuLaunchKernelEx,
       {handle_manual_cuLaunchKernelEx, "cuLaunchKernelEx"}},
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
      {LUPINE_RPC_lupineManagedHostFlush,
       {handle_manual_lupineManagedHostFlush, "lupineManagedHostFlush"}},
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

struct lupine_lane {
  uint64_t id = 0;
  std::mutex mutex;
  std::condition_variable cond;
  bool ready = false;
  int op = 0;
  std::thread worker;
};

int client_handler(lupine_socket_t connfd) {
  conn_t conn = {};
  conn.connfd = connfd;
  conn.request_id = 1;
  conn.local_request_parity = conn.request_id & 1;
  if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.write_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.call_mutex, NULL) < 0 ||
      pthread_cond_init(&conn.read_cond, NULL) < 0 ||
      rpc_http2_server_init(&conn) < 0) {
    LUPINE_LOG_ERROR("Error initializing mutex.");
    return lupine_server_checkpoint_child_finish();
  }
  if (!lupine_server_initialize_connection(&conn)) {
    LUPINE_LOG_ERROR("Error initializing per-connection staging state.");
    lupine_socket_close(connfd);
    rpc_conn_destroy(&conn);
    return lupine_server_checkpoint_child_finish();
  }

  LUPINE_LOG_DEBUG("Client connected.");

  std::unordered_map<uint64_t, std::shared_ptr<lupine_lane>> lanes;
  bool connection_ready = false;
  while (!conn.closed) {
    if (pthread_mutex_lock(&conn.read_mutex) != 0) {
      break;
    }
    while (conn.read_id != 0 && !conn.closed) {
      pthread_cond_wait(&conn.read_cond, &conn.read_mutex);
    }
    if (conn.closed) {
      pthread_mutex_unlock(&conn.read_mutex);
      break;
    }

    int request_id = 0;
    if (rpc_read(&conn, &request_id, sizeof(request_id)) !=
            sizeof(request_id) ||
        request_id == 0) {
      pthread_mutex_unlock(&conn.read_mutex);
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }

    conn.read_id = request_id;
    if (request_id % 2 == conn.local_request_parity) {
      if (pthread_cond_broadcast(&conn.read_cond) < 0 ||
          pthread_mutex_unlock(&conn.read_mutex) < 0) {
        break;
      }
      continue;
    }

    if (rpc_read(&conn, &conn.read_lane_id, sizeof(conn.read_lane_id)) !=
            sizeof(conn.read_lane_id) ||
        rpc_read(&conn, &conn.read_op, sizeof(conn.read_op)) !=
            sizeof(conn.read_op)) {
      pthread_mutex_unlock(&conn.read_mutex);
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }
    uint64_t lane_id = conn.read_lane_id;
    int op = conn.read_op;

    if (!connection_ready) {
      if (!lupine_server_checkpoint_connection_ready(
              rpc_http2_session_id(&conn))) {
        pthread_mutex_unlock(&conn.read_mutex);
        LUPINE_LOG_ERROR("Failed to restore connection checkpoint.");
        break;
      }
      connection_ready = true;
    }

    std::shared_ptr<lupine_lane> lane;
    auto it = lanes.find(lane_id);
    if (it == lanes.end()) {
      if (lanes.size() >= MAX_LANES) {
        pthread_mutex_unlock(&conn.read_mutex);
        LUPINE_LOG_ERROR("Too many active RPC lanes.");
        break;
      }
      lane = std::make_shared<lupine_lane>();
      lane->id = lane_id;
      lane->worker = std::thread([&conn, lane]() {
        for (;;) {
          int op = 0;
          {
            std::unique_lock<std::mutex> lock(lane->mutex);
            lane->cond.wait(lock, [&lane]() { return lane->ready; });
            op = lane->op;
            lane->ready = false;
          }

          conn.read_lane_id = lane->id;
          conn.read_op = op;
          if (conn.read_id == -1 || op == LUPINE_RPC_TERMINATE_LANE) {
            rpc_read_end(&conn);
            return;
          }
          {
            lupine_checkpoint::cuda_call_guard dispatch_guard;
            if (lupine_handle_rpc_request(&conn, op) >= 0) {
              continue;
            }
          }
          rpc_read_end(&conn);
          return;
        }
      });
      lanes.emplace(lane_id, lane);
    } else {
      lane = it->second;
    }

    {
      std::lock_guard<std::mutex> lock(lane->mutex);
      lane->op = op;
      lane->ready = true;
    }
    lane->cond.notify_one();
    if (pthread_cond_broadcast(&conn.read_cond) < 0 ||
        pthread_mutex_unlock(&conn.read_mutex) < 0) {
      break;
    }

    if (op == LUPINE_RPC_TERMINATE_LANE) {
      if (lane->worker.joinable()) {
        lane->worker.join();
      }
      lanes.erase(lane_id);
    }
  }

  for (auto &entry : lanes) {
    auto &lane = entry.second;
    if (pthread_mutex_lock(&conn.read_mutex) != 0) {
      break;
    }
    while (conn.read_id != 0) {
      pthread_cond_wait(&conn.read_cond, &conn.read_mutex);
    }
    conn.read_id = -1;
    {
      std::lock_guard<std::mutex> lock(lane->mutex);
      lane->op = LUPINE_RPC_TERMINATE_LANE;
      lane->ready = true;
    }
    lane->cond.notify_one();
    if (pthread_cond_broadcast(&conn.read_cond) < 0 ||
        pthread_mutex_unlock(&conn.read_mutex) < 0) {
      break;
    }
    if (lane->worker.joinable()) {
      lane->worker.join();
    }
  }

  conn.closed = 1;
  pthread_cond_broadcast(&conn.read_cond);
  lupine_socket_close(connfd);
  if (conn.rpc_thread != 0) {
    pthread_join(conn.rpc_thread, nullptr);
    conn.rpc_thread = 0;
  }

  // Preserve the connection's CUDA state through checkpointing. Cleanup can
  // release CUDA-owned staging resources and must happen afterward.
  int checkpoint_result = lupine_server_checkpoint_child_finish();
  lupine_server_cleanup_connection(&conn);
  rpc_conn_destroy(&conn);
  return checkpoint_result;
}

int main() {
  int port = DEFAULT_PORT;
  struct sockaddr_in servaddr, cli;
  if (lupine_socket_init() < 0) {
    LUPINE_LOG_ERROR("Socket initialization failed.");
    exit(EXIT_FAILURE);
  }

  lupine_socket_t sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == LUPINE_INVALID_SOCKET) {
    LUPINE_LOG_ERROR("Socket creation failed.");
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
      LUPINE_LOG_ERROR("Invalid LUPINE_PORT '" << p << "'; expected 1-65535.");
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
    LUPINE_LOG_ERROR("Socket bind failed.");
    exit(EXIT_FAILURE);
  }

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
    LUPINE_LOG_ERROR("Socket bind failed.");
    exit(EXIT_FAILURE);
  }

  if (listen(sockfd, MAX_CLIENTS) != 0) {
    LUPINE_LOG_ERROR("Listen failed.");
    exit(EXIT_FAILURE);
  }

  LUPINE_LOG_DEBUG("Server listening on port " << port << "...");

#ifndef _WIN32
  if (!lupine_install_parent_signal_handlers()) {
    LUPINE_LOG_ERROR("Failed to install server signal handlers.");
    lupine_socket_close(sockfd);
    exit(EXIT_FAILURE);
  }
  std::unordered_set<pid_t> connection_children;
#endif

  // Server loop
  while (1) {
#ifndef _WIN32
    if (lupine_parent_child_exited != 0) {
      lupine_reap_connection_children(connection_children);
    }
    if (lupine_parent_termination_requested != 0) {
      break;
    }
#endif
    socklen_t len = sizeof(cli);
    lupine_socket_t connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

    if (connfd == LUPINE_INVALID_SOCKET) {
#ifndef _WIN32
      if (lupine_parent_child_exited != 0) {
        lupine_reap_connection_children(connection_children);
      }
      if (lupine_parent_termination_requested != 0) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
#endif
      LUPINE_LOG_ERROR("Server accept failed.");
      continue;
    }

#ifndef _WIN32
    if (lupine_parent_termination_requested != 0) {
      lupine_socket_close(connfd);
      break;
    }
#endif

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

    sigset_t term_mask;
    sigset_t previous_mask;
    sigemptyset(&term_mask);
    sigaddset(&term_mask, SIGTERM);
    if (sigprocmask(SIG_BLOCK, &term_mask, &previous_mask) != 0) {
      LUPINE_LOG_ERROR("Failed to block SIGTERM around server fork.");
      lupine_socket_close(connfd);
      continue;
    }

    pid_t pid = fork();
    if (pid < 0) {
      (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
      LUPINE_LOG_ERROR("Server fork failed.");
      lupine_socket_close(connfd);
      continue;
    }
    if (pid == 0) {
      struct sigaction child_action = {};
      child_action.sa_handler = SIG_DFL;
      sigemptyset(&child_action.sa_mask);
      (void)sigaction(SIGCHLD, &child_action, nullptr);

      lupine_socket_close(sockfd);
      if (!lupine_server_checkpoint_child_start(connfd)) {
        LUPINE_LOG_ERROR("Failed to initialize graceful child shutdown.");
        lupine_socket_close(connfd);
        exit(EXIT_FAILURE);
      }
      (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
      int checkpoint_result = client_handler(connfd);
      exit(checkpoint_result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    connection_children.insert(pid);
    (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
    lupine_socket_close(connfd);
#else
    // Windows has no fork; connections share the server process.
    std::thread client_thread(client_handler, connfd);

    // detach the thread so it runs independently
    client_thread.detach();
#endif
  }

  lupine_socket_close(sockfd);

#ifndef _WIN32
  // Every connection owns its CUDA state in a dedicated child, so each child
  // must quiesce and optionally checkpoint itself.
  lupine_reap_connection_children(connection_children);
  for (pid_t child : connection_children) {
    (void)kill(child, SIGTERM);
  }

  int shutdown_result = EXIT_SUCCESS;
  while (!connection_children.empty()) {
    int status = 0;
    pid_t child = waitpid(-1, &status, 0);
    if (child > 0) {
      connection_children.erase(child);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
        shutdown_result = EXIT_FAILURE;
      }
      continue;
    }
    if (child < 0 && errno == EINTR) {
      continue;
    }
    if (child < 0 && errno == ECHILD) {
      connection_children.clear();
      break;
    }
    shutdown_result = EXIT_FAILURE;
    break;
  }
  return shutdown_result;
#else
  return 0;
#endif
}
