#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <unordered_map>

#ifndef _WIN32
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "lupine_log.h"
#include "manual_server.h"
#include "rpc.h"
#include "snapshot.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

static constexpr int LUPINE_RPC_cuFuncGetParamLayout = 1000001;
static constexpr int LUPINE_RPC_cuCtxCreate_v2 = 1000002;
static constexpr int LUPINE_RPC_cuMemPoolSetAttribute = 1000003;
static constexpr int LUPINE_RPC_cuMemPoolGetAttribute = 1000004;
static constexpr int LUPINE_RPC_cuLaunchHostFunc = 1000005;
static constexpr int LUPINE_RPC_cuPointerGetAttribute = 1000006;
static constexpr int LUPINE_RPC_cuGetExportTableMetadata = 1000007;
static constexpr int LUPINE_RPC_cuPrivateGetModuleNode = 1000008;
static constexpr int LUPINE_RPC_cuPointerGetAttributes = 1000009;
static constexpr int LUPINE_RPC_cuStreamAddCallback = 1000010;
static constexpr int LUPINE_RPC_cuGraphConditionalHandleCreate = 1000011;
static constexpr int LUPINE_RPC_cuGraphAddNode_v2 = 1000012;
static constexpr int LUPINE_RPC_cuStreamBeginCaptureToGraph = 1000013;
static constexpr int LUPINE_RPC_cuStreamUpdateCaptureDependencies_v2 = 1000014;
static constexpr int LUPINE_RPC_cuStreamGetCaptureInfo_v3 = 1000015;
static constexpr int LUPINE_RPC_cuDeviceGetGraphMemAttribute = 1000016;
static constexpr int LUPINE_RPC_cuDeviceSetGraphMemAttribute = 1000017;
static constexpr int LUPINE_RPC_cuLinkAddData_v2 = 1000018;
static constexpr int LUPINE_RPC_cuGraphGetEdges = 1000019;
static constexpr int LUPINE_RPC_cuGraphNodeGetDependencies = 1000020;
static constexpr int LUPINE_RPC_cuGraphNodeGetDependentNodes = 1000021;

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
      {LUPINE_RPC_cuCtxCreate_v2,
       {handle_manual_cuCtxCreate_v2, "cuCtxCreate_v2"}},
      {LUPINE_RPC_cuMemPoolSetAttribute,
       {handle_manual_cuMemPoolSetAttribute, "cuMemPoolSetAttribute"}},
      {LUPINE_RPC_cuMemPoolGetAttribute,
       {handle_manual_cuMemPoolGetAttribute, "cuMemPoolGetAttribute"}},
      {LUPINE_RPC_cuPointerGetAttribute,
       {handle_manual_cuPointerGetAttribute, "cuPointerGetAttribute"}},
      {LUPINE_RPC_cuPointerGetAttributes,
       {handle_manual_cuPointerGetAttributes, "cuPointerGetAttributes"}},
      {RPC_cuLinkCreate_v2, {handle_manual_cuLinkCreate_v2, "cuLinkCreate_v2"}},
      {LUPINE_RPC_cuLinkAddData_v2,
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
      {LUPINE_RPC_cuDeviceGetGraphMemAttribute,
       {handle_manual_cuDeviceGetGraphMemAttribute,
        "cuDeviceGetGraphMemAttribute"}},
      {LUPINE_RPC_cuDeviceSetGraphMemAttribute,
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
      {LUPINE_RPC_cuGraphGetEdges,
       {handle_manual_cuGraphGetEdges, "cuGraphGetEdges"}},
      {LUPINE_RPC_cuGraphNodeGetDependencies,
       {handle_manual_cuGraphNodeGetDependencies,
        "cuGraphNodeGetDependencies"}},
      {LUPINE_RPC_cuGraphNodeGetDependentNodes,
       {handle_manual_cuGraphNodeGetDependentNodes,
        "cuGraphNodeGetDependentNodes"}},
      {LUPINE_RPC_snapshot_create,
       {handle_lupine_snapshot_create, "lupine snapshot create"}},
      {LUPINE_RPC_snapshot_status,
       {handle_lupine_snapshot_status, "lupine snapshot status"}},
      {LUPINE_RPC_snapshot_load,
       {handle_lupine_snapshot_load, "lupine snapshot load"}},
      {LUPINE_RPC_snapshot_delete,
       {handle_lupine_snapshot_delete, "lupine snapshot delete"}},
      {LUPINE_RPC_snapshot_save_and_exit,
       {handle_lupine_snapshot_save_and_exit, "lupine snapshot save and exit"}},
      {RPC_cuGraphHostNodeGetParams,
       {handle_manual_cuGraphHostNodeGetParams, "cuGraphHostNodeGetParams"}},
      {RPC_cuGraphHostNodeSetParams,
       {handle_manual_cuGraphHostNodeSetParams, "cuGraphHostNodeSetParams"}},
      {RPC_cuGraphExecHostNodeSetParams,
       {handle_manual_cuGraphExecHostNodeSetParams,
        "cuGraphExecHostNodeSetParams"}},
      {LUPINE_RPC_cuLaunchHostFunc,
       {handle_manual_cuLaunchHostFunc, "cuLaunchHostFunc"}},
      {LUPINE_RPC_cuStreamAddCallback,
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
      {LUPINE_RPC_cuStreamUpdateCaptureDependencies_v2,
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

void client_handler(lupine_socket_t connfd) {
#ifndef _WIN32
  int fd_flags = fcntl(connfd, F_GETFD);
  int socket_error = 0;
  socklen_t socket_error_len = sizeof(socket_error);
  int socket_error_result =
      getsockopt(connfd, SOL_SOCKET, SO_ERROR, &socket_error, &socket_error_len);
  LUPINE_LOG_DEBUG("Starting client handler for fd " << connfd
                                                     << " fd_flags=" << fd_flags
                                                     << " so_error_result="
                                                     << socket_error_result
                                                     << " so_error="
                                                     << socket_error);
#endif
  conn_t conn = {connfd, 1};
  conn.request_id = 1;
  conn.local_request_parity = conn.request_id & 1;
  int init_result = pthread_mutex_init(&conn.read_mutex, NULL);
  if (init_result != 0) {
    LUPINE_LOG_ERROR("Error initializing read mutex: " << init_result);
    return;
  }
  init_result = pthread_mutex_init(&conn.write_mutex, NULL);
  if (init_result != 0) {
    LUPINE_LOG_ERROR("Error initializing write mutex: " << init_result);
    return;
  }
  init_result = pthread_mutex_init(&conn.call_mutex, NULL);
  if (init_result != 0) {
    LUPINE_LOG_ERROR("Error initializing call mutex: " << init_result);
    return;
  }
  init_result = pthread_cond_init(&conn.read_cond, NULL);
  if (init_result != 0) {
    LUPINE_LOG_ERROR("Error initializing read condition: " << init_result);
    return;
  }
  if (rpc_http2_server_init(&conn) < 0) {
    LUPINE_LOG_ERROR("Error initializing HTTP/2 server transport.");
    return;
  }

  printf("Client connected.\n");

  while (1) {
    int op = rpc_dispatch(&conn, 0);
    if (op < 0) {
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }
    LUPINE_TRACE_LOG("LUPINE server handling op " << op);

    const auto &manual_handlers = lupine_manual_handlers();
    auto manual = manual_handlers.find(op);
    if (manual != manual_handlers.end()) {
      if (manual->second.handler(&conn) < 0) {
        lupine_log_manual_handler_error(manual->second.name);
        break;
      }
      continue;
    }

    auto opHandler = get_handler(op);
    if (opHandler == nullptr) {
      LUPINE_LOG_ERROR("No RPC handler for op " << op << "; closing client.");
      break;
    }
    if (opHandler(&conn) < 0) {
      LUPINE_LOG_ERROR("Error handling request.");
      break;
    }
  }

  if (pthread_mutex_destroy(&conn.read_mutex) < 0 ||
      pthread_mutex_destroy(&conn.write_mutex) < 0)
    LUPINE_LOG_ERROR("Error destroying mutex.");

  lupine_socket_close(connfd);
}

static void restored_or_fresh_client_handler(lupine_socket_t connfd) {
  char snapshot_id[LUPINE_SNAPSHOT_ID_BUFFER_BYTES] = {};
  int bootstrap = lupine_snapshot_read_bootstrap(connfd, snapshot_id);
  if (bootstrap < 0) {
    LUPINE_LOG_ERROR("Error reading snapshot bootstrap.");
    lupine_socket_close(connfd);
    return;
  }
  if (bootstrap > 0) {
    int restore = lupine_snapshot_restore_for_connection(snapshot_id, connfd);
    if (restore == 0) {
      return;
    }
    if (restore < 0) {
      LUPINE_LOG_ERROR("Error restoring snapshot " << snapshot_id);
      lupine_socket_close(connfd);
      return;
    }
  }
  client_handler(connfd);
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
    port = atoi(p);
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
      restored_or_fresh_client_handler(connfd);
      exit(0);
    }
    lupine_socket_close(connfd);
#else
    // Windows has no fork; connections share the server process.
    std::thread client_thread(restored_or_fresh_client_handler, connfd);

    // detach the thread so it runs independently
    client_thread.detach();
#endif
  }

  lupine_socket_close(sockfd);
  return 0;
}
