#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <thread>

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

static bool lupine_server_trace_enabled() {
  static int enabled = []() {
    const char *env = getenv("LUPINE_SERVER_TRACE");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
  }();
  return enabled != 0;
}

static void lupine_log_manual_handler_error(const char *name) {
  LUPINE_LOG_ERROR("Error handling manual " << name << " request.");
}

void client_handler(lupine_socket_t connfd) {
  conn_t conn = {connfd, 1};
  conn.request_id = 1;
  conn.local_request_parity = conn.request_id & 1;
  if (pthread_mutex_init(&conn.read_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.write_mutex, NULL) < 0 ||
      pthread_mutex_init(&conn.call_mutex, NULL) < 0 ||
      pthread_cond_init(&conn.read_cond, NULL) < 0 ||
      rpc_http2_server_init(&conn) < 0) {
    LUPINE_LOG_ERROR("Error initializing mutex.");
    return;
  }

  printf("Client connected.\n");

  while (1) {
    int op = rpc_dispatch(&conn, 0);
    if (op < 0) {
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }
    if (lupine_server_trace_enabled()) {
      LUPINE_LOG_DEBUG("LUPINE server handling op " << op);
    }

    if (op == LUPINE_RPC_cuGetExportTableMetadata) {
      if (handle_manual_cuGetExportTableMetadata(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuGetExportTable metadata request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuPrivateGetModuleNode) {
      if (handle_manual_cuPrivateGetModuleNode(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual private module node request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuModuleLoad) {
      if (handle_manual_cuModuleLoad(&conn) < 0) {
        lupine_log_manual_handler_error("cuModuleLoad");
        break;
      }
      continue;
    }
    if (op == RPC_cuModuleLoadData) {
      if (handle_manual_cuModuleLoadData(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuModuleLoadData request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuLibraryLoadData) {
      if (handle_manual_cuLibraryLoadData(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuLibraryLoadData request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuCtxCreate_v2) {
      if (handle_manual_cuCtxCreate_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuCtxCreate_v2 request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuMemPoolSetAttribute) {
      if (handle_manual_cuMemPoolSetAttribute(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuMemPoolSetAttribute request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuMemPoolGetAttribute) {
      if (handle_manual_cuMemPoolGetAttribute(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuMemPoolGetAttribute request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuPointerGetAttribute) {
      if (handle_manual_cuPointerGetAttribute(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuPointerGetAttribute request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuPointerGetAttributes) {
      if (handle_manual_cuPointerGetAttributes(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuPointerGetAttributes request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuArrayCreate_v2) {
      if (handle_manual_cuArrayCreate_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuArrayCreate_v2 request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuArray3DCreate_v2) {
      if (handle_manual_cuArray3DCreate_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuArray3DCreate_v2 request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuLinkCreate_v2) {
      if (handle_manual_cuLinkCreate_v2(&conn) < 0) {
        lupine_log_manual_handler_error("cuLinkCreate_v2");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuLinkAddData_v2) {
      if (handle_manual_cuLinkAddData_v2(&conn) < 0) {
        lupine_log_manual_handler_error("cuLinkAddData_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuLinkAddFile_v2) {
      if (handle_manual_cuLinkAddFile_v2(&conn) < 0) {
        lupine_log_manual_handler_error("cuLinkAddFile_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuLinkComplete) {
      if (handle_manual_cuLinkComplete(&conn) < 0) {
        lupine_log_manual_handler_error("cuLinkComplete");
        break;
      }
      continue;
    }
    if (op == RPC_cuLinkDestroy) {
      if (handle_manual_cuLinkDestroy(&conn) < 0) {
        lupine_log_manual_handler_error("cuLinkDestroy");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy3D_v2) {
      if (handle_manual_cuMemcpy3D_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuMemcpy3D_v2 request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy2D_v2) {
      if (handle_manual_cuMemcpy2D_v2(&conn) < 0) {
        lupine_log_manual_handler_error("cuMemcpy2D_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy2DUnaligned_v2) {
      if (handle_manual_cuMemcpy2DUnaligned_v2(&conn) < 0) {
        lupine_log_manual_handler_error("cuMemcpy2DUnaligned_v2");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpy2DAsync_v2) {
      if (handle_manual_cuMemcpy2DAsync_v2(&conn) < 0) {
        lupine_log_manual_handler_error("cuMemcpy2DAsync_v2");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuDeviceGetGraphMemAttribute) {
      if (handle_manual_cuDeviceGetGraphMemAttribute(&conn) < 0) {
        lupine_log_manual_handler_error("cuDeviceGetGraphMemAttribute");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuDeviceSetGraphMemAttribute) {
      if (handle_manual_cuDeviceSetGraphMemAttribute(&conn) < 0) {
        lupine_log_manual_handler_error("cuDeviceSetGraphMemAttribute");
        break;
      }
      continue;
    }
    if (op == RPC_cuLibraryGetModule) {
      if (handle_manual_cuLibraryGetModule(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuLibraryGetModule request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuLibraryUnload) {
      if (handle_manual_cuLibraryUnload(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuLibraryUnload request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuModuleGetGlobal_v2) {
      if (handle_manual_cuModuleGetGlobal_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuModuleGetGlobal_v2 request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuFuncGetParamLayout) {
      if (handle_manual_cuFuncGetParamLayout(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuFuncGetParamLayout request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuOccupancyMaxPotentialBlockSize) {
      if (handle_manual_cuOccupancyMaxPotentialBlockSize(&conn, false) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuOccupancyMaxPotentialBlockSize request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuOccupancyMaxPotentialBlockSizeWithFlags) {
      if (handle_manual_cuOccupancyMaxPotentialBlockSize(&conn, true) < 0) {
        LUPINE_LOG_ERROR("Error handling manual "
                         "cuOccupancyMaxPotentialBlockSizeWithFlags request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuLaunchKernel) {
      if (handle_manual_cuLaunchKernel(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuLaunchKernel request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuLaunchCooperativeKernel) {
      if (handle_manual_cuLaunchCooperativeKernel(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuLaunchCooperativeKernel request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddKernelNode_v2) {
      if (handle_manual_cuGraphAddKernelNode(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphAddKernelNode request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddMemcpyNode) {
      if (handle_manual_cuGraphAddMemcpyNode(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphAddMemcpyNode request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddMemsetNode) {
      if (handle_manual_cuGraphAddMemsetNode(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphAddMemsetNode request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphAddHostNode) {
      if (handle_manual_cuGraphAddHostNode(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphAddHostNode request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphExecKernelNodeSetParams_v2) {
      if (handle_manual_cuGraphExecKernelNodeSetParams(&conn) < 0) {
        lupine_log_manual_handler_error("cuGraphExecKernelNodeSetParams_v2");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuGraphConditionalHandleCreate) {
      if (handle_manual_cuGraphConditionalHandleCreate(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphConditionalHandleCreate "
                         "request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuGraphAddNode_v2) {
      if (handle_manual_cuGraphAddNode(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphAddNode request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphLaunch) {
      if (handle_manual_cuGraphLaunch(&conn) < 0) {
        lupine_log_manual_handler_error("cuGraphLaunch");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphGetNodes) {
      if (handle_manual_cuGraphGetNodes(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphGetNodes request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuLaunchHostFunc) {
      if (handle_manual_cuLaunchHostFunc(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuLaunchHostFunc request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuStreamAddCallback) {
      if (handle_manual_cuStreamAddCallback(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuStreamAddCallback request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuEventRecord) {
      if (handle_manual_cuEventRecord(&conn, false) < 0) {
        lupine_log_manual_handler_error("cuEventRecord");
        break;
      }
      continue;
    }
    if (op == RPC_cuEventRecordWithFlags) {
      if (handle_manual_cuEventRecord(&conn, true) < 0) {
        lupine_log_manual_handler_error("cuEventRecordWithFlags");
        break;
      }
      continue;
    }
    if (op == RPC_cuEventQuery) {
      if (handle_manual_cuEventQuery(&conn) < 0) {
        lupine_log_manual_handler_error("cuEventQuery");
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamWaitEvent) {
      if (handle_manual_cuStreamWaitEvent(&conn) < 0) {
        lupine_log_manual_handler_error("cuStreamWaitEvent");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuStreamBeginCaptureToGraph) {
      if (handle_manual_cuStreamBeginCaptureToGraph(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuStreamBeginCaptureToGraph "
                         "request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuStreamUpdateCaptureDependencies_v2) {
      if (handle_manual_cuStreamUpdateCaptureDependencies(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual "
                         "cuStreamUpdateCaptureDependencies request.");
        break;
      }
      continue;
    }
    if (op == LUPINE_RPC_cuStreamGetCaptureInfo_v3) {
      if (handle_manual_cuStreamGetCaptureInfo(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuStreamGetCaptureInfo request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamBeginCapture_v2) {
      if (handle_manual_cuStreamBeginCapture(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuStreamBeginCapture request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamEndCapture) {
      if (handle_manual_cuStreamEndCapture(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuStreamEndCapture request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphClone) {
      if (handle_manual_cuGraphClone(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphClone request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphInstantiateWithFlags) {
      if (handle_manual_cuGraphInstantiateWithFlags(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuGraphInstantiateWithFlags request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphInstantiateWithParams) {
      if (handle_manual_cuGraphInstantiateWithParams(&conn) < 0) {
        LUPINE_LOG_ERROR(
            "Error handling manual cuGraphInstantiateWithParams request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphExecDestroy) {
      if (handle_manual_cuGraphExecDestroy(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphExecDestroy request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuGraphDestroy) {
      if (handle_manual_cuGraphDestroy(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuGraphDestroy request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpyHtoD_v2) {
      if (handle_manual_cuMemcpyHtoD_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuMemcpyHtoD_v2 request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpyHtoDAsync_v2) {
      if (handle_manual_cuMemcpyHtoDAsync_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuMemcpyHtoDAsync_v2 request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuMemcpyDtoHAsync_v2) {
      if (handle_manual_cuMemcpyDtoHAsync_v2(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuMemcpyDtoHAsync_v2 request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuCtxSynchronize) {
      if (handle_manual_cuCtxSynchronize(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuCtxSynchronize request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuStreamSynchronize) {
      if (handle_manual_cuStreamSynchronize(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuStreamSynchronize request.");
        break;
      }
      continue;
    }
    if (op == RPC_cuEventSynchronize) {
      if (handle_manual_cuEventSynchronize(&conn) < 0) {
        LUPINE_LOG_ERROR("Error handling manual cuEventSynchronize request.");
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
