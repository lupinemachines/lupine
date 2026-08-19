#ifndef LUPINE_RPC_SERVER_H
#define LUPINE_RPC_SERVER_H

#include "rpc.h"

#include <stddef.h>
#include <unordered_map>

typedef int (*RequestHandler)(conn_t *conn);

struct rpc_handler;

struct rpc_backend {
  const char *name;
  bool (*child_start)(lupine_socket_t connfd);
  int (*child_finish)();
  bool (*connection_open)(conn_t *conn);
  bool (*connection_ready)(conn_t *conn, const char *session_id);
  int (*dispatch)(const rpc_handler &handler, conn_t *conn);
  void (*connection_close)(conn_t *conn);
};

struct rpc_handler {
  rpc_handler(RequestHandler request_handler = nullptr,
              const rpc_backend *request_backend = nullptr)
      : handler(request_handler), backend(request_backend) {}

  RequestHandler handler;
  const rpc_backend *backend;
};

using rpc_handler_registry = std::unordered_map<int, rpc_handler>;

bool rpc_server_validate(const rpc_handler_registry &handlers,
                         const rpc_backend *const *backends, size_t count);
int rpc_server_dispatch(const rpc_handler_registry &handlers, conn_t *conn,
                        int op);
bool rpc_server_child_start(const rpc_backend *const *backends, size_t count,
                            lupine_socket_t connfd);
int rpc_server_child_finish(const rpc_backend *const *backends, size_t count);
bool rpc_server_connection_open(const rpc_backend *const *backends,
                                size_t count, conn_t *conn);
bool rpc_server_connection_ready(const rpc_backend *const *backends,
                                 size_t count, conn_t *conn,
                                 const char *session_id);
void rpc_server_connection_close(const rpc_backend *const *backends,
                                 size_t count, conn_t *conn);

extern const rpc_backend lupine_cuda_backend;
extern const rpc_backend lupine_nvml_backend;

const rpc_handler_registry &lupine_rpc_handlers();
const rpc_backend *const *lupine_compiled_backends(size_t *count);

#endif
