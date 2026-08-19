#include "rpc_server.h"

#include "lupine_log.h"

#include <unordered_set>

bool rpc_server_validate(const rpc_handler_registry &handlers,
                         const rpc_backend *const *backends, size_t count) {
  std::unordered_set<const rpc_backend *> compiled_backends;
  for (size_t i = 0; i < count; ++i) {
    const rpc_backend *backend = backends[i];
    if (backend == nullptr || backend->name == nullptr) {
      return false;
    }
    compiled_backends.insert(backend);
  }
  for (const auto &entry : handlers) {
    if (entry.second.handler == nullptr ||
        compiled_backends.count(entry.second.backend) == 0) {
      return false;
    }
  }
  return true;
}

int rpc_server_dispatch(const rpc_handler_registry &handlers, conn_t *conn,
                        int op) {
  LUPINE_TRACE_LOG("LUPINE server handling op " << op);
  auto it = handlers.find(op);
  if (it == handlers.end()) {
    LUPINE_LOG_ERROR("No RPC handler for op " << op << "; closing client.");
    return -1;
  }
  const rpc_handler &handler = it->second;
  const rpc_backend &backend = *handler.backend;
  int result = backend.dispatch != nullptr ? backend.dispatch(handler, conn)
                                           : handler.handler(conn);
  if (result >= 0) {
    return 0;
  }
  if (handler.error_style == rpc_handler_error_style::manual) {
    LUPINE_LOG_ERROR("Error handling manual " << handler.name << " request.");
  } else {
    LUPINE_LOG_ERROR("Error handling " << backend.name << " request for op "
                                       << op << ".");
  }
  return -1;
}

bool rpc_server_child_start(const rpc_backend *const *backends, size_t count,
                            lupine_socket_t connfd) {
  for (size_t i = 0; i < count; ++i) {
    if (backends[i]->child_start != nullptr &&
        !backends[i]->child_start(connfd)) {
      for (size_t j = i; j > 0; --j) {
        if (backends[j - 1]->child_finish != nullptr) {
          (void)backends[j - 1]->child_finish();
        }
      }
      return false;
    }
  }
  return true;
}

int rpc_server_child_finish(const rpc_backend *const *backends, size_t count) {
  int result = 0;
  for (size_t i = count; i > 0; --i) {
    if (backends[i - 1]->child_finish != nullptr) {
      int backend_result = backends[i - 1]->child_finish();
      if (result == 0) {
        result = backend_result;
      }
    }
  }
  return result;
}

bool rpc_server_connection_open(const rpc_backend *const *backends,
                                size_t count, conn_t *conn) {
  for (size_t i = 0; i < count; ++i) {
    if (backends[i]->connection_open != nullptr &&
        !backends[i]->connection_open(conn)) {
      for (size_t j = i; j > 0; --j) {
        if (backends[j - 1]->connection_close != nullptr) {
          backends[j - 1]->connection_close(conn);
        }
      }
      return false;
    }
  }
  return true;
}

bool rpc_server_connection_ready(const rpc_backend *const *backends,
                                 size_t count, conn_t *conn,
                                 const char *session_id) {
  for (size_t i = 0; i < count; ++i) {
    if (backends[i]->connection_ready != nullptr &&
        !backends[i]->connection_ready(conn, session_id)) {
      return false;
    }
  }
  return true;
}

void rpc_server_connection_close(const rpc_backend *const *backends,
                                 size_t count, conn_t *conn) {
  for (size_t i = count; i > 0; --i) {
    if (backends[i - 1]->connection_close != nullptr) {
      backends[i - 1]->connection_close(conn);
    }
  }
}
