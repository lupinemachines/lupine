#ifndef LUPINE_CLIENT_CALLBACK_H
#define LUPINE_CLIENT_CALLBACK_H

#include "cuda_compat.h"
#include "rpc.h"

#include <stddef.h>

struct lupine_wire_callback {
  void *function_token = nullptr;
  void *user_data_token = nullptr;
};

// Callback tokens are valid pointers to client-owned registration records. The
// server only stores and echoes them; it never dereferences or invokes them.
bool lupine_register_host_callback(conn_t *conn, CUhostFn callback,
                                   void *user_data, bool persistent,
                                   lupine_wire_callback *wire);
bool lupine_register_stream_callback(conn_t *conn, CUstreamCallback callback,
                                     void *user_data,
                                     lupine_wire_callback *wire);
void lupine_revoke_callback(conn_t *conn, void *function_token);
void lupine_clear_callbacks(conn_t *conn);

bool lupine_resolve_host_callback(conn_t *conn,
                                  const lupine_wire_callback &wire,
                                  bool consume, CUhostFn *callback,
                                  void **user_data);
bool lupine_resolve_stream_callback(conn_t *conn,
                                    const lupine_wire_callback &wire,
                                    CUstreamCallback *callback,
                                    void **user_data);

struct lupine_host_callback_dispatch_options {
  void *(*allocate)(size_t) = nullptr;
  void (*deallocate)(void *) = nullptr;
  void (*commit_copy)(void *dst, const void *src, size_t size) = nullptr;
  size_t max_drain_bytes = 64U * 1024U * 1024U;
};

enum lupine_callback_dispatch_result {
  LUPINE_CALLBACK_DISPATCH_FATAL = -1,
  LUPINE_CALLBACK_DISPATCH_COMPLETE = 0,
  LUPINE_CALLBACK_DISPATCH_ABORTED = 1,
};

// Reads the body of an op=1 callback message after rpc_dispatch() has consumed
// its prefix. On a bounded allocation failure the complete message is drained,
// the callback is revoked without being invoked, and ABORTED is returned. Any
// framing or transport failure poisons the connection and returns FATAL.
int lupine_dispatch_host_callback(
    conn_t *conn, const lupine_host_callback_dispatch_options *options,
    int *request_id);

#endif
