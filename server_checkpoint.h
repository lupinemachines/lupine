#pragma once

#include "lupine_platform.h"
#include "rpc.h"

// Initializes SIGTERM coordination and attempts to load the optional LupineCR
// provider. Must be called in the connection child before its first CUDA call.
bool lupine_server_checkpoint_child_start(lupine_socket_t connection);

// Supplies the stable identifier discovered during connection bootstrap and
// asks the optional provider to restore it before the first CUDA RPC. A null or
// empty identifier represents an unkeyed connection and does not restore.
bool lupine_server_checkpoint_connection_ready(const char *connection_id,
                                               const char *required = nullptr);

// Stops signal coordination. If SIGTERM was received, drains all admitted
// CUDA handlers and invokes the optional provider. Returns zero when shutdown
// can proceed, including when no provider is installed.
int lupine_server_checkpoint_child_finish();

// Optional provider override; null leaves the native CUDA implementation in
// use.
void *lupine_server_checkpoint_cuda_symbol(const char *name);

// CUDA-specific catalog and preemption hooks stay out of the optional-provider
// loader, which is also tested without a CUDA driver.
void lupine_server_checkpoint_attach(conn_t *conn, int (*notice)(conn_t *),
                                     int (*save)(const char *),
                                     int (*restore)(const char *));
int lupine_server_checkpoint_commit(const char *checkpoint_id,
                                    const uint64_t *pointers,
                                    const uint64_t *sizes,
                                    void *const *contexts, size_t count);

std::string lupine_server_checkpoint_catalog_path(const char *checkpoint_id);
