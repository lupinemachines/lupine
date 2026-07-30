#pragma once

#include "lupine_platform.h"

// Initializes SIGTERM coordination and attempts to load the optional LupineCR
// provider. Must be called in the connection child before its first CUDA call.
bool lupine_server_checkpoint_child_start(lupine_socket_t connection);

// Supplies the stable identifier discovered during connection bootstrap and
// asks the optional provider to restore it before the first CUDA RPC. A null or
// empty identifier represents an unkeyed connection and does not restore.
bool lupine_server_checkpoint_connection_ready(const char *connection_id);

// Stops signal coordination. If SIGTERM was received, drains all admitted
// CUDA handlers and invokes the optional provider. Returns zero when shutdown
// can proceed, including when no provider is installed.
int lupine_server_checkpoint_child_finish();
