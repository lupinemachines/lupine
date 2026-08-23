#ifndef LUPINE_SERVER_VA_H
#define LUPINE_SERVER_VA_H

#include <cuda.h>

#include <cstddef>

struct conn_t;

// Runs the native CUDA managed allocator while steering its anonymous VA pool
// reservation into the connection's negotiated arena.
CUresult lupine_server_va_alloc_managed(conn_t *conn, CUdeviceptr *pointer,
                                        size_t size, unsigned int flags);

// Replaces a monotonically claimed part of the connection's PROT_NONE arena
// with an anonymous mapping.
void *lupine_server_va_map(conn_t *conn, size_t size, size_t alignment,
                           int protection, int flags);

#endif
