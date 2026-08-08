#ifndef LUPINE_COPY_PIPELINE_H
#define LUPINE_COPY_PIPELINE_H

#include <cstddef>
#include <cuda.h>

#include "rpc.h"

#ifdef LUPINE_RPC_SERVER
bool lupine_server_initialize_connection(conn_t *conn);
void lupine_server_cleanup_connection(conn_t *conn);

int lupine_server_copy_htod_async(conn_t *conn, int framed,
                                  CUdeviceptr destination, size_t bytes,
                                  CUstream stream, CUresult &result);

int handle_manual_cuDevicePrimaryCtxRetain(conn_t *conn);
int handle_manual_cuDevicePrimaryCtxRelease_v2(conn_t *conn);
int handle_manual_cuDevicePrimaryCtxReset_v2(conn_t *conn);
int handle_manual_cuCtxAttach(conn_t *conn);
int handle_manual_cuCtxCreate_v2(conn_t *conn);
int handle_manual_cuCtxDestroy_v2(conn_t *conn);
int handle_manual_cuCtxDetach(conn_t *conn);
int handle_manual_cuMemcpyHtoD_v2(conn_t *conn);
int handle_manual_cuMemcpyDtoH_v2(conn_t *conn);
#endif

#endif
