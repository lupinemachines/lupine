#ifndef LUPINE_GPU_SNAPSHOT_H
#define LUPINE_GPU_SNAPSHOT_H

#include <cuda.h>
#include <stddef.h>

#include "rpc.h"

// GPU-state snapshots without CRIU or cuda-checkpoint.
//
// Device allocations are served from a single VMM virtual-address arena
// (cuMemAddressReserve + cuMemCreate + cuMemMap). Because the server runs with
// ASLR disabled, the driver picks the same arena base in every worker process,
// so a CUdeviceptr handed to the client is a stable offset that can be
// reproduced on restore. A snapshot records the arena base + each live
// allocation's (offset, size, contents); restore re-reserves the SAME base in a
// fresh worker and copies the bytes back, reproducing every device pointer.
//
// Opaque handles (CUstream/CUmodule/...) get new values in a fresh worker, but
// the client never dereferences them -- it only passes them back through RPCs --
// so the server remaps them (added in a later phase).

// Custom RPC op ids (kept out of the codegen range).
static const int LUPINE_RPC_gpu_snapshot_save = 1000201;
static const int LUPINE_RPC_gpu_snapshot_restore = 1000202;

#ifdef __cplusplus
extern "C" {
#endif

// VMM-backed replacements for cuMemAlloc_v2 / cuMemFree_v2.
CUresult lupine_gpu_alloc(CUdeviceptr *dptr, size_t bytesize);
CUresult lupine_gpu_free(CUdeviceptr dptr);
// True when dptr was handed out by lupine_gpu_alloc.
int lupine_gpu_owns(CUdeviceptr dptr);

// Serialize / restore live device allocations for snapshot `id`.
CUresult lupine_gpu_snapshot_save(const char *id);
CUresult lupine_gpu_snapshot_restore(const char *id);

// RPC handlers (registered as manual overrides in server.cpp).
int handle_manual_cuMemAlloc_v2(conn_t *conn);
int handle_manual_cuMemFree_v2(conn_t *conn);
int handle_gpu_snapshot_save(conn_t *conn);
int handle_gpu_snapshot_restore(conn_t *conn);

#ifdef __cplusplus
}
#endif

#endif // LUPINE_GPU_SNAPSHOT_H
