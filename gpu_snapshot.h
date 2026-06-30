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

// Object tracking. A fresh worker created on restore replays these so the
// client's pre-snapshot opaque handles can be translated to live ones.
void lupine_gpu_track_module(CUmodule m, unsigned int kind, const void *image,
                             size_t size);
void lupine_gpu_track_function(CUfunction fn, CUmodule m, const char *name);
void lupine_gpu_track_library(CUlibrary lib, unsigned int kind,
                              const void *image, size_t size);
void lupine_gpu_track_kernel(CUkernel k, CUlibrary lib, const char *name);
void lupine_gpu_track_kernel_function(CUfunction fn, CUkernel k);
// Device globals live in the library/module image (outside the VMM arena), so
// their contents must be captured and restored after the image is replayed.
void lupine_gpu_track_library_global(CUlibrary lib, const char *name);
void lupine_gpu_track_module_global(CUmodule mod, const char *name);
void lupine_gpu_track_stream(CUstream s, unsigned int flags);
void lupine_gpu_track_event(CUevent e, unsigned int flags);
void lupine_gpu_track_primary_ctx(CUcontext ctx);

// Translate a client (pre-snapshot) handle to the live one; identity if not
// remapped (e.g. objects created after restore, or the default stream).
CUfunction lupine_gpu_xlate_function(CUfunction fn);
CUmodule lupine_gpu_xlate_module(CUmodule m);
CUlibrary lupine_gpu_xlate_library(CUlibrary lib);
CUkernel lupine_gpu_xlate_kernel(CUkernel k);
CUstream lupine_gpu_xlate_stream(CUstream s);
CUevent lupine_gpu_xlate_event(CUevent e);
CUcontext lupine_gpu_xlate_context(CUcontext ctx);

// RPC handlers (registered as manual overrides in server.cpp).
int handle_manual_cuMemAlloc_v2(conn_t *conn);
int handle_manual_cuMemFree_v2(conn_t *conn);
int handle_gpu_snapshot_save(conn_t *conn);
int handle_gpu_snapshot_restore(conn_t *conn);

// Overrides of codegen handlers that additionally track the created object so it
// can be replayed/remapped on restore.
int handle_manual_cuModuleGetFunction_tracked(conn_t *conn);
int handle_manual_cuLibraryGetKernel_tracked(conn_t *conn);
int handle_manual_cuKernelGetFunction_tracked(conn_t *conn);
int handle_manual_cuStreamCreate_tracked(conn_t *conn);
int handle_manual_cuEventCreate_tracked(conn_t *conn);
int handle_manual_cuDevicePrimaryCtxRetain_tracked(conn_t *conn);

#ifdef __cplusplus
}
#endif

#endif // LUPINE_GPU_SNAPSHOT_H
