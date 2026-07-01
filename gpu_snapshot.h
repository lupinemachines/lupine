#ifndef LUPINE_GPU_SNAPSHOT_H
#define LUPINE_GPU_SNAPSHOT_H

#include <cuda.h>
#include <stddef.h>

#include "rpc.h"

// GPU-state snapshots for Lupine's reconnect-to-a-fresh-worker path.
//
// The server asks CUDA to checkpoint itself, copies the driver-created host
// checkpoint payload mapping to disk, then immediately restores/unlocks itself.
// A fresh worker replays the tracked module/library/function shape, checkpoints
// itself, overlays the saved payload into its own CUDA checkpoint payload
// mapping, and restores/unlocks. This avoids CRIU, cuda-checkpoint, manual
// allocation manifests, and stable server-side GPU pointer assumptions.
//
// Opaque module/library/function/kernel handles still get new values in a fresh
// worker before payload restore. The server records enough object creation
// inputs to recreate that shape and translate old client handles to live ones.
// Streams are intentionally normalized to the default stream after restore.

// Custom RPC op ids (kept out of the codegen range).
static const int LUPINE_RPC_gpu_snapshot_save = 1000201;
static const int LUPINE_RPC_gpu_snapshot_restore = 1000202;

#ifdef __cplusplus
extern "C" {
#endif

// Allocation hooks are retained for ABI/routing compatibility. They currently
// pass through to CUDA; snapshotting no longer owns server allocation layout.
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
void lupine_gpu_track_library_module(CUmodule m, CUlibrary lib);
void lupine_gpu_track_function(CUfunction fn, CUmodule m, const char *name);
void lupine_gpu_track_library(CUlibrary lib, unsigned int kind,
                              const void *image, size_t size);
void lupine_gpu_track_kernel(CUkernel k, CUlibrary lib, const char *name);
void lupine_gpu_track_kernel_function(CUfunction fn, CUkernel k);

// Translate a client (pre-snapshot) handle to the live one; identity if not
// remapped (e.g. objects created after restore).
CUfunction lupine_gpu_xlate_function(CUfunction fn);
CUmodule lupine_gpu_xlate_module(CUmodule m);
CUlibrary lupine_gpu_xlate_library(CUlibrary lib);
CUkernel lupine_gpu_xlate_kernel(CUkernel k);
CUcontext lupine_gpu_xlate_context(CUcontext ctx);
// Streams are intentionally not replayed. After restore, non-default streams
// are lazily replaced so cached pre-snapshot stream handles do not poison
// launches and async copies.
CUstream lupine_gpu_xlate_stream(CUstream s);
int lupine_gpu_restored(void);

// RPC handlers (registered as manual overrides in server.cpp).
int handle_manual_cuMemAlloc_v2(conn_t *conn);
int handle_manual_cuMemFree_v2(conn_t *conn);
int handle_gpu_snapshot_save(conn_t *conn);
int handle_gpu_snapshot_restore(conn_t *conn);

// Overrides of codegen handlers that additionally track the created object so
// it can be replayed/remapped on restore.
int handle_manual_cuModuleGetFunction_tracked(conn_t *conn);
int handle_manual_cuLibraryGetKernel_tracked(conn_t *conn);
int handle_manual_cuKernelGetFunction_tracked(conn_t *conn);

#ifdef __cplusplus
}
#endif

#endif // LUPINE_GPU_SNAPSHOT_H
