#ifndef LUPINE_GPU_SNAPSHOT_XLATE_H
#define LUPINE_GPU_SNAPSHOT_XLATE_H

// Include this AFTER <cuda.h> in the server handler translation units
// (gen_server.cpp, manual_server.cpp). It redefines the driver entry points
// that consume a client-held handle so the handle is translated old->new after
// a snapshot restore. Before any restore the remaps are empty, so translation
// is an identity lookup.
//
// Each wrapper calls the real driver function via the (name)(args)
// parenthesization, which suppresses the function-like macro so there is no
// recursion. Object *creators* (cuStreamCreate, cuModuleLoadData,
// cuLibraryGetKernel, cuKernelGetFunction, cuDevicePrimaryCtxRetain) are NOT
// wrapped -- they are tracked instead. cuLaunchKernel is handled in its manual
// handler (it must translate before reading the kernel's param layout).

#include <cuda.h>

#include "gpu_snapshot.h"

#ifdef __cplusplus
#define LUPINE_XLATE_INLINE static inline
#else
#define LUPINE_XLATE_INLINE static inline
#endif

// --- stream-consuming -----------------------------------------------------
LUPINE_XLATE_INLINE CUresult lupine_x_cuStreamSynchronize(CUstream s) {
  return (cuStreamSynchronize)(lupine_gpu_xlate_stream(s));
}
#define cuStreamSynchronize lupine_x_cuStreamSynchronize

LUPINE_XLATE_INLINE CUresult lupine_x_cuStreamQuery(CUstream s) {
  return (cuStreamQuery)(lupine_gpu_xlate_stream(s));
}
#define cuStreamQuery lupine_x_cuStreamQuery

LUPINE_XLATE_INLINE CUresult lupine_x_cuStreamGetFlags(CUstream s,
                                                       unsigned int *f) {
  return (cuStreamGetFlags)(lupine_gpu_xlate_stream(s), f);
}
#define cuStreamGetFlags lupine_x_cuStreamGetFlags

LUPINE_XLATE_INLINE CUresult lupine_x_cuStreamGetPriority(CUstream s, int *p) {
  return (cuStreamGetPriority)(lupine_gpu_xlate_stream(s), p);
}
#define cuStreamGetPriority lupine_x_cuStreamGetPriority

LUPINE_XLATE_INLINE CUresult lupine_x_cuStreamWaitEvent(CUstream s, CUevent e,
                                                        unsigned int fl) {
  return (cuStreamWaitEvent)(lupine_gpu_xlate_stream(s),
                             lupine_gpu_xlate_event(e), fl);
}
#define cuStreamWaitEvent lupine_x_cuStreamWaitEvent

// --- event-consuming ------------------------------------------------------
LUPINE_XLATE_INLINE CUresult lupine_x_cuEventQuery(CUevent e) {
  return (cuEventQuery)(lupine_gpu_xlate_event(e));
}
#define cuEventQuery lupine_x_cuEventQuery

LUPINE_XLATE_INLINE CUresult lupine_x_cuEventSynchronize(CUevent e) {
  return (cuEventSynchronize)(lupine_gpu_xlate_event(e));
}
#define cuEventSynchronize lupine_x_cuEventSynchronize

LUPINE_XLATE_INLINE CUresult lupine_x_cuEventElapsedTime(float *ms, CUevent a,
                                                         CUevent b) {
  return (cuEventElapsedTime)(ms, lupine_gpu_xlate_event(a),
                              lupine_gpu_xlate_event(b));
}
#define cuEventElapsedTime lupine_x_cuEventElapsedTime

LUPINE_XLATE_INLINE CUresult lupine_x_cuLaunchHostFunc(CUstream s, CUhostFn fn,
                                                       void *u) {
  return (cuLaunchHostFunc)(lupine_gpu_xlate_stream(s), fn, u);
}
#define cuLaunchHostFunc lupine_x_cuLaunchHostFunc

// --- async copies / memsets (stream is the last arg) ----------------------
LUPINE_XLATE_INLINE CUresult lupine_x_cuMemcpyHtoDAsync_v2(CUdeviceptr d,
                                                           const void *s,
                                                           size_t n,
                                                           CUstream st) {
  return (cuMemcpyHtoDAsync_v2)(d, s, n, lupine_gpu_xlate_stream(st));
}
#define cuMemcpyHtoDAsync_v2 lupine_x_cuMemcpyHtoDAsync_v2

LUPINE_XLATE_INLINE CUresult lupine_x_cuMemcpyDtoHAsync_v2(void *d,
                                                           CUdeviceptr s,
                                                           size_t n,
                                                           CUstream st) {
  return (cuMemcpyDtoHAsync_v2)(d, s, n, lupine_gpu_xlate_stream(st));
}
#define cuMemcpyDtoHAsync_v2 lupine_x_cuMemcpyDtoHAsync_v2

LUPINE_XLATE_INLINE CUresult lupine_x_cuMemcpyDtoDAsync_v2(CUdeviceptr d,
                                                           CUdeviceptr s,
                                                           size_t n,
                                                           CUstream st) {
  return (cuMemcpyDtoDAsync_v2)(d, s, n, lupine_gpu_xlate_stream(st));
}
#define cuMemcpyDtoDAsync_v2 lupine_x_cuMemcpyDtoDAsync_v2

LUPINE_XLATE_INLINE CUresult lupine_x_cuMemsetD8Async(CUdeviceptr d,
                                                      unsigned char v, size_t n,
                                                      CUstream st) {
  return (cuMemsetD8Async)(d, v, n, lupine_gpu_xlate_stream(st));
}
#define cuMemsetD8Async lupine_x_cuMemsetD8Async

LUPINE_XLATE_INLINE CUresult lupine_x_cuMemsetD16Async(CUdeviceptr d,
                                                       unsigned short v,
                                                       size_t n, CUstream st) {
  return (cuMemsetD16Async)(d, v, n, lupine_gpu_xlate_stream(st));
}
#define cuMemsetD16Async lupine_x_cuMemsetD16Async

LUPINE_XLATE_INLINE CUresult lupine_x_cuMemsetD32Async(CUdeviceptr d,
                                                       unsigned int v, size_t n,
                                                       CUstream st) {
  return (cuMemsetD32Async)(d, v, n, lupine_gpu_xlate_stream(st));
}
#define cuMemsetD32Async lupine_x_cuMemsetD32Async

// --- event record (stream arg) --------------------------------------------
LUPINE_XLATE_INLINE CUresult lupine_x_cuEventRecord(CUevent e, CUstream s) {
  return (cuEventRecord)(lupine_gpu_xlate_event(e), lupine_gpu_xlate_stream(s));
}
#define cuEventRecord lupine_x_cuEventRecord

LUPINE_XLATE_INLINE CUresult lupine_x_cuEventRecordWithFlags(CUevent e,
                                                             CUstream s,
                                                             unsigned int fl) {
  return (cuEventRecordWithFlags)(lupine_gpu_xlate_event(e),
                                  lupine_gpu_xlate_stream(s), fl);
}
#define cuEventRecordWithFlags lupine_x_cuEventRecordWithFlags

// --- context-consuming ----------------------------------------------------
LUPINE_XLATE_INLINE CUresult lupine_x_cuCtxSetCurrent(CUcontext c) {
  return (cuCtxSetCurrent)(lupine_gpu_xlate_context(c));
}
#define cuCtxSetCurrent lupine_x_cuCtxSetCurrent

LUPINE_XLATE_INLINE CUresult lupine_x_cuCtxPushCurrent_v2(CUcontext c) {
  return (cuCtxPushCurrent_v2)(lupine_gpu_xlate_context(c));
}
#define cuCtxPushCurrent_v2 lupine_x_cuCtxPushCurrent_v2

// --- function-consuming ---------------------------------------------------
LUPINE_XLATE_INLINE CUresult lupine_x_cuFuncGetAttribute(
    int *pi, CUfunction_attribute a, CUfunction f) {
  return (cuFuncGetAttribute)(pi, a, lupine_gpu_xlate_function(f));
}
#define cuFuncGetAttribute lupine_x_cuFuncGetAttribute

LUPINE_XLATE_INLINE CUresult lupine_x_cuFuncSetAttribute(
    CUfunction f, CUfunction_attribute a, int v) {
  return (cuFuncSetAttribute)(lupine_gpu_xlate_function(f), a, v);
}
#define cuFuncSetAttribute lupine_x_cuFuncSetAttribute

// --- library/module-consuming getters (client handle may be pre-snapshot) ---
LUPINE_XLATE_INLINE CUresult lupine_x_cuLibraryGetModule(CUmodule *m,
                                                         CUlibrary lib) {
  return (cuLibraryGetModule)(m, lupine_gpu_xlate_library(lib));
}
#define cuLibraryGetModule lupine_x_cuLibraryGetModule

LUPINE_XLATE_INLINE CUresult lupine_x_cuLibraryUnload(CUlibrary lib) {
  return (cuLibraryUnload)(lupine_gpu_xlate_library(lib));
}
#define cuLibraryUnload lupine_x_cuLibraryUnload

LUPINE_XLATE_INLINE CUresult lupine_x_cuLibraryGetManaged(CUdeviceptr *d,
                                                          size_t *b,
                                                          CUlibrary lib,
                                                          const char *n) {
  return (cuLibraryGetManaged)(d, b, lupine_gpu_xlate_library(lib), n);
}
#define cuLibraryGetManaged lupine_x_cuLibraryGetManaged

LUPINE_XLATE_INLINE CUresult lupine_x_cuModuleUnload(CUmodule m) {
  return (cuModuleUnload)(lupine_gpu_xlate_module(m));
}
#define cuModuleUnload lupine_x_cuModuleUnload

LUPINE_XLATE_INLINE CUresult lupine_x_cuModuleGetGlobal_v2(CUdeviceptr *d,
                                                           size_t *b,
                                                           CUmodule m,
                                                           const char *n) {
  return (cuModuleGetGlobal_v2)(d, b, lupine_gpu_xlate_module(m), n);
}
#define cuModuleGetGlobal_v2 lupine_x_cuModuleGetGlobal_v2

#endif // LUPINE_GPU_SNAPSHOT_XLATE_H
