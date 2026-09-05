// CUDA runtime API (libcudart) on the LUPINE client.
//
// Most of the surface is generated: each call is forwarded over the CUDA
// driver shim's connections and executed by the server's own libcudart, so the
// same server child owns both the driver state and the runtime state, and one
// lane per client thread keeps the two APIs ordered.
//
// The calls in this file cannot be forwarded that way, because their arguments
// name client-side things the server cannot see: host memory, the module
// images nvcc embedded in the caller, host entry-point addresses, and host
// callbacks. Those run on the driver shim's exported `cu*` API, which is the
// same forwarding path one layer down and already owns the client's identity
// VA arenas, host mirrors, and module images.

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"
#include "lupine_fatbin.h"

namespace {

cudaError_t rpc_error() { return cudaErrorDevicesUnavailable; }

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  return lupine_rpc_write_start_request(conn, op);
}
int rpc_write(conn_t *conn, const void *data, size_t size) {
  return lupine_rpc_write(conn, data, size);
}
int rpc_wait_for_response(conn_t *conn) {
  return lupine_rpc_wait_for_response(conn);
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  return lupine_rpc_read(conn, data, size);
}
int rpc_read_end(conn_t *conn) { return lupine_rpc_read_end(conn); }

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

// The sticky error every call feeds and cudaGetLastError drains.
thread_local cudaError_t local_error = cudaSuccess;

cudaError_t record(cudaError_t error) {
  if (error != cudaSuccess) {
    local_error = error;
  }
  return error;
}

// The two enumerations share their numbering for every code a driver call can
// return; only the reused values below disagree.
cudaError_t translate(CUresult result) {
  switch (result) {
  case CUDA_SUCCESS:
    return cudaSuccess;
  case CUDA_ERROR_DEINITIALIZED:
    return cudaErrorCudartUnloading;
  case CUDA_ERROR_INVALID_CONTEXT:
    return cudaErrorDeviceUninitialized;
  default:
    return static_cast<cudaError_t>(result);
  }
}

cudaError_t from_driver(CUresult result) { return record(translate(result)); }

// ---------------------------------------------------------------------------
// Device and context state
// ---------------------------------------------------------------------------

// The runtime's current device is per thread, and a thread's RPCs share one
// lane, so the server's runtime sees the same per-thread device once
// cudaSetDevice has been forwarded there. The driver shim's current context
// has to follow as well: it decides where the driver-backed calls below land.
thread_local int current_device = 0;
thread_local bool current_device_bound = false;

void ensure_init() {
  static std::once_flag once;
  std::call_once(once, [] { (void)cuInit(0); });
}

cudaError_t bind_current_device() {
  if (current_device_bound) {
    return cudaSuccess;
  }
  CUcontext context = nullptr;
  CUresult result = cuDevicePrimaryCtxRetain(&context, current_device);
  if (result == CUDA_SUCCESS) {
    result = cuCtxSetCurrent(context);
  }
  if (result != CUDA_SUCCESS) {
    return translate(result);
  }
  current_device_bound = true;
  return cudaSuccess;
}

// Every driver-backed entry point starts here: the driver has to be
// initialized and the thread's device selected before the call routes.
cudaError_t enter() {
  ensure_init();
  return bind_current_device();
}

conn_t *connection_for_device(int *device) {
  ensure_init();
  return lupine_rpc_conn_for_device(device);
}

conn_t *connection() {
  if (enter() != cudaSuccess) {
    return nullptr;
  }
  int device = current_device;
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_stream(cudaStream_t stream) {
  ensure_init();
  return stream == nullptr ? connection() : lupine_rpc_conn_for_stream(stream);
}

conn_t *connection_for_event(cudaEvent_t event) {
  ensure_init();
  return event == nullptr ? connection() : lupine_rpc_conn_for_event(event);
}

void note_stream_owner(conn_t *conn, cudaStream_t stream) {
  lupine_rpc_note_stream_owner(conn, stream);
}

void note_event_owner(conn_t *conn, cudaEvent_t event) {
  lupine_rpc_note_event_owner(conn, event);
}

} // namespace

#include "codegen/gen_cudart_client.inc"

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  if (count == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  ensure_init();
  return lupine_rpc_device_count(count) < 0 ? record(rpc_error()) : cudaSuccess;
}

extern "C" cudaError_t cudaGetDevice(int *device) {
  if (device == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  *device = current_device;
  return cudaSuccess;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  int remote_device = device;
  conn_t *conn = connection_for_device(&remote_device);
  if (conn == nullptr) {
    return record(cudaErrorInvalidDevice);
  }
  cudaError_t result = lupine_rpc_cudaSetDevice(conn, remote_device);
  if (result != cudaSuccess) {
    return record(result);
  }
  current_device = device;
  current_device_bound = false;
  return record(bind_current_device());
}

#if CUDART_VERSION >= 12000
extern "C" cudaError_t cudaInitDevice(int device, unsigned int deviceFlags,
                                      unsigned int flags) {
  int remote_device = device;
  conn_t *conn = connection_for_device(&remote_device);
  if (conn == nullptr) {
    return record(cudaErrorInvalidDevice);
  }
  cudaError_t result =
      lupine_rpc_cudaInitDevice(conn, remote_device, deviceFlags, flags);
  if (result != cudaSuccess) {
    return record(result);
  }
  CUcontext context = nullptr;
  return from_driver(cuDevicePrimaryCtxRetain(&context, device));
}
#endif

extern "C" cudaError_t cudaSetValidDevices(int *device_arr, int len) {
  if (len < 0 || (len != 0 && device_arr == nullptr)) {
    return record(cudaErrorInvalidValue);
  }
  std::vector<int> remote_devices(static_cast<size_t>(len));
  conn_t *conn = nullptr;
  for (int i = 0; i < len; ++i) {
    remote_devices[i] = device_arr[i];
    conn_t *device_conn = connection_for_device(&remote_devices[i]);
    if (device_conn == nullptr) {
      return record(cudaErrorInvalidDevice);
    }
    // The list is a single preference order, so it can only be forwarded to
    // one server; devices behind another connection are not expressible.
    if (conn != nullptr && device_conn != conn) {
      return record(cudaErrorNotSupported);
    }
    conn = device_conn;
  }
  if (conn == nullptr) {
    conn = connection();
  }
  return record(
      lupine_rpc_cudaSetValidDevices(conn, remote_devices.data(), len));
}

extern "C" cudaError_t cudaDeviceReset() {
  conn_t *conn = connection();
  cudaError_t result = lupine_rpc_cudaDeviceReset(conn);
  current_device_bound = false;
  local_error = cudaSuccess;
  return record(result);
}

// Every generated wrapper records its result, and so does every call this file
// answers itself, so the sticky error lives here instead of in the server's
// runtime: keeping it in both would lose the order the two were set in.
extern "C" cudaError_t cudaGetLastError() {
  const cudaError_t error = local_error;
  local_error = cudaSuccess;
  return error;
}

extern "C" cudaError_t cudaPeekAtLastError() { return local_error; }

namespace {

// The server's runtime owns the message strings, so they are fetched once and
// held per thread: the returned pointer stays valid until this thread's next
// lookup of the same kind. The shim's own unreachable-server code is answered
// locally, because its occurrence means the forward path may already be gone.
const char *error_text(int op, cudaError_t error, bool want_name) {
  if (error == cudaErrorDevicesUnavailable) {
    return want_name ? "cudaErrorDevicesUnavailable"
                     : "CUDA-capable device(s) is/are busy or unavailable";
  }
  thread_local std::string name_result;
  thread_local std::string description_result;
  std::string &result = want_name ? name_result : description_result;

  constexpr uint32_t kMaxLength = 4096;
  conn_t *conn = connection();
  uint32_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &error, sizeof(error)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 || length > kMaxLength) {
    return want_name ? "cudaErrorUnknown" : "unknown error";
  }
  std::string text(length, '\0');
  if ((length != 0 && rpc_read(conn, &text[0], length) < 0) ||
      rpc_read_end(conn) < 0) {
    return want_name ? "cudaErrorUnknown" : "unknown error";
  }
  result = std::move(text);
  return result.c_str();
}

} // namespace

extern "C" const char *cudaGetErrorName(cudaError_t error) {
  return error_text(RPC_cudaGetErrorName, error, true);
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
  return error_text(RPC_cudaGetErrorString, error, false);
}

extern "C" struct cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w,
                      enum cudaChannelFormatKind f) {
  struct cudaChannelFormatDesc desc;
  desc.x = x;
  desc.y = y;
  desc.z = z;
  desc.w = w;
  desc.f = f;
  return desc;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  if (devPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  if (size == 0) {
    *devPtr = nullptr;
    return cudaSuccess;
  }
  CUdeviceptr pointer = 0;
  CUresult result = cuMemAlloc(&pointer, size);
  *devPtr = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaFree(void *devPtr) {
  if (devPtr == nullptr) {
    return cudaSuccess;
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuMemFree(reinterpret_cast<CUdeviceptr>(devPtr)));
}

extern "C" cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
                                       size_t width, size_t height) {
  if (devPtr == nullptr || pitch == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  // The runtime asks for the widest alignment the driver offers.
  CUresult result = cuMemAllocPitch(&pointer, pitch, width, height, 16);
  *devPtr = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
                                    struct cudaExtent extent) {
  if (pitchedDevPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  size_t pitch = 0;
  CUresult result = cuMemAllocPitch(&pointer, &pitch, extent.width,
                                    extent.height * extent.depth, 16);
  pitchedDevPtr->ptr = reinterpret_cast<void *>(pointer);
  pitchedDevPtr->pitch = pitch;
  pitchedDevPtr->xsize = extent.width;
  pitchedDevPtr->ysize = extent.height;
  return from_driver(result);
}

extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                         unsigned int flags) {
  if (devPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  CUresult result = cuMemAllocManaged(&pointer, size, flags);
  *devPtr = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaMallocAsync(void **devPtr, size_t size,
                                       cudaStream_t hStream) {
  if (devPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  CUresult result =
      cuMemAllocAsync(&pointer, size, reinterpret_cast<CUstream>(hStream));
  *devPtr = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaMallocFromPoolAsync(void **ptr, size_t size,
                                               cudaMemPool_t memPool,
                                               cudaStream_t stream) {
  if (ptr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  CUresult result = cuMemAllocFromPoolAsync(
      &pointer, size, reinterpret_cast<CUmemoryPool>(memPool),
      reinterpret_cast<CUstream>(stream));
  *ptr = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
  if (devPtr == nullptr) {
    return cudaSuccess;
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(devPtr),
                                    reinterpret_cast<CUstream>(hStream)));
}

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
  if (ptr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuMemHostAlloc(ptr, size, 0));
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size,
                                     unsigned int flags) {
  if (pHost == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  // cudaHostAlloc* and CU_MEMHOSTALLOC_* share their bit values.
  return from_driver(cuMemHostAlloc(pHost, size, flags));
}

extern "C" cudaError_t cudaFreeHost(void *ptr) {
  if (ptr == nullptr) {
    return cudaSuccess;
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuMemFreeHost(ptr));
}

extern "C" cudaError_t cudaHostRegister(void *ptr, size_t size,
                                        unsigned int flags) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  // cudaHostRegister* and CU_MEMHOSTREGISTER_* share their bit values.
  return from_driver(cuMemHostRegister(ptr, size, flags));
}

extern "C" cudaError_t cudaHostUnregister(void *ptr) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuMemHostUnregister(ptr));
}

extern "C" cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                                unsigned int flags) {
  if (pDevice == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  CUresult result = cuMemHostGetDevicePointer(&pointer, pHost, flags);
  *pDevice = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
  if (pFlags == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuMemHostGetFlags(pFlags, pHost));
}

namespace {

bool is_device_pointer(const void *pointer) {
  if (pointer == nullptr) {
    return false;
  }
  unsigned int type = CU_MEMORYTYPE_HOST;
  if (cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            reinterpret_cast<CUdeviceptr>(pointer)) !=
      CUDA_SUCCESS) {
    return false;
  }
  return type == CU_MEMORYTYPE_DEVICE || type == CU_MEMORYTYPE_UNIFIED;
}

// cudaMemcpyDefault leaves the direction to the pointers, so it is resolved
// the same way the runtime does: by asking the driver what each one is.
void resolve_kind(void *dst, const void *src, cudaMemcpyKind kind,
                  bool *dst_is_device, bool *src_is_device) {
  switch (kind) {
  case cudaMemcpyHostToDevice:
    *dst_is_device = true;
    *src_is_device = false;
    return;
  case cudaMemcpyDeviceToHost:
    *dst_is_device = false;
    *src_is_device = true;
    return;
  case cudaMemcpyDeviceToDevice:
    *dst_is_device = true;
    *src_is_device = true;
    return;
  case cudaMemcpyHostToHost:
    *dst_is_device = false;
    *src_is_device = false;
    return;
  default:
    *dst_is_device = is_device_pointer(dst);
    *src_is_device = is_device_pointer(src);
    return;
  }
}

CUresult copy_1d(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                 CUstream stream, bool async) {
  bool dst_is_device = false;
  bool src_is_device = false;
  resolve_kind(dst, src, kind, &dst_is_device, &src_is_device);
  const CUdeviceptr dst_device = reinterpret_cast<CUdeviceptr>(dst);
  const CUdeviceptr src_device = reinterpret_cast<CUdeviceptr>(src);
  if (dst_is_device && src_is_device) {
    return async ? cuMemcpyDtoDAsync(dst_device, src_device, count, stream)
                 : cuMemcpyDtoD(dst_device, src_device, count);
  }
  if (dst_is_device) {
    return async ? cuMemcpyHtoDAsync(dst_device, src, count, stream)
                 : cuMemcpyHtoD(dst_device, src, count);
  }
  if (src_is_device) {
    return async ? cuMemcpyDtoHAsync(dst, src_device, count, stream)
                 : cuMemcpyDtoH(dst, src_device, count);
  }
  std::memcpy(dst, src, count);
  return CUDA_SUCCESS;
}

void set_2d_source(CUDA_MEMCPY2D *copy, const void *src, size_t pitch,
                   bool device) {
  copy->srcMemoryType = device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
  copy->srcHost = device ? nullptr : src;
  copy->srcDevice = device ? reinterpret_cast<CUdeviceptr>(src) : 0;
  copy->srcPitch = pitch;
}

void set_2d_destination(CUDA_MEMCPY2D *copy, void *dst, size_t pitch,
                        bool device) {
  copy->dstMemoryType = device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
  copy->dstHost = device ? nullptr : dst;
  copy->dstDevice = device ? reinterpret_cast<CUdeviceptr>(dst) : 0;
  copy->dstPitch = pitch;
}

CUresult copy_2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                 size_t width, size_t height, cudaMemcpyKind kind,
                 CUstream stream, bool async) {
  bool dst_is_device = false;
  bool src_is_device = false;
  resolve_kind(dst, src, kind, &dst_is_device, &src_is_device);
  CUDA_MEMCPY2D copy = {};
  set_2d_source(&copy, src, spitch, src_is_device);
  set_2d_destination(&copy, dst, dpitch, dst_is_device);
  copy.WidthInBytes = width;
  copy.Height = height;
  return async ? cuMemcpy2DAsync(&copy, stream) : cuMemcpy2D(&copy);
}

CUresult copy_3d(const struct cudaMemcpy3DParms *parms, CUstream stream,
                 bool async) {
  if (parms->srcArray != nullptr || parms->dstArray != nullptr) {
    // A CUDA array position is measured in elements, so translating it needs
    // the array's element size; array copies are not implemented yet.
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  bool dst_is_device = false;
  bool src_is_device = false;
  resolve_kind(parms->dstPtr.ptr, parms->srcPtr.ptr, parms->kind,
               &dst_is_device, &src_is_device);
  CUDA_MEMCPY3D copy = {};
  copy.srcXInBytes = parms->srcPos.x;
  copy.srcY = parms->srcPos.y;
  copy.srcZ = parms->srcPos.z;
  copy.srcMemoryType =
      src_is_device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
  copy.srcHost = src_is_device ? nullptr : parms->srcPtr.ptr;
  copy.srcDevice =
      src_is_device ? reinterpret_cast<CUdeviceptr>(parms->srcPtr.ptr) : 0;
  copy.srcPitch = parms->srcPtr.pitch;
  copy.srcHeight = parms->srcPtr.ysize;
  copy.dstXInBytes = parms->dstPos.x;
  copy.dstY = parms->dstPos.y;
  copy.dstZ = parms->dstPos.z;
  copy.dstMemoryType =
      dst_is_device ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
  copy.dstHost = dst_is_device ? nullptr : parms->dstPtr.ptr;
  copy.dstDevice =
      dst_is_device ? reinterpret_cast<CUdeviceptr>(parms->dstPtr.ptr) : 0;
  copy.dstPitch = parms->dstPtr.pitch;
  copy.dstHeight = parms->dstPtr.ysize;
  copy.WidthInBytes = parms->extent.width;
  copy.Height = parms->extent.height;
  copy.Depth = parms->extent.depth;
  return async ? cuMemcpy3DAsync(&copy, stream) : cuMemcpy3D(&copy);
}

CUresult primary_context(int device, CUcontext *context) {
  return cuDevicePrimaryCtxRetain(context, device);
}

} // namespace

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  enum cudaMemcpyKind kind) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(copy_1d(dst, src, count, kind, nullptr, false));
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                       enum cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(
      copy_1d(dst, src, count, kind, reinterpret_cast<CUstream>(stream), true));
}

extern "C" cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                                    size_t spitch, size_t width, size_t height,
                                    enum cudaMemcpyKind kind) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(
      copy_2d(dst, dpitch, src, spitch, width, height, kind, nullptr, false));
}

extern "C" cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
                                         const void *src, size_t spitch,
                                         size_t width, size_t height,
                                         enum cudaMemcpyKind kind,
                                         cudaStream_t stream) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(copy_2d(dst, dpitch, src, spitch, width, height, kind,
                             reinterpret_cast<CUstream>(stream), true));
}

extern "C" cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
  if (p == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(copy_3d(p, nullptr, false));
}

extern "C" cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
                                         cudaStream_t stream) {
  if (p == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(copy_3d(p, reinterpret_cast<CUstream>(stream), true));
}

extern "C" cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                                      int srcDevice, size_t count) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUcontext dst_context = nullptr;
  CUcontext src_context = nullptr;
  CUresult result = primary_context(dstDevice, &dst_context);
  if (result == CUDA_SUCCESS) {
    result = primary_context(srcDevice, &src_context);
  }
  if (result != CUDA_SUCCESS) {
    return from_driver(result);
  }
  return from_driver(
      cuMemcpyPeer(reinterpret_cast<CUdeviceptr>(dst), dst_context,
                   reinterpret_cast<CUdeviceptr>(src), src_context, count));
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                           const void *src, int srcDevice,
                                           size_t count, cudaStream_t stream) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUcontext dst_context = nullptr;
  CUcontext src_context = nullptr;
  CUresult result = primary_context(dstDevice, &dst_context);
  if (result == CUDA_SUCCESS) {
    result = primary_context(srcDevice, &src_context);
  }
  if (result != CUDA_SUCCESS) {
    return from_driver(result);
  }
  return from_driver(
      cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dst), dst_context,
                        reinterpret_cast<CUdeviceptr>(src), src_context, count,
                        reinterpret_cast<CUstream>(stream)));
}

namespace {

CUresult copy_3d_peer(const struct cudaMemcpy3DPeerParms *parms,
                      CUstream stream, bool async) {
  if (parms->srcArray != nullptr || parms->dstArray != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUcontext dst_context = nullptr;
  CUcontext src_context = nullptr;
  CUresult result = primary_context(parms->dstDevice, &dst_context);
  if (result == CUDA_SUCCESS) {
    result = primary_context(parms->srcDevice, &src_context);
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  CUDA_MEMCPY3D_PEER copy = {};
  copy.srcXInBytes = parms->srcPos.x;
  copy.srcY = parms->srcPos.y;
  copy.srcZ = parms->srcPos.z;
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = reinterpret_cast<CUdeviceptr>(parms->srcPtr.ptr);
  copy.srcContext = src_context;
  copy.srcPitch = parms->srcPtr.pitch;
  copy.srcHeight = parms->srcPtr.ysize;
  copy.dstXInBytes = parms->dstPos.x;
  copy.dstY = parms->dstPos.y;
  copy.dstZ = parms->dstPos.z;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = reinterpret_cast<CUdeviceptr>(parms->dstPtr.ptr);
  copy.dstContext = dst_context;
  copy.dstPitch = parms->dstPtr.pitch;
  copy.dstHeight = parms->dstPtr.ysize;
  copy.WidthInBytes = parms->extent.width;
  copy.Height = parms->extent.height;
  copy.Depth = parms->extent.depth;
  return async ? cuMemcpy3DPeerAsync(&copy, stream) : cuMemcpy3DPeer(&copy);
}

} // namespace

extern "C" cudaError_t cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p) {
  if (p == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(copy_3d_peer(p, nullptr, false));
}

extern "C" cudaError_t
cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p,
                      cudaStream_t stream) {
  if (p == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(copy_3d_peer(p, reinterpret_cast<CUstream>(stream), true));
}

extern "C" cudaError_t
cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                         const void *ptr) {
  if (attributes == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  std::memset(attributes, 0, sizeof(*attributes));
  const CUdeviceptr pointer = reinterpret_cast<CUdeviceptr>(ptr);
  unsigned int memory_type = 0;
  CUresult result = cuPointerGetAttribute(
      &memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result != CUDA_SUCCESS) {
    // An unregistered host pointer is not an error for the runtime.
    attributes->type = cudaMemoryTypeUnregistered;
    return cudaSuccess;
  }
  int ordinal = 0;
  (void)cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                              pointer);
  attributes->device = ordinal;
  CUdeviceptr device_pointer = 0;
  if (cuPointerGetAttribute(&device_pointer,
                            CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                            pointer) == CUDA_SUCCESS) {
    attributes->devicePointer = reinterpret_cast<void *>(device_pointer);
  }
  void *host_pointer = nullptr;
  if (cuPointerGetAttribute(&host_pointer, CU_POINTER_ATTRIBUTE_HOST_POINTER,
                            pointer) == CUDA_SUCCESS) {
    attributes->hostPointer = host_pointer;
  }
  int is_managed = 0;
  (void)cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED,
                              pointer);
  if (is_managed != 0) {
    attributes->type = cudaMemoryTypeManaged;
  } else if (memory_type == CU_MEMORYTYPE_DEVICE) {
    attributes->type = cudaMemoryTypeDevice;
  } else if (memory_type == CU_MEMORYTYPE_HOST) {
    attributes->type = cudaMemoryTypeHost;
  } else {
    attributes->type = cudaMemoryTypeUnregistered;
  }
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Attribute values whose width the attribute itself decides
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(
      cuMemPoolGetAttribute(reinterpret_cast<CUmemoryPool>(memPool),
                            static_cast<CUmemPool_attribute>(attr), value));
}

extern "C" cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(
      cuMemPoolSetAttribute(reinterpret_cast<CUmemoryPool>(memPool),
                            static_cast<CUmemPool_attribute>(attr), value));
}

extern "C" cudaError_t
cudaDeviceGetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                               void *value) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuDeviceGetGraphMemAttribute(
      device, static_cast<CUgraphMem_attribute>(attr), value));
}

extern "C" cudaError_t
cudaDeviceSetGraphMemAttribute(int device, enum cudaGraphMemAttributeType attr,
                               void *value) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuDeviceSetGraphMemAttribute(
      device, static_cast<CUgraphMem_attribute>(attr), value));
}

extern "C" cudaError_t cudaMemRangeGetAttributes(
    void **data, size_t *dataSizes, enum cudaMemRangeAttribute *attributes,
    size_t numAttributes, const void *devPtr, size_t count) {
  if ((numAttributes != 0 &&
       (data == nullptr || dataSizes == nullptr || attributes == nullptr))) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  for (size_t i = 0; i < numAttributes; ++i) {
    CUresult result = cuMemRangeGetAttribute(
        data[i], dataSizes[i],
        static_cast<CUmem_range_attribute>(attributes[i]),
        reinterpret_cast<CUdeviceptr>(devPtr), count);
    if (result != CUDA_SUCCESS) {
      return from_driver(result);
    }
  }
  return cudaSuccess;
}

// ---------------------------------------------------------------------------
// Module registration and kernels
//
// nvcc emits a registration call per translation unit that hands the runtime
// the fatbin it embedded, then one call per kernel and per device variable.
// Those images and host addresses only exist in the client, so the shim keeps
// its own registry and resolves each entry through the driver shim's module
// loader on first use.
// ---------------------------------------------------------------------------

namespace {

struct fatbin_registration {
  const void *image = nullptr;
  std::unordered_map<int, CUmodule> modules;
};

struct function_registration {
  void **fatbin = nullptr;
  std::string name;
  std::unordered_map<int, CUfunction> functions;
};

struct variable_registration {
  void **fatbin = nullptr;
  std::string name;
  size_t size = 0;
};

std::mutex &registry_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<void **, fatbin_registration> &fatbins() {
  static auto *map = new std::unordered_map<void **, fatbin_registration>();
  return *map;
}

std::unordered_map<const void *, function_registration> &
registered_functions() {
  static auto *map =
      new std::unordered_map<const void *, function_registration>();
  return *map;
}

std::unordered_map<const void *, variable_registration> &
registered_variables() {
  static auto *map =
      new std::unordered_map<const void *, variable_registration>();
  return *map;
}

// registry_mutex() must be held.
CUresult module_for(void **fatbin, int device, CUmodule *module) {
  auto entry = fatbins().find(fatbin);
  if (entry == fatbins().end()) {
    return CUDA_ERROR_NOT_FOUND;
  }
  auto loaded = entry->second.modules.find(device);
  if (loaded != entry->second.modules.end()) {
    *module = loaded->second;
    return CUDA_SUCCESS;
  }
  CUmodule created = nullptr;
  CUresult result = cuModuleLoadData(&created, entry->second.image);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  entry->second.modules[device] = created;
  *module = created;
  return CUDA_SUCCESS;
}

CUresult function_for(const void *host_function, CUfunction *function) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = registered_functions().find(host_function);
  if (entry == registered_functions().end()) {
    return CUDA_ERROR_INVALID_IMAGE;
  }
  auto resolved = entry->second.functions.find(current_device);
  if (resolved != entry->second.functions.end()) {
    *function = resolved->second;
    return CUDA_SUCCESS;
  }
  CUmodule module = nullptr;
  CUresult result = module_for(entry->second.fatbin, current_device, &module);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  CUfunction created = nullptr;
  result = cuModuleGetFunction(&created, module, entry->second.name.c_str());
  if (result != CUDA_SUCCESS) {
    return result;
  }
  entry->second.functions[current_device] = created;
  *function = created;
  return CUDA_SUCCESS;
}

CUresult variable_for(const void *symbol, CUdeviceptr *pointer, size_t *size) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = registered_variables().find(symbol);
  if (entry == registered_variables().end()) {
    return CUDA_ERROR_NOT_FOUND;
  }
  CUmodule module = nullptr;
  CUresult result = module_for(entry->second.fatbin, current_device, &module);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  size_t resolved_size = 0;
  result = cuModuleGetGlobal(pointer, &resolved_size, module,
                             entry->second.name.c_str());
  if (result == CUDA_SUCCESS && size != nullptr) {
    *size = resolved_size;
  }
  return result;
}

// Runs a block against a device other than the calling thread's. Modules load
// into the current context, and the per-device caches are keyed by the current
// device, so both have to move together and move back.
class device_scope {
public:
  explicit device_scope(int device) : previous_(current_device) {
    if (device == current_device) {
      return;
    }
    CUcontext context = nullptr;
    CUresult result = cuDevicePrimaryCtxRetain(&context, device);
    if (result == CUDA_SUCCESS) {
      result = cuCtxSetCurrent(context);
    }
    if (result != CUDA_SUCCESS) {
      status_ = translate(result);
      return;
    }
    current_device = device;
    restore_ = true;
  }

  ~device_scope() {
    if (!restore_) {
      return;
    }
    current_device = previous_;
    current_device_bound = false;
    (void)bind_current_device();
  }

  cudaError_t status() const { return status_; }

private:
  int previous_;
  bool restore_ = false;
  cudaError_t status_ = cudaSuccess;
};

cudaError_t resolve_function(const void *host_function, CUfunction *function) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return entered;
  }
  return translate(function_for(host_function, function));
}

struct call_configuration {
  dim3 grid;
  dim3 block;
  size_t shared;
  cudaStream_t stream;
};

thread_local std::vector<call_configuration> call_configurations;

// The runtime and driver launch configurations describe the same thing, and
// their attribute lists share a layout, so only the dimensions are unpacked.
CUlaunchConfig to_driver_config(const cudaLaunchConfig_t &config) {
  static_assert(sizeof(cudaLaunchAttribute) == sizeof(CUlaunchAttribute),
                "launch attributes must share their layout");
  CUlaunchConfig driver_config = {};
  driver_config.gridDimX = config.gridDim.x;
  driver_config.gridDimY = config.gridDim.y;
  driver_config.gridDimZ = config.gridDim.z;
  driver_config.blockDimX = config.blockDim.x;
  driver_config.blockDimY = config.blockDim.y;
  driver_config.blockDimZ = config.blockDim.z;
  driver_config.sharedMemBytes =
      static_cast<unsigned int>(config.dynamicSmemBytes);
  driver_config.hStream = reinterpret_cast<CUstream>(config.stream);
  driver_config.attrs = reinterpret_cast<CUlaunchAttribute *>(config.attrs);
  driver_config.numAttrs = config.numAttrs;
  return driver_config;
}

} // namespace

extern "C" void **__cudaRegisterFatBinary(void *fatCubin) {
  const void *image = fatCubin;
  const auto *wrapper = static_cast<const lupine_fatbin_wrapper *>(fatCubin);
  if (wrapper != nullptr && wrapper->magic == LUPINE_FATBINC_MAGIC) {
    image = wrapper->data;
  }
  auto *handle = new void *[1];
  handle[0] = fatCubin;
  std::lock_guard<std::mutex> lock(registry_mutex());
  fatbins()[handle].image = image;
  return handle;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  // Modules load on first use, so there is nothing to finish here.
  (void)fatCubinHandle;
}

extern "C" void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = fatbins().find(fatCubinHandle);
  if (entry == fatbins().end()) {
    return;
  }
  for (const auto &module : entry->second.modules) {
    (void)cuModuleUnload(module.second);
  }
  fatbins().erase(entry);
  delete[] fatCubinHandle;
}

extern "C" char __cudaInitModule(void **fatCubinHandle) {
  (void)fatCubinHandle;
  return 1;
}

extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *bid, dim3 *bDim,
                                       dim3 *gDim, int *wSize) {
  (void)deviceFun;
  (void)thread_limit;
  (void)tid;
  (void)bid;
  (void)bDim;
  (void)gDim;
  (void)wSize;
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto &entry = registered_functions()[hostFun];
  entry.fatbin = fatCubinHandle;
  entry.name = deviceName;
}

extern "C" void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                  char *deviceAddress, const char *deviceName,
                                  int ext, size_t size, int constant,
                                  int global) {
  (void)deviceAddress;
  (void)ext;
  (void)constant;
  (void)global;
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto &entry = registered_variables()[hostVar];
  entry.fatbin = fatCubinHandle;
  entry.name = deviceName;
  entry.size = size;
}

extern "C" void
__cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                         char *deviceAddress, const char *deviceName, int ext,
                         size_t size, int constant, int global) {
  (void)deviceAddress;
  (void)ext;
  (void)constant;
  (void)global;
  if (hostVarPtrAddress == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto &entry = registered_variables()[hostVarPtrAddress];
  entry.fatbin = fatCubinHandle;
  entry.name = deviceName;
  entry.size = size;
}

extern "C" void __cudaRegisterTexture(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim, int norm,
                                      int ext) {
  // Texture references were removed from the device API; only texture objects
  // are supported, and those carry no host-side registration.
  (void)fatCubinHandle;
  (void)hostVar;
  (void)deviceAddress;
  (void)deviceName;
  (void)dim;
  (void)norm;
  (void)ext;
}

extern "C" void __cudaRegisterSurface(void **fatCubinHandle,
                                      const void *hostVar,
                                      const void **deviceAddress,
                                      const char *deviceName, int dim,
                                      int ext) {
  (void)fatCubinHandle;
  (void)hostVar;
  (void)deviceAddress;
  (void)deviceName;
  (void)dim;
  (void)ext;
}

extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem,
                                                void *stream) {
  call_configurations.push_back(
      {gridDim, blockDim, sharedMem, static_cast<cudaStream_t>(stream)});
  return 0;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                                  size_t *sharedMem,
                                                  void *stream) {
  if (call_configurations.empty()) {
    return cudaErrorInvalidConfiguration;
  }
  const call_configuration configuration = call_configurations.back();
  call_configurations.pop_back();
  if (gridDim != nullptr) {
    *gridDim = configuration.grid;
  }
  if (blockDim != nullptr) {
    *blockDim = configuration.block;
  }
  if (sharedMem != nullptr) {
    *sharedMem = configuration.shared;
  }
  if (stream != nullptr) {
    *static_cast<cudaStream_t *>(stream) = configuration.stream;
  }
  return cudaSuccess;
}

#if CUDART_VERSION >= 13000
// nvcc's launch lowering resolves the host entry point once per launch site
// and passes the handle back on every launch. A kernel is per device here, so
// the handle stays the entry-point address and the launch resolves it against
// the calling thread's device.
extern "C" cudaError_t __cudaGetKernel(cudaKernel_t *kernel,
                                       const void *entryFuncAddr) {
  if (kernel == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  *kernel = reinterpret_cast<cudaKernel_t>(const_cast<void *>(entryFuncAddr));
  return cudaSuccess;
}

extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                                          dim3 blockDim, void **args,
                                          size_t sharedMem,
                                          cudaStream_t stream) {
  return cudaLaunchKernel(reinterpret_cast<const void *>(kernel), gridDim,
                          blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel,
                                               dim3 gridDim, dim3 blockDim,
                                               void **args, size_t sharedMem,
                                               cudaStream_t stream) {
  return cudaLaunchKernel(reinterpret_cast<const void *>(kernel), gridDim,
                          blockDim, args, sharedMem, stream);
}
#endif

// Host variables carry no device storage, so registering one only has to be
// accepted. The same is true of the unified-addressing table nvcc hands over.
extern "C" void __cudaRegisterHostVar(void **fatCubinHandle,
                                      const char *deviceName, char *hostVar,
                                      size_t size) {
  (void)fatCubinHandle;
  (void)deviceName;
  (void)hostVar;
  (void)size;
}

extern "C" void __cudaRegisterUnifiedTable(void *table) { (void)table; }

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream) {
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(cuLaunchKernel(
      function, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
      blockDim.z, static_cast<unsigned int>(sharedMem),
      reinterpret_cast<CUstream>(stream), args, nullptr));
}

extern "C" cudaError_t cudaLaunchCooperativeKernel(const void *func,
                                                   dim3 gridDim, dim3 blockDim,
                                                   void **args,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(cuLaunchCooperativeKernel(
      function, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
      blockDim.z, static_cast<unsigned int>(sharedMem),
      reinterpret_cast<CUstream>(stream), args));
}

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                           const void *func, void **args) {
  if (config == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  const CUlaunchConfig driver_config = to_driver_config(*config);
  return from_driver(cuLaunchKernelEx(&driver_config, function, args, nullptr));
}

extern "C" cudaError_t
cudaOccupancyMaxPotentialClusterSize(int *clusterSize, const void *func,
                                     const cudaLaunchConfig_t *launchConfig) {
  if (clusterSize == nullptr || launchConfig == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  const CUlaunchConfig driver_config = to_driver_config(*launchConfig);
  return from_driver(cuOccupancyMaxPotentialClusterSize(clusterSize, function,
                                                        &driver_config));
}

extern "C" cudaError_t
cudaOccupancyMaxActiveClusters(int *numClusters, const void *func,
                               const cudaLaunchConfig_t *launchConfig) {
  if (numClusters == nullptr || launchConfig == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  const CUlaunchConfig driver_config = to_driver_config(*launchConfig);
  return from_driver(
      cuOccupancyMaxActiveClusters(numClusters, function, &driver_config));
}

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                             const void *func) {
  if (attr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  std::memset(attr, 0, sizeof(*attr));
  struct {
    CUfunction_attribute attribute;
    int *value;
  } const integers[] = {
      {CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, &attr->maxThreadsPerBlock},
      {CU_FUNC_ATTRIBUTE_NUM_REGS, &attr->numRegs},
      {CU_FUNC_ATTRIBUTE_PTX_VERSION, &attr->ptxVersion},
      {CU_FUNC_ATTRIBUTE_BINARY_VERSION, &attr->binaryVersion},
      {CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, &attr->cacheModeCA},
      {CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
       &attr->maxDynamicSharedSizeBytes},
      {CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
       &attr->preferredShmemCarveout},
  };
  for (const auto &entry : integers) {
    int value = 0;
    CUresult result = cuFuncGetAttribute(&value, entry.attribute, function);
    if (result != CUDA_SUCCESS) {
      return from_driver(result);
    }
    *entry.value = value;
  }
  struct {
    CUfunction_attribute attribute;
    size_t *value;
  } const sizes[] = {
      {CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, &attr->sharedSizeBytes},
      {CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, &attr->constSizeBytes},
      {CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, &attr->localSizeBytes},
  };
  for (const auto &entry : sizes) {
    int value = 0;
    CUresult result = cuFuncGetAttribute(&value, entry.attribute, function);
    if (result != CUDA_SUCCESS) {
      return from_driver(result);
    }
    *entry.value = static_cast<size_t>(value);
  }
  return cudaSuccess;
}

extern "C" cudaError_t
cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(cuFuncSetAttribute(
      function, static_cast<CUfunction_attribute>(attr), value));
}

extern "C" cudaError_t cudaFuncSetCacheConfig(const void *func,
                                              enum cudaFuncCache cacheConfig) {
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(
      cuFuncSetCacheConfig(function, static_cast<CUfunc_cache>(cacheConfig)));
}

extern "C" cudaError_t cudaGetFuncBySymbol(cudaFunction_t *functionPtr,
                                           const void *symbolPtr) {
  if (functionPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(symbolPtr, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  *functionPtr = reinterpret_cast<cudaFunction_t>(function);
  return cudaSuccess;
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize) {
  if (numBlocks == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, function, blockSize, dynamicSMemSize));
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (numBlocks == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      numBlocks, function, blockSize, dynamicSMemSize, flags));
}

extern "C" cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(
    size_t *dynamicSmemSize, const void *func, int numBlocks, int blockSize) {
  if (dynamicSmemSize == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(cuOccupancyAvailableDynamicSMemPerBlock(
      dynamicSmemSize, function, numBlocks, blockSize));
}

// ---------------------------------------------------------------------------
// Device symbols
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr, const void *symbol) {
  if (devPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  CUresult result = variable_for(symbol, &pointer, nullptr);
  *devPtr = reinterpret_cast<void *>(pointer);
  return from_driver(result);
}

extern "C" cudaError_t cudaGetSymbolSize(size_t *size, const void *symbol) {
  if (size == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  return from_driver(variable_for(symbol, &pointer, size));
}

namespace {

cudaError_t copy_symbol(const void *symbol, void *host, size_t count,
                        size_t offset, cudaMemcpyKind kind, CUstream stream,
                        bool async, bool to_symbol) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  CUdeviceptr pointer = 0;
  size_t size = 0;
  CUresult resolved = variable_for(symbol, &pointer, &size);
  if (resolved != CUDA_SUCCESS) {
    return from_driver(resolved);
  }
  if (offset > size || count > size - offset) {
    return record(cudaErrorInvalidValue);
  }
  void *device = reinterpret_cast<void *>(pointer + offset);
  const cudaMemcpyKind resolved_kind =
      kind == cudaMemcpyDefault
          ? (to_symbol ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost)
          : kind;
  return from_driver(
      to_symbol ? copy_1d(device, host, count, resolved_kind, stream, async)
                : copy_1d(host, device, count, resolved_kind, stream, async));
}

} // namespace

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                          size_t count, size_t offset,
                                          enum cudaMemcpyKind kind) {
  return copy_symbol(symbol, const_cast<void *>(src), count, offset, kind,
                     nullptr, false, true);
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                            size_t count, size_t offset,
                                            enum cudaMemcpyKind kind) {
  return copy_symbol(symbol, dst, count, offset, kind, nullptr, false, false);
}

extern "C" cudaError_t cudaMemcpyToSymbolAsync(const void *symbol,
                                               const void *src, size_t count,
                                               size_t offset,
                                               enum cudaMemcpyKind kind,
                                               cudaStream_t stream) {
  return copy_symbol(symbol, const_cast<void *>(src), count, offset, kind,
                     reinterpret_cast<CUstream>(stream), true, true);
}

extern "C" cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
                                                 size_t count, size_t offset,
                                                 enum cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  return copy_symbol(symbol, dst, count, offset, kind,
                     reinterpret_cast<CUstream>(stream), true, false);
}

// ---------------------------------------------------------------------------
// Host callbacks
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn,
                                          void *userData) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuLaunchHostFunc(reinterpret_cast<CUstream>(stream),
                                      reinterpret_cast<CUhostFn>(fn),
                                      userData));
}

namespace {

// The two callback signatures differ only in the spelling of their handle and
// result types, so the trampoline just re-types the arguments.
struct stream_callback {
  cudaStreamCallback_t callback;
  void *user_data;
};

void CUDA_CB stream_callback_trampoline(CUstream stream, CUresult status,
                                        void *data) {
  auto *forward = static_cast<stream_callback *>(data);
  forward->callback(reinterpret_cast<cudaStream_t>(stream), translate(status),
                    forward->user_data);
  delete forward;
}

} // namespace

extern "C" cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                             cudaStreamCallback_t callback,
                                             void *userData,
                                             unsigned int flags) {
  if (callback == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  auto *forward = new stream_callback{callback, userData};
  CUresult result =
      cuStreamAddCallback(reinterpret_cast<CUstream>(stream),
                          stream_callback_trampoline, forward, flags);
  if (result != CUDA_SUCCESS) {
    delete forward;
  }
  return from_driver(result);
}

// ---------------------------------------------------------------------------
// Entry points, libraries, and kernel metadata
// ---------------------------------------------------------------------------

#if CUDART_VERSION >= 12000
extern "C" cudaError_t
cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                        unsigned long long flags,
                        enum cudaDriverEntryPointQueryResult *driverStatus) {
  if (symbol == nullptr || funcPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  ensure_init();
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult result =
      cuGetProcAddress(symbol, funcPtr, CUDA_VERSION, flags, &status);
  if (driverStatus != nullptr) {
    *driverStatus = static_cast<enum cudaDriverEntryPointQueryResult>(status);
  }
  return from_driver(result);
}
#else
extern "C" cudaError_t cudaGetDriverEntryPoint(const char *symbol,
                                               void **funcPtr,
                                               unsigned long long flags) {
  if (symbol == nullptr || funcPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  ensure_init();
  return from_driver(cuGetProcAddress(symbol, funcPtr, CUDA_VERSION, flags));
}
#endif

#if CUDART_VERSION >= 13000
extern "C" cudaError_t cudaGetDriverEntryPointByVersion(
    const char *symbol, void **funcPtr, unsigned int cudaVersion,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus) {
  if (symbol == nullptr || funcPtr == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  ensure_init();
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult result = cuGetProcAddress(
      symbol, funcPtr, static_cast<int>(cudaVersion), flags, &status);
  if (driverStatus != nullptr) {
    *driverStatus = static_cast<enum cudaDriverEntryPointQueryResult>(status);
  }
  return from_driver(result);
}
#endif

extern "C" cudaError_t cudaGetExportTable(const void **ppExportTable,
                                          const cudaUUID_t *pExportTableId) {
  ensure_init();
  return from_driver(cuGetExportTable(
      ppExportTable, reinterpret_cast<const CUuuid *>(pExportTableId)));
}

extern "C" cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph,
                                              const char *path,
                                              unsigned int flags) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(
      cuGraphDebugDotPrint(reinterpret_cast<CUgraph>(graph), path, flags));
}

#if CUDART_VERSION >= 12000
extern "C" cudaError_t cudaFuncGetName(const char **name, const void *func) {
  if (name == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto entry = registered_functions().find(func);
  if (entry == registered_functions().end()) {
    return record(cudaErrorInvalidDeviceFunction);
  }
  *name = entry->second.name.c_str();
  return cudaSuccess;
}

extern "C" cudaError_t cudaFuncGetParamInfo(const void *func, size_t paramIndex,
                                            size_t *paramOffset,
                                            size_t *paramSize) {
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  return from_driver(
      cuFuncGetParamInfo(function, paramIndex, paramOffset, paramSize));
}
#endif

#if CUDART_VERSION >= 13000
extern "C" cudaError_t cudaFuncGetParamCount(const void *func,
                                             size_t *paramCount) {
  if (paramCount == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  CUfunction function = nullptr;
  cudaError_t resolved = resolve_function(func, &function);
  if (resolved != cudaSuccess) {
    return record(resolved);
  }
  // The driver has no count query, so the parameters are walked until it
  // reports there is no further index.
  size_t count = 0;
  for (;; ++count) {
    size_t offset = 0;
    size_t size = 0;
    if (cuFuncGetParamInfo(function, count, &offset, &size) != CUDA_SUCCESS) {
      break;
    }
  }
  *paramCount = count;
  return cudaSuccess;
}

extern "C" cudaError_t cudaGetKernel(cudaKernel_t *kernelPtr,
                                     const void *entryFuncAddr) {
  return __cudaGetKernel(kernelPtr, entryFuncAddr);
}

extern "C" cudaError_t cudaKernelSetAttributeForDevice(
    cudaKernel_t kernel, enum cudaFuncAttribute attr, int value, int device) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  // The attribute is set on the kernel as the named device sees it, so both
  // the resolution and the call run under that device's context.
  device_scope scope(device);
  if (scope.status() != cudaSuccess) {
    return record(scope.status());
  }
  CUfunction function = nullptr;
  CUresult resolved =
      function_for(reinterpret_cast<const void *>(kernel), &function);
  if (resolved != CUDA_SUCCESS) {
    return from_driver(resolved);
  }
  return from_driver(cuFuncSetAttribute(
      function, static_cast<CUfunction_attribute>(attr), value));
}

extern "C" cudaError_t cudaLibraryLoadData(
    cudaLibrary_t *library, const void *code, enum cudaJitOption *jitOptions,
    void **jitOptionsValues, unsigned int numJitOptions,
    enum cudaLibraryOption *libraryOptions, void **libraryOptionValues,
    unsigned int numLibraryOptions) {
  if (library == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuLibraryLoadData(
      reinterpret_cast<CUlibrary *>(library), code,
      reinterpret_cast<CUjit_option *>(jitOptions), jitOptionsValues,
      numJitOptions, reinterpret_cast<CUlibraryOption *>(libraryOptions),
      libraryOptionValues, numLibraryOptions));
}

extern "C" cudaError_t cudaLibraryLoadFromFile(
    cudaLibrary_t *library, const char *fileName,
    enum cudaJitOption *jitOptions, void **jitOptionsValues,
    unsigned int numJitOptions, enum cudaLibraryOption *libraryOptions,
    void **libraryOptionValues, unsigned int numLibraryOptions) {
  if (library == nullptr) {
    return record(cudaErrorInvalidValue);
  }
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuLibraryLoadFromFile(
      reinterpret_cast<CUlibrary *>(library), fileName,
      reinterpret_cast<CUjit_option *>(jitOptions), jitOptionsValues,
      numJitOptions, reinterpret_cast<CUlibraryOption *>(libraryOptions),
      libraryOptionValues, numLibraryOptions));
}

extern "C" cudaError_t cudaLibraryGetUnifiedFunction(void **fptr,
                                                     cudaLibrary_t library,
                                                     const char *symbol) {
  cudaError_t entered = enter();
  if (entered != cudaSuccess) {
    return record(entered);
  }
  return from_driver(cuLibraryGetUnifiedFunction(
      fptr, reinterpret_cast<CUlibrary>(library), symbol));
}
#endif
