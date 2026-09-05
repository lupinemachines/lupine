// Exercises the real generated client/server marshallers against a fake
// libcudart, and verifies that copy/completion entry points call libcuda.
// No GPU, network connection, or implicit driver initialization is needed.
#undef NDEBUG
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cuda_client_rpc.h"
#include "lupine_fatbin.h"

#include "codegen/gen_cudart_server.h"
#include "codegen/gen_rpc_ids.h"

int handle___cudaRegisterFatBinary(conn_t *);
extern "C" void **__cudaRegisterFatBinary(void *);
extern "C" void __cudaRegisterFatBinaryEnd(void **);
extern "C" void __cudaUnregisterFatBinary(void **);
extern "C" void __cudaRegisterFunction(void **, const char *, char *,
                                       const char *, int, uint3 *, uint3 *,
                                       dim3 *, dim3 *, int *);
extern "C" void __cudaRegisterVar(void **, char *, char *, const char *, int,
                                  size_t, int, int);
extern "C" void __cudaRegisterManagedVar(void **, void **, char *, const char *,
                                         int, size_t, int, int);
extern "C" void __cudaRegisterTexture(void **, const void *, const void **,
                                      const char *, int, int, int);
extern "C" void __cudaRegisterSurface(void **, const void *, const void **,
                                      const char *, int, int);
extern "C" void __cudaRegisterUnifiedTable(void *);
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t, void *);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *, dim3 *, size_t *,
                                                  void *);

static conn_t *const servers[] = {reinterpret_cast<conn_t *>(0x100),
                                  reinterpret_cast<conn_t *>(0x200)};
struct write_span {
  const void *data;
  size_t size;
};
static thread_local std::vector<write_span> writes;
static thread_local std::vector<unsigned char> request, response;
static thread_local size_t read_offset;
static thread_local int operation, server;
static thread_local bool fail_request;
static int registrations, functions, variables, managed, textures, surfaces;
static std::vector<const char *> retained_names;
static std::vector<const uint3 *> retained_indices;
static std::vector<void **> managed_slots;

struct allocation {
  size_t size;
  conn_t *conn;
};
static std::unordered_map<uintptr_t, allocation> allocations;
static std::string driver_call;
static CUresult driver_result = CUDA_SUCCESS;
static CUdeviceptr driver_dst, driver_src;
static size_t driver_count;
static CUstream driver_stream;
static CUDA_MEMCPY3D driver_3d;

extern "C" int lupine_rpc_device_count(int *count) {
  *count = 2;
  return 0;
}
extern "C" conn_t *lupine_rpc_conn_for_device(int *device) {
  assert(*device == 0 || *device == 1);
  conn_t *conn = servers[*device];
  *device = 0;
  return conn;
}
extern "C" conn_t *lupine_rpc_conn_for_stream(cudaStream_t) {
  return servers[1];
}
extern "C" conn_t *lupine_rpc_conn_for_deviceptr(CUdeviceptr ptr) {
  for (const auto &[base, allocation] : allocations) {
    if (ptr >= base && ptr - base < allocation.size)
      return allocation.conn;
  }
  assert(false && "device allocation was not tracked");
  return nullptr;
}
extern "C" void lupine_rpc_note_allocation(conn_t *conn, const void *ptr,
                                           size_t size) {
  allocations[reinterpret_cast<uintptr_t>(ptr)] = {size, conn};
}
extern "C" void lupine_rpc_forget_allocation(const void *ptr) {
  allocations.erase(reinterpret_cast<uintptr_t>(ptr));
}

static void append(std::vector<unsigned char> &bytes, const void *data,
                   size_t size) {
  if (size == 0)
    return;
  const auto *p = static_cast<const unsigned char *>(data);
  bytes.insert(bytes.end(), p, p + size);
}
extern "C" int lupine_rpc_write_start_request(conn_t *conn, int op) {
  if (fail_request)
    return -1;
  assert(conn == servers[0] || conn == servers[1]);
  server = conn == servers[1];
  operation = op;
  writes.clear();
  request.clear();
  response.clear();
  read_offset = 0;
  return 0;
}
extern "C" int lupine_rpc_write(conn_t *, const void *data, size_t size) {
  writes.push_back({data, size}); // Intentionally defer reading caller storage.
  return 0;
}
int rpc_read(conn_t *, void *data, size_t size) {
  assert(read_offset + size <= request.size());
  if (size)
    std::memcpy(data, request.data() + read_offset, size);
  read_offset += size;
  return 0;
}
int rpc_read_end(conn_t *) {
  assert(read_offset == request.size());
  return 1;
}
int rpc_write_start_response(conn_t *, int) { return 0; }
int rpc_write(conn_t *, const void *data, size_t size) {
  append(response, data, size);
  return 0;
}
int rpc_write_end(conn_t *) { return 1; }
extern "C" int lupine_rpc_wait_for_response(conn_t *conn) {
  for (const auto &span : writes)
    append(request, span.data, span.size);
  int result = -1;
#define DISPATCH(name)                                                         \
  case RPC_##name:                                                             \
    result = handle_##name(conn);                                              \
    break
  switch (operation) {
    DISPATCH(__cudaRegisterFatBinary);
    DISPATCH(__cudaRegisterFatBinaryEnd);
    DISPATCH(__cudaUnregisterFatBinary);
    DISPATCH(__cudaRegisterFunction);
    DISPATCH(__cudaRegisterVar);
    DISPATCH(__cudaRegisterManagedVar);
    DISPATCH(__cudaRegisterTexture);
    DISPATCH(__cudaRegisterSurface);
    DISPATCH(__cudaPushCallConfiguration);
    DISPATCH(__cudaPopCallConfiguration);
    DISPATCH(cudaSetDevice);
    DISPATCH(cudaMalloc);
    DISPATCH(cudaMallocAsync);
    DISPATCH(cudaMallocPitch);
    DISPATCH(cudaFree);
    DISPATCH(cudaGetSymbolAddress);
    DISPATCH(cudaGetSymbolSize);
  default:
    assert(false && "unexpected runtime RPC");
  }
#undef DISPATCH
  assert(result == 0);
  read_offset = 0;
  return 0;
}
extern "C" int lupine_rpc_read(conn_t *, void *data, size_t size) {
  assert(read_offset + size <= response.size());
  if (size)
    std::memcpy(data, response.data() + read_offset, size);
  read_offset += size;
  return 0;
}
extern "C" int lupine_rpc_read_end(conn_t *) {
  assert(read_offset == response.size());
  return 0;
}

static void **vendor_fatbin(void *image) {
  auto *wrapper = static_cast<lupine_fatbin_wrapper *>(image);
  assert(wrapper->magic == LUPINE_FATBINC_MAGIC);
  ++registrations;
  return reinterpret_cast<void **>(0x1000 + server * 0x100);
}
static void vendor_end(void **) {
  // These writes happen after the registration RPC's temporary buffers died.
  for (void **slot : managed_slots)
    *slot = reinterpret_cast<void *>(0xdead);
}
static void vendor_unregister(void **) {}
static void vendor_function(void **, const char *host, char *device,
                            const char *name, int limit, uint3 *tid, uint3 *bid,
                            dim3 *bd, dim3 *gd, int *ws) {
  assert(host == reinterpret_cast<const char *>(0xabc));
  assert(std::strcmp(device, "kernel") == 0 &&
         std::strcmp(name, "kernel") == 0);
  assert(limit == -1);
  if (tid) {
    assert(tid->x == 7 && bid->y == 8 && bd->z == 9 && gd->x == 10 &&
           *ws == 32);
    retained_indices.push_back(tid);
  } else {
    assert(!bid && !bd && !gd && !ws);
  }
  retained_names.push_back(name);
  ++functions;
}
static void vendor_var(void **, char *host, char *device, const char *name, int,
                       size_t size, int, int) {
  assert(host == reinterpret_cast<char *>(0xdef) && size == 32);
  assert(std::strcmp(device, "variable") == 0);
  retained_names.push_back(name);
  ++variables;
}
static void vendor_managed(void **, void **slot, char *, const char *name, int,
                           size_t, int, int) {
  assert(slot != reinterpret_cast<void **>(0xbad));
  managed_slots.push_back(slot);
  retained_names.push_back(name);
  ++managed;
}
static void vendor_texture(void **, const void *, const void **device,
                           const char *name, int, int, int) {
  assert(std::strcmp(reinterpret_cast<const char *>(device), "texture") == 0);
  retained_names.push_back(name);
  ++textures;
}
static void vendor_surface(void **, const void *, const void **device,
                           const char *name, int, int) {
  assert(std::strcmp(reinterpret_cast<const char *>(device), "surface") == 0);
  retained_names.push_back(name);
  ++surfaces;
}
struct configuration {
  dim3 grid, block;
  size_t shared;
  void *stream;
};
static thread_local std::vector<configuration> configurations;
static unsigned vendor_push(dim3 grid, dim3 block, size_t shared,
                            void *stream) {
  assert(server == 0);
  configurations.push_back({grid, block, shared, stream});
  return 0;
}
static cudaError_t vendor_pop(dim3 *grid, dim3 *block, size_t *shared,
                              void *stream) {
  assert(server == 0);
  if (configurations.empty())
    return cudaErrorInvalidConfiguration;
  auto config = configurations.back();
  configurations.pop_back();
  if (grid)
    *grid = config.grid;
  if (block)
    *block = config.block;
  if (shared)
    *shared = config.shared;
  if (stream)
    *static_cast<void **>(stream) = config.stream;
  return cudaSuccess;
}
static cudaError_t vendor_set_device(int device) {
  assert(device == 0);
  return cudaSuccess;
}
static cudaError_t vendor_malloc(void **ptr, size_t) {
  *ptr = reinterpret_cast<void *>(0x1000000 + server * 0x1000000);
  return cudaSuccess;
}
static cudaError_t vendor_malloc_async(void **ptr, size_t size, cudaStream_t) {
  return vendor_malloc(ptr, size);
}
static cudaError_t vendor_malloc_pitch(void **ptr, size_t *pitch, size_t width,
                                       size_t) {
  *pitch = width + 64;
  return vendor_malloc(ptr, 0);
}
static cudaError_t vendor_free(void *) { return cudaSuccess; }
static cudaError_t vendor_symbol(void **ptr, const void *) {
  *ptr = reinterpret_cast<void *>(0x3000000 + server * 0x1000000);
  return cudaSuccess;
}
static cudaError_t vendor_symbol_size(size_t *size, const void *) {
  *size = 32;
  return cudaSuccess;
}
extern "C" void *__wrap_dlopen(const char *, int) {
  return reinterpret_cast<void *>(1);
}
extern "C" void *__wrap_dlsym(void *, const char *name) {
#define SYMBOL(api, fn)                                                        \
  if (std::strcmp(name, #api) == 0)                                            \
  return reinterpret_cast<void *>(&fn)
  SYMBOL(__cudaRegisterFatBinary, vendor_fatbin);
  SYMBOL(__cudaRegisterFatBinaryEnd, vendor_end);
  SYMBOL(__cudaUnregisterFatBinary, vendor_unregister);
  SYMBOL(__cudaRegisterFunction, vendor_function);
  SYMBOL(__cudaRegisterVar, vendor_var);
  SYMBOL(__cudaRegisterManagedVar, vendor_managed);
  SYMBOL(__cudaRegisterTexture, vendor_texture);
  SYMBOL(__cudaRegisterSurface, vendor_surface);
  SYMBOL(__cudaPushCallConfiguration, vendor_push);
  SYMBOL(__cudaPopCallConfiguration, vendor_pop);
  SYMBOL(cudaSetDevice, vendor_set_device);
  SYMBOL(cudaMalloc, vendor_malloc);
  SYMBOL(cudaMallocAsync, vendor_malloc_async);
  SYMBOL(cudaMallocPitch, vendor_malloc_pitch);
  SYMBOL(cudaFree, vendor_free);
  SYMBOL(cudaGetSymbolAddress, vendor_symbol);
  SYMBOL(cudaGetSymbolSize, vendor_symbol_size);
#undef SYMBOL
  assert(false && "unexpected vendor symbol");
  return nullptr;
}

extern "C" CUresult cuInit(unsigned) {
  assert(false && "implicit initialization");
  return CUDA_ERROR_NOT_INITIALIZED;
}
extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice device) {
  *ctx = reinterpret_cast<CUcontext>(0x8000 + device);
  return CUDA_SUCCESS;
}
extern "C" CUresult cuDevicePrimaryCtxRelease(CUdevice) { return CUDA_SUCCESS; }
extern "C" CUresult cuCtxSetCurrent(CUcontext) { return CUDA_SUCCESS; }
static CUresult copy(const char *name, CUdeviceptr dst, CUdeviceptr src,
                     size_t count, CUstream stream) {
  driver_call = name;
  driver_dst = dst;
  driver_src = src;
  driver_count = count;
  driver_stream = stream;
  return driver_result;
}
extern "C" CUresult cuMemcpy(CUdeviceptr d, CUdeviceptr s, size_t n) {
  return copy("default", d, s, n, nullptr);
}
extern "C" CUresult cuMemcpyAsync(CUdeviceptr d, CUdeviceptr s, size_t n,
                                  CUstream st) {
  return copy("default_async", d, s, n, st);
}
extern "C" CUresult cuMemcpyHtoD(CUdeviceptr d, const void *s, size_t n) {
  return copy("htod", d, reinterpret_cast<CUdeviceptr>(s), n, nullptr);
}
extern "C" CUresult cuMemcpyDtoH(void *d, CUdeviceptr s, size_t n) {
  return copy("dtoh", reinterpret_cast<CUdeviceptr>(d), s, n, nullptr);
}
extern "C" CUresult cuMemcpyDtoD(CUdeviceptr d, CUdeviceptr s, size_t n) {
  return copy("dtod", d, s, n, nullptr);
}
extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr d, const void *s, size_t n,
                                      CUstream st) {
  return copy("htod_async", d, reinterpret_cast<CUdeviceptr>(s), n, st);
}
extern "C" CUresult cuMemcpyDtoHAsync(void *d, CUdeviceptr s, size_t n,
                                      CUstream st) {
  return copy("dtoh_async", reinterpret_cast<CUdeviceptr>(d), s, n, st);
}
extern "C" CUresult cuMemcpyDtoDAsync(CUdeviceptr d, CUdeviceptr s, size_t n,
                                      CUstream st) {
  return copy("dtod_async", d, s, n, st);
}
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D *p) {
  driver_3d = *p;
  driver_call = "3d";
  return driver_result;
}
extern "C" CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *p, CUstream st) {
  driver_3d = *p;
  driver_stream = st;
  driver_call = "3d_async";
  return driver_result;
}
extern "C" CUresult cuMemcpyPeer(CUdeviceptr d, CUcontext dc, CUdeviceptr s,
                                 CUcontext sc, size_t n) {
  assert(dc == reinterpret_cast<CUcontext>(0x8001) &&
         sc == reinterpret_cast<CUcontext>(0x8000));
  return copy("peer", d, s, n, nullptr);
}
extern "C" CUresult cuMemcpyPeerAsync(CUdeviceptr d, CUcontext dc,
                                      CUdeviceptr s, CUcontext sc, size_t n,
                                      CUstream st) {
  cuMemcpyPeer(d, dc, s, sc, n);
  driver_call = "peer_async";
  driver_stream = st;
  return driver_result;
}
extern "C" CUresult cuCtxSynchronize() {
  driver_call = "device_sync";
  return driver_result;
}
extern "C" CUresult cuStreamSynchronize(CUstream) {
  driver_call = "stream_sync";
  return driver_result;
}
extern "C" CUresult cuStreamQuery(CUstream) {
  driver_call = "stream_query";
  return driver_result;
}
extern "C" CUresult cuEventSynchronize(CUevent) {
  driver_call = "event_sync";
  return driver_result;
}
extern "C" CUresult cuEventQuery(CUevent) {
  driver_call = "event_query";
  return driver_result;
}
extern "C" CUresult cuEventRecord(CUevent, CUstream) {
  driver_call = "event_record";
  return driver_result;
}
extern "C" CUresult cuEventRecordWithFlags(CUevent, CUstream, unsigned) {
  driver_call = "event_record_flags";
  return driver_result;
}
extern "C" CUresult cuEventDestroy(CUevent) {
  driver_call = "event_destroy";
  return driver_result;
}

int main() {
  lupine_fatbin_header image = {};
  image.magic = LUPINE_FATBIN_MAGIC;
  image.header_size = sizeof(image);
  void **handle = __cudaRegisterFatBinary(&image);
  assert(handle && registrations == 2);
  uint3 tid{7, 0, 0}, bid{0, 8, 0};
  dim3 block(1, 1, 9), grid(10, 1, 1);
  int warp = 32;
  char kernel[] = "kernel", variable[] = "variable", texture[] = "texture",
       surface[] = "surface";
  __cudaRegisterFunction(handle, reinterpret_cast<const char *>(0xabc), kernel,
                         kernel, -1, &tid, &bid, &block, &grid, &warp);
  __cudaRegisterFunction(handle, reinterpret_cast<const char *>(0xabc), kernel,
                         kernel, -1, nullptr, nullptr, nullptr, nullptr,
                         nullptr);
  __cudaRegisterVar(handle, reinterpret_cast<char *>(0xdef), variable, variable,
                    0, 32, 0, 1);
  __cudaRegisterManagedVar(handle, reinterpret_cast<void **>(0xbad), variable,
                           variable, 0, 32, 0, 1);
  __cudaRegisterTexture(handle, nullptr,
                        reinterpret_cast<const void **>(texture), texture, 2, 0,
                        0);
  __cudaRegisterSurface(
      handle, nullptr, reinterpret_cast<const void **>(surface), surface, 2, 0);
  std::memset(kernel, '!', 6);
  std::memset(variable, '!', 8);
  assert(functions == 4 && variables == 2 && managed == 2 && textures == 2 &&
         surfaces == 2);
  for (const char *name : retained_names)
    assert(std::strchr(name, '!') == nullptr);
  tid.x = 99;
  for (const uint3 *index : retained_indices)
    assert(index->x == 7);
  __cudaRegisterFatBinaryEnd(handle);
  for (void **slot : managed_slots)
    assert(*slot == reinterpret_cast<void *>(0xdead));
  __cudaUnregisterFatBinary(handle);
  assert(cudaPeekAtLastError() == cudaSuccess);

  auto stream = reinterpret_cast<cudaStream_t>(0x123456789abcdef0ULL);
  assert(__cudaPushCallConfiguration(dim3(2, 3, 4), dim3(5, 6, 7), 8192,
                                     stream) == 0);
  assert(cudaSetDevice(1) == cudaSuccess); // Pop must not change servers.
  assert(__cudaPushCallConfiguration(dim3(9), dim3(8), 1024, nullptr) == 0);
  size_t shared = 0;
  void *out_stream = stream;
  assert(__cudaPopCallConfiguration(&grid, &block, &shared, &out_stream) ==
         cudaSuccess);
  assert(grid.x == 9 && block.x == 8 && shared == 1024 &&
         out_stream == nullptr);
  assert(__cudaPopCallConfiguration(&grid, &block, &shared, &out_stream) ==
         cudaSuccess);
  assert(grid.z == 4 && block.z == 7 && shared == 8192 && out_stream == stream);
  assert(__cudaPopCallConfiguration(nullptr, nullptr, nullptr, nullptr) ==
         cudaErrorInvalidConfiguration);
  assert(cudaGetLastError() == cudaErrorInvalidConfiguration);
  fail_request = true;
  assert(__cudaPushCallConfiguration(grid, block, 0, nullptr) ==
         cudaErrorDevicesUnavailable);
  fail_request = false;
  assert(cudaGetLastError() == cudaErrorDevicesUnavailable);
  std::thread([] {
    assert(__cudaPushCallConfiguration(dim3(42), dim3(1), 0, nullptr) == 0);
    dim3 g;
    assert(__cudaPopCallConfiguration(&g, nullptr, nullptr, nullptr) ==
               cudaSuccess &&
           g.x == 42);
  }).join();

  void *device = nullptr;
  assert(cudaMalloc(&device, 256) == cudaSuccess);
  assert(lupine_rpc_conn_for_deviceptr(reinterpret_cast<CUdeviceptr>(device) +
                                       128) == servers[1]);
  assert(cudaSetDevice(0) == cudaSuccess);
  assert(lupine_rpc_conn_for_deviceptr(reinterpret_cast<CUdeviceptr>(device) +
                                       128) == servers[1]);
  assert(cudaFree(device) == cudaSuccess && allocations.empty());
  assert(cudaMallocAsync(&device, 64, stream) == cudaSuccess);
  assert(lupine_rpc_conn_for_deviceptr(reinterpret_cast<CUdeviceptr>(device)) ==
         servers[1]);
  assert(cudaFree(device) == cudaSuccess);
  size_t pitch;
  assert(cudaMallocPitch(&device, &pitch, 32, 3) == cudaSuccess);
  assert(allocations.at(reinterpret_cast<uintptr_t>(device)).size == pitch * 3);

  char src[32] = "copy bytes", dst[32] = {};
  assert(cudaMemcpy(dst, src, sizeof(src), cudaMemcpyHostToHost) ==
         cudaSuccess);
  assert(std::memcmp(dst, src, sizeof(src)) == 0);
  const cudaMemcpyKind kinds[] = {cudaMemcpyHostToDevice,
                                  cudaMemcpyDeviceToHost,
                                  cudaMemcpyDeviceToDevice, cudaMemcpyDefault};
  const char *names[] = {"htod", "dtoh", "dtod", "default"};
  for (int i = 0; i < 4; ++i) {
    assert(cudaMemcpy(dst, src, 17, kinds[i]) == cudaSuccess);
    assert(driver_call == names[i] &&
           driver_dst == reinterpret_cast<CUdeviceptr>(dst) &&
           driver_src == reinterpret_cast<CUdeviceptr>(src) &&
           driver_count == 17);
    assert(cudaMemcpyAsync(dst, src, 17, kinds[i], stream) == cudaSuccess);
    assert(driver_call == std::string(names[i]) + "_async" &&
           driver_stream == stream);
    assert(cudaMemcpy2DAsync(dst, 16, src, 24, 8, 2, kinds[i], stream) ==
           cudaSuccess);
    assert(driver_call == "3d_async" && driver_3d.srcPitch == 24 &&
           driver_3d.dstPitch == 16 && driver_3d.WidthInBytes == 8 &&
           driver_3d.Height == 2 && driver_3d.Depth == 1 && driver_stream == stream);
  }
  assert(driver_3d.srcMemoryType == CU_MEMORYTYPE_UNIFIED &&
         driver_3d.dstMemoryType == CU_MEMORYTYPE_UNIFIED);
  assert(cudaMemcpy2D(dst, 16, src, 24, 8, 2, cudaMemcpyHostToDevice) ==
         cudaSuccess);
  assert(driver_call == "3d" && driver_3d.srcMemoryType == CU_MEMORYTYPE_HOST &&
         driver_3d.dstMemoryType == CU_MEMORYTYPE_DEVICE);
  assert(cudaMemcpy(nullptr, nullptr, 0, cudaMemcpyDefault) == cudaSuccess);
  assert(cudaMemcpy(dst, src, 1, static_cast<cudaMemcpyKind>(99)) ==
         cudaErrorInvalidMemcpyDirection);
  assert(cudaMemcpy2D(dst, 4, src, 4, 8, 2, cudaMemcpyHostToHost) ==
         cudaErrorInvalidPitchValue);
  assert(cudaMemcpyToSymbol(variable, src, 4, 8, cudaMemcpyHostToDevice) ==
         cudaSuccess);
  assert(driver_call == "htod" && driver_dst == 0x3000008);
  assert(cudaMemcpyFromSymbolAsync(dst, variable, 4, 8, cudaMemcpyDeviceToHost,
                                   stream) == cudaSuccess);
  assert(driver_call == "dtoh_async" && driver_src == 0x4000008 &&
         driver_stream == stream);
  assert(cudaMemcpyToSymbol(variable, src, 4, 30, cudaMemcpyHostToDevice) ==
         cudaErrorInvalidValue);
  assert(cudaMemcpyPeer(dst, 1, src, 0, 4) == cudaSuccess &&
         driver_call == "peer");
  assert(cudaMemcpyPeerAsync(dst, 1, src, 0, 4, stream) == cudaSuccess &&
         driver_call == "peer_async");

  auto event = reinterpret_cast<cudaEvent_t>(0x123);
  assert(cudaEventRecord(event, stream) == cudaSuccess &&
         driver_call == "event_record");
  assert(cudaEventRecordWithFlags(event, stream, 0) == cudaSuccess &&
         driver_call == "event_record_flags");
  assert(cudaEventSynchronize(event) == cudaSuccess &&
         driver_call == "event_sync");
  assert(cudaEventQuery(event) == cudaSuccess && driver_call == "event_query");
  assert(cudaEventDestroy(event) == cudaSuccess &&
         driver_call == "event_destroy");
  assert(cudaStreamSynchronize(stream) == cudaSuccess &&
         driver_call == "stream_sync");
  assert(cudaStreamQuery(stream) == cudaSuccess &&
         driver_call == "stream_query");
  assert(cudaDeviceSynchronize() == cudaSuccess &&
         driver_call == "device_sync");
  driver_result = CUDA_ERROR_DEINITIALIZED;
  assert(cudaMemcpy(dst, src, 1, cudaMemcpyDefault) ==
         cudaErrorDeviceUninitialized);
  assert(cudaPeekAtLastError() == cudaErrorDeviceUninitialized);
  __cudaRegisterUnifiedTable(nullptr);
  assert(cudaGetLastError() == cudaErrorNotSupported);
}
