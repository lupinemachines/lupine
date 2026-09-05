// Exercises the real generated client/server marshallers against a fake
// libcudart, and verifies that copy/completion entry points call libcuda.
// No GPU, network connection, or implicit driver initialization is needed.
#undef NDEBUG
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cuda_client_rpc.h"
#include "lupine_fatbin.h"
#include "rpc.h"

#include "codegen/gen_cudart_server.h"
#include "codegen/gen_rpc_ids.h"

// Wrap only transport I/O, leaving connection setup and sequencing in rpc.cpp.
extern "C" int __wrap__Z8rpc_readP6conn_tPvm(conn_t *, void *, size_t);
extern "C" int __wrap__Z12rpc_read_endP6conn_t(conn_t *);
extern "C" int __wrap__Z24rpc_write_start_responseP6conn_ti(conn_t *, int);
extern "C" int __wrap__Z9rpc_writeP6conn_tPKvm(conn_t *, const void *, size_t);
extern "C" int __wrap__Z13rpc_write_endP6conn_t(conn_t *);

int handle___cudaRegisterFatBinary(conn_t *);
int handle_lupineCudartFuncParamLayout(conn_t *);
int handle_cudaLaunchKernel(conn_t *);
int handle_cudaLaunchCooperativeKernel(conn_t *);
int handle_cudaLaunchKernelExC(conn_t *);
int handle___cudaLaunchKernel(conn_t *);
int handle_cudaOccupancyMaxPotentialClusterSize(conn_t *);
int handle_cudaOccupancyMaxActiveClusters(conn_t *);
extern "C" void **__cudaRegisterFatBinary(void *);
extern "C" char __cudaInitModule(void **);
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t, dim3, dim3, void **,
                                          size_t, cudaStream_t);
extern "C" cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t, dim3, dim3,
                                               void **, size_t, cudaStream_t);
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
extern "C" void __cudaRegisterHostVar(void **, const char *, char *, size_t);
extern "C" void __cudaRegisterUnifiedTable(void **, void *, size_t, void *, size_t);
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t, void *);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *, dim3 *, size_t *,
                                                  void *);

static conn_t server_state[2];
static conn_t *const servers[] = {&server_state[0], &server_state[1]};
struct write_span {
  const void *data;
  size_t size;
};
static thread_local std::vector<write_span> writes;
static thread_local std::vector<unsigned char> request, response;
static thread_local size_t read_offset;
static thread_local int operation, server;
static thread_local bool fail_request;
static thread_local bool async_request;
static int response_waits, launches;
struct queued_launch {
  conn_t *conn;
  int operation;
  std::vector<unsigned char> bytes;
};
static std::vector<queued_launch> queued_launches;
static thread_local int failing_server = -1;
static int registrations, functions, variables, managed, textures, surfaces,
    host_variables;
static int ended[2], unregistered[2];
static int unified_tables[2];
static void *unified_functions, *unified_data;
static size_t unified_function_size, unified_data_size;
static int module_inits[2];
static char module_result = static_cast<char>(0xa5);
static bool missing_init_module;
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
static CUevent driver_event;
static unsigned driver_flags;
static CUDA_MEMCPY3D driver_3d;
static cudaError_t attribute_result = cudaSuccess;
static uint64_t attribute_value = 0xfedcba9876543210ULL;
static size_t attribute_request_size;
static int attribute_server;
static int expected_launch_server;
static const void *expected_entry;
static const void *zero_arg_entry = reinterpret_cast<const void *>(0x777);
static int expected_attributes;
static std::string launch_call;

extern "C" int lupine_rpc_device_count(int *count) {
  *count = 3; // Devices 0 and 2 share a server and one fatbin registration.
  return 0;
}
extern "C" conn_t *lupine_rpc_conn_for_device(int *device) {
  assert(*device >= 0 && *device < 3);
  conn_t *conn = servers[*device == 1];
  *device = *device == 2 ? 1 : 0;
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
  assert(conn == servers[0] || conn == servers[1]);
  server = conn == servers[1];
  if (fail_request || server == failing_server)
    return -1;
  async_request = false;
  operation = op;
  writes.clear();
  request.clear();
  response.clear();
  read_offset = 0;
  return 0;
}
// The transport is mocked; ticket consumption uses the real RPC core.
extern "C" int lupine_rpc_write_start_async_request(conn_t *conn, int op,
                                                    uint64_t *sequence) {
  if (lupine_rpc_write_start_request(conn, op) < 0)
    return -1;
  async_request = true;
  *sequence = conn->issued_async_sequence++;
  return 0;
}
extern "C" int lupine_rpc_write(conn_t *, const void *data, size_t size) {
  writes.push_back({data, size}); // Intentionally defer reading caller storage.
  return 0;
}
extern "C" int __wrap__Z8rpc_readP6conn_tPvm(conn_t *, void *data,
                                             size_t size) {
  assert(read_offset + size <= request.size());
  if (size)
    std::memcpy(data, request.data() + read_offset, size);
  read_offset += size;
  return 0;
}
extern "C" int __wrap__Z12rpc_read_endP6conn_t(conn_t *) {
  assert(read_offset == request.size());
  return 1;
}
extern "C" int __wrap__Z24rpc_write_start_responseP6conn_ti(conn_t *, int) {
  return 0;
}
extern "C" int __wrap__Z9rpc_writeP6conn_tPKvm(conn_t *, const void *data,
                                               size_t size) {
  append(response, data, size);
  return 0;
}
static void dispatch_request(conn_t *conn) {
  int result = -1;
#define DISPATCH(name)                                                         \
  case RPC_##name:                                                             \
    result = handle_##name(conn);                                              \
    break
  switch (operation) {
    DISPATCH(__cudaRegisterFatBinary);
    DISPATCH(__cudaInitModule);
    DISPATCH(__cudaRegisterFatBinaryEnd);
    DISPATCH(__cudaUnregisterFatBinary);
    DISPATCH(__cudaRegisterFunction);
    DISPATCH(__cudaRegisterVar);
    DISPATCH(__cudaRegisterManagedVar);
    DISPATCH(__cudaRegisterTexture);
    DISPATCH(__cudaRegisterSurface);
    DISPATCH(__cudaRegisterHostVar);
    DISPATCH(__cudaRegisterUnifiedTable);
    DISPATCH(__cudaPushCallConfiguration);
    DISPATCH(__cudaPopCallConfiguration);
    DISPATCH(cudaSetDevice);
    DISPATCH(cudaMalloc);
    DISPATCH(cudaMallocAsync);
    DISPATCH(cudaMallocPitch);
    DISPATCH(cudaFree);
    DISPATCH(cudaGetSymbolAddress);
    DISPATCH(cudaGetSymbolSize);
    DISPATCH(cudaMemPoolGetAttribute);
    DISPATCH(cudaMemPoolSetAttribute);
    DISPATCH(cudaDeviceGetGraphMemAttribute);
    DISPATCH(cudaDeviceSetGraphMemAttribute);
    DISPATCH(lupineCudartFuncParamLayout);
    DISPATCH(cudaLaunchKernel);
    DISPATCH(cudaLaunchCooperativeKernel);
    DISPATCH(cudaLaunchKernelExC);
    DISPATCH(__cudaLaunchKernel);
    DISPATCH(cudaOccupancyMaxPotentialClusterSize);
    DISPATCH(cudaOccupancyMaxActiveClusters);
  default:
    assert(false && "unexpected runtime RPC");
  }
#undef DISPATCH
  assert(result == 0);
}
extern "C" int lupine_rpc_wait_for_response(conn_t *conn) {
  assert(!async_request && "kernel launch waited for the server");
  ++response_waits;
  for (const auto &span : writes)
    append(request, span.data, span.size);
  dispatch_request(conn);
  read_offset = 0;
  return 0;
}
extern "C" int __wrap__Z13rpc_write_endP6conn_t(conn_t *conn) {
  if (!async_request)
    return 1;
  for (const auto &span : writes)
    append(request, span.data, span.size);
  queued_launches.push_back({conn, operation, std::move(request)});
  return 0;
}
extern "C" int lupine_rpc_write_end(conn_t *conn) {
  return __wrap__Z13rpc_write_endP6conn_t(conn);
}
static void drain_launches() {
  for (auto &queued : queued_launches) {
    operation = queued.operation;
    server = queued.conn == servers[1];
    request = std::move(queued.bytes);
    response.clear();
    read_offset = 0;
    uint64_t sequence;
    std::memcpy(&sequence, request.data(), sizeof(sequence));
    assert(sequence == queued.conn->serving_async_sequence);
    dispatch_request(queued.conn);
    assert(response.empty() &&
           queued.conn->serving_async_sequence == sequence + 1);
  }
  queued_launches.clear();
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
static void vendor_end(void **handle) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  ++ended[server];
  // These writes happen after the registration RPC's temporary buffers died.
  for (void **slot : managed_slots)
    *slot = reinterpret_cast<void *>(0xdead);
}
static void vendor_unified_table(void **handle, void *function_table,
                                 size_t function_size, void *data_table,
                                 size_t data_size) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  assert(function_table == unified_functions &&
         function_size == unified_function_size);
  assert(data_table == unified_data && data_size == unified_data_size);
  ++unified_tables[server];
}
static void vendor_unregister(void **handle) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  ++unregistered[server];
}
static char vendor_init_module(void **handle) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  ++module_inits[server];
  return module_result;
}
static void vendor_function(void **handle, const char *host, char *device,
                            const char *name, int limit, uint3 *tid, uint3 *bid,
                            dim3 *bd, dim3 *gd, int *ws) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
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
static void vendor_var(void **handle, char *host, char *device,
                       const char *name, int, size_t size, int, int) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  assert(host == reinterpret_cast<char *>(0xdef) && size == 32);
  assert(std::strcmp(device, "variable") == 0);
  retained_names.push_back(name);
  ++variables;
}
static void vendor_managed(void **handle, void **slot, char *, const char *name,
                           int, size_t, int, int) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  assert(slot != reinterpret_cast<void **>(0xbad));
  managed_slots.push_back(slot);
  retained_names.push_back(name);
  ++managed;
}
static void vendor_texture(void **handle, const void *, const void **device,
                           const char *name, int, int, int) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  assert(std::strcmp(reinterpret_cast<const char *>(device), "texture") == 0);
  retained_names.push_back(name);
  ++textures;
}
static void vendor_surface(void **handle, const void *, const void **device,
                           const char *name, int, int) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  assert(std::strcmp(reinterpret_cast<const char *>(device), "surface") == 0);
  retained_names.push_back(name);
  ++surfaces;
}
static void vendor_host_var(void **handle, const char *name, char *host,
                            size_t size) {
  assert(handle == reinterpret_cast<void **>(0x1000 + server * 0x100));
  assert(host == reinterpret_cast<char *>(0xdef) && size == 32);
  retained_names.push_back(name);
  ++host_variables;
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
static cudaError_t vendor_pool_get(cudaMemPool_t pool, cudaMemPoolAttr attr,
                                   void *value) {
  assert(pool == reinterpret_cast<cudaMemPool_t>(0x555));
  attribute_request_size = request.size();
  attribute_server = server;
  if (attribute_result != cudaSuccess)
    return attribute_result;
  if (attr == cudaMemPoolReuseFollowEventDependencies ||
      attr == cudaMemPoolReuseAllowOpportunistic ||
      attr == cudaMemPoolReuseAllowInternalDependencies
#if CUDART_VERSION >= 13020
      || attr == cudaMemPoolAttrAllocationType ||
      attr == cudaMemPoolAttrExportHandleTypes ||
      attr == cudaMemPoolAttrLocationId ||
      attr == cudaMemPoolAttrLocationType ||
      attr == cudaMemPoolAttrHwDecompressEnabled
#endif
  ) {
    int enabled = 1;
    std::memcpy(value, &enabled, sizeof(enabled));
  } else {
    std::memcpy(value, &attribute_value, sizeof(attribute_value));
  }
  return cudaSuccess;
}
static cudaError_t vendor_pool_set(cudaMemPool_t pool, cudaMemPoolAttr attr,
                                   void *value) {
  assert(pool == reinterpret_cast<cudaMemPool_t>(0x555));
  attribute_request_size = request.size();
  attribute_server = server;
  uint64_t expected = attribute_value;
  if (attr == cudaMemPoolReuseFollowEventDependencies ||
      attr == cudaMemPoolReuseAllowOpportunistic ||
      attr == cudaMemPoolReuseAllowInternalDependencies
#if CUDART_VERSION >= 13020
      || attr == cudaMemPoolAttrAllocationType ||
      attr == cudaMemPoolAttrExportHandleTypes ||
      attr == cudaMemPoolAttrLocationId ||
      attr == cudaMemPoolAttrLocationType ||
      attr == cudaMemPoolAttrHwDecompressEnabled
#endif
  ) {
    expected = 0;
    int enabled = 1;
    std::memcpy(&expected, &enabled, sizeof(enabled));
  }
  assert(std::memcmp(value, &expected, sizeof(expected)) == 0);
  return attribute_result;
}
static cudaError_t vendor_graph_get(int device, cudaGraphMemAttributeType,
                                    void *value) {
  assert(device == 0);
  attribute_server = server;
  if (attribute_result == cudaSuccess)
    std::memcpy(value, &attribute_value, sizeof(attribute_value));
  return attribute_result;
}
static cudaError_t vendor_graph_set(int device, cudaGraphMemAttributeType,
                                    void *value) {
  assert(device == 0);
  attribute_server = server;
  assert(std::memcmp(value, &attribute_value, sizeof(attribute_value)) == 0);
  return attribute_result;
}
static cudaError_t vendor_param_info(const void *func, size_t index,
                                     size_t *offset, size_t *size) {
  assert(func == expected_entry);
  if (func == zero_arg_entry || index >= 2)
    return cudaErrorInvalidValue;
  *offset = index == 0 ? 0 : 8;
  *size = index == 0 ? sizeof(int) : sizeof(uint64_t);
  return cudaSuccess;
}
static cudaError_t vendor_launch(const void *func, dim3 grid, dim3 block,
                                 void **args, size_t shared,
                                 cudaStream_t stream) {
  assert(func == expected_entry && server == expected_launch_server);
  assert(grid.x == 2 && grid.y == 3 && grid.z == 4);
  assert(block.x == 5 && block.y == 6 && block.z == 7 && shared == 8192);
  assert(stream == nullptr ||
         stream == reinterpret_cast<cudaStream_t>(0x123456789abcdef0ULL));
  assert(pthread_mutex_trylock(&servers[server]->async_mutex) == EBUSY);
  if (func != zero_arg_entry) {
    int first;
    uint64_t second;
    std::memcpy(&first, args[0], sizeof(first));
    std::memcpy(&second, args[1], sizeof(second));
    assert(first == 42 && second == 0x123456789abcdef0ULL);
  }
  ++launches;
  launch_call = "kernel";
  return cudaSuccess;
}
static cudaError_t vendor_cooperative(const void *func, dim3 grid, dim3 block,
                                      void **args, size_t shared,
                                      cudaStream_t stream) {
  vendor_launch(func, grid, block, args, shared, stream);
  launch_call = "cooperative";
  return cudaSuccess;
}
static cudaError_t vendor_private_launch(cudaKernel_t kernel, dim3 grid,
                                         dim3 block, void **args, size_t shared,
                                         cudaStream_t stream) {
  vendor_launch(kernel, grid, block, args, shared, stream);
  launch_call = "private";
  return cudaSuccess;
}
static void check_config(const cudaLaunchConfig_t *config) {
  assert(config->gridDim.z == 4 && config->blockDim.y == 6);
  assert(config->dynamicSmemBytes == 8192 &&
         config->numAttrs == static_cast<unsigned>(expected_attributes));
  if (config->numAttrs != 0) {
    assert(config->attrs[0].id == cudaLaunchAttributeCooperative &&
           config->attrs[0].val.cooperative == 1);
  }
}
static cudaError_t vendor_launch_ex(const cudaLaunchConfig_t *config,
                                    const void *func, void **args) {
  check_config(config);
  vendor_launch(func, config->gridDim, config->blockDim, args,
                config->dynamicSmemBytes, config->stream);
  launch_call = "extended";
  return cudaSuccess;
}
static cudaError_t vendor_cluster_size(int *value, const void *func,
                                       const cudaLaunchConfig_t *config) {
  assert(func == expected_entry && server == expected_launch_server);
  check_config(config);
  *value = 8;
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
  if (std::strcmp(name, "__cudaInitModule") == 0)
    return missing_init_module ? nullptr
                               : reinterpret_cast<void *>(&vendor_init_module);
  SYMBOL(__cudaRegisterFunction, vendor_function);
  SYMBOL(__cudaRegisterVar, vendor_var);
  SYMBOL(__cudaRegisterManagedVar, vendor_managed);
  SYMBOL(__cudaRegisterTexture, vendor_texture);
  SYMBOL(__cudaRegisterSurface, vendor_surface);
  SYMBOL(__cudaRegisterHostVar, vendor_host_var);
  SYMBOL(__cudaRegisterUnifiedTable, vendor_unified_table);
  SYMBOL(__cudaPushCallConfiguration, vendor_push);
  SYMBOL(__cudaPopCallConfiguration, vendor_pop);
  SYMBOL(cudaSetDevice, vendor_set_device);
  SYMBOL(cudaMalloc, vendor_malloc);
  SYMBOL(cudaMallocAsync, vendor_malloc_async);
  SYMBOL(cudaMallocPitch, vendor_malloc_pitch);
  SYMBOL(cudaFree, vendor_free);
  SYMBOL(cudaGetSymbolAddress, vendor_symbol);
  SYMBOL(cudaGetSymbolSize, vendor_symbol_size);
  SYMBOL(cudaMemPoolGetAttribute, vendor_pool_get);
  SYMBOL(cudaMemPoolSetAttribute, vendor_pool_set);
  SYMBOL(cudaDeviceGetGraphMemAttribute, vendor_graph_get);
  SYMBOL(cudaDeviceSetGraphMemAttribute, vendor_graph_set);
  SYMBOL(cudaFuncGetParamInfo, vendor_param_info);
  SYMBOL(cudaLaunchKernel, vendor_launch);
  SYMBOL(cudaLaunchCooperativeKernel, vendor_cooperative);
  SYMBOL(cudaLaunchKernelExC, vendor_launch_ex);
  SYMBOL(__cudaLaunchKernel, vendor_private_launch);
  SYMBOL(cudaOccupancyMaxPotentialClusterSize, vendor_cluster_size);
  SYMBOL(cudaOccupancyMaxActiveClusters, vendor_cluster_size);
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
extern "C" CUresult cuStreamSynchronize(CUstream stream) {
  driver_call = "stream_sync";
  driver_stream = stream;
  return driver_result;
}
extern "C" CUresult cuStreamQuery(CUstream stream) {
  driver_call = "stream_query";
  driver_stream = stream;
  return driver_result;
}
extern "C" CUresult cuEventSynchronize(CUevent event) {
  driver_call = "event_sync";
  driver_event = event;
  return driver_result;
}
extern "C" CUresult cuEventQuery(CUevent event) {
  driver_call = "event_query";
  driver_event = event;
  return driver_result;
}
extern "C" CUresult cuEventRecord(CUevent event, CUstream stream) {
  driver_call = "event_record";
  driver_event = event;
  driver_stream = stream;
  return driver_result;
}
extern "C" CUresult cuEventRecordWithFlags(CUevent event, CUstream stream,
                                           unsigned flags) {
  driver_call = "event_record_flags";
  driver_event = event;
  driver_stream = stream;
  driver_flags = flags;
  return driver_result;
}
extern "C" CUresult cuEventDestroy(CUevent event) {
  driver_call = "event_destroy";
  driver_event = event;
  return driver_result;
}

int main() {
  for (conn_t *conn : servers)
    assert(rpc_conn_init(conn, LUPINE_INVALID_SOCKET, 0) == 0);
  lupine_fatbin_header image = {};
  image.magic = LUPINE_FATBIN_MAGIC;
  image.header_size = sizeof(image);
  void **handle = __cudaRegisterFatBinary(&image);
  assert(handle && registrations == 2);
  // Unmapped address identities must reach every server unchanged. Attempting
  // to serialize window bytes would fault, and allocating copies would change
  // the addresses used by host-originated unified pointers.
  unified_functions = reinterpret_cast<void *>(0x123400000000ULL);
  unified_data = reinterpret_cast<void *>(0x567800000000ULL);
  unified_function_size = 128;
  unified_data_size = 16;
  __cudaRegisterUnifiedTable(handle, unified_functions, unified_function_size,
                             unified_data, unified_data_size);
  assert(unified_tables[0] == 1 && unified_tables[1] == 1);
  unified_functions = unified_data = nullptr;
  unified_function_size = unified_data_size = 0;
  __cudaRegisterUnifiedTable(handle, nullptr, 0, nullptr, 0);
  assert(unified_tables[0] == 2 && unified_tables[1] == 2);
  assert(cudaPeekAtLastError() == cudaSuccess);
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
  __cudaRegisterHostVar(handle, variable, reinterpret_cast<char *>(0xdef), 32);
  std::memset(kernel, '!', 6);
  std::memset(variable, '!', 8);
  assert(functions == 4 && variables == 2 && managed == 2 && textures == 2 &&
         surfaces == 2 && host_variables == 2);
  for (const char *name : retained_names)
    assert(std::strchr(name, '!') == nullptr);
  tid.x = 99;
  for (const uint3 *index : retained_indices)
    assert(index->x == 7);
  assert(static_cast<unsigned char>(__cudaInitModule(handle)) == 0xa5);
  assert(module_inits[0] == 1 && module_inits[1] == 0);
  assert(cudaSetDevice(1) == cudaSuccess);
  module_result = static_cast<char>(0x7e);
  assert(__cudaInitModule(handle) == 0x7e);
  assert(module_inits[0] == 1 && module_inits[1] == 1);
  assert(cudaSetDevice(0) == cudaSuccess);
  module_result = 0;
  assert(__cudaInitModule(handle) == 0);
  assert(cudaPeekAtLastError() ==
         cudaSuccess); // Raw bytes are not error codes.
  missing_init_module = true;
  assert(__cudaInitModule(handle) == 0);
  assert(cudaGetLastError() == cudaErrorNotSupported);
  missing_init_module = false;
  fail_request = true;
  assert(__cudaInitModule(handle) == 0);
  assert(cudaGetLastError() == cudaErrorDevicesUnavailable);
  fail_request = false;
  assert(__cudaInitModule(nullptr) == 0);
  assert(cudaGetLastError() == cudaErrorInvalidResourceHandle);
  __cudaRegisterFatBinaryEnd(handle);
  assert(ended[0] == 1 && ended[1] == 1);
  for (void **slot : managed_slots)
    assert(*slot == reinterpret_cast<void *>(0xdead));
  failing_server = 0;
  __cudaRegisterFatBinaryEnd(handle);
  assert(ended[0] == 1 && ended[1] == 2);
  assert(cudaPeekAtLastError() == cudaErrorDevicesUnavailable);
  failing_server = -1;
  __cudaUnregisterFatBinary(handle);
  assert(unregistered[0] == 1 && unregistered[1] == 1);
  assert(cudaGetLastError() == cudaErrorDevicesUnavailable);
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

  auto pool = reinterpret_cast<cudaMemPool_t>(0x555);
  size_t narrow_request_size = 0;
  for (auto attr : {
           cudaMemPoolReuseFollowEventDependencies,
           cudaMemPoolReuseAllowOpportunistic,
           cudaMemPoolReuseAllowInternalDependencies,
#if CUDART_VERSION >= 13020
           cudaMemPoolAttrAllocationType,
           cudaMemPoolAttrExportHandleTypes,
           cudaMemPoolAttrLocationId,
           cudaMemPoolAttrLocationType,
           cudaMemPoolAttrHwDecompressEnabled,
#endif
       }) {
    int narrow = 1; // Exactly four bytes: ASan catches any eight-byte copy.
    assert(cudaMemPoolSetAttribute(pool, attr, &narrow) == cudaSuccess);
    narrow_request_size = attribute_request_size;
    narrow = 0;
    assert(cudaMemPoolGetAttribute(pool, attr, &narrow) == cudaSuccess);
    assert(narrow == 1 && attribute_request_size == narrow_request_size);
    attribute_result = cudaErrorInvalidValue;
    narrow = 19;
    assert(cudaMemPoolGetAttribute(pool, attr, &narrow) ==
           cudaErrorInvalidValue);
    assert(narrow == 19); // Failed queries leave the caller's output alone.
    attribute_result = cudaSuccess;
  }
  uint64_t wide = attribute_value;
  assert(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                 &wide) == cudaSuccess);
  assert(attribute_request_size == narrow_request_size);
  wide = 0;
  assert(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                 &wide) == cudaSuccess);
  assert(wide == attribute_value &&
         attribute_request_size == narrow_request_size);
  assert(cudaDeviceSetGraphMemAttribute(1, cudaGraphMemAttrUsedMemHigh,
                                        &wide) == cudaSuccess &&
         attribute_server == 1);
  wide = 0;
  assert(cudaDeviceGetGraphMemAttribute(1, cudaGraphMemAttrUsedMemHigh,
                                        &wide) == cudaSuccess &&
         wide == attribute_value && attribute_server == 1);
  attribute_result = cudaErrorInvalidValue;
  wide = 17;
  assert(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh,
                                        &wide) == cudaErrorInvalidValue &&
         wide == 17 && attribute_server == 0);
  attribute_result = cudaSuccess;
  assert(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                 nullptr) == cudaErrorInvalidValue);
  assert(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                 nullptr) == cudaErrorInvalidValue);
  assert(cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh,
                                        nullptr) == cudaErrorInvalidValue);
  assert(cudaDeviceSetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh,
                                        nullptr) == cudaErrorInvalidValue);
  assert(cudaGetLastError() == cudaErrorInvalidValue);

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

  expected_entry = reinterpret_cast<const void *>(0xabc);
  expected_launch_server = 0;
  int first_arg = 42;
  uint64_t second_arg = 0x123456789abcdef0ULL;
  void *args[] = {&first_arg, &second_arg};
  int waits = response_waits;
  assert(cudaLaunchKernel(expected_entry, dim3(2, 3, 4), dim3(5, 6, 7), args,
                          8192, nullptr) == cudaSuccess);
  assert(launches == 0 && queued_launches.size() == 1);
  assert(response_waits == waits + 1); // Only the initial layout lookup waits.
  first_arg = 0;
  second_arg = 0;
  drain_launches();
  assert(launches == 1 && launch_call == "kernel");
  first_arg = 42;
  second_arg = 0x123456789abcdef0ULL;
  waits = response_waits;
  expected_launch_server = 1;
  assert(cudaLaunchCooperativeKernel(expected_entry, dim3(2, 3, 4),
                                     dim3(5, 6, 7), args, 8192,
                                     stream) == cudaSuccess);
  assert(response_waits == waits && launches == 1);
  drain_launches();
  assert(launch_call == "cooperative");
  for (int attrs = 0; attrs < 2; ++attrs) {
    expected_attributes = attrs;
    cudaLaunchAttribute attribute = {};
    attribute.id = cudaLaunchAttributeCooperative;
    attribute.val.cooperative = 1;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(2, 3, 4);
    config.blockDim = dim3(5, 6, 7);
    config.dynamicSmemBytes = 8192;
    config.stream = stream;
    config.numAttrs = attrs;
    config.attrs = attrs ? &attribute : nullptr;
    int clusters = 0;
    assert(cudaOccupancyMaxPotentialClusterSize(&clusters, expected_entry,
                                                &config) == cudaSuccess &&
           clusters == 8);
    assert(cudaOccupancyMaxActiveClusters(&clusters, expected_entry, &config) ==
               cudaSuccess &&
           clusters == 8);
    waits = response_waits;
    assert(cudaLaunchKernelExC(&config, expected_entry, args) == cudaSuccess);
    assert(response_waits == waits);
    config.gridDim = dim3(99);
    attribute.val.cooperative = 0;
    drain_launches();
    assert(launch_call == "extended");
  }
  expected_entry = reinterpret_cast<const void *>(0x999);
  auto kernel_handle =
      reinterpret_cast<cudaKernel_t>(const_cast<void *>(expected_entry));
  assert(__cudaLaunchKernel(kernel_handle, dim3(2, 3, 4), dim3(5, 6, 7), args,
                            8192, stream) == cudaSuccess);
  drain_launches();
  assert(launch_call == "private");
  waits = response_waits;
  assert(__cudaLaunchKernel_ptsz(kernel_handle, dim3(2, 3, 4), dim3(5, 6, 7),
                                 args, 8192, stream) == cudaSuccess);
  assert(response_waits == waits);
  drain_launches();
  assert(cudaLaunchKernel(expected_entry, dim3(2), dim3(5), nullptr, 0,
                          stream) == cudaErrorInvalidValue);
  void *invalid_args[] = {nullptr, &second_arg};
  assert(cudaLaunchKernel(expected_entry, dim3(2), dim3(5), invalid_args, 0,
                          stream) == cudaErrorInvalidValue);
  assert(cudaLaunchKernelExC(nullptr, expected_entry, args) ==
         cudaErrorInvalidValue);
  cudaLaunchConfig_t invalid_config = {};
  invalid_config.numAttrs = 1;
  assert(cudaLaunchKernelExC(&invalid_config, expected_entry, args) ==
         cudaErrorInvalidValue);
  assert(queued_launches.empty());
  fail_request = true;
  assert(cudaLaunchKernel(expected_entry, dim3(2, 3, 4), dim3(5, 6, 7), args,
                          8192, stream) == cudaErrorDevicesUnavailable);
  fail_request = false;
  assert(queued_launches.empty());
  expected_entry = zero_arg_entry;
  expected_launch_server = 0;
  assert(cudaLaunchKernel(expected_entry, dim3(2, 3, 4), dim3(5, 6, 7), nullptr,
                          8192, nullptr) == cudaSuccess);
  drain_launches();
  assert(servers[0]->issued_async_sequence ==
         servers[0]->serving_async_sequence);
  assert(servers[1]->issued_async_sequence ==
         servers[1]->serving_async_sequence);
  assert(cudaGetLastError() == cudaErrorDevicesUnavailable);

  auto event = reinterpret_cast<cudaEvent_t>(0x123);
  for (CUresult result :
       {CUDA_SUCCESS, CUDA_ERROR_NOT_READY, CUDA_ERROR_DEINITIALIZED}) {
    driver_result = result;
    auto expected = result == CUDA_ERROR_DEINITIALIZED
                        ? cudaErrorDeviceUninitialized
                        : static_cast<cudaError_t>(result);
    driver_event = nullptr;
    driver_stream = nullptr;
    assert(cudaEventRecord(event, stream) == expected &&
           driver_call == "event_record" && driver_event == event &&
           driver_stream == stream);
    driver_event = nullptr;
    driver_stream = nullptr;
    driver_flags = 0;
    assert(cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal) ==
               expected &&
           driver_call == "event_record_flags" && driver_event == event &&
           driver_stream == stream && driver_flags == cudaEventRecordExternal);
    driver_event = nullptr;
    assert(cudaEventSynchronize(event) == expected &&
           driver_call == "event_sync" && driver_event == event);
    driver_event = nullptr;
    assert(cudaEventQuery(event) == expected && driver_call == "event_query" &&
           driver_event == event);
    driver_event = nullptr;
    assert(cudaEventDestroy(event) == expected &&
           driver_call == "event_destroy" && driver_event == event);
    driver_stream = nullptr;
    assert(cudaStreamSynchronize(stream) == expected &&
           driver_call == "stream_sync" && driver_stream == stream);
    driver_stream = nullptr;
    assert(cudaStreamQuery(stream) == expected &&
           driver_call == "stream_query" && driver_stream == stream);
    assert(cudaDeviceSynchronize() == expected && driver_call == "device_sync");
  }
  driver_result = CUDA_SUCCESS;
  assert(cudaDeviceSynchronize() == cudaSuccess);
  assert(cudaGetLastError() == cudaErrorDeviceUninitialized);
  assert(cudaPeekAtLastError() == cudaSuccess);
  driver_result = CUDA_ERROR_DEINITIALIZED;
  assert(cudaMemcpy(dst, src, 1, cudaMemcpyDefault) ==
         cudaErrorDeviceUninitialized);
  assert(cudaPeekAtLastError() == cudaErrorDeviceUninitialized);
  assert(cudaGetLastError() == cudaErrorDeviceUninitialized);
  for (conn_t *conn : servers)
    rpc_conn_destroy(conn);
}
