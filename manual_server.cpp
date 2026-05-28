#include <arpa/inet.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <errno.h>
#include <future>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <dlfcn.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <vector>

#include <list>
#include <map>
#include <algorithm>

#include "cuda_compat.h"

#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "codegen/ptx_fatbin.hpp"
#include "manual_server.h"
#include "rpc.h"

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags,
                                           CUdevice dev);

static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;
static constexpr size_t LUPINE_HTOD_CHUNK_BYTES = 64 * 1024 * 1024;

struct lupine_captured_stdout {
  int saved_stdout = -1;
  FILE *capture_file = nullptr;
  bool active = false;
  std::string output;
};

static pthread_mutex_t lupine_stdout_capture_mutex = PTHREAD_MUTEX_INITIALIZER;

static bool lupine_start_stdout_capture(lupine_captured_stdout *capture) {
  if (capture == nullptr) {
    return false;
  }
  capture->saved_stdout = -1;
  capture->capture_file = nullptr;
  capture->active = false;
  capture->output.clear();

  if (pthread_mutex_lock(&lupine_stdout_capture_mutex) != 0) {
    return false;
  }

  fflush(stdout);
  std::cout.flush();

  capture->saved_stdout = dup(STDOUT_FILENO);
  capture->capture_file = tmpfile();
  if (capture->saved_stdout < 0 || capture->capture_file == nullptr) {
    if (capture->saved_stdout >= 0) {
      close(capture->saved_stdout);
      capture->saved_stdout = -1;
    }
    if (capture->capture_file != nullptr) {
      fclose(capture->capture_file);
      capture->capture_file = nullptr;
    }
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  if (dup2(fileno(capture->capture_file), STDOUT_FILENO) < 0) {
    fclose(capture->capture_file);
    capture->capture_file = nullptr;
    close(capture->saved_stdout);
    capture->saved_stdout = -1;
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  capture->active = true;
  return true;
}

static void lupine_finish_stdout_capture(lupine_captured_stdout *capture) {
  if (capture == nullptr || !capture->active) {
    return;
  }

  fflush(stdout);
  std::cout.flush();
  dup2(capture->saved_stdout, STDOUT_FILENO);
  close(capture->saved_stdout);
  capture->saved_stdout = -1;
  if (capture->capture_file != nullptr) {
    int capture_fd = fileno(capture->capture_file);
    if (capture_fd >= 0 && lseek(capture_fd, 0, SEEK_SET) >= 0) {
      char buffer[4096];
      for (;;) {
        ssize_t bytes = read(capture_fd, buffer, sizeof(buffer));
        if (bytes > 0) {
          capture->output.append(buffer, static_cast<size_t>(bytes));
          continue;
        }
        if (bytes == 0) {
          break;
        }
        if (errno == EINTR) {
          continue;
        }
        break;
      }
    }
    fclose(capture->capture_file);
    capture->capture_file = nullptr;
  }
  capture->active = false;
  pthread_mutex_unlock(&lupine_stdout_capture_mutex);
}

static int lupine_write_captured_stdout(conn_t *conn,
                                        const lupine_captured_stdout &capture,
                                        uint64_t *output_size) {
  if (output_size == nullptr) {
    return -1;
  }
  *output_size = capture.output.size();
  if (rpc_write(conn, output_size, sizeof(*output_size)) < 0) {
    return -1;
  }
  if (*output_size != 0 &&
      rpc_write(conn, capture.output.data(), capture.output.size()) < 0) {
    return -1;
  }
  return 0;
}

struct lupine_kernel_param_layout {
  uint32_t count = 0;
  size_t offsets[64] = {};
  size_t sizes[64] = {};
};

struct lupine_private_module_node_capture {
  void *node = nullptr;
  uint64_t owner = 0;
  uint64_t count = 0;
};

static bool lupine_pointer_attribute_size(CUpointer_attribute attr,
                                          size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_POINTER_ATTRIBUTE_CONTEXT:
    *size = sizeof(CUcontext);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_HOST_POINTER:
    *size = sizeof(void *);
    return true;
  case CU_POINTER_ATTRIBUTE_P2P_TOKENS:
    *size = sizeof(CUDA_POINTER_ATTRIBUTE_P2P_TOKENS);
    return true;
  case CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
  case CU_POINTER_ATTRIBUTE_IS_MANAGED:
  case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
  case CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE:
  case CU_POINTER_ATTRIBUTE_MAPPED:
  case CU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
  case CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
#if CUDA_VERSION >= 13000
  case CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE:
#endif
    *size = sizeof(int);
    return true;
  case CU_POINTER_ATTRIBUTE_BUFFER_ID:
  case CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID:
    *size = sizeof(unsigned long long);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
  case CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR:
    *size = sizeof(CUdeviceptr);
    return true;
  case CU_POINTER_ATTRIBUTE_RANGE_SIZE:
  case CU_POINTER_ATTRIBUTE_MAPPING_SIZE:
    *size = sizeof(size_t);
    return true;
  case CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
    *size = sizeof(unsigned int);
    return true;
  case CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
    *size = sizeof(CUmemoryPool);
    return true;
  default:
    return false;
  }
}

static std::unordered_map<CUmodule, CUlibrary> &lupine_module_libraries() {
  static std::unordered_map<CUmodule, CUlibrary> libraries;
  return libraries;
}

struct lupine_graph_host_copy {
  void *client_dst = nullptr;
  void *server_src = nullptr;
  size_t bytes = 0;
};

struct lupine_pending_dtoh_copy {
  CUstream stream = nullptr;
  void *client_dst = nullptr;
  void *server_src = nullptr;
  size_t bytes = 0;
  bool pinned = false;
};

struct lupine_graph_resources;

struct lupine_host_callback_data {
  conn_t *conn = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  std::shared_ptr<lupine_graph_resources> resources;
};

struct lupine_stream_callback_data {
  conn_t *conn = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
};

struct lupine_graph_resources {
  std::vector<CUdeviceptr> device_allocs;
  std::vector<void *> host_allocs;
  std::vector<void *> pageable_allocs;
  std::vector<void *> callback_allocs;
  std::vector<lupine_graph_host_copy> dtoh_copies;
  void *capture_scratch = nullptr;
  size_t capture_scratch_size = 0;
  size_t capture_scratch_offset = 0;

  ~lupine_graph_resources() {
    std::unordered_set<CUdeviceptr> freed_device_allocs;
    for (CUdeviceptr ptr : device_allocs) {
      if (ptr != 0 && freed_device_allocs.insert(ptr).second) {
        cuMemFree_v2(ptr);
      }
    }
    std::unordered_set<void *> freed_host_allocs;
    for (void *ptr : host_allocs) {
      if (ptr != nullptr && freed_host_allocs.insert(ptr).second) {
        cuMemFreeHost(ptr);
      }
    }
    std::unordered_set<void *> freed_pageable_allocs;
    for (void *ptr : pageable_allocs) {
      if (ptr != nullptr && freed_pageable_allocs.insert(ptr).second) {
        free(ptr);
      }
    }
    std::unordered_set<void *> freed_callbacks;
    for (void *ptr : callback_allocs) {
      if (ptr != nullptr && freed_callbacks.insert(ptr).second) {
        delete static_cast<lupine_host_callback_data *>(ptr);
      }
    }
  }
};

static std::unordered_map<CUgraph, std::shared_ptr<lupine_graph_resources>> &
lupine_graph_resource_map() {
  static std::unordered_map<CUgraph, std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static std::unordered_map<CUgraphExec,
                          std::shared_ptr<lupine_graph_resources>> &
lupine_graph_exec_resource_map() {
  static std::unordered_map<CUgraphExec,
                            std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static std::unordered_map<CUstream, std::shared_ptr<lupine_graph_resources>> &
lupine_stream_capture_resource_map() {
  static std::unordered_map<CUstream, std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static std::unordered_map<CUevent, std::shared_ptr<lupine_graph_resources>> &
lupine_event_capture_resource_map() {
  static std::unordered_map<CUevent, std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static void lupine_retire_graph_resources(
    const std::shared_ptr<lupine_graph_resources> &resources) {
  if (!resources) {
    return;
  }
  static auto *retired = new std::vector<std::shared_ptr<lupine_graph_resources>>;
  retired->push_back(resources);
}

static std::unordered_map<conn_t *, std::vector<lupine_pending_dtoh_copy>> &
lupine_pending_dtoh_copies() {
  static std::unordered_map<conn_t *, std::vector<lupine_pending_dtoh_copy>>
      copies;
  return copies;
}

static std::shared_ptr<lupine_graph_resources>
lupine_get_graph_resources(CUgraph graph) {
  auto &resources = lupine_graph_resource_map();
  auto it = resources.find(graph);
  if (it != resources.end()) {
    return it->second;
  }
  auto created = std::make_shared<lupine_graph_resources>();
  resources[graph] = created;
  return created;
}

static bool lupine_mem_pool_attribute_size(CUmemPool_attribute attr,
                                           size_t *size) {
  if (size == nullptr) {
    return false;
  }
  switch (attr) {
  case CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC:
  case CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES:
    *size = sizeof(int);
    return true;
  case CU_MEMPOOL_ATTR_RELEASE_THRESHOLD:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH:
  case CU_MEMPOOL_ATTR_USED_MEM_CURRENT:
  case CU_MEMPOOL_ATTR_USED_MEM_HIGH:
    *size = sizeof(cuuint64_t);
    return true;
  default:
    return false;
  }
}

static bool lupine_server_trace_enabled() {
  static bool enabled = [] {
    const char *value = getenv("LUPINE_SERVER_TRACE");
    return value != nullptr && strcmp(value, "0") != 0;
  }();
  return enabled;
}

static uint64_t lupine_fnv1a64(const void *data, size_t size) {
  static constexpr uint64_t kOffset = 14695981039346656037ull;
  static constexpr uint64_t kPrime = 1099511628211ull;
  const auto *bytes = static_cast<const unsigned char *>(data);
  uint64_t hash = kOffset;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
  return hash;
}

static uint64_t lupine_export_slot_hash(const void *fn) {
  if (fn == nullptr) {
    return 0;
  }
  Dl_info info = {};
  if (dladdr(fn, &info) == 0 || info.dli_fname == nullptr) {
    return 0;
  }
  return lupine_fnv1a64(fn, 32);
}

int handle_manual_cuGetExportTableMetadata(conn_t *conn) {
  CUuuid uuid = {};
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  uint64_t byte_size = 0;
  uint32_t slot_count = 0;
  uint32_t trusted = 0;
  uint64_t hashes[LUPINE_PRIVATE_EXPORT_MAX_SLOTS] = {};

  if (rpc_read(conn, uuid.bytes, sizeof(uuid.bytes)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  const void *export_table = nullptr;
  cuInit(0);
  result = cuGetExportTable(&export_table, &uuid);
  if (result == CUDA_SUCCESS && export_table != nullptr) {
    const auto *slots = static_cast<const void *const *>(export_table);
    byte_size = reinterpret_cast<uintptr_t>(slots[0]);
    if (byte_size >= sizeof(void *) && byte_size % sizeof(void *) == 0 &&
        byte_size / sizeof(void *) <= LUPINE_PRIVATE_EXPORT_MAX_SLOTS) {
      trusted = 1;
      slot_count = static_cast<uint32_t>(byte_size / sizeof(void *));
      for (uint32_t i = 1; i < slot_count; ++i) {
        hashes[i] = lupine_export_slot_hash(slots[i]);
      }
    }
  }

  if (lupine_server_trace_enabled()) {
    std::cerr << "LUPINE server cuGetExportTable metadata result=" << result
              << " bytes=" << byte_size << " slots=" << slot_count
              << " trusted=" << trusted << std::endl;
  }

  size_t hash_bytes = static_cast<size_t>(slot_count) * sizeof(uint64_t);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &byte_size, sizeof(byte_size)) < 0 ||
      rpc_write(conn, &slot_count, sizeof(slot_count)) < 0 ||
      rpc_write(conn, &trusted, sizeof(trusted)) < 0 ||
      (hash_bytes != 0 && rpc_write(conn, hashes, hash_bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static void lupine_private_module_node_callback(void *opaque, void *node,
                                                uint64_t owner) {
  auto *capture = static_cast<lupine_private_module_node_capture *>(opaque);
  if (capture == nullptr || capture->node != nullptr) {
    return;
  }
  capture->node = node;
  capture->owner = owner;
  capture->count = 1;
}

int handle_manual_cuPrivateGetModuleNode(conn_t *conn) {
  static constexpr unsigned char PRIVATE_MODULE_ITERATOR_UUID[16] = {
      0x6e, 0x16, 0x3f, 0xbe, 0xb9, 0x58, 0x44, 0x4d,
      0x83, 0x5c, 0xe1, 0x82, 0xaf, 0xf1, 0x99, 0x1e};

  CUcontext context = nullptr;
  CUmodule module = nullptr;
  int request_id;
  CUfunction node = nullptr;
  uint64_t owner = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &context, sizeof(context)) < 0 ||
      rpc_read(conn, &module, sizeof(module)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUuuid uuid = {};
  memcpy(uuid.bytes, PRIVATE_MODULE_ITERATOR_UUID, sizeof(uuid.bytes));
  const void *export_table = nullptr;
  result = cuGetExportTable(&export_table, &uuid);
  if (result == CUDA_SUCCESS && export_table != nullptr) {
    const auto *slots = static_cast<const void *const *>(export_table);
    size_t byte_size = reinterpret_cast<uintptr_t>(slots[0]);
    if (byte_size <= 7 * sizeof(void *) || slots[7] == nullptr) {
      result = CUDA_ERROR_NOT_FOUND;
    } else {
      using private_module_iterator = uint64_t (*)(
          uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
      auto iterator = reinterpret_cast<private_module_iterator>(
          const_cast<void *>(slots[7]));
      lupine_private_module_node_capture capture;
      CUcontext previous = nullptr;
      cuCtxGetCurrent(&previous);
      if (context != nullptr) {
        cuCtxSetCurrent(context);
      }
      uint64_t count = iterator(
          reinterpret_cast<uint64_t>(context),
          reinterpret_cast<uint64_t>(module),
          reinterpret_cast<uint64_t>(&lupine_private_module_node_callback),
          reinterpret_cast<uint64_t>(&capture),
          reinterpret_cast<uint64_t>(module), 0);
      if (previous != context) {
        cuCtxSetCurrent(previous);
      }
      if (capture.node != nullptr) {
        node = reinterpret_cast<CUfunction>(capture.node);
        owner = capture.owner;
        result = CUDA_SUCCESS;
      } else {
        result = count == 0 ? CUDA_ERROR_NOT_FOUND : CUDA_ERROR_UNKNOWN;
      }
      if (lupine_server_trace_enabled()) {
        std::cerr << "LUPINE server private module node module=" << module
                  << " context=" << context << " count=" << count
                  << " node=" << node
                  << " owner=" << reinterpret_cast<void *>(owner)
                  << " result=" << static_cast<int>(result) << std::endl;
      }
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &node, sizeof(node)) < 0 ||
      rpc_write(conn, &owner, sizeof(owner)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static size_t lupine_memcpy3d_host_span_bytes(const CUDA_MEMCPY3D &params,
                                              bool source) {
  size_t width = params.WidthInBytes;
  size_t height = params.Height == 0 ? 1 : params.Height;
  size_t depth = params.Depth == 0 ? 1 : params.Depth;
  size_t pitch = source ? params.srcPitch : params.dstPitch;
  if (pitch == 0) {
    pitch = width;
  }
  return pitch * height * depth;
}

static int lupine_read_graph_dependencies(conn_t *conn,
                                          std::vector<CUgraphNode> *deps) {
  size_t count = 0;
  if (deps == nullptr || rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  deps->resize(count);
  if (count != 0 &&
      rpc_read(conn, deps->data(), count * sizeof(CUgraphNode)) < 0) {
    return -1;
  }
  return 0;
}

static void *lupine_alloc_host_resource(
    const std::shared_ptr<lupine_graph_resources> &resources, size_t bytes) {
  void *ptr = nullptr;
  if (bytes == 0) {
    return nullptr;
  }
  if (cuMemAllocHost(&ptr, bytes) != CUDA_SUCCESS) {
    ptr = malloc(bytes);
    if (ptr != nullptr) {
      resources->pageable_allocs.push_back(ptr);
    }
    return ptr;
  }
  if (ptr != nullptr) {
    resources->host_allocs.push_back(ptr);
  }
  return ptr;
}

static uint32_t lupine_count_pending_dtoh_copies(conn_t *conn, CUstream stream,
                                                 bool all_streams) {
  auto &pending = lupine_pending_dtoh_copies()[conn];
  uint32_t count = 0;
  for (const auto &copy : pending) {
    if (all_streams || copy.stream == stream) {
      ++count;
    }
  }
  return count;
}

static int lupine_write_pending_dtoh_copies(conn_t *conn, CUstream stream,
                                            bool all_streams,
                                            bool write_count = true) {
  auto &pending = lupine_pending_dtoh_copies()[conn];
  if (write_count) {
    uint32_t count =
        lupine_count_pending_dtoh_copies(conn, stream, all_streams);
    if (rpc_write(conn, &count, sizeof(count)) < 0) {
      return -1;
    }
  }
  for (auto &copy : pending) {
    if (!all_streams && copy.stream != stream) {
      continue;
    }
    if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
        rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
        (copy.bytes != 0 && rpc_write(conn, copy.server_src, copy.bytes) < 0)) {
      return -1;
    }
  }
  return 0;
}

static void lupine_cleanup_pending_dtoh_copies(conn_t *conn, CUstream stream,
                                               bool all_streams) {
  auto &pending = lupine_pending_dtoh_copies()[conn];
  auto keep = std::remove_if(
      pending.begin(), pending.end(), [&](lupine_pending_dtoh_copy &copy) {
        if (!all_streams && copy.stream != stream) {
          return false;
        }
        if (copy.server_src != nullptr) {
          if (copy.pinned) {
            cuMemFreeHost(copy.server_src);
          } else {
            free(copy.server_src);
          }
          copy.server_src = nullptr;
        }
        return true;
      });
  pending.erase(keep, pending.end());
}

static void *lupine_alloc_capture_scratch(
    const std::shared_ptr<lupine_graph_resources> &resources, size_t bytes) {
  if (!resources || bytes == 0) {
    return nullptr;
  }
  size_t offset = (resources->capture_scratch_offset + 255) & ~size_t(255);
  if (offset + bytes > resources->capture_scratch_size) {
    return nullptr;
  }
  auto *ptr = static_cast<unsigned char *>(resources->capture_scratch) + offset;
  resources->capture_scratch_offset = offset + bytes;
  return ptr;
}

int rpc_write(const void *conn, const void *data, const size_t size) {
  ((conn_t *)conn)->write_iov[((conn_t *)conn)->write_iov_count++] =
      (struct iovec){(void *)data, size};
  return 0;
}

int handle_manual_cuModuleLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUmodule module = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 || rpc_read(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (kind == lupine::kModuleImageFatbinWrapperV1 ||
      kind == lupine::kModuleImageFatbinWrapperV2) {
    result = cuModuleLoadFatBinary(&module, image.data());
  } else if (kind == lupine::kModuleImageFatbinRaw) {
    result = cuModuleLoadData(&module, image.data());
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUlibrary library = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 || rpc_read(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (kind == lupine::kModuleImageFatbinWrapperV1 ||
      kind == lupine::kModuleImageFatbinWrapperV2) {
    lupine::fatbin_wrapper wrapper = {
        lupine::kFatbinWrapperMagic,
        kind == lupine::kModuleImageFatbinWrapperV2 ? 2U : 1U,
        image.data(),
        nullptr,
    };
    result = cuLibraryLoadData(&library, &wrapper, nullptr, nullptr, 0, nullptr,
                               nullptr, 0);
  } else if (kind == lupine::kModuleImageFatbinRaw) {
    result = cuLibraryLoadData(&library, image.data(), nullptr, nullptr, 0,
                               nullptr, nullptr, 0);
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuCtxCreate_v2(conn_t *conn) {
  unsigned int flags = 0;
  CUdevice dev = 0;
  int request_id;
  CUcontext ctx = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &dev, sizeof(dev)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuCtxCreate_v2(&ctx, flags, dev);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolSetAttribute(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemPool_attribute attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_mem_pool_attribute_size(attr, &expected_size) ||
      value_size != expected_size) {
    return -1;
  }

  std::vector<unsigned char> value(value_size);
  if (rpc_read(conn, value.data(), value_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuMemPoolSetAttribute(pool, attr, value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolGetAttribute(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemPool_attribute attr = CU_MEMPOOL_ATTR_RELEASE_THRESHOLD;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_mem_pool_attribute_size(attr, &expected_size) ||
      value_size != expected_size) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> value(value_size);
  result = cuMemPoolGetAttribute(pool, attr, value.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, value.data(), value_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerGetAttribute(conn_t *conn) {
  CUpointer_attribute attribute;
  CUdeviceptr ptr = 0;
  size_t value_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  unsigned char value[64] = {};

  if (rpc_read(conn, &attribute, sizeof(attribute)) < 0 ||
      rpc_read(conn, &ptr, sizeof(ptr)) < 0 ||
      rpc_read(conn, &value_size, sizeof(value_size)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_pointer_attribute_size(attribute, &expected_size) ||
      value_size != expected_size || value_size > sizeof(value)) {
    result = CUDA_ERROR_NOT_SUPPORTED;
    value_size = 0;
  } else {
    result = cuPointerGetAttribute(value, attribute, ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, value, value_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuPointerGetAttributes(conn_t *conn) {
  unsigned int num_attributes = 0;
  CUdeviceptr ptr = 0;
  int request_id;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &num_attributes, sizeof(num_attributes)) < 0) {
    return -1;
  }
  std::vector<CUpointer_attribute> attributes(num_attributes);
  if (num_attributes != 0 &&
      rpc_read(conn, attributes.data(),
               num_attributes * sizeof(CUpointer_attribute)) < 0) {
    return -1;
  }
  if (rpc_read(conn, &ptr, sizeof(ptr)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<size_t> value_sizes(num_attributes, 0);
  std::vector<std::vector<unsigned char>> values(num_attributes);
  std::vector<void *> data(num_attributes, nullptr);
  for (unsigned int i = 0; i < num_attributes; ++i) {
    size_t value_size = 0;
    if (!lupine_pointer_attribute_size(attributes[i], &value_size) ||
        value_size > 64) {
      result = CUDA_ERROR_NOT_SUPPORTED;
      break;
    }
    value_sizes[i] = value_size;
    values[i].resize(value_size);
    data[i] = values[i].data();
  }

  if (result == CUDA_SUCCESS) {
    result = cuPointerGetAttributes(num_attributes, attributes.data(),
                                    data.data(), ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_attributes; ++i) {
    size_t value_size = result == CUDA_SUCCESS ? value_sizes[i] : 0;
    if (rpc_write(conn, &value_size, sizeof(value_size)) < 0 ||
        (value_size != 0 &&
         rpc_write(conn, values[i].data(), value_size) < 0)) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuArrayCreate_v2(conn_t *conn) {
  CUDA_ARRAY_DESCRIPTOR desc = {};
  CUarray handle = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &desc, sizeof(desc)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuArrayCreate_v2(&handle, &desc);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuArray3DCreate_v2(conn_t *conn) {
  CUDA_ARRAY3D_DESCRIPTOR desc = {};
  CUarray handle = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &desc, sizeof(desc)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuArray3DCreate_v2(&handle, &desc);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

struct lupine_jit_server_state {
  std::vector<CUjit_option> options;
  std::vector<uintptr_t> raw_values;
  std::vector<void *> option_values;
  float wall_time = 0.0f;
  std::vector<char> info_log;
  std::vector<char> error_log;
  bool capture_wall_time = false;
  bool capture_info_log = false;
  bool capture_error_log = false;
};

static std::unordered_map<CUlinkState,
                          std::unique_ptr<lupine_jit_server_state>> &
lupine_jit_server_states() {
  static std::unordered_map<CUlinkState,
                            std::unique_ptr<lupine_jit_server_state>>
      states;
  return states;
}

static size_t lupine_find_jit_size_option(const lupine_jit_server_state &state,
                                          CUjit_option option) {
  for (size_t i = 0; i < state.options.size(); ++i) {
    if (state.options[i] == option) {
      return static_cast<size_t>(state.raw_values[i]);
    }
  }
  return 0;
}

static void lupine_prepare_jit_options(lupine_jit_server_state *state) {
  if (state == nullptr) {
    return;
  }
  size_t info_size =
      lupine_find_jit_size_option(*state, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
  size_t error_size =
      lupine_find_jit_size_option(*state, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
  state->option_values.resize(state->options.size());
  if (info_size != 0) {
    state->info_log.assign(info_size, '\0');
  }
  if (error_size != 0) {
    state->error_log.assign(error_size, '\0');
  }
  for (size_t i = 0; i < state->options.size(); ++i) {
    switch (state->options[i]) {
    case CU_JIT_WALL_TIME:
      state->capture_wall_time = true;
      state->option_values[i] = &state->wall_time;
      break;
    case CU_JIT_INFO_LOG_BUFFER:
      state->capture_info_log = !state->info_log.empty();
      state->option_values[i] =
          state->info_log.empty() ? nullptr : state->info_log.data();
      break;
    case CU_JIT_ERROR_LOG_BUFFER:
      state->capture_error_log = !state->error_log.empty();
      state->option_values[i] =
          state->error_log.empty() ? nullptr : state->error_log.data();
      break;
    default:
      state->option_values[i] = reinterpret_cast<void *>(state->raw_values[i]);
      break;
    }
  }
}

static int lupine_read_jit_options(conn_t *conn,
                                   lupine_jit_server_state *state) {
  unsigned int num_options = 0;
  if (rpc_read(conn, &num_options, sizeof(num_options)) < 0) {
    return -1;
  }
  if (state == nullptr) {
    return -1;
  }
  state->options.resize(num_options);
  state->raw_values.resize(num_options);
  if (num_options != 0 && (rpc_read(conn, state->options.data(),
                                    num_options * sizeof(CUjit_option)) < 0 ||
                           rpc_read(conn, state->raw_values.data(),
                                    num_options * sizeof(uintptr_t)) < 0)) {
    return -1;
  }
  lupine_prepare_jit_options(state);
  return 0;
}

static uint32_t lupine_jit_output_count(const lupine_jit_server_state &state) {
  uint32_t count = 0;
  if (state.capture_wall_time) {
    ++count;
  }
  if (state.capture_info_log) {
    ++count;
  }
  if (state.capture_error_log) {
    ++count;
  }
  return count;
}

int handle_manual_cuLinkCreate_v2(conn_t *conn) {
  auto jit_state = std::make_unique<lupine_jit_server_state>();
  CUlinkState state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (lupine_read_jit_options(conn, jit_state.get()) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkCreate_v2(
      static_cast<unsigned int>(jit_state->options.size()),
      jit_state->options.empty() ? nullptr : jit_state->options.data(),
      jit_state->option_values.empty() ? nullptr
                                       : jit_state->option_values.data(),
      &state);
  if (result == CUDA_SUCCESS) {
    lupine_jit_server_states()[state] = std::move(jit_state);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &state, sizeof(state)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkAddData_v2(conn_t *conn) {
  CUlinkState state = nullptr;
  CUjitInputType type = CU_JIT_INPUT_PTX;
  size_t size = 0;
  size_t name_len = 0;
  lupine_jit_server_state jit_state;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> data(size);
  if ((size != 0 && rpc_read(conn, data.data(), size) < 0) ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0) {
    return -1;
  }
  std::vector<char> name(name_len == 0 ? 1 : name_len, '\0');
  if ((name_len != 0 && rpc_read(conn, name.data(), name_len) < 0) ||
      lupine_read_jit_options(conn, &jit_state) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkAddData_v2(
      state, type, data.data(), data.size(),
      name_len == 0 ? nullptr : name.data(),
      static_cast<unsigned int>(jit_state.options.size()),
      jit_state.options.empty() ? nullptr : jit_state.options.data(),
      jit_state.option_values.empty() ? nullptr
                                      : jit_state.option_values.data());
  uint32_t output_count = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &output_count, sizeof(output_count)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkAddFile_v2(conn_t *conn) {
  CUlinkState state = nullptr;
  CUjitInputType type = CU_JIT_INPUT_LIBRARY;
  size_t path_len = 0;
  uint8_t has_file_data = 0;
  uint64_t file_size = 0;
  lupine_jit_server_state jit_state;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      rpc_read(conn, &path_len, sizeof(path_len)) < 0) {
    return -1;
  }
  std::vector<char> path(path_len == 0 ? 1 : path_len, '\0');
  if ((path_len != 0 && rpc_read(conn, path.data(), path_len) < 0) ||
      rpc_read(conn, &has_file_data, sizeof(has_file_data)) < 0 ||
      rpc_read(conn, &file_size, sizeof(file_size)) < 0 ||
      file_size > (1ull << 32) ||
      (file_size != 0 && has_file_data == 0)) {
    return -1;
  }
  std::vector<char> file_data;
  if (has_file_data != 0) {
    file_data.resize(static_cast<size_t>(file_size));
    if (!file_data.empty() &&
        rpc_read(conn, file_data.data(), file_data.size()) < 0) {
      return -1;
    }
  }
  if (lupine_read_jit_options(conn, &jit_state) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  if (!file_data.empty()) {
    result = cuLinkAddData_v2(
        state, type, file_data.data(), file_data.size(),
        path_len == 0 ? nullptr : path.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  } else {
    result = cuLinkAddFile_v2(
        state, type, path_len == 0 ? nullptr : path.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  }
  uint32_t output_count = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &output_count, sizeof(output_count)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkComplete(conn_t *conn) {
  CUlinkState state = nullptr;
  void *cubin = nullptr;
  size_t size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkComplete(state, &cubin, &size);
  size_t returned_size = result == CUDA_SUCCESS ? size : 0;
  auto jit_it = lupine_jit_server_states().find(state);
  const lupine_jit_server_state empty_jit_state;
  const auto &jit_state = jit_it == lupine_jit_server_states().end()
                              ? empty_jit_state
                              : *jit_it->second;
  std::vector<CUjit_option> output_options;
  std::vector<size_t> output_sizes;
  std::vector<const void *> output_data;
  output_options.reserve(lupine_jit_output_count(jit_state));
  output_sizes.reserve(output_options.capacity());
  output_data.reserve(output_options.capacity());
  if (jit_state.capture_wall_time) {
    output_options.push_back(CU_JIT_WALL_TIME);
    output_sizes.push_back(sizeof(jit_state.wall_time));
    output_data.push_back(&jit_state.wall_time);
  }
  if (jit_state.capture_info_log) {
    output_options.push_back(CU_JIT_INFO_LOG_BUFFER);
    output_sizes.push_back(jit_state.info_log.size());
    output_data.push_back(jit_state.info_log.data());
  }
  if (jit_state.capture_error_log) {
    output_options.push_back(CU_JIT_ERROR_LOG_BUFFER);
    output_sizes.push_back(jit_state.error_log.size());
    output_data.push_back(jit_state.error_log.data());
  }
  uint32_t output_count = static_cast<uint32_t>(output_options.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_size, sizeof(returned_size)) < 0 ||
      (returned_size != 0 && rpc_write(conn, cubin, returned_size) < 0) ||
      rpc_write(conn, &output_count, sizeof(output_count)) < 0) {
    return -1;
  }
  for (size_t i = 0; i < output_options.size(); ++i) {
    if (rpc_write(conn, &output_options[i], sizeof(output_options[i])) < 0 ||
        rpc_write(conn, &output_sizes[i], sizeof(output_sizes[i])) < 0 ||
        (output_sizes[i] != 0 &&
         rpc_write(conn, output_data[i], output_sizes[i]) < 0)) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkDestroy(conn_t *conn) {
  CUlinkState state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read(conn, &state, sizeof(state)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkDestroy(state);
  lupine_jit_server_states().erase(state);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpy3D_v2(conn_t *conn) {
  CUDA_MEMCPY3D copy = {};
  size_t src_host_size = 0;
  size_t dst_host_size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &src_host_size, sizeof(src_host_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> src_host(src_host_size);
  if (src_host_size != 0 &&
      rpc_read(conn, src_host.data(), src_host_size) < 0) {
    return -1;
  }
  if (rpc_read(conn, &dst_host_size, sizeof(dst_host_size)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> dst_host(dst_host_size);
  if (copy.srcMemoryType == CU_MEMORYTYPE_HOST) {
    copy.srcHost = src_host.empty() ? nullptr : src_host.data();
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = dst_host.empty() ? nullptr : dst_host.data();
  }

  result = cuMemcpy3D_v2(&copy);
  size_t returned_dst_size = result == CUDA_SUCCESS ? dst_host.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0 ||
      (returned_dst_size != 0 &&
       rpc_write(conn, dst_host.data(), returned_dst_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int handle_manual_cuMemcpy2D_common(conn_t *conn, bool async,
                                           bool unaligned) {
  CUDA_MEMCPY2D copy = {};
  size_t src_host_size = 0;
  size_t dst_host_size = 0;
  CUstream stream = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &copy, sizeof(copy)) < 0 ||
      rpc_read(conn, &src_host_size, sizeof(src_host_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> src_host(src_host_size);
  if (src_host_size != 0 &&
      rpc_read(conn, src_host.data(), src_host_size) < 0) {
    return -1;
  }
  if (rpc_read(conn, &dst_host_size, sizeof(dst_host_size)) < 0 ||
      (async && rpc_read(conn, &stream, sizeof(stream)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<unsigned char> dst_host(dst_host_size);
  if (copy.srcMemoryType == CU_MEMORYTYPE_HOST) {
    copy.srcHost = src_host.empty() ? nullptr : src_host.data();
  }
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = dst_host.empty() ? nullptr : dst_host.data();
  }

  if (async) {
    result = cuMemcpy2DAsync_v2(&copy, stream);
    if (result == CUDA_SUCCESS) {
      result = cuStreamSynchronize(stream);
    }
  } else if (unaligned) {
    result = cuMemcpy2DUnaligned_v2(&copy);
  } else {
    result = cuMemcpy2D_v2(&copy);
  }

  size_t returned_dst_size = result == CUDA_SUCCESS ? dst_host.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_dst_size, sizeof(returned_dst_size)) < 0 ||
      (returned_dst_size != 0 &&
       rpc_write(conn, dst_host.data(), returned_dst_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpy2D_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, false, false);
}

int handle_manual_cuMemcpy2DUnaligned_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, false, true);
}

int handle_manual_cuMemcpy2DAsync_v2(conn_t *conn) {
  return handle_manual_cuMemcpy2D_common(conn, true, false);
}

int handle_manual_cuGraphAddMemAllocNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEM_ALLOC_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemAllocNode(&graphNode, hGraph,
                                  deps.empty() ? nullptr : deps.data(),
                                  deps.size(), &nodeParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemFreeNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUdeviceptr dptr = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &dptr, sizeof(dptr)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemFreeNode(&graphNode, hGraph,
                                 deps.empty() ? nullptr : deps.data(),
                                 deps.size(), dptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDeviceGetGraphMemAttribute(conn_t *conn) {
  CUdevice device = 0;
  CUgraphMem_attribute attr = CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
  cuuint64_t value = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuDeviceGetGraphMemAttribute(device, attr, &value);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &value, sizeof(value)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuDeviceSetGraphMemAttribute(conn_t *conn) {
  CUdevice device = 0;
  CUgraphMem_attribute attr = CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT;
  cuuint64_t value = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      rpc_read(conn, &attr, sizeof(attr)) < 0 ||
      rpc_read(conn, &value, sizeof(value)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuDeviceSetGraphMemAttribute(device, attr, &value);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuTexObjectCreate(conn_t *conn) {
  CUDA_RESOURCE_DESC resDesc = {};
  CUDA_TEXTURE_DESC texDesc = {};
  CUDA_RESOURCE_VIEW_DESC resViewDesc = {};
  uint32_t hasTexDesc = 0;
  uint32_t hasResViewDesc = 0;
  CUtexObject texObject = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &resDesc, sizeof(resDesc)) < 0 ||
      rpc_read(conn, &hasTexDesc, sizeof(hasTexDesc)) < 0 ||
      (hasTexDesc != 0 && rpc_read(conn, &texDesc, sizeof(texDesc)) < 0) ||
      rpc_read(conn, &hasResViewDesc, sizeof(hasResViewDesc)) < 0 ||
      (hasResViewDesc != 0 &&
       rpc_read(conn, &resViewDesc, sizeof(resViewDesc)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuTexObjectCreate(&texObject, &resDesc,
                             hasTexDesc != 0 ? &texDesc : nullptr,
                             hasResViewDesc != 0 ? &resViewDesc : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &texObject, sizeof(texObject)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryGetModule(conn_t *conn) {
  CUlibrary library = nullptr;
  CUmodule module = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuLibraryGetModule(&module, library);
  if (result == CUDA_SUCCESS) {
    lupine_module_libraries()[module] = library;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLibraryUnload(conn_t *conn) {
  CUlibrary library = nullptr;
  int request_id;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  (void)library;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuModuleGetGlobal_v2(conn_t *conn) {
  CUdeviceptr dptr = 0;
  size_t bytes = 0;
  CUmodule module = nullptr;
  std::size_t name_len = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &module, sizeof(module)) < 0 ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0) {
    return -1;
  }
  std::vector<char> name(name_len);
  if (name_len == 0 || rpc_read(conn, name.data(), name_len) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuModuleGetGlobal_v2(&dptr, &bytes, module, name.data());
  if (result != CUDA_SUCCESS) {
    auto library_it = lupine_module_libraries().find(module);
    if (library_it != lupine_module_libraries().end()) {
      CUdeviceptr library_dptr = 0;
      size_t library_bytes = 0;
      CUresult library_result = cuLibraryGetGlobal(
          &library_dptr, &library_bytes, library_it->second, name.data());
      if (library_result == CUDA_SUCCESS) {
        dptr = library_dptr;
        bytes = library_bytes;
        result = library_result;
      }
    }
  }
  if (lupine_server_trace_enabled()) {
    std::cerr << "LUPINE cuModuleGetGlobal name=" << name.data()
              << " result=" << static_cast<int>(result) << " bytes=" << bytes
              << std::endl;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(dptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

CUresult lupine_get_kernel_param_layout(CUfunction f,
                                        lupine_kernel_param_layout *layout) {
  if (layout == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *layout = {};
  bool use_kernel_info = false;
  for (uint32_t i = 0; i < 64; ++i) {
    size_t offset = 0;
    size_t size = 0;
    CUresult result = use_kernel_info
                          ? cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f),
                                                 i, &offset, &size)
                          : cuFuncGetParamInfo(f, i, &offset, &size);
    if (result == CUDA_ERROR_INVALID_VALUE) {
      return CUDA_SUCCESS;
    }
    if (i == 0 && result == CUDA_ERROR_INVALID_HANDLE) {
      use_kernel_info = true;
      result = cuKernelGetParamInfo(reinterpret_cast<CUkernel>(f), i, &offset,
                                    &size);
      if (result == CUDA_ERROR_INVALID_VALUE) {
        return CUDA_SUCCESS;
      }
    }
    if (result != CUDA_SUCCESS) {
      return i == 0 ? result : CUDA_SUCCESS;
    }
    layout->offsets[i] = offset;
    layout->sizes[i] = size;
    layout->count = i + 1;
  }
  return CUDA_SUCCESS;
}

int handle_manual_cuFuncGetParamLayout(conn_t *conn) {
  CUfunction f = nullptr;
  int request_id;
  lupine_kernel_param_layout layout;
  CUresult result;

  if (rpc_read(conn, &f, sizeof(f)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = lupine_get_kernel_param_layout(f, &layout);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &layout.count, sizeof(layout.count)) < 0 ||
      rpc_write(conn, layout.offsets, sizeof(layout.offsets)) < 0 ||
      rpc_write(conn, layout.sizes, sizeof(layout.sizes)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchKernel(conn_t *conn) {
  CUfunction f = nullptr;
  unsigned int gridDimX = 0;
  unsigned int gridDimY = 0;
  unsigned int gridDimZ = 0;
  unsigned int blockDimX = 0;
  unsigned int blockDimY = 0;
  unsigned int blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  uint32_t param_count = 0;
  size_t packed_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_kernel_param_layout layout;
  result = lupine_get_kernel_param_layout(f, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      result =
          cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                         blockDimZ, sharedMemBytes, hStream,
                         param_count == 0 ? nullptr : params.data(), nullptr);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchCooperativeKernel(conn_t *conn) {
  CUfunction f = nullptr;
  unsigned int gridDimX = 0;
  unsigned int gridDimY = 0;
  unsigned int gridDimZ = 0;
  unsigned int blockDimX = 0;
  unsigned int blockDimY = 0;
  unsigned int blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  uint32_t param_count = 0;
  size_t packed_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &gridDimX, sizeof(gridDimX)) < 0 ||
      rpc_read(conn, &gridDimY, sizeof(gridDimY)) < 0 ||
      rpc_read(conn, &gridDimZ, sizeof(gridDimZ)) < 0 ||
      rpc_read(conn, &blockDimX, sizeof(blockDimX)) < 0 ||
      rpc_read(conn, &blockDimY, sizeof(blockDimY)) < 0 ||
      rpc_read(conn, &blockDimZ, sizeof(blockDimZ)) < 0 ||
      rpc_read(conn, &sharedMemBytes, sizeof(sharedMemBytes)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_kernel_param_layout layout;
  result = lupine_get_kernel_param_layout(f, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      result = cuLaunchCooperativeKernel(
          f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
          sharedMemBytes, hStream, param_count == 0 ? nullptr : params.data());
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static void CUDA_CB lupine_graph_host_callback(void *userData) {
  auto *callback = static_cast<lupine_host_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr) {
    return;
  }

  int transfer_count = 0;
  if (callback->resources) {
    transfer_count = static_cast<int>(callback->resources->dtoh_copies.size());
  }

  conn_t *conn = callback->conn;
  if (rpc_write_start_request(conn, 1) < 0 ||
      rpc_write(conn, &transfer_count, sizeof(transfer_count)) < 0) {
    return;
  }
  if (callback->resources) {
    for (const auto &copy : callback->resources->dtoh_copies) {
      if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
          rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
          rpc_write(conn, copy.server_src, copy.bytes) < 0) {
        return;
      }
    }
  }
  void *fn = reinterpret_cast<void *>(callback->fn);
  void *response = nullptr;
  if (rpc_write(conn, &fn, sizeof(fn)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &response, sizeof(response)) < 0 ||
      rpc_read_end(conn) < 0) {
    return;
  }
}

static void CUDA_CB lupine_stream_callback(CUstream stream, CUresult status,
                                           void *userData) {
  auto *callback = static_cast<lupine_stream_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr ||
      callback->callback == nullptr) {
    delete callback;
    return;
  }

  conn_t *conn = callback->conn;
  void *fn = reinterpret_cast<void *>(callback->callback);
  void *client_user_data = callback->userData;
  void *response = nullptr;
  if (rpc_write_start_request(conn, 2) >= 0 &&
      lupine_write_pending_dtoh_copies(conn, stream, false) >= 0 &&
      rpc_write(conn, &stream, sizeof(stream)) >= 0 &&
      rpc_write(conn, &status, sizeof(status)) >= 0 &&
      rpc_write(conn, &fn, sizeof(fn)) >= 0 &&
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) >= 0 &&
      rpc_wait_for_response(conn) >= 0) {
    lupine_cleanup_pending_dtoh_copies(conn, stream, false);
    rpc_read(conn, &response, sizeof(response));
    rpc_read_end(conn);
  }
  delete callback;
}

int handle_manual_cuGraphAddKernelNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t packed_size = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUfunction func = nodeParams.func;
#if CUDA_VERSION >= 12000
  if (func == nullptr) {
    func = reinterpret_cast<CUfunction>(nodeParams.kern);
  }
#endif
  lupine_kernel_param_layout layout;
  result = lupine_get_kernel_param_layout(func, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      nodeParams.kernelParams = params.data();
      nodeParams.extra = nullptr;
      result = cuGraphAddKernelNode_v2(&graphNode, hGraph,
                                       deps.empty() ? nullptr : deps.data(),
                                       deps.size(), &nodeParams);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecKernelNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec = nullptr;
  CUgraphNode hNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t packed_size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUfunction func = nodeParams.func;
#if CUDA_VERSION >= 12000
  if (func == nullptr) {
    func = reinterpret_cast<CUfunction>(nodeParams.kern);
  }
#endif
  lupine_kernel_param_layout layout;
  result = lupine_get_kernel_param_layout(func, &layout);
  if (result == CUDA_SUCCESS && layout.count == param_count) {
    std::vector<void *> params(param_count);
    for (uint32_t i = 0; i < param_count; ++i) {
      if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
        result = CUDA_ERROR_INVALID_VALUE;
        break;
      }
      params[i] = packed.data() + layout.offsets[i];
    }
    if (result == CUDA_SUCCESS) {
      nodeParams.kernelParams = params.data();
      nodeParams.extra = nullptr;
      result =
          cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, &nodeParams);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemcpyNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEMCPY3D copyParams = {};
  CUcontext ctx = nullptr;
  size_t host_src_bytes = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &copyParams, sizeof(copyParams)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &host_src_bytes, sizeof(host_src_bytes)) < 0) {
    return -1;
  }

  auto resources = lupine_get_graph_resources(hGraph);
  if (host_src_bytes != 0) {
    void *host = lupine_alloc_host_resource(resources, host_src_bytes);
    if (host == nullptr || rpc_read(conn, host, host_src_bytes) < 0) {
      return -1;
    }
    copyParams.srcHost = host;
  }

  if (copyParams.dstMemoryType == CU_MEMORYTYPE_HOST) {
    size_t host_dst_bytes = lupine_memcpy3d_host_span_bytes(copyParams, false);
    void *host = lupine_alloc_host_resource(resources, host_dst_bytes);
    if (host == nullptr && host_dst_bytes != 0) {
      return -1;
    }
    resources->dtoh_copies.push_back(
        {copyParams.dstHost, host, host_dst_bytes});
    copyParams.dstHost = host;
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemcpyNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size(), &copyParams, ctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddMemsetNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_MEMSET_NODE_PARAMS memsetParams = {};
  CUcontext ctx = nullptr;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &memsetParams, sizeof(memsetParams)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphAddMemsetNode(&graphNode, hGraph,
                                deps.empty() ? nullptr : deps.data(),
                                deps.size(), &memsetParams, ctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphAddHostNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_HOST_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback =
      new lupine_host_callback_data{conn, nodeParams.fn, nodeParams.userData,
                                    lupine_get_graph_resources(hGraph)};
  CUDA_HOST_NODE_PARAMS serverParams = {};
  serverParams.fn = lupine_graph_host_callback;
  serverParams.userData = callback;
  callback->resources->callback_allocs.push_back(callback);

  result = cuGraphAddHostNode(&graphNode, hGraph,
                              deps.empty() ? nullptr : deps.data(), deps.size(),
                              &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphConditionalHandleCreate(conn_t *conn) {
  CUgraph hGraph = nullptr;
  CUcontext ctx = nullptr;
  unsigned int defaultLaunchValue = 0;
  unsigned int flags = 0;
  CUgraphConditionalHandle handle = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
      rpc_read(conn, &defaultLaunchValue, sizeof(defaultLaunchValue)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphConditionalHandleCreate(&handle, hGraph, ctx,
                                          defaultLaunchValue, flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static CUresult lupine_server_cuGraphAddNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    CUgraphNodeParams *nodeParams) {
#if CUDA_VERSION >= 12060
  return cuGraphAddNode_v2(phGraphNode, hGraph, dependencies, dependencyData,
                           numDependencies, nodeParams);
#else
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuGraphAddNode(phGraphNode, hGraph, dependencies, numDependencies,
                        nodeParams);
#endif
}

int handle_manual_cuGraphAddNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUgraphNodeParams nodeParams = {};
  uint32_t param_count = 0;
  size_t packed_size = 0;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &packed_size, sizeof(packed_size)) < 0) {
    return -1;
  }
  std::vector<unsigned char> packed(packed_size);
  if (packed_size != 0 && rpc_read(conn, packed.data(), packed_size) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraph> child_graphs;
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    if (param_count == 0 && packed_size == 0) {
      nodeParams.kernel.kernelParams = nullptr;
      nodeParams.kernel.extra = nullptr;
      result = lupine_server_cuGraphAddNode(
          &graphNode, hGraph, deps.empty() ? nullptr : deps.data(), nullptr,
          deps.size(), &nodeParams);
    } else {
      CUfunction func =
          nodeParams.kernel.func != nullptr
              ? nodeParams.kernel.func
              : reinterpret_cast<CUfunction>(nodeParams.kernel.kern);
      lupine_kernel_param_layout layout;
      result = lupine_get_kernel_param_layout(func, &layout);
      if (result == CUDA_SUCCESS && layout.count == param_count) {
        std::vector<void *> params(param_count);
        for (uint32_t i = 0; i < param_count; ++i) {
          if (layout.offsets[i] + layout.sizes[i] > packed.size()) {
            result = CUDA_ERROR_INVALID_VALUE;
            break;
          }
          params[i] = packed.data() + layout.offsets[i];
        }
        if (result == CUDA_SUCCESS) {
          nodeParams.kernel.kernelParams = params.data();
          nodeParams.kernel.extra = nullptr;
          result = lupine_server_cuGraphAddNode(
              &graphNode, hGraph, deps.empty() ? nullptr : deps.data(), nullptr,
              deps.size(), &nodeParams);
        }
      }
    }
  } else if (nodeParams.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    child_graphs.resize(nodeParams.conditional.size);
    nodeParams.conditional.phGraph_out = nullptr;
    result = lupine_server_cuGraphAddNode(&graphNode, hGraph,
                                          deps.empty() ? nullptr : deps.data(),
                                          nullptr, deps.size(), &nodeParams);
    if (result == CUDA_SUCCESS &&
        nodeParams.conditional.phGraph_out != nullptr) {
      for (size_t i = 0; i < child_graphs.size(); ++i) {
        child_graphs[i] = nodeParams.conditional.phGraph_out[i];
      }
    }
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (lupine_server_trace_enabled()) {
    std::cerr << "LUPINE cuGraphAddNode type=" << nodeParams.type
              << " param_count=" << param_count
              << " packed_size=" << packed_size << " graph=" << hGraph
              << " node=" << graphNode << " result=" << result << std::endl;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      (!child_graphs.empty() &&
       rpc_write(conn, child_graphs.data(),
                 child_graphs.size() * sizeof(CUgraph)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphGetNodes(conn_t *conn) {
  CUgraph hGraph = nullptr;
  size_t requested = 0;
  size_t count = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &requested, sizeof(requested)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (requested == 0) {
    result = cuGraphGetNodes(hGraph, nullptr, &count);
  } else {
    count = requested;
  }
  std::vector<CUgraphNode> nodes(count);
  if (result == CUDA_SUCCESS && requested != 0) {
    result = cuGraphGetNodes(hGraph, nodes.data(), &count);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (requested != 0 && count != 0 &&
       rpc_write(conn, nodes.data(), count * sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLaunchHostFunc(conn_t *conn) {
  CUstream stream = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &fn, sizeof(fn)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto &stream_resources = lupine_stream_capture_resource_map();
  auto &resources = stream_resources[stream];
  if (!resources) {
    resources = std::make_shared<lupine_graph_resources>();
  }
  auto *callback = new lupine_host_callback_data{conn, fn, userData, resources};
  resources->callback_allocs.push_back(callback);
  result = cuLaunchHostFunc(stream, lupine_graph_host_callback, callback);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamAddCallback(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &userData, sizeof(userData)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *data = new lupine_stream_callback_data{conn, callback, userData};
  result = cuStreamAddCallback(stream, lupine_stream_callback, data, flags);
  if (result != CUDA_SUCCESS) {
    delete data;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventRecord(conn_t *conn, bool with_flags) {
  CUevent event = nullptr;
  CUstream stream = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto stream_it = lupine_stream_capture_resource_map().find(stream);
  if (stream_it != lupine_stream_capture_resource_map().end()) {
    lupine_event_capture_resource_map()[event] = stream_it->second;
  }

  result = with_flags ? cuEventRecordWithFlags(event, stream, flags)
                      : cuEventRecord(event, stream);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventQuery(conn_t *conn) {
  CUevent event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult result = cuEventQuery(event);

  if (rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  if (result == CUDA_SUCCESS) {
    if (lupine_write_pending_dtoh_copies(conn, nullptr, true) < 0) {
      return -1;
    }
  } else {
    uint32_t copy_count = 0;
    if (rpc_write(conn, &copy_count, sizeof(copy_count)) < 0) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  if (result == CUDA_SUCCESS) {
    lupine_cleanup_pending_dtoh_copies(conn, nullptr, true);
  }
  return 0;
}

int handle_manual_cuStreamWaitEvent(conn_t *conn) {
  CUstream stream = nullptr;
  CUevent event = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto event_it = lupine_event_capture_resource_map().find(event);
  if (event_it != lupine_event_capture_resource_map().end()) {
    auto &stream_resources = lupine_stream_capture_resource_map();
    auto &resources = stream_resources[stream];
    if (!resources) {
      resources = event_it->second;
    }
  }

  result = cuStreamWaitEvent(stream, event, flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamBeginCaptureToGraph(conn_t *conn) {
  CUstream stream = nullptr;
  CUgraph graph = nullptr;
  std::vector<CUgraphNode> deps;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamBeginCaptureToGraph(stream, graph,
                                       deps.empty() ? nullptr : deps.data(),
                                       nullptr, deps.size(), mode);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static CUresult lupine_server_cuStreamUpdateCaptureDependencies(
    CUstream stream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
#if CUDA_VERSION >= 12060
  return cuStreamUpdateCaptureDependencies_v2(
      stream, dependencies, dependencyData, numDependencies, flags);
#else
  if (dependencyData != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuStreamUpdateCaptureDependencies(stream, dependencies,
                                           numDependencies, flags);
#endif
}

int handle_manual_cuStreamUpdateCaptureDependencies(conn_t *conn) {
  CUstream stream = nullptr;
  std::vector<CUgraphNode> deps;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = lupine_server_cuStreamUpdateCaptureDependencies(
      stream, deps.empty() ? nullptr : deps.data(), nullptr, deps.size(),
      flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamGetCaptureInfo(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  const CUgraphNode *deps_ptr = nullptr;
  const CUgraphEdgeData *edge_ptr = nullptr;
  size_t dep_count = 0;
  bool has_edge_data = false;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamGetCaptureInfo_v3(stream, &status, &id, &graph, &deps_ptr,
                                     &edge_ptr, &dep_count);
  has_edge_data = edge_ptr != nullptr && dep_count != 0;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &id, sizeof(id)) < 0 ||
      rpc_write(conn, &graph, sizeof(graph)) < 0 ||
      rpc_write(conn, &dep_count, sizeof(dep_count)) < 0 ||
      rpc_write(conn, &has_edge_data, sizeof(has_edge_data)) < 0 ||
      (dep_count != 0 && deps_ptr != nullptr &&
       rpc_write(conn, deps_ptr, dep_count * sizeof(CUgraphNode)) < 0) ||
      (has_edge_data &&
       rpc_write(conn, edge_ptr, dep_count * sizeof(CUgraphEdgeData)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamBeginCapture(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &mode, sizeof(mode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto &stream_resources = lupine_stream_capture_resource_map();
  auto &resources = stream_resources[stream];
  if (!resources) {
    resources = std::make_shared<lupine_graph_resources>();
  }
  if (resources->capture_scratch == nullptr) {
    static constexpr size_t scratch_size = 128ull * 1024ull * 1024ull;
    void *scratch = nullptr;
    if (cuMemAllocHost(&scratch, scratch_size) == CUDA_SUCCESS) {
      resources->capture_scratch = scratch;
      resources->capture_scratch_size = scratch_size;
      resources->host_allocs.push_back(scratch);
    }
  }

  result = resources->capture_scratch == nullptr
               ? CUDA_ERROR_OUT_OF_MEMORY
               : cuStreamBeginCapture_v2(stream, mode);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuStreamEndCapture(conn_t *conn) {
  CUstream stream = nullptr;
  CUgraph *graph_out = nullptr;
  CUgraph graph = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph_out, sizeof(graph_out)) < 0 ||
      (graph_out != nullptr && rpc_read(conn, &graph, sizeof(graph)) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuStreamEndCapture(stream, &graph);
  if (result == CUDA_SUCCESS) {
    auto &stream_resources = lupine_stream_capture_resource_map();
    auto stream_it = stream_resources.find(stream);
    if (stream_it != stream_resources.end()) {
      lupine_graph_resource_map()[graph] = stream_it->second;
      stream_resources.erase(stream_it);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graph_out, sizeof(graph_out)) < 0 ||
      (graph_out != nullptr && rpc_write(conn, &graph, sizeof(graph)) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphClone(conn_t *conn) {
  CUgraph clone = nullptr;
  CUgraph original = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &clone, sizeof(clone)) < 0 ||
      rpc_read(conn, &original, sizeof(original)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphClone(&clone, original);
  if (result == CUDA_SUCCESS) {
    auto original_it = lupine_graph_resource_map().find(original);
    if (original_it != lupine_graph_resource_map().end()) {
      lupine_graph_resource_map()[clone] = original_it->second;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &clone, sizeof(clone)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphInstantiateWithFlags(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  unsigned long long flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphInstantiateWithFlags(&exec, graph, flags);
  if (result == CUDA_SUCCESS) {
    auto graph_it = lupine_graph_resource_map().find(graph);
    if (graph_it != lupine_graph_resource_map().end()) {
      lupine_graph_exec_resource_map()[exec] = graph_it->second;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphInstantiateWithParams(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  CUDA_GRAPH_INSTANTIATE_PARAMS params = {};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphInstantiateWithParams(&exec, graph, &params);
  if (result == CUDA_SUCCESS) {
    auto graph_it = lupine_graph_resource_map().find(graph);
    if (graph_it != lupine_graph_resource_map().end()) {
      lupine_graph_exec_resource_map()[exec] = graph_it->second;
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      rpc_write(conn, &params, sizeof(params)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecDestroy(conn_t *conn) {
  CUgraphExec exec = nullptr;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphExecDestroy(exec);
  auto exec_resource_it = lupine_graph_exec_resource_map().find(exec);
  if (exec_resource_it != lupine_graph_exec_resource_map().end()) {
    lupine_retire_graph_resources(exec_resource_it->second);
    lupine_graph_exec_resource_map().erase(exec_resource_it);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphDestroy(conn_t *conn) {
  CUgraph graph = nullptr;
  if (rpc_read(conn, &graph, sizeof(graph)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphDestroy(graph);
  auto graph_resource_it = lupine_graph_resource_map().find(graph);
  if (graph_resource_it != lupine_graph_resource_map().end()) {
    lupine_retire_graph_resources(graph_resource_it->second);
    lupine_graph_resource_map().erase(graph_resource_it);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyHtoD_v2(conn_t *conn) {
  CUdeviceptr dstDevice = 0;
  size_t byteCount = 0;
  CUresult result = CUDA_SUCCESS;
  void *host = nullptr;

  if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  size_t chunk_bytes = std::min(LUPINE_HTOD_CHUNK_BYTES, byteCount);
  if (chunk_bytes != 0) {
    result = cuMemAllocHost(&host, chunk_bytes);
  }
  if (result != CUDA_SUCCESS) {
    if (rpc_drain(conn, byteCount) < 0) {
      return -1;
    }
  }

  size_t offset = 0;
  while (result == CUDA_SUCCESS && offset < byteCount) {
    size_t chunk = std::min(chunk_bytes, byteCount - offset);
    if (rpc_read(conn, host, chunk) < 0) {
      if (host != nullptr) {
        cuMemFreeHost(host);
      }
      return -1;
    }
    result = cuMemcpyHtoD_v2(dstDevice + offset, host, chunk);
    offset += chunk;
    if (result != CUDA_SUCCESS && rpc_drain(conn, byteCount - offset) < 0) {
      cuMemFreeHost(host);
      return -1;
    }
  }

  if (host != nullptr) {
    cuMemFreeHost(host);
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyHtoDAsync_v2(conn_t *conn) {
  CUdeviceptr dstDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  void *capture_host = nullptr;

  if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto &stream_resources = lupine_stream_capture_resource_map();
    auto &resources = stream_resources[stream];
    if (!resources) {
      resources = std::make_shared<lupine_graph_resources>();
    }
    capture_host = lupine_alloc_capture_scratch(resources, byteCount);
    if (capture_host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_drain(conn, byteCount) < 0) {
        return -1;
      }
    } else {
      if (byteCount != 0 && rpc_read(conn, capture_host, byteCount) < 0) {
        return -1;
      }
    }
  } else {
    result = CUDA_SUCCESS;
    void *host = nullptr;
    if (byteCount != 0) {
      result = cuMemAllocHost(&host, byteCount);
    }
    if (result != CUDA_SUCCESS) {
      if (rpc_drain(conn, byteCount) < 0) {
        return -1;
      }
    }
    size_t offset = 0;
    while (result == CUDA_SUCCESS && offset < byteCount) {
      size_t chunk = std::min(LUPINE_HTOD_CHUNK_BYTES, byteCount - offset);
      auto *chunk_host = static_cast<unsigned char *>(host) + offset;
      if (rpc_read(conn, chunk_host, chunk) < 0) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
        return -1;
      }

      CUresult copy_result =
          cuMemcpyHtoDAsync_v2(dstDevice + offset, chunk_host, chunk, stream);
      if (copy_result != CUDA_SUCCESS) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
        result = copy_result;
        offset += chunk;
        if (rpc_drain(conn, byteCount - offset) < 0) {
          return -1;
        }
        host = nullptr;
        break;
      }
      offset += chunk;
    }
    if (host != nullptr && result == CUDA_SUCCESS) {
      result = cuLaunchHostFunc(stream,
                                [](void *userData) {
                                  cuMemFreeHost(userData);
                                },
                                host);
      if (result != CUDA_SUCCESS) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
      }
    }
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE &&
      result != CUDA_ERROR_OUT_OF_MEMORY) {
    result = cuMemcpyHtoDAsync_v2(dstDevice, capture_host, byteCount, stream);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemcpyDtoHAsync_v2(conn_t *conn) {
  void *dstHost = nullptr;
  CUdeviceptr srcDevice = 0;
  size_t byteCount = 0;
  CUstream stream = nullptr;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }

  void *host = nullptr;
  CUresult alloc_result = CUDA_ERROR_INVALID_VALUE;
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto &stream_resources = lupine_stream_capture_resource_map();
    auto &resources = stream_resources[stream];
    if (!resources) {
      resources = std::make_shared<lupine_graph_resources>();
    }
    host = lupine_alloc_capture_scratch(resources, byteCount);
    if (host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS) {
        resources->dtoh_copies.push_back({dstHost, host, byteCount});
      }
      host = nullptr;
    }
  } else {
    auto &pending = lupine_pending_dtoh_copies()[conn];
    bool reused_pending_host = false;
    for (auto &copy : pending) {
      if (copy.client_dst == dstHost && copy.bytes == byteCount) {
        host = copy.server_src;
        reused_pending_host = true;
        break;
      }
    }
    if (!reused_pending_host) {
      alloc_result = cuMemAllocHost(&host, byteCount);
      if (alloc_result != CUDA_SUCCESS) {
        host = byteCount == 0 ? nullptr : malloc(byteCount);
      }
    }
    if (byteCount != 0 && host == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (reused_pending_host) {
        host = nullptr;
      }
      if (result == CUDA_SUCCESS && byteCount != 0 && !reused_pending_host) {
        pending.push_back(
            {stream, dstHost, host, byteCount, alloc_result == CUDA_SUCCESS});
        host = nullptr;
      }
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    if (alloc_result == CUDA_SUCCESS && host != nullptr) {
      cuMemFreeHost(host);
    } else if (host != nullptr) {
      free(host);
    }
    return -1;
  }

  if (alloc_result == CUDA_SUCCESS && host != nullptr) {
    cuMemFreeHost(host);
  } else if (host != nullptr) {
    free(host);
  }
  return 0;
}

int handle_manual_cuCtxSynchronize(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  CUresult result = cuCtxSynchronize();
  lupine_finish_stdout_capture(&capture);
  uint64_t stdout_size = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_pending_dtoh_copies(conn, nullptr, true) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(conn, nullptr, true);
  return 0;
}

int handle_manual_cuStreamSynchronize(conn_t *conn) {
  CUstream stream = nullptr;
  if (rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  CUresult result = cuStreamSynchronize(stream);
  lupine_finish_stdout_capture(&capture);
  std::shared_ptr<lupine_graph_resources> resources;
  uint32_t copy_count = 0;
  auto stream_it = lupine_stream_capture_resource_map().find(stream);
  if (stream_it != lupine_stream_capture_resource_map().end()) {
    resources = stream_it->second;
  }
  uint32_t graph_copy_count =
      resources == nullptr
          ? 0
          : static_cast<uint32_t>(resources->dtoh_copies.size());
  uint32_t pending_copy_count =
      lupine_count_pending_dtoh_copies(conn, nullptr, true);
  copy_count = graph_copy_count + pending_copy_count;
  uint64_t stdout_size = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &copy_count, sizeof(copy_count)) < 0 ||
      (resources != nullptr &&
       std::any_of(resources->dtoh_copies.begin(), resources->dtoh_copies.end(),
                   [&](const lupine_graph_host_copy &copy) {
                     return rpc_write(conn, &copy.client_dst,
                                      sizeof(copy.client_dst)) < 0 ||
                            rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) <
                                0 ||
                            (copy.bytes != 0 &&
                             rpc_write(conn, copy.server_src, copy.bytes) < 0);
                   })) ||
      lupine_write_pending_dtoh_copies(conn, nullptr, true, false) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(conn, nullptr, true);
  return 0;
}

int handle_manual_cuGraphLaunch(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUstream stream = nullptr;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuGraphLaunch(exec, stream);
  auto exec_it = lupine_graph_exec_resource_map().find(exec);
  if (result == CUDA_SUCCESS &&
      exec_it != lupine_graph_exec_resource_map().end()) {
    lupine_stream_capture_resource_map()[stream] = exec_it->second;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuEventSynchronize(conn_t *conn) {
  CUevent event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  lupine_captured_stdout capture;
  lupine_start_stdout_capture(&capture);
  CUresult result = cuEventSynchronize(event);
  lupine_finish_stdout_capture(&capture);
  uint64_t stdout_size = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_pending_dtoh_copies(conn, nullptr, true) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(conn, nullptr, true);
  return 0;
}

int handle_manual_cuOccupancyMaxPotentialBlockSize(conn_t *conn,
                                                   bool with_flags) {
  CUfunction func = nullptr;
  size_t dynamicSMemSize = 0;
  int blockSizeLimit = 0;
  unsigned int flags = 0;
  int request_id;
  int minGridSize = 0;
  int blockSize = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &func, sizeof(func)) < 0 ||
      rpc_read(conn, &dynamicSMemSize, sizeof(dynamicSMemSize)) < 0 ||
      rpc_read(conn, &blockSizeLimit, sizeof(blockSizeLimit)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (with_flags) {
    result = cuOccupancyMaxPotentialBlockSizeWithFlags(
        &minGridSize, &blockSize, func, nullptr, dynamicSMemSize,
        blockSizeLimit, flags);
  } else {
    result = cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func,
                                              nullptr, dynamicSMemSize,
                                              blockSizeLimit);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &minGridSize, sizeof(minGridSize)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(blockSize)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
