#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <errno.h>
#if defined(__linux__)
#include <sys/mman.h> // memfd_create
#endif
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <vector>

#include <list>
#include <map>

#include "cuda_compat.h"

#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "lupine_attr_sizes.h"
#include "lupine_fatbin.h"
#include "lupine_log.h"
#include "manual_server.h"
#include "rpc.h"
#include "server_registry.h"

#if CUDA_VERSION < 12020
#ifdef CU_MEM_LOCATION_TYPE_HOST
static constexpr CUmemLocationType LUPINE_CU_MEM_LOCATION_TYPE_HOST =
    CU_MEM_LOCATION_TYPE_HOST;
#else
static constexpr CUmemLocationType LUPINE_CU_MEM_LOCATION_TYPE_HOST =
    static_cast<CUmemLocationType>(2);
#endif
#endif

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags,
                                           CUdevice dev);

static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;
static constexpr size_t LUPINE_HTOD_CHUNK_BYTES = 64 * 1024 * 1024;

// cuMemAllocHost / cuMemFreeHost page-lock and unlock host memory on every
// call, which costs on the order of a millisecond even for a tiny transfer and
// dominates the latency of small, frequent copies. The server forks one
// process per connection and runs handlers serially on that connection, so a
// single process-global staging buffer can be reused across memcpy calls.
// Buffers up to LUPINE_STAGING_RETAIN_BYTES are kept page-locked and reused;
// larger requests fall back to a per-call allocation (whose cost is amortized
// over the large transfer) so a process never pins a huge buffer indefinitely.
static constexpr size_t LUPINE_STAGING_RETAIN_BYTES = 8 * 1024 * 1024;

static lupine_handle_registry<CUlibrary, std::vector<unsigned char>> &
lupine_preserved_library_images() {
  static lupine_handle_registry<CUlibrary, std::vector<unsigned char>> images;
  return images;
}

static lupine_handle_registry<CUlinkState,
                              std::shared_ptr<rpc_jit_server_state>> &
lupine_jit_server_states() {
  static lupine_handle_registry<CUlinkState,
                                std::shared_ptr<rpc_jit_server_state>>
      states;
  return states;
}

struct lupine_staging {
  void *ptr = nullptr;
  bool owned = false; // true => caller must release (a per-call allocation)
  bool pinned = false;
};

// Returns a host buffer of at least `bytes`. On success ptr != nullptr; when
// owned is true the caller must release it via lupine_release_staging, when
// false it borrows the retained buffer and must not free it.
static lupine_staging lupine_acquire_staging(size_t bytes) {
  lupine_staging out;
  if (bytes == 0) {
    return out;
  }
  if (bytes > LUPINE_STAGING_RETAIN_BYTES) {
    if (cuMemAllocHost(&out.ptr, bytes) == CUDA_SUCCESS) {
      out.owned = true;
      out.pinned = true;
    } else if ((out.ptr = malloc(bytes)) != nullptr) {
      out.owned = true;
    }
    return out;
  }
  static void *retained = nullptr;
  static size_t retained_size = 0;
  if (retained_size < bytes) {
    void *grown = nullptr;
    if (cuMemAllocHost(&grown, bytes) != CUDA_SUCCESS) {
      return out;
    }
    if (retained != nullptr) {
      cuMemFreeHost(retained);
    }
    retained = grown;
    retained_size = bytes;
  }
  out.ptr = retained;
  out.pinned = true;
  return out;
}

static void lupine_release_staging(const lupine_staging &s) {
  if (s.ptr != nullptr && s.owned) {
    if (s.pinned) {
      cuMemFreeHost(s.ptr);
    } else {
      free(s.ptr);
    }
  }
}

struct lupine_captured_stdout {
  int saved_stdout = -1;
  bool active = false;
  std::string output;
};

static pthread_mutex_t lupine_stdout_capture_mutex = PTHREAD_MUTEX_INITIALIZER;

// Device printf output is drained by the CUDA driver as a write to fd 1
// (process stdout) during synchronization (see issue #294). We capture it by
// temporarily redirecting fd 1 to a backing file we can read back. The lupine
// server writes all of its own diagnostics to stderr, so fd 1 is exclusively
// the device-printf channel and nothing else can contaminate the capture.
//
// This returns a single process-global, reusable backing file: created once
// on first use and kept open for the process lifetime, so the per-synchronize
// hot path performs no filesystem open/close. On Linux it is an anonymous
// in-memory file from memfd_create() (no path, no /tmp, no inode, no page
// cache of a real file); other platforms (and old kernels without memfd)
// fall back to a single tmpfile() created once. The file is reset (truncated
// to empty) at the start of each capture.
static FILE *lupine_stdout_capture_file() {
  static FILE *file = []() -> FILE * {
#if defined(__linux__)
    int fd = memfd_create("lupine-stdout-capture", MFD_CLOEXEC);
    if (fd >= 0) {
      FILE *f = fdopen(fd, "w+");
      if (f != nullptr) {
        return f;
      }
      // fdopen failed; reclaim the fd and fall through to tmpfile().
      close(fd);
    }
#endif
    return tmpfile();
  }();
  return file;
}

static bool lupine_start_stdout_capture(lupine_captured_stdout *capture) {
  if (capture == nullptr) {
    return false;
  }
  capture->saved_stdout = -1;
  capture->active = false;
  capture->output.clear();

  FILE *capture_file = lupine_stdout_capture_file();
  if (capture_file == nullptr) {
    return false;
  }
  int capture_fd = lupine_fd_fileno(capture_file);
  if (capture_fd < 0) {
    return false;
  }

  if (pthread_mutex_lock(&lupine_stdout_capture_mutex) != 0) {
    return false;
  }

  fflush(stdout);
  std::cout.flush();

  // Reset the reused backing file to empty so this capture only contains
  // output produced during the synchronization below.
  if (lupine_fd_truncate(capture_fd, 0) != 0 ||
      lupine_fd_seek(capture_fd, 0, SEEK_SET) < 0) {
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  capture->saved_stdout = lupine_fd_dup(LUPINE_STDOUT_FD);
  if (capture->saved_stdout < 0) {
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  if (lupine_fd_dup2(capture_fd, LUPINE_STDOUT_FD) < 0) {
    lupine_fd_close(capture->saved_stdout);
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
  lupine_fd_dup2(capture->saved_stdout, LUPINE_STDOUT_FD);
  lupine_fd_close(capture->saved_stdout);
  capture->saved_stdout = -1;

  // The backing file is process-global and reused, so read it back without
  // closing it. Its extent is exactly the bytes written during this capture
  // (it was truncated to empty on entry).
  FILE *capture_file = lupine_stdout_capture_file();
  if (capture_file != nullptr) {
    int capture_fd = lupine_fd_fileno(capture_file);
    if (capture_fd >= 0 && lupine_fd_seek(capture_fd, 0, SEEK_SET) >= 0) {
      char buffer[4096];
      for (;;) {
        ssize_t bytes = lupine_fd_read(capture_fd, buffer, sizeof(buffer));
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

static lupine_handle_registry<CUmodule, CUlibrary> &lupine_module_libraries() {
  static lupine_handle_registry<CUmodule, CUlibrary> libraries;
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
  mutable std::mutex mutex;
  std::vector<CUdeviceptr> device_allocs;
  std::vector<void *> host_allocs;
  std::vector<void *> pageable_allocs;
  std::vector<void *> callback_allocs;
  std::vector<lupine_graph_host_copy> dtoh_copies;
  void *capture_scratch = nullptr;
  size_t capture_scratch_size = 0;
  size_t capture_scratch_offset = 0;

  void add_host_allocation(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    host_allocs.push_back(ptr);
  }

  void add_pageable_allocation(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    pageable_allocs.push_back(ptr);
  }

  void add_callback(void *callback) {
    std::lock_guard<std::mutex> lock(mutex);
    callback_allocs.push_back(callback);
  }

  void add_dtoh_copy(lupine_graph_host_copy copy) {
    std::lock_guard<std::mutex> lock(mutex);
    dtoh_copies.push_back(copy);
  }

  std::vector<lupine_graph_host_copy> dtoh_copy_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return dtoh_copies;
  }

  bool has_capture_scratch() const {
    std::lock_guard<std::mutex> lock(mutex);
    return capture_scratch != nullptr;
  }

  bool install_capture_scratch(void *scratch, size_t size) {
    if (scratch == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex);
    if (capture_scratch != nullptr) {
      return false;
    }
    capture_scratch = scratch;
    capture_scratch_size = size;
    host_allocs.push_back(scratch);
    return true;
  }

  void *allocate_capture_scratch(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex);
    size_t offset = (capture_scratch_offset + 255) & ~size_t(255);
    if (capture_scratch == nullptr || offset > capture_scratch_size ||
        bytes > capture_scratch_size - offset) {
      return nullptr;
    }
    auto *ptr = static_cast<unsigned char *>(capture_scratch) + offset;
    capture_scratch_offset = offset + bytes;
    return ptr;
  }

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

static lupine_handle_registry<CUgraph,
                              std::shared_ptr<lupine_graph_resources>> &
lupine_graph_resource_map() {
  static lupine_handle_registry<CUgraph,
                                std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static lupine_handle_registry<CUgraphExec,
                              std::shared_ptr<lupine_graph_resources>> &
lupine_graph_exec_resource_map() {
  static lupine_handle_registry<CUgraphExec,
                                std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static lupine_handle_registry<CUstream,
                              std::shared_ptr<lupine_graph_resources>> &
lupine_stream_capture_resource_map() {
  static lupine_handle_registry<CUstream,
                                std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static lupine_handle_registry<CUevent,
                              std::shared_ptr<lupine_graph_resources>> &
lupine_event_capture_resource_map() {
  static lupine_handle_registry<CUevent,
                                std::shared_ptr<lupine_graph_resources>>
      resources;
  return resources;
}

static lupine_retirement_list<std::shared_ptr<lupine_graph_resources>> &
lupine_retired_graph_resources() {
  static auto *retired =
      new lupine_retirement_list<std::shared_ptr<lupine_graph_resources>>;
  return *retired;
}

static void lupine_retire_graph_resources(
    const std::shared_ptr<lupine_graph_resources> &resources) {
  if (!resources) {
    return;
  }
  lupine_retired_graph_resources().retire(resources);
}

using lupine_pending_dtoh_streams =
    std::unordered_map<CUstream, std::vector<lupine_pending_dtoh_copy>>;

// Detaching one stream can also remove its connection entry, so these nested
// updates use one explicitly lock-owned registry rather than independent map
// operations.
struct lupine_pending_dtoh_registry {
  std::mutex mutex;
  std::unordered_map<conn_t *, lupine_pending_dtoh_streams> copies;
};

static lupine_pending_dtoh_registry &lupine_pending_dtoh_copies() {
  static lupine_pending_dtoh_registry registry;
  return registry;
}

static std::shared_ptr<lupine_graph_resources>
lupine_get_graph_resources(CUgraph graph) {
  return lupine_graph_resource_map().get_or_create(
      graph, []() { return std::make_shared<lupine_graph_resources>(); });
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
#ifdef _WIN32
  MEMORY_BASIC_INFORMATION info = {};
  if (VirtualQuery(fn, &info, sizeof(info)) == 0 ||
      info.AllocationBase == nullptr) {
    return 0;
  }
#else
  Dl_info info = {};
  if (dladdr(fn, &info) == 0 || info.dli_fname == nullptr) {
    return 0;
  }
#endif
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

  LUPINE_TRACE_LOG("LUPINE server cuGetExportTable metadata result="
                   << result << " bytes=" << byte_size
                   << " slots=" << slot_count << " trusted=" << trusted);

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
      LUPINE_TRACE_LOG("LUPINE server private module node module="
                       << module << " context=" << context << " count=" << count
                       << " node=" << node
                       << " owner=" << reinterpret_cast<void *>(owner)
                       << " result=" << static_cast<int>(result));
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
      resources->add_pageable_allocation(ptr);
    }
    return ptr;
  }
  if (ptr != nullptr) {
    resources->add_host_allocation(ptr);
  }
  return ptr;
}

static std::vector<lupine_pending_dtoh_copy>
lupine_detach_pending_dtoh_copies(conn_t *conn, CUstream stream,
                                  bool all_streams) {
  std::vector<lupine_pending_dtoh_copy> copies;
  auto &registry = lupine_pending_dtoh_copies();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto conn_it = registry.copies.find(conn);
  if (conn_it == registry.copies.end()) {
    return copies;
  }
  auto &streams = conn_it->second;
  if (all_streams) {
    for (auto &entry : streams) {
      auto &stream_copies = entry.second;
      copies.insert(copies.end(), stream_copies.begin(), stream_copies.end());
    }
    registry.copies.erase(conn_it);
    return copies;
  }

  auto stream_it = streams.find(stream);
  if (stream_it != streams.end()) {
    copies.swap(stream_it->second);
    streams.erase(stream_it);
    if (streams.empty()) {
      registry.copies.erase(conn_it);
    }
  }
  return copies;
}

static int lupine_write_pending_dtoh_copies(
    uint32_t *copy_count, conn_t *conn,
    const std::vector<lupine_pending_dtoh_copy> &pending) {
  if (copy_count != nullptr) {
    *copy_count = static_cast<uint32_t>(pending.size());
    if (rpc_write(conn, copy_count, sizeof(*copy_count)) < 0) {
      return -1;
    }
  }
  for (const auto &copy : pending) {
    if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
        rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
        (copy.bytes != 0 &&
         rpc_write_payload(conn, copy.server_src, copy.bytes) < 0)) {
      return -1;
    }
  }
  return 0;
}

static void lupine_cleanup_pending_dtoh_copies(
    std::vector<lupine_pending_dtoh_copy> *pending) {
  if (pending == nullptr) {
    return;
  }
  for (auto &copy : *pending) {
    if (copy.server_src != nullptr) {
      if (copy.pinned) {
        cuMemFreeHost(copy.server_src);
      } else {
        free(copy.server_src);
      }
      copy.server_src = nullptr;
    }
  }
  pending->clear();
}

static void *lupine_alloc_capture_scratch(
    const std::shared_ptr<lupine_graph_resources> &resources, size_t bytes) {
  if (!resources || bytes == 0) {
    return nullptr;
  }
  return resources->allocate_capture_scratch(bytes);
}

int handle_manual_cuModuleLoad(conn_t *conn) {
  CUmodule module = nullptr;
  size_t image_size = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size + 1, 0);
  if (image_size == 0 || rpc_read_payload(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuModuleLoadData(&module, image.data());

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
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
  if (image_size == 0 || rpc_read_payload(conn, image.data(), image_size) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (kind == LUPINE_MODULE_IMAGE_FATBINC_V1 ||
      kind == LUPINE_MODULE_IMAGE_FATBINC_V2) {
    result = cuModuleLoadFatBinary(&module, image.data());
  } else if (kind == LUPINE_MODULE_IMAGE_FATBIN_RAW) {
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
  rpc_jit_server_state jit_state;
  bool has_library_option_values = false;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0) {
    return -1;
  }

  std::vector<unsigned char> image(image_size);
  if (image_size == 0 || rpc_read_payload(conn, image.data(), image_size) < 0) {
    return -1;
  }
  if (rpc_read_jit_options(conn, &jit_state) < 0) {
    return -1;
  }
  std::vector<CUlibraryOption> library_options;
  std::vector<uintptr_t> library_raw_values;
  if (rpc_read_library_options(conn, &library_options, &library_raw_values,
                               &has_library_option_values) < 0) {
    return -1;
  }
  unsigned int num_library_options =
      static_cast<unsigned int>(library_options.size());
  std::vector<void *> library_option_values(num_library_options);
  for (unsigned int i = 0; i < num_library_options; ++i) {
    library_option_values[i] = reinterpret_cast<void *>(library_raw_values[i]);
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUjit_option *jit_opts =
      jit_state.options.empty() ? nullptr : jit_state.options.data();
  void **jit_vals = jit_state.option_values.empty()
                        ? nullptr
                        : jit_state.option_values.data();
  CUlibraryOption *lib_opts =
      library_options.empty() ? nullptr : library_options.data();
  void **lib_vals = !has_library_option_values || library_option_values.empty()
                        ? nullptr
                        : library_option_values.data();
  unsigned int num_jit_options =
      static_cast<unsigned int>(jit_state.options.size());

  if (kind == LUPINE_MODULE_IMAGE_FATBINC_V1 ||
      kind == LUPINE_MODULE_IMAGE_FATBINC_V2) {
    lupine_fatbin_wrapper wrapper = {
        LUPINE_FATBINC_MAGIC,
        kind == LUPINE_MODULE_IMAGE_FATBINC_V2 ? 2U : 1U,
        image.data(),
        nullptr,
    };
    result = cuLibraryLoadData(&library, &wrapper, jit_opts, jit_vals,
                               num_jit_options, lib_opts, lib_vals,
                               num_library_options);
  } else if (kind == LUPINE_MODULE_IMAGE_FATBIN_RAW) {
    result = cuLibraryLoadData(&library, image.data(), jit_opts, jit_vals,
                               num_jit_options, lib_opts, lib_vals,
                               num_library_options);
  } else {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (result == CUDA_SUCCESS) {
    lupine_preserved_library_images().store(library, std::move(image));
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      rpc_write_jit_outputs(conn, &jit_state) < 0 ||
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
    if (!lupine_pointer_attribute_size(attributes[i], &value_size)) {
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
  if (result != CUDA_SUCCESS) {
    std::fill(value_sizes.begin(), value_sizes.end(), 0);
  }

  if (rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_attributes; ++i) {
    if (rpc_write(conn, &value_sizes[i], sizeof(value_sizes[i])) < 0 ||
        (value_sizes[i] != 0 &&
         rpc_write(conn, values[i].data(), value_sizes[i]) < 0)) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPrefetchAsync(conn_t *conn) {
  CUdeviceptr devPtr;
  size_t count;
  int location_type;
  int location_id;
  unsigned int flags;
  CUstream hStream;
  int request_id;
  CUresult result;
  if (rpc_read(conn, &devPtr, sizeof(devPtr)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      rpc_read(conn, &location_type, sizeof(location_type)) < 0 ||
      rpc_read(conn, &location_id, sizeof(location_id)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &hStream, sizeof(hStream)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUmemLocation location = {};
  location.type = static_cast<CUmemLocationType>(location_type);
  location.id = location_id;
#if CUDA_VERSION >= 12020
  result = cuMemPrefetchAsync_v2(devPtr, count, location, flags, hStream);
#else
  if (flags != 0 || (location.type != CU_MEM_LOCATION_TYPE_DEVICE &&
                     location.type != LUPINE_CU_MEM_LOCATION_TYPE_HOST)) {
    result = CUDA_ERROR_INVALID_VALUE;
  } else {
    CUdevice dstDevice = location.type == CU_MEM_LOCATION_TYPE_DEVICE
                             ? location.id
                             : CU_DEVICE_CPU;
    result = cuMemPrefetchAsync(devPtr, count, dstDevice, hStream);
  }
#endif

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuLinkCreate_v2(conn_t *conn) {
  auto jit_state = std::make_shared<rpc_jit_server_state>();
  CUlinkState state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read_jit_options(conn, jit_state.get()) < 0) {
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
    lupine_jit_server_states().store(state, jit_state);
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
  rpc_jit_server_state jit_state;
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
      rpc_read_jit_options(conn, &jit_state) < 0) {
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
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_jit_outputs(conn, &jit_state) < 0 ||
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
  rpc_jit_server_state jit_state;
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
      file_size > (1ull << 32) || (file_size != 0 && has_file_data == 0)) {
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
  if (rpc_read_jit_options(conn, &jit_state) < 0) {
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
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write_jit_outputs(conn, &jit_state) < 0 ||
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
  std::shared_ptr<rpc_jit_server_state> stored_jit_state;
  lupine_jit_server_states().lookup(state, &stored_jit_state);
  rpc_jit_server_state empty_jit_state;
  rpc_jit_server_state *jit_state =
      stored_jit_state == nullptr ? &empty_jit_state : stored_jit_state.get();
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_size, sizeof(returned_size)) < 0 ||
      (returned_size != 0 && rpc_write(conn, cubin, returned_size) < 0) ||
      rpc_write_jit_outputs(conn, jit_state) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
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
  lupine_jit_server_states().remove(state);
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
    lupine_module_libraries().store(module, library);
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
  std::vector<unsigned char> retired_image;
  lupine_preserved_library_images().remove(library, &retired_image);

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
    CUlibrary library = nullptr;
    if (lupine_module_libraries().lookup(module, &library)) {
      CUdeviceptr library_dptr = 0;
      size_t library_bytes = 0;
      CUresult library_result = cuLibraryGetGlobal(
          &library_dptr, &library_bytes, library, name.data());
      if (library_result == CUDA_SUCCESS) {
        dptr = library_dptr;
        bytes = library_bytes;
        result = library_result;
      }
    }
  }
  LUPINE_TRACE_LOG("LUPINE cuModuleGetGlobal name=" << name.data() << " result="
                                                    << static_cast<int>(result)
                                                    << " bytes=" << bytes);

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
  CUcontext ctx = nullptr;
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
      rpc_read(conn, &ctx, sizeof(ctx)) < 0 ||
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

  if (ctx != nullptr) {
    CUcontext previous = nullptr;
    result = cuCtxGetCurrent(&previous);
    if (result == CUDA_SUCCESS && previous != ctx) {
      result = cuCtxSetCurrent(ctx);
    }
  } else {
    result = CUDA_SUCCESS;
  }

  lupine_kernel_param_layout layout;
  if (result == CUDA_SUCCESS) {
    result = lupine_get_kernel_param_layout(f, &layout);
  }
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

  // Fire-and-forget: no response is sent.
  (void)request_id;
  (void)result;
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

  std::vector<lupine_graph_host_copy> copies =
      callback->resources ? callback->resources->dtoh_copy_snapshot()
                          : std::vector<lupine_graph_host_copy>();
  int transfer_count = static_cast<int>(copies.size());

  conn_t *conn = callback->conn;
  if (rpc_write_start_request(conn, 1) < 0 ||
      rpc_write(conn, &transfer_count, sizeof(transfer_count)) < 0) {
    return;
  }
  for (const auto &copy : copies) {
    if (rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) < 0 ||
        rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
        rpc_write_payload(conn, copy.server_src, copy.bytes) < 0) {
      return;
    }
  }
  CUhostFn fn = callback->fn;
  void *client_user_data = callback->userData;
  void *response = nullptr;
  if (rpc_write(conn, &fn, sizeof(fn)) < 0 ||
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
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
  uint32_t copy_count = 0;
  auto pending = lupine_detach_pending_dtoh_copies(conn, stream, false);
  if (rpc_write_start_request(conn, 2) >= 0 &&
      lupine_write_pending_dtoh_copies(&copy_count, conn, pending) >= 0 &&
      rpc_write(conn, &stream, sizeof(stream)) >= 0 &&
      rpc_write(conn, &status, sizeof(status)) >= 0 &&
      rpc_write(conn, &fn, sizeof(fn)) >= 0 &&
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) >= 0 &&
      rpc_wait_for_response(conn) >= 0) {
    rpc_read(conn, &response, sizeof(response));
    rpc_read_end(conn);
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
  delete callback;
}

struct lupine_kernel_param_payload {
  std::vector<unsigned char> storage;
  std::vector<void *> params;
};

static CUresult
lupine_read_kernel_param_values(conn_t *conn,
                                const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                uint32_t paramCount, size_t payloadSize,
                                lupine_kernel_param_payload *payload);

int handle_manual_cuGraphAddKernelNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  result = lupine_read_kernel_param_values(conn, nodeParams, param_count,
                                           payload_size, &payload);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    nodeParams.kernelParams =
        payload.params.empty() ? nullptr : payload.params.data();
    nodeParams.extra = nullptr;
    result = cuGraphAddKernelNode_v2(&graphNode, hGraph,
                                     deps.empty() ? nullptr : deps.data(),
                                     deps.size(), &nodeParams);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static CUfunction
lupine_kernel_node_function(const CUDA_KERNEL_NODE_PARAMS &params) {
  CUfunction func = params.func;
#if CUDA_VERSION >= 12000
  if (func == nullptr) {
    func = reinterpret_cast<CUfunction>(params.kern);
  }
#endif
  return func;
}

static CUresult
lupine_prepare_server_kernel_params(const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                    CUDA_KERNEL_NODE_PARAMS *serialParams,
                                    lupine_kernel_param_layout *layout,
                                    size_t *payloadSize) {
  if (serialParams == nullptr || layout == nullptr || payloadSize == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *layout = {};
  *payloadSize = 0;
  if (nodeParams.extra != nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUfunction func = lupine_kernel_node_function(nodeParams);
  CUresult result = lupine_get_kernel_param_layout(func, layout);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (layout->count > 64) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (layout->count != 0 && nodeParams.kernelParams == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  for (uint32_t i = 0; i < layout->count; ++i) {
    if (nodeParams.kernelParams[i] == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  *serialParams = nodeParams;
  serialParams->kernelParams = nullptr;
  serialParams->extra = nullptr;
  for (uint32_t i = 0; i < layout->count; ++i) {
    *payloadSize += layout->sizes[i];
  }
  return CUDA_SUCCESS;
}

static int
lupine_write_kernel_param_values(conn_t *conn,
                                 const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                 const lupine_kernel_param_layout &layout) {
  return rpc_write_kernel_param_values(conn, layout.count, layout.sizes,
                                       nodeParams.kernelParams);
}

static CUresult
lupine_read_kernel_param_values(conn_t *conn,
                                const CUDA_KERNEL_NODE_PARAMS &nodeParams,
                                uint32_t paramCount, size_t payloadSize,
                                lupine_kernel_param_payload *payload) {
  if (conn == nullptr || payload == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  payload->storage.clear();
  payload->params.clear();

  CUfunction func = lupine_kernel_node_function(nodeParams);
  lupine_kernel_param_layout layout;
  CUresult result = lupine_get_kernel_param_layout(func, &layout);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  if (layout.count != paramCount) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  size_t expected_payload_size = 0;
  size_t storage_size = 0;
  for (uint32_t i = 0; i < layout.count; ++i) {
    expected_payload_size += layout.sizes[i];
    storage_size = std::max(storage_size, layout.offsets[i] + layout.sizes[i]);
  }
  if (payloadSize != expected_payload_size) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  payload->storage.assign(storage_size, 0);
  payload->params.resize(paramCount);
  if (rpc_read_kernel_param_values(
          conn, layout.count, layout.offsets, layout.sizes, payloadSize,
          payload->storage.data(), payload->storage.size(),
          payload->params.data()) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

int handle_manual_cuGraphKernelNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  int request_id;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  CUDA_KERNEL_NODE_PARAMS serialParams = {};
  lupine_kernel_param_layout layout = {};
  size_t payloadSize = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphKernelNodeGetParams_v2(hNode, &nodeParams);
  if (result == CUDA_SUCCESS) {
    serialParams = nodeParams;
    serialParams.kernelParams = nullptr;
    serialParams.extra = nullptr;
    result = lupine_prepare_server_kernel_params(nodeParams, &serialParams,
                                                 &layout, &payloadSize);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &serialParams, sizeof(serialParams)) < 0 ||
      rpc_write(conn, &layout, sizeof(layout)) < 0 ||
      rpc_write(conn, &payloadSize, sizeof(payloadSize)) < 0 ||
      (result == CUDA_SUCCESS &&
       lupine_write_kernel_param_values(conn, nodeParams, layout) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphKernelNodeSetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  uint32_t param_count = 0;
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  result = lupine_read_kernel_param_values(conn, nodeParams, param_count,
                                           payload_size, &payload);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    nodeParams.kernelParams =
        payload.params.empty() ? nullptr : payload.params.data();
    nodeParams.extra = nullptr;
    result = cuGraphKernelNodeSetParams_v2(hNode, &nodeParams);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
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
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  result = lupine_read_kernel_param_values(conn, nodeParams, param_count,
                                           payload_size, &payload);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    nodeParams.kernelParams =
        payload.params.empty() ? nullptr : payload.params.data();
    nodeParams.extra = nullptr;
    result = cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, &nodeParams);
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
    resources->add_dtoh_copy({copyParams.dstHost, host, host_dst_bytes});
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
  callback->resources->add_callback(callback);

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
  size_t payload_size = 0;
  lupine_kernel_param_payload payload;
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      rpc_read(conn, &param_count, sizeof(param_count)) < 0 ||
      rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    result = lupine_read_kernel_param_values(
        conn, *reinterpret_cast<CUDA_KERNEL_NODE_PARAMS *>(&nodeParams.kernel),
        param_count, payload_size, &payload);
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraph> child_graphs;
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    if (result == CUDA_SUCCESS) {
      nodeParams.kernel.kernelParams =
          payload.params.empty() ? nullptr : payload.params.data();
      nodeParams.kernel.extra = nullptr;
      result = lupine_server_cuGraphAddNode(
          &graphNode, hGraph, deps.empty() ? nullptr : deps.data(), nullptr,
          deps.size(), &nodeParams);
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
  LUPINE_TRACE_LOG("LUPINE cuGraphAddNode type="
                   << nodeParams.type << " param_count=" << param_count
                   << " payload_size=" << payload_size << " graph=" << hGraph
                   << " node=" << graphNode << " result=" << result);

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

static int lupine_handle_node_dependency_query(conn_t *conn, bool dependent) {
  CUgraphNode hNode = nullptr;
  size_t requested = 0;
  uint8_t want_edge = 0;
  size_t count = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &requested, sizeof(requested)) < 0 ||
      rpc_read(conn, &want_edge, sizeof(want_edge)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraphNode> nodes;
#if CUDA_VERSION >= 12030
  std::vector<CUgraphEdgeData> edges;
  auto call = [&](CUgraphNode *out, CUgraphEdgeData *edge,
                  size_t *n) -> CUresult {
    return dependent ? cuGraphNodeGetDependentNodes_v2(hNode, out, edge, n)
                     : cuGraphNodeGetDependencies_v2(hNode, out, edge, n);
  };
  if (requested == 0) {
    result = call(nullptr, nullptr, &count);
  } else {
    count = requested;
    nodes.resize(count);
    if (want_edge) {
      edges.resize(count);
    }
    result = call(nodes.data(), want_edge ? edges.data() : nullptr, &count);
  }
#else
  auto call = [&](CUgraphNode *out, size_t *n) -> CUresult {
    return dependent ? cuGraphNodeGetDependentNodes(hNode, out, n)
                     : cuGraphNodeGetDependencies(hNode, out, n);
  };
  if (requested == 0) {
    result = call(nullptr, &count);
  } else {
    count = requested;
    nodes.resize(count);
    result = call(nodes.data(), &count);
  }
#endif

  bool send_arrays = requested != 0 && count != 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (send_arrays &&
       rpc_write(conn, nodes.data(), count * sizeof(CUgraphNode)) < 0)) {
    return -1;
  }
#if CUDA_VERSION >= 12030
  if (send_arrays && want_edge &&
      rpc_write(conn, edges.data(), count * sizeof(CUgraphEdgeData)) < 0) {
    return -1;
  }
#endif
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphNodeGetDependencies(conn_t *conn) {
  return lupine_handle_node_dependency_query(conn, /*dependent=*/false);
}

int handle_manual_cuGraphNodeGetDependentNodes(conn_t *conn) {
  return lupine_handle_node_dependency_query(conn, /*dependent=*/true);
}

// cuGraphGetEdges: two parallel out node arrays (from/to) plus an optional
// CUgraphEdgeData array, all sized by an in/out count.
int handle_manual_cuGraphGetEdges(conn_t *conn) {
  CUgraph hGraph = nullptr;
  size_t requested = 0;
  uint8_t want_edge = 0;
  size_t count = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      rpc_read(conn, &requested, sizeof(requested)) < 0 ||
      rpc_read(conn, &want_edge, sizeof(want_edge)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraphNode> from;
  std::vector<CUgraphNode> to;
#if CUDA_VERSION >= 12030
  std::vector<CUgraphEdgeData> edges;
  if (requested == 0) {
    result = cuGraphGetEdges_v2(hGraph, nullptr, nullptr, nullptr, &count);
  } else {
    count = requested;
    from.resize(count);
    to.resize(count);
    if (want_edge) {
      edges.resize(count);
    }
    result = cuGraphGetEdges_v2(hGraph, from.data(), to.data(),
                                want_edge ? edges.data() : nullptr, &count);
  }
#else
  if (requested == 0) {
    result = cuGraphGetEdges(hGraph, nullptr, nullptr, &count);
  } else {
    count = requested;
    from.resize(count);
    to.resize(count);
    result = cuGraphGetEdges(hGraph, from.data(), to.data(), &count);
  }
#endif

  bool send_arrays = requested != 0 && count != 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0 ||
      (send_arrays &&
       rpc_write(conn, from.data(), count * sizeof(CUgraphNode)) < 0) ||
      (send_arrays &&
       rpc_write(conn, to.data(), count * sizeof(CUgraphNode)) < 0)) {
    return -1;
  }
#if CUDA_VERSION >= 12030
  if (send_arrays && want_edge &&
      rpc_write(conn, edges.data(), count * sizeof(CUgraphEdgeData)) < 0) {
    return -1;
  }
#endif
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// Host-node callbacks set after node creation must be wrapped in the same
// trampoline as cuGraphAddHostNode. cuGraphHostNodeSetParams /
// cuGraphExecHostNodeSetParams give us no graph handle to attach the callback's
// lifetime to, so keep them alive for the (per-connection) server process.
struct lupine_host_setparams_callback_store {
  std::mutex mutex;
  std::vector<lupine_host_callback_data *> callbacks;
};

static lupine_host_setparams_callback_store &lupine_host_setparams_callbacks() {
  static lupine_host_setparams_callback_store callbacks;
  return callbacks;
}

static lupine_host_callback_data *
lupine_make_host_setparams_callback(conn_t *conn,
                                    const CUDA_HOST_NODE_PARAMS &params) {
  auto *callback =
      new lupine_host_callback_data{conn, params.fn, params.userData, nullptr};
  auto &callbacks = lupine_host_setparams_callbacks();
  std::lock_guard<std::mutex> guard(callbacks.mutex);
  callbacks.callbacks.push_back(callback);
  return callback;
}

int handle_manual_cuGraphHostNodeSetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback = lupine_make_host_setparams_callback(conn, params);
  CUDA_HOST_NODE_PARAMS serverParams{};
  serverParams.fn = lupine_graph_host_callback;
  serverParams.userData = callback;
  result = cuGraphHostNodeSetParams(hNode, &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphExecHostNodeSetParams(conn_t *conn) {
  CUgraphExec hGraphExec = nullptr;
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  auto *callback = lupine_make_host_setparams_callback(conn, params);
  CUDA_HOST_NODE_PARAMS serverParams{};
  serverParams.fn = lupine_graph_host_callback;
  serverParams.userData = callback;
  result = cuGraphExecHostNodeSetParams(hGraphExec, hNode, &serverParams);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphHostNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  result = cuGraphHostNodeGetParams(hNode, &params);
  // Unwrap the trampoline so the client sees the fn/userData it registered.
  if (result == CUDA_SUCCESS && params.fn == lupine_graph_host_callback &&
      params.userData != nullptr) {
    auto *callback = static_cast<lupine_host_callback_data *>(params.userData);
    params.fn = callback->fn;
    params.userData = callback->userData;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &params, sizeof(params)) < 0 ||
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

  auto resources = lupine_stream_capture_resource_map().get_or_create(
      stream, []() { return std::make_shared<lupine_graph_resources>(); });
  auto *callback = new lupine_host_callback_data{conn, fn, userData, resources};
  resources->add_callback(callback);
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

  std::shared_ptr<lupine_graph_resources> resources;
  if (lupine_stream_capture_resource_map().lookup(stream, &resources)) {
    lupine_event_capture_resource_map().store(event, resources);
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
  uint32_t copy_count = 0;
  std::vector<lupine_pending_dtoh_copy> pending;
  if (result == CUDA_SUCCESS) {
    pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
    if (lupine_write_pending_dtoh_copies(&copy_count, conn, pending) < 0) {
      lupine_cleanup_pending_dtoh_copies(&pending);
      return -1;
    }
  } else {
    if (rpc_write(conn, &copy_count, sizeof(copy_count)) < 0) {
      return -1;
    }
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
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

  std::shared_ptr<lupine_graph_resources> event_resources;
  if (lupine_event_capture_resource_map().lookup(event, &event_resources)) {
    lupine_stream_capture_resource_map().get_or_create(
        stream, [&event_resources]() { return event_resources; });
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

  auto resources = lupine_stream_capture_resource_map().get_or_create(
      stream, []() { return std::make_shared<lupine_graph_resources>(); });
  if (!resources->has_capture_scratch()) {
    static constexpr size_t scratch_size = 128ull * 1024ull * 1024ull;
    void *scratch = nullptr;
    if (cuMemAllocHost(&scratch, scratch_size) == CUDA_SUCCESS) {
      if (!resources->install_capture_scratch(scratch, scratch_size)) {
        cuMemFreeHost(scratch);
      }
    }
  }

  result = !resources->has_capture_scratch()
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
    std::shared_ptr<lupine_graph_resources> resources;
    if (lupine_stream_capture_resource_map().remove(stream, &resources)) {
      lupine_graph_resource_map().store(graph, resources);
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
    std::shared_ptr<lupine_graph_resources> resources;
    if (lupine_graph_resource_map().lookup(original, &resources)) {
      lupine_graph_resource_map().store(clone, resources);
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
    std::shared_ptr<lupine_graph_resources> resources;
    if (lupine_graph_resource_map().lookup(graph, &resources)) {
      lupine_graph_exec_resource_map().store(exec, resources);
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
    std::shared_ptr<lupine_graph_resources> resources;
    if (lupine_graph_resource_map().lookup(graph, &resources)) {
      lupine_graph_exec_resource_map().store(exec, resources);
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
  std::shared_ptr<lupine_graph_resources> resources;
  if (lupine_graph_exec_resource_map().remove(exec, &resources)) {
    lupine_retire_graph_resources(resources);
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
  std::shared_ptr<lupine_graph_resources> resources;
  if (lupine_graph_resource_map().remove(graph, &resources)) {
    lupine_retire_graph_resources(resources);
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

  if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  int framed = lupine_payload_framed(conn, byteCount);
  size_t chunk_bytes = std::min(LUPINE_HTOD_CHUNK_BYTES, byteCount);
  lupine_staging staging = lupine_acquire_staging(chunk_bytes);
  void *host = staging.ptr;
  if (chunk_bytes != 0 && host == nullptr) {
    result = CUDA_ERROR_OUT_OF_MEMORY;
    if (rpc_drain_payload(conn, framed, byteCount) < 0) {
      return -1;
    }
  }

  size_t offset = 0;
  while (result == CUDA_SUCCESS && offset < byteCount) {
    size_t chunk = std::min(chunk_bytes, byteCount - offset);
    if (rpc_read_payload_part(conn, framed, host, chunk) < 0) {
      lupine_release_staging(staging);
      return -1;
    }
    result = cuMemcpyHtoD_v2(dstDevice + offset, host, chunk);
    offset += chunk;
    if (result != CUDA_SUCCESS &&
        rpc_drain_payload(conn, framed, byteCount - offset) < 0) {
      lupine_release_staging(staging);
      return -1;
    }
  }

  lupine_release_staging(staging);

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

  int framed = lupine_payload_framed(conn, byteCount);
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto resources = lupine_stream_capture_resource_map().get_or_create(
        stream, []() { return std::make_shared<lupine_graph_resources>(); });
    capture_host = lupine_alloc_capture_scratch(resources, byteCount);
    if (capture_host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_drain_payload(conn, framed, byteCount) < 0) {
        return -1;
      }
    } else {
      if (byteCount != 0 &&
          rpc_read_payload_part(conn, framed, capture_host, byteCount) < 0) {
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
      if (rpc_drain_payload(conn, framed, byteCount) < 0) {
        return -1;
      }
    }
    size_t offset = 0;
    while (result == CUDA_SUCCESS && offset < byteCount) {
      size_t chunk = std::min(LUPINE_HTOD_CHUNK_BYTES, byteCount - offset);
      auto *chunk_host = static_cast<unsigned char *>(host) + offset;
      if (rpc_read_payload_part(conn, framed, chunk_host, chunk) < 0) {
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
        if (rpc_drain_payload(conn, framed, byteCount - offset) < 0) {
          return -1;
        }
        host = nullptr;
        break;
      }
      offset += chunk;
    }
    if (host != nullptr && result == CUDA_SUCCESS) {
      result = cuLaunchHostFunc(
          stream, [](void *userData) { cuMemFreeHost(userData); }, host);
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

int handle_manual_cuMemcpyDtoH_v2(conn_t *conn) {
  CUdeviceptr srcDevice = 0;
  size_t byteCount = 0;
  int request_id = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned char> dstHost;

  if (rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t staging_size =
      std::min(byteCount, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
  if (staging_size != 0) {
    try {
      dstHost.resize(staging_size);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &result, sizeof(result)) < 0 ||
          rpc_write_end(conn) < 0) {
        return -1;
      }
      return 0;
    }
  }

  size_t offset = 0;
  do {
    size_t chunk = std::min(byteCount - offset, staging_size);
    void *chunk_dst = chunk == 0 ? nullptr : dstHost.data();
    result = cuMemcpyDtoH_v2(chunk_dst, srcDevice + offset, chunk);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS && chunk != 0 &&
         rpc_write_payload(conn, dstHost.data(), chunk) < 0) ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      return 0;
    }
    offset += chunk;
  } while (offset < byteCount);

  return 0;
}

int handle_manual_cuMemcpyAtoH_v2(conn_t *conn) {
  CUarray srcArray = nullptr;
  size_t srcOffset = 0;
  size_t byteCount = 0;
  int request_id = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  std::vector<unsigned char> dstHost;

  if (rpc_read(conn, &srcArray, sizeof(srcArray)) < 0 ||
      rpc_read(conn, &srcOffset, sizeof(srcOffset)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t staging_size =
      std::min(byteCount, (size_t)LUPINE_COMPRESS_BLOCK_BYTES);
  if (staging_size != 0) {
    try {
      dstHost.resize(staging_size);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
      if (rpc_write_start_response(conn, request_id) < 0 ||
          rpc_write(conn, &result, sizeof(result)) < 0 ||
          rpc_write_end(conn) < 0) {
        return -1;
      }
      return 0;
    }
  }

  size_t offset = 0;
  do {
    size_t chunk = std::min(byteCount - offset, staging_size);
    void *chunk_dst = chunk == 0 ? nullptr : dstHost.data();
    result = cuMemcpyAtoH_v2(chunk_dst, srcArray, srcOffset + offset, chunk);
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        (result == CUDA_SUCCESS && chunk != 0 &&
         rpc_write(conn, dstHost.data(), chunk) < 0) ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    if (result != CUDA_SUCCESS) {
      return 0;
    }
    offset += chunk;
  } while (offset < byteCount);

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
    auto resources = lupine_stream_capture_resource_map().get_or_create(
        stream, []() { return std::make_shared<lupine_graph_resources>(); });
    host = lupine_alloc_capture_scratch(resources, byteCount);
    if (host == nullptr && byteCount != 0) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS) {
        resources->add_dtoh_copy({dstHost, host, byteCount});
      }
      host = nullptr;
    }
  } else {
    alloc_result = cuMemAllocHost(&host, byteCount);
    if (alloc_result != CUDA_SUCCESS) {
      host = byteCount == 0 ? nullptr : malloc(byteCount);
    }
    if (byteCount != 0 && host == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      result = cuMemcpyDtoHAsync_v2(host, srcDevice, byteCount, stream);
      if (result == CUDA_SUCCESS && byteCount != 0) {
        auto &registry = lupine_pending_dtoh_copies();
        std::lock_guard<std::mutex> lock(registry.mutex);
        auto &pending = registry.copies[conn][stream];
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
  uint32_t copy_count = 0;
  uint64_t stdout_size = 0;
  auto pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_pending_dtoh_copies(&copy_count, conn, pending) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
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
  lupine_stream_capture_resource_map().lookup(stream, &resources);
  std::vector<lupine_graph_host_copy> graph_copies =
      resources == nullptr ? std::vector<lupine_graph_host_copy>()
                           : resources->dtoh_copy_snapshot();
  uint32_t graph_copy_count = static_cast<uint32_t>(graph_copies.size());
  bool all_pending_streams = stream == nullptr;
  auto pending =
      lupine_detach_pending_dtoh_copies(conn, stream, all_pending_streams);
  uint32_t pending_copy_count = static_cast<uint32_t>(pending.size());
  copy_count = graph_copy_count + pending_copy_count;
  uint64_t stdout_size = 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &copy_count, sizeof(copy_count)) < 0 ||
      std::any_of(
          graph_copies.begin(), graph_copies.end(),
          [&](const lupine_graph_host_copy &copy) {
            return rpc_write(conn, &copy.client_dst, sizeof(copy.client_dst)) <
                       0 ||
                   rpc_write(conn, &copy.bytes, sizeof(copy.bytes)) < 0 ||
                   (copy.bytes != 0 &&
                    rpc_write_payload(conn, copy.server_src, copy.bytes) < 0);
          }) ||
      lupine_write_pending_dtoh_copies(nullptr, conn, pending) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
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
  std::shared_ptr<lupine_graph_resources> resources;
  if (result == CUDA_SUCCESS &&
      lupine_graph_exec_resource_map().lookup(exec, &resources)) {
    lupine_stream_capture_resource_map().store(stream, resources);
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
  uint32_t copy_count = 0;
  uint64_t stdout_size = 0;
  auto pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_pending_dtoh_copies(&copy_count, conn, pending) < 0 ||
      lupine_write_captured_stdout(conn, capture, &stdout_size) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    lupine_cleanup_pending_dtoh_copies(&pending);
    return -1;
  }
  lupine_cleanup_pending_dtoh_copies(&pending);
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
