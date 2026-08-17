#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
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
#include <new>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>

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

#include "cache.h"
#include "codegen/gen_api.h"
#include "codegen/gen_server.h"
#include "copy_pipeline.h"
#include "ipc.h"
#include "lupine_attr_sizes.h"
#include "lupine_fatbin.h"
#include "lupine_log.h"
#include "manual_server.h"
#include "rpc.h"

#ifdef _WIN32
#include <io.h>
#define lupine_ipc_close_fd _close
#else
#include <unistd.h>
#define lupine_ipc_close_fd close
#endif
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

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

static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;

using lupine_kernel_param_values_owner =
    std::unique_ptr<void *, decltype(&rpc_free_kernel_param_values)>;

static lupine_kernel_param_values_owner
lupine_own_kernel_param_values(void **values) {
  return lupine_kernel_param_values_owner(values, rpc_free_kernel_param_values);
}

static constexpr size_t lupine_param_info_copy_size() {
  return (sizeof(CUresult) + alignof(size_t) - 1) / alignof(size_t) *
             alignof(size_t) +
         2 * sizeof(size_t);
}

static constexpr size_t lupine_graph_kernel_node_response_copy_size() {
  return (sizeof(CUresult) + alignof(CUDA_KERNEL_NODE_PARAMS) - 1) /
             alignof(CUDA_KERNEL_NODE_PARAMS) *
             alignof(CUDA_KERNEL_NODE_PARAMS) +
         sizeof(CUDA_KERNEL_NODE_PARAMS);
}

template <typename T> static constexpr size_t lupine_cuda_value_copy_size() {
  return (sizeof(CUresult) + alignof(T) - 1) / alignof(T) * alignof(T) +
         sizeof(T);
}

static constexpr size_t lupine_attribute_copy_size() {
  return (sizeof(CUresult) + alignof(int) - 1) / alignof(int) * alignof(int) +
         2 * sizeof(int);
}

template <typename Query>
static int lupine_write_param_values(conn_t *conn, void *const *values,
                                     Query query) {
  for (size_t i = 0;; ++i) {
    if (rpc_write_buffer_reserve(conn, lupine_param_info_copy_size(),
                                 std::max(alignof(CUresult), alignof(size_t))) <
        0) {
      return -1;
    }
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    auto *offset = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    auto *size = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    if (result == nullptr || offset == nullptr || size == nullptr) {
      return -1;
    }

    *offset = 0;
    *size = 0;
    *result = query(i, offset, size);
    if (*result == CUDA_SUCCESS &&
        (values == nullptr || values[i] == nullptr)) {
      *result = CUDA_ERROR_INVALID_VALUE;
    }
    if (*result != CUDA_SUCCESS) {
      return 0;
    }
    if (rpc_write(conn, values[i], *size) < 0) {
      return -1;
    }
  }
}

static int lupine_write_func_param_values(conn_t *conn, CUfunction function,
                                          void *const *values) {
  return lupine_write_param_values(
      conn, values, [function](size_t index, size_t *offset, size_t *size) {
        return cuFuncGetParamInfo(function, index, offset, size);
      });
}

#if CUDA_VERSION >= 12000
static int lupine_write_kernel_param_values(conn_t *conn, CUkernel kernel,
                                            void *const *values) {
  return lupine_write_param_values(
      conn, values, [kernel](size_t index, size_t *offset, size_t *size) {
        return cuKernelGetParamInfo(kernel, index, offset, size);
      });
}
#endif

// The CUDA linker requires output option buffers to remain valid for the
// lifetime of CUlinkState. The mutex serializes cuLinkDestroy against a
// concurrent cuLinkComplete on another RPC lane: the cubin buffer returned by
// cuLinkComplete is driver-owned and freed by cuLinkDestroy, so it must stay
// locked out until the response holding that buffer has been flushed.
struct lupine_link_state {
  CUlinkState cuda_state = nullptr;
  rpc_jit_server_state jit;
  std::mutex mutex;
};

static int lupine_write_jit_outputs(conn_t *conn, rpc_jit_server_state *state) {
  if (conn == nullptr || state == nullptr ||
      rpc_copy_alloc(conn, sizeof(uint32_t)) < 0) {
    return -1;
  }

  auto *output_count = static_cast<uint32_t *>(
      rpc_write_buffer(conn, sizeof(uint32_t), alignof(uint32_t)));
  if (output_count == nullptr) {
    return -1;
  }
  *output_count = static_cast<uint32_t>(state->capture_wall_time) +
                  static_cast<uint32_t>(state->capture_info_log) +
                  static_cast<uint32_t>(state->capture_error_log);

  if (state->capture_wall_time) {
    auto *option = static_cast<CUjit_option *>(
        rpc_write_buffer(conn, sizeof(CUjit_option), alignof(CUjit_option)));
    if (option == nullptr) {
      return -1;
    }
    *option = CU_JIT_WALL_TIME;
    auto *size = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    if (size == nullptr) {
      return -1;
    }
    *size = sizeof(state->wall_time);
    if (rpc_write(conn, &state->wall_time, *size) < 0) {
      return -1;
    }
  }

  if (state->capture_info_log) {
    auto *option = static_cast<CUjit_option *>(
        rpc_write_buffer(conn, sizeof(CUjit_option), alignof(CUjit_option)));
    if (option == nullptr) {
      return -1;
    }
    *option = CU_JIT_INFO_LOG_BUFFER;
    auto *size = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    if (size == nullptr) {
      return -1;
    }
    *size = state->info_log.size();
    if (*size != 0 && rpc_write(conn, state->info_log.data(), *size) < 0) {
      return -1;
    }
  }

  if (state->capture_error_log) {
    auto *option = static_cast<CUjit_option *>(
        rpc_write_buffer(conn, sizeof(CUjit_option), alignof(CUjit_option)));
    if (option == nullptr) {
      return -1;
    }
    *option = CU_JIT_ERROR_LOG_BUFFER;
    auto *size = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    if (size == nullptr) {
      return -1;
    }
    *size = state->error_log.size();
    if (*size != 0 && rpc_write(conn, state->error_log.data(), *size) < 0) {
      return -1;
    }
  }
  return 0;
}

static lupine_link_state *lupine_link_state_from_handle(CUlinkState state) {
  return reinterpret_cast<lupine_link_state *>(state);
}

static CUlinkState lupine_link_state_to_handle(lupine_link_state *state) {
  return reinterpret_cast<CUlinkState>(state);
}

#ifdef _WIN32
static constexpr size_t LUPINE_HTOD_CHUNK_BYTES = 64 * 1024 * 1024;

struct lupine_deferred_host_free {
  void *ptr = nullptr;
  lupine_deferred_host_free *next = nullptr;
};

struct lupine_host_free_queue {
  std::mutex mutex;
  std::condition_variable condition;
  lupine_deferred_host_free *head = nullptr;
  lupine_deferred_host_free *tail = nullptr;
};

static lupine_host_free_queue &lupine_host_frees() {
  // The server process owns this detached worker until exit. Keep its queue
  // alive for the same lifetime so static destruction cannot race the worker.
  static auto *queue = new lupine_host_free_queue();
  return *queue;
}

static void lupine_reap_host_frees() {
  auto &queue = lupine_host_frees();
  for (;;) {
    lupine_deferred_host_free *item = nullptr;
    {
      std::unique_lock<std::mutex> lock(queue.mutex);
      queue.condition.wait(lock, [&queue] { return queue.head != nullptr; });
      item = queue.head;
      queue.head = item->next;
      if (queue.head == nullptr) {
        queue.tail = nullptr;
      }
    }

    CUresult result = cuMemFreeHost(item->ptr);
    if (result != CUDA_SUCCESS) {
      LUPINE_LOG_ERROR("Deferred cuMemFreeHost failed for " << item->ptr << ": "
                                                            << result);
    }
    delete item;
  }
}

static bool lupine_start_host_free_reaper() {
  static std::once_flag once;
  try {
    (void)lupine_host_frees();
    std::call_once(once, [] { std::thread(lupine_reap_host_frees).detach(); });
    return true;
  } catch (...) {
    return false;
  }
}

static void CUDA_CB lupine_queue_host_free(void *userData) {
  // CUDA host functions cannot call CUDA APIs. Hand the completed allocation
  // to a normal host thread before calling cuMemFreeHost.
  auto *item = static_cast<lupine_deferred_host_free *>(userData);
  auto &queue = lupine_host_frees();
  {
    std::lock_guard<std::mutex> lock(queue.mutex);
    if (queue.tail == nullptr) {
      queue.head = item;
    } else {
      queue.tail->next = item;
    }
    queue.tail = item;
  }
  queue.condition.notify_one();
}

static CUresult lupine_defer_host_free(CUstream stream, void *ptr) {
  auto *item = new (std::nothrow) lupine_deferred_host_free{ptr, nullptr};
  if (item == nullptr || !lupine_start_host_free_reaper()) {
    delete item;
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  CUresult result = cuLaunchHostFunc(stream, lupine_queue_host_free, item);
  if (result != CUDA_SUCCESS) {
    delete item;
  }
  return result;
}
#endif

struct lupine_captured_stdout {
  int saved_stdout = -1;
  bool active = false;
  std::string output;
};

static pthread_mutex_t lupine_stdout_capture_mutex = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> lupine_stdout_capture_required{false};

static bool lupine_image_contains(const unsigned char *image, size_t image_size,
                                  const char *needle, size_t needle_size) {
  return image != nullptr && needle != nullptr && needle_size != 0 &&
         image_size >= needle_size &&
         std::search(image, image + image_size, needle, needle + needle_size) !=
             image + image_size;
}

static bool lupine_image_may_use_device_stdout(const unsigned char *image,
                                               size_t image_size) {
  static constexpr char vprintf_symbol[] = "vprintf";
  if (lupine_image_contains(image, image_size, vprintf_symbol,
                            sizeof(vprintf_symbol) - 1)) {
    return true;
  }

  // PTX names vprintf directly and cubins retain it in their symbol data. A
  // fatbin whose members are all compressed exposes neither representation,
  // so keep capture enabled for that unknown case rather than dropping output.
  uint32_t magic = 0;
  if (image_size >= sizeof(magic)) {
    memcpy(&magic, image, sizeof(magic));
  }
  static constexpr char elf_magic[] = "\177ELF";
  static constexpr char ptx_version[] = ".version";
  return magic == LUPINE_FATBIN_MAGIC &&
         !lupine_image_contains(image, image_size, elf_magic,
                                sizeof(elf_magic) - 1) &&
         !lupine_image_contains(image, image_size, ptx_version,
                                sizeof(ptx_version) - 1);
}

static void lupine_note_device_stdout_image(const unsigned char *image,
                                            size_t image_size) {
  if (lupine_image_may_use_device_stdout(image, image_size)) {
    lupine_stdout_capture_required.store(true, std::memory_order_release);
  }
}

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

  // Redirecting fd 1 is process-global, so only pay the serialization and
  // syscall cost after a loaded image has shown that device stdout may be used.
  if (!lupine_stdout_capture_required.load(std::memory_order_acquire)) {
    return false;
  }

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
                                        const lupine_captured_stdout &capture) {
  auto *output_size = static_cast<uint64_t *>(
      rpc_write_buffer(conn, sizeof(uint64_t), alignof(uint64_t)));
  if (output_size == nullptr) {
    return -1;
  }
  *output_size = capture.output.size();
  if (*output_size != 0 &&
      rpc_write(conn, capture.output.data(), capture.output.size()) < 0) {
    return -1;
  }
  return 0;
}

struct lupine_private_module_node_capture {
  void *node = nullptr;
  uint64_t owner = 0;
  uint64_t count = 0;
};

static libcuckoo::cuckoohash_map<CUmodule, CUlibrary> &
lupine_module_libraries() {
  static auto *libraries = new libcuckoo::cuckoohash_map<CUmodule, CUlibrary>();
  return *libraries;
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
  lupine_graph_resources *resources = nullptr;
};

struct lupine_stream_callback_data {
  conn_t *conn = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
};

struct lupine_graph_host_copy_node {
  explicit lupine_graph_host_copy_node(lupine_graph_host_copy node_copy)
      : copy(node_copy) {}

  lupine_graph_host_copy copy;
  lupine_graph_host_copy_node *next = nullptr;
};

struct lupine_graph_capture_scratch {
  lupine_graph_capture_scratch(void *scratch_ptr, size_t scratch_size)
      : ptr(scratch_ptr), size(scratch_size) {}

  void *ptr;
  size_t size;
  std::atomic<size_t> offset{0};
};

struct lupine_graph_resources {
  void add_dtoh_copy(lupine_graph_host_copy copy) {
    auto *node = new lupine_graph_host_copy_node(copy);
    node->next = dtoh_copies.load(std::memory_order_relaxed);
    while (!dtoh_copies.compare_exchange_weak(node->next, node,
                                              std::memory_order_release,
                                              std::memory_order_relaxed)) {
    }
  }

  std::vector<lupine_graph_host_copy> dtoh_copy_snapshot() const {
    std::vector<lupine_graph_host_copy> copies;
    for (auto *node = dtoh_copies.load(std::memory_order_acquire);
         node != nullptr; node = node->next) {
      copies.push_back(node->copy);
    }
    std::reverse(copies.begin(), copies.end());
    return copies;
  }

  bool has_capture_scratch() const {
    return capture_scratch.load(std::memory_order_acquire) != nullptr;
  }

  bool install_capture_scratch(void *scratch, size_t size) {
    if (scratch == nullptr) {
      return false;
    }
    auto *candidate = new lupine_graph_capture_scratch(scratch, size);
    lupine_graph_capture_scratch *expected = nullptr;
    if (!capture_scratch.compare_exchange_strong(expected, candidate,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
      delete candidate;
      return false;
    }
    return true;
  }

  void *allocate_capture_scratch(size_t bytes) {
    if (bytes == 0) {
      return nullptr;
    }
    auto *scratch = capture_scratch.load(std::memory_order_acquire);
    if (scratch == nullptr) {
      return nullptr;
    }
    size_t current = scratch->offset.load(std::memory_order_relaxed);
    for (;;) {
      if (current > scratch->size || current > SIZE_MAX - 255) {
        return nullptr;
      }
      size_t aligned = (current + 255) & ~size_t(255);
      if (aligned > scratch->size || bytes > scratch->size - aligned) {
        return nullptr;
      }
      if (scratch->offset.compare_exchange_weak(current, aligned + bytes,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
        return static_cast<unsigned char *>(scratch->ptr) + aligned;
      }
    }
  }

  std::atomic<lupine_graph_host_copy_node *> dtoh_copies{nullptr};
  std::atomic<lupine_graph_capture_scratch *> capture_scratch{nullptr};
};

// Graph host buffers and callback metadata must remain valid for any queued
// launch or callback. The per-connection process already owned them until exit;
// stable raw pointers make that lifetime explicit and avoid reference-count
// synchronization on every map access.
static libcuckoo::cuckoohash_map<CUgraph, lupine_graph_resources *> &
lupine_graph_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraph, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUgraphExec, lupine_graph_resources *> &
lupine_graph_exec_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUgraphExec, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *> &
lupine_stream_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUstream, lupine_graph_resources *>();
  return *resources;
}

static libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *> &
lupine_event_capture_resource_map() {
  static auto *resources =
      new libcuckoo::cuckoohash_map<CUevent, lupine_graph_resources *>();
  return *resources;
}

using lupine_pending_dtoh_streams =
    std::unordered_map<CUstream, std::vector<lupine_pending_dtoh_copy>>;

static libcuckoo::cuckoohash_map<conn_t *, lupine_pending_dtoh_streams> &
lupine_pending_dtoh_copies() {
  static libcuckoo::cuckoohash_map<conn_t *, lupine_pending_dtoh_streams>
      copies;
  return copies;
}

static lupine_graph_resources *lupine_get_graph_resources(CUgraph graph) {
  auto *candidate = new lupine_graph_resources();
  auto *resources = candidate;
  lupine_graph_resource_map().upsert(
      graph,
      [&resources](lupine_graph_resources *&existing,
                   libcuckoo::UpsertContext) { resources = existing; },
      candidate);
  if (resources != candidate) {
    delete candidate;
  }
  return resources;
}

static lupine_graph_resources *lupine_get_stream_resources(CUstream stream) {
  auto *candidate = new lupine_graph_resources();
  auto *resources = candidate;
  lupine_stream_capture_resource_map().upsert(
      stream,
      [&resources](lupine_graph_resources *&existing,
                   libcuckoo::UpsertContext) { resources = existing; },
      candidate);
  if (resources != candidate) {
    delete candidate;
  }
  return resources;
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

static void *lupine_alloc_process_host_buffer(size_t bytes) {
  void *ptr = nullptr;
  if (bytes == 0) {
    return nullptr;
  }
  if (cuMemAllocHost(&ptr, bytes) != CUDA_SUCCESS) {
    return malloc(bytes);
  }
  return ptr;
}

static std::vector<lupine_pending_dtoh_copy>
lupine_detach_pending_dtoh_copies(conn_t *conn, CUstream stream,
                                  bool all_streams) {
  std::vector<lupine_pending_dtoh_copy> copies;
  lupine_pending_dtoh_copies().erase_fn(
      conn, [&](lupine_pending_dtoh_streams &streams) {
        if (all_streams) {
          for (auto &entry : streams) {
            auto &stream_copies = entry.second;
            copies.insert(copies.end(), stream_copies.begin(),
                          stream_copies.end());
          }
          return true;
        }

        auto stream_it = streams.find(stream);
        if (stream_it == streams.end()) {
          return false;
        }
        copies.swap(stream_it->second);
        streams.erase(stream_it);
        return streams.empty();
      });
  return copies;
}

static int lupine_write_pending_dtoh_copies(
    conn_t *conn, const std::vector<lupine_pending_dtoh_copy> &pending,
    bool include_count) {
  if (include_count) {
    auto *copy_count = static_cast<uint32_t *>(
        rpc_write_buffer(conn, sizeof(uint32_t), alignof(uint32_t)));
    if (copy_count == nullptr) {
      return -1;
    }
    *copy_count = static_cast<uint32_t>(pending.size());
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

static void *lupine_alloc_capture_scratch(lupine_graph_resources *resources,
                                          size_t bytes) {
  if (resources == nullptr || bytes == 0) {
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
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image.data(), image_size);
  }

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
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image.data(), image.size());
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

static int lupine_write_function_attributes(conn_t *conn, CUfunction function) {
  if (conn == nullptr) {
    return -1;
  }
  auto *count = static_cast<uint32_t *>(
      rpc_write_buffer(conn, sizeof(uint32_t), alignof(uint32_t)));
  if (count == nullptr) {
    return -1;
  }
  *count = CU_FUNC_ATTRIBUTE_MAX;
  for (int attribute = 0; attribute < CU_FUNC_ATTRIBUTE_MAX; ++attribute) {
    if (rpc_write_buffer_reserve(conn, lupine_attribute_copy_size(),
                                 std::max(alignof(CUresult), alignof(int))) <
        0) {
      return -1;
    }
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    auto *wire_attribute =
        static_cast<int *>(rpc_write_buffer(conn, sizeof(int), alignof(int)));
    auto *value =
        static_cast<int *>(rpc_write_buffer(conn, sizeof(int), alignof(int)));
    if (result == nullptr || wire_attribute == nullptr || value == nullptr) {
      return -1;
    }
    *wire_attribute = attribute;
    *value = 0;
    *result = function == nullptr
                  ? CUDA_ERROR_INVALID_HANDLE
                  : cuFuncGetAttribute(
                        value, static_cast<CUfunction_attribute>(attribute),
                        function);
  }
  return 0;
}

static constexpr size_t lupine_attribute_snapshot_copy_size() {
  return sizeof(uint32_t) + static_cast<size_t>(CU_FUNC_ATTRIBUTE_MAX) *
                                lupine_attribute_copy_size();
}

static int lupine_write_kernel_attributes(conn_t *conn, CUkernel kernel,
                                          CUdevice device) {
  if (conn == nullptr) {
    return -1;
  }
  auto *count = static_cast<uint32_t *>(
      rpc_write_buffer(conn, sizeof(uint32_t), alignof(uint32_t)));
  if (count == nullptr) {
    return -1;
  }
  *count = CU_FUNC_ATTRIBUTE_MAX;
#if CUDA_VERSION < 12000
  (void)kernel;
  (void)device;
#endif
  for (int attribute = 0; attribute < CU_FUNC_ATTRIBUTE_MAX; ++attribute) {
    if (rpc_write_buffer_reserve(conn, lupine_attribute_copy_size(),
                                 std::max(alignof(CUresult), alignof(int))) <
        0) {
      return -1;
    }
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    auto *wire_attribute =
        static_cast<int *>(rpc_write_buffer(conn, sizeof(int), alignof(int)));
    auto *value =
        static_cast<int *>(rpc_write_buffer(conn, sizeof(int), alignof(int)));
    if (result == nullptr || wire_attribute == nullptr || value == nullptr) {
      return -1;
    }
    *wire_attribute = attribute;
    *value = 0;
#if CUDA_VERSION >= 12000
    *result = kernel == nullptr
                  ? CUDA_ERROR_INVALID_HANDLE
                  : cuKernelGetAttribute(
                        value, static_cast<CUfunction_attribute>(attribute),
                        kernel, device);
#else
    *result = CUDA_ERROR_NOT_SUPPORTED;
#endif
  }
  return 0;
}

int handle_manual_lupineFunctionParamLayoutSnapshot(conn_t *conn) {
  CUfunction function = nullptr;
  if (rpc_read(conn, &function, sizeof(function)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_copy_alloc(conn, lupine_param_info_copy_size()) < 0) {
    return -1;
  }

  for (uint32_t i = 0;; ++i) {
    if (rpc_write_buffer_reserve(conn, lupine_param_info_copy_size(),
                                 std::max(alignof(CUresult), alignof(size_t))) <
        0) {
      return -1;
    }
    auto *result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    auto *offset = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    auto *size = static_cast<size_t *>(
        rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
    if (result == nullptr || offset == nullptr || size == nullptr) {
      return -1;
    }

    *offset = 0;
    *size = 0;
    *result = cuFuncGetParamInfo(function, i, offset, size);
    if (*result != CUDA_SUCCESS) {
      break;
    }
  }
  return rpc_write_end(conn);
}

int handle_manual_lupineFunctionAttributeSnapshot(conn_t *conn) {
  CUfunction function = nullptr;
  if (rpc_read(conn, &function, sizeof(function)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_copy_alloc(conn, lupine_attribute_snapshot_copy_size()) < 0 ||
      lupine_write_function_attributes(conn, function) < 0 ||
      rpc_write_end(conn) < 0) {
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
    // The client-side image is not the buffer passed to CUDA on this process.
    // Clear the preservation hint so the driver retains its own copy rather
    // than requiring a global library-to-image lifetime table.
    if (library_options[i] == CU_LIBRARY_BINARY_IS_PRESERVED) {
      library_option_values[i] = nullptr;
    }
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
    lupine_note_device_stdout_image(image.data(), image.size());
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      lupine_write_jit_outputs(conn, &jit_state) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_lupineLibrarySnapshot(conn_t *conn) {
  CUlibrary library = nullptr;
  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  struct kernel_record {
    std::string name;
    CUkernel kernel = nullptr;
    CUfunction function = nullptr;
    uint32_t name_len = 0;
    uint32_t param_count = 0;
    std::vector<uint64_t> params;
  };
  std::vector<kernel_record> records;
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;
#if CUDA_VERSION >= 12040
  unsigned int kernel_count = 0;
  std::vector<CUkernel> kernels;
  result = cuLibraryGetKernelCount(&kernel_count, library);
  if (result == CUDA_SUCCESS && kernel_count != 0) {
    kernels.resize(kernel_count);
    result = cuLibraryEnumerateKernels(kernels.data(), kernel_count, library);
    if (result != CUDA_SUCCESS) {
      kernels.clear();
    }
  }
  if (result == CUDA_SUCCESS) {
    for (CUkernel kernel : kernels) {
      const char *name = nullptr;
      if (kernel == nullptr || cuKernelGetName(&name, kernel) != CUDA_SUCCESS ||
          name == nullptr) {
        continue;
      }
      kernel_record record;
      record.name = name;
      record.name_len = static_cast<uint32_t>(record.name.size() + 1);
      record.kernel = kernel;
      if (cuKernelGetFunction(&record.function, kernel) != CUDA_SUCCESS) {
        record.function = nullptr;
      }
      for (size_t index = 0;; ++index) {
        size_t offset = 0;
        size_t size = 0;
        if (cuKernelGetParamInfo(kernel, index, &offset, &size) !=
            CUDA_SUCCESS) {
          break;
        }
        record.params.push_back(static_cast<uint64_t>(offset));
        record.params.push_back(static_cast<uint64_t>(size));
      }
      record.param_count = static_cast<uint32_t>(record.params.size() / 2);
      records.push_back(std::move(record));
    }
  }
#endif

  uint32_t table_count = static_cast<uint32_t>(records.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &table_count, sizeof(table_count)) < 0) {
    return -1;
  }
  for (const auto &record : records) {
    if (rpc_write(conn, &record.name_len, sizeof(record.name_len)) < 0 ||
        rpc_write(conn, record.name.c_str(), record.name_len) < 0 ||
        rpc_write(conn, &record.kernel, sizeof(record.kernel)) < 0 ||
        rpc_write(conn, &record.function, sizeof(record.function)) < 0 ||
        rpc_write(conn, &record.param_count, sizeof(record.param_count)) < 0 ||
        (record.param_count != 0 &&
         rpc_write(conn, record.params.data(),
                   record.params.size() * sizeof(uint64_t)) < 0)) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
}

int handle_manual_lupineLibraryAttributeSnapshot(conn_t *conn) {
  CUlibrary library = nullptr;
  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

#if CUDA_VERSION >= 12040
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_copy_alloc(conn, lupine_cuda_value_copy_size<CUdevice>()) < 0) {
    return -1;
  }

  if (rpc_write_buffer_reserve(conn, lupine_cuda_value_copy_size<CUdevice>(),
                               std::max(alignof(CUresult), alignof(CUdevice))) <
      0) {
    return -1;
  }
  auto *device_result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  auto *device = static_cast<CUdevice *>(
      rpc_write_buffer(conn, sizeof(CUdevice), alignof(CUdevice)));
  if (device_result == nullptr || device == nullptr) {
    return -1;
  }
  *device = -1;
  *device_result = cuCtxGetDevice(device);
  if (*device_result != CUDA_SUCCESS) {
    return rpc_write_end(conn) < 0 ? -1 : 0;
  }
  CUdevice snapshot_device = *device;

  if (rpc_write_buffer_reserve(
          conn, lupine_cuda_value_copy_size<unsigned int>(),
          std::max(alignof(CUresult), alignof(unsigned int))) < 0) {
    return -1;
  }
  auto *count_result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  auto *kernel_count = static_cast<unsigned int *>(
      rpc_write_buffer(conn, sizeof(unsigned int), alignof(unsigned int)));
  if (count_result == nullptr || kernel_count == nullptr) {
    return -1;
  }
  *kernel_count = 0;
  *count_result = cuLibraryGetKernelCount(kernel_count, library);
  if (*count_result != CUDA_SUCCESS || *kernel_count == 0) {
    return rpc_write_end(conn) < 0 ? -1 : 0;
  }
  unsigned int snapshot_kernel_count = *kernel_count;

  std::vector<CUkernel> kernels(snapshot_kernel_count);

  auto *enumerate_result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  if (enumerate_result == nullptr) {
    return -1;
  }
  *enumerate_result =
      cuLibraryEnumerateKernels(kernels.data(), snapshot_kernel_count, library);
  if (*enumerate_result != CUDA_SUCCESS) {
    return rpc_write_end(conn) < 0 ? -1 : 0;
  }

  for (const CUkernel &kernel : kernels) {
    if (rpc_write(conn, &kernel, sizeof(kernel)) < 0) {
      return -1;
    }
    if (rpc_write_buffer_reserve(
            conn, lupine_cuda_value_copy_size<CUfunction>(),
            std::max(alignof(CUresult), alignof(CUfunction))) < 0) {
      return -1;
    }
    auto *function_result = static_cast<CUresult *>(
        rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
    auto *function = static_cast<CUfunction *>(
        rpc_write_buffer(conn, sizeof(CUfunction), alignof(CUfunction)));
    if (function_result == nullptr || function == nullptr) {
      return -1;
    }
    *function = nullptr;
    *function_result = kernel == nullptr
                           ? CUDA_ERROR_INVALID_HANDLE
                           : cuKernelGetFunction(function, kernel);
    if ((*function_result == CUDA_SUCCESS &&
         lupine_write_function_attributes(conn, *function) < 0) ||
        lupine_write_kernel_attributes(conn, kernel, snapshot_device) < 0) {
      return -1;
    }
  }
#else
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_copy_alloc(conn, sizeof(CUresult)) < 0) {
    return -1;
  }
  auto *result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  if (result == nullptr) {
    return -1;
  }
  *result = CUDA_ERROR_NOT_SUPPORTED;
#endif
  return rpc_write_end(conn) < 0 ? -1 : 0;
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

// POSIX-fd shareable handles cannot cross the wire as raw fd numbers. Export
// parks the real fd with the parent-process broker under a random token and
// returns the token; import redeems a token for the real fd (see ipc.h).

int handle_manual_cuMemExportToShareableHandle(conn_t *conn) {
  CUmemGenericAllocationHandle handle = 0;
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  lupine_ipc_token token = {};
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;

  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    int shareable_fd = -1;
    result =
        cuMemExportToShareableHandle(&shareable_fd, handle, handleType, flags);
    if (result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION,
                                        &token, shareable_fd) < 0) {
        result = CUDA_ERROR_UNKNOWN;
      }
      lupine_ipc_close_fd(shareable_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemImportFromShareableHandle(conn_t *conn) {
  lupine_ipc_token token = {};
  CUmemAllocationHandleType handleType;
  CUmemGenericAllocationHandle handle = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token);
    if (import_fd >= 0) {
      result = cuMemImportFromShareableHandle(
          &handle, reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd)),
          handleType);
      lupine_ipc_close_fd(import_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolExportToShareableHandle(conn_t *conn) {
  CUmemoryPool pool = nullptr;
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  lupine_ipc_token token = {};
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;

  if (rpc_read(conn, &pool, sizeof(pool)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    int shareable_fd = -1;
    result = cuMemPoolExportToShareableHandle(&shareable_fd, pool, handleType,
                                              flags);
    if (result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token,
                                        shareable_fd) < 0) {
        result = CUDA_ERROR_UNKNOWN;
      }
      lupine_ipc_close_fd(shareable_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &token, sizeof(token)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemPoolImportFromShareableHandle(conn_t *conn) {
  lupine_ipc_token token = {};
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  CUmemoryPool pool = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token);
    if (import_fd >= 0) {
      result = cuMemPoolImportFromShareableHandle(
          &pool, reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd)),
          handleType, flags);
      lupine_ipc_close_fd(import_fd);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pool, sizeof(pool)) < 0 ||
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
      value_size != expected_size) {
    result = CUDA_ERROR_INVALID_VALUE;
    value_size = 0;
  } else if (value_size > sizeof(value)) {
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

int handle_manual_cuPointerSetAttribute(conn_t *conn) {
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
  // A payload larger than the largest pointer attribute cannot be a request
  // this server understands; drop the connection rather than leave unread
  // bytes desynchronizing the stream.
  if (value_size > sizeof(value)) {
    return -1;
  }
  if (value_size != 0 && rpc_read(conn, value, value_size) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  size_t expected_size = 0;
  if (!lupine_settable_pointer_attribute_size(attribute, &expected_size) ||
      value_size != expected_size) {
    result = CUDA_ERROR_INVALID_VALUE;
  } else {
    result = cuPointerSetAttribute(value, attribute, ptr);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
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
      result = CUDA_ERROR_INVALID_VALUE;
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
  auto link_state = std::make_unique<lupine_link_state>();
  CUlinkState client_state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  if (rpc_read_jit_options(conn, &link_state->jit) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  result = cuLinkCreate_v2(
      static_cast<unsigned int>(link_state->jit.options.size()),
      link_state->jit.options.empty() ? nullptr
                                      : link_state->jit.options.data(),
      link_state->jit.option_values.empty()
          ? nullptr
          : link_state->jit.option_values.data(),
      &link_state->cuda_state);
  if (result == CUDA_SUCCESS) {
    client_state = lupine_link_state_to_handle(link_state.release());
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &client_state, sizeof(client_state)) < 0 ||
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
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state != nullptr) {
    result = cuLinkAddData_v2(
        link_state->cuda_state, type, data.data(), data.size(),
        name_len == 0 ? nullptr : name.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_jit_outputs(conn, &jit_state) < 0 ||
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
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state == nullptr) {
    result = CUDA_ERROR_INVALID_HANDLE;
  } else if (!file_data.empty()) {
    result = cuLinkAddData_v2(
        link_state->cuda_state, type, file_data.data(), file_data.size(),
        path_len == 0 ? nullptr : path.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  } else {
    result = cuLinkAddFile_v2(
        link_state->cuda_state, type, path_len == 0 ? nullptr : path.data(),
        static_cast<unsigned int>(jit_state.options.size()),
        jit_state.options.empty() ? nullptr : jit_state.options.data(),
        jit_state.option_values.empty() ? nullptr
                                        : jit_state.option_values.data());
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_jit_outputs(conn, &jit_state) < 0 ||
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
  auto *link_state = lupine_link_state_from_handle(state);
  rpc_jit_server_state empty_jit_state;
  rpc_jit_server_state *jit_state = &empty_jit_state;
  // Held until the response is flushed: the cubin buffer belongs to the
  // driver's link state, and a concurrent cuLinkDestroy would free it while
  // the queued iovec still points at it.
  std::unique_lock<std::mutex> lock;
  if (link_state != nullptr) {
    lock = std::unique_lock<std::mutex>(link_state->mutex);
    result = cuLinkComplete(link_state->cuda_state, &cubin, &size);
    jit_state = &link_state->jit;
  }
  size_t returned_size = result == CUDA_SUCCESS ? size : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &returned_size, sizeof(returned_size)) < 0 ||
      (returned_size != 0 && rpc_write(conn, cubin, returned_size) < 0) ||
      lupine_write_jit_outputs(conn, jit_state) < 0 ||
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
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state != nullptr) {
    // Wait for any cuLinkComplete on another lane to finish flushing the
    // driver-owned cubin buffer before cuLinkDestroy frees it.
    std::lock_guard<std::mutex> lock(link_state->mutex);
    result = cuLinkDestroy(link_state->cuda_state);
  }
  // Retain the opaque handle wrapper until process exit so stale client
  // handles never dereference freed memory.
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
    lupine_module_libraries().insert_or_assign(module, library);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// The library stays loaded: the client mirrors libraries onto other routes and
// caches the CUkernel handles this one owns, but it only ever sends the unload
// to the route that loaded it, so freeing here would dangle those handles.
int handle_manual_cuLibraryUnload(conn_t *conn) {
  CUlibrary library = nullptr;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  if (rpc_read_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuModuleGetGlobal_v2(conn_t *conn) {
  CUdeviceptr *dptr_null_check = nullptr;
  size_t *bytes_null_check = nullptr;
  CUdeviceptr dptr = 0;
  size_t bytes = 0;
  CUmodule module = nullptr;
  std::size_t name_len = 0;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dptr_null_check, sizeof(dptr_null_check)) < 0 ||
      rpc_read(conn, &bytes_null_check, sizeof(bytes_null_check)) < 0 ||
      rpc_read(conn, &module, sizeof(module)) < 0 ||
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

  result = cuModuleGetGlobal_v2(dptr_null_check ? &dptr : nullptr,
                                bytes_null_check ? &bytes : nullptr, module,
                                name.data());
  if (result != CUDA_SUCCESS) {
    CUlibrary library = nullptr;
    if (lupine_module_libraries().find(module, library)) {
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
      rpc_write(conn, &dptr_null_check, sizeof(dptr_null_check)) < 0 ||
      (dptr_null_check && rpc_write(conn, &dptr, sizeof(dptr)) < 0) ||
      rpc_write(conn, &bytes_null_check, sizeof(bytes_null_check)) < 0 ||
      (bytes_null_check && rpc_write(conn, &bytes, sizeof(bytes)) < 0) ||
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
  bool kernel_handle = false;
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
      rpc_read(conn, &kernel_handle, sizeof(kernel_handle)) < 0) {
    return -1;
  }

  result = CUDA_SUCCESS;

  void **params = nullptr;
  CUresult param_result = CUDA_SUCCESS;
#if CUDA_VERSION >= 12000
  int read_result =
      kernel_handle
          ? rpc_read_kernel_param_values(
                conn, &params, reinterpret_cast<CUkernel>(f), &param_result)
          : rpc_read_func_param_values(conn, &params, f, &param_result);
#else
  int read_result = rpc_read_func_param_values(conn, &params, f, &param_result);
#endif
  if (read_result < 0) {
    return -1;
  }
  if (result == CUDA_SUCCESS) {
    result = param_result;
  }
  auto params_owner = lupine_own_kernel_param_values(params);
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    result =
        cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                       blockDimZ, sharedMemBytes, hStream, params, nullptr);
  }

  (void)request_id;
  (void)result;
  return 0;
}

int handle_manual_cuLaunchKernelEx(conn_t *conn) {
  CUlaunchConfig config = {};
  CUfunction f = nullptr;
  bool kernel_handle = false;
  int request_id;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  std::vector<CUlaunchAttribute> attributes;
  if (rpc_read_launch_config(conn, &config, &attributes) < 0 ||
      rpc_read(conn, &f, sizeof(f)) < 0 ||
      rpc_read(conn, &kernel_handle, sizeof(kernel_handle)) < 0) {
    return -1;
  }

#if CUDA_VERSION < 11080
  result = CUDA_ERROR_NOT_SUPPORTED;
#else
  result = CUDA_SUCCESS;
#endif

  void **params = nullptr;
  CUresult param_result = CUDA_SUCCESS;
#if CUDA_VERSION >= 12000
  int read_result =
      kernel_handle
          ? rpc_read_kernel_param_values(
                conn, &params, reinterpret_cast<CUkernel>(f), &param_result)
          : rpc_read_func_param_values(conn, &params, f, &param_result);
#else
  int read_result = rpc_read_func_param_values(conn, &params, f, &param_result);
#endif
  if (read_result < 0) {
    return -1;
  }
  if (result == CUDA_SUCCESS) {
    result = param_result;
  }
  auto params_owner = lupine_own_kernel_param_values(params);
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

#if CUDA_VERSION >= 11080
  if (result == CUDA_SUCCESS) {
    result = cuLaunchKernelEx(&config, f, params, nullptr);
  }
#endif

  // Mirror the client: attribute-free launches are fire-and-forget, launches
  // carrying attributes expect a synchronous result.
  if (config.numAttrs != 0) {
    if (rpc_write_start_response(conn, request_id) < 0 ||
        rpc_write(conn, &result, sizeof(result)) < 0 ||
        rpc_write_end(conn) < 0) {
      return -1;
    }
    return 0;
  }

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
  bool kernel_handle = false;
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
      rpc_read(conn, &kernel_handle, sizeof(kernel_handle)) < 0) {
    return -1;
  }

  result = CUDA_SUCCESS;
  void **params = nullptr;
  int read_result;
#if CUDA_VERSION >= 12000
  read_result = kernel_handle
                    ? rpc_read_kernel_param_values(
                          conn, &params, reinterpret_cast<CUkernel>(f), &result)
                    : rpc_read_func_param_values(conn, &params, f, &result);
#else
  read_result = rpc_read_func_param_values(conn, &params, f, &result);
#endif
  if (read_result < 0) {
    return -1;
  }
  auto params_owner = lupine_own_kernel_param_values(params);
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
    result = cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ,
                                       blockDimX, blockDimY, blockDimZ,
                                       sharedMemBytes, hStream, params);
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
  auto pending = lupine_detach_pending_dtoh_copies(conn, stream, false);
  if (rpc_write_start_request(conn, 2) >= 0 &&
      rpc_copy_alloc(conn, sizeof(uint32_t)) >= 0 &&
      lupine_write_pending_dtoh_copies(conn, pending, true) >= 0 &&
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

int handle_manual_cuGraphAddKernelNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  std::vector<CUgraphNode> deps;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read_kernel_node_params(conn, &nodeParams) < 0 ||
      rpc_read_kernel_node_param_values(conn, &nodeParams, &result) < 0) {
    return -1;
  }
  auto params_owner = lupine_own_kernel_param_values(nodeParams.kernelParams);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
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

int handle_manual_cuGraphKernelNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  int request_id;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0) {
    return -1;
  }
  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_copy_alloc(conn, lupine_graph_kernel_node_response_copy_size()) < 0) {
    return -1;
  }
  auto *result = static_cast<CUresult *>(
      rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
  auto *node_params = static_cast<CUDA_KERNEL_NODE_PARAMS *>(rpc_write_buffer(
      conn, sizeof(CUDA_KERNEL_NODE_PARAMS), alignof(CUDA_KERNEL_NODE_PARAMS)));
  if (result == nullptr || node_params == nullptr) {
    return -1;
  }
  *node_params = {};
  *result = cuGraphKernelNodeGetParams_v2(hNode, node_params);
  if (*result == CUDA_SUCCESS && node_params->extra != nullptr) {
    *result = CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult call_result = *result;
  CUfunction function = node_params->func;
#if CUDA_VERSION >= 12000
  CUkernel kernel = node_params->kern;
#endif
  void **kernel_params = node_params->kernelParams;
  if (call_result == CUDA_SUCCESS) {
    int write_result = -1;
    if (function != nullptr) {
      write_result =
          lupine_write_func_param_values(conn, function, kernel_params);
    }
#if CUDA_VERSION >= 12000
    else if (kernel != nullptr) {
      write_result =
          lupine_write_kernel_param_values(conn, kernel, kernel_params);
    }
#endif
    else {
      if (rpc_write_buffer_reserve(
              conn, lupine_param_info_copy_size(),
              std::max(alignof(CUresult), alignof(size_t))) == 0) {
        auto *param_result = static_cast<CUresult *>(
            rpc_write_buffer(conn, sizeof(CUresult), alignof(CUresult)));
        auto *offset = static_cast<size_t *>(
            rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
        auto *size = static_cast<size_t *>(
            rpc_write_buffer(conn, sizeof(size_t), alignof(size_t)));
        if (param_result != nullptr && offset != nullptr && size != nullptr) {
          *param_result = CUDA_ERROR_INVALID_HANDLE;
          *offset = 0;
          *size = 0;
          write_result = 0;
        }
      }
    }
    if (write_result < 0) {
      return -1;
    }
  }
  if (rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGraphKernelNodeSetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read_kernel_node_params(conn, &nodeParams) < 0 ||
      rpc_read_kernel_node_param_values(conn, &nodeParams, &result) < 0) {
    return -1;
  }
  auto params_owner = lupine_own_kernel_param_values(nodeParams.kernelParams);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
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
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0 ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read_kernel_node_params(conn, &nodeParams) < 0 ||
      rpc_read_kernel_node_param_values(conn, &nodeParams, &result) < 0) {
    return -1;
  }
  auto params_owner = lupine_own_kernel_param_values(nodeParams.kernelParams);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (result == CUDA_SUCCESS) {
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
    void *host = lupine_alloc_process_host_buffer(host_src_bytes);
    if (host == nullptr || rpc_read(conn, host, host_src_bytes) < 0) {
      return -1;
    }
    copyParams.srcHost = host;
  }

  if (copyParams.dstMemoryType == CU_MEMORYTYPE_HOST) {
    size_t host_dst_bytes = lupine_memcpy3d_host_span_bytes(copyParams, false);
    void *host = lupine_alloc_process_host_buffer(host_dst_bytes);
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
  CUgraphNode graphNode = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 ||
      lupine_read_graph_dependencies(conn, &deps) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0) {
    return -1;
  }
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    int read_result = -1;
    if (nodeParams.kernel.func != nullptr) {
      read_result =
          rpc_read_func_param_values(conn, &nodeParams.kernel.kernelParams,
                                     nodeParams.kernel.func, &result);
    }
#if CUDA_VERSION >= 12000
    else if (nodeParams.kernel.kern != nullptr) {
      read_result =
          rpc_read_kernel_param_values(conn, &nodeParams.kernel.kernelParams,
                                       nodeParams.kernel.kern, &result);
    }
#endif
    else {
      read_result = rpc_read_func_param_values(
          conn, &nodeParams.kernel.kernelParams, nullptr, &result);
    }
    if (read_result < 0) {
      return -1;
    }
    if (result == CUDA_SUCCESS) {
      nodeParams.kernel.extra = nullptr;
    }
  }
  auto params_owner = lupine_own_kernel_param_values(
      nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL
          ? nodeParams.kernel.kernelParams
          : nullptr);
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::vector<CUgraph> child_graphs;
  if (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL) {
    if (result == CUDA_SUCCESS) {
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
                   << nodeParams.type << " graph=" << hGraph
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

// Host-node callbacks set after node creation have no graph handle to own their
// trampoline data. The server process is their lifetime boundary, so allocate
// them directly without a synchronized container that never reclaimed them.
static lupine_host_callback_data *
lupine_make_host_setparams_callback(conn_t *conn,
                                    const CUDA_HOST_NODE_PARAMS &params) {
  return new lupine_host_callback_data{conn, params.fn, params.userData,
                                       nullptr};
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

  auto *resources = lupine_get_stream_resources(stream);
  auto *callback = new lupine_host_callback_data{conn, fn, userData, resources};
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

  lupine_graph_resources *resources = nullptr;
  if (lupine_stream_capture_resource_map().find(stream, resources)) {
    lupine_event_capture_resource_map().insert_or_assign(event, resources);
  }

  result = with_flags ? cuEventRecordWithFlags(event, stream, flags)
                      : cuEventRecord(event, stream);
  (void)result;
  (void)request_id;
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
  std::vector<lupine_pending_dtoh_copy> pending;
  if (result == CUDA_SUCCESS) {
    pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  }
  bool failed = rpc_copy_alloc(conn, sizeof(uint32_t)) < 0 ||
                lupine_write_pending_dtoh_copies(conn, pending, true) < 0 ||
                rpc_write(conn, &result, sizeof(result)) < 0 ||
                rpc_write_end(conn) < 0;
  lupine_cleanup_pending_dtoh_copies(&pending);
  return failed ? -1 : 0;
}

int handle_manual_lupineEventQueryBatch(conn_t *conn) {
  uint32_t count = 0;
  if (rpc_read(conn, &count, sizeof(count)) < 0 || count == 0 ||
      count > LUPINE_EVENT_QUERY_BATCH_MAX) {
    return -1;
  }
  CUevent events[LUPINE_EVENT_QUERY_BATCH_MAX];
  if (rpc_read(conn, events, count * sizeof(events[0])) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult results[LUPINE_EVENT_QUERY_BATCH_MAX];
  for (uint32_t i = 0; i < count; ++i) {
    results[i] = cuEventQuery(events[i]);
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, results, count * sizeof(results[0])) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
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

  lupine_graph_resources *event_resources = nullptr;
  if (lupine_event_capture_resource_map().find(event, event_resources)) {
    lupine_stream_capture_resource_map().insert(stream, event_resources);
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

  auto *resources = lupine_get_stream_resources(stream);
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
    lupine_graph_resources *resources = nullptr;
    if (lupine_stream_capture_resource_map().erase_fn(
            stream, [&resources](lupine_graph_resources *&stored) {
              resources = stored;
              return true;
            })) {
      lupine_graph_resource_map().insert_or_assign(graph, resources);
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
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(original, resources)) {
      lupine_graph_resource_map().insert_or_assign(clone, resources);
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
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(graph, resources)) {
      lupine_graph_exec_resource_map().insert_or_assign(exec, resources);
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
    lupine_graph_resources *resources = nullptr;
    if (lupine_graph_resource_map().find(graph, resources)) {
      lupine_graph_exec_resource_map().insert_or_assign(exec, resources);
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
  lupine_graph_exec_resource_map().erase(exec);
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
  lupine_graph_resource_map().erase(graph);
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
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  void *capture_host = nullptr;

  if (rpc_read(conn, &dstDevice, sizeof(dstDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  int framed = lupine_payload_framed(conn, byteCount);
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  CUresult capture_query_result = CUDA_SUCCESS;
  if (stream != nullptr) {
    capture_query_result = cuStreamIsCapturing(stream, &capture_status);
  }
  if (capture_query_result != CUDA_SUCCESS) {
    result = capture_query_result;
    if (rpc_drain_payload(conn, framed, byteCount) < 0) {
      return -1;
    }
  } else if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto *resources = lupine_get_stream_resources(stream);
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
#ifdef _WIN32
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
      result = lupine_defer_host_free(stream, host);
      if (result != CUDA_SUCCESS) {
        cuStreamSynchronize(stream);
        cuMemFreeHost(host);
      }
    }
#else
    if (lupine_server_copy_htod_async(conn, framed, dstDevice, byteCount,
                                      stream, result) < 0) {
      return -1;
    }
#endif
  }

  if (rpc_read_end(conn) < 0) {
    return -1;
  }

  if (capture_query_result == CUDA_SUCCESS &&
      capture_status != CU_STREAM_CAPTURE_STATUS_NONE &&
      result != CUDA_ERROR_OUT_OF_MEMORY) {
    result = cuMemcpyHtoDAsync_v2(dstDevice, capture_host, byteCount, stream);
  }

  return 0;
}

// Fire-and-forget: connection ordering already guarantees the flush is
// applied before any later request, so no response is sent.
int handle_manual_lupineManagedHostFlush(conn_t *conn) {
  uint32_t count = 0;

  if (rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }

  for (uint32_t i = 0; i < count; ++i) {
    void *server_host_ptr = nullptr;
    size_t bytes = 0;
    if (rpc_read(conn, &server_host_ptr, sizeof(server_host_ptr)) < 0 ||
        rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
        rpc_read(conn, server_host_ptr, bytes) < 0) {
      return -1;
    }
  }

  return rpc_read_end(conn) < 0 ? -1 : 0;
}
// Serves LUPINE_RPC_lupineDeviceSnapshot: every immutable per-device value the
// client caches, for every device, in one response. Mutable state (primary
// context state, context limits) is deliberately excluded. The response is all
// or nothing: any query failure fails the whole RPC and the client falls back
// to the per-call paths. Individual attributes the driver rejects are simply
// absent from the pair list; that is expected, not an error.
struct lupine_device_snapshot_record {
  char name[LUPINE_DEVICE_SNAPSHOT_NAME_BYTES] = {};
  CUuuid uuid = {};
  uint64_t total_mem = 0;
  uint32_t pair_count = 0;
  std::vector<int32_t> pairs;
};

static CUresult
lupine_build_device_snapshot_record(size_t ordinal,
                                    lupine_device_snapshot_record *record) {
  CUdevice device = 0;
  size_t bytes = 0;
  CUresult result = cuDeviceGet(&device, static_cast<int>(ordinal));
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetName(record->name, sizeof(record->name), device);
    record->name[sizeof(record->name) - 1] = '\0';
  }
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetUuid_v2(&record->uuid, device);
  }
  if (result == CUDA_SUCCESS) {
    result = cuDeviceTotalMem_v2(&bytes, device);
    record->total_mem = bytes;
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }

  try {
    record->pairs.reserve(static_cast<size_t>(CU_DEVICE_ATTRIBUTE_MAX - 1) * 2);
  } catch (...) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  for (int attrib = 1; attrib < CU_DEVICE_ATTRIBUTE_MAX; ++attrib) {
    int value = 0;
    if (cuDeviceGetAttribute(&value, static_cast<CUdevice_attribute>(attrib),
                             device) == CUDA_SUCCESS) {
      record->pairs.push_back(static_cast<int32_t>(attrib));
      record->pairs.push_back(static_cast<int32_t>(value));
    }
  }
  record->pair_count = static_cast<uint32_t>(record->pairs.size() / 2);
  return CUDA_SUCCESS;
}

int handle_manual_lupineDeviceSnapshot(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  int device_count = 0;
  CUresult result = cuDeviceGetCount(&device_count);
  if (result == CUDA_SUCCESS && device_count < 0) {
    result = CUDA_ERROR_UNKNOWN;
  }

  // rpc_write queues iovecs that are only sent at rpc_write_end, so all
  // records are built first in storage that stays stable until then.
  std::vector<lupine_device_snapshot_record> records;
  std::vector<CUresult> record_results;
  if (result == CUDA_SUCCESS) {
    try {
      records.resize(static_cast<size_t>(device_count));
      record_results.resize(records.size(), CUDA_ERROR_UNKNOWN);
    } catch (...) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    }
  }
  if (result == CUDA_SUCCESS && !records.empty()) {
    std::vector<std::thread> workers;
    auto build_record = [&records, &record_results](size_t ordinal) {
      record_results[ordinal] =
          lupine_build_device_snapshot_record(ordinal, &records[ordinal]);
    };
    size_t next_ordinal = 1;
    try {
      workers.reserve(records.size() - 1);
      for (; next_ordinal < records.size(); ++next_ordinal) {
        workers.emplace_back(build_record, next_ordinal);
      }
    } catch (...) {
      // Any unlaunched devices fall back to this RPC thread below.
    }

    build_record(0);
    for (size_t ordinal = next_ordinal; ordinal < records.size(); ++ordinal) {
      build_record(ordinal);
    }
    for (auto &worker : workers) {
      worker.join();
    }
    for (CUresult record_result : record_results) {
      if (record_result != CUDA_SUCCESS) {
        result = record_result;
        break;
      }
    }
  }

  uint32_t devices = static_cast<uint32_t>(records.size());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0) {
    return -1;
  }
  if (result != CUDA_SUCCESS) {
    return rpc_write_end(conn) < 0 ? -1 : 0;
  }
  if (rpc_write(conn, &devices, sizeof(devices)) < 0) {
    return -1;
  }
  for (const auto &record : records) {
    if (rpc_write(conn, record.name, sizeof(record.name)) < 0 ||
        rpc_write(conn, &record.uuid, sizeof(record.uuid)) < 0 ||
        rpc_write(conn, &record.total_mem, sizeof(record.total_mem)) < 0 ||
        rpc_write(conn, &record.pair_count, sizeof(record.pair_count)) < 0 ||
        (record.pair_count != 0 &&
         rpc_write(conn, record.pairs.data(),
                   record.pairs.size() * sizeof(int32_t)) < 0)) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
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
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &dstHost, sizeof(dstHost)) < 0 ||
      rpc_read(conn, &srcDevice, sizeof(srcDevice)) < 0 ||
      rpc_read(conn, &byteCount, sizeof(byteCount)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }

  if (rpc_read_end(conn) < 0) {
    return -1;
  }

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (stream != nullptr) {
    cuStreamIsCapturing(stream, &capture_status);
  }

  void *host = nullptr;
  CUresult alloc_result = CUDA_ERROR_INVALID_VALUE;
  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    auto *resources = lupine_get_stream_resources(stream);
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
        lupine_pending_dtoh_copy copy{stream, dstHost, host, byteCount,
                                      alloc_result == CUDA_SUCCESS};
        lupine_pending_dtoh_copies().upsert(
            conn,
            [stream, &copy](lupine_pending_dtoh_streams &streams,
                            libcuckoo::UpsertContext) {
              streams[stream].push_back(copy);
            },
            lupine_pending_dtoh_streams{});
        host = nullptr;
      }
    }
  }

  // A fire-and-forget copy drops an immediate validation error, matching launch
  // semantics: an execution failure poisons the context and the driver reports
  // it from the client's next synchronize.
  if (alloc_result == CUDA_SUCCESS && host != nullptr) {
    cuMemFreeHost(host);
  } else if (host != nullptr) {
    free(host);
  }
  return 0;
}

// Resolve the device alias here so the client does not need a second round
// trip for it. A mapped allocation whose alias cannot be resolved still
// succeeds; the 0 tells the client to query it on first use instead.
int handle_manual_cuMemHostAlloc(conn_t *conn) {
  void *pp = nullptr;
  size_t bytesize = 0;
  unsigned int flags = 0;
  if (rpc_read(conn, &pp, sizeof(pp)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(bytesize)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUdeviceptr device_ptr = 0;
  CUresult result = cuMemHostAlloc(&pp, bytesize, flags);
  if (result == CUDA_SUCCESS && (flags & CU_MEMHOSTALLOC_DEVICEMAP) != 0 &&
      cuMemHostGetDevicePointer(&device_ptr, pp, 0) != CUDA_SUCCESS) {
    device_ptr = 0;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pp, sizeof(pp)) < 0 ||
      rpc_write(conn, &device_ptr, sizeof(device_ptr)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuMemHostGetFlags(conn_t *conn) {
  unsigned int flags = 0;
  void *p = nullptr;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &p, sizeof(p)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUresult result = cuMemHostGetFlags(&flags, p);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &flags, sizeof(flags)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
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
  auto pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  bool failed = rpc_write_start_response(conn, request_id) < 0 ||
                rpc_copy_alloc(conn, sizeof(uint32_t) + sizeof(uint64_t)) < 0 ||
                lupine_write_pending_dtoh_copies(conn, pending, true) < 0 ||
                lupine_write_captured_stdout(conn, capture) < 0 ||
                rpc_write(conn, &result, sizeof(result)) < 0 ||
                rpc_write_end(conn) < 0;
  lupine_cleanup_pending_dtoh_copies(&pending);
  return failed ? -1 : 0;
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
  lupine_graph_resources *resources = nullptr;
  uint32_t copy_count = 0;
  lupine_stream_capture_resource_map().find(stream, resources);
  std::vector<lupine_graph_host_copy> graph_copies =
      resources == nullptr ? std::vector<lupine_graph_host_copy>()
                           : resources->dtoh_copy_snapshot();
  uint32_t graph_copy_count = static_cast<uint32_t>(graph_copies.size());
  bool all_pending_streams = stream == nullptr;
  auto pending =
      lupine_detach_pending_dtoh_copies(conn, stream, all_pending_streams);
  uint32_t pending_copy_count = static_cast<uint32_t>(pending.size());
  copy_count = graph_copy_count + pending_copy_count;
  bool failed =
      rpc_write_start_response(conn, request_id) < 0 ||
      rpc_copy_alloc(conn, sizeof(uint64_t)) < 0 ||
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
      lupine_write_pending_dtoh_copies(conn, pending, false) < 0 ||
      lupine_write_captured_stdout(conn, capture) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0;
  lupine_cleanup_pending_dtoh_copies(&pending);
  return failed ? -1 : 0;
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
  lupine_graph_resources *resources = nullptr;
  if (result == CUDA_SUCCESS &&
      lupine_graph_exec_resource_map().find(exec, resources)) {
    lupine_stream_capture_resource_map().insert_or_assign(stream, resources);
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
  auto pending = lupine_detach_pending_dtoh_copies(conn, nullptr, true);
  bool failed = rpc_write_start_response(conn, request_id) < 0 ||
                rpc_copy_alloc(conn, sizeof(uint32_t) + sizeof(uint64_t)) < 0 ||
                lupine_write_pending_dtoh_copies(conn, pending, true) < 0 ||
                lupine_write_captured_stdout(conn, capture) < 0 ||
                rpc_write(conn, &result, sizeof(result)) < 0 ||
                rpc_write_end(conn) < 0;
  lupine_cleanup_pending_dtoh_copies(&pending);
  return failed ? -1 : 0;
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

// The generated marshaller cannot receive a string of unknown length, and the
// driver hands back a static pointer rather than filling a caller buffer, so
// these two forward the answer as an explicit length plus bytes.
static int lupine_handle_error_string(conn_t *conn,
                                      CUresult (*lookup)(CUresult,
                                                         const char **)) {
  CUresult error;
  if (rpc_read(conn, &error, sizeof(error)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  const char *text = nullptr;
  CUresult result = lookup(error, &text);
  uint32_t length = (result == CUDA_SUCCESS && text != nullptr)
                        ? static_cast<uint32_t>(strlen(text))
                        : 0;

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_write(conn, text, length) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_manual_cuGetErrorName(conn_t *conn) {
  return lupine_handle_error_string(conn, cuGetErrorName);
}

int handle_manual_cuGetErrorString(conn_t *conn) {
  return lupine_handle_error_string(conn, cuGetErrorString);
}
