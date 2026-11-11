#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <errno.h>
#if defined(__linux__)
#include <sys/mman.h> // memfd_create
#include <sys/syscall.h>
#endif
#include <future>
#include <iostream>
#include <iterator>
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

#include "cuda_compat.h"

#include "cache.h"
#include "codegen/gen_rpc_ids.h"
#include "cuda_server.h"
#include "cuda_server_memcpy.h"
#include "events.h"
#include "ipc.h"
#include "lupine_attr_sizes.h"
#include "lupine_fatbin.h"
#include "lupine_log.h"
#include "monitoring.h"
#include "rpc.h"
#include "server_checkpoint.h"

#ifdef _WIN32
#include <io.h>
#define lupine_ipc_close_fd _close
#else
#include <unistd.h>
#define lupine_ipc_close_fd close
#endif
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

#if defined(__linux__)
static thread_local conn_t *lupine_server_va_connection = nullptr;
#endif

// The mmap interposer places the driver's managed VA reservations made inside
// `allocate` in the connection's identity arena: cuMemAllocManaged and the
// __managed__ globals a module load creates.
template <typename F>
static CUresult lupine_in_va_arena(conn_t *conn, F &&allocate) {
#if defined(__linux__)
  if (conn->va_size != 0) {
    conn_t *previous = lupine_server_va_connection;
    lupine_server_va_connection = conn;
    CUresult result = allocate();
    lupine_server_va_connection = previous;
    return result;
  }
#endif
  return allocate();
}

namespace {

// Metadata warming uses expected failing CUDA queries to find parameter-list
// boundaries and unsupported attributes. These are Lupine's queries, rather
// than application calls, and must not be surfaced to application log hooks.
thread_local unsigned lupine_internal_cuda_query_depth = 0;

class lupine_internal_cuda_query {
public:
  lupine_internal_cuda_query() { ++lupine_internal_cuda_query_depth; }
  ~lupine_internal_cuda_query() { --lupine_internal_cuda_query_depth; }
  lupine_internal_cuda_query(const lupine_internal_cuda_query &) = delete;
  lupine_internal_cuda_query &
  operator=(const lupine_internal_cuda_query &) = delete;
};

bool lupine_is_internal_cuda_query() {
  return lupine_internal_cuda_query_depth != 0;
}

} // namespace

#ifdef cuGraphInstantiate_v2
#undef cuGraphInstantiate_v2
#endif
extern "C" CUresult CUDAAPI cuGraphInstantiate_v2(CUgraphExec *phGraphExec,
                                                  CUgraph hGraph,
                                                  CUgraphNode *phErrorNode,
                                                  char *logBuffer,
                                                  size_t bufferSize);

static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V1 = 1;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBIN_RAW = 2;
static constexpr uint32_t LUPINE_MODULE_IMAGE_FATBINC_V2 = 3;
static constexpr uint32_t LUPINE_PRIVATE_EXPORT_MAX_SLOTS = 256;

template <typename T, typename Tag> static T &lupine_leaked(Tag) {
  static T *instance = new T();
  return *instance;
}

struct lupine_raw_bytes {
  const void *data;
  size_t size;
};

template <typename T> static int lupine_write_one(conn_t *conn, const T &val) {
  return rpc_write(conn, &val, sizeof(val));
}

static int lupine_write_one(conn_t *conn, lupine_raw_bytes raw) {
  return raw.size == 0 ? 0 : rpc_write(conn, raw.data, raw.size);
}

template <typename... Args>
static bool lupine_read_all(conn_t *conn, Args &...args) {
  return ((rpc_read(conn, &args, sizeof(args)) >= 0) && ...);
}

template <typename... Args>
static int lupine_write_response(conn_t *conn, int request_id,
                                 const Args &...args) {
  return rpc_write_start_response(conn, request_id) < 0 ||
                 ((lupine_write_one(conn, args) < 0) || ...) ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

template <typename T, size_t N, size_t Align = alignof(T)>
struct lupine_inline_buf {
  alignas(Align) T stack[N];
  std::unique_ptr<T[]> heap;
  size_t cap = N;
  T *ptr = stack;

  T *alloc(size_t n) {
    if (n <= cap) {
      ptr = cap == N ? stack : heap.get();
      return ptr;
    }
    if (n > SIZE_MAX / sizeof(T)) {
      return nullptr;
    }
    heap.reset(new (std::nothrow) T[n]);
    if (!heap) {
      cap = N;
      return nullptr;
    }
    cap = n;
    return ptr = heap.get();
  }

  void reset() {
    heap.reset();
    cap = N;
    ptr = stack;
  }
};

template <typename T, size_t N> struct lupine_inline_vec {
  T stack[N];
  std::vector<T> overflow;
  size_t count = 0;

  void push(const T &val) {
    if (overflow.empty() && count < N) {
      stack[count++] = val;
      return;
    }
    if (overflow.empty()) {
      overflow.reserve(N * 2);
      overflow.assign(stack, stack + count);
    }
    overflow.push_back(val);
    ++count;
  }

  const T *data() const { return overflow.empty() ? stack : overflow.data(); }
};

struct lupine_attribute_record {
  CUresult result = CUDA_ERROR_UNKNOWN;
  int attribute = 0;
  int value = 0;
};
static_assert(sizeof(lupine_attribute_record) == 12);

static int
lupine_write_attribute_records(conn_t *conn,
                               const lupine_attribute_record *attributes,
                               const uint32_t &attribute_count) {
  if (rpc_write(conn, &attribute_count, sizeof(attribute_count)) < 0) {
    return -1;
  }
  return attribute_count == 0 ||
                 rpc_write(conn, attributes,
                           attribute_count * sizeof(lupine_attribute_record)) >=
                     0
             ? 0
             : -1;
}

struct lupine_jit_state {
  static constexpr unsigned int kInlineCap = 16;
  unsigned int num_options = 0;
  lupine_inline_buf<CUjit_option, kInlineCap> opt_buf;
  lupine_inline_buf<void *, kInlineCap> val_buf;
  lupine_inline_buf<size_t, kInlineCap> len_buf;
  CUjit_option *options = nullptr;
  void **option_values = nullptr;
  size_t *output_lengths = nullptr;
  size_t info_log_size = 0;
  char *info_log = nullptr;
  size_t error_log_size = 0;
  char *error_log = nullptr;

  lupine_jit_state() = default;
  lupine_jit_state(const lupine_jit_state &) = delete;
  lupine_jit_state &operator=(const lupine_jit_state &) = delete;
  ~lupine_jit_state() { clear(); }

  void clear() {
    opt_buf.reset();
    val_buf.reset();
    len_buf.reset();
    std::free(info_log);
    std::free(error_log);
    num_options = 0;
    options = nullptr;
    option_values = nullptr;
    output_lengths = nullptr;
    info_log_size = 0;
    info_log = nullptr;
    error_log_size = 0;
    error_log = nullptr;
  }
};

static int lupine_alloc_jit_log(char **log_buf, size_t size, void **slot) {
  if (slot == nullptr) {
    return 0;
  }
  *log_buf = static_cast<char *>(std::malloc(size));
  if (size != 0) {
    if (*log_buf == nullptr) {
      return -1;
    }
    std::memset(*log_buf, 0, size);
  }
  *slot = *log_buf;
  return 0;
}

static int lupine_read_jit_options(conn_t *conn, lupine_jit_state *jit) {
  if (rpc_read(conn, &jit->num_options, sizeof(jit->num_options)) < 0) {
    return -1;
  }
  if (jit->num_options == 0) {
    jit->options = nullptr;
    jit->option_values = nullptr;
    jit->output_lengths = nullptr;
    return 0;
  }
  jit->options = jit->opt_buf.alloc(jit->num_options);
  jit->option_values = jit->val_buf.alloc(jit->num_options);
  jit->output_lengths = jit->len_buf.alloc(jit->num_options);
  if (!jit->options || !jit->option_values || !jit->output_lengths ||
      rpc_read(conn, jit->options, jit->num_options * sizeof(*jit->options)) <
          0 ||
      rpc_read(conn, jit->option_values,
               jit->num_options * sizeof(*jit->option_values)) < 0) {
    return -1;
  }
  void **info_slot = nullptr;
  void **error_slot = nullptr;
  for (unsigned int i = 0; i < jit->num_options; ++i) {
    if (jit->options[i] == CU_JIT_INFO_LOG_BUFFER && jit->option_values[i]) {
      info_slot = &jit->option_values[i];
    } else if (jit->options[i] == CU_JIT_ERROR_LOG_BUFFER &&
               jit->option_values[i]) {
      error_slot = &jit->option_values[i];
    } else if (jit->options[i] == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES) {
      jit->info_log_size = reinterpret_cast<uintptr_t>(jit->option_values[i]);
    } else if (jit->options[i] == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES) {
      jit->error_log_size = reinterpret_cast<uintptr_t>(jit->option_values[i]);
    }
  }
  return lupine_alloc_jit_log(&jit->info_log, jit->info_log_size, info_slot) <
                     0 ||
                 lupine_alloc_jit_log(&jit->error_log, jit->error_log_size,
                                      error_slot) < 0
             ? -1
             : 0;
}

// Every JIT option value the driver may rewrite is echoed back: the log
// buffers as the bytes written into them, everything else as the option value
// word the driver left in the server's array (CU_JIT_WALL_TIME and the log
// sizes are rewritten there by value, not through the caller's pointer).
static int lupine_write_jit_outputs(conn_t *conn, lupine_jit_state *jit) {
  for (unsigned int i = 0; i < jit->num_options; ++i) {
    const bool is_log = jit->options[i] == CU_JIT_INFO_LOG_BUFFER ||
                        jit->options[i] == CU_JIT_ERROR_LOG_BUFFER;
    const void *output = &jit->option_values[i];
    size_t output_length = sizeof(jit->option_values[i]);
    if (jit->options[i] == CU_JIT_INFO_LOG_BUFFER) {
      output = jit->info_log;
      output_length = output == nullptr ? 0 : jit->info_log_size;
    } else if (jit->options[i] == CU_JIT_ERROR_LOG_BUFFER) {
      output = jit->error_log;
      output_length = output == nullptr ? 0 : jit->error_log_size;
    }
    if (is_log && output != nullptr && output_length != 0) {
      const auto *end =
          static_cast<const char *>(std::memchr(output, '\0', output_length));
      if (end != nullptr) {
        output_length = end - static_cast<const char *>(output);
      }
    }
    jit->output_lengths[i] = output_length;
    if (rpc_write(conn, &jit->output_lengths[i],
                  sizeof(jit->output_lengths[i])) < 0 ||
        rpc_write(conn, output, output_length) < 0) {
      return -1;
    }
  }
  return 0;
}

// The CUDA linker requires output option buffers to remain valid for the
// lifetime of CUlinkState. The mutex serializes cuLinkDestroy against a
// concurrent cuLinkComplete on another RPC lane: the cubin buffer returned by
// cuLinkComplete is driver-owned and freed by cuLinkDestroy, so it must stay
// locked out until the response holding that buffer has been flushed.
struct lupine_link_state {
  CUlinkState cuda_state = nullptr;
  lupine_jit_state jit;
  std::mutex mutex;
};

static lupine_link_state *lupine_link_state_from_handle(CUlinkState state) {
  return reinterpret_cast<lupine_link_state *>(state);
}

static CUlinkState lupine_link_state_to_handle(lupine_link_state *state) {
  return reinterpret_cast<CUlinkState>(state);
}

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
    lupine_require_stdout_capture();
  }
}

struct lupine_private_module_node_capture {
  void *node = nullptr;
  uint64_t owner = 0;
  uint64_t count = 0;
};

#if CUDA_VERSION >= 12090
struct lupine_logs_callback_data {
  std::atomic<conn_t *> conn;
  CUlogsCallback callback = nullptr;
  void *user_data = nullptr;
};

static std::mutex &lupine_logs_callback_mutex() {
  return lupine_leaked<std::mutex>([] {});
}

static std::unordered_map<CUlogsCallbackHandle, lupine_logs_callback_data *> &
lupine_logs_callbacks() {
  return lupine_leaked<
      std::unordered_map<CUlogsCallbackHandle, lupine_logs_callback_data *>>(
      [] {});
}
#endif

// A fire-and-forget failure is reported by the next synchronizing response.
bool lupine_stream_query_unsettled(conn_t *conn, CUstream stream) {
  CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
  return rpc_request_unsettled(conn) &&
         cuStreamIsCapturing(stream, &capture) == CUDA_SUCCESS &&
         capture == CU_STREAM_CAPTURE_STATUS_NONE;
}

void lupine_keep_async_error(conn_t *conn, CUresult result) {
  int none = CUDA_SUCCESS;
  if (result != CUDA_SUCCESS) {
    __atomic_compare_exchange_n(&conn->first_async_error, &none,
                                static_cast<int>(result), false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
}

bool lupine_unanswered(conn_t *conn, CUresult result) {
  if (rpc_request_awaited(conn)) {
    return false;
  }
  lupine_keep_async_error(conn, result);
  return true;
}

CUresult lupine_take_async_error(conn_t *conn, CUresult result) {
  if (__atomic_load_n(&conn->first_async_error, __ATOMIC_RELAXED) == 0) {
    return result;
  }
  int error =
      __atomic_exchange_n(&conn->first_async_error, 0, __ATOMIC_RELAXED);
  return error == CUDA_SUCCESS ? result : static_cast<CUresult>(error);
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

int handle_cuGetExportTableMetadata(conn_t *conn) {
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

  return lupine_write_response(
      conn, request_id, result, byte_size, slot_count, trusted,
      lupine_raw_bytes{hashes,
                       static_cast<size_t>(slot_count) * sizeof(uint64_t)});
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

int handle_cuPrivateGetModuleNode(conn_t *conn) {
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

  return lupine_write_response(conn, request_id, node, owner, result);
}

struct lupine_server_graph_deps {
  lupine_inline_buf<CUgraphNode, 16> buf;
  size_t count = 0;

  int read(conn_t *conn) {
    if (rpc_read(conn, &count, sizeof(count)) < 0) {
      return -1;
    }
    CUgraphNode *nodes = buf.alloc(count);
    if (!nodes || (count != 0 &&
                   rpc_read(conn, nodes, count * sizeof(CUgraphNode)) < 0)) {
      return -1;
    }
    lupine_graph_push_dependencies(nodes, count);
    return 0;
  }

  const CUgraphNode *data() const { return count != 0 ? buf.ptr : nullptr; }
  size_t size() const { return count; }
};

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

#if CUDA_VERSION >= 12040
using lupine_attribute_records =
    std::array<lupine_attribute_record, CU_FUNC_ATTRIBUTE_MAX>;
struct lupine_param_record {
  size_t offset = 0;
  size_t size = 0;
};
static_assert(sizeof(lupine_param_record) == 16);
struct lupine_kernel_record {
  const char *name = nullptr;
  uint32_t name_length = 0;
  CUkernel kernel = nullptr;
  CUfunction function = nullptr;
  uint32_t param_count = 0;
  uint32_t function_attribute_count = 0;
  uint32_t kernel_attribute_count = 0;
  std::vector<lupine_param_record> params;
  lupine_attribute_records function_attributes;
  lupine_attribute_records kernel_attributes;
};

static constexpr uint32_t LUPINE_PARAM_COUNT_UNKNOWN = UINT32_MAX;

template <typename Query>
static void lupine_collect_params(lupine_kernel_record &record, Query query) {
  lupine_internal_cuda_query probe;
  for (lupine_param_record param;; ++record.param_count) {
    CUresult result = query(record.param_count, &param.offset, &param.size);
    if (result == CUDA_SUCCESS) {
      record.params.push_back(param);
      continue;
    }
    if (result != CUDA_ERROR_INVALID_VALUE) {
      record.params.clear();
      record.param_count = LUPINE_PARAM_COUNT_UNKNOWN;
    }
    return;
  }
}

template <typename Query>
static void lupine_collect_attributes(lupine_attribute_records &attributes,
                                      uint32_t attribute_count, Query query) {
  lupine_internal_cuda_query probe;
  for (uint32_t i = 0; i < attribute_count; ++i) {
    attributes[i].attribute = static_cast<int>(i);
    attributes[i].result =
        query(&attributes[i].value, static_cast<CUfunction_attribute>(i));
  }
}

static int lupine_write_attributes(conn_t *conn,
                                   const lupine_attribute_records &attributes,
                                   const uint32_t &attribute_count) {
  return lupine_write_attribute_records(conn, attributes.data(),
                                        attribute_count);
}

static int lupine_write_params(conn_t *conn,
                               const lupine_kernel_record &record) {
  return record.params.empty() ||
                 rpc_write(conn, record.params.data(),
                           record.params.size() *
                               sizeof(lupine_param_record)) >= 0
             ? 0
             : -1;
}
static int lupine_write_kernel_record(conn_t *conn,
                                      const lupine_kernel_record &record,
                                      bool is_library) {
  return rpc_write(conn, &record.name_length, sizeof(record.name_length)) < 0 ||
                 rpc_write(conn, record.name, record.name_length) < 0 ||
                 (is_library &&
                  rpc_write(conn, &record.kernel, sizeof(record.kernel)) < 0) ||
                 rpc_write(conn, &record.function, sizeof(record.function)) <
                     0 ||
                 rpc_write(conn, &record.param_count,
                           sizeof(record.param_count)) < 0 ||
                 lupine_write_params(conn, record) < 0 ||
                 lupine_write_attributes(conn, record.function_attributes,
                                         record.function_attribute_count) < 0 ||
                 (is_library &&
                  lupine_write_attributes(conn, record.kernel_attributes,
                                          record.kernel_attribute_count) < 0)
             ? -1
             : 0;
}
#endif

struct lupine_module_snapshot {
  uint32_t count = 0;
#if CUDA_VERSION >= 12040
  std::vector<lupine_kernel_record> records;
#endif
};

// A module is immutable once loaded, so its function set, each function's
// parameter layout, and its attributes cannot change before the module
// unloads. Driver versions without the enumeration entry points report no
// functions at all and the client falls back to asking per name.
//
// cuFuncLoad reproduces the side effect the client's cuModuleGetFunction would
// have had: under the default lazy module loading a function's code is not
// resident until it is looked up by name, and an unloaded function answers
// CUDA_ERROR_FUNCTION_NOT_LOADED to every attribute query. A function that
// will not load is left out of the snapshot entirely rather than reported with
// answers that came from an unloaded function.
//
// Like a library's, this runs before the response takes the RPC write lock.
static lupine_module_snapshot lupine_collect_module(CUmodule module) {
  lupine_internal_cuda_query probe;
  lupine_module_snapshot snapshot;
#if CUDA_VERSION >= 12040
  unsigned int function_count = 0;
  if (module == nullptr ||
      cuModuleGetFunctionCount(&function_count, module) != CUDA_SUCCESS ||
      function_count == 0) {
    return snapshot;
  }
  lupine_inline_buf<CUfunction, 64> fn_buf;
  CUfunction *functions = fn_buf.alloc(function_count);
  if (!functions || cuModuleEnumerateFunctions(functions, function_count,
                                               module) != CUDA_SUCCESS) {
    return snapshot;
  }
  for (unsigned int i = 0; i < function_count; ++i) {
    CUfunction function = functions[i];
    const char *name = nullptr;
    if (function == nullptr || cuFuncGetName(&name, function) != CUDA_SUCCESS ||
        name == nullptr || cuFuncLoad(function) != CUDA_SUCCESS) {
      continue;
    }
    auto &record = snapshot.records.emplace_back();
    record.name = name;
    record.name_length = static_cast<uint32_t>(std::strlen(name) + 1);
    record.function = function;
    record.function_attribute_count = CU_FUNC_ATTRIBUTE_MAX;
    lupine_collect_params(
        record, [function](size_t index, size_t *offset, size_t *size) {
          return cuFuncGetParamInfo(function, index, offset, size);
        });
    lupine_collect_attributes(
        record.function_attributes, record.function_attribute_count,
        [function](int *value, CUfunction_attribute attribute) {
          return cuFuncGetAttribute(value, attribute, function);
        });
  }
  snapshot.count = static_cast<uint32_t>(snapshot.records.size());
#endif
  return snapshot;
}

// Queue fields from stable typed storage. Names remain owned by the loaded
// module, and the snapshot stays alive until the response is submitted.
static int lupine_write_module(conn_t *conn,
                               const lupine_module_snapshot &snapshot) {
  if (rpc_write(conn, &snapshot.count, sizeof(snapshot.count)) < 0) {
    return -1;
  }
#if CUDA_VERSION >= 12040
  for (const auto &record : snapshot.records) {
    if (lupine_write_kernel_record(conn, record, false) < 0) {
      return -1;
    }
  }
#endif
  return 0;
}

static int lupine_finish_module_load(conn_t *conn, int request_id,
                                     CUmodule module, CUresult result,
                                     const unsigned char *image,
                                     size_t image_size,
                                     lupine_jit_state *jit = nullptr) {
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image, image_size);
  }
  auto snapshot =
      lupine_collect_module(result == CUDA_SUCCESS ? module : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      (jit != nullptr && lupine_write_jit_outputs(conn, jit) < 0) ||
      rpc_write(conn, &module, sizeof(module)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      lupine_write_module(conn, snapshot) < 0) {
    return -1;
  }
  return rpc_write_end(conn);
}

static int lupine_handle_module_load(conn_t *conn, bool has_kind,
                                     bool has_jit) {
  uint32_t kind = LUPINE_MODULE_IMAGE_FATBIN_RAW;
  size_t image_size = 0;
  lupine_jit_state jit;
  CUmodule module = nullptr;

  if ((has_kind && rpc_read(conn, &kind, sizeof(kind)) < 0) ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0 || image_size == 0) {
    return -1;
  }

  size_t alloc_size = has_kind ? image_size : image_size + 1;
  if (alloc_size < image_size) {
    return -1;
  }
  std::unique_ptr<unsigned char[]> image(
      new (std::nothrow) unsigned char[alloc_size]);
  if (!image || rpc_read(conn, image.get(), image_size) < 0 ||
      (has_jit && lupine_read_jit_options(conn, &jit) < 0)) {
    return -1;
  }
  if (!has_kind) {
    image[image_size] = 0;
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult result = lupine_in_va_arena(conn, [&] {
    if (has_jit) {
      // The client always unwraps fat binary containers, and
      // cuModuleLoadDataEx takes a cubin, fatbin, or PTX image, so the kind
      // tag makes no difference here.
      return cuModuleLoadDataEx(&module, image.get(), jit.num_options,
                                jit.options, jit.option_values);
    }
    if (kind == LUPINE_MODULE_IMAGE_FATBINC_V1 ||
        kind == LUPINE_MODULE_IMAGE_FATBINC_V2) {
      return cuModuleLoadFatBinary(&module, image.get());
    }
    if (kind == LUPINE_MODULE_IMAGE_FATBIN_RAW) {
      return cuModuleLoadData(&module, image.get());
    }
    return CUDA_ERROR_NOT_SUPPORTED;
  });
  return lupine_finish_module_load(conn, request_id, module, result,
                                   image.get(), image_size,
                                   has_jit ? &jit : nullptr);
}

int handle_cuModuleLoad(conn_t *conn) {
  return lupine_handle_module_load(conn, false, false);
}

int handle_cuModuleLoadData(conn_t *conn) {
  return lupine_handle_module_load(conn, true, false);
}

int handle_cuModuleLoadDataEx(conn_t *conn) {
  return lupine_handle_module_load(conn, true, true);
}

int handle_lupineFunctionParamLayoutSnapshot(conn_t *conn) {
  CUfunction function = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &function, sizeof(function)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  struct wire_rec {
    unsigned char bytes[2 * sizeof(size_t) + sizeof(CUresult)];
  };
  lupine_inline_vec<wire_rec, 65> records;
  {
    lupine_internal_cuda_query probe;
    for (uint32_t i = 0;; ++i) {
      size_t offset = 0, size = 0;
      CUresult param_result = cuFuncGetParamInfo(function, i, &offset, &size);
      wire_rec rec;
      std::memcpy(rec.bytes, &offset, sizeof(offset));
      std::memcpy(rec.bytes + sizeof(offset), &size, sizeof(size));
      std::memcpy(rec.bytes + 2 * sizeof(size_t), &param_result,
                  sizeof(param_result));
      records.push(rec);
      if (param_result != CUDA_SUCCESS) {
        break;
      }
    }
  }

  return lupine_write_response(
      conn, request_id,
      lupine_raw_bytes{records.data(), records.count * sizeof(wire_rec)});
}

int handle_lupineFunctionAttributeSnapshot(conn_t *conn) {
  CUfunction function = nullptr;
  if (rpc_read(conn, &function, sizeof(function)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  std::array<lupine_attribute_record,
             static_cast<size_t>(CU_FUNC_ATTRIBUTE_MAX) + 1>
      records;
  uint32_t count = static_cast<uint32_t>(records.size());
  {
    lupine_internal_cuda_query probe;
    for (int attribute = LUPINE_FUNC_ATTRIBUTE_DEVICE;
         attribute < CU_FUNC_ATTRIBUTE_MAX; ++attribute) {
      auto &entry = records[static_cast<size_t>(attribute -
                                                LUPINE_FUNC_ATTRIBUTE_DEVICE)];
      entry.attribute = attribute;
      entry.value = 0;
      entry.result =
          attribute == LUPINE_FUNC_ATTRIBUTE_DEVICE
              ? cuCtxGetDevice(&entry.value)
              : cuFuncGetAttribute(&entry.value,
                                   static_cast<CUfunction_attribute>(attribute),
                                   function);
    }
  }

  if (rpc_write_start_response(conn, request_id) < 0 ||
      lupine_write_attribute_records(conn, records.data(), count) < 0) {
    return -1;
  }
  return rpc_write_end(conn);
}

struct lupine_library_snapshot {
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;
  CUdevice device = -1;
  uint32_t count = 0;
#if CUDA_VERSION >= 12040
  std::vector<lupine_kernel_record> records;
#endif
};

// Resolving a kernel to a function can synchronize CUDA work, including a
// pinned HtoD callback that needs the RPC write lock to fetch client bytes.
// Finish every CUDA query before starting the response and taking that lock.
static lupine_library_snapshot lupine_collect_library(CUlibrary library) {
  lupine_internal_cuda_query probe;
  lupine_library_snapshot snapshot;
#if CUDA_VERSION >= 12040
  CUdevice current_device = -1;
  if (cuCtxGetDevice(&current_device) == CUDA_SUCCESS) {
    snapshot.device = current_device;
  }
  unsigned int kernel_count = 0;
  snapshot.result = cuLibraryGetKernelCount(&kernel_count, library);
  lupine_inline_buf<CUkernel, 64> kern_buf;
  CUkernel *kernels = nullptr;
  if (snapshot.result == CUDA_SUCCESS && kernel_count != 0) {
    kernels = kern_buf.alloc(kernel_count);
    snapshot.result =
        kernels ? cuLibraryEnumerateKernels(kernels, kernel_count, library)
                : CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (snapshot.result != CUDA_SUCCESS) {
    return snapshot;
  }
  snapshot.records.reserve(kernel_count);
  for (unsigned int i = 0; i < kernel_count; ++i) {
    CUkernel kernel = kernels[i];
    const char *name = nullptr;
    if (kernel == nullptr || cuKernelGetName(&name, kernel) != CUDA_SUCCESS ||
        name == nullptr) {
      continue;
    }
    auto &record = snapshot.records.emplace_back();
    record.name = name;
    record.name_length = static_cast<uint32_t>(std::strlen(name) + 1);
    record.kernel = kernel;
    if (cuKernelGetFunction(&record.function, kernel) != CUDA_SUCCESS) {
      record.function = nullptr;
    }
    lupine_collect_params(
        record, [kernel](size_t index, size_t *offset, size_t *size) {
          return cuKernelGetParamInfo(kernel, index, offset, size);
        });
    record.function_attribute_count =
        record.function != nullptr ? CU_FUNC_ATTRIBUTE_MAX : 0;
    record.kernel_attribute_count =
        snapshot.device >= 0 ? CU_FUNC_ATTRIBUTE_MAX : 0;
    lupine_collect_attributes(
        record.function_attributes, record.function_attribute_count,
        [&](int *value, CUfunction_attribute attribute) {
          return cuFuncGetAttribute(value, attribute, record.function);
        });
    lupine_collect_attributes(record.kernel_attributes,
                              record.kernel_attribute_count,
                              [&](int *value, CUfunction_attribute attribute) {
                                return cuKernelGetAttribute(
                                    value, attribute, kernel, snapshot.device);
                              });
  }
  snapshot.count = static_cast<uint32_t>(snapshot.records.size());
#endif
  return snapshot;
}

// Queue fields from stable typed storage. Names remain owned by the loaded
// library, and the snapshot stays alive until the response is submitted.
static int lupine_write_library(conn_t *conn,
                                const lupine_library_snapshot &snapshot) {
  if (rpc_write(conn, &snapshot.result, sizeof(snapshot.result)) < 0 ||
      rpc_write(conn, &snapshot.device, sizeof(snapshot.device)) < 0 ||
      rpc_write(conn, &snapshot.count, sizeof(snapshot.count)) < 0) {
    return -1;
  }
#if CUDA_VERSION >= 12040
  for (const auto &record : snapshot.records) {
    if (lupine_write_kernel_record(conn, record, true) < 0) {
      return -1;
    }
  }
#endif
  return 0;
}

struct lupine_library_options {
  unsigned int count = 0;
  bool has_values = false;
  lupine_inline_buf<CUlibraryOption, 16> opt_buf;
  lupine_inline_buf<void *, 16> val_buf;

  CUlibraryOption *options() { return opt_buf.ptr; }
  void **values() { return val_buf.ptr; }
};

static CUresult lupine_load_library(conn_t *conn, CUlibrary *library,
                                    uint32_t kind, const unsigned char *image,
                                    size_t image_size,
                                    const lupine_jit_state &jit,
                                    lupine_library_options &options) {
  lupine_fatbin_wrapper wrapper = {
      LUPINE_FATBINC_MAGIC,
      kind == LUPINE_MODULE_IMAGE_FATBINC_V2 ? 2U : 1U,
      image,
      nullptr,
  };
  const void *code = kind == LUPINE_MODULE_IMAGE_FATBIN_RAW
                         ? static_cast<const void *>(image)
                         : &wrapper;
  if (kind != LUPINE_MODULE_IMAGE_FATBINC_V1 &&
      kind != LUPINE_MODULE_IMAGE_FATBINC_V2 &&
      kind != LUPINE_MODULE_IMAGE_FATBIN_RAW) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = lupine_in_va_arena(conn, [&] {
    return cuLibraryLoadData(library, code, jit.options, jit.option_values,
                             jit.num_options, options.options(),
                             options.has_values ? options.values() : nullptr,
                             options.count);
  });
  if (result == CUDA_SUCCESS) {
    lupine_note_device_stdout_image(image, image_size);
  }
  return result;
}

static int lupine_read_library_options(conn_t *conn,
                                       lupine_library_options *opts) {
  unsigned int num_options = 0;
  if (rpc_read(conn, &num_options, sizeof(num_options)) < 0) {
    return -1;
  }
  opts->count = num_options;
  opts->has_values = false;
  if (num_options == 0) {
    return 0;
  }
  CUlibraryOption *options = opts->opt_buf.alloc(num_options);
  void **values = opts->val_buf.alloc(num_options);
  if (!options || !values ||
      rpc_read(conn, options, num_options * sizeof(CUlibraryOption)) < 0 ||
      rpc_read(conn, values, num_options * sizeof(void *)) < 0) {
    return -1;
  }
  for (unsigned int i = 0; i < num_options; ++i) {
    opts->has_values |= reinterpret_cast<uintptr_t>(values[i]) !=
                        ~static_cast<uintptr_t>(options[i]);
  }
  if (opts->has_values) {
    // The client-side image is not the buffer passed to CUDA on this process.
    // Clear the preservation hint so the driver retains its own copy rather
    // than requiring a global library-to-image lifetime table.
    for (unsigned int i = 0; i < num_options; ++i) {
      if (options[i] == CU_LIBRARY_BINARY_IS_PRESERVED) {
        values[i] = nullptr;
      }
    }
  }
  return 0;
}

int handle_cuLibraryLoadData(conn_t *conn) {
  uint32_t kind = 0;
  size_t image_size = 0;
  int request_id;
  CUlibrary library = nullptr;
  lupine_jit_state jit_state;

  if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
      rpc_read(conn, &image_size, sizeof(image_size)) < 0 || image_size == 0) {
    return -1;
  }

  std::unique_ptr<unsigned char[]> image(
      new (std::nothrow) unsigned char[image_size]);
  lupine_library_options library_options;
  if (!image || rpc_read(conn, image.get(), image_size) < 0 ||
      lupine_read_jit_options(conn, &jit_state) < 0 ||
      lupine_read_library_options(conn, &library_options) < 0) {
    return -1;
  }

  request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult result = lupine_load_library(conn, &library, kind, image.get(),
                                        image_size, jit_state, library_options);
  lupine_library_snapshot snapshot;
  if (result == CUDA_SUCCESS) {
    snapshot = lupine_collect_library(library);
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &library, sizeof(library)) < 0 ||
      lupine_write_jit_outputs(conn, &jit_state) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      (result == CUDA_SUCCESS && lupine_write_library(conn, snapshot) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// Each entry answers like a cuLibraryLoadData response without JIT outputs.
int handle_lupineLibraryLoadBatch(conn_t *conn) {
  struct entry {
    CUlibrary library = nullptr;
    CUresult result = CUDA_ERROR_INVALID_CONTEXT;
    lupine_library_snapshot snapshot;
  };
  CUcontext context = nullptr;
  uint32_t count = 0;
  if (rpc_read(conn, &context, sizeof(context)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 || count > 4096) {
    return -1;
  }
  // Every image loads as soon as it is read, before rpc_read_end: this lane
  // belongs to the client's helper thread and nothing else waits on it.
  std::vector<entry> entries(count);
  std::unique_ptr<unsigned char[]> image;
  size_t image_capacity = 0;
  lupine_library_options options;
  lupine_jit_state jit;
  bool pushed = cuCtxPushCurrent(context) == CUDA_SUCCESS;
  auto pop_ctx = [&] {
    if (pushed) {
      CUcontext popped = nullptr;
      cuCtxPopCurrent(&popped);
    }
  };
  for (entry &e : entries) {
    uint32_t kind = 0;
    size_t image_size = 0;
    if (rpc_read(conn, &kind, sizeof(kind)) < 0 ||
        rpc_read(conn, &image_size, sizeof(image_size)) < 0 ||
        image_size == 0) {
      pop_ctx();
      return -1;
    }
    if (image_size > image_capacity) {
      image.reset(new (std::nothrow) unsigned char[image_size]);
      if (!image) {
        pop_ctx();
        return -1;
      }
      image_capacity = image_size;
    }
    if (rpc_read(conn, image.get(), image_size) < 0 ||
        lupine_read_library_options(conn, &options) < 0) {
      pop_ctx();
      return -1;
    }
    if (pushed) {
      e.result = lupine_load_library(conn, &e.library, kind, image.get(),
                                     image_size, jit, options);
    }
    if (e.result == CUDA_SUCCESS) {
      e.snapshot = lupine_collect_library(e.library);
    }
  }
  pop_ctx();
  int request_id = rpc_read_end(conn);
  if (request_id < 0 || rpc_write_start_response(conn, request_id) < 0) {
    return -1;
  }
  for (entry &e : entries) {
    if (rpc_write(conn, &e.library, sizeof(e.library)) < 0 ||
        rpc_write(conn, &e.result, sizeof(e.result)) < 0 ||
        (e.result == CUDA_SUCCESS &&
         lupine_write_library(conn, e.snapshot) < 0)) {
      return -1;
    }
  }
  return rpc_write_end(conn);
}

// POSIX-fd shareable handles cannot cross the wire as raw fd numbers. Export
// parks the real fd with the parent-process broker under a random token and
// returns the token; import redeems a token for the real fd (see ipc.h).

template <typename Handle, typename ExportFn>
static int lupine_handle_shareable_export(conn_t *conn, uint32_t kind,
                                          ExportFn export_fn) {
  Handle handle{};
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  lupine_ipc_token token = {};
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;
  int request_id = -1;

  if (!lupine_read_all(conn, handle, handleType, flags) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      lupine_ipc_make_token(&token) == 0) {
    int shareable_fd = -1;
    result = export_fn(&shareable_fd, handle, handleType, flags);
    if (result == CUDA_SUCCESS) {
      if (lupine_ipc_broker_register_fd(kind, &token, shareable_fd) < 0) {
        result = CUDA_ERROR_UNKNOWN;
      }
      lupine_ipc_close_fd(shareable_fd);
    }
  }
  return lupine_write_response(conn, request_id, token, result);
}

template <typename Handle, bool HasFlags, typename ImportFn>
static int lupine_handle_shareable_import(conn_t *conn, uint32_t kind,
                                          ImportFn import_fn) {
  lupine_ipc_token token = {};
  CUmemAllocationHandleType handleType;
  unsigned long long flags = 0;
  Handle handle{};
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  int request_id = -1;

  if (rpc_read(conn, &token, sizeof(token)) < 0 ||
      rpc_read(conn, &handleType, sizeof(handleType)) < 0 ||
      (HasFlags && rpc_read(conn, &flags, sizeof(flags)) < 0) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  if (handleType == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int import_fd = lupine_ipc_broker_get_fd(kind, &token);
    if (import_fd >= 0) {
      void *os_handle =
          reinterpret_cast<void *>(static_cast<uintptr_t>(import_fd));
      result = import_fn(&handle, os_handle, handleType, flags);
      lupine_ipc_close_fd(import_fd);
    }
  }
  return lupine_write_response(conn, request_id, handle, result);
}

int handle_cuMemExportToShareableHandle(conn_t *conn) {
  return lupine_handle_shareable_export<CUmemGenericAllocationHandle>(
      conn, LUPINE_IPC_FD_KIND_VMM_ALLOCATION, cuMemExportToShareableHandle);
}

int handle_cuMemImportFromShareableHandle(conn_t *conn) {
  return lupine_handle_shareable_import<CUmemGenericAllocationHandle, false>(
      conn, LUPINE_IPC_FD_KIND_VMM_ALLOCATION,
      [](CUmemGenericAllocationHandle *h, void *os,
         CUmemAllocationHandleType type, unsigned long long) {
        return cuMemImportFromShareableHandle(h, os, type);
      });
}

int handle_cuMemPoolExportToShareableHandle(conn_t *conn) {
  return lupine_handle_shareable_export<CUmemoryPool>(
      conn, LUPINE_IPC_FD_KIND_MEMORY_POOL, cuMemPoolExportToShareableHandle);
}

int handle_cuMemPoolImportFromShareableHandle(conn_t *conn) {
  return lupine_handle_shareable_import<CUmemoryPool, true>(
      conn, LUPINE_IPC_FD_KIND_MEMORY_POOL, cuMemPoolImportFromShareableHandle);
}

int handle_cuPointerGetAttributes(conn_t *conn) {
  unsigned int num_attributes = 0;
  CUdeviceptr ptr = 0;
  int request_id = -1;
  CUresult result = CUDA_SUCCESS;

  if (rpc_read(conn, &num_attributes, sizeof(num_attributes)) < 0) {
    return -1;
  }
  lupine_inline_buf<CUpointer_attribute, 16> attr_buf;
  CUpointer_attribute *attributes = attr_buf.alloc(num_attributes);
  if (!attributes ||
      (num_attributes != 0 &&
       rpc_read(conn, attributes,
                num_attributes * sizeof(CUpointer_attribute)) < 0) ||
      rpc_read(conn, &ptr, sizeof(ptr)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  size_t total_size = 0;
  for (unsigned int i = 0; i < num_attributes; ++i) {
    size_t value_size = 0;
    if (!lupine_pointer_attribute_size(attributes[i], &value_size)) {
      result = CUDA_ERROR_INVALID_VALUE;
      break;
    }
    total_size += value_size;
  }

  lupine_inline_buf<unsigned char, 256, alignof(std::max_align_t)> val_buf;
  lupine_inline_buf<void *, 16> data_buf;
  unsigned char *values = nullptr;
  if (result == CUDA_SUCCESS) {
    values = val_buf.alloc(total_size);
    void **data = data_buf.alloc(num_attributes);
    if (!values || !data) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      if (total_size != 0) {
        std::memset(values, 0, total_size);
      }
      size_t offset = 0;
      for (unsigned int i = 0; i < num_attributes; ++i) {
        size_t value_size = 0;
        lupine_pointer_attribute_size(attributes[i], &value_size);
        data[i] = values + offset;
        offset += value_size;
      }
      result = cuPointerGetAttributes(num_attributes, attributes, data, ptr);
    }
  }

  return lupine_write_response(
      conn, request_id, result,
      lupine_raw_bytes{values, result == CUDA_SUCCESS ? total_size : 0});
}

int handle_cuLinkCreate_v2(conn_t *conn) {
  auto link_state = std::make_unique<lupine_link_state>();
  CUlinkState client_state = nullptr;
  int request_id = -1;
  if (lupine_read_jit_options(conn, &link_state->jit) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUresult result =
      cuLinkCreate_v2(link_state->jit.num_options, link_state->jit.options,
                      link_state->jit.option_values, &link_state->cuda_state);
  if (result == CUDA_SUCCESS) {
    client_state = lupine_link_state_to_handle(link_state.release());
  }
  return lupine_write_response(conn, request_id, client_state, result);
}

static int lupine_handle_link_add(conn_t *conn, bool is_file) {
  CUlinkState state = nullptr;
  CUjitInputType type = is_file ? CU_JIT_INPUT_LIBRARY : CU_JIT_INPUT_PTX;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0) {
    return -1;
  }

  lupine_inline_buf<char, 512> name_buf;
  char *name_ptr = nullptr;
  auto read_name = [&]() -> int {
    size_t len = 0;
    if (rpc_read(conn, &len, sizeof(len)) < 0 || len > (1U << 20)) {
      return -1;
    }
    if (len != 0) {
      name_ptr = name_buf.alloc(len + 1);
      if (!name_ptr || rpc_read(conn, name_ptr, len) < 0) {
        return -1;
      }
      name_ptr[len] = '\0';
    }
    return 0;
  };

  size_t data_size = 0;
  std::unique_ptr<unsigned char[]> data;
  auto read_payload = [&](size_t bytes) -> int {
    data_size = bytes;
    if (bytes == 0) {
      return 0;
    }
    data.reset(new (std::nothrow) unsigned char[bytes + 1]);
    if (!data || rpc_read(conn, data.get(), bytes) < 0) {
      return -1;
    }
    data[bytes] = 0;
    return 0;
  };

  if (is_file) {
    uint8_t has_file_data = 0;
    uint64_t file_size = 0;
    if (read_name() < 0 ||
        rpc_read(conn, &has_file_data, sizeof(has_file_data)) < 0 ||
        rpc_read(conn, &file_size, sizeof(file_size)) < 0 ||
        has_file_data == 0 || file_size > (1ull << 32) ||
        read_payload(static_cast<size_t>(file_size)) < 0) {
      return -1;
    }
  } else {
    size_t size = 0;
    if (rpc_read(conn, &size, sizeof(size)) < 0 || size == SIZE_MAX ||
        read_payload(size) < 0 || read_name() < 0) {
      return -1;
    }
  }

  lupine_jit_state jit_state;
  int request_id = -1;
  if (lupine_read_jit_options(conn, &jit_state) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  CUresult result =
      link_state == nullptr
          ? CUDA_ERROR_INVALID_HANDLE
          : cuLinkAddData_v2(link_state->cuda_state, type, data.get(),
                             data_size, name_ptr, jit_state.num_options,
                             jit_state.options, jit_state.option_values);
  return rpc_write_start_response(conn, request_id) < 0 ||
                 lupine_write_jit_outputs(conn, &jit_state) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

int handle_cuLinkAddData_v2(conn_t *conn) {
  return lupine_handle_link_add(conn, false);
}

int handle_cuLinkAddFile_v2(conn_t *conn) {
  return lupine_handle_link_add(conn, true);
}

int handle_cuLinkComplete(conn_t *conn) {
  CUlinkState state = nullptr;
  void *cubin = nullptr;
  size_t size = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  int request_id = -1;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  lupine_jit_state empty_jit_outputs;
  lupine_jit_state *jit_outputs = &empty_jit_outputs;
  // Held until the response is flushed: the cubin buffer belongs to the
  // driver's link state, and a concurrent cuLinkDestroy would free it while
  // the queued cursor still points at it.
  std::unique_lock<std::mutex> lock;
  if (link_state != nullptr) {
    lock = std::unique_lock<std::mutex>(link_state->mutex);
    result = cuLinkComplete(link_state->cuda_state, &cubin, &size);
    jit_outputs = &link_state->jit;
  }
  size_t returned_size = result == CUDA_SUCCESS ? size : 0;
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &returned_size, sizeof(returned_size)) < 0 ||
                 rpc_write(conn, cubin, returned_size) < 0 ||
                 lupine_write_jit_outputs(conn, jit_outputs) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

int handle_cuLinkDestroy(conn_t *conn) {
  CUlinkState state = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;
  int request_id = -1;
  if (rpc_read(conn, &state, sizeof(state)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  auto *link_state = lupine_link_state_from_handle(state);
  if (link_state != nullptr) {
    // Wait for any cuLinkComplete on another lane to finish flushing the
    // driver-owned cubin buffer before cuLinkDestroy frees it.
    std::lock_guard<std::mutex> lock(link_state->mutex);
    result = cuLinkDestroy(link_state->cuda_state);
    if (result == CUDA_SUCCESS) {
      link_state->jit.clear();
    }
  }
  // Retain the opaque handle wrapper until process exit so stale client
  // handles never dereference freed memory.
  return lupine_write_response(conn, request_id, result);
}

// The library stays loaded: the client mirrors libraries onto other routes and
// caches the CUkernel handles this one owns, but it only ever sends the unload
// to the route that loaded it, so freeing here would dangle those handles.
int handle_cuLibraryUnload(conn_t *conn) {
  CUlibrary library = nullptr;

  if (rpc_read(conn, &library, sizeof(library)) < 0) {
    return -1;
  }
  if (rpc_read_end(conn) < 0) {
    return -1;
  }
  return 0;
}

struct lupine_server_launch_params {
  lupine_inline_buf<void *, 64> param_buf;
  lupine_inline_buf<unsigned char, 4096, alignof(std::max_align_t)> storage_buf;
  uint32_t count = 0;

  int read(conn_t *conn) {
    if (rpc_read(conn, &count, sizeof(count)) < 0) {
      return -1;
    }
    lupine_inline_buf<size_t, 64> size_buf;
    size_t *sizes = size_buf.alloc(count);
    void **params = param_buf.alloc(count);
    if (!sizes || !params ||
        rpc_read(conn, sizes, count * sizeof(*sizes)) != 0) {
      return -1;
    }
    size_t storage_size = 0;
    for (size_t i = 0; i < count; ++i) {
      storage_size += sizes[i];
    }
    unsigned char *storage = storage_buf.alloc(storage_size);
    if (!storage || rpc_read(conn, storage, storage_size) != 0) {
      return -1;
    }
    for (size_t i = 0; i < count; ++i) {
      params[i] = storage;
      storage += sizes[i];
    }
    return 0;
  }

  template <typename KernelNodeParams>
  int read_node(conn_t *conn, KernelNodeParams *node_params) {
    if (read(conn) < 0) {
      return -1;
    }
    node_params->kernelParams = data();
    node_params->extra = nullptr;
    return 0;
  }

  void **data() const { return count != 0 ? param_buf.ptr : nullptr; }
};

static int handle_cuLaunchKernelCommon(conn_t *conn, bool cooperative) {
  CUfunction f = nullptr;
  unsigned int gridDimX = 0, gridDimY = 0, gridDimZ = 0;
  unsigned int blockDimX = 0, blockDimY = 0, blockDimZ = 0;
  unsigned int sharedMemBytes = 0;
  CUstream hStream = nullptr;
  lupine_server_launch_params launch_params;

  if (!lupine_read_all(conn, f, gridDimX, gridDimY, gridDimZ, blockDimX,
                       blockDimY, blockDimZ, sharedMemBytes, hStream) ||
      launch_params.read(conn) < 0 || rpc_read_end(conn) < 0) {
    return -1;
  }

  CUresult result =
      cooperative ? cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ,
                                              blockDimX, blockDimY, blockDimZ,
                                              sharedMemBytes, hStream,
                                              launch_params.data())
                  : cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                   blockDimY, blockDimZ, sharedMemBytes,
                                   hStream, launch_params.data(), nullptr);
  lupine_keep_async_error(conn, result);
  return 0;
}

int handle_cuLaunchKernel(conn_t *conn) {
  return handle_cuLaunchKernelCommon(conn, false);
}

int handle_cuLaunchKernelEx(conn_t *conn) {
  CUlaunchConfig config = {};
  CUfunction f = nullptr;
  lupine_server_launch_params launch_params;
  lupine_inline_buf<CUlaunchAttribute, 8> attr_buf;

  if (rpc_read(conn, &config, sizeof(config)) < 0) {
    return -1;
  }
  CUlaunchAttribute *attributes = attr_buf.alloc(config.numAttrs);
  if (!attributes ||
      (config.numAttrs != 0 &&
       rpc_read(conn, attributes, config.numAttrs * sizeof(*attributes)) < 0) ||
      rpc_read(conn, &f, sizeof(f)) < 0 || launch_params.read(conn) < 0 ||
      rpc_read_end(conn) < 0) {
    return -1;
  }
  config.attrs = config.numAttrs != 0 ? attributes : nullptr;

#if CUDA_VERSION < 11080
  CUresult result = CUDA_ERROR_NOT_SUPPORTED;
#else
  CUresult result = cuLaunchKernelEx(&config, f, launch_params.data(), nullptr);
#endif
  lupine_keep_async_error(conn, result);
  return 0;
}

int handle_cuLaunchCooperativeKernel(conn_t *conn) {
  return handle_cuLaunchKernelCommon(conn, true);
}

namespace {

struct lupine_host_callback_data {
  conn_t *conn = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  // Uncaptured launches run once; a captured one is replayed by its graph.
  bool one_shot = false;
};

struct lupine_stream_callback_data {
  conn_t *conn = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
};

} // namespace

template <typename... Args>
static void lupine_send_side_effect(conn_t *conn, int op, const Args &...args) {
  void *response = nullptr;
  if (rpc_write_start_request(conn, op) >= 0 &&
      ((rpc_write(conn, &args, sizeof(args)) >= 0) && ...) &&
      rpc_wait_for_response(conn) >= 0) {
    rpc_read(conn, &response, sizeof(response));
    rpc_read_end(conn);
  }
}

// A host callback carries the deliveries numbered when it started: the copies
// ordered before it, on any stream, have pushed their bytes by then, and the
// client applies them before running the callback.
static void CUDA_CB lupine_graph_host_callback(void *userData) {
  auto *callback = static_cast<lupine_host_callback_data *>(userData);
  if (callback == nullptr || callback->conn == nullptr) {
    return;
  }

  conn_t *conn = callback->conn;
  uint64_t delivered = lupine_dtoh_watermark(conn);
  lupine_send_side_effect(conn, LUPINE_SIDE_EFFECT_HOST_FUNCTION, delivered,
                          callback->fn, callback->userData);
  if (callback->one_shot) {
    delete callback;
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
  uint64_t delivered = lupine_dtoh_watermark(conn);
  void *fn = reinterpret_cast<void *>(callback->callback);
  lupine_send_side_effect(conn, LUPINE_SIDE_EFFECT_STREAM_CALLBACK, delivered,
                          stream, status, fn, callback->userData);
  delete callback;
}

#if CUDA_VERSION >= 12090
static void CUDA_CB lupine_logs_callback(void *user_data, CUlogLevel level,
                                         char *message, size_t length) {
  if (lupine_is_internal_cuda_query()) {
    return;
  }
  auto *callback = static_cast<lupine_logs_callback_data *>(user_data);
  if (callback == nullptr) {
    return;
  }

  conn_t *conn = nullptr;
  CUlogsCallback client_callback = nullptr;
  void *client_user_data = nullptr;
  {
    std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
    conn = callback->conn.load(std::memory_order_acquire);
    client_callback = callback->callback;
    client_user_data = callback->user_data;
  }
  if (conn == nullptr || client_callback == nullptr) {
    return;
  }

  void *response = nullptr;
  int32_t origin_stream_id = rpc_current_http2_stream(conn);
  if (rpc_write_start_request(conn, LUPINE_SIDE_EFFECT_LOG_CALLBACK) >= 0 &&
      rpc_write(conn, &level, sizeof(level)) >= 0 &&
      rpc_write(conn, &origin_stream_id, sizeof(origin_stream_id)) >= 0 &&
      rpc_write(conn, &length, sizeof(length)) >= 0 &&
      (length == 0 || rpc_write(conn, message, length) >= 0) &&
      rpc_write(conn, &client_callback, sizeof(client_callback)) >= 0 &&
      rpc_write(conn, &client_user_data, sizeof(client_user_data)) >= 0 &&
      rpc_wait_for_response(conn) >= 0) {
    rpc_read(conn, &response, sizeof(response));
    rpc_read_end(conn);
  }
}
#endif

int handle_cuGraphAddKernelNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  lupine_server_graph_deps deps;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  CUgraphNode graphNode = nullptr;
  lupine_server_launch_params params;
  int request_id = -1;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 || deps.read(conn) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      params.read_node(conn, &nodeParams) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = cuGraphAddKernelNode_v2(&graphNode, hGraph, deps.data(),
                                            deps.size(), &nodeParams);
  return lupine_write_response(conn, request_id, graphNode, result);
}

int handle_cuGraphKernelNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  int request_id = -1;
  CUDA_KERNEL_NODE_PARAMS node_params = {};

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = cuGraphKernelNodeGetParams_v2(hNode, &node_params);
  if (result == CUDA_SUCCESS && node_params.extra != nullptr) {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  struct wire_hdr {
    unsigned char bytes[sizeof(size_t) + sizeof(CUresult)];
  };
  lupine_inline_vec<wire_hdr, 65> headers;
  lupine_inline_vec<size_t, 64> sizes;
  if (result == CUDA_SUCCESS) {
    lupine_internal_cuda_query probe;
    for (size_t i = 0;; ++i) {
      size_t offset = 0, size = 0;
      CUresult param_result = CUDA_ERROR_INVALID_HANDLE;
      if (node_params.func != nullptr) {
        param_result = cuFuncGetParamInfo(node_params.func, i, &offset, &size);
      }
#if CUDA_VERSION >= 12000
      else if (node_params.kern != nullptr) {
        param_result =
            cuKernelGetParamInfo(node_params.kern, i, &offset, &size);
      }
#endif
      wire_hdr hdr;
      std::memcpy(hdr.bytes, &size, sizeof(size));
      std::memcpy(hdr.bytes + sizeof(size), &param_result,
                  sizeof(param_result));
      headers.push(hdr);
      if (param_result != CUDA_SUCCESS) {
        break;
      }
      sizes.push(size);
    }
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_write(conn, &node_params, sizeof(node_params)) < 0) {
    return -1;
  }
  if (result == CUDA_SUCCESS) {
    const wire_hdr *hdrs = headers.data();
    const size_t *sz = sizes.data();
    for (size_t i = 0; i < sizes.count; ++i) {
      if (rpc_write(conn, &hdrs[i], sizeof(wire_hdr)) < 0 ||
          rpc_write(conn, node_params.kernelParams[i], sz[i]) < 0) {
        return -1;
      }
    }
    if (rpc_write(conn, &hdrs[sizes.count], sizeof(wire_hdr)) < 0) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
}

static int handle_cuGraphKernelNodeSetParamsCommon(conn_t *conn,
                                                   bool exec_node) {
  CUgraphExec hGraphExec = nullptr;
  CUgraphNode hNode = nullptr;
  CUDA_KERNEL_NODE_PARAMS nodeParams = {};
  lupine_server_launch_params params;
  int request_id = -1;

  if ((exec_node && rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0) ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      params.read_node(conn, &nodeParams) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result =
      exec_node
          ? cuGraphExecKernelNodeSetParams_v2(
                hGraphExec, lupine_htod_graph_exec_node(hGraphExec, hNode),
                &nodeParams)
          : cuGraphKernelNodeSetParams_v2(hNode, &nodeParams);
  return lupine_write_response(conn, request_id, result);
}

int handle_cuGraphKernelNodeSetParams(conn_t *conn) {
  return handle_cuGraphKernelNodeSetParamsCommon(conn, false);
}

int handle_cuGraphExecKernelNodeSetParams(conn_t *conn) {
  return handle_cuGraphKernelNodeSetParamsCommon(conn, true);
}

int handle_cuGraphAddMemcpyNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  lupine_server_graph_deps deps;
  CUDA_MEMCPY3D copyParams = {};
  CUcontext ctx = nullptr;
  size_t host_src_bytes = 0;
  CUgraphNode graphNode = nullptr;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 || deps.read(conn) < 0 ||
      !lupine_read_all(conn, copyParams, ctx, host_src_bytes)) {
    return -1;
  }

  if (host_src_bytes != 0) {
    void *host = lupine_alloc_process_host_buffer(host_src_bytes);
    if (host == nullptr || rpc_read(conn, host, host_src_bytes) < 0) {
      return -1;
    }
    lupine_pack_host_source(copyParams);
    copyParams.srcHost = host;
  }

  lupine_host_rows client;
  if (copyParams.dstMemoryType == CU_MEMORYTYPE_HOST) {
    client = lupine_host_destination_rows(copyParams);
    lupine_pack_host_destination(copyParams);
    void *host = lupine_alloc_process_host_buffer(client.bytes());
    if (host == nullptr && client.bytes() != 0) {
      return -1;
    }
    copyParams.dstHost = host;
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult result = cuGraphAddMemcpyNode(&graphNode, hGraph, deps.data(),
                                         deps.size(), &copyParams, ctx);
  // The copy lands in graph-owned staging; a host node behind it pushes the
  // bytes at every launch.
  if (copyParams.dstMemoryType == CU_MEMORYTYPE_HOST && client.bytes() != 0) {
    CUresult pushed = lupine_graph_add_dtoh_push(
        conn, hGraph, result == CUDA_SUCCESS ? graphNode : nullptr, client,
        copyParams.dstHost);
    result = result == CUDA_SUCCESS ? pushed : result;
  }
  return lupine_write_response(conn, request_id, graphNode, result);
}

template <typename Call>
static CUresult lupine_call_with_host_node_trampoline(
    conn_t *conn, const CUDA_HOST_NODE_PARAMS &params, Call &&call) {
  if (params.fn == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  auto *callback =
      new lupine_host_callback_data{conn, params.fn, params.userData};
  CUDA_HOST_NODE_PARAMS server_params{lupine_graph_host_callback, callback};
  CUresult result = call(&server_params);
  if (result != CUDA_SUCCESS) {
    delete callback;
  }
  return result;
}

int handle_cuGraphAddHostNode(conn_t *conn) {
  CUgraphNode graphNode = nullptr;
  CUgraph hGraph = nullptr;
  lupine_server_graph_deps deps;
  CUDA_HOST_NODE_PARAMS nodeParams = {};
  int request_id = -1;

  if (rpc_read(conn, &graphNode, sizeof(graphNode)) < 0 ||
      rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 || deps.read(conn) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = lupine_call_with_host_node_trampoline(
      conn, nodeParams, [&](const CUDA_HOST_NODE_PARAMS *server_params) {
        return cuGraphAddHostNode(&graphNode, hGraph, deps.data(), deps.size(),
                                  server_params);
      });
  return lupine_write_response(conn, request_id, graphNode, result);
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

int handle_cuGraphAddNode(conn_t *conn) {
  CUgraph hGraph = nullptr;
  lupine_server_graph_deps deps;
  CUgraphNodeParams nodeParams = {};
  CUgraphNode graphNode = nullptr;
  lupine_server_launch_params params;

  if (rpc_read(conn, &hGraph, sizeof(hGraph)) < 0 || deps.read(conn) < 0 ||
      rpc_read(conn, &nodeParams, sizeof(nodeParams)) < 0 ||
      (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL &&
       params.read_node(conn, &nodeParams.kernel) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  if (nodeParams.type == CU_GRAPH_NODE_TYPE_CONDITIONAL) {
    nodeParams.conditional.phGraph_out = nullptr;
  }
  CUresult result =
      (nodeParams.type == CU_GRAPH_NODE_TYPE_KERNEL ||
       nodeParams.type == CU_GRAPH_NODE_TYPE_CONDITIONAL)
          ? lupine_server_cuGraphAddNode(&graphNode, hGraph, deps.data(),
                                         nullptr, deps.size(), &nodeParams)
          : CUDA_ERROR_NOT_SUPPORTED;
  LUPINE_TRACE_LOG("LUPINE cuGraphAddNode type="
                   << nodeParams.type << " graph=" << hGraph
                   << " node=" << graphNode << " result=" << result);

  size_t cond_bytes = (result == CUDA_SUCCESS &&
                       nodeParams.type == CU_GRAPH_NODE_TYPE_CONDITIONAL)
                          ? nodeParams.conditional.size * sizeof(CUgraph)
                          : 0;
  return lupine_write_response(
      conn, request_id, graphNode, result,
      lupine_raw_bytes{nodeParams.conditional.phGraph_out, cond_bytes});
}

// Host-node callbacks set after node creation have no graph handle to own their
// trampoline data. The server process is their lifetime boundary, so allocate
// them directly without a synchronized container that never reclaimed them.
static int handle_cuGraphHostNodeSetParamsCommon(conn_t *conn, bool exec_node) {
  CUgraphExec hGraphExec = nullptr;
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  int request_id = -1;

  if ((exec_node && rpc_read(conn, &hGraphExec, sizeof(hGraphExec)) < 0) ||
      rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = lupine_call_with_host_node_trampoline(
      conn, params, [&](const CUDA_HOST_NODE_PARAMS *server_params) {
        return exec_node ? cuGraphExecHostNodeSetParams(
                               hGraphExec,
                               lupine_htod_graph_exec_node(hGraphExec, hNode),
                               server_params)
                         : cuGraphHostNodeSetParams(hNode, server_params);
      });
  return lupine_write_response(conn, request_id, result);
}

int handle_cuGraphHostNodeSetParams(conn_t *conn) {
  return handle_cuGraphHostNodeSetParamsCommon(conn, false);
}

int handle_cuGraphExecHostNodeSetParams(conn_t *conn) {
  return handle_cuGraphHostNodeSetParamsCommon(conn, true);
}

int handle_cuGraphHostNodeGetParams(conn_t *conn) {
  CUgraphNode hNode = nullptr;
  CUDA_HOST_NODE_PARAMS params{};
  int request_id = -1;

  if (rpc_read(conn, &hNode, sizeof(hNode)) < 0 ||
      rpc_read(conn, &params, sizeof(params)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = cuGraphHostNodeGetParams(hNode, &params);
  // Unwrap the trampoline so the client sees the fn/userData it registered.
  if (result == CUDA_SUCCESS && params.fn == lupine_graph_host_callback &&
      params.userData != nullptr) {
    auto *callback = static_cast<lupine_host_callback_data *>(params.userData);
    params.fn = callback->fn;
    params.userData = callback->userData;
  }
  return lupine_write_response(conn, request_id, params, result);
}

int handle_cuLaunchHostFunc(conn_t *conn) {
  // Unanswered: a failure reaches the client from its next synchronize.
  CUstream stream = nullptr;
  CUhostFn fn = nullptr;
  void *userData = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (!lupine_read_all(conn, stream, fn, userData) || rpc_read_end(conn) < 0) {
    return -1;
  }

  if (fn != nullptr) {
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    if (stream != nullptr) {
      cuStreamIsCapturing(stream, &capture_status);
    }
    auto *callback = new lupine_host_callback_data{
        conn, fn, userData, capture_status == CU_STREAM_CAPTURE_STATUS_NONE};
    result = cuLaunchHostFunc(stream, lupine_graph_host_callback, callback);
    if (result != CUDA_SUCCESS) {
      delete callback;
    }
  }

  lupine_keep_async_error(conn, result);
  return 0;
}

int handle_cuStreamAddCallback(conn_t *conn) {
  // Unanswered: a failure reaches the client from its next synchronize.
  CUstream stream = nullptr;
  CUstreamCallback callback = nullptr;
  void *userData = nullptr;
  unsigned int flags = 0;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (!lupine_read_all(conn, stream, callback, userData, flags) ||
      rpc_read_end(conn) < 0) {
    return -1;
  }

  if (callback != nullptr) {
    auto *data = new lupine_stream_callback_data{conn, callback, userData};
    result = cuStreamAddCallback(stream, lupine_stream_callback, data, flags);
    if (result != CUDA_SUCCESS) {
      delete data;
    }
  }

  lupine_keep_async_error(conn, result);
  return 0;
}

#if CUDA_VERSION >= 12090
int handle_cuLogsRegisterCallback(conn_t *conn) {
  CUlogsCallback callback = nullptr;
  void *user_data = nullptr;
  CUlogsCallbackHandle callback_handle = nullptr;
  CUresult result = CUDA_ERROR_INVALID_VALUE;

  if (rpc_read(conn, &callback, sizeof(callback)) < 0 ||
      rpc_read(conn, &user_data, sizeof(user_data)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  lupine_logs_callback_data *data = nullptr;
  if (callback != nullptr) {
    data = new (std::nothrow) lupine_logs_callback_data;
    if (data == nullptr) {
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      data->conn.store(conn, std::memory_order_release);
      data->callback = callback;
      data->user_data = user_data;
      result =
          cuLogsRegisterCallback(lupine_logs_callback, data, &callback_handle);
      if (result == CUDA_SUCCESS) {
        std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
        lupine_logs_callbacks()[callback_handle] = data;
      } else {
        delete data;
      }
    }
  }

  return lupine_write_response(conn, request_id, callback_handle, result);
}

int handle_cuLogsUnregisterCallback(conn_t *conn) {
  CUlogsCallbackHandle callback_handle = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &callback_handle, sizeof(callback_handle)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = cuLogsUnregisterCallback(callback_handle);
  lupine_logs_callback_data *data = nullptr;
  if (result == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
    auto it = lupine_logs_callbacks().find(callback_handle);
    if (it != lupine_logs_callbacks().end()) {
      data = it->second;
      lupine_logs_callbacks().erase(it);
    }
  }
  delete data;
  return lupine_write_response(conn, request_id, result);
}
#endif

void lupine_server_cleanup_log_callbacks(conn_t *conn) {
#if CUDA_VERSION >= 12090
  std::vector<std::pair<CUlogsCallbackHandle, lupine_logs_callback_data *>>
      callbacks;
  {
    std::lock_guard<std::mutex> lock(lupine_logs_callback_mutex());
    for (auto it = lupine_logs_callbacks().begin();
         it != lupine_logs_callbacks().end();) {
      if (it->second->conn.load(std::memory_order_acquire) == conn) {
        it->second->conn.store(nullptr, std::memory_order_release);
        callbacks.emplace_back(it->first, it->second);
        it = lupine_logs_callbacks().erase(it);
      } else {
        ++it;
      }
    }
  }
  for (const auto &[handle, data] : callbacks) {
    if (cuLogsUnregisterCallback(handle) == CUDA_SUCCESS) {
      delete data;
    }
  }
#else
  (void)conn;
#endif
}

static int handle_cuEventRecordCommon(conn_t *conn, bool with_flags) {
  CUevent event = nullptr;
  CUstream stream = nullptr;
  unsigned int flags = 0;

  if (rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (with_flags && rpc_read(conn, &flags, sizeof(flags)) < 0)) {
    return -1;
  }
  if (rpc_read_end(conn) < 0) {
    return -1;
  }

  CUresult result = with_flags ? cuEventRecordWithFlags(event, stream, flags)
                               : cuEventRecord(event, stream);
  if (result == CUDA_SUCCESS) {
    lupine_record_event_capture_resources(event, stream);
  }
  lupine_keep_async_error(conn, result);
  return 0;
}

int handle_cuEventRecord(conn_t *conn) {
  return handle_cuEventRecordCommon(conn, false);
}

int handle_cuEventRecordWithFlags(conn_t *conn) {
  return handle_cuEventRecordCommon(conn, true);
}

int handle_cuEventDestroy_v2(conn_t *conn) {
  // Unanswered: a failure reaches the client from its next synchronize.
  CUevent event = nullptr;
  if (rpc_read(conn, &event, sizeof(event)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUresult result = cuEventDestroy_v2(event);
  if (result == CUDA_SUCCESS) {
    lupine_forget_event_capture_resources(event);
  }
  lupine_keep_async_error(conn, result);
  return 0;
}

int handle_cuEventQuery(conn_t *conn) {
  CUevent events[kLupineEventQueryBatch + 1] = {};
  uint64_t tickets[kLupineEventQueryBatch + 1] = {};
  uint32_t count = 0;
  if (rpc_read(conn, &events[0], sizeof(events[0])) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      count > kLupineEventQueryBatch ||
      rpc_read(conn, events + 1, count * sizeof(*events)) < 0 ||
      rpc_read(conn, tickets + 1, count * sizeof(*tickets)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  // A record the client sent that is not submitted yet makes the answer not
  // ready; the driver would still report the event's previous record.
  CUresult results[kLupineEventQueryBatch + 1];
  for (uint32_t i = 0; i <= count; ++i) {
    bool unsettled = i == 0 ? rpc_request_unsettled(conn)
                            : !rpc_async_ticket_released(conn, tickets[i]);
    results[i] = unsettled ? CUDA_ERROR_NOT_READY : cuEventQuery(events[i]);
  }
  // Every success above is covered: copies its event follows have numbered
  // their deliveries by now. The client applies them before using any of it.
  uint64_t delivered = lupine_dtoh_watermark(conn);
  results[0] = lupine_take_async_error(conn, results[0]);
  return lupine_write_response(
      conn, request_id, delivered,
      lupine_raw_bytes{results, (count + 1) * sizeof(*results)});
}

int handle_cuStreamWaitEvent(conn_t *conn) {
  CUstream stream = nullptr;
  CUevent event = nullptr;
  unsigned int flags = 0;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &event, sizeof(event)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0 || rpc_read_end(conn) < 0) {
    return -1;
  }
  lupine_wait_event_capture_resources(stream, event);
  // Fire-and-forget like cuEventRecord: a failure surfaces at the next sync.
  lupine_keep_async_error(conn, cuStreamWaitEvent(stream, event, flags));
  return 0;
}

static int lupine_handle_stream_begin_capture(conn_t *conn, bool to_graph) {
  CUstream stream = nullptr;
  CUgraph graph = nullptr;
  lupine_server_graph_deps deps;
  CUstreamCaptureMode mode = CU_STREAM_CAPTURE_MODE_GLOBAL;
  int request_id = -1;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (to_graph &&
       (rpc_read(conn, &graph, sizeof(graph)) < 0 || deps.read(conn) < 0)) ||
      rpc_read(conn, &mode, sizeof(mode)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  lupine_capture_id capture_id = lupine_begin_stream_capture_resources(stream);
  CUresult result = lupine_server_prepare_htod_capture(conn);
  if (result == CUDA_SUCCESS) {
    result = to_graph ? cuStreamBeginCaptureToGraph(stream, graph, deps.data(),
                                                    nullptr, deps.size(), mode)
                      : cuStreamBeginCapture_v2(stream, mode);
  }
  if (result != CUDA_SUCCESS) {
    lupine_discard_stream_capture_resources(capture_id);
  }
  return lupine_write_response(conn, request_id, result);
}

int handle_cuStreamBeginCaptureToGraph(conn_t *conn) {
  return lupine_handle_stream_begin_capture(conn, true);
}

int handle_cuStreamGetCaptureInfo(conn_t *conn) {
  CUstream stream = nullptr;
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  cuuint64_t id = 0;
  CUgraph graph = nullptr;
  const CUgraphNode *deps_ptr = nullptr;
  const CUgraphEdgeData *edge_ptr = nullptr;
  size_t dep_count = 0;
  int request_id = -1;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = cuStreamGetCaptureInfo_v3(stream, &status, &id, &graph,
                                              &deps_ptr, &edge_ptr, &dep_count);
  bool has_edge_data = edge_ptr != nullptr && dep_count != 0;

  return lupine_write_response(
      conn, request_id, status, id, graph, dep_count, has_edge_data,
      lupine_raw_bytes{deps_ptr,
                       deps_ptr ? dep_count * sizeof(CUgraphNode) : 0},
      lupine_raw_bytes{edge_ptr,
                       has_edge_data ? dep_count * sizeof(CUgraphEdgeData) : 0},
      result);
}

int handle_cuStreamBeginCapture(conn_t *conn) {
  return lupine_handle_stream_begin_capture(conn, false);
}

int handle_cuStreamEndCapture(conn_t *conn) {
  CUstream stream = nullptr;
  uint8_t graph_present = 0;
  CUgraph graph = nullptr;
  int request_id = -1;

  if (rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      rpc_read(conn, &graph_present, sizeof(graph_present)) < 0 ||
      (graph_present && rpc_read(conn, &graph, sizeof(graph)) < 0) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result =
      cuStreamEndCapture(stream, graph_present ? &graph : nullptr);
  lupine_finish_stream_capture_resources(stream, graph, result == CUDA_SUCCESS);

  return lupine_write_response(
      conn, request_id,
      lupine_raw_bytes{&graph, graph_present ? sizeof(graph) : 0}, result);
}

int handle_cuGraphClone(conn_t *conn) {
  CUgraph clone = nullptr, original = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &clone, sizeof(clone)) < 0 ||
      rpc_read(conn, &original, sizeof(original)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUresult result = cuGraphClone(&clone, original);
  if (result == CUDA_SUCCESS) {
    lupine_clone_graph_resources(clone, original);
  }
  return lupine_write_response(conn, request_id, clone, result);
}

template <typename InstantiateFn>
static CUresult lupine_instantiate_prepared_graph(CUgraph graph,
                                                  CUgraphExec *exec,
                                                  CUgraphNode *error_node,
                                                  InstantiateFn instantiate) {
  lupine_htod_graph_binding binding;
  CUresult result = lupine_prepare_graph_exec_resources(graph, &binding);
  if (result == CUDA_SUCCESS) {
    result = instantiate(binding.prepared);
  }
  if (result != CUDA_SUCCESS && error_node != nullptr) {
    *error_node = lupine_original_htod_graph_node(binding, *error_node);
  }
  if (result == CUDA_SUCCESS) {
    result = lupine_associate_graph_exec_resources(*exec, binding);
    if (result != CUDA_SUCCESS) {
      (void)cuGraphExecDestroy(*exec);
      *exec = nullptr;
    }
  }
  lupine_release_htod_graph_binding(&binding);
  return result;
}

int handle_cuGraphInstantiate_v2(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  uint8_t error_node_present = 0;
  CUgraphNode error_node = nullptr;
  size_t log_buffer_size = 0;
  char *log_buffer = nullptr;
  uint8_t log_buffer_null = 0;

  if (rpc_read(conn, &graph, sizeof(graph)) < 0 ||
      rpc_read(conn, &error_node_present, sizeof(error_node_present)) < 0 ||
      (error_node_present &&
       rpc_read(conn, &error_node, sizeof(error_node)) < 0) ||
      rpc_read(conn, &log_buffer_size, sizeof(log_buffer_size)) < 0 ||
      rpc_read(conn, &log_buffer_null, sizeof(log_buffer_null)) < 0) {
    return -1;
  }
  if (!log_buffer_null) {
    log_buffer = static_cast<char *>(
        calloc(log_buffer_size != 0 ? log_buffer_size : 1, sizeof(char)));
    if (log_buffer == nullptr) {
      return -1;
    }
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    free(log_buffer);
    return -1;
  }

  CUresult result = lupine_instantiate_prepared_graph(
      graph, &exec, error_node_present ? &error_node : nullptr,
      [&](CUgraph prepared) {
        return cuGraphInstantiate_v2(&exec, prepared,
                                     error_node_present ? &error_node : nullptr,
                                     log_buffer, log_buffer_size);
      });

  uint8_t log_buffer_has_data = !log_buffer_null && result != CUDA_SUCCESS;
  bool failed =
      rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &exec, sizeof(exec)) < 0 ||
      (error_node_present &&
       rpc_write(conn, &error_node, sizeof(error_node)) < 0) ||
      rpc_write(conn, &log_buffer_has_data, sizeof(log_buffer_has_data)) < 0 ||
      (log_buffer_has_data != 0 && log_buffer_size != 0 &&
       rpc_write(conn, log_buffer, log_buffer_size) < 0) ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0;
  free(log_buffer);
  return failed ? -1 : 0;
}

int handle_cuGraphInstantiateWithFlags(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  unsigned long long flags = 0;
  int request_id = -1;

  if (!lupine_read_all(conn, exec, graph, flags) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUresult result = lupine_instantiate_prepared_graph(
      graph, &exec, nullptr, [&](CUgraph prepared) {
        return cuGraphInstantiateWithFlags(&exec, prepared, flags);
      });
  return lupine_write_response(conn, request_id, exec, result);
}

int handle_cuGraphInstantiateWithParams(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUgraph graph = nullptr;
  CUDA_GRAPH_INSTANTIATE_PARAMS params = {};
  int request_id = -1;

  if (!lupine_read_all(conn, exec, graph, params) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUgraphNode *err_node = nullptr;
#if CUDA_VERSION >= 12000
  err_node = &params.hErrNode_out;
#endif
  CUresult result = lupine_instantiate_prepared_graph(
      graph, &exec, err_node, [&](CUgraph prepared) {
        return cuGraphInstantiateWithParams(&exec, prepared, &params);
      });

  return lupine_write_response(conn, request_id, exec, params, result);
}

int handle_cuGraphExecDestroy(conn_t *conn) {
  CUgraphExec exec = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUresult result = cuGraphExecDestroy(exec);
  rpc_async_sequence_end(conn);
  if (result == CUDA_SUCCESS) {
    result = lupine_release_graph_exec_resources(exec);
  }
  return lupine_unanswered(conn, result)
             ? 0
             : lupine_write_response(conn, request_id, result);
}

// Unanswered, as the generated client sends it: a failure reaches the client
// from its next synchronize.
int handle_cuGraphDestroy(conn_t *conn) {
  CUgraph graph = nullptr;
  if (rpc_read(conn, &graph, sizeof(graph)) < 0 || rpc_read_end(conn) < 0) {
    return -1;
  }
  CUresult result = cuGraphDestroy(graph);
  lupine_erase_graph_resources(graph);
  lupine_keep_async_error(conn, result);
  return 0;
}

// Unanswered: the bytes land before the ticket is released, so later
// requests from every lane see them.
int handle_lupineManagedHostFlush(conn_t *conn) {
  uint32_t count = 0;

  if (rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }

  for (uint32_t i = 0; i < count; ++i) {
    void *server_host_ptr = nullptr;
    size_t bytes = 0;
    // Host mappings and native identity-VA managed allocations are both CPU
    // accessible. Reading directly avoids a context-dependent pinned
    // allocation in the flush handler.
    if (rpc_read(conn, &server_host_ptr, sizeof(server_host_ptr)) < 0 ||
        rpc_read(conn, &bytes, sizeof(bytes)) < 0 ||
        rpc_read(conn, server_host_ptr, bytes) < 0) {
      return -1;
    }
  }

  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  // A host callback's acknowledgement waits for its writes to land.
  if (rpc_request_awaited(conn) &&
      (rpc_write_start_response(conn, request_id) < 0 ||
       rpc_write_end(conn) < 0)) {
    return -1;
  }
  return 0;
}
// Serves LUPINE_RPC_lupineDeviceSnapshot: every immutable per-device value the
// client caches, for every device, in one response. Mutable state (primary
// context state, context limits) is deliberately excluded. The response is all
// or nothing: any query failure fails the whole RPC and the client falls back
// to the per-call paths. Individual attributes the driver rejects are simply
// absent from the pair list; that is expected, not an error.
struct lupine_device_snapshot_record {
  char name[256] = {};
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

int handle_lupineDeviceSnapshot(conn_t *conn) {
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
        rpc_write(conn, record.pairs.data(),
                  record.pairs.size() * sizeof(int32_t)) < 0) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
}

namespace {

#if defined(__linux__)
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

static constexpr uintptr_t LUPINE_SERVER_VA_ALIGNMENT = UINT64_C(0x200000);

void *lupine_server_mmap(conn_t *conn, void *address, size_t size,
                         size_t alignment, int protection, int flags, int fd,
                         off_t offset) {
  if (conn != nullptr) {
    uintptr_t claimed = 0;
    if (!lupine_va_claim(conn, size, alignment, &claimed)) {
      errno = ENOMEM;
      return MAP_FAILED;
    }
    address = reinterpret_cast<void *>(claimed);
    flags = (flags & ~(MAP_FIXED | MAP_FIXED_NOREPLACE)) | MAP_FIXED;
  }
  return reinterpret_cast<void *>(
      syscall(SYS_mmap, address, size, protection, flags, fd, offset));
}
#endif

enum class lupine_server_allocation_kind { host, managed };

struct lupine_server_allocation {
  conn_t *conn = nullptr;
  size_t size = 0;
  unsigned int flags = 0;
  lupine_server_allocation_kind kind = lupine_server_allocation_kind::host;
};

struct lupine_server_allocation_state {
  std::mutex mutex;
  std::unordered_map<uintptr_t, lupine_server_allocation> allocations;
};

lupine_server_allocation_state lupine_server_state;

bool lupine_record_server_allocation(
    uintptr_t address, const lupine_server_allocation &allocation) {
  std::lock_guard<std::mutex> lock(lupine_server_state.mutex);
  try {
    return lupine_server_state.allocations.emplace(address, allocation).second;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

CUresult lupine_server_map_host_allocation(conn_t *conn, void **pointer,
                                           CUdeviceptr *device_pointer,
                                           size_t bytes, unsigned int flags,
                                           unsigned int register_flags) {
#if !defined(__linux__)
  return CUDA_ERROR_NOT_SUPPORTED;
#else
  void *requested = *pointer;
  *pointer = nullptr;
  *device_pointer = 0;
  long configured_page_size = sysconf(_SC_PAGESIZE);
  size_t page_size = configured_page_size > 0
                         ? static_cast<size_t>(configured_page_size)
                         : static_cast<size_t>(4096);
  if (bytes > SIZE_MAX - (page_size - 1)) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  size_t storage_size = (bytes + page_size - 1) & ~(page_size - 1);
  if (reinterpret_cast<uintptr_t>(requested) % page_size != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // Portable replicas use the original device address. A foreign address must
  // never replace an existing mapping, including another connection's arena.
  conn_t *arena = requested == nullptr && conn->va_size != 0 ? conn : nullptr;
  int map_flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (requested != nullptr) {
    map_flags |= MAP_FIXED_NOREPLACE;
  }
  void *mapping = lupine_server_mmap(arena, requested, storage_size, page_size,
                                     PROT_READ | PROT_WRITE, map_flags, -1, 0);
  uintptr_t address = reinterpret_cast<uintptr_t>(mapping);
  if (mapping == MAP_FAILED) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  // Older kernels may ignore MAP_FIXED_NOREPLACE and choose another address.
  if (requested != nullptr && mapping != requested) {
    munmap(mapping, storage_size);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  CUresult result = cuMemHostRegister_v2(mapping, storage_size, register_flags);
  bool registered = result == CUDA_SUCCESS;
  CUdeviceptr mapped_device_pointer = 0;
  if (registered) {
    result = cuMemHostGetDevicePointer(&mapped_device_pointer, mapping, 0);
  }
  // Pointer-rich mapped memory requires the CPU and GPU views to use the same
  // numeric address. Do not silently create an allocation that violates the
  // connection invariant on systems without identical host UVA aliases.
  if (result == CUDA_SUCCESS && mapped_device_pointer != address) {
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (result != CUDA_SUCCESS ||
      !lupine_record_server_allocation(
          address,
          {conn, storage_size, flags, lupine_server_allocation_kind::host})) {
    if (registered) {
      (void)cuMemHostUnregister(mapping);
    }
    munmap(mapping, storage_size);
    return result == CUDA_SUCCESS ? CUDA_ERROR_OUT_OF_MEMORY : result;
  }
  *pointer = mapping;
  *device_pointer = mapped_device_pointer;
  return CUDA_SUCCESS;
#endif
}

CUresult lupine_server_host_alloc(conn_t *conn, void **pointer,
                                  CUdeviceptr *device_pointer, size_t bytes,
                                  unsigned int flags) {
  if (conn->va_size == 0 && *pointer == nullptr) {
    CUresult result = cuMemHostAlloc(pointer, bytes, flags);
    if (result == CUDA_SUCCESS && (flags & CU_MEMHOSTALLOC_DEVICEMAP) != 0 &&
        cuMemHostGetDevicePointer(device_pointer, *pointer, 0) !=
            CUDA_SUCCESS) {
      *device_pointer = 0;
    }
    return result;
  }
  unsigned int register_flags = CU_MEMHOSTREGISTER_DEVICEMAP;
  if ((flags & CU_MEMHOSTALLOC_PORTABLE) != 0) {
    register_flags |= CU_MEMHOSTREGISTER_PORTABLE;
  }
  return lupine_server_map_host_allocation(conn, pointer, device_pointer, bytes,
                                           flags, register_flags);
}

CUresult lupine_server_managed_alloc(conn_t *conn, CUdeviceptr *pointer,
                                     size_t bytes, unsigned int flags) {
  if (conn->va_size == 0) {
    return cuMemAllocManaged(pointer, bytes, flags);
  }
#if !defined(__linux__)
  return CUDA_ERROR_NOT_SUPPORTED;
#else
  CUresult result = lupine_in_va_arena(
      conn, [&] { return cuMemAllocManaged(pointer, bytes, flags); });
  if (result != CUDA_SUCCESS) {
    return result;
  }
  uintptr_t address = *pointer;
  if (!lupine_va_contains(conn, address, bytes) ||
      !lupine_record_server_allocation(
          address,
          {conn, bytes, flags, lupine_server_allocation_kind::managed})) {
    (void)cuMemFree_v2(address);
    *pointer = 0;
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return CUDA_SUCCESS;
#endif
}

CUresult
lupine_free_server_allocation(uintptr_t address,
                              const lupine_server_allocation &allocation) {
  CUresult result = CUDA_SUCCESS;
  if (allocation.kind == lupine_server_allocation_kind::host) {
    result = cuMemHostUnregister(reinterpret_cast<void *>(address));
#if defined(__linux__)
    if (result == CUDA_SUCCESS &&
        munmap(reinterpret_cast<void *>(address), allocation.size) != 0) {
      result = CUDA_ERROR_UNKNOWN;
    }
#endif
  } else {
    result = cuMemFree_v2(address);
  }
  return result;
}

// The native free runs unlocked: cuMemFree synchronizes the device, and other
// lanes' allocations must not wait behind it.
bool lupine_take_server_allocation(uintptr_t address,
                                   lupine_server_allocation_kind kind,
                                   lupine_server_allocation *allocation) {
  std::lock_guard<std::mutex> lock(lupine_server_state.mutex);
  auto &allocations = lupine_server_state.allocations;
  auto it = allocations.find(address);
  if (it == allocations.end() || it->second.kind != kind) {
    return false;
  }
  *allocation = it->second;
  allocations.erase(it);
  return true;
}

CUresult
lupine_release_server_allocation(uintptr_t address,
                                 const lupine_server_allocation &allocation) {
  CUresult result = lupine_free_server_allocation(address, allocation);
  if (result != CUDA_SUCCESS) {
    (void)lupine_record_server_allocation(address, allocation);
  }
  return result;
}

bool lupine_server_host_allocation_flags(void *pointer, unsigned int *flags) {
  std::lock_guard<std::mutex> lock(lupine_server_state.mutex);
  auto it = lupine_server_state.allocations.find(
      reinterpret_cast<uintptr_t>(pointer));
  if (it == lupine_server_state.allocations.end() ||
      it->second.kind != lupine_server_allocation_kind::host) {
    return false;
  }
  *flags = it->second.flags;
  return true;
}

} // namespace

#if defined(__linux__)
extern "C" __attribute__((visibility("default"))) void *
mmap(void *address, size_t size, int protection, int flags, int fd,
     off_t offset) {
  conn_t *conn = lupine_server_va_connection;
  if (conn == nullptr || address != nullptr || fd != -1 ||
      protection != PROT_NONE || (flags & MAP_ANONYMOUS) == 0 ||
      (flags & MAP_STACK) != 0) {
    conn = nullptr;
  }
  return lupine_server_mmap(conn, address, size, LUPINE_SERVER_VA_ALIGNMENT,
                            protection, flags, fd, offset);
}

extern "C" __attribute__((visibility("default"))) void *
mmap64(void *address, size_t size, int protection, int flags, int fd,
       off64_t offset) {
  static_assert(sizeof(off_t) == sizeof(off64_t));
  return mmap(address, size, protection, flags, fd, static_cast<off_t>(offset));
}
#endif

// Resolve the device alias here so the client does not need a second round
// trip for it.
int handle_cuMemHostAlloc(conn_t *conn) {
  void *pp = nullptr;
  size_t bytesize = 0;
  unsigned int flags = 0;
  int request_id = -1;
  if (!lupine_read_all(conn, pp, bytesize, flags) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUdeviceptr device_ptr = 0;
  CUresult result =
      lupine_server_host_alloc(conn, &pp, &device_ptr, bytesize, flags);
  return lupine_write_response(conn, request_id, pp, device_ptr, result);
}

int handle_cuMemHostGetFlags(conn_t *conn) {
  unsigned int flags = 0;
  void *p = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &p, sizeof(p)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUresult result = lupine_server_host_allocation_flags(p, &flags)
                        ? CUDA_SUCCESS
                        : cuMemHostGetFlags(&flags, p);
  return lupine_write_response(conn, request_id, flags, result);
}

int handle_cuMemFreeHost(conn_t *conn) {
  void *pointer = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &pointer, sizeof(pointer)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  lupine_server_allocation allocation;
  CUresult result =
      pointer == nullptr
          ? CUDA_SUCCESS
          : (lupine_take_server_allocation(
                 address, lupine_server_allocation_kind::host, &allocation)
                 ? lupine_release_server_allocation(address, allocation)
                 : cuMemFreeHost(pointer));
  if (result == CUDA_SUCCESS) {
    lupine_dtoh_sync_contexts(conn);
  }
  // Copies into the memory have numbered their deliveries by now; the client
  // applies them before it unmaps its view.
  uint64_t delivered = lupine_dtoh_watermark(conn);
  return lupine_write_response(conn, request_id, delivered, result);
}

int handle_cuMemAllocManaged(conn_t *conn) {
  CUdeviceptr pointer = 0;
  size_t bytes = 0;
  unsigned int flags = 0;
  int request_id = -1;
  if (!lupine_read_all(conn, pointer, bytes, flags) ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUresult result = lupine_server_managed_alloc(conn, &pointer, bytes, flags);
  return lupine_write_response(conn, request_id, pointer, result);
}

int handle_cuMemFree_v2(conn_t *conn) {
  CUdeviceptr pointer = 0;
  int request_id = -1;
  if (rpc_read(conn, &pointer, sizeof(pointer)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  lupine_server_allocation allocation;
  CUresult result;
  if (lupine_take_server_allocation(
          pointer, lupine_server_allocation_kind::managed, &allocation)) {
    result = lupine_release_server_allocation(pointer, allocation);
  } else {
    auto release = reinterpret_cast<decltype(&cuMemFree_v2)>(
        lupine_server_checkpoint_cuda_symbol("cuMemFree_v2"));
    result = release != nullptr ? release(pointer) : cuMemFree_v2(pointer);
  }
  return lupine_write_response(conn, request_id, result);
}

void lupine_server_cleanup_identity_allocations(conn_t *conn) {
  std::lock_guard<std::mutex> lock(lupine_server_state.mutex);
  auto &allocations = lupine_server_state.allocations;
  for (auto it = allocations.begin(); it != allocations.end();) {
    if (it->second.conn != conn) {
      ++it;
      continue;
    }
    (void)lupine_free_server_allocation(it->first, it->second);
    it = allocations.erase(it);
  }
}

int handle_cuGraphLaunch(conn_t *conn) {
  CUgraphExec exec = nullptr;
  CUstream stream = nullptr;
  if (rpc_read(conn, &exec, sizeof(exec)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 || rpc_read_end(conn) < 0) {
    return -1;
  }
  lupine_dtoh_wait_for_room(conn);
  lupine_keep_async_error(conn, cuGraphLaunch(exec, stream));
  return 0;
}

// Context handlers. The staging bookkeeping they drive lives in
// cuda_server_memcpy.cpp, because it exists to own copy staging buffers.
extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *context,
                                           unsigned int flags, CUdevice device);

static int lupine_handle_context_acquire(conn_t *conn, bool primary) {
  unsigned int flags = 0;
  CUdevice device = 0;
  int request_id = -1;
  if ((!primary && rpc_read(conn, &flags, sizeof(flags)) < 0) ||
      rpc_read(conn, &device, sizeof(device)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }
  CUcontext context = nullptr;
  lupine_server_begin_lifecycle_transaction(conn);
  lupine_monitoring_begin_context_create(device);
  CUresult result = primary ? cuDevicePrimaryCtxRetain(&context, device)
                            : cuCtxCreate_v2(&context, flags, device);
  lupine_monitoring_end_context_create(result == CUDA_SUCCESS);
  if (primary) {
    lupine_server_note_primary_context(conn, device, context, result);
  } else {
    lupine_server_note_created_context(conn, context, result);
  }
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_response(conn, request_id, context, result);
}

int handle_cuCtxCreate_v2(conn_t *conn) {
  return lupine_handle_context_acquire(conn, false);
}

// The first event is the caller's allocation; later creations are speculative.
// A failure stops the batch. If the first creation failed, report it without
// modifying the caller's output; otherwise hand out only real, created events.
int handle_lupineEventCreateBatch(conn_t *conn) {
  constexpr uint32_t kBatchSize = 32;
  CUcontext context = nullptr;
  unsigned int flags = 0;
  if (rpc_read(conn, &context, sizeof(context)) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  CUevent events[kBatchSize];
  CUdevice device = -1;
  uint32_t count = 0;
  CUresult result = cuCtxPushCurrent(context);
  if (result == CUDA_SUCCESS) {
    result = cuCtxGetDevice(&device);
    if (result == CUDA_SUCCESS) {
      for (; count < kBatchSize; ++count) {
        result = cuEventCreate(&events[count], flags);
        if (result != CUDA_SUCCESS) {
          break;
        }
      }
    }
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }
  if (count != 0) {
    result = CUDA_SUCCESS;
  }
  return lupine_write_response(
      conn, request_id, result, device, count,
      lupine_raw_bytes{events, count * sizeof(*events)});
}

// Creates the client's stream pool under the context it names in one round
// trip: kStreamPoolPerPriority streams at each usable priority from the least
// downward, capped at kStreamPoolMaxPriorities, in torch's c10 creation order
// (stream-major, priority-minor), all CU_STREAM_NON_BLOCKING. Creation stops
// at the first failure; the streams already created are still returned.
int handle_lupineStreamPoolInit(conn_t *conn) {
  constexpr uint32_t kStreamPoolPerPriority = 32;
  constexpr uint32_t kStreamPoolMaxPriorities = 4;
  CUcontext context = nullptr;
  if (rpc_read(conn, &context, sizeof(context)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }

  CUstream streams[kStreamPoolPerPriority * kStreamPoolMaxPriorities];
  uint32_t created = 0;
  uint32_t priorities = 0;
  int least = 0;
  int greatest = 0;
  if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
    priorities = 1;
    if (cuCtxGetStreamPriorityRange(&least, &greatest) == CUDA_SUCCESS) {
      priorities =
          std::min<uint32_t>(kStreamPoolMaxPriorities,
                             static_cast<uint32_t>(least - greatest) + 1);
    }
    for (; created < kStreamPoolPerPriority * priorities; ++created) {
      int priority = least - static_cast<int>(created % priorities);
      if (cuStreamCreateWithPriority(&streams[created], CU_STREAM_NON_BLOCKING,
                                     priority) != CUDA_SUCCESS) {
        break;
      }
    }
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
  }

  return lupine_write_response(
      conn, request_id, created, priorities, least,
      lupine_raw_bytes{streams, created * sizeof(*streams)});
}

int handle_cuDevicePrimaryCtxRetain(conn_t *conn) {
  return lupine_handle_context_acquire(conn, true);
}

static int handle_cuDevicePrimaryCtxTeardown(conn_t *conn, bool reset) {
  CUdevice device = 0;
  int request_id = -1;
  if (rpc_read(conn, &device, sizeof(device)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  lupine_server_begin_lifecycle_transaction(conn);
  lupine_server_prepare_primary_context(conn, device);
  CUresult result = reset ? cuDevicePrimaryCtxReset_v2(device)
                          : cuDevicePrimaryCtxRelease_v2(device);
  lupine_server_finish_primary_context(conn, device, reset, result);
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_cuDevicePrimaryCtxRelease_v2(conn_t *conn) {
  return handle_cuDevicePrimaryCtxTeardown(conn, false);
}

int handle_cuDevicePrimaryCtxReset_v2(conn_t *conn) {
  return handle_cuDevicePrimaryCtxTeardown(conn, true);
}

int handle_cuCtxAttach(conn_t *conn) {
  unsigned int flags = 0;
  int request_id = -1;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  CUcontext context = nullptr;
  lupine_server_begin_lifecycle_transaction(conn);
  CUresult result = cuCtxAttach(&context, flags);
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_response(conn, request_id, context, result);
}

static int handle_cuCtxTeardown(conn_t *conn, bool detach) {
  CUcontext context = nullptr;
  int request_id = -1;
  if (rpc_read(conn, &context, sizeof(context)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0) {
    return -1;
  }

  lupine_server_begin_lifecycle_transaction(conn);
  lupine_server_prepare_context_destroy(conn, context);
  CUresult result = detach ? cuCtxDetach(context) : cuCtxDestroy_v2(context);
  if (detach) {
    lupine_server_finish_context_detach(conn, context, result);
  } else {
    lupine_server_finish_context_destroy(conn, context, result);
  }
  lupine_server_end_lifecycle_transaction(conn);
  return lupine_write_lifecycle_response(conn, request_id, result);
}

int handle_cuCtxDestroy_v2(conn_t *conn) {
  return handle_cuCtxTeardown(conn, false);
}

int handle_cuCtxDetach(conn_t *conn) {
  return handle_cuCtxTeardown(conn, true);
}
