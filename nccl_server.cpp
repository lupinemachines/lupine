#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>

#include <dlfcn.h>

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

// A call the machine's libnccl predates.
ncclResult_t function_not_found() { return ncclInvalidUsage; }

// The library this server was compiled against; the handlers marshal its
// enums and configuration structs, so another major is not a match.
void *nccl_library() {
  static void *library = []() -> void * {
    const std::string name = "libnccl.so." + std::to_string(NCCL_MAJOR);
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }();
  return library;
}

template <typename Fn> Fn nccl_symbol(const char *name) {
  void *lib = nccl_library();
  return lib == nullptr ? nullptr : reinterpret_cast<Fn>(dlsym(lib, name));
}

struct wire_string {
  const char *data = nullptr;
  uint32_t length = 0;
};

wire_string text(const char *string) {
  return {string,
          string == nullptr ? 0 : static_cast<uint32_t>(std::strlen(string))};
}

// The call's status, then its strings as a count and, for each, its length
// and bytes; no strings after a failure.
int write_strings(conn_t *conn, int request_id, ncclResult_t status,
                  const wire_string *strings, uint32_t string_count) {
  const uint32_t count = status == ncclSuccess ? string_count : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (rpc_write(conn, &strings[i].length, sizeof(strings[i].length)) < 0 ||
        (strings[i].length != 0 &&
         rpc_write(conn, strings[i].data, strings[i].length) < 0)) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
}

int write_string(conn_t *conn, int request_id, ncclResult_t status,
                 wire_string string) {
  return write_strings(conn, request_id, status, &string, 1);
}

template <typename Arg>
int handle_single_string(conn_t *conn, const char *(*fn)(Arg)) {
  Arg arg;
  if (rpc_read(conn, &arg, sizeof(arg)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  return write_string(conn, request_id, ncclSuccess,
                      text(fn == nullptr ? nullptr : fn(arg)));
}

} // namespace

int handle_ncclGetErrorString(conn_t *conn) {
  static const auto fn =
      nccl_symbol<const char *(*)(ncclResult_t)>("ncclGetErrorString");
  return handle_single_string<ncclResult_t>(conn, fn);
}

int handle_ncclGetLastError(conn_t *conn) {
  static const auto fn =
      nccl_symbol<const char *(*)(ncclComm_t)>("ncclGetLastError");
  return handle_single_string<ncclComm_t>(conn, fn);
}

#if NCCL_VERSION_CODE >= 23007
int handle_ncclParamGetStr(conn_t *conn) {
  ncclParamHandle_t h;
  if (rpc_read(conn, &h, sizeof(h)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = ncclResult_t (*)(ncclParamHandle_t, const char **);
  static const fn_t fn = nccl_symbol<fn_t>("ncclParamGetStr");
  const char *string = nullptr;
  const ncclResult_t status =
      fn == nullptr ? function_not_found() : fn(h, &string);
  return write_string(conn, request_id, status, text(string));
}

int handle_ncclParamGetParameter(conn_t *conn) {
  uint32_t key_length = 0;
  if (rpc_read(conn, &key_length, sizeof(key_length)) < 0 || key_length == 0 ||
      key_length > (1U << 20)) {
    return -1;
  }
  lupine_scratch_buffer<256> key_storage;
  char *key_buf = static_cast<char *>(key_storage.alloc_uninit(key_length));
  if (key_buf == nullptr || rpc_read(conn, key_buf, key_length) < 0) {
    return -1;
  }
  key_buf[key_length - 1] = '\0';
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = ncclResult_t (*)(const char *, const char **, int *);
  static const fn_t fn = nccl_symbol<fn_t>("ncclParamGetParameter");
  const char *value = nullptr;
  int length = 0;
  const ncclResult_t status =
      fn == nullptr ? function_not_found() : fn(key_buf, &value, &length);
  return write_string(
      conn, request_id, status,
      {value, value == nullptr ? 0 : static_cast<uint32_t>(length)});
}

int handle_ncclParamGetAllParameterKeys(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = ncclResult_t (*)(const char ***, int *);
  static const fn_t fn = nccl_symbol<fn_t>("ncclParamGetAllParameterKeys");
  const char **table = nullptr;
  int count = 0;
  const ncclResult_t status =
      fn == nullptr ? function_not_found() : fn(&table, &count);
  lupine_scratch_buffer<64 * sizeof(wire_string)> key_storage;
  wire_string *keys = nullptr;
  uint32_t key_count = 0;
  if (status == ncclSuccess && count > 0) {
    keys = static_cast<wire_string *>(key_storage.alloc_array_uninit(
        static_cast<size_t>(count), sizeof(wire_string)));
    if (keys == nullptr) {
      return -1;
    }
    key_count = static_cast<uint32_t>(count);
    for (int i = 0; i < count; ++i) {
      keys[i] = text(table[i]);
    }
  }
  return write_strings(conn, request_id, status, keys, key_count);
}
#endif

#include "codegen/gen_nccl_server.inc"
