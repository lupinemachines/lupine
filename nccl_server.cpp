#include <nccl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

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
int handle_single_string(conn_t *conn, const char *symbol) {
  Arg arg;
  if (rpc_read(conn, &arg, sizeof(arg)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  auto fn = nccl_symbol<const char *(*)(Arg)>(symbol);
  return write_string(conn, request_id, ncclSuccess,
                      text(fn == nullptr ? nullptr : fn(arg)));
}

} // namespace

int handle_ncclGetErrorString(conn_t *conn) {
  return handle_single_string<ncclResult_t>(conn, "ncclGetErrorString");
}

int handle_ncclGetLastError(conn_t *conn) {
  return handle_single_string<ncclComm_t>(conn, "ncclGetLastError");
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
  fn_t fn = nccl_symbol<fn_t>("ncclParamGetStr");
  const char *string = nullptr;
  const ncclResult_t status =
      fn == nullptr ? function_not_found() : fn(h, &string);
  return write_string(conn, request_id, status, text(string));
}

int handle_ncclParamGetParameter(conn_t *conn) {
  uint32_t key_length = 0;
  if (rpc_read(conn, &key_length, sizeof(key_length)) < 0 || key_length == 0) {
    return -1;
  }
  char stack_key[256];
  std::string heap_key;
  char *key_buf = stack_key;
  if (key_length > sizeof(stack_key)) {
    heap_key.resize(key_length);
    key_buf = &heap_key[0];
  }
  if (rpc_read(conn, key_buf, key_length) < 0) {
    return -1;
  }
  key_buf[key_length - 1] = '\0';
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = ncclResult_t (*)(const char *, const char **, int *);
  fn_t fn = nccl_symbol<fn_t>("ncclParamGetParameter");
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
  fn_t fn = nccl_symbol<fn_t>("ncclParamGetAllParameterKeys");
  const char **table = nullptr;
  int count = 0;
  const ncclResult_t status =
      fn == nullptr ? function_not_found() : fn(&table, &count);
  std::vector<wire_string> keys;
  if (status == ncclSuccess && count > 0) {
    keys.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      keys.push_back(text(table[i]));
    }
  }
  return write_strings(conn, request_id, status, keys.data(),
                       static_cast<uint32_t>(keys.size()));
}
#endif

#include "codegen/gen_nccl_server.inc"
