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

// The call's status, then its strings as a count and, for each, its length
// and bytes; no strings after a failure.
int write_strings(conn_t *conn, int request_id, ncclResult_t status,
                  const std::vector<std::pair<const char *, size_t>> &strings) {
  const uint32_t count =
      status == ncclSuccess ? static_cast<uint32_t>(strings.size()) : 0;
  std::vector<uint32_t> lengths(count);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  for (uint32_t i = 0; i < count; ++i) {
    lengths[i] = static_cast<uint32_t>(strings[i].second);
    if (rpc_write(conn, &lengths[i], sizeof(lengths[i])) < 0 ||
        (lengths[i] != 0 &&
         rpc_write(conn, strings[i].first, lengths[i]) < 0)) {
      return -1;
    }
  }
  return rpc_write_end(conn) < 0 ? -1 : 0;
}

std::pair<const char *, size_t> text(const char *string) {
  return {string, string == nullptr ? 0 : std::strlen(string)};
}

} // namespace

int handle_ncclGetErrorString(conn_t *conn) {
  ncclResult_t result;
  if (rpc_read(conn, &result, sizeof(result)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(ncclResult_t);
  fn_t fn = nccl_symbol<fn_t>("ncclGetErrorString");
  const char *string = fn == nullptr ? nullptr : fn(result);
  return write_strings(conn, request_id, ncclSuccess, {text(string)});
}

int handle_ncclGetLastError(conn_t *conn) {
  ncclComm_t comm;
  if (rpc_read(conn, &comm, sizeof(comm)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(ncclComm_t);
  fn_t fn = nccl_symbol<fn_t>("ncclGetLastError");
  const char *string = fn == nullptr ? nullptr : fn(comm);
  return write_strings(conn, request_id, ncclSuccess, {text(string)});
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
  return write_strings(conn, request_id, status, {text(string)});
}

int handle_ncclParamGetParameter(conn_t *conn) {
  uint32_t key_length = 0;
  std::string key;
  if (rpc_read(conn, &key_length, sizeof(key_length)) < 0) {
    return -1;
  }
  key.resize(key_length);
  if ((key_length != 0 && rpc_read(conn, &key[0], key_length) < 0)) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = ncclResult_t (*)(const char *, const char **, int *);
  fn_t fn = nccl_symbol<fn_t>("ncclParamGetParameter");
  const char *value = nullptr;
  int length = 0;
  const ncclResult_t status =
      fn == nullptr ? function_not_found() : fn(key.c_str(), &value, &length);
  return write_strings(
      conn, request_id, status,
      {{value, value == nullptr ? 0 : static_cast<size_t>(length)}});
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
  std::vector<std::pair<const char *, size_t>> keys;
  for (int i = 0; status == ncclSuccess && i < count; ++i) {
    keys.push_back(text(table[i]));
  }
  return write_strings(conn, request_id, status, keys);
}
#endif

#include "codegen/gen_nccl_server.inc"
