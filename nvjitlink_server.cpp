#include <cuda.h>
#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

// nvJitLink has no status for a missing library; a server without one reports
// an internal error.
nvJitLinkResult function_not_found() { return NVJITLINK_ERROR_INTERNAL; }

// The library of the toolkit this server was compiled against.
void *nvjitlink_library() {
  static void *library = []() -> void * {
    const int major = CUDA_VERSION / 1000;
#ifdef _WIN32
    const std::string name =
        "nvJitLink_" + std::to_string(major * 10) + "_0.dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libnvJitLink.so." + std::to_string(major);
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

void *lookup(void *lib, const std::string &name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      GetProcAddress(static_cast<HMODULE>(lib), name.c_str()));
#else
  return dlsym(lib, name.c_str());
#endif
}

// Libraries before 12.4 export each call only under its release-suffixed
// name; the newest release the machine's library carries is the one its
// unsuffixed name would call.
template <typename Fn> Fn nvjitlink_symbol(const char *name) {
  void *lib = nvjitlink_library();
  if (lib == nullptr) {
    return nullptr;
  }
  if (void *symbol = lookup(lib, name)) {
    return reinterpret_cast<Fn>(symbol);
  }
  const std::string release = "__" + std::string(name) + "_" +
                              std::to_string(CUDA_VERSION / 1000) + "_";
  for (int minor = 99; minor >= 0; --minor) {
    if (void *symbol = lookup(lib, release + std::to_string(minor))) {
      return reinterpret_cast<Fn>(symbol);
    }
  }
  return nullptr;
}

// A string travels as a presence byte, then its length and bytes when present.
struct wire_string {
  bool present = false;
  std::string text;
  const char *get() const { return present ? text.c_str() : nullptr; }
};

int read_string(conn_t *conn, wire_string *out) {
  uint8_t present = 0;
  uint64_t length = 0;
  if (rpc_read(conn, &present, sizeof(present)) < 0) {
    return -1;
  }
  out->present = present != 0;
  if (!out->present) {
    return 0;
  }
  if (rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  out->text.resize(length);
  return length == 0 ? 0 : rpc_read(conn, &out->text[0], length);
}

// Every output a Get<X> call copies out is sized by its Get<X>Size call, so
// the handler asks for the size itself and the response carries the status,
// then the size and bytes after a success. The logs write their terminator
// even when they report no size.
int handle_output(conn_t *conn, const char *size_symbol, const char *get_symbol,
                  bool terminated) {
  nvJitLinkHandle handle;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using size_fn_t = nvJitLinkResult (*)(nvJitLinkHandle, size_t *);
  using get_fn_t = nvJitLinkResult (*)(nvJitLinkHandle, void *);
  size_fn_t size_fn = nvjitlink_symbol<size_fn_t>(size_symbol);
  get_fn_t get_fn = nvjitlink_symbol<get_fn_t>(get_symbol);
  size_t size = 0;
  std::vector<char> bytes;
  nvJitLinkResult status = size_fn == nullptr || get_fn == nullptr
                               ? function_not_found()
                               : size_fn(handle, &size);
  if (status == NVJITLINK_SUCCESS) {
    bytes.assign(terminated && size == 0 ? 1 : size, '\0');
    status = get_fn(handle, bytes.data());
  }
  const uint64_t length = status == NVJITLINK_SUCCESS ? bytes.size() : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_write(conn, bytes.data(), length) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int write_status(conn_t *conn, int request_id, nvJitLinkResult status) {
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &status, sizeof(status)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

} // namespace

// The options travel as a presence byte, then each option as a string.
int handle_nvJitLinkCreate(conn_t *conn) {
  uint32_t num_options = 0;
  uint8_t present = 0;
  if (rpc_read(conn, &num_options, sizeof(num_options)) < 0 ||
      rpc_read(conn, &present, sizeof(present)) < 0) {
    return -1;
  }
  std::vector<wire_string> options(present ? num_options : 0);
  for (wire_string &option : options) {
    if (read_string(conn, &option) < 0) {
      return -1;
    }
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  std::vector<const char *> pointers;
  for (const wire_string &option : options) {
    pointers.push_back(option.get());
  }
  using fn_t = nvJitLinkResult (*)(nvJitLinkHandle *, uint32_t, const char **);
  fn_t fn = nvjitlink_symbol<fn_t>("nvJitLinkCreate");
  nvJitLinkHandle handle = nullptr;
  const nvJitLinkResult status =
      fn == nullptr
          ? function_not_found()
          : fn(&handle, num_options, present ? pointers.data() : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &handle, sizeof(handle)) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// The data travels as a presence byte, then its size and bytes when present,
// followed by the name as a string.
int handle_nvJitLinkAddData(conn_t *conn) {
  nvJitLinkHandle handle;
  int32_t input_type = 0;
  uint8_t present = 0;
  uint64_t size = 0;
  std::vector<char> data;
  wire_string name;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &input_type, sizeof(input_type)) < 0 ||
      rpc_read(conn, &present, sizeof(present)) < 0 ||
      rpc_read(conn, &size, sizeof(size)) < 0) {
    return -1;
  }
  if (present) {
    data.resize(size);
    if (size != 0 && rpc_read(conn, data.data(), size) < 0) {
      return -1;
    }
  }
  if (read_string(conn, &name) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType,
                                   const void *, size_t, const char *);
  fn_t fn = nvjitlink_symbol<fn_t>("nvJitLinkAddData");
  const nvJitLinkResult status =
      fn == nullptr ? function_not_found()
                    : fn(handle, static_cast<nvJitLinkInputType>(input_type),
                         present ? data.data() : nullptr, size, name.get());
  return write_status(conn, request_id, status);
}

// Only a file the client could not open reaches the server, which reports it
// the way the library does for its own missing files.
int handle_nvJitLinkAddFile(conn_t *conn) {
  nvJitLinkHandle handle;
  int32_t input_type = 0;
  wire_string file_name;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &input_type, sizeof(input_type)) < 0 ||
      read_string(conn, &file_name) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t =
      nvJitLinkResult (*)(nvJitLinkHandle, nvJitLinkInputType, const char *);
  fn_t fn = nvjitlink_symbol<fn_t>("nvJitLinkAddFile");
  const nvJitLinkResult status =
      fn == nullptr ? function_not_found()
                    : fn(handle, static_cast<nvJitLinkInputType>(input_type),
                         file_name.get());
  return write_status(conn, request_id, status);
}

int handle_nvJitLinkGetLinkedCubin(conn_t *conn) {
  return handle_output(conn, "nvJitLinkGetLinkedCubinSize",
                       "nvJitLinkGetLinkedCubin", false);
}

int handle_nvJitLinkGetLinkedPtx(conn_t *conn) {
  return handle_output(conn, "nvJitLinkGetLinkedPtxSize",
                       "nvJitLinkGetLinkedPtx", false);
}

#if LUPINE_NVJITLINK_HAS_LTOIR
int handle_nvJitLinkGetLinkedLTOIR(conn_t *conn) {
  return handle_output(conn, "nvJitLinkGetLinkedLTOIRSize",
                       "nvJitLinkGetLinkedLTOIR", false);
}
#endif

int handle_nvJitLinkGetErrorLog(conn_t *conn) {
  return handle_output(conn, "nvJitLinkGetErrorLogSize", "nvJitLinkGetErrorLog",
                       true);
}

int handle_nvJitLinkGetInfoLog(conn_t *conn) {
  return handle_output(conn, "nvJitLinkGetInfoLogSize", "nvJitLinkGetInfoLog",
                       true);
}

#include "codegen/gen_nvjitlink_server.inc"
