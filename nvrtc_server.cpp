#include <cuda.h>
#include <nvrtc.h>

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

// NVRTC has no status for a missing library; a server without one reports the
// failure it gives for a compiler that cannot start.
nvrtcResult function_not_found() {
  return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
}

// The library of the toolkit this server was compiled against. NVRTC 11.x
// kept the 11.2 soname across every 11 release.
void *nvrtc_library() {
  static void *library = []() -> void * {
    const int major = CUDA_VERSION / 1000;
#ifdef _WIN32
    const std::string name =
        "nvrtc64_" +
        std::string(major == 11 ? "112" : std::to_string(major * 10)) +
        "_0.dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name =
        "libnvrtc.so." +
        std::string(major == 11 ? "11.2" : std::to_string(major));
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn nvrtc_symbol(const char *name) {
  void *lib = nvrtc_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
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

// rpc_write queues the caller's bytes until the response is sent, so strings
// are encoded into a buffer the handler keeps alive until then.
void encode_string(std::vector<char> *out, const char *text) {
  const uint8_t present = text != nullptr;
  out->push_back(static_cast<char>(present));
  if (!present) {
    return;
  }
  const uint64_t length = strlen(text);
  const size_t at = out->size();
  out->resize(at + sizeof(length) + length);
  memcpy(out->data() + at, &length, sizeof(length));
  memcpy(out->data() + at + sizeof(length), text, length);
}

// An array of strings travels as a presence byte, then each element as a
// string, for the count the call names.
int read_string_array(conn_t *conn, int count,
                      std::vector<wire_string> *strings,
                      std::vector<const char *> *pointers, bool *present) {
  uint8_t flag = 0;
  if (rpc_read(conn, &flag, sizeof(flag)) < 0) {
    return -1;
  }
  *present = flag != 0;
  if (!*present) {
    return 0;
  }
  strings->resize(count < 0 ? 0 : count);
  for (wire_string &text : *strings) {
    if (read_string(conn, &text) < 0) {
      return -1;
    }
  }
  for (const wire_string &text : *strings) {
    pointers->push_back(text.get());
  }
  return 0;
}

// Every output an nvrtcGet<X> call copies out is sized by its nvrtcGet<X>Size
// call, so the handler asks for the size itself and the response carries the
// status, then the size and bytes after a success.
int handle_output(conn_t *conn, const char *size_symbol,
                  const char *get_symbol) {
  nvrtcProgram prog;
  if (rpc_read(conn, &prog, sizeof(prog)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using size_fn_t = nvrtcResult (*)(nvrtcProgram, size_t *);
  using get_fn_t = nvrtcResult (*)(nvrtcProgram, char *);
  size_fn_t size_fn = nvrtc_symbol<size_fn_t>(size_symbol);
  get_fn_t get_fn = nvrtc_symbol<get_fn_t>(get_symbol);
  size_t size = 0;
  std::vector<char> bytes;
  nvrtcResult status = size_fn == nullptr || get_fn == nullptr
                           ? function_not_found()
                           : size_fn(prog, &size);
  if (status == NVRTC_SUCCESS) {
    bytes.resize(size);
    status = get_fn(prog, bytes.data());
  }
  const uint64_t length = status == NVRTC_SUCCESS ? size : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_write(conn, bytes.data(), length) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#if CUDA_VERSION >= 13030
// The bundled headers calls take an optional error log slot; the response
// carries the status, then the log the library pointed it at, if any.
int write_bundled_response(conn_t *conn, int request_id, nvrtcResult status,
                           const char *log) {
  std::vector<char> response;
  encode_string(&response, log);
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &status, sizeof(status)) < 0 ||
                 rpc_write(conn, response.data(), response.size()) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}
#endif

} // namespace

// Static strings travel as their length and bytes.
int handle_nvrtcGetErrorString(conn_t *conn) {
  nvrtcResult result;
  if (rpc_read(conn, &result, sizeof(result)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(nvrtcResult);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcGetErrorString");
  std::vector<char> response;
  encode_string(&response, fn == nullptr ? nullptr : fn(result));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, response.data(), response.size()) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// The response carries the status, then the count and architectures.
int handle_nvrtcGetSupportedArchs(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using count_fn_t = nvrtcResult (*)(int *);
  using fn_t = nvrtcResult (*)(int *);
  count_fn_t count_fn = nvrtc_symbol<count_fn_t>("nvrtcGetNumSupportedArchs");
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcGetSupportedArchs");
  int count = 0;
  std::vector<int> archs;
  nvrtcResult status = count_fn == nullptr || fn == nullptr
                           ? function_not_found()
                           : count_fn(&count);
  if (status == NVRTC_SUCCESS) {
    archs.resize(count);
    status = fn(archs.data());
  }
  const int32_t length = status == NVRTC_SUCCESS ? count : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      (length != 0 &&
       rpc_write(conn, archs.data(), length * sizeof(int)) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nvrtcCreateProgram(conn_t *conn) {
  wire_string src, name;
  int32_t num_headers = 0;
  std::vector<wire_string> headers, include_names;
  std::vector<const char *> header_pointers, include_pointers;
  bool headers_present = false, includes_present = false;
  if (rpc_read(conn, &num_headers, sizeof(num_headers)) < 0 ||
      read_string(conn, &src) < 0 || read_string(conn, &name) < 0 ||
      read_string_array(conn, num_headers, &headers, &header_pointers,
                        &headers_present) < 0 ||
      read_string_array(conn, num_headers, &include_names, &include_pointers,
                        &includes_present) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvrtcResult (*)(nvrtcProgram *, const char *, const char *, int,
                               const char *const *, const char *const *);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcCreateProgram");
  nvrtcProgram prog = nullptr;
  const nvrtcResult status =
      fn == nullptr ? function_not_found()
                    : fn(&prog, src.get(), name.get(), num_headers,
                         headers_present ? header_pointers.data() : nullptr,
                         includes_present ? include_pointers.data() : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &prog, sizeof(prog)) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nvrtcCompileProgram(conn_t *conn) {
  nvrtcProgram prog;
  int32_t num_options = 0;
  std::vector<wire_string> options;
  std::vector<const char *> option_pointers;
  bool options_present = false;
  if (rpc_read(conn, &prog, sizeof(prog)) < 0 ||
      rpc_read(conn, &num_options, sizeof(num_options)) < 0 ||
      read_string_array(conn, num_options, &options, &option_pointers,
                        &options_present) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvrtcResult (*)(nvrtcProgram, int, const char *const *);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcCompileProgram");
  const nvrtcResult status =
      fn == nullptr ? function_not_found()
                    : fn(prog, num_options,
                         options_present ? option_pointers.data() : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nvrtcGetPTX(conn_t *conn) {
  return handle_output(conn, "nvrtcGetPTXSize", "nvrtcGetPTX");
}

int handle_nvrtcGetCUBIN(conn_t *conn) {
  return handle_output(conn, "nvrtcGetCUBINSize", "nvrtcGetCUBIN");
}

int handle_nvrtcGetProgramLog(conn_t *conn) {
  return handle_output(conn, "nvrtcGetProgramLogSize", "nvrtcGetProgramLog");
}

#if CUDA_VERSION >= 12000
int handle_nvrtcGetLTOIR(conn_t *conn) {
  return handle_output(conn, "nvrtcGetLTOIRSize", "nvrtcGetLTOIR");
}

int handle_nvrtcGetOptiXIR(conn_t *conn) {
  return handle_output(conn, "nvrtcGetOptiXIRSize", "nvrtcGetOptiXIR");
}
#endif

#if CUDA_VERSION < 13000
int handle_nvrtcGetNVVM(conn_t *conn) {
  return handle_output(conn, "nvrtcGetNVVMSize", "nvrtcGetNVVM");
}
#endif

#if CUDA_VERSION >= 13020
int handle_nvrtcGetTileIR(conn_t *conn) {
  return handle_output(conn, "nvrtcGetTileIRSize", "nvrtcGetTileIR");
}
#endif

// The response carries the status, then the lowered name after a success.
int handle_nvrtcGetLoweredName(conn_t *conn) {
  nvrtcProgram prog;
  wire_string expression;
  if (rpc_read(conn, &prog, sizeof(prog)) < 0 ||
      read_string(conn, &expression) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvrtcResult (*)(nvrtcProgram, const char *, const char **);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcGetLoweredName");
  const char *lowered = nullptr;
  const nvrtcResult status = fn == nullptr
                                 ? function_not_found()
                                 : fn(prog, expression.get(), &lowered);
  std::vector<char> response;
  encode_string(&response, status == NVRTC_SUCCESS ? lowered : nullptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, response.data(), response.size()) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#if CUDA_VERSION >= 13030
int handle_nvrtcInstallBundledHeaders(conn_t *conn) {
  wire_string path;
  uint32_t flags = 0;
  uint8_t wants_log = 0;
  if (read_string(conn, &path) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0 ||
      rpc_read(conn, &wants_log, sizeof(wants_log)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvrtcResult (*)(const char *, unsigned int, const char **);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcInstallBundledHeaders");
  const char *log = nullptr;
  const nvrtcResult status =
      fn == nullptr ? function_not_found()
                    : fn(path.get(), flags, wants_log ? &log : nullptr);
  return write_bundled_response(conn, request_id, status, log);
}

// The response carries the information after the status and log.
int handle_nvrtcGetBundledHeadersInfo(conn_t *conn) {
  uint8_t wants_info = 0, wants_log = 0;
  if (rpc_read(conn, &wants_info, sizeof(wants_info)) < 0 ||
      rpc_read(conn, &wants_log, sizeof(wants_log)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvrtcResult (*)(nvrtcBundledHeadersInfo *, const char **);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcGetBundledHeadersInfo");
  nvrtcBundledHeadersInfo info{};
  const char *log = nullptr;
  const nvrtcResult status = fn == nullptr ? function_not_found()
                                           : fn(wants_info ? &info : nullptr,
                                                wants_log ? &log : nullptr);
  std::vector<char> response;
  encode_string(&response, log);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, response.data(), response.size()) < 0 ||
      rpc_write(conn, &info, sizeof(info)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nvrtcRemoveBundledHeaders(conn_t *conn) {
  wire_string path;
  uint8_t wants_log = 0;
  if (read_string(conn, &path) < 0 ||
      rpc_read(conn, &wants_log, sizeof(wants_log)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvrtcResult (*)(const char *, const char **);
  fn_t fn = nvrtc_symbol<fn_t>("nvrtcRemoveBundledHeaders");
  const char *log = nullptr;
  const nvrtcResult status = fn == nullptr
                                 ? function_not_found()
                                 : fn(path.get(), wants_log ? &log : nullptr);
  return write_bundled_response(conn, request_id, status, log);
}
#endif

#include "codegen/gen_nvrtc_server.inc"
