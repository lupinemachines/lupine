// NVRTC (libnvrtc) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libnvrtc, sent on the
// CUDA driver shim's connections, so a program compiles on the machine that
// loads its output. Most of the surface is generated; the calls in this file
// carry string arrays, copy out outputs sized by an earlier call, or return
// strings the library owns.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

namespace {

// NVRTC has no status for an unreachable server; it reports the failure it
// gives for a compiler that cannot start.
nvrtcResult rpc_error() { return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE; }

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  return lupine_rpc_write_start_request(conn, op);
}
int rpc_write(conn_t *conn, const void *data, size_t size) {
  return lupine_rpc_write(conn, data, size);
}
int rpc_wait_for_response(conn_t *conn) {
  return lupine_rpc_wait_for_response(conn);
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  return lupine_rpc_read(conn, data, size);
}
int rpc_read_end(conn_t *conn) { return lupine_rpc_read_end(conn); }

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

// A program is created on the runtime's current device and routes every later
// call to that connection. The handle the caller holds is the client's: the
// server's program behind it is replaced when compiling needs files only the
// client has, and a handle the server frees is never handed out again while
// the caller still holds it. The creation arguments and name expressions are
// kept to create that replacement. Lowered names are strings the library owns
// until the program is destroyed, so the client keeps a copy of each for as
// long and hands out that copy's address.
struct program_state {
  conn_t *conn;
  nvrtcProgram server;
  std::string src;
  std::unique_ptr<std::string> name;
  std::vector<std::string> headers;
  std::vector<std::string> include_names;
  std::vector<std::string> name_expressions;
  std::unordered_map<std::string, std::string> lowered_names;
};

std::mutex programs_mutex;
std::unordered_map<void *, std::unique_ptr<program_state>> programs;

program_state *state_of(nvrtcProgram prog) {
  std::lock_guard<std::mutex> lock(programs_mutex);
  auto it = programs.find(prog);
  return it == programs.end() ? nullptr : it->second.get();
}

nvrtcProgram server_program(nvrtcProgram prog) {
  program_state *state = state_of(prog);
  return state == nullptr ? nullptr : state->server;
}

void note_name_expression(nvrtcProgram prog, const char *name_expression) {
  std::lock_guard<std::mutex> lock(programs_mutex);
  auto it = programs.find(prog);
  if (it != programs.end()) {
    it->second->name_expressions.push_back(name_expression);
  }
}

// A call without a program goes to the runtime's current device, which the
// runtime shim answers locally.
conn_t *connection() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  return lupine_rpc_conn_for_device(&device);
}

conn_t *connection_for_handle(void *handle) {
  program_state *state = state_of(static_cast<nvrtcProgram>(handle));
  return state == nullptr ? nullptr : state->conn;
}

// rpc_write queues the caller's bytes until the request is sent, so strings
// are encoded into a buffer the calling function keeps alive until then. A
// string is a presence byte, then its length and bytes when present.
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

int read_string(conn_t *conn, bool *present, std::string *text) {
  uint8_t flag = 0;
  uint64_t length = 0;
  if (rpc_read(conn, &flag, sizeof(flag)) < 0) {
    return -1;
  }
  *present = flag != 0;
  if (!*present) {
    return 0;
  }
  if (rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  text->assign(length, '\0');
  return length == 0 ? 0 : rpc_read(conn, &(*text)[0], length);
}

// An array of strings is a presence byte, then each element as a string, for
// the count the call names.
void encode_string_array(std::vector<char> *out, int count,
                         const char *const *strings) {
  out->push_back(static_cast<char>(strings != nullptr));
  for (int i = 0; strings != nullptr && i < count; ++i) {
    encode_string(out, strings[i]);
  }
}

// An output an nvrtcGet<X> call copies out: the caller's buffer holds the size
// its nvrtcGet<X>Size call reported, and nothing is written after a failure.
nvrtcResult copy_output(int op, nvrtcProgram program, char *out) {
  conn_t *conn = connection_for_handle(program);
  const nvrtcProgram prog = server_program(program);
  nvrtcResult status = rpc_error();
  uint64_t length = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &prog, sizeof(prog)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 ||
      (length != 0 && rpc_read(conn, out, length) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

#if CUDA_VERSION >= 13030
// The library points an error log slot at a string the next call on the same
// thread may replace; the copy handed out lives as long.
thread_local std::string bundled_headers_log;

nvrtcResult finish_bundled_call(conn_t *conn, nvrtcResult *status,
                                const char **errorLog) {
  bool present = false;
  std::string log;
  if (rpc_read(conn, status, sizeof(*status)) < 0 ||
      read_string(conn, &present, &log) < 0) {
    return rpc_error();
  }
  if (present && errorLog != nullptr) {
    bundled_headers_log = std::move(log);
    *errorLog = bundled_headers_log.c_str();
  }
  return NVRTC_SUCCESS;
}
#endif

} // namespace

#include "codegen/gen_nvrtc_client.inc"

namespace {

nvrtcResult create_on_server(program_state *state, nvrtcProgram *server) {
  const int32_t count = static_cast<int32_t>(state->headers.size());
  std::vector<const char *> headers, include_names;
  for (int32_t i = 0; i < count; ++i) {
    headers.push_back(state->headers[i].c_str());
    include_names.push_back(state->include_names[i].c_str());
  }
  std::vector<char> request;
  encode_string(&request, state->src.c_str());
  encode_string(&request, state->name ? state->name->c_str() : nullptr);
  encode_string_array(&request, count, count == 0 ? nullptr : headers.data());
  encode_string_array(&request, count,
                      count == 0 ? nullptr : include_names.data());
  nvrtcResult status = rpc_error();
  if (rpc_write_start_request(state->conn, RPC_nvrtcCreateProgram) < 0 ||
      rpc_write(state->conn, &count, sizeof(count)) < 0 ||
      rpc_write(state->conn, request.data(), request.size()) < 0 ||
      rpc_wait_for_response(state->conn) < 0 ||
      rpc_read(state->conn, server, sizeof(*server)) < 0 ||
      rpc_read(state->conn, &status, sizeof(status)) < 0 ||
      rpc_read_end(state->conn) < 0) {
    return rpc_error();
  }
  return status;
}

// ---------------------------------------------------------------------------
// Included files
// ---------------------------------------------------------------------------

bool read_file(const std::string &path, std::string *contents) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  contents->assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
  return !in.bad();
}

std::string directory_of(const std::string &path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

std::string join(const std::string &directory, const std::string &name) {
  if (directory.empty() || name.empty() || name[0] == '/' || name[0] == '\\' ||
      (name.size() > 1 && name[1] == ':')) {
    return name;
  }
  const char last = directory.back();
  return last == '/' || last == '\\' ? directory + name
                                     : directory + "/" + name;
}

// Every #include directive in the text, with whether it names its file in
// quotes. Conditional compilation is not evaluated: a file that is never used
// only costs its bytes.
std::vector<std::pair<std::string, bool>>
include_directives(const std::string &text) {
  std::vector<std::pair<std::string, bool>> names;
  size_t line = 0;
  while (line < text.size()) {
    size_t end = text.find('\n', line);
    if (end == std::string::npos) {
      end = text.size();
    }
    size_t i = text.find_first_not_of(" \t", line);
    if (i < end && text[i] == '#') {
      i = text.find_first_not_of(" \t", i + 1);
      if (i < end && text.compare(i, 7, "include") == 0) {
        i = text.find_first_not_of(" \t", i + 7);
        if (i < end && (text[i] == '"' || text[i] == '<')) {
          const char close = text[i] == '"' ? '"' : '>';
          const size_t stop = text.find(close, i + 1);
          if (stop < end) {
            names.emplace_back(text.substr(i + 1, stop - i - 1), close == '"');
          }
        }
      }
    }
    line = end + 1;
  }
  return names;
}

// The server compiles, so the files a program includes from the client's disk
// travel as headers named the way the directives name them. Quoted names are
// looked for beside the including file first, then every name in the include
// paths the options give, in order, as NVRTC looks for them. Names the
// program's own headers already provide, and names found nowhere, are left to
// the server's compiler.
std::vector<std::pair<std::string, std::string>>
included_files(const program_state &state, int numOptions,
               const char *const *options) {
  std::vector<std::string> include_paths;
  for (int i = 0; options != nullptr && i < numOptions; ++i) {
    const std::string option = options[i] == nullptr ? "" : options[i];
    if (option.compare(0, 2, "-I") == 0 && option.size() > 2) {
      include_paths.push_back(option.substr(2));
    } else if (option.compare(0, 15, "--include-path=") == 0) {
      include_paths.push_back(option.substr(15));
    }
  }
  std::set<std::string> known(state.include_names.begin(),
                              state.include_names.end());
  std::vector<std::pair<std::string, std::string>> files;
  std::deque<std::pair<std::string, std::string>> pending;
  pending.emplace_back(state.src, state.name ? directory_of(*state.name) : "");
  while (!pending.empty()) {
    const std::pair<std::string, std::string> source =
        std::move(pending.front());
    pending.pop_front();
    for (const auto &directive : include_directives(source.first)) {
      if (!known.insert(directive.first).second) {
        continue;
      }
      std::vector<std::string> candidates;
      if (directive.second) {
        candidates.push_back(join(source.second, directive.first));
      }
      for (const std::string &path : include_paths) {
        candidates.push_back(join(path, directive.first));
      }
      for (const std::string &candidate : candidates) {
        std::string contents;
        if (read_file(candidate, &contents)) {
          pending.emplace_back(contents, directory_of(candidate));
          files.emplace_back(directive.first, std::move(contents));
          break;
        }
      }
    }
  }
  return files;
}

// Headers can only be given when a program is created, so the server's program
// is created again with the included files and the name expressions added so
// far, and the old one is freed.
nvrtcResult
recreate_with(program_state *state,
              std::vector<std::pair<std::string, std::string>> files) {
  for (auto &file : files) {
    state->include_names.push_back(std::move(file.first));
    state->headers.push_back(std::move(file.second));
  }
  nvrtcProgram replacement = nullptr;
  nvrtcResult status = create_on_server(state, &replacement);
  if (status != NVRTC_SUCCESS) {
    return status;
  }
  for (const std::string &expression : state->name_expressions) {
    status = lupine_rpc_nvrtcAddNameExpression(state->conn, replacement,
                                               expression.c_str());
    if (status != NVRTC_SUCCESS) {
      lupine_rpc_nvrtcDestroyProgram(state->conn, &replacement);
      return status;
    }
  }
  nvrtcProgram previous = state->server;
  state->server = replacement;
  return lupine_rpc_nvrtcDestroyProgram(state->conn, &previous);
}

} // namespace

extern "C" const char *nvrtcGetErrorString(nvrtcResult result) {
  static std::mutex mutex;
  static std::unordered_map<int, std::string> texts;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = texts.find(result);
    if (it != texts.end()) {
      return it->second.c_str();
    }
  }
  conn_t *conn = connection();
  bool present = false;
  std::string text;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvrtcGetErrorString) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      read_string(conn, &present, &text) < 0 || rpc_read_end(conn) < 0 ||
      !present) {
    return "NVRTC_ERROR unknown";
  }
  std::lock_guard<std::mutex> lock(mutex);
  return texts.emplace(result, std::move(text)).first->second.c_str();
}

extern "C" nvrtcResult nvrtcGetSupportedArchs(int *supportedArchs) {
  conn_t *conn = connection();
  nvrtcResult status = rpc_error();
  int32_t count = 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvrtcGetSupportedArchs) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0 ||
      (count != 0 && rpc_read(conn, supportedArchs, count * sizeof(int)) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src,
                                          const char *name, int numHeaders,
                                          const char *const *headers,
                                          const char *const *includeNames) {
  auto state = std::make_unique<program_state>();
  state->conn = connection();
  state->src = src == nullptr ? "" : src;
  if (name != nullptr) {
    state->name = std::make_unique<std::string>(name);
  }
  for (int i = 0; i < numHeaders; ++i) {
    state->headers.push_back(headers[i] == nullptr ? "" : headers[i]);
    state->include_names.push_back(
        includeNames[i] == nullptr ? "" : includeNames[i]);
  }
  if (state->conn == nullptr) {
    return rpc_error();
  }
  const nvrtcResult status = create_on_server(state.get(), &state->server);
  if (status == NVRTC_SUCCESS) {
    *prog = static_cast<nvrtcProgram>(static_cast<void *>(state.get()));
    std::lock_guard<std::mutex> lock(programs_mutex);
    programs.emplace(*prog, std::move(state));
  }
  return status;
}

extern "C" nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog) {
  program_state *state = state_of(*prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  nvrtcProgram server = state->server;
  const nvrtcResult status =
      lupine_rpc_nvrtcDestroyProgram(state->conn, &server);
  if (status == NVRTC_SUCCESS) {
    std::lock_guard<std::mutex> lock(programs_mutex);
    programs.erase(*prog);
    *prog = nullptr;
  }
  return status;
}

extern "C" nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                           const char *const *options) {
  program_state *state = state_of(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  auto files = included_files(*state, numOptions, options);
  if (!files.empty()) {
    const nvrtcResult recreated = recreate_with(state, std::move(files));
    if (recreated != NVRTC_SUCCESS) {
      return recreated;
    }
  }
  nvrtcResult status = rpc_error();
  const int32_t count = numOptions;
  std::vector<char> request;
  encode_string_array(&request, numOptions, options);
  if (rpc_write_start_request(state->conn, RPC_nvrtcCompileProgram) < 0 ||
      rpc_write(state->conn, &state->server, sizeof(state->server)) < 0 ||
      rpc_write(state->conn, &count, sizeof(count)) < 0 ||
      rpc_write(state->conn, request.data(), request.size()) < 0 ||
      rpc_wait_for_response(state->conn) < 0 ||
      rpc_read(state->conn, &status, sizeof(status)) < 0 ||
      rpc_read_end(state->conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx) {
  return copy_output(RPC_nvrtcGetPTX, prog, ptx);
}

extern "C" nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin) {
  return copy_output(RPC_nvrtcGetCUBIN, prog, cubin);
}

extern "C" nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log) {
  return copy_output(RPC_nvrtcGetProgramLog, prog, log);
}

#if CUDA_VERSION >= 12000
extern "C" nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *LTOIR) {
  return copy_output(RPC_nvrtcGetLTOIR, prog, LTOIR);
}

extern "C" nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char *optixir) {
  return copy_output(RPC_nvrtcGetOptiXIR, prog, optixir);
}
#endif

#if CUDA_VERSION < 13000
extern "C" nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm) {
  return copy_output(RPC_nvrtcGetNVVM, prog, nvvm);
}
#endif

#if CUDA_VERSION >= 13020
extern "C" nvrtcResult nvrtcGetTileIR(nvrtcProgram prog, char *TileIR) {
  return copy_output(RPC_nvrtcGetTileIR, prog, TileIR);
}
#endif

extern "C" nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                           const char *const name_expression,
                                           const char **lowered_name) {
  program_state *state = state_of(prog);
  if (state == nullptr) {
    return NVRTC_ERROR_INVALID_PROGRAM;
  }
  {
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto name = name_expression == nullptr
                    ? state->lowered_names.end()
                    : state->lowered_names.find(name_expression);
    if (name != state->lowered_names.end()) {
      *lowered_name = name->second.c_str();
      return NVRTC_SUCCESS;
    }
  }
  nvrtcResult status = rpc_error();
  bool present = false;
  std::string lowered;
  std::vector<char> request;
  encode_string(&request, name_expression);
  if (rpc_write_start_request(state->conn, RPC_nvrtcGetLoweredName) < 0 ||
      rpc_write(state->conn, &state->server, sizeof(state->server)) < 0 ||
      rpc_write(state->conn, request.data(), request.size()) < 0 ||
      rpc_wait_for_response(state->conn) < 0 ||
      rpc_read(state->conn, &status, sizeof(status)) < 0 ||
      read_string(state->conn, &present, &lowered) < 0 ||
      rpc_read_end(state->conn) < 0) {
    return rpc_error();
  }
  if (status != NVRTC_SUCCESS || !present) {
    return status;
  }
  std::lock_guard<std::mutex> lock(programs_mutex);
  *lowered_name =
      state->lowered_names.emplace(name_expression, std::move(lowered))
          .first->second.c_str();
  return status;
}

#if CUDA_VERSION >= 13030
extern "C" nvrtcResult nvrtcInstallBundledHeaders(const char *installPath,
                                                  unsigned int flags,
                                                  const char **errorLog) {
  conn_t *conn = connection();
  nvrtcResult status = rpc_error();
  const uint32_t wire_flags = flags;
  const uint8_t wants_log = errorLog != nullptr;
  std::vector<char> request;
  encode_string(&request, installPath);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvrtcInstallBundledHeaders) < 0 ||
      rpc_write(conn, request.data(), request.size()) < 0 ||
      rpc_write(conn, &wire_flags, sizeof(wire_flags)) < 0 ||
      rpc_write(conn, &wants_log, sizeof(wants_log)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      finish_bundled_call(conn, &status, errorLog) != NVRTC_SUCCESS ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" nvrtcResult nvrtcGetBundledHeadersInfo(nvrtcBundledHeadersInfo *info,
                                                  const char **errorLog) {
  conn_t *conn = connection();
  nvrtcResult status = rpc_error();
  const uint8_t wants_info = info != nullptr;
  const uint8_t wants_log = errorLog != nullptr;
  nvrtcBundledHeadersInfo received{};
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvrtcGetBundledHeadersInfo) < 0 ||
      rpc_write(conn, &wants_info, sizeof(wants_info)) < 0 ||
      rpc_write(conn, &wants_log, sizeof(wants_log)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      finish_bundled_call(conn, &status, errorLog) != NVRTC_SUCCESS ||
      rpc_read(conn, &received, sizeof(received)) < 0 ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (status == NVRTC_SUCCESS) {
    *info = received;
  }
  return status;
}

extern "C" nvrtcResult nvrtcRemoveBundledHeaders(const char *installPath,
                                                 const char **errorLog) {
  conn_t *conn = connection();
  nvrtcResult status = rpc_error();
  const uint8_t wants_log = errorLog != nullptr;
  std::vector<char> request;
  encode_string(&request, installPath);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_nvrtcRemoveBundledHeaders) < 0 ||
      rpc_write(conn, request.data(), request.size()) < 0 ||
      rpc_write(conn, &wants_log, sizeof(wants_log)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      finish_bundled_call(conn, &status, errorLog) != NVRTC_SUCCESS ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}
#endif
