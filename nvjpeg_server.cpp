#include <nvjpeg.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

// nvJPEG documents IMPLEMENTATION_NOT_SUPPORTED for a feature the library does
// not provide, which includes a server with no libnvjpeg at all.
nvjpegStatus_t function_not_found() {
  return NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
}

// The library this server was compiled against; the handlers marshal its
// structures, so another major on the machine is not a match.
void *nvjpeg_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(NVJPEG_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "nvjpeg64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libnvjpeg.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn nvjpeg_symbol(const char *name) {
  void *lib = nvjpeg_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// ---------------------------------------------------------------------------
// Retained bitstreams
// ---------------------------------------------------------------------------

// A bitstream object parsed without saving its stream decodes from the
// caller's bytes later, and a decode state keeps the tables it parsed the same
// way, so the server keeps the bytes each parse was given until the next parse
// of that kind replaces them or the object is destroyed. The new bytes are
// stored only after the library has switched to them.
enum retained_kind : int { stream_bytes, stream_tables, batched_tables };

std::mutex retained_mutex;
std::map<std::pair<void *, int>, std::vector<unsigned char>> retained;

void retain(void *object, retained_kind kind,
            std::vector<unsigned char> bytes) {
  std::lock_guard<std::mutex> lock(retained_mutex);
  retained[{object, kind}] = std::move(bytes);
}

void release(void *object) {
  std::lock_guard<std::mutex> lock(retained_mutex);
  retained.erase(retained.lower_bound({object, stream_bytes}),
                 retained.upper_bound({object, batched_tables}));
}

int read_bitstream(conn_t *conn, std::vector<unsigned char> *bytes) {
  size_t length = 0;
  if (rpc_read(conn, &length, sizeof(length)) < 0) {
    return -1;
  }
  bytes->resize(length);
  return length == 0 ? 0 : rpc_read(conn, bytes->data(), length);
}

int write_status(conn_t *conn, int request_id, nvjpegStatus_t status) {
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &status, sizeof(status)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

int handle_stream_parse(conn_t *conn, const char *symbol, retained_kind kind) {
  nvjpegHandle_t handle;
  std::vector<unsigned char> bytes;
  nvjpegJpegStream_t jpeg_stream;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      read_bitstream(conn, &bytes) < 0 ||
      rpc_read(conn, &jpeg_stream, sizeof(jpeg_stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char *, size_t,
                                  nvjpegJpegStream_t);
  fn_t fn = nvjpeg_symbol<fn_t>(symbol);
  const nvjpegStatus_t status =
      fn == nullptr ? function_not_found()
                    : fn(handle, bytes.data(), bytes.size(), jpeg_stream);
  retain(jpeg_stream, kind, std::move(bytes));
  return write_status(conn, request_id, status);
}

int handle_destroy(conn_t *conn, const char *symbol) {
  void *object;
  if (rpc_read(conn, &object, sizeof(object)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvjpegStatus_t (*)(void *);
  fn_t fn = nvjpeg_symbol<fn_t>(symbol);
  const nvjpegStatus_t status =
      fn == nullptr ? function_not_found() : fn(object);
  release(object);
  return write_status(conn, request_id, status);
}

// ---------------------------------------------------------------------------
// Batched decoding
// ---------------------------------------------------------------------------

int handle_decode_batched(conn_t *conn, bool with_params) {
  nvjpegHandle_t handle;
  nvjpegJpegState_t jpeg_handle;
  uint64_t count = 0;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &jpeg_handle, sizeof(jpeg_handle)) < 0 ||
      rpc_read(conn, &count, sizeof(count)) < 0) {
    return -1;
  }
  std::vector<size_t> lengths(count);
  if (count != 0 &&
      rpc_read(conn, lengths.data(), count * sizeof(size_t)) < 0) {
    return -1;
  }
  std::vector<std::vector<unsigned char>> bitstreams(count);
  std::vector<const unsigned char *> data(count);
  for (uint64_t i = 0; i < count; ++i) {
    bitstreams[i].resize(lengths[i]);
    if (lengths[i] != 0 &&
        rpc_read(conn, bitstreams[i].data(), lengths[i]) < 0) {
      return -1;
    }
    data[i] = bitstreams[i].data();
  }
  std::vector<nvjpegImage_t> destinations(count);
  std::vector<nvjpegDecodeParams_t> decode_params(with_params ? count : 0);
  cudaStream_t stream;
  if ((count != 0 && rpc_read(conn, destinations.data(),
                              count * sizeof(nvjpegImage_t)) < 0) ||
      (with_params && count != 0 &&
       rpc_read(conn, decode_params.data(),
                count * sizeof(nvjpegDecodeParams_t)) < 0) ||
      rpc_read(conn, &stream, sizeof(stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  nvjpegStatus_t status = function_not_found();
  if (with_params) {
    using fn_t = nvjpegStatus_t (*)(
        nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *const *,
        const size_t *, nvjpegImage_t *, nvjpegDecodeParams_t *, cudaStream_t);
    if (fn_t fn = nvjpeg_symbol<fn_t>("nvjpegDecodeBatchedEx")) {
      status = fn(handle, jpeg_handle, data.data(), lengths.data(),
                  destinations.data(), decode_params.data(), stream);
    }
  } else {
    using fn_t = nvjpegStatus_t (*)(
        nvjpegHandle_t, nvjpegJpegState_t, const unsigned char *const *,
        const size_t *, nvjpegImage_t *, cudaStream_t);
    if (fn_t fn = nvjpeg_symbol<fn_t>("nvjpegDecodeBatched")) {
      status = fn(handle, jpeg_handle, data.data(), lengths.data(),
                  destinations.data(), stream);
    }
  }
  return write_status(conn, request_id, status);
}

} // namespace

int handle_nvjpegJpegStreamParse(conn_t *conn) {
  nvjpegHandle_t handle;
  std::vector<unsigned char> bytes;
  int save_metadata = 0;
  int save_stream = 0;
  nvjpegJpegStream_t jpeg_stream;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      read_bitstream(conn, &bytes) < 0 ||
      rpc_read(conn, &save_metadata, sizeof(save_metadata)) < 0 ||
      rpc_read(conn, &save_stream, sizeof(save_stream)) < 0 ||
      rpc_read(conn, &jpeg_stream, sizeof(jpeg_stream)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvjpegStatus_t (*)(nvjpegHandle_t, const unsigned char *, size_t,
                                  int, int, nvjpegJpegStream_t);
  fn_t fn = nvjpeg_symbol<fn_t>("nvjpegJpegStreamParse");
  const nvjpegStatus_t status =
      fn == nullptr ? function_not_found()
                    : fn(handle, bytes.data(), bytes.size(), save_metadata,
                         save_stream, jpeg_stream);
  retain(jpeg_stream, stream_bytes, std::move(bytes));
  return write_status(conn, request_id, status);
}

int handle_nvjpegJpegStreamParseHeader(conn_t *conn) {
  return handle_stream_parse(conn, "nvjpegJpegStreamParseHeader", stream_bytes);
}

int handle_nvjpegJpegStreamParseTables(conn_t *conn) {
  return handle_stream_parse(conn, "nvjpegJpegStreamParseTables",
                             stream_tables);
}

int handle_nvjpegDecodeBatchedParseJpegTables(conn_t *conn) {
  nvjpegHandle_t handle;
  nvjpegJpegState_t jpeg_handle;
  std::vector<unsigned char> bytes;
  if (rpc_read(conn, &handle, sizeof(handle)) < 0 ||
      rpc_read(conn, &jpeg_handle, sizeof(jpeg_handle)) < 0 ||
      read_bitstream(conn, &bytes) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = nvjpegStatus_t (*)(nvjpegHandle_t, nvjpegJpegState_t,
                                  const unsigned char *, const size_t);
  fn_t fn = nvjpeg_symbol<fn_t>("nvjpegDecodeBatchedParseJpegTables");
  const nvjpegStatus_t status =
      fn == nullptr ? function_not_found()
                    : fn(handle, jpeg_handle, bytes.data(), bytes.size());
  retain(jpeg_handle, batched_tables, std::move(bytes));
  return write_status(conn, request_id, status);
}

int handle_nvjpegJpegStreamDestroy(conn_t *conn) {
  return handle_destroy(conn, "nvjpegJpegStreamDestroy");
}

int handle_nvjpegJpegStateDestroy(conn_t *conn) {
  return handle_destroy(conn, "nvjpegJpegStateDestroy");
}

int handle_nvjpegDecodeBatched(conn_t *conn) {
  return handle_decode_batched(conn, false);
}

int handle_nvjpegDecodeBatchedEx(conn_t *conn) {
  return handle_decode_batched(conn, true);
}

#include "codegen/gen_nvjpeg_server.inc"
