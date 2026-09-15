// Exercise the public shims, server handlers, and log packet decoder with an
// in-memory transport. The native-library fixture supplies deterministic logs
// without CUDA initialization or a GPU. Transport scheduling itself is covered
// by h2_test; here each callback must wait for the caller's response to finish.
#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"
#include "library_logging.h"
#include "pending_log_callbacks.h"
#include "rpc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <deque>
#include <string>
#include <thread>
#include <vector>

#define HANDLER(name) int handle_##name(conn_t *);
HANDLER(cublasSetLoggerCallback)
HANDLER(cublasGetLoggerCallback)
HANDLER(cublasGetProperty)
HANDLER(cublasLoggerConfigure)
HANDLER(cublasLtLoggerSetCallback)
HANDLER(cublasLtLoggerSetFile)
HANDLER(cublasLtLoggerOpenFile)
HANDLER(cublasLtLoggerSetLevel)
HANDLER(cublasLtLoggerSetMask)
HANDLER(cublasLtLoggerForceDisable)
HANDLER(cublasLtHeuristicsCacheSetCapacity)
HANDLER(cublasLtMatrixLayoutInit_internal)
#undef HANDLER

namespace {
void require(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}
void success(cublasStatus_t status) {
  require(status == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed");
}
conn_t client{}, server{};
pending_log_callbacks pending;
struct packet {
  int op;
  std::vector<char> input, output;
  size_t offset = 0;
  bool response = false;
};
thread_local std::deque<packet> packets;
bool fail_response = false;

int dispatch(int op) {
  switch (op) {
#define HANDLER(name)                                                          \
  case RPC_##name:                                                             \
    return handle_##name(&server);
    HANDLER(cublasSetLoggerCallback)
    HANDLER(cublasGetLoggerCallback)
    HANDLER(cublasGetProperty)
    HANDLER(cublasLoggerConfigure)
    HANDLER(cublasLtLoggerSetCallback)
    HANDLER(cublasLtLoggerSetFile)
    HANDLER(cublasLtLoggerOpenFile)
    HANDLER(cublasLtLoggerSetLevel)
    HANDLER(cublasLtLoggerSetMask)
    HANDLER(cublasLtLoggerForceDisable)
    HANDLER(cublasLtHeuristicsCacheSetCapacity)
    HANDLER(cublasLtMatrixLayoutInit_internal)
#undef HANDLER
  default:
    require(false, "unexpected request");
    return -1;
  }
}

std::thread::id caller;
int cublas_calls = 0, replacement_calls = 0, lt_calls = 0;
std::string last_function, last_message;
FILE *replace_file_in_callback = nullptr;
void check_callback_thread() {
  require(std::this_thread::get_id() == caller, "callback changed threads");
  require(packets.empty(), "callback ran before the response was consumed");
}
void cublas_callback(const char *message) {
  check_callback_thread();
  ++cublas_calls;
  last_message = message;
  // This nested RPC would deadlock if the callback still held the caller lane.
  cublasLogCallback installed = nullptr;
  success(cublasGetLoggerCallback(&installed));
}
void replacement_callback(const char *) {
  check_callback_thread();
  ++replacement_calls;
}
void lt_callback(int level, const char *function, const char *message) {
  check_callback_thread();
  require(level == 5, "log level changed");
  ++lt_calls;
  last_function = function;
  last_message = message;
  if (replace_file_in_callback != nullptr) {
    FILE *old = replace_file_in_callback;
    replace_file_in_callback = nullptr;
    require(std::ftell(old) > 0, "file bytes must precede the user callback");
    success(cublasLtLoggerSetFile(nullptr));
    std::fclose(old);
  }
}
void emit_lt() { success(cublasLtHeuristicsCacheSetCapacity(16)); }
std::string file_text(FILE *file) {
  std::fflush(file);
  std::rewind(file);
  std::string text;
  char buffer[1024];
  size_t bytes;
  while ((bytes = std::fread(buffer, 1, sizeof(buffer), file)) != 0) {
    text.append(buffer, bytes);
  }
  return text;
}
} // namespace

// Only these transport/routing primitives are replaced. All serialization and
// native symbol lookup run through the same code as the shipped libraries.
extern "C" cudaError_t cudaGetDevice(int *device) {
  *device = 0;
  return cudaSuccess;
}
extern "C" conn_t *lupine_rpc_conn_for_device(int *) { return &client; }
extern "C" conn_t *lupine_rpc_conn_for_stream(CUstream_st *) { return &client; }
int rpc_write_start_request(conn_t *, int op) {
  packets.push_back({op, {}, {}, 0, false});
  return 0;
}
int rpc_write(conn_t *, const void *data, size_t size) {
  if (size != 0) {
    const char *bytes = static_cast<const char *>(data);
    packets.back().output.insert(packets.back().output.end(), bytes,
                                 bytes + size);
  }
  return 0;
}
int rpc_read(conn_t *, void *data, size_t size) {
  auto &packet = packets.back();
  if (size > packet.input.size() - packet.offset) {
    return -1;
  }
  if (size != 0) {
    std::memcpy(data, packet.input.data() + packet.offset, size);
  }
  packet.offset += size;
  return 0;
}
int rpc_read_end(conn_t *) {
  auto &packet = packets.back();
  require(packet.offset == packet.input.size(), "unconsumed packet bytes");
  if (packet.response) {
    packets.pop_back();
  }
  return 1;
}
int rpc_write_start_response(conn_t *, int) {
  packets.back().output.clear();
  return 0;
}
int rpc_write_end(conn_t *) { return 0; }
int32_t rpc_current_http2_stream(conn_t *conn) {
  return conn == &client ? 2 : 1;
}
int rpc_wait_for_response(conn_t *) {
  auto &packet = packets.back();
  packet.input = std::move(packet.output);
  packet.offset = 0;
  int result = packet.op == LUPINE_SIDE_EFFECT_LIBRARY_LOG
                   ? lupine_read_library_log(&client, pending)
                   : dispatch(packet.op);
  packet.input = std::move(packet.output);
  packet.offset = 0;
  packet.response = true;
  return result;
}
extern "C" int lupine_rpc_write_start_request(conn_t *conn, int op) {
  return rpc_write_start_request(conn, op);
}
extern "C" int lupine_rpc_write(conn_t *conn, const void *data, size_t size) {
  return rpc_write(conn, data, size);
}
extern "C" int lupine_rpc_wait_for_response(conn_t *conn) {
  int result = rpc_wait_for_response(conn);
  if (fail_response) {
    packets.clear();
    pending.discard(conn);
    return -1;
  }
  return result;
}
extern "C" int lupine_rpc_read(conn_t *conn, void *data, size_t size) {
  return rpc_read(conn, data, size);
}
extern "C" int lupine_rpc_read_end(conn_t *conn) {
  int result = rpc_read_end(conn);
  pending.complete(conn, 1);
  return result;
}

int main(int argc, char **argv) {
  caller = std::this_thread::get_id();
  if (argc == 2 && std::string(argv[1]) == "--native") {
    success(cublasSetLoggerCallback(cublas_callback));
    success(cublasLoggerConfigure(1, 0, 0, nullptr));
    int version = 0;
    success(cublasGetProperty(MAJOR_VERSION, &version));
    require(cublas_calls > 0, "native cuBLAS emitted no callbacks");
    success(cublasSetLoggerCallback(nullptr));
    success(cublasLtLoggerSetLevel(5));
    success(cublasLtLoggerSetCallback(lt_callback));
    FILE *file = std::tmpfile();
    require(file != nullptr, "tmpfile failed");
    success(cublasLtLoggerSetFile(file));
    cublasLtMatrixLayout_t layout = nullptr;
    success(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, 2, 2, 2));
    require(lt_calls > 0 &&
                last_function.find("MatrixLayout") != std::string::npos,
            "native cuBLASLt metadata missing");
    require(file_text(file).find("[cublasLt]") != std::string::npos &&
                file_text(file).find("MatrixLayout") != std::string::npos,
            "native cuBLASLt file output missing");
    success(cublasLtMatrixLayoutDestroy(layout));
    success(cublasLtLoggerSetFile(nullptr));
    success(cublasLtLoggerSetCallback(nullptr));
    std::fclose(file);
    lupine_cublas_cleanup_logs(&server);
    std::puts("native cuBLAS/cuBLASLt logging tests passed (no GPU required)");
    return 0;
  }
  cublasLogCallback callback = cublas_callback;
  require(cublasGetLoggerCallback(nullptr) == CUBLAS_STATUS_INVALID_VALUE,
          "null callback output must be rejected");
  success(cublasGetLoggerCallback(&callback));
  require(callback == nullptr, "initial callback must be null");
  success(cublasSetLoggerCallback(cublas_callback));
  success(cublasGetLoggerCallback(&callback));
  require(callback == cublas_callback,
          "getter returned a server function pointer");
  setenv("LUPINE_TEST_LOGGER_FAIL", "1", 1);
  require(cublasSetLoggerCallback(replacement_callback) ==
              CUBLAS_STATUS_EXECUTION_FAILED,
          "failed callback setter status changed");
  unsetenv("LUPINE_TEST_LOGGER_FAIL");
  success(cublasGetLoggerCallback(&callback));
  require(callback == cublas_callback, "failed setter replaced the callback");
  cublas_calls = 0;
  int version = 0;
  success(cublasGetProperty(MAJOR_VERSION, &version));
  require(cublas_calls == 1 &&
              last_message == "cublasGetProperty: fixture message",
          "cuBLAS callback message was lost");
  success(cublasSetLoggerCallback(replacement_callback));
  require(cublas_calls == 2,
          "replacement lost the old callback's final message");
  success(cublasGetProperty(MAJOR_VERSION, &version));
  require(replacement_calls == 1, "replacement callback did not run");
  success(cublasSetLoggerCallback(nullptr));
  success(cublasGetLoggerCallback(&callback));
  require(callback == nullptr, "removed callback is still installed");
  int removed_count = replacement_calls;
  success(cublasGetProperty(MAJOR_VERSION, &version));
  require(replacement_calls == removed_count, "callback ran after removal");

  success(cublasLtLoggerSetCallback(lt_callback));
  emit_lt();
  require(lt_calls == 1 &&
              last_function == "cublasLtHeuristicsCacheSetCapacity" &&
              last_message == "fixture message",
          "cuBLASLt metadata changed");
  FILE *first = std::tmpfile(), *second = std::tmpfile();
  require(first && second, "tmpfile failed");
  success(cublasLtLoggerSetFile(first));
  emit_lt();
  require(
      file_text(first) ==
          "[native][cublasLt][Api][cublasLtHeuristicsCacheSetCapacity] fixture "
          "message\n",
      "native file formatting changed or file was not flushed before return");
  setenv("LUPINE_TEST_LOGGER_FAIL", "1", 1);
  require(cublasLtLoggerSetFile(second) == CUBLAS_STATUS_EXECUTION_FAILED,
          "failed FILE setter status changed");
  require(cublasLtLoggerSetCallback(nullptr) == CUBLAS_STATUS_EXECUTION_FAILED,
          "failed Lt callback setter status changed");
  unsetenv("LUPINE_TEST_LOGGER_FAIL");
  auto old_size = file_text(first).size();
  int old_callback_count = lt_calls;
  emit_lt();
  require(file_text(first).size() > old_size && file_text(second).empty() &&
              lt_calls > old_callback_count,
          "failed setter changed the active file or callback");
  success(cublasLtLoggerSetFile(second));
  std::string old_text = file_text(first);
  require(old_text.find("cublasLtLoggerSetFile") != std::string::npos,
          "file replacement lost its final log");
  flockfile(first);
  emit_lt();
  funlockfile(first);
  require(file_text(first) == old_text && !file_text(second).empty(),
          "file replacement wrote to the wrong sink");
  require(cublasLtLoggerOpenFile("/missing/log") == CUBLAS_STATUS_INVALID_VALUE,
          "failed OpenFile status changed");
  auto before = file_text(second).size();
  emit_lt();
  require(file_text(second).size() > before,
          "failed OpenFile removed the old sink");
  success(cublasLtLoggerSetCallback(nullptr));
  int removed_lt_count = lt_calls;
  before = file_text(second).size();
  emit_lt();
  require(lt_calls == removed_lt_count && file_text(second).size() > before,
          "file and callback registrations are not independent");
  success(cublasLtLoggerOpenFile("new-server-file"));
  before = file_text(second).size();
  emit_lt();
  require(file_text(second).size() == before,
          "OpenFile kept the old FILE sink");
  success(cublasLtLoggerSetFile(second));
  success(cublasLtLoggerSetMask(0));
  before = file_text(second).size();
  emit_lt();
  require(file_text(second).size() == before, "mask was ignored");
  success(cublasLtLoggerSetLevel(5));
  emit_lt();
  require(file_text(second).size() > before, "level did not re-enable logging");
  success(cublasLtLoggerSetCallback(lt_callback));
  FILE *reentrant = std::tmpfile();
  require(reentrant != nullptr, "tmpfile failed");
  success(cublasLtLoggerSetFile(reentrant));
  replace_file_in_callback = reentrant;
  emit_lt();
  require(replace_file_in_callback == nullptr,
          "file replacement callback did not run");
  success(cublasLtLoggerSetCallback(nullptr));
  success(cublasLtLoggerSetFile(second));
  success(cublasLtLoggerForceDisable());
  before = file_text(second).size();
  emit_lt();
  require(file_text(second).size() == before, "ForceDisable was ignored");
  success(cublasLtLoggerSetFile(nullptr));
  std::fclose(first);
  std::fclose(second);

  success(cublasSetLoggerCallback(cublas_callback));
  int previous = cublas_calls;
  fail_response = true;
  require(cublasGetProperty(MAJOR_VERSION, &version) ==
              CUBLAS_STATUS_NOT_INITIALIZED,
          "transport failure must propagate");
  fail_response = false;
  pending.complete(&client, 1);
  require(cublas_calls == previous,
          "failed connection delivered a stale callback");
  lupine_cublas_cleanup_logs(&server);
  success(cublasGetProperty(MAJOR_VERSION, &version));
  require(cublas_calls == previous, "server cleanup retained a callback route");

  // Independent lanes/connections and callback re-entry must not interfere.
  int completions = 0;
  pending.enqueue(&client, 3, [&] {
    ++completions;
    pending.enqueue(&client, 3, [&] { ++completions; });
  });
  pending.enqueue(&server, 3, [&] { completions += 100; });
  pending.complete(&client, 2);
  require(completions == 0, "wrong lane drained callbacks");
  std::thread worker([&] { pending.complete(&client, 3); });
  worker.join();
  require(completions == 1, "callback re-entry failed");
  pending.complete(&client, 3);
  require(completions == 2, "nested callback was lost");
  pending.discard(&server);
  pending.complete(&server, 3);
  require(completions == 2, "connection discard failed");
  std::puts("cuBLAS logging forwarding tests passed");
}
