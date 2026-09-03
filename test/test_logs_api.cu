#include <atomic>
#include <cstdio>
#include <cstring>
#include <cuda.h>

#if CUDA_VERSION >= 12090

struct CallbackState {
  std::atomic<int> count{0};
  std::atomic<int> malformed{0};
};

static void CUDA_CB log_callback(void *user_data, CUlogLevel level,
                                 char *message, size_t length) {
  auto *state = static_cast<CallbackState *>(user_data);
  if (state == nullptr || level != CU_LOG_LEVEL_ERROR || message == nullptr ||
      length == 0) {
    if (state != nullptr) {
      state->malformed.store(1, std::memory_order_relaxed);
    }
    return;
  }
  state->count.fetch_add(1, std::memory_order_relaxed);
}

static bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "FAIL: %s: %s (%d)\n", operation,
               name == nullptr ? "unknown" : name, static_cast<int>(result));
  return false;
}

static bool generate_error_log() {
  CUlogIterator ignored = 0;
  size_t size = 1;
  CUresult result = cuLogsDumpToMemory(&ignored, nullptr, &size, 0);
  if (result == CUDA_ERROR_INVALID_VALUE) {
    return true;
  }
  std::fprintf(stderr, "FAIL: cuLogsDumpToMemory(nullptr) returned %d\n",
               static_cast<int>(result));
  return false;
}

int main() {
  if (!check(cuInit(0), "cuInit")) {
    return 1;
  }

  CallbackState state;
  CUlogsCallbackHandle callback = nullptr;
  if (!check(cuLogsRegisterCallback(log_callback, &state, &callback),
             "cuLogsRegisterCallback") ||
      callback == nullptr) {
    return 1;
  }

  if (cuLogsCurrent(nullptr, 0) != CUDA_ERROR_INVALID_VALUE ||
      cuLogsDumpToFile(nullptr, nullptr, 0) != CUDA_ERROR_INVALID_VALUE) {
    std::fprintf(stderr, "FAIL: log APIs accepted invalid output pointers\n");
    return 1;
  }

  CUlogIterator memory_iterator = 0;
  if (!check(cuLogsCurrent(&memory_iterator, 0), "cuLogsCurrent") ||
      !generate_error_log()) {
    return 1;
  }

  char logs[25600];
  std::memset(logs, 0x7f, sizeof(logs));
  size_t size = sizeof(logs);
  if (!check(cuLogsDumpToMemory(&memory_iterator, logs, &size, 0),
             "cuLogsDumpToMemory") ||
      size == 0 || size >= sizeof(logs) || logs[size] != '\0') {
    std::fprintf(stderr,
                 "FAIL: invalid memory dump size/termination: %zu bytes\n",
                 size);
    return 1;
  }

  CUlogIterator file_iterator = 0;
  if (!check(cuLogsCurrent(&file_iterator, 0), "cuLogsCurrent(file)") ||
      !generate_error_log() ||
      !check(cuLogsDumpToFile(&file_iterator, "/dev/null", 0),
             "cuLogsDumpToFile")) {
    return 1;
  }

  int callbacks_before_unregister = state.count.load(std::memory_order_relaxed);
  if (callbacks_before_unregister < 2 ||
      state.malformed.load(std::memory_order_relaxed) != 0) {
    std::fprintf(stderr, "FAIL: callback count=%d malformed=%d\n",
                 callbacks_before_unregister,
                 state.malformed.load(std::memory_order_relaxed));
    return 1;
  }

  if (!check(cuLogsUnregisterCallback(callback), "cuLogsUnregisterCallback") ||
      !generate_error_log()) {
    return 1;
  }
  if (state.count.load(std::memory_order_relaxed) !=
      callbacks_before_unregister) {
    std::fprintf(stderr, "FAIL: callback fired after unregister\n");
    return 1;
  }

  std::printf("PASS: CUDA log APIs and callbacks are forwarded\n");
  return 0;
}

#else

int main() {
  std::printf("SKIP: CUDA log APIs require CUDA 12.9 or newer\n");
  return 0;
}

#endif
