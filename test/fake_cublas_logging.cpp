// A device-free library fixture. Deliberately logs from inside setters too,
// so replacement tests exercise the old callback/file's last message.
#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>
#include <string>

namespace {
#ifdef TEST_CUBLASLT
cublasLtLoggerCallback_t callback;
FILE *file;
int level = 5;
void emit(const char *name) {
  if (level == 0) {
    return;
  }
  if (file != nullptr) {
    std::fprintf(file, "[native][cublasLt][Api][%s] fixture message\n", name);
  }
  if (callback != nullptr) {
    callback(5, name, "fixture message");
  }
}
#else
cublasLogCallback callback;
#endif
} // namespace

#ifdef TEST_CUBLASLT
extern "C" cublasStatus_t
cublasLtLoggerSetCallback(cublasLtLoggerCallback_t next) {
  emit("cublasLtLoggerSetCallback");
  if (std::getenv("LUPINE_TEST_LOGGER_FAIL")) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  callback = next;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLtLoggerSetFile(FILE *next) {
  emit("cublasLtLoggerSetFile");
  if (std::getenv("LUPINE_TEST_LOGGER_FAIL")) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  file = next;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLtLoggerOpenFile(const char *name) {
  emit("cublasLtLoggerOpenFile");
  if (name == nullptr || std::string(name) == "/missing/log") {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  file = nullptr;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLtLoggerSetLevel(int next) {
  level = next;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLtLoggerSetMask(int mask) {
  level = mask == 0 ? 0 : 5;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLtLoggerForceDisable() {
  level = 0;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLtHeuristicsCacheSetCapacity(size_t) {
  emit("cublasLtHeuristicsCacheSetCapacity");
  return CUBLAS_STATUS_SUCCESS;
}
#else
extern "C" cublasStatus_t cublasSetLoggerCallback(cublasLogCallback next) {
  if (callback != nullptr) {
    callback("cublasSetLoggerCallback: old callback");
  }
  if (std::getenv("LUPINE_TEST_LOGGER_FAIL")) {
    return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  callback = next;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *out) {
  *out = callback;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasGetProperty(libraryPropertyType, int *value) {
  if (callback != nullptr) {
    callback("cublasGetProperty: fixture message");
  }
  *value = 13;
  return CUBLAS_STATUS_SUCCESS;
}
extern "C" cublasStatus_t cublasLoggerConfigure(int, int, int, const char *) {
  return CUBLAS_STATUS_SUCCESS;
}
#endif
