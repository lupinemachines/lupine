// Check the libraries the executable actually linked, not a separately opened
// filename. In particular CUDA 11 links libcudart.so.11.0, not libcudart.so.11.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/stat.h>

static bool check_library(const char *symbol, const void *address,
                          const char *expected) {
  Dl_info info = {};
  struct stat actual_stat = {}, expected_stat = {};
  if (expected == nullptr || dladdr(address, &info) == 0 ||
      stat(info.dli_fname, &actual_stat) != 0 ||
      stat(expected, &expected_stat) != 0 ||
      actual_stat.st_dev != expected_stat.st_dev ||
      actual_stat.st_ino != expected_stat.st_ino) {
    std::fprintf(stderr, "%s loaded from %s; expected %s\n", symbol,
                 info.dli_fname ? info.dli_fname : "<unresolved>",
                 expected ? expected : "<unset>");
    return false;
  }
  std::printf("%s: %s\n", symbol, info.dli_fname);
  return true;
}

int main() {
  bool driver = check_library("cuInit", reinterpret_cast<const void *>(&cuInit),
                              std::getenv("LUPINE_LIB"));
  bool runtime = check_library(
      "cudaGetDeviceCount", reinterpret_cast<const void *>(&cudaGetDeviceCount),
      std::getenv("LUPINE_EXPECTED_CUDART"));
  return driver && runtime ? 0 : 1;
}
