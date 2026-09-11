#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  const char *variant = std::getenv("LUPINE_TEST_VARIANT");
  if (variant != nullptr && std::strcmp(variant, "driver-runtime") == 0) {
    std::printf("SKIP: the runtime shim opens the configured backend before "
                "main\n");
    return 0;
  }

  unsetenv("LUPINE_SERVER");
  CUresult result = cuInit(0);

  if (result == CUDA_ERROR_NO_DEVICE) {
    std::printf("PASS: cuInit without an available backend returned %d\n",
                static_cast<int>(result));
    return 0;
  }

  std::fprintf(stderr,
               "FAIL: cuInit without an available backend returned %d, "
               "expected %d\n",
               static_cast<int>(result),
               static_cast<int>(CUDA_ERROR_NO_DEVICE));
  return 1;
}
