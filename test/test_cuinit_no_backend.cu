#include <cuda.h>

#include <cstdio>
#include <cstdlib>

int main() {
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
