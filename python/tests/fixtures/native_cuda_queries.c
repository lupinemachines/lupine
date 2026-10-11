#include <cuda.h>
#include <cudaTypedefs.h>

// Current CUDA headers redirect the public name to the five-argument API.
// The literal older export retains PFN_cuGetProcAddress_v11030's signature.
#undef cuGetProcAddress
extern CUresult cuGetProcAddress(const char *, void **, int, cuuint64_t);

static int match(int modern, const char *name, void *expected) {
  void *pointer = 0;
  CUdriverProcAddressQueryResult status = 99;
  CUresult result = modern
                        ? cuGetProcAddress_v2(name, &pointer, 13000, 0, &status)
                        : cuGetProcAddress(name, &pointer, 13000, 0);
  return result == 0 && pointer == expected && (!modern || status == 0);
}

int check_legacy(void) {
  return match(0, "cuLaunchKernel", (void *)cuLaunchKernel);
}

int check_modern(void) {
  return match(1, "cuLaunchKernel", (void *)cuLaunchKernel);
}

int check_recursive(void) {
  void *query = 0, *pointer = 0;
  CUdriverProcAddressQueryResult status = 99;
  if (cuGetProcAddress("cuGetProcAddress", &query, 13000, 0) ||
      query != (void *)cuGetProcAddress_v2)
    return 0;
  PFN_cuGetProcAddress_v12000 function = (PFN_cuGetProcAddress_v12000)query;
  return function("cuLaunchKernel", &pointer, 13000, 0, &status) == 0 &&
         pointer == (void *)cuLaunchKernel && status == 0;
}

int check_unsupported_legacy(void) {
  void *pointer = (void *)1;
  return cuGetProcAddress("cuMemAdvise", &pointer, 13000, 0) == 500 && !pointer;
}

int check_unsupported_modern(void) {
  void *pointer = (void *)1;
  CUdriverProcAddressQueryResult status = 99;
  return cuGetProcAddress_v2("cuMemAdvise", &pointer, 13000, 0, &status) == 500 &&
         !pointer;
}
