#include <cuda.h>

#include <cstdio>
#include <cstdlib>

static const char *result_name(CUresult result) {
  const char *name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "UNKNOWN";
}

static void check(CUresult result, const char *expr, int line) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 result_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

int main() {
  CHECK(cuInit(0));

  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  char pci_bus_id[64] = {};
  CHECK(cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));

  CUresult result = cuDeviceGetByPCIBusId(nullptr, pci_bus_id);
  if (result != CUDA_ERROR_INVALID_VALUE) {
    std::fprintf(stderr,
                 "cuDeviceGetByPCIBusId(nullptr, %s) returned %s (%d), "
                 "expected CUDA_ERROR_INVALID_VALUE\n",
                 pci_bus_id, result_name(result), static_cast<int>(result));
    return EXIT_FAILURE;
  }

  std::printf("PASS: output-only null pointer returns CUDA invalid value\n");
  return EXIT_SUCCESS;
}
