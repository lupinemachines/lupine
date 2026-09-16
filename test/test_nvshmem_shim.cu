// Checks what the nvSHMEM shim promises: that libnvshmem_host.so.3 loads and
// links, that the queries with a true answer give it, and that everything a
// real nvSHMEM job would need refuses instead of returning a plausible value.
// A GPU allocation on either side of the calls confirms the library being
// loaded leaves the rest of the stack alone.

#include <cuda_runtime.h>
#include <nvshmem_host.h>

#include <cstdio>

namespace {

int failures = 0;

void check(bool condition, const char *what) {
  printf("%s: %s\n", what, condition ? "passed" : "FAILED");
  if (!condition) {
    failures++;
  }
}

bool device_roundtrip() {
  int *device = nullptr;
  if (cudaMalloc(&device, sizeof(int)) != cudaSuccess) {
    return false;
  }
  const int sent = 0x5eed;
  int received = 0;
  const bool ok =
      cudaMemcpy(device, &sent, sizeof(sent), cudaMemcpyHostToDevice) ==
          cudaSuccess &&
      cudaMemcpy(&received, device, sizeof(received), cudaMemcpyDeviceToHost) ==
          cudaSuccess &&
      received == sent;
  cudaFree(device);
  return ok;
}

} // namespace

int main() {
  check(device_roundtrip(), "device before nvSHMEM");

  int major = -1;
  int minor = -1;
  nvshmem_info_get_version(&major, &minor);
  printf("nvSHMEM API %d.%d\n", major, minor);
  check(major == NVSHMEM_MAJOR_VERSION && minor == NVSHMEM_MINOR_VERSION,
        "API version");

  int vendor_major = -1;
  int vendor_minor = -1;
  int vendor_patch = -1;
  nvshmemx_vendor_get_version_info(&vendor_major, &vendor_minor,
                                   &vendor_patch);
  printf("vendor %d.%d.%d\n", vendor_major, vendor_minor, vendor_patch);
  check(vendor_major == NVSHMEM_VENDOR_MAJOR_VERSION, "vendor version");

  char name[NVSHMEM_MAX_NAME_LEN] = {};
  nvshmem_info_get_name(name);
  printf("library name: %s\n", name);
  check(name[0] != '\0', "library name");

  check(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED,
        "status before init");

  nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
  check(nvshmemx_get_uniqueid(&id) != 0, "unique id refused");

  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  check(nvshmemx_hostlib_init_attr(0, &attr) != 0, "init refused");
  check(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED,
        "status after refused init");

  check(nvshmem_malloc(1024) == nullptr, "symmetric allocation refused");
  check(nvshmem_ptr(nullptr, 0) == nullptr, "peer pointer refused");
  check(nvshmem_n_pes() != 0, "PE count refused");
  check(nvshmem_barrier(NVSHMEM_TEAM_WORLD) != 0, "barrier refused");
  check(nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, nullptr, nullptr, 0) != 0,
        "reduction refused");

  check(device_roundtrip(), "device after nvSHMEM");

  printf("nvshmem shim: %s\n",
         failures == 0 ? "all checks passed" : "checks FAILED");
  return failures == 0 ? 0 : 1;
}
