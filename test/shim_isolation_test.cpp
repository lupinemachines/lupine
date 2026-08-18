#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << std::endl;
    std::exit(1);
  }
}

void require_hidden(void *library, const char *symbol) {
  dlerror();
  require(dlsym(library, symbol) == nullptr, symbol);
  (void)dlerror();
}

} // namespace

int main(int argc, char **argv) {
  require(argc == 3, "usage: shim_isolation_test LIBCUDA LIBNVML");
  void *cuda = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  require(cuda != nullptr, dlerror());
  void *nvml = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  require(nvml != nullptr, dlerror());

  require(dlsym(cuda, "cuInit") != nullptr, "CUDA entry point unavailable");
  require(dlsym(nvml, "nvmlInit_v2") != nullptr,
          "NVML entry point unavailable");

  for (const char *symbol :
       {"rpc_dispatch", "rpc_set_lifecycle_hooks",
        "lupine_client_transport_open",
        "lupine_client_transport_connection"}) {
    require_hidden(cuda, symbol);
    require_hidden(nvml, symbol);
  }

  require(dlclose(nvml) == 0, "NVML shim close failed");
  require(dlclose(cuda) == 0, "CUDA shim close failed");
  std::cout << "shim_isolation_test: PASS" << std::endl;
  return 0;
}
