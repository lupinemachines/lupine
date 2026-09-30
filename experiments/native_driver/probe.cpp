// Loads the actual NVIDIA library, not Lupine's CUDA API shim. Success includes
// kernel execution, readback, and destruction of the native client objects.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s NATIVE_LIBCUDA CUBIN\n", argv[0]);
    return 2;
  }
  void *driver = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!driver) {
    std::fprintf(stderr, "%s\n", dlerror());
    return 2;
  }
  auto init = reinterpret_cast<int (*)(unsigned)>(dlsym(driver, "cuInit"));
  auto count =
      reinterpret_cast<int (*)(int *)>(dlsym(driver, "cuDeviceGetCount"));
  Dl_info origin{};
  dladdr(reinterpret_cast<void *>(init), &origin);
  std::printf("native cuInit comes from %s\n", origin.dli_fname);
  int result = init(0), devices = -1;
  int counted = count(&devices);
  std::printf("native cuInit=%d cuDeviceGetCount=%d count=%d\n", result,
              counted, devices);
  if (result || counted || devices < 1)
    return 1;
  auto create = reinterpret_cast<int (*)(void **, unsigned, int)>(
      dlsym(driver, "cuCtxCreate_v2"));
  auto destroy =
      reinterpret_cast<int (*)(void *)>(dlsym(driver, "cuCtxDestroy_v2"));
  void *ctx = nullptr;
  std::printf("creating native client context\n");
  std::fflush(stdout);
  int made = create(&ctx, 0, 0);
  std::printf("native cuCtxCreate=%d client context=%p\n", made, ctx);
  std::fflush(stdout);
  if (made)
    return 1;
  auto alloc = reinterpret_cast<int (*)(uint64_t *, size_t)>(
      dlsym(driver, "cuMemAlloc_v2"));
  auto release =
      reinterpret_cast<int (*)(uint64_t)>(dlsym(driver, "cuMemFree_v2"));
  auto load = reinterpret_cast<int (*)(void **, const void *)>(
      dlsym(driver, "cuModuleLoadData"));
  auto function = reinterpret_cast<int (*)(void **, void *, const char *)>(
      dlsym(driver, "cuModuleGetFunction"));
  auto launch = reinterpret_cast<int (*)(
      void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
      unsigned, void *, void **, void **)>(dlsym(driver, "cuLaunchKernel"));
  auto sync = reinterpret_cast<int (*)()>(dlsym(driver, "cuCtxSynchronize"));
  auto readback = reinterpret_cast<int (*)(void *, uint64_t, size_t)>(
      dlsym(driver, "cuMemcpyDtoH_v2"));
  auto unload =
      reinterpret_cast<int (*)(void *)>(dlsym(driver, "cuModuleUnload"));
  auto check = [](int result, const char *operation) {
    std::printf("native %s=%d\n", operation, result);
    std::fflush(stdout);
    if (result)
      std::exit(1);
  };
  FILE *input = std::fopen(argv[2], "rb");
  if (!input)
    return 2;
  std::fseek(input, 0, SEEK_END);
  size_t size = std::ftell(input);
  std::rewind(input);
  std::vector<unsigned char> cubin(size);
  if (std::fread(cubin.data(), 1, size, input) != size)
    return 2;
  std::fclose(input);
  uint64_t gpu = 0;
  void *module = nullptr, *kernel = nullptr;
  check(alloc(&gpu, sizeof(uint32_t)), "cuMemAlloc");
  check(load(&module, cubin.data()), "cuModuleLoadData");
  check(function(&kernel, module, "write_value"), "cuModuleGetFunction");
  std::printf("client module=%p function=%p gpu=%llx\n", module, kernel,
              (unsigned long long)gpu);
  std::fflush(stdout);
  void *args[] = {&gpu};
  check(launch(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr),
        "cuLaunchKernel");
  check(sync(), "cuCtxSynchronize");
  uint32_t value = 0;
  check(readback(&value, gpu, sizeof(value)), "cuMemcpyDtoH");
  std::printf("native kernel readback=%08x expected=12345678\n", value);
  std::fflush(stdout);
  if (value != 0x12345678)
    return 1;
  check(unload(module), "cuModuleUnload");
  check(release(gpu), "cuMemFree");
  int destroyed = destroy(ctx);
  std::printf("native cuCtxDestroy=%d\n", destroyed);
  if (destroyed)
    return 1;
  return 0;
}
