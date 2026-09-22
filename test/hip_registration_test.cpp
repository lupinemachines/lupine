#include <hip/hip_runtime_api.h>

#include "hip_fatbin.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" void **__hipRegisterFatBinary(const void *);
extern "C" void __hipUnregisterFatBinary(void **);
extern "C" void __hipRegisterFunction(void **, const void *, char *,
                                      const char *, unsigned int, uint3 *,
                                      uint3 *, dim3 *, dim3 *, int *);

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%d: %s\n", __LINE__, #condition);                       \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

static void launch_stack(unsigned int value) {
  dim3 grid, block;
  size_t shared = 0;
  hipStream_t stream = nullptr;
  EXPECT(__hipPopCallConfiguration(&grid, &block, &shared, &stream) ==
         hipErrorInvalidValue);
  EXPECT(__hipPushCallConfiguration(dim3(value), dim3(2, 3, 4), 123, nullptr) ==
         hipSuccess);
  EXPECT(__hipPushCallConfiguration(dim3(value + 1), dim3(5), 456, nullptr) ==
         hipSuccess);
  EXPECT(__hipPopCallConfiguration(nullptr, &block, &shared, &stream) ==
         hipErrorInvalidValue);
  EXPECT(__hipPopCallConfiguration(&grid, &block, &shared, &stream) ==
         hipSuccess);
  EXPECT(grid.x == value + 1 && block.x == 5 && shared == 456);
  EXPECT(__hipPopCallConfiguration(&grid, &block, &shared, &stream) ==
         hipSuccess);
  EXPECT(grid.x == value && block.x == 2 && block.y == 3 && block.z == 4 &&
         shared == 123 && stream == nullptr);
}

static void host_stub() {}

int main() {
  // No GPU or server is involved in compiler registration or launch stacks.
  EXPECT(__hipPushCallConfiguration(dim3(99), dim3(1), 0, nullptr) == hipSuccess);
  std::thread first(launch_stack, 7);
  std::thread second(launch_stack, 11);
  first.join();
  second.join();
  dim3 grid, block;
  size_t shared;
  hipStream_t stream;
  EXPECT(__hipPopCallConfiguration(&grid, &block, &shared, &stream) == hipSuccess);
  EXPECT(grid.x == 99);

  std::vector<unsigned char> image(128, 0);
  memcpy(image.data(), "__CLANG_OFFLOAD_BUNDLE__", 24);
  const uint64_t fields[] = {1, 64, 64, 4};
  memcpy(image.data() + 24, fields, sizeof(fields));
  memcpy(image.data() + 56, "test", 4);
  EXPECT(lupine_hip_bundle_size(image.data(), image.size()) == image.size());
  for (size_t size = 0; size < image.size(); ++size) {
    EXPECT(lupine_hip_bundle_size(image.data(), size) == 0);
  }
  uint64_t invalid = UINT64_MAX;
  for (size_t offset : {24, 32, 40, 48}) {
    auto malformed = image;
    memcpy(malformed.data() + offset, &invalid, sizeof(invalid));
    EXPECT(lupine_hip_bundle_size(malformed.data(), malformed.size()) == 0);
  }
  auto overlapping = image;
  uint64_t offset = 40;
  memcpy(overlapping.data() + 32, &offset, sizeof(offset));
  EXPECT(lupine_hip_bundle_size(overlapping.data(), overlapping.size()) == 0);

  struct {
    uint32_t magic = 0x48495046;
    uint32_t version = 1;
    const void *image;
    void *reserved = nullptr;
  } wrapper;
  wrapper.image = image.data();
  EXPECT(__hipRegisterFatBinary(nullptr) == nullptr);
  void **handle = __hipRegisterFatBinary(&wrapper);
  EXPECT(handle != nullptr);
  const void *host = reinterpret_cast<const void *>(host_stub);
  __hipRegisterFunction(handle, host, nullptr, "kernel", 0, nullptr, nullptr,
                        nullptr, nullptr, nullptr);
  __hipUnregisterFatBinary(handle);
  __hipUnregisterFatBinary(handle);
  EXPECT(hipLaunchKernel(host, dim3(1), dim3(1), nullptr, 0, nullptr) ==
         hipErrorInvalidDeviceFunction);
  EXPECT(hipModuleLoadData(nullptr, image.data()) == hipErrorInvalidValue);
  EXPECT(hipGetDevice(nullptr) == hipErrorInvalidValue);
  puts("PASS");
}
