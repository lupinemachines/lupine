// Regression test for cuPointerGetAttribute and cuPointerGetAttributes
// CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, CU_POINTER_ATTRIBUTE_RANGE_SIZE, and
// CU_POINTER_ATTRIBUTE_HOST_POINTER parity on host, registered, managed, and
// device pointers.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

bool expect_status(CUresult got, CUresult want, const char *what) {
  if (got == want) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: got %d, want %d\n", what,
               static_cast<int>(got), static_cast<int>(want));
  return false;
}

int check_range_queries(const char *label, CUdeviceptr query,
                        CUdeviceptr expected_base, size_t expected_size) {
  int failures = 0;

  CUdeviceptr single_start = 0xdeadbeefULL;
  size_t single_size = 0xdeadbeefULL;
  failures += !expect_status(
      cuPointerGetAttribute(&single_start,
                            CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, query),
      CUDA_SUCCESS, label);
  failures += !expect_status(
      cuPointerGetAttribute(&single_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                            query),
      CUDA_SUCCESS, label);
  if (single_start != expected_base) {
    std::fprintf(stderr,
                 "FAIL: %s single RANGE_START_ADDR got 0x%llx, want 0x%llx\n",
                 label, static_cast<unsigned long long>(single_start),
                 static_cast<unsigned long long>(expected_base));
    ++failures;
  }
  if (single_size != expected_size) {
    std::fprintf(stderr, "FAIL: %s single RANGE_SIZE got %zu, want %zu\n",
                 label, single_size, expected_size);
    ++failures;
  }

  for (int pass = 0; pass < 2; ++pass) {
    CUpointer_attribute attrs[3] = {
        CU_POINTER_ATTRIBUTE_HOST_POINTER,
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        CU_POINTER_ATTRIBUTE_RANGE_SIZE,
    };
    void *batch_host = reinterpret_cast<void *>(0xdeadbeefULL);
    CUdeviceptr batch_start = 0xdeadbeefULL;
    size_t batch_size = 0xdeadbeefULL;
    void *ptrs[3] = {&batch_host, &batch_start, &batch_size};
    failures += !expect_status(cuPointerGetAttributes(3, attrs, ptrs, query),
                               CUDA_SUCCESS, label);
    if (batch_host != reinterpret_cast<void *>(query)) {
      std::fprintf(stderr,
                   "FAIL: %s batch[p%d] HOST_POINTER got %p, want %p\n", label,
                   pass, batch_host, reinterpret_cast<void *>(query));
      ++failures;
    }
    if (batch_start != expected_base) {
      std::fprintf(
          stderr,
          "FAIL: %s batch[p%d] RANGE_START_ADDR got 0x%llx, want 0x%llx\n",
          label, pass, static_cast<unsigned long long>(batch_start),
          static_cast<unsigned long long>(expected_base));
      ++failures;
    }
    if (batch_size != expected_size) {
      std::fprintf(stderr,
                   "FAIL: %s batch[p%d] RANGE_SIZE got %zu, want %zu\n", label,
                   pass, batch_size, expected_size);
      ++failures;
    }
  }

  return failures;
}

int test_device(int ordinal) {
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  if (cuDeviceGet(&dev, ordinal) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: setup device %d\n", ordinal);
    return 1;
  }

  int failures = 0;
  constexpr size_t kSize = 4096;

  void *unreg = aligned_alloc(4096, kSize);
  void *pinned = nullptr;
  void *mapped = nullptr;
  void *reg = aligned_alloc(4096, kSize);
  CUdeviceptr dptr = 0;
  CUdeviceptr uptr = 0;

  if (unreg == nullptr || reg == nullptr ||
      cuMemAlloc_v2(&dptr, kSize) != CUDA_SUCCESS ||
      cuMemAllocHost_v2(&pinned, kSize) != CUDA_SUCCESS ||
      cuMemHostAlloc(&mapped, kSize, CU_MEMHOSTALLOC_DEVICEMAP) !=
          CUDA_SUCCESS ||
      cuMemHostRegister_v2(reg, kSize, CU_MEMHOSTREGISTER_DEVICEMAP) !=
          CUDA_SUCCESS ||
      cuMemAllocManaged(&uptr, kSize, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: allocations on device %d\n", ordinal);
    return 1;
  }

  // 1. CU_POINTER_ATTRIBUTE_HOST_POINTER on a pure device pointer returns
  // CUDA_ERROR_INVALID_VALUE and zeroes the caller's pointer output, whereas
  // an unregistered host pointer leaves the output untouched.
  void *sentinel = reinterpret_cast<void *>(0xdeadbeefULL);
  void *got_host = sentinel;
  failures += !expect_status(
      cuPointerGetAttribute(&got_host, CU_POINTER_ATTRIBUTE_HOST_POINTER, dptr),
      CUDA_ERROR_INVALID_VALUE, "cuPointerGetAttribute(HOST_POINTER, dptr)");
  if (got_host != nullptr) {
    std::fprintf(stderr, "FAIL: dptr HOST_POINTER got %p, want nullptr\n",
                 got_host);
    ++failures;
  }

  got_host = sentinel;
  failures += !expect_status(
      cuPointerGetAttribute(&got_host, CU_POINTER_ATTRIBUTE_HOST_POINTER,
                            dptr + 64),
      CUDA_ERROR_INVALID_VALUE, "cuPointerGetAttribute(HOST_POINTER, dptr+64)");
  if (got_host != nullptr) {
    std::fprintf(stderr, "FAIL: dptr+64 HOST_POINTER got %p, want nullptr\n",
                 got_host);
    ++failures;
  }

  got_host = sentinel;
  failures += !expect_status(
      cuPointerGetAttribute(&got_host, CU_POINTER_ATTRIBUTE_HOST_POINTER,
                            reinterpret_cast<CUdeviceptr>(unreg)),
      CUDA_ERROR_INVALID_VALUE, "cuPointerGetAttribute(HOST_POINTER, unreg)");
  if (got_host != sentinel) {
    std::fprintf(stderr, "FAIL: unreg HOST_POINTER modified output to %p\n",
                 got_host);
    ++failures;
  }

  // 2. Range start and size across pinned, mapped, registered, and managed.
  CUdeviceptr pinned_base = reinterpret_cast<CUdeviceptr>(pinned);
  CUdeviceptr mapped_base = reinterpret_cast<CUdeviceptr>(mapped);
  CUdeviceptr reg_base = reinterpret_cast<CUdeviceptr>(reg);

  failures += check_range_queries("pinned_base", pinned_base, pinned_base,
                                  kSize);
  failures += check_range_queries("pinned_off", pinned_base + 64, pinned_base,
                                  kSize);
  failures += check_range_queries("mapped_base", mapped_base, mapped_base,
                                  kSize);
  failures += check_range_queries("mapped_off", mapped_base + 64, mapped_base,
                                  kSize);
  failures += check_range_queries("reg_base", reg_base, reg_base, kSize);
  failures += check_range_queries("reg_off", reg_base + 64, reg_base, kSize);
  failures += check_range_queries("managed_base", uptr, uptr, kSize);
  failures += check_range_queries("managed_off", uptr + 64, uptr, kSize);

  (void)cuMemFree_v2(uptr);
  (void)cuMemHostUnregister(reg);
  std::free(reg);
  (void)cuMemFreeHost(mapped);
  (void)cuMemFreeHost(pinned);
  (void)cuMemFree_v2(dptr);
  std::free(unreg);
  (void)cuDevicePrimaryCtxRelease(dev);
  return failures;
}

} // namespace

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: cuInit\n");
    return EXIT_FAILURE;
  }
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  int failures = 0;
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    failures += test_device(ordinal);
  }

  if (failures != 0) {
    std::fprintf(stderr, "FAILED with %d error(s)\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS\n");
  return EXIT_SUCCESS;
}
