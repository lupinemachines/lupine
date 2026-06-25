#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

namespace {

using CuUserObjectCreateFn = CUresult(CUDAAPI *)(CUuserObject *, void *,
                                                 CUhostFn, unsigned int,
                                                 unsigned int);

void CUDA_CB destroy_user_object(void *ptr) {
  auto *value = static_cast<std::atomic<int> *>(ptr);
  value->store(0x51A7E11, std::memory_order_release);
}

const char *result_name(CUresult result) {
  const char *name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "CUresult";
}

int fail_cuda(CUresult result, const char *call) {
  std::fprintf(stderr, "%s failed: %s (%d)\n", call, result_name(result),
               static_cast<int>(result));
  return 1;
}

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    CUresult _result = (call);                                                 \
    if (_result != CUDA_SUCCESS) {                                             \
      return fail_cuda(_result, #call);                                        \
    }                                                                          \
  } while (0)

} // namespace

int main() {
  CHECK_CUDA(cuInit(0));

  void *raw_create = nullptr;
  CUdriverProcAddressQueryResult symbol_status =
      CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  CHECK_CUDA(cuGetProcAddress("cuUserObjectCreate", &raw_create, CUDA_VERSION,
                              CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status));
  if (raw_create == nullptr ||
      symbol_status != CU_GET_PROC_ADDRESS_SUCCESS) {
    std::fprintf(stderr,
                 "cuUserObjectCreate lookup failed: ptr=%p status=%d\n",
                 raw_create, static_cast<int>(symbol_status));
    return 1;
  }

  auto create = reinterpret_cast<CuUserObjectCreateFn>(raw_create);
  std::atomic<int> payload{0};
  CUuserObject object = nullptr;
  CHECK_CUDA(create(&object, &payload, destroy_user_object, 1,
                    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC));
  CHECK_CUDA(cuUserObjectRelease(object, 1));
  for (int i = 0; i < 100; ++i) {
    if (payload.load(std::memory_order_acquire) == 0x51A7E11) {
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::fprintf(stderr, "user-object destructor callback did not run\n");
  return 1;
}
