// Regression test for thread-local CUDA context stack semantics:
//  1. cuCtxPopCurrent on an empty stack returns CUDA_ERROR_INVALID_CONTEXT and
//     leaves *pctx untouched (so draining the stack in a while-loop terminates).
//  2. cuCtxPushCurrent(nullptr) returns CUDA_ERROR_INVALID_VALUE.
//  3. cuCtxPushCurrent / cuCtxCreate when no context is current does not push a
//     spurious nullptr underneath the new top of stack.
//  4. cuCtxSetCurrent(nullptr) pops the top of the calling thread's context
//     stack (or is a no-op when the stack is empty).
//  5. cuCtxDestroy on the current context pops it and restores the context
//     pushed underneath.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>

namespace {

bool expect(bool cond, const char *what) {
  if (cond) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s\n", what);
  return false;
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

  CUdevice dev = 0;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    return EXIT_FAILURE;
  }

  int failures = 0;
  CUcontext sentinel = reinterpret_cast<CUcontext>(0xdeadbeefULL);
  CUcontext popped = sentinel;
  CUcontext cur = sentinel;

  // 1. Empty stack pop and null push.
  failures += !expect(cuCtxPopCurrent(&popped) == CUDA_ERROR_INVALID_CONTEXT &&
                          popped == sentinel,
                      "cuCtxPopCurrent on fresh empty stack");
  failures += !expect(cuCtxPopCurrent(nullptr) == CUDA_ERROR_INVALID_CONTEXT,
                      "cuCtxPopCurrent(nullptr) on empty stack");
  failures += !expect(cuCtxPushCurrent(nullptr) == CUDA_ERROR_INVALID_VALUE,
                      "cuCtxPushCurrent(nullptr)");
  failures += !expect(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS,
                      "cuCtxSetCurrent(nullptr) on empty stack");

  CUcontext primary = nullptr;
  if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS) {
    return EXIT_FAILURE;
  }

  // 2. SetCurrent + PushCurrent + PopCurrent loop termination.
  failures += !expect(cuCtxSetCurrent(primary) == CUDA_SUCCESS,
                      "cuCtxSetCurrent(primary)");
  failures += !expect(cuCtxPushCurrent(primary) == CUDA_SUCCESS,
                      "cuCtxPushCurrent(primary)");
  int drained = 0;
  while (cuCtxPopCurrent(&popped) == CUDA_SUCCESS) {
    failures += !expect(popped == primary, "popped == primary");
    if (++drained > 8) {
      break;
    }
  }
  failures += !expect(drained == 2, "drained exactly 2 stack entries");
  failures += !expect(cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == nullptr,
                      "current context is nullptr after draining stack");

  // 3. cuCtxSetCurrent(nullptr) pops one entry from a non-empty stack.
  failures += !expect(cuCtxPushCurrent(primary) == CUDA_SUCCESS,
                      "push 1 from empty");
  failures += !expect(cuCtxPushCurrent(primary) == CUDA_SUCCESS, "push 2");
  failures += !expect(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS,
                      "cuCtxSetCurrent(nullptr) pops top entry");
  failures += !expect(cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == primary,
                      "underlying context restored after SetCurrent(nullptr)");
  popped = nullptr;
  failures += !expect(cuCtxPopCurrent(&popped) == CUDA_SUCCESS &&
                          popped == primary,
                      "pop remaining entry after SetCurrent(nullptr)");
  failures += !expect(cuCtxPopCurrent(&popped) == CUDA_ERROR_INVALID_CONTEXT,
                      "stack empty after second pop");

  // 4. cuCtxCreate pushes onto the stack; cuCtxDestroy pops the destroyed
  // current context and restores the context underneath.
  failures += !expect(cuCtxSetCurrent(primary) == CUDA_SUCCESS,
                      "SetCurrent(primary) before cuCtxCreate");
  CUcontext sub = nullptr;
#if CUDA_VERSION >= 12050
  CUresult create_rc = cuCtxCreate_v4(&sub, nullptr, 0, dev);
#else
  CUresult create_rc = cuCtxCreate_v2(&sub, 0, dev);
#endif
  failures += !expect(create_rc == CUDA_SUCCESS && sub != nullptr,
                      "cuCtxCreate");
  failures += !expect(cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == sub,
                      "cuCtxCreate made sub current");
  failures += !expect(cuCtxDestroy(sub) == CUDA_SUCCESS, "cuCtxDestroy(sub)");
  failures += !expect(cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == primary,
                      "cuCtxDestroy restored primary from stack");
  failures += !expect(cuCtxPopCurrent(&popped) == CUDA_SUCCESS &&
                          popped == primary,
                      "pop primary after cuCtxDestroy");
  failures += !expect(cuCtxPopCurrent(&popped) == CUDA_ERROR_INVALID_CONTEXT,
                      "stack empty at end");

  cuDevicePrimaryCtxRelease(dev);
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d context stack checks failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: CUDA context stack semantics match native driver\n");
  return EXIT_SUCCESS;
}
