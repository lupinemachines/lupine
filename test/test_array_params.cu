// Array-valued driver parameters must cross the wire whole: every
// cuStreamBatchMemOp op must run, and all four border color channels must
// round-trip.
#include <cuda.h>
#include <stdio.h>
#include <string.h>

static const char *cn(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

static int check_batch_mem_op() {
  const unsigned int count = 4;
  CUdeviceptr words = 0;
  CUresult r = cuMemAlloc(&words, count * sizeof(cuuint32_t));
  if (r != CUDA_SUCCESS || cuMemsetD32(words, 0, count) != CUDA_SUCCESS) {
    printf("RESULT: ERROR alloc %s\n", cn(r));
    return 2;
  }
  CUstreamBatchMemOpParams ops[count];
  memset(ops, 0, sizeof(ops));
  for (unsigned int i = 0; i < count; ++i) {
    ops[i].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    ops[i].writeValue.address = words + i * sizeof(cuuint32_t);
    ops[i].writeValue.value = 0x11111111u * (i + 1);
  }
  r = cuStreamBatchMemOp_v2(0, count, ops, 0);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuStreamBatchMemOp_v2 %s\n", cn(r));
    return 1;
  }
  cuuint32_t readback[count] = {};
  r = cuMemcpyDtoH(readback, words, sizeof(readback));
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuMemcpyDtoH %s\n", cn(r));
    return 1;
  }
  for (unsigned int i = 0; i < count; ++i) {
    if (readback[i] != 0x11111111u * (i + 1)) {
      printf("RESULT: FAIL batch op %u wrote 0x%x\n", i, readback[i]);
      return 1;
    }
  }
  cuMemFree(words);
  return 0;
}

static int check_border_color() {
  CUtexref tex = nullptr;
  CUresult r = cuTexRefCreate(&tex);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: ERROR cuTexRefCreate %s\n", cn(r));
    return 2;
  }
  float color[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  r = cuTexRefSetBorderColor(tex, color);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuTexRefSetBorderColor %s\n", cn(r));
    return 1;
  }
  float got[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
  r = cuTexRefGetBorderColor(got, tex);
  if (r != CUDA_SUCCESS) {
    printf("RESULT: FAIL cuTexRefGetBorderColor %s\n", cn(r));
    return 1;
  }
  if (memcmp(got, color, sizeof(color)) != 0) {
    printf("RESULT: FAIL border color %g %g %g %g\n", got[0], got[1], got[2],
           got[3]);
    return 1;
  }
  cuTexRefDestroy(tex);
  return 0;
}

int main() {
  cuInit(0);
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("RESULT: ERROR context\n");
    return 2;
  }
  int rc = check_batch_mem_op();
  if (rc == 0)
    rc = check_border_color();
  cuDevicePrimaryCtxRelease(dev);
  if (rc == 0)
    printf("RESULT: OK\n");
  return rc;
}
