// Integration test for module-image and module-file error codes: unrecognized
// bytes and empty files are INVALID_IMAGE, unopenable paths are FILE_NOT_FOUND,
// and PTX behind a block comment loads.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *cn(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}

static int failures = 0;

static void expect(const char *what, CUresult got, CUresult want) {
  if (got == want) {
    printf("ok: %s -> %s\n", what, cn(got));
    return;
  }
  printf("FAIL: %s -> %s, expected %s\n", what, cn(got), cn(want));
  ++failures;
}

static const char kCommentPtx[] = "/* leading block comment */\n"
                                  ".version 6.4\n"
                                  ".target sm_52\n"
                                  ".address_size 64\n"
                                  ".visible .entry nop() { ret; }\n";

int main() {
  cuInit(0);
  CUcontext ctx = nullptr;
  if (cuDevicePrimaryCtxRetain(&ctx, 0) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    printf("RESULT: ERROR context\n");
    return 2;
  }

  unsigned char garbage[64];
  for (size_t i = 0; i + 1 < sizeof(garbage); ++i) {
    garbage[i] = (unsigned char)(0xde + i * 7);
  }
  garbage[sizeof(garbage) - 1] = 0;

  CUmodule module = nullptr;
  expect("cuModuleLoadData(garbage bytes)", cuModuleLoadData(&module, garbage),
         CUDA_ERROR_INVALID_IMAGE);
  expect("cuModuleLoadData(text that is not ptx)",
         cuModuleLoadData(&module, "hello world, not ptx\n"),
         CUDA_ERROR_INVALID_IMAGE);
  expect("cuModuleLoadData(ptx behind a block comment)",
         cuModuleLoadData(&module, kCommentPtx), CUDA_SUCCESS);

  char empty_path[] = "/tmp/lupine_empty_module_XXXXXX";
  int empty_fd = mkstemp(empty_path);
  if (empty_fd < 0) {
    printf("RESULT: ERROR mkstemp\n");
    return 2;
  }

  expect("cuModuleLoad(empty file)", cuModuleLoad(&module, empty_path),
         CUDA_ERROR_INVALID_IMAGE);
  expect("cuModuleLoad(missing file)",
         cuModuleLoad(&module, "/tmp/lupine_no_such_module_file"),
         CUDA_ERROR_FILE_NOT_FOUND);

  CUlinkState link = nullptr;
  if (cuLinkCreate(0, nullptr, nullptr, &link) == CUDA_SUCCESS) {
    expect("cuLinkAddFile(empty file)",
           cuLinkAddFile(link, CU_JIT_INPUT_PTX, empty_path, 0, nullptr,
                         nullptr),
           CUDA_ERROR_INVALID_IMAGE);
    expect("cuLinkAddFile(missing file)",
           cuLinkAddFile(link, CU_JIT_INPUT_PTX,
                         "/tmp/lupine_no_such_module_file", 0, nullptr,
                         nullptr),
           CUDA_ERROR_FILE_NOT_FOUND);
    cuLinkDestroy(link);
  }

  close(empty_fd);
  remove(empty_path);

  printf("RESULT: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
