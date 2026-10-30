// cuGetErrorName and cuGetErrorString return strings with static lifetime: a
// pointer from one lookup must stay valid, and unchanged, after later lookups
// of other codes on the same thread.
#include <cuda.h>

#include <stdio.h>
#include <string.h>

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      g_failures++;                                                            \
    } else {                                                                   \
      fprintf(stderr, "ok:   %s\n", msg);                                      \
    }                                                                          \
  } while (0)

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    fprintf(stderr, "FATAL: cuInit\n");
    return 1;
  }

  const char *name = nullptr;
  const char *other_name = nullptr;
  const char *again = nullptr;
  CHECK(cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &name) == CUDA_SUCCESS &&
            cuGetErrorName(CUDA_ERROR_OUT_OF_MEMORY, &other_name) ==
                CUDA_SUCCESS &&
            cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &again) == CUDA_SUCCESS,
        "names resolve");
  CHECK(name != nullptr && strcmp(name, "CUDA_ERROR_INVALID_VALUE") == 0,
        "an earlier name survives a later lookup");
  CHECK(other_name != nullptr &&
            strcmp(other_name, "CUDA_ERROR_OUT_OF_MEMORY") == 0,
        "the later name is its own");
  CHECK(name == again, "a code always yields the same pointer");

  const char *description = nullptr;
  const char *other_description = nullptr;
  const char *description_again = nullptr;
  CHECK(cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &description) ==
                CUDA_SUCCESS &&
            cuGetErrorString(CUDA_ERROR_OUT_OF_MEMORY, &other_description) ==
                CUDA_SUCCESS &&
            cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &description_again) ==
                CUDA_SUCCESS,
        "descriptions resolve");
  CHECK(description != nullptr && other_description != nullptr &&
            description != other_description &&
            strcmp(description, other_description) != 0,
        "an earlier description is not overwritten by a later one");
  CHECK(description == description_again,
        "a code always yields the same description pointer");

  printf("%s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
