#pragma once

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#if defined(LUPINE_CUDA_CLIENT_API_EXPORTS)
#define LUPINE_CUDA_CLIENT_API_EXPORT __declspec(dllexport)
#else
#define LUPINE_CUDA_CLIENT_API_EXPORT __declspec(dllimport)
#endif
#else
#define LUPINE_CUDA_CLIENT_API_EXPORT __attribute__((visibility("default")))
#endif

static constexpr std::uint32_t LUPINE_CUDA_CLIENT_API_VERSION = 1;

enum lupine_cuda_call_result : std::int32_t {
  LUPINE_CUDA_CALL_SUCCESS = 0,
  LUPINE_CUDA_CALL_UNAVAILABLE = -1,
  LUPINE_CUDA_CALL_INVALID_OWNER = -2,
  LUPINE_CUDA_CALL_OWNER_MISMATCH = -3,
};

enum lupine_cuda_owner_kind : std::uint32_t {
  LUPINE_CUDA_OWNER_CURRENT_CONTEXT = 1,
  LUPINE_CUDA_OWNER_CUBLAS_HANDLE = 2,
  LUPINE_CUDA_OWNER_STREAM = 3,
  LUPINE_CUDA_OWNER_DEVICE_POINTER = 4,
};

struct lupine_cuda_owner {
  lupine_cuda_owner_kind kind;
  std::uintptr_t value;
};

struct lupine_cuda_call;

struct lupine_cuda_client_api {
  std::uint32_t version;
  int (*call_begin)(int operation, const lupine_cuda_owner *owners,
                    std::size_t owner_count, lupine_cuda_call **call);
  int (*call_write)(lupine_cuda_call *call, const void *data, std::size_t size);
  int (*call_invoke)(lupine_cuda_call *call);
  int (*call_read)(lupine_cuda_call *call, void *data, std::size_t size);
  int (*call_finish)(lupine_cuda_call *call);
  int (*record_owner)(lupine_cuda_call *call, lupine_cuda_owner_kind kind,
                      std::uintptr_t value);
  void (*forget_owner)(lupine_cuda_owner_kind kind, std::uintptr_t value);
  void (*call_destroy)(lupine_cuda_call *call);
};

extern "C" LUPINE_CUDA_CLIENT_API_EXPORT const lupine_cuda_client_api *
lupine_cuda_client_api_v1();
