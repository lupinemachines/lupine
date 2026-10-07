#pragma once

// Native CUDA can open its own devices during initialization or a local GPU
// call. Those opens must bypass the remote device hooks, including in mixed
// local/remote configurations.
inline thread_local unsigned int lupine_native_cuda_call_depth = 0;

struct lupine_native_cuda_call_guard {
  lupine_native_cuda_call_guard() { ++lupine_native_cuda_call_depth; }
  ~lupine_native_cuda_call_guard() { --lupine_native_cuda_call_depth; }
  lupine_native_cuda_call_guard(const lupine_native_cuda_call_guard &) = delete;
  lupine_native_cuda_call_guard &
  operator=(const lupine_native_cuda_call_guard &) = delete;
};
