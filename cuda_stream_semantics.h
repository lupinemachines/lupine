#pragma once
#include <cuda.h>

// Only implicit-stream API wrappers set this scope. Explicit-stream variants
// pass CU_STREAM_PER_THREAD through the normal transport to the same lane.
inline thread_local bool lupine_per_thread_default_stream = false;

class lupine_per_thread_stream_scope {
  bool previous = lupine_per_thread_default_stream;

public:
  lupine_per_thread_stream_scope() { lupine_per_thread_default_stream = true; }
  ~lupine_per_thread_stream_scope() {
    lupine_per_thread_default_stream = previous;
  }
};

inline CUstream lupine_per_thread_stream(CUstream stream) {
  return stream == nullptr ? CU_STREAM_PER_THREAD : stream;
}

inline CUstream lupine_implicit_stream() {
  return lupine_per_thread_default_stream ? CU_STREAM_PER_THREAD
                                          : CU_STREAM_LEGACY;
}
