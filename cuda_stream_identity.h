#pragma once

#include <atomic>
#include <cstdint>
#include <cuda.h>
#include <functional>

// Default streams belong to a context, and the per-thread stream also belongs
// to a lane. Never use the sentinel handles alone as global bookkeeping keys.
struct lupine_stream_key {
  CUstream stream = nullptr;
  CUcontext context = nullptr;
  uint64_t lane = 0;
  bool operator==(const lupine_stream_key &other) const {
    return stream == other.stream && context == other.context &&
           lane == other.lane;
  }
  bool operator!=(const lupine_stream_key &other) const {
    return !(*this == other);
  }
};

namespace std {
template <> struct hash<lupine_stream_key> {
  size_t operator()(const lupine_stream_key &key) const {
    return hash<CUstream>{}(key.stream) ^
           (hash<CUcontext>{}(key.context) << 1) ^
           (hash<uint64_t>{}(key.lane) << 2);
  }
};
} // namespace std

inline bool lupine_is_default_stream(CUstream stream) {
  return stream == nullptr || stream == CU_STREAM_LEGACY ||
         stream == CU_STREAM_PER_THREAD;
}

inline lupine_stream_key lupine_stream_identity(CUstream stream,
                                                CUcontext context) {
  if (!lupine_is_default_stream(stream))
    return {stream};
  if (stream != CU_STREAM_PER_THREAD)
    return {CU_STREAM_LEGACY, context};
  static std::atomic<uint64_t> next_lane{1};
  static thread_local uint64_t lane = next_lane.fetch_add(1);
  return {stream, context, lane};
}

// Capture this identity before entering a CUDA callback: callbacks execute on
// another thread and must not call CUDA APIs.
inline lupine_stream_key lupine_server_stream_identity(CUstream stream) {
  CUcontext context = nullptr;
  if (lupine_is_default_stream(stream))
    (void)cuCtxGetCurrent(&context);
  return lupine_stream_identity(stream, context);
}
