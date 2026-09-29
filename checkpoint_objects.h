#pragma once

#include "rpc.h"
#include <cstdint>
#include <cuda.h>
#include <string>
#include <type_traits>
#include <vector>

namespace lupine_objects {
enum class kind : uint32_t { context, module, function, stream, event };
bool enabled();
void unsupported(const char *reason);
uintptr_t native(kind type, uintptr_t id);
uintptr_t wire(kind type, uintptr_t handle);
void context(CUcontext handle, CUdevice device, unsigned flags, bool primary);
void module(CUmodule handle, const void *image, size_t size, bool fatbinary);
void function(CUfunction handle, CUmodule parent, const char *name);
void stream(CUstream handle, unsigned flags, int priority);
void event(CUevent handle, unsigned flags);
void erase(kind type, uintptr_t handle);
int save(const char *path);
int restore(const char *path);
const void *wire_values(kind type, const void *handles, size_t size);
void finish_call();
void observe_call(int operation);

template <typename T> struct handle_type : std::false_type {};
#define LUPINE_OBJECT_TYPE(Type, Kind)                                         \
  template <> struct handle_type<Type> : std::true_type {                      \
    static constexpr kind type = kind::Kind;                                   \
  };
LUPINE_OBJECT_TYPE(CUcontext, context)
LUPINE_OBJECT_TYPE(CUmodule, module)
LUPINE_OBJECT_TYPE(CUfunction, function)
LUPINE_OBJECT_TYPE(CUstream, stream)
LUPINE_OBJECT_TYPE(CUevent, event)
#undef LUPINE_OBJECT_TYPE
} // namespace lupine_objects

// Translate opaque handles at the RPC boundary. Device addresses and the
// client's pointer tables are never rewritten. Server helpers use native
// handles throughout, including callback and staging state.
template <typename T,
          std::enable_if_t<lupine_objects::handle_type<T>::value, int> = 0>
int rpc_read(conn_t *conn, T *handles, size_t size) {
  int result = rpc_read(conn, static_cast<void *>(handles), size);
  if (result >= 0 && lupine_objects::enabled()) {
    for (size_t i = 0; i < size / sizeof(T); ++i)
      handles[i] = reinterpret_cast<T>(
          lupine_objects::native(lupine_objects::handle_type<T>::type,
                                 reinterpret_cast<uintptr_t>(handles[i])));
  }
  return result;
}

template <typename T,
          std::enable_if_t<lupine_objects::handle_type<T>::value, int> = 0>
int rpc_write(conn_t *conn, const T *handles, size_t size) {
  if (!lupine_objects::enabled())
    return rpc_write(conn, static_cast<const void *>(handles), size);
  return rpc_write(conn,
                   lupine_objects::wire_values(
                       lupine_objects::handle_type<T>::type, handles, size),
                   size);
}
