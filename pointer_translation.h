#pragma once

#include <cuda.h>

// Optional runtime provider ABI. Lupine never links against a provider; it
// loads the shared object named by LUPINE_POINTER_TRANSLATOR when needed.
using lupine_pointer_translate_fn = int (*)(void *context,
                                            CUdeviceptr old_pointer,
                                            CUdeviceptr *new_pointer);
using lupine_pointer_provider_fn = int (*)(
    lupine_pointer_translate_fn *translate, void **context);

// Returns true when an optional provider translated a device pointer.
bool lupine_translate_device_pointer(CUdeviceptr old_pointer,
                                     CUdeviceptr *new_pointer);

// Returns true when an optional provider translated the serialized value.
bool lupine_translate_rpc_pointer(const void *data, size_t size,
                                  void **owned_copy);
