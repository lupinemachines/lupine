// Device registration for the torch backend: what libtorch demands in C++
// (allocator, device guard, hooks, generator, the metadata kernels). Every
// operator is implemented by the native Python bindings and GPU kernels.
#ifndef LUPINE_TORCH_BACKEND_H
#define LUPINE_TORCH_BACKEND_H

#include <cstdint>

namespace lupine_torch {

// A storage's data pointer is its handle shifted left, so Python reads the
// handle back from any tensor (untyped_storage().data_ptr() >> kHandleShift)
// while the native runtime maps each handle to a real CUDA allocation.
constexpr int kHandleShift = 20;

using release_fn = void (*)(uint64_t handle);
using sync_fn = void (*)();

// Register the native PrivateUse1 device. Release drops a storage allocation;
// sync waits for the native runtime through the device guard.
void register_backend(int device_count, release_fn release, sync_fn sync);
// Stops the callbacks (interpreter exit).
void clear_callbacks();
int device_count();
int current_device();
void set_device(int index);
int exchange_device(int index);

} // namespace lupine_torch

#endif
