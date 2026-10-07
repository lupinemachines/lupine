// Device registration for the torch backend: what libtorch demands in C++
// (allocator, device guard, hooks, generator, the metadata kernels). Every
// operator and every byte on the wire is Python's (lupine._backend.forward).
#ifndef LUPINE_TORCH_BACKEND_H
#define LUPINE_TORCH_BACKEND_H

#include <cstdint>

namespace lupine_torch {

// A storage's data pointer is its handle shifted left, so Python reads the
// handle back from any tensor (untyped_storage().data_ptr() >> kHandleShift)
// and the worker's own handles, numbered from 1 << 40, stay distinct.
constexpr int kHandleShift = 20;

using release_fn = void (*)(uint64_t handle);
using sync_fn = void (*)();

// Registers the backend for PrivateUse1 and, when `dual` (the host torch has
// no CUDA build), for the in-tree CUDA key as well. `release` is told the
// handle of every storage the host drops; `sync` waits for the worker when
// torch synchronizes through the device guard.
void register_backend(int device_count, bool dual, release_fn release,
                      sync_fn sync);
// Stops the callbacks (interpreter exit).
void clear_callbacks();
// Runs at extension load, before torch first asks for its CUDA hooks.
void register_cuda_hooks();
int device_count();
int current_device();
void set_device(int index);
int exchange_device(int index);

} // namespace lupine_torch

#endif
