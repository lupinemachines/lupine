#ifndef LUPINE_TORCH_BACKEND_H
#define LUPINE_TORCH_BACKEND_H

#include <c10/core/Device.h>

#include <cstdint>
#include <string>

namespace lupine_torch {

namespace host {
// Connects to the worker at host:port, registers the backend kernels for
// PrivateUse1 and, when `dual` (the host torch has no CUDA build), for the
// in-tree CUDA key as well. Returns the worker's JSON description.
std::string connect(const std::string &address, bool dual);
// Runs at extension load, before torch first asks for its CUDA hooks.
void register_cuda_hooks();
bool connected();
// Closes the worker connection so the worker can exit on its own.
void disconnect();
int device_count();
c10::DeviceIndex current_device();
void set_device(c10::DeviceIndex index);
c10::DeviceIndex exchange_device(c10::DeviceIndex index);
// Blocks until every op issued so far has executed; raises the first error a
// fire-and-forget op produced since the last sync.
void synchronize();
void exec(const std::string &code);
std::string eval(const std::string &code);
} // namespace host

namespace worker {
// Serves one host connection on an accepted socket until it closes.
void serve(int fd);
// Sets the CUDA stream every lane thread runs ops on (a packed c10::Stream).
void set_stream(int64_t stream_id, int16_t device_index);
} // namespace worker

} // namespace lupine_torch

#endif
