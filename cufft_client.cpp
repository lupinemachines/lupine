// cuFFT (libcufft) on the LUPINE client.
//
// Every call is an RPC the server answers with its own libcufft, sent on the
// CUDA driver shim's connections: the server child that owns the driver and
// runtime state runs the library too, and one lane per client thread keeps
// the three APIs ordered. Nearly all of the surface is generated. The calls
// in this file carry something the generated marshalling cannot: a multi-GPU
// descriptor the caller reads, or an array with one entry per GPU of the
// plan.

#include <cuda_runtime_api.h>
#include <cufftXt.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"

namespace {

cufftResult rpc_error() { return CUFFT_INTERNAL_ERROR; }

// The generated code speaks the RPC core's vocabulary; the driver shim exports
// it under its own prefix so both can be declared in one translation unit.
int rpc_write_start_request(conn_t *conn, int op) {
  return lupine_rpc_write_start_request(conn, op);
}
int rpc_write(conn_t *conn, const void *data, size_t size) {
  return lupine_rpc_write(conn, data, size);
}
int rpc_wait_for_response(conn_t *conn) {
  return lupine_rpc_wait_for_response(conn);
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  return lupine_rpc_read(conn, data, size);
}
int rpc_read_end(conn_t *conn) { return lupine_rpc_read_end(conn); }

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

// A plan is created on the runtime's current device and routes every later
// call to that connection. Its GPU count sizes the per-GPU arrays the work
// area and callback setters take, which the wire cannot infer.
struct plan_state {
  conn_t *conn;
  int gpus;
};

std::mutex plans_mutex;
std::unordered_map<cufftHandle, plan_state> plans;

conn_t *connection() { return lupine_cudart_connection(); }

conn_t *connection_for_handle(cufftHandle plan) {
  std::lock_guard<std::mutex> lock(plans_mutex);
  auto it = plans.find(plan);
  return it == plans.end() ? nullptr : it->second.conn;
}

void note_handle_owner(conn_t *conn, cufftHandle plan) {
  std::lock_guard<std::mutex> lock(plans_mutex);
  plans[plan] = {conn, 1};
}

int plan_gpu_count(cufftHandle plan) {
  std::lock_guard<std::mutex> lock(plans_mutex);
  auto it = plans.find(plan);
  return it == plans.end() ? 1 : it->second.gpus;
}

// ---------------------------------------------------------------------------
// Multi-GPU descriptors
// ---------------------------------------------------------------------------

// cufftXtMalloc hands out a descriptor the caller reads (the per-GPU data
// pointers and sizes), so the client keeps a copy of the server's and maps it
// back to the server's for the calls that take one. The data pointers are the
// server's device addresses, which the caller passes on unchanged.
struct descriptor_mirror {
  cudaLibXtDesc lib;
  cudaXtDesc desc;
  cudaLibXtDesc *remote;
  conn_t *conn;
};

std::mutex descriptors_mutex;
std::unordered_map<cudaLibXtDesc *, descriptor_mirror *> descriptors;

descriptor_mirror *mirror_of(cudaLibXtDesc *descriptor) {
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  auto it = descriptors.find(descriptor);
  return it == descriptors.end() ? nullptr : it->second;
}

cudaLibXtDesc *remote_descriptor(cudaLibXtDesc *descriptor) {
  descriptor_mirror *mirror = mirror_of(descriptor);
  return mirror == nullptr ? nullptr : mirror->remote;
}

size_t descriptor_bytes(const descriptor_mirror *mirror) {
  size_t bytes = 0;
  for (int i = 0; i < mirror->desc.nGPUs && i < MAX_CUDA_DESCRIPTOR_GPUS; ++i) {
    bytes += mirror->desc.size[i];
  }
  return bytes;
}

} // namespace

#include "codegen/gen_cufft_client.inc"

// ---------------------------------------------------------------------------
// Plan state the client mirrors
// ---------------------------------------------------------------------------

extern "C" cufftResult cufftDestroy(cufftHandle plan) {
  conn_t *conn = connection_for_handle(plan);
  cufftResult status = lupine_rpc_cufftDestroy(conn, plan);
  std::lock_guard<std::mutex> lock(plans_mutex);
  plans.erase(plan);
  return status;
}

// The caller names virtual device ordinals; the server's library wants its
// own. A plan cannot span servers, so each is rewritten as if on the plan's.
extern "C" cufftResult cufftXtSetGPUs(cufftHandle handle, int nGPUs,
                                      int *whichGPUs) {
  conn_t *conn = connection_for_handle(handle);
  std::vector<int> devices;
  if (whichGPUs != nullptr && nGPUs > 0) {
    devices.assign(whichGPUs, whichGPUs + nGPUs);
    for (int &device : devices) {
      lupine_rpc_conn_for_device(&device);
    }
    whichGPUs = devices.data();
  }
  cufftResult status =
      lupine_rpc_cufftXtSetGPUs(conn, handle, nGPUs, whichGPUs);
  if (status == CUFFT_SUCCESS) {
    std::lock_guard<std::mutex> lock(plans_mutex);
    auto it = plans.find(handle);
    if (it != plans.end()) {
      it->second.gpus = nGPUs;
    }
  }
  return status;
}

// ---------------------------------------------------------------------------
// Multi-GPU descriptors
// ---------------------------------------------------------------------------

extern "C" cufftResult cufftXtMalloc(cufftHandle plan,
                                     cudaLibXtDesc **descriptor,
                                     cufftXtSubFormat format) {
  conn_t *conn = connection_for_handle(plan);
  descriptor_mirror *mirror = new descriptor_mirror();
  cufftResult status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cufftXtMalloc) < 0 ||
      rpc_write(conn, &plan, sizeof(plan)) < 0 ||
      rpc_write(conn, &format, sizeof(format)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      (status == CUFFT_SUCCESS &&
       (rpc_read(conn, &mirror->remote, sizeof(mirror->remote)) < 0 ||
        rpc_read(conn, &mirror->lib, sizeof(mirror->lib)) < 0 ||
        rpc_read(conn, &mirror->desc, sizeof(mirror->desc)) < 0)) ||
      rpc_read_end(conn) < 0) {
    delete mirror;
    return rpc_error();
  }
  if (status != CUFFT_SUCCESS) {
    delete mirror;
    return status;
  }
  mirror->lib.descriptor = &mirror->desc;
  mirror->conn = conn;
  *descriptor = &mirror->lib;
  std::lock_guard<std::mutex> lock(descriptors_mutex);
  descriptors[&mirror->lib] = mirror;
  return CUFFT_SUCCESS;
}

extern "C" cufftResult cufftXtFree(cudaLibXtDesc *descriptor) {
  descriptor_mirror *mirror = mirror_of(descriptor);
  if (mirror == nullptr) {
    return CUFFT_INVALID_VALUE;
  }
  conn_t *conn = mirror->conn;
  cufftResult status = rpc_error();
  if (rpc_write_start_request(conn, RPC_cufftXtFree) < 0 ||
      rpc_write(conn, &mirror->remote, sizeof(mirror->remote)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (status == CUFFT_SUCCESS) {
    std::lock_guard<std::mutex> lock(descriptors_mutex);
    descriptors.erase(descriptor);
    delete mirror;
  }
  return status;
}

// The host side of a copy is the descriptor's data end to end, which the
// mirror sizes; the device sides travel as the server's descriptors.
extern "C" cufftResult cufftXtMemcpy(cufftHandle plan, void *dstPointer,
                                     void *srcPointer, cufftXtCopyType type) {
  conn_t *conn = connection_for_handle(plan);
  descriptor_mirror *dst = nullptr;
  descriptor_mirror *src = nullptr;
  size_t bytes = 0;
  switch (type) {
  case CUFFT_COPY_HOST_TO_DEVICE:
    dst = mirror_of(static_cast<cudaLibXtDesc *>(dstPointer));
    bytes = dst == nullptr ? 0 : descriptor_bytes(dst);
    break;
  case CUFFT_COPY_DEVICE_TO_HOST:
    src = mirror_of(static_cast<cudaLibXtDesc *>(srcPointer));
    bytes = src == nullptr ? 0 : descriptor_bytes(src);
    break;
  case CUFFT_COPY_DEVICE_TO_DEVICE:
    dst = mirror_of(static_cast<cudaLibXtDesc *>(dstPointer));
    src = mirror_of(static_cast<cudaLibXtDesc *>(srcPointer));
    break;
  default:
    return CUFFT_INVALID_VALUE;
  }
  cudaLibXtDesc *remote_dst = dst == nullptr ? nullptr : dst->remote;
  cudaLibXtDesc *remote_src = src == nullptr ? nullptr : src->remote;
  cufftResult status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cufftXtMemcpy) < 0 ||
      rpc_write(conn, &plan, sizeof(plan)) < 0 ||
      rpc_write(conn, &type, sizeof(type)) < 0 ||
      rpc_write(conn, &remote_dst, sizeof(remote_dst)) < 0 ||
      rpc_write(conn, &remote_src, sizeof(remote_src)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      (type == CUFFT_COPY_HOST_TO_DEVICE && bytes != 0 &&
       rpc_write(conn, srcPointer, bytes) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 ||
      (type == CUFFT_COPY_DEVICE_TO_HOST && status == CUFFT_SUCCESS &&
       bytes != 0 && rpc_read(conn, dstPointer, bytes) < 0) ||
      rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" cufftResult cufftXtExecDescriptor(cufftHandle plan,
                                             cudaLibXtDesc *input,
                                             cudaLibXtDesc *output,
                                             int direction) {
  return lupine_rpc_cufftXtExecDescriptor(connection_for_handle(plan), plan,
                                          remote_descriptor(input),
                                          remote_descriptor(output), direction);
}

extern "C" cufftResult cufftXtExecDescriptorC2C(cufftHandle plan,
                                                cudaLibXtDesc *input,
                                                cudaLibXtDesc *output,
                                                int direction) {
  return lupine_rpc_cufftXtExecDescriptorC2C(
      connection_for_handle(plan), plan, remote_descriptor(input),
      remote_descriptor(output), direction);
}

extern "C" cufftResult cufftXtExecDescriptorC2R(cufftHandle plan,
                                                cudaLibXtDesc *input,
                                                cudaLibXtDesc *output) {
  return lupine_rpc_cufftXtExecDescriptorC2R(connection_for_handle(plan), plan,
                                             remote_descriptor(input),
                                             remote_descriptor(output));
}

extern "C" cufftResult cufftXtExecDescriptorD2Z(cufftHandle plan,
                                                cudaLibXtDesc *input,
                                                cudaLibXtDesc *output) {
  return lupine_rpc_cufftXtExecDescriptorD2Z(connection_for_handle(plan), plan,
                                             remote_descriptor(input),
                                             remote_descriptor(output));
}

extern "C" cufftResult cufftXtExecDescriptorR2C(cufftHandle plan,
                                                cudaLibXtDesc *input,
                                                cudaLibXtDesc *output) {
  return lupine_rpc_cufftXtExecDescriptorR2C(connection_for_handle(plan), plan,
                                             remote_descriptor(input),
                                             remote_descriptor(output));
}

extern "C" cufftResult cufftXtExecDescriptorZ2D(cufftHandle plan,
                                                cudaLibXtDesc *input,
                                                cudaLibXtDesc *output) {
  return lupine_rpc_cufftXtExecDescriptorZ2D(connection_for_handle(plan), plan,
                                             remote_descriptor(input),
                                             remote_descriptor(output));
}

extern "C" cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan,
                                                cudaLibXtDesc *input,
                                                cudaLibXtDesc *output,
                                                int direction) {
  return lupine_rpc_cufftXtExecDescriptorZ2Z(
      connection_for_handle(plan), plan, remote_descriptor(input),
      remote_descriptor(output), direction);
}

// ---------------------------------------------------------------------------
// Arrays with one entry per GPU of the plan
// ---------------------------------------------------------------------------

namespace {

// A null array travels as a zero count; a present one as the plan's GPU count
// and that many pointers.
int write_pointer_array(conn_t *conn, void **pointers, int count) {
  const uint32_t n = pointers == nullptr ? 0 : static_cast<uint32_t>(count);
  if (rpc_write(conn, &n, sizeof(n)) < 0 ||
      (n != 0 && rpc_write(conn, pointers, n * sizeof(void *)) < 0)) {
    return -1;
  }
  return 0;
}

cufftResult set_jit_callback(int op, cufftHandle plan,
                             const char *lto_callback_symbol_name,
                             const void *lto_callback_fatbin,
                             size_t lto_callback_fatbin_size,
                             cufftXtCallbackType type, void **caller_info) {
  conn_t *conn = connection_for_handle(plan);
  const uint32_t name_length =
      lto_callback_symbol_name == nullptr
          ? 0
          : static_cast<uint32_t>(strlen(lto_callback_symbol_name));
  const uint64_t fatbin_size = lto_callback_fatbin_size;
  cufftResult status = rpc_error();
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_write(conn, &plan, sizeof(plan)) < 0 ||
      rpc_write(conn, &name_length, sizeof(name_length)) < 0 ||
      (name_length != 0 &&
       rpc_write(conn, lto_callback_symbol_name, name_length) < 0) ||
      rpc_write(conn, &fatbin_size, sizeof(fatbin_size)) < 0 ||
      (fatbin_size != 0 &&
       rpc_write(conn, lto_callback_fatbin, fatbin_size) < 0) ||
      rpc_write(conn, &type, sizeof(type)) < 0 ||
      write_pointer_array(conn, caller_info, plan_gpu_count(plan)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

} // namespace

extern "C" cufftResult cufftXtSetWorkArea(cufftHandle plan, void **workArea) {
  conn_t *conn = connection_for_handle(plan);
  cufftResult status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cufftXtSetWorkArea) < 0 ||
      rpc_write(conn, &plan, sizeof(plan)) < 0 ||
      write_pointer_array(conn, workArea, plan_gpu_count(plan)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

extern "C" cufftResult cufftXtSetCallback(cufftHandle plan,
                                          void **callback_routine,
                                          cufftXtCallbackType cbType,
                                          void **caller_info) {
  conn_t *conn = connection_for_handle(plan);
  const int gpus = plan_gpu_count(plan);
  cufftResult status = rpc_error();
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cufftXtSetCallback) < 0 ||
      rpc_write(conn, &plan, sizeof(plan)) < 0 ||
      rpc_write(conn, &cbType, sizeof(cbType)) < 0 ||
      write_pointer_array(conn, callback_routine, gpus) < 0 ||
      write_pointer_array(conn, caller_info, gpus) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return status;
}

#if CUFFT_VERSION >= 12000
extern "C" cufftResult
cufftXtSetJITCallback(cufftHandle plan, const char *lto_callback_symbol_name,
                      const void *lto_callback_fatbin,
                      size_t lto_callback_fatbin_size, cufftXtCallbackType type,
                      void **caller_info) {
  return set_jit_callback(RPC_cufftXtSetJITCallback, plan,
                          lto_callback_symbol_name, lto_callback_fatbin,
                          lto_callback_fatbin_size, type, caller_info);
}
#endif

#if CUFFT_VERSION >= 11300 && CUFFT_VERSION < 12000
extern "C" cufftResult __cufftXtSetJITCallback_12_7(
    cufftHandle plan, const char *lto_callback_symbol_name,
    const void *lto_callback_fatbin, size_t lto_callback_fatbin_size,
    cufftXtCallbackType type, void **caller_info) {
  return set_jit_callback(RPC___cufftXtSetJITCallback_12_7, plan,
                          lto_callback_symbol_name, lto_callback_fatbin,
                          lto_callback_fatbin_size, type, caller_info);
}
#endif
