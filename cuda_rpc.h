#ifndef LUPINE_CUDA_RPC_H
#define LUPINE_CUDA_RPC_H

#include "cuda_compat.h"
#include "rpc.h"

#include <vector>

// Events the explicit lupineEventQueryBatch cache-warming RPC may carry. Its
// request is a count followed by that many handles, and its response is one
// CUresult per handle in the same order.
#define LUPINE_EVENT_QUERY_BATCH_MAX 16

// Wire layout for LUPINE_RPC_lupineDeviceSnapshot. The response is all or
// nothing: a non-success result carries no payload, otherwise every device
// record holds a fixed-size name buffer, uuid, total memory, and a
// count-prefixed list of (attribute, value) pairs.
#define LUPINE_DEVICE_SNAPSHOT_NAME_BYTES 256

struct rpc_jit_output_binding {
  CUjit_option option;
  void *dst;
  size_t size;
};

extern int
rpc_read_jit_outputs(conn_t *conn,
                     const std::vector<rpc_jit_output_binding> &bindings);

#endif
