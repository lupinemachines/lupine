#ifndef LUPINE_CUDA_SERVER_MEMCPY_H
#define LUPINE_CUDA_SERVER_MEMCPY_H

#include <cuda.h>

#include <vector>

#include "rpc.h"
#include "third_party/libcuckoo/libcuckoo/cuckoohash_map.hh"

struct lupine_staging_state;
struct lupine_graph_resources;

// A device-to-host copy the server holds until a synchronize collects it. The
// copy handlers here produce these; the stream, event and context synchronize
// handlers in cuda_server.cpp drain them, so the registry is shared rather than
// duplicated.
struct lupine_pending_dtoh_item {
  CUevent event = nullptr;
  void *client_dst = nullptr;
  void *server_src = nullptr;
  size_t bytes = 0;
  bool pinned = false;
};

using lupine_pending_dtoh_items = std::vector<lupine_pending_dtoh_item>;
using lupine_pending_dtoh_streams =
    std::unordered_map<CUstream, lupine_pending_dtoh_items>;

libcuckoo::cuckoohash_map<conn_t *, lupine_pending_dtoh_streams> &
lupine_pending_dtoh_copies();
lupine_graph_resources *lupine_get_stream_resources(CUstream stream);
void *lupine_alloc_capture_scratch(lupine_graph_resources *resources,
                                   size_t bytes);
void lupine_graph_note_dtoh_copy(lupine_graph_resources *resources,
                                 void *client_dst, void *server_src,
                                 size_t bytes);

// Frees a pinned staging block once the stream that is still reading from it
// drains. Only the Windows host-to-device path defers a free this way.
CUresult lupine_defer_host_free(CUstream stream, void *ptr);

// Staging bookkeeping shared with the context handlers in cuda_server.cpp: a
// context cannot go away until the staging buffers it owns have been retired.
void lupine_server_begin_lifecycle_transaction(conn_t *conn);
void lupine_server_end_lifecycle_transaction(conn_t *conn);
void lupine_server_finish_context_destroy(conn_t *conn, CUcontext context,
                                          CUresult result);
void lupine_server_finish_context_detach(conn_t *conn, CUcontext context,
                                         CUresult result);
void lupine_server_finish_primary_context(conn_t *conn, CUdevice device,
                                          bool reset, CUresult result);
void lupine_server_note_created_context(conn_t *conn, CUcontext context,
                                        CUresult result);
void lupine_server_note_primary_context(conn_t *conn, CUdevice device,
                                        CUcontext context, CUresult result);
void lupine_server_prepare_context_destroy(conn_t *conn, CUcontext context);
void lupine_server_prepare_primary_context(conn_t *conn, CUdevice device);
int lupine_write_lifecycle_response(conn_t *conn, int request_id,
                                    CUresult result);
int lupine_copy_htod_serial(conn_t *conn, CUdeviceptr destination, size_t bytes,
                            lupine_staging_state &state, CUresult *result);

#endif
