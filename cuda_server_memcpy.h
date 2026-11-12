#ifndef LUPINE_CUDA_SERVER_MEMCPY_H
#define LUPINE_CUDA_SERVER_MEMCPY_H

#include "device_stdout.h"

#include <cstdint>
#include <cuda.h>

#include <memory>
#include <string>
#include <vector>

#include "cuda_compat.h"
#include "rpc.h"

using lupine_capture_id = uint64_t;

// Device-to-host copies into client memory are pushed to the client as their
// stream runs them. A host function queued behind each copy snapshots its
// bytes and numbers it; the connection's push lane sends deliveries in that
// order. Every answer and server callback carries the last number assigned
// when it was produced, and the client applies deliveries up to that number
// before acting on it, so nothing has to guess which copies an answer covers.
uint64_t lupine_dtoh_watermark(conn_t *conn);
void lupine_dtoh_sync_contexts(conn_t *conn);
int handle_lupineDtohPush(conn_t *conn);
void lupine_keep_async_error(conn_t *conn, CUresult result);
// True when the client does not wait for this request's answer, which then
// gets none: a failure is kept for its next synchronize.
bool lupine_unanswered(conn_t *conn, CUresult result);
// Whether a stream query may answer not ready without asking the driver,
// because work sent before it is not submitted yet. A capturing stream, or one
// whose capture state cannot be read, still asks: the driver's answer there is
// an error, and invalidates the capture.
bool lupine_stream_query_unsettled(conn_t *conn, CUstream stream);
CUresult lupine_take_async_error(conn_t *conn, CUresult result);
lupine_capture_id lupine_begin_stream_capture_resources(CUstream stream);
void lupine_discard_stream_capture_resources(lupine_capture_id capture_id);
void lupine_finish_stream_capture_resources(CUstream stream, CUgraph graph,
                                            bool success);
void lupine_record_event_capture_resources(CUevent event, CUstream stream);
void lupine_forget_event_capture_resources(CUevent event);
void lupine_wait_event_capture_resources(CUstream stream, CUevent event);
void lupine_clone_graph_resources(CUgraph clone, CUgraph original);
void lupine_erase_graph_resources(CUgraph graph);
// Applies backpressure to a handler about to queue device-to-host work.
void lupine_dtoh_wait_for_room(conn_t *conn);
// Replaces each dependency on a graph device-to-host copy with the node that
// pushes its bytes, so later nodes see them. Every handler that takes graph
// dependencies, generated or not, passes them through this.
void lupine_graph_push_dependencies(CUgraphNode *deps, size_t count);
struct lupine_htod_graph_binding {
  CUgraph original = nullptr;
  CUgraph prepared = nullptr;
  std::shared_ptr<void> resources;
};
CUresult
lupine_prepare_graph_exec_resources(CUgraph graph,
                                    lupine_htod_graph_binding *binding);
CUresult
lupine_associate_graph_exec_resources(CUgraphExec exec,
                                      const lupine_htod_graph_binding &binding);
CUgraphNode
lupine_original_htod_graph_node(const lupine_htod_graph_binding &binding,
                                CUgraphNode node);
CUgraphNode lupine_htod_graph_exec_node(CUgraphExec exec, CUgraphNode node);
void lupine_release_htod_graph_binding(lupine_htod_graph_binding *binding);
CUresult lupine_release_graph_exec_resources(CUgraphExec exec);
CUresult lupine_prepare_graph_exec_update(CUgraphExec exec, CUgraph graph,
                                          lupine_htod_graph_binding *binding);
void lupine_finish_graph_exec_update(CUgraphExec exec, CUgraph graph,
                                     lupine_htod_graph_binding *binding,
                                     CUresult result, CUgraphNode *error_node);
void lupine_finish_graph_exec_update(CUgraphExec exec, CUgraph graph,
                                     lupine_htod_graph_binding *binding,
                                     CUresult result,
                                     CUgraphExecUpdateResultInfo *info);
// Adds a node pushing an explicit graph copy's bytes after `copy`.
CUresult lupine_graph_add_dtoh_push(conn_t *conn, CUgraph graph,
                                    CUgraphNode copy,
                                    const lupine_host_rows &client,
                                    void *server_src);

// Where a copy's host destination rows sit in client memory.
inline lupine_host_rows
lupine_host_destination_rows(const CUDA_MEMCPY3D &copy) {
  size_t slice = copy.dstHeight * copy.dstPitch;
  return {static_cast<unsigned char *>(copy.dstHost) + copy.dstZ * slice +
              copy.dstY * copy.dstPitch + copy.dstXInBytes,
          copy.WidthInBytes,
          copy.Height,
          copy.dstPitch,
          copy.Depth,
          slice};
}

// A host side staged on the server is dense: the client sends and receives
// only the copied rows and keeps its own pitch and offsets.
inline size_t lupine_pack_host_destination(CUDA_MEMCPY3D &copy) {
  copy.dstXInBytes = 0;
  copy.dstY = 0;
  copy.dstZ = 0;
  copy.dstPitch = copy.WidthInBytes;
  copy.dstHeight = copy.Height;
  return copy.WidthInBytes * copy.Height * copy.Depth;
}
inline size_t lupine_pack_host_source(CUDA_MEMCPY3D &copy) {
  copy.srcXInBytes = 0;
  copy.srcY = 0;
  copy.srcZ = 0;
  copy.srcPitch = copy.WidthInBytes;
  copy.srcHeight = copy.Height;
  return copy.WidthInBytes * copy.Height * copy.Depth;
}

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
CUresult lupine_server_prepare_htod_capture(conn_t *conn);
int lupine_write_lifecycle_response(conn_t *conn, int request_id,
                                    CUresult result);

#endif
