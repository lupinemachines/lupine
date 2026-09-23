#ifndef LUPINE_CUDA_CLIENT_EPOCHS_H
#define LUPINE_CUDA_CLIENT_EPOCHS_H

#include "rpc_epochs.h"
#include <cuda.h>

rpc_epoch_call lupine_cuda_stream_call(CUstream stream, CUevent event = nullptr,
                                       bool record_event = false,
                                       conn_t *connection = nullptr);
rpc_epoch_call
lupine_cuda_context_call(CUcontext context = nullptr, bool all_contexts = false,
                         conn_t *connection = nullptr,
                         const std::vector<CUevent> &events = {});
rpc_epoch_call lupine_cuda_event_call(const std::vector<CUevent> &events,
                                      conn_t *connection = nullptr);
void lupine_cuda_stream_created(CUstream stream, conn_t *conn,
                                CUcontext context, unsigned int flags);
void lupine_cuda_epochs_forget_connection(conn_t *conn);

#endif
