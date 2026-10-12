#ifndef LUPINE_NVML_SERVER_H
#define LUPINE_NVML_SERVER_H

#include "rpc.h"

#include "codegen/gen_nvml_server.h"

int handle_nvmlDeviceGetComputeRunningProcesses(conn_t *conn);
int handle_nvmlDeviceGetComputeRunningProcesses_v2(conn_t *conn);
int handle_nvmlDeviceGetGraphicsRunningProcesses(conn_t *conn);
int handle_nvmlDeviceGetGraphicsRunningProcesses_v2(conn_t *conn);
int handle_nvmlDeviceGetMPSComputeRunningProcesses(conn_t *conn);
int handle_nvmlDeviceGetMPSComputeRunningProcesses_v2(conn_t *conn);

#endif
