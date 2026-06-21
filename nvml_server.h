#ifndef LUPINE_NVML_SERVER_H
#define LUPINE_NVML_SERVER_H

#include "rpc.h"

#include "codegen/gen_nvml_server.h"

int handle_nvmlInit_v2(conn_t *conn);
int handle_nvmlInitWithFlags(conn_t *conn);
int handle_nvmlShutdown(conn_t *conn);
int handle_nvmlDeviceGetCount_v2(conn_t *conn);
int handle_nvmlDeviceGetHandleByIndex_v2(conn_t *conn);
int handle_nvmlDeviceGetName(conn_t *conn);
int handle_nvmlDeviceGetIndex(conn_t *conn);
int handle_nvmlDeviceGetComputeRunningProcesses(conn_t *conn);
int handle_nvmlDeviceGetComputeRunningProcesses_v2(conn_t *conn);
int handle_nvmlDeviceGetGraphicsRunningProcesses(conn_t *conn);
int handle_nvmlDeviceGetGraphicsRunningProcesses_v2(conn_t *conn);
int handle_nvmlDeviceGetMPSComputeRunningProcesses(conn_t *conn);
int handle_nvmlDeviceGetMPSComputeRunningProcesses_v2(conn_t *conn);
int handle_nvmlEventSetCreate(conn_t *conn);
int handle_nvmlEventSetFree(conn_t *conn);
int handle_nvmlEventSetWait_v2(conn_t *conn);
int handle_nvmlDeviceRegisterEvents(conn_t *conn);

#endif
