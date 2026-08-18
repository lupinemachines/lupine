#ifndef GEN_SERVER_H
#define GEN_SERVER_H

#include "rpc_server.h"

#include <stddef.h>

rpc_handler get_cuda_handler(int op);
const int *get_cuda_handler_operations(size_t *count);
rpc_handler get_nvml_handler(int op);
const int *get_nvml_handler_operations(size_t *count);

#endif
