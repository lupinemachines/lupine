#ifndef GEN_SERVER_H
#define GEN_SERVER_H

#include "rpc_server.h"

const rpc_handler_registry &get_cuda_handlers();
const rpc_handler_registry &get_nvml_handlers();

#endif
