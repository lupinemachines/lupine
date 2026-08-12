#ifndef LUPINE_HIP_SERVER_H
#define LUPINE_HIP_SERVER_H

#include "rpc.h"

#include "codegen/gen_hip_server.h"

// Hand-written HIP server handler prototypes go here as compute-path
// functions (hipMalloc/hipMemcpy/hipModuleLaunchKernel/...) are added in
// follow-up PRs. v1 only marshals device enumeration/properties, which the
// codegen emits into gen_hip_server.inc.

#endif