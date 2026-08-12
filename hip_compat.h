#ifndef LUPINE_HIP_COMPAT_H
#define LUPINE_HIP_COMPAT_H

// HIP version-compat shims, analogous to cuda_compat.h. The generated HIP
// client/server units pull in the real HIP runtime through this header so
// type definitions (hipDeviceProp_t, hipUUID, hipDeviceAttribute_t, ...) are
// available on whatever ROCm/HIP SDK the build runs against.
//
// Unlike CUDA, the v1 HIP surface only marshals whole structs/opaque values
// (it never reaches into struct members from generated code), so no field
// shims are needed yet. Add them here as compute-path functions land.
//
// hip_common.h only auto-selects __HIP_PLATFORM_AMD__ under hip-clang; its own
// comment notes that other host compilers (MSVC, GCC, ...) must define one of
// the platform macros explicitly or hip_runtime.h #errors out. Every TU that
// needs the HIP types goes through this header, so set it once here.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime.h>

#endif