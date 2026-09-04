/*
 * Minimal host-side expansion of the nvcc-internal crt/host_defines.h macros.
 *
 * The redistributable CUDA header wheels (nvidia-cuda-runtime) ship
 * cuda_runtime_api.h and its dependencies except for the compiler-internal
 * crt/ headers, which only exist inside a full CUDA toolkit install. Host
 * code consuming the runtime API never needs the device expansions, so this
 * file provides the host-only definitions that cuda_runtime_api.h relies on:
 * qualifiers expand to nothing, CUDARTAPI to the platform calling convention,
 * and __dv to ordinary default arguments.
 *
 * Builds with a real toolkit (server images, Windows CI) never see this file;
 * it is only placed as crt/host_defines.h by the hermetic client-only builds.
 */

#ifndef LUPINE_CUDART_HOST_DEFINES_H
#define LUPINE_CUDART_HOST_DEFINES_H

#if defined(__CUDACC__) || defined(__CUDADEVRT_INTERNAL__)
#error "This stub is for host-only builds; use the real toolkit headers with nvcc."
#endif

#define __host__
#define __device__
#define __device_builtin__
#define __device_builtin_runtime__
#if defined(_MSC_VER)
#define __align__(n) __declspec(align(n))
#define __builtin_align__(n) __declspec(align(n))
#else
#define __align__(n) __attribute__((aligned(n)))
#define __builtin_align__(n) __attribute__((aligned(n)))
#endif
#define __global__
#define __forceinline__ inline
#define __no_return__ [[noreturn]]
#define __annotate__(a)
#define __location__(a)
#define __launch_bounds__(...)
#if defined(_MSC_VER)
#define __restrict__ __restrict
#define __inline__ __inline
#else
#define __inline__ inline
#endif
#define __thread__ __thread
#define __import__
#define __export__
#define __cdecl

#if defined(_MSC_VER) && defined(_M_IX86)
#define CUDARTAPI __stdcall
#else
#define CUDARTAPI
#endif
#define CUDARTAPI_CDECL

#define __dv(x) = x
#define __cudart_builtin__
#define __cudart_builtin_device__

#endif /* LUPINE_CUDART_HOST_DEFINES_H */
