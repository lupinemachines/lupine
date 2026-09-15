#include <cuda.h>
#include <nvrtc.h>

// NVRTC. The client forwards each call over the CUDA driver shim's
// connections, so a program compiles on the server that loads its output: the
// generated code matches that machine's toolkit and driver. The server
// handlers resolve the machine's libnvrtc by name at runtime. A program is a
// handle the server's library hands out and routes to the connection that
// created it; a call without one goes to the runtime's current device.
//
// NVRTC copies every string it is given during the call (source, name,
// headers, include names, options, name expressions), so nothing outlives a
// handler. The files a program includes are on the client, though, so the
// client may recreate the server's program with them as headers before it
// compiles; a program handle is therefore the client's, and each call that
// takes one substitutes the server's current program. The calls hand-written in
// nvrtc_client.cpp and nvrtc_server.cpp carry string arrays, return an output
// whose size an earlier call reported, or return strings the library owns: a
// lowered name lives until its program is destroyed, an error string for the
// life of the process, and a bundled headers error log until the next call on
// the thread.
//
// Guards follow the headers each toolkit ships: LTO IR and OptiX IR from
// 12.0, precompiled headers and the flow callback from 12.8, NVVM bitcode up
// to 12.9, Tile IR from 13.2 and bundled headers from 13.3. The NVVM calls are
// declared here because the newest header no longer has them.

/**
 * @disabled
 */
const char *nvrtcGetErrorString(nvrtcResult result);
/**
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 */
nvrtcResult nvrtcVersion(int *major, int *minor);
/**
 * @param numArchs RECV_ONLY
 */
nvrtcResult nvrtcGetNumSupportedArchs(int *numArchs);
/**
 * @disabled
 */
nvrtcResult nvrtcGetSupportedArchs(int *supportedArchs);
/**
 * @disabled
 */
nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog, const char *src,
                               const char *name, int numHeaders,
                               const char *const *headers,
                               const char *const *includeNames);
/**
 * @disabled client
 * @param prog SEND_RECV DEREF
 */
nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);
/**
 * @disabled
 */
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions,
                                const char *const *options);
/**
 * @param prog SEND_ONLY
 * @param ptxSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);
/**
 * @param prog SEND_ONLY
 * @param cubinSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin);
/**
 * @guard CUDA_VERSION >= 12000
 * @param prog SEND_ONLY
 * @param LTOIRSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t *LTOIRSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 * @guard CUDA_VERSION >= 12000
 */
nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *LTOIR);
/**
 * @guard CUDA_VERSION >= 12000
 * @param prog SEND_ONLY
 * @param optixirSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t *optixirSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 * @guard CUDA_VERSION >= 12000
 */
nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char *optixir);
/**
 * @guard CUDA_VERSION < 13000
 * @param prog SEND_ONLY
 * @param nvvmSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t *nvvmSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 * @guard CUDA_VERSION < 13000
 */
nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm);
/**
 * @param prog SEND_ONLY
 * @param logSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);
/**
 * @param prog SEND_ONLY
 * @param name_expression SEND_ONLY NULL_TERMINATED
 */
// clang-format off
nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char *name_expression) {
  const nvrtcProgram program = prog;
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  if (return_value == NVRTC_SUCCESS) {
    note_name_expression(program, name_expression);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled
 */
nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                const char *const name_expression,
                                const char **lowered_name);
/**
 * @guard CUDA_VERSION >= 12080
 * @param ret RECV_ONLY
 */
nvrtcResult nvrtcGetPCHHeapSize(size_t *ret);
/**
 * @guard CUDA_VERSION >= 12080
 * @param size SEND_ONLY
 */
nvrtcResult nvrtcSetPCHHeapSize(size_t size);
/**
 * @guard CUDA_VERSION >= 12080
 * @param prog SEND_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @guard CUDA_VERSION >= 12080
 * @param prog SEND_ONLY
 * @param size RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t *size) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
// The callback is a client function the server's compiler cannot call.
/**
 * @guard CUDA_VERSION >= 12080
 */
nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog,
                                 int (*callback)(void *, void *),
                                 void *payload);
/**
 * @guard CUDA_VERSION >= 13020
 * @param prog SEND_ONLY
 * @param TileIRSizeRet RECV_ONLY
 */
// clang-format off
nvrtcResult nvrtcGetTileIRSize(nvrtcProgram prog, size_t *TileIRSizeRet) {
  prog = server_program(prog);
  nvrtcResult return_value = LUPINE_GENERATED_CALL();
  return return_value;
}
// clang-format on
/**
 * @disabled
 * @guard CUDA_VERSION >= 13020
 */
nvrtcResult nvrtcGetTileIR(nvrtcProgram prog, char *TileIR);
/**
 * @disabled
 * @guard CUDA_VERSION >= 13030
 */
nvrtcResult nvrtcInstallBundledHeaders(const char *installPath,
                                       unsigned int flags,
                                       const char **errorLog);
/**
 * @disabled
 * @guard CUDA_VERSION >= 13030
 */
nvrtcResult nvrtcGetBundledHeadersInfo(nvrtcBundledHeadersInfo *info,
                                       const char **errorLog);
/**
 * @disabled
 * @guard CUDA_VERSION >= 13030
 */
nvrtcResult nvrtcRemoveBundledHeaders(const char *installPath,
                                      const char **errorLog);
