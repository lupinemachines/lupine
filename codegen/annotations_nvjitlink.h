#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>

// nvJitLink. The client forwards each call over the CUDA driver shim's
// connections, so a link runs on the server that loads its output: the linked
// code matches that machine's toolkit and driver. The server handlers resolve
// the machine's libnvJitLink by name at runtime. A link handle is created on
// the runtime's current device and routes to the connection that created it.
//
// nvJitLink copies the options and every input it is given during the call,
// so nothing outlives a handler. A file named to nvJitLinkAddFile is on the
// client, so the client reads it and adds its bytes under the same name. The
// calls hand-written in nvjitlink_client.cpp and nvjitlink_server.cpp carry
// string arrays or nullable names, or copy out an output whose size an earlier
// call reported. The release-suffixed names consumers built against the
// header reference (__nvJitLinkCreate_13_3 and its siblings) are defined in
// nvjitlink_client.cpp.
//
// The linked LTO IR calls are guarded by the header that declares them.

/**
 * @disabled
 */
nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions,
                                const char **options);
/**
 * @disabled client
 * @param handle SEND_RECV DEREF
 */
nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle *handle);
/**
 * @disabled
 */
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle,
                                 nvJitLinkInputType inputType, const void *data,
                                 size_t size, const char *name);
/**
 * @disabled
 */
nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle handle,
                                 nvJitLinkInputType inputType,
                                 const char *fileName);
/**
 * @param handle SEND_ONLY
 */
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle);
/**
 * @param handle SEND_ONLY
 * @param size RECV_ONLY
 */
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle,
                                            size_t *size);
/**
 * @disabled
 */
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin);
/**
 * @param handle SEND_ONLY
 * @param size RECV_ONLY
 */
nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle handle, size_t *size);
/**
 * @disabled
 */
nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle handle, char *ptx);
/**
 * @guard LUPINE_NVJITLINK_HAS_LTOIR
 * @param handle SEND_ONLY
 * @param size RECV_ONLY
 */
nvJitLinkResult nvJitLinkGetLinkedLTOIRSize(nvJitLinkHandle handle,
                                            size_t *size);
/**
 * @disabled
 * @guard LUPINE_NVJITLINK_HAS_LTOIR
 */
nvJitLinkResult nvJitLinkGetLinkedLTOIR(nvJitLinkHandle handle, void *ltoir);
/**
 * @param handle SEND_ONLY
 * @param size RECV_ONLY
 */
nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle handle, size_t *size);
/**
 * @disabled
 */
nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle handle, char *log);
/**
 * @param handle SEND_ONLY
 * @param size RECV_ONLY
 */
nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle handle, size_t *size);
/**
 * @disabled
 */
nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle handle, char *log);
/**
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 */
nvJitLinkResult nvJitLinkVersion(unsigned int *major, unsigned int *minor);
