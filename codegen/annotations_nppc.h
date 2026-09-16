#include <npp.h>

// NPP core (libnppc). The client forwards each call over the CUDA driver shim's
// connections like every NPP library; the server handlers resolve the
// machine's libnppc by name at runtime. A core call goes to the runtime's
// current device, except the global stream calls: nppSetStream records the
// stream it sets, and nppGetStream, nppGetStreamContext and the stream thread
// counts go to the server that owns that stream. nppGetStreamContext names
// the caller's current device, not the server's. nppGetLibVersion and
// nppGetGpuName return memory the library owns, so the client keeps a copy of
// each for the life of the process, the name one per device.
//
// NPP 13 exports the stream calls without declaring them, so they are declared
// here as NPP 12 declares them. nppGetPointerMemoryType is exported but never
// declared, and is left out.

/**
 * @disabled
 */
const NppLibraryVersion *nppGetLibVersion();
int nppGetGpuNumSMs();
int nppGetMaxThreadsPerBlock();
int nppGetMaxThreadsPerSM();
/**
 * @param pMaxThreadsPerSM SEND_RECV
 * @param pMaxThreadsPerBlock SEND_RECV
 * @param pNumberOfSMs SEND_RECV
 */
int nppGetGpuDeviceProperties(int *pMaxThreadsPerSM, int *pMaxThreadsPerBlock,
                              int *pNumberOfSMs);
/**
 * @disabled
 */
const char *nppGetGpuName();
/**
 * @disabled client
 */
cudaStream_t nppGetStream();
/**
 * @disabled client
 * @param pNppStreamContext SEND_RECV
 */
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
/**
 * @disabled client
 */
unsigned int nppGetStreamNumSMs();
/**
 * @disabled client
 */
unsigned int nppGetStreamMaxThreadsPerSM();
/**
 * @disabled client
 * @param hStream SEND_ONLY
 */
NppStatus nppSetStream(cudaStream_t hStream);
