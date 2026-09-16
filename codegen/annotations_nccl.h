#include <nccl.h>

// NCCL API. The client forwards each call over the CUDA driver shim's
// connections, so the server child that owns the driver and runtime state
// runs the library too; the server handlers resolve the machine's libnccl by
// name at runtime. A communicator or parameter handle is a pointer the
// server's library hands out; it routes to the connection that created it,
// and a call without one goes to the runtime's current device.
//
// Buffers, streams and windows are server values and travel as addresses. A
// configuration travels as the caller's bytes with its strings (`VERSIONED`),
// so a caller built against another NCCL release still fits.
//
// Collectives and point-to-point calls are `@async`: inside a group NCCL only
// queues them until ncclGroupEnd, so the client submits them fire-and-forget;
// outside a group NCCL may block until a peer joins, so they are ordinary
// requests. The client sends ncclGroupStart to a server the first time a group
// reaches it and ncclGroupEnd when the outermost group ends.
//
// Hand-written: the static and last-error strings, the parameter strings, and
// the client sides that route or rewrite what the generated marshalling
// cannot (group brackets, device lists, ncclMemFree).
//
// The build requires NCCL 2.14.3; calls that arrived later carry the matching
// @guard.

#if NCCL_VERSION_CODE >= 21903
/**
 * @guard NCCL_VERSION_CODE >= 21903
 * @param ptr RECV_ONLY
 * @param size SEND_ONLY
 */
// clang-format off
ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == ncclSuccess) {
    note_allocation(conn, *ptr, size);
  }
  return return_value;
}
// clang-format on
/**
 * @guard NCCL_VERSION_CODE >= 21903
 * @disabled client - routes by the allocation's owner
 * @param ptr SEND_ONLY
 */
ncclResult_t ncclMemFree(void *ptr);
#endif
/**
 * @param version RECV_ONLY
 */
ncclResult_t ncclGetVersion(int *version);
/**
 * @param uniqueId RECV_ONLY
 */
ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId);
// clang-format off
/**
 * @param comm RECV_ONLY
 * @param nranks SEND_ONLY
 * @param commId SEND_ONLY
 * @param rank SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:netName,commName MEMBERGUARD:netName=NCCL_VERSION_CODE>=21701 MEMBERGUARD:commName=NCCL_VERSION_CODE>=22703
 */
ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t *config) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == ncclSuccess || return_value == ncclInProgress) {
    note_handle_owner(conn, *comm);
  }
  return return_value;
}
// clang-format on
/**
 * @param comm RECV_ONLY
 * @param nranks SEND_ONLY
 * @param commId SEND_ONLY
 * @param rank SEND_ONLY
 */
// clang-format off
ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == ncclSuccess || return_value == ncclInProgress) {
    note_handle_owner(conn, *comm);
  }
  return return_value;
}
// clang-format on
/**
 * @disabled client - maps each device to its server
 * @param ndev SEND_ONLY
 * @param comm RECV_ONLY LENGTH:ndev
 * @param devlist SEND_ONLY NULLABLE LENGTH:ndev
 */
ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist);
/**
 * @param comm SEND_ONLY
 */
ncclResult_t ncclCommFinalize(ncclComm_t comm);
/**
 * @param comm SEND_ONLY
 */
// clang-format off
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(comm);
  return return_value;
}
// clang-format on
/**
 * @param comm SEND_ONLY
 */
// clang-format off
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  forget_handle(comm);
  return return_value;
}
// clang-format on
#if NCCL_VERSION_CODE >= 22807
/**
 * @guard NCCL_VERSION_CODE >= 22807
 * @param comm SEND_ONLY
 * @param revokeFlags SEND_ONLY
 */
ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags);
#endif
#if NCCL_VERSION_CODE >= 21801
// clang-format off
/**
 * @guard NCCL_VERSION_CODE >= 21801
 * @param comm SEND_ONLY
 * @param color SEND_ONLY
 * @param key SEND_ONLY
 * @param newcomm RECV_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:netName,commName MEMBERGUARD:netName=NCCL_VERSION_CODE>=21701 MEMBERGUARD:commName=NCCL_VERSION_CODE>=22703
 */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if ((return_value == ncclSuccess || return_value == ncclInProgress) && *newcomm != nullptr) {
    note_handle_owner(conn, *newcomm);
  }
  return return_value;
}
// clang-format on
#endif
#if NCCL_VERSION_CODE >= 22703
// clang-format off
/**
 * @guard NCCL_VERSION_CODE >= 22703
 * @param comm SEND_ONLY
 * @param excludeRanksCount SEND_ONLY
 * @param excludeRanksList SEND_ONLY LENGTH:excludeRanksCount
 * @param newcomm RECV_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:netName,commName MEMBERGUARD:netName=NCCL_VERSION_CODE>=21701 MEMBERGUARD:commName=NCCL_VERSION_CODE>=22703
 * @param shrinkFlags SEND_ONLY
 */
ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t *config, int shrinkFlags) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if ((return_value == ncclSuccess || return_value == ncclInProgress) && *newcomm != nullptr) {
    note_handle_owner(conn, *newcomm);
  }
  return return_value;
}
// clang-format on
#endif
#if NCCL_VERSION_CODE >= 22902
/**
 * @guard NCCL_VERSION_CODE >= 22902
 * @param comm SEND_ONLY
 * @param uniqueId RECV_ONLY
 */
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId);
// clang-format off
/**
 * @guard NCCL_VERSION_CODE >= 22902
 * @param comm SEND_ONLY
 * @param nRanks SEND_ONLY
 * @param uniqueId SEND_ONLY NULLABLE
 * @param rank SEND_ONLY
 * @param newcomm RECV_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:netName,commName MEMBERGUARD:netName=NCCL_VERSION_CODE>=21701 MEMBERGUARD:commName=NCCL_VERSION_CODE>=22703
 */
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank, ncclComm_t *newcomm, ncclConfig_t *config) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if ((return_value == ncclSuccess || return_value == ncclInProgress) && *newcomm != nullptr) {
    note_handle_owner(conn, *newcomm);
  }
  return return_value;
}
// clang-format on
#endif
#if NCCL_VERSION_CODE >= 22304
// clang-format off
/**
 * @guard NCCL_VERSION_CODE >= 22304
 * @param newcomm RECV_ONLY
 * @param nranks SEND_ONLY
 * @param myrank SEND_ONLY
 * @param nId SEND_ONLY
 * @param commIds SEND_ONLY LENGTH:nId
 * @param config SEND_ONLY VERSIONED STRINGS:netName,commName MEMBERGUARD:netName=NCCL_VERSION_CODE>=21701 MEMBERGUARD:commName=NCCL_VERSION_CODE>=22703
 */
ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks, int myrank, int nId, ncclUniqueId *commIds, ncclConfig_t *config) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == ncclSuccess || return_value == ncclInProgress) {
    note_handle_owner(conn, *newcomm);
  }
  return return_value;
}
// clang-format on
#endif
/**
 * @disabled
 */
const char *ncclGetErrorString(ncclResult_t result);
/**
 * @disabled
 */
const char *ncclGetLastError(ncclComm_t comm);
#if NCCL_VERSION_CODE >= 22403
/**
 * @guard NCCL_VERSION_CODE >= 22403
 */
void ncclResetDebugInit();
#endif
/**
 * @param comm SEND_ONLY
 * @param asyncError RECV_ONLY
 */
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
/**
 * @param comm SEND_ONLY
 * @param count RECV_ONLY
 */
ncclResult_t ncclCommCount(const ncclComm_t comm, int *count);
/**
 * @param comm SEND_ONLY
 * @param device RECV_ONLY
 */
// clang-format off
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == ncclSuccess) {
    *device = lupine_local_device_for_remote(conn, *device);
  }
  return return_value;
}
// clang-format on
/**
 * @param comm SEND_ONLY
 * @param rank RECV_ONLY
 */
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank);
#if NCCL_VERSION_CODE >= 21903
/**
 * @guard NCCL_VERSION_CODE >= 21903
 * @param comm SEND_ONLY
 * @param buff SEND_ONLY
 * @param size SEND_ONLY
 * @param handle RECV_ONLY
 */
ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size,
                              void **handle);
/**
 * @guard NCCL_VERSION_CODE >= 21903
 * @param comm SEND_ONLY
 * @param handle SEND_ONLY
 */
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle);
#endif
#if NCCL_VERSION_CODE >= 22907
/**
 * @guard NCCL_VERSION_CODE >= 22907
 * @param comm SEND_ONLY
 * @param flags SEND_ONLY
 */
ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags);
/**
 * @guard NCCL_VERSION_CODE >= 22907
 * @param comm SEND_ONLY
 */
ncclResult_t ncclCommResume(ncclComm_t comm);
/**
 * @guard NCCL_VERSION_CODE >= 22907
 * @param comm SEND_ONLY
 * @param stat SEND_ONLY
 * @param value RECV_ONLY
 */
ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat,
                              uint64_t *value);
#endif
#if NCCL_VERSION_CODE >= 22703
/**
 * @guard NCCL_VERSION_CODE >= 22703
 * @param comm SEND_ONLY
 * @param buff SEND_ONLY
 * @param size SEND_ONLY
 * @param win RECV_ONLY
 * @param winFlags SEND_ONLY
 */
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size,
                                    ncclWindow_t *win, int winFlags);
/**
 * @guard NCCL_VERSION_CODE >= 22703
 * @param comm SEND_ONLY
 * @param win SEND_ONLY
 */
ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
#endif
#if NCCL_VERSION_CODE >= 22902
/**
 * @guard NCCL_VERSION_CODE >= 22902
 * @param comm SEND_ONLY
 * @param win SEND_ONLY
 * @param outUserPtr RECV_ONLY
 */
ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win,
                               void **outUserPtr);
#endif

// The device API's host half, which nccl_device.h declares rather than
// nccl.h; its opaque handles are all of that header the shim needs. A device
// communicator is a kernel's handle on NCCL's device-side resources: the
// requirement list building one is a chain of nodes the library writes
// handles back into, and the communicator it mints is laid out for device
// code linked against that same NCCL release. Neither survives the wire, so
// the calls that build one report the configuration unusable. The window
// queries do cross: they answer with server device addresses, which is what
// every pointer this shim hands back already is.
typedef struct ncclDevComm ncclDevComm_t;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;
#if NCCL_VERSION_CODE >= 22800
/**
 * @guard NCCL_VERSION_CODE >= 22800
 */
ncclResult_t ncclDevCommCreate(ncclComm_t comm,
                               const ncclDevCommRequirements_t *reqs,
                               ncclDevComm_t *outDevComm);
/**
 * @guard NCCL_VERSION_CODE >= 22800
 */
ncclResult_t ncclDevCommDestroy(ncclComm_t comm, const ncclDevComm_t *devComm);
#endif
#if NCCL_VERSION_CODE >= 22900
/**
 * @guard NCCL_VERSION_CODE >= 22900
 * @param window SEND_ONLY
 * @param offset SEND_ONLY
 * @param outPtr RECV_ONLY
 */
ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset,
                                             void **outPtr);
/**
 * @guard NCCL_VERSION_CODE >= 22900
 * @param window SEND_ONLY
 * @param offset SEND_ONLY
 * @param peer SEND_ONLY
 * @param outPtr RECV_ONLY
 */
ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset,
                                      int peer, void **outPtr);
#endif
/**
 * @param op RECV_ONLY
 * @param scalar SEND_ONLY SCALAR:residence SIZE:nccl_type_width(datatype)
 * @param datatype SEND_ONLY
 * @param residence SEND_ONLY
 * @param comm SEND_ONLY
 */
ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm);
/**
 * @param op SEND_ONLY
 * @param comm SEND_ONLY
 */
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
/**
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param op SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream);
/**
 * @async
 * @param buff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, cudaStream_t stream);
/**
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream);
/**
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param op SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
/**
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param recvcount SEND_ONLY
 * @param datatype SEND_ONLY
 * @param op SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);
/**
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param sendcount SEND_ONLY
 * @param datatype SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);
#if NCCL_VERSION_CODE >= 22803
/**
 * @guard NCCL_VERSION_CODE >= 22803
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t stream);
/**
 * @guard NCCL_VERSION_CODE >= 22803
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream);
/**
 * @guard NCCL_VERSION_CODE >= 22803
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count,
                         ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream);
#endif
#if NCCL_VERSION_CODE >= 23102
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param op SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclAllReduceConfig(const void *sendbuff, void *recvbuff,
                                 size_t count, ncclDataType_t datatype,
                                 ncclRedOp_t op, ncclComm_t comm,
                                 cudaStream_t stream,
                                 const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclBroadcastConfig(const void *sendbuff, void *recvbuff,
                                 size_t count, ncclDataType_t datatype,
                                 int root, ncclComm_t comm, cudaStream_t stream,
                                 const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param op SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclReduceConfig(const void *sendbuff, void *recvbuff,
                              size_t count, ncclDataType_t datatype,
                              ncclRedOp_t op, int root, ncclComm_t comm,
                              cudaStream_t stream,
                              const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param sendcount SEND_ONLY
 * @param datatype SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclAllGatherConfig(const void *sendbuff, void *recvbuff,
                                 size_t sendcount, ncclDataType_t datatype,
                                 ncclComm_t comm, cudaStream_t stream,
                                 const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param recvcount SEND_ONLY
 * @param datatype SEND_ONLY
 * @param op SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclReduceScatterConfig(const void *sendbuff, void *recvbuff,
                                     size_t recvcount, ncclDataType_t datatype,
                                     ncclRedOp_t op, ncclComm_t comm,
                                     cudaStream_t stream,
                                     const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclAlltoAllConfig(const void *sendbuff, void *recvbuff,
                                size_t count, ncclDataType_t datatype,
                                ncclComm_t comm, cudaStream_t stream,
                                const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclGatherConfig(const void *sendbuff, void *recvbuff,
                              size_t count, ncclDataType_t datatype, int root,
                              ncclComm_t comm, cudaStream_t stream,
                              const ncclCollConfig_t *config);
/**
 * @guard NCCL_VERSION_CODE >= 23102
 * @async
 * @param sendbuff SEND_ONLY
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param root SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 * @param config SEND_ONLY VERSIONED STRINGS:algSelection CLEARED:ext
 */
ncclResult_t ncclScatterConfig(const void *sendbuff, void *recvbuff,
                               size_t count, ncclDataType_t datatype, int root,
                               ncclComm_t comm, cudaStream_t stream,
                               const ncclCollConfig_t *config);
#endif
/**
 * @async
 * @param sendbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param peer SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclSend(const void *sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream);
/**
 * @async
 * @param recvbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param peer SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream);
#if NCCL_VERSION_CODE >= 22902
/**
 * @guard NCCL_VERSION_CODE >= 22902
 * @async
 * @param localbuff SEND_ONLY
 * @param count SEND_ONLY
 * @param datatype SEND_ONLY
 * @param peer SEND_ONLY
 * @param peerWin SEND_ONLY
 * @param peerWinOffset SEND_ONLY
 * @param sigIdx SEND_ONLY
 * @param ctx SEND_ONLY
 * @param flags SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclPutSignal(const void *localbuff, size_t count,
                           ncclDataType_t datatype, int peer,
                           ncclWindow_t peerWin, size_t peerWinOffset,
                           int sigIdx, int ctx, unsigned int flags,
                           ncclComm_t comm, cudaStream_t stream);
/**
 * @guard NCCL_VERSION_CODE >= 22902
 * @async
 * @param peer SEND_ONLY
 * @param sigIdx SEND_ONLY
 * @param ctx SEND_ONLY
 * @param flags SEND_ONLY
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags,
                        ncclComm_t comm, cudaStream_t stream);
/**
 * @guard NCCL_VERSION_CODE >= 22902
 * @async
 * @param nDesc SEND_ONLY
 * @param signalDescs SEND_ONLY LENGTH:nDesc
 * @param comm SEND_ONLY
 * @param stream SEND_ONLY
 */
ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t *signalDescs,
                            ncclComm_t comm, cudaStream_t stream);
#endif
/**
 * @disabled client - opens the group on each server it reaches
 * @async
 */
ncclResult_t ncclGroupStart();
/**
 * @disabled client - closes the group on the server it reached
 */
ncclResult_t ncclGroupEnd();
#if NCCL_VERSION_CODE >= 22203
/**
 * @guard NCCL_VERSION_CODE >= 22203
 * @disabled client - closes the group on the server it reached
 * @param simInfo SEND_RECV DEREF
 */
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo);
#endif
#if NCCL_VERSION_CODE >= 23007
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param out RECV_ONLY
 * @param key SEND_ONLY NULL_TERMINATED
 */
// clang-format off
ncclResult_t ncclParamBind(ncclParamHandle_t *out, const char *key) {
  ncclResult_t return_value = LUPINE_GENERATED_CALL();
  if (return_value == ncclSuccess) {
    note_handle_owner(conn, *out);
  }
  return return_value;
}
// clang-format on
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetI8(ncclParamHandle_t h, int8_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetI16(ncclParamHandle_t h, int16_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetI32(ncclParamHandle_t h, int32_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetI64(ncclParamHandle_t h, int64_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetU8(ncclParamHandle_t h, uint8_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetU16(ncclParamHandle_t h, uint16_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetU32(ncclParamHandle_t h, uint32_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param out RECV_ONLY
 */
ncclResult_t ncclParamGetU64(ncclParamHandle_t h, uint64_t *out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @disabled
 */
ncclResult_t ncclParamGetStr(ncclParamHandle_t h, const char **out);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @param h SEND_ONLY
 * @param maxLen SEND_ONLY
 * @param out SEND_RECV LENGTH:maxLen
 * @param len RECV_ONLY
 */
ncclResult_t ncclParamGet(ncclParamHandle_t h, void *out, int maxLen, int *len);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @disabled
 */
ncclResult_t ncclParamGetParameter(const char *key, const char **value,
                                   int *valueLen);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 * @disabled
 */
ncclResult_t ncclParamGetAllParameterKeys(const char ***table, int *tableLen);
/**
 * @guard NCCL_VERSION_CODE >= 23007
 */
void ncclParamDumpAll(void);
#endif
