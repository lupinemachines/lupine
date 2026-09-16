// CUPTI (libcupti) on the LUPINE client.
//
// CUPTI measures a process from inside that process: it hooks the driver and
// runtime entry points the process itself calls, and hands back the activity
// records the driver produced for them. A LUPINE client calls neither. Its
// CUDA calls are RPCs, and the work they describe runs in a server process on
// another machine, so there is nothing on this side for CUPTI to measure and
// this shim exists to make libcupti loadable, not to profile.
//
// Forwarding was considered and rejected. A server-side CUPTI measures the
// server: its process and thread ids, its monotonic clock, correlation ids for
// driver calls this client never made, and, on a server holding connections
// for more than one client, records that belong to the others. The callback
// API compounds it, since cuptiSubscribe registers a function pointer CUPTI
// calls from inside the driver call being profiled, which would put a client
// round trip in the server's hot path for every call it makes. The event,
// metric, PC-sampling and profiler APIs reprogram the GPU's performance
// counters and serialize its kernels, which one client cannot be allowed to do
// to a shared server. A timeline stitched out of any of that would be wrong in
// ways its reader cannot see, and a wrong measurement is worse than a missing
// one.
//
// So every entry point that measures, subscribes, or configures a measurement
// returns CUPTI_ERROR_NOT_SUPPORTED - "the attempted operation is not
// supported on the current system or device", which is the case here - and
// says so once on stderr, so an empty timeline is explained rather than
// silent. What can be answered without a GPU is answered: cuptiGetVersion
// reports the API version this shim was built against, cuptiGetResultString
// and cuptiGetErrorMessage name a result code, and cuptiGetLastError returns
// and clears the calling thread's last refusal. Nothing crosses the wire, so
// there is no annotations file, no RPC id, and no server side.

#include "lupine_log.h"

#include <cupti_result.h>
#include <cupti_version.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {

// Indexed by the numeric CUptiResult, which is append-only, so the table names
// no enumerator a build's header may not declare. 36 and 37 are unassigned,
// and NVIDIA's libcupti answers those and any other unknown code the way the
// null entries here do.
constexpr const char *const result_names[] = {
    "CUPTI_SUCCESS",
    "CUPTI_ERROR_INVALID_PARAMETER",
    "CUPTI_ERROR_INVALID_DEVICE",
    "CUPTI_ERROR_INVALID_CONTEXT",
    "CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID",
    "CUPTI_ERROR_INVALID_EVENT_ID",
    "CUPTI_ERROR_INVALID_EVENT_NAME",
    "CUPTI_ERROR_INVALID_OPERATION",
    "CUPTI_ERROR_OUT_OF_MEMORY",
    "CUPTI_ERROR_HARDWARE",
    "CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT",
    "CUPTI_ERROR_API_NOT_IMPLEMENTED",
    "CUPTI_ERROR_MAX_LIMIT_REACHED",
    "CUPTI_ERROR_NOT_READY",
    "CUPTI_ERROR_NOT_COMPATIBLE",
    "CUPTI_ERROR_NOT_INITIALIZED",
    "CUPTI_ERROR_INVALID_METRIC_ID",
    "CUPTI_ERROR_INVALID_METRIC_NAME",
    "CUPTI_ERROR_QUEUE_EMPTY",
    "CUPTI_ERROR_INVALID_HANDLE",
    "CUPTI_ERROR_INVALID_STREAM",
    "CUPTI_ERROR_INVALID_KIND",
    "CUPTI_ERROR_INVALID_EVENT_VALUE",
    "CUPTI_ERROR_DISABLED",
    "CUPTI_ERROR_INVALID_MODULE",
    "CUPTI_ERROR_INVALID_METRIC_VALUE",
    "CUPTI_ERROR_HARDWARE_BUSY",
    "CUPTI_ERROR_NOT_SUPPORTED",
    "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED",
    "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE",
    "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES",
    "CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS",
    "CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED",
    "CUPTI_ERROR_VIRTUALIZED_DEVICE_NOT_SUPPORTED",
    "CUPTI_ERROR_CUDA_COMPILER_NOT_COMPATIBLE",
    "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES",
    nullptr,
    nullptr,
    "CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED",
    "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED",
    "CUPTI_ERROR_VIRTUALIZED_DEVICE_INSUFFICIENT_PRIVILEGES",
    "CUPTI_ERROR_CONFIDENTIAL_COMPUTING_NOT_SUPPORTED",
    "CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED",
    "CUPTI_ERROR_MIG_DEVICE_NOT_SUPPORTED",
    "CUPTI_ERROR_SLI_DEVICE_NOT_SUPPORTED",
    "CUPTI_ERROR_WSL_DEVICE_NOT_SUPPORTED",
};

const char *name_of(CUptiResult result) {
  if (result == CUPTI_ERROR_UNKNOWN) {
    return "CUPTI_ERROR_UNKNOWN";
  }
  const size_t code = static_cast<size_t>(result);
  if (code >= sizeof(result_names) / sizeof(result_names[0])) {
    return nullptr;
  }
  return result_names[code];
}

thread_local CUptiResult last_error = CUPTI_SUCCESS;

CUptiResult refuse() {
  static std::atomic<bool> announced{false};
  if (!announced.exchange(true)) {
    LUPINE_LOG_ERROR(
        "lupine: CUPTI cannot profile a LUPINE client - the work it would "
        "measure runs on the server - so every cupti* entry point but the "
        "version and error-string queries returns CUPTI_ERROR_NOT_SUPPORTED");
  }
  last_error = CUPTI_ERROR_NOT_SUPPORTED;
  return CUPTI_ERROR_NOT_SUPPORTED;
}

} // namespace

extern "C" CUptiResult cuptiGetVersion(uint32_t *version) {
  *version = CUPTI_API_VERSION;
  return CUPTI_SUCCESS;
}

extern "C" CUptiResult cuptiGetResultString(CUptiResult result,
                                            const char **str) {
  const char *name = name_of(result);
  *str = name == nullptr ? "<unknown>" : name;
  return name == nullptr ? CUPTI_ERROR_INVALID_PARAMETER : CUPTI_SUCCESS;
}

// The detailed message NVIDIA's libcupti has for a handful of codes explains a
// hardware or privilege condition none of them describe here, and for every
// other code it is the code's name.
extern "C" CUptiResult cuptiGetErrorMessage(CUptiResult result,
                                            const char **str) {
  const char *name = name_of(result);
  *str = name == nullptr ? "<unknown>" : name;
  return CUPTI_SUCCESS;
}

// Declared in cupti_activity.h, which this file does not include for the
// reason the block below gives.
extern "C" CUptiResult cuptiGetLastError(void) {
  const CUptiResult result = last_error;
  last_error = CUPTI_SUCCESS;
  return result;
}

// Each entry point below is declared with no parameters and defined to ignore
// whatever the caller passes: the refusal does not depend on the arguments,
// and a callee that reads none of them returns correctly to a caller that
// passed any. One list therefore serves every CUPTI generation - a name's
// parameters, and the types they are spelled with, change between releases
// while the name and its CUptiResult do not - and this file includes no header
// that declares them, since a declaration would contradict these definitions.
//
// The names are every cupti* symbol NVIDIA's libcupti exports, as of CUDA
// 11.8, 12.9 and 13.3. The three CUDA 13 dropped stay: the same list serves a
// libcupti.so.11.8 or libcupti.so.12 build of this shim.
#define LUPINE_CUPTI_REFUSED(name)                                             \
  extern "C" CUptiResult name(void) { return refuse(); }

LUPINE_CUPTI_REFUSED(cuptiActivityConfigurePCSampling)
LUPINE_CUPTI_REFUSED(cuptiActivityConfigureUnifiedMemoryCounter)
LUPINE_CUPTI_REFUSED(cuptiActivityDisable)
LUPINE_CUPTI_REFUSED(cuptiActivityDisableContext)
LUPINE_CUPTI_REFUSED(cuptiActivityDisable_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityEnable)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableAllSyncRecords)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableAllocationSource)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableAndDump)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableAndDump_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableBufferSummary)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableContext)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableCudaEventDeviceTimestamps)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableDeviceGraph)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableDriverApi)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableDriverApi_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableHWTrace)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableLatencyTimestamps)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableLaunchAttributes)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableRawTimestamps)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableRawTimestamps_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableRuntimeApi)
LUPINE_CUPTI_REFUSED(cuptiActivityEnableRuntimeApi_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityEnable_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityFlush)
LUPINE_CUPTI_REFUSED(cuptiActivityFlushAll)
LUPINE_CUPTI_REFUSED(cuptiActivityFlushPeriod)
LUPINE_CUPTI_REFUSED(cuptiActivityGetAttribute)
LUPINE_CUPTI_REFUSED(cuptiActivityGetAttribute_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityGetEnabledKinds)
LUPINE_CUPTI_REFUSED(cuptiActivityGetNextRecord)
LUPINE_CUPTI_REFUSED(cuptiActivityGetNextRecord_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityGetNumDroppedRecords)
LUPINE_CUPTI_REFUSED(cuptiActivityGetNumDroppedRecords_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityGetNvtxExtPayloadAttr)
LUPINE_CUPTI_REFUSED(cuptiActivityGetNvtxExtPayloadEntryTypeInfo)
LUPINE_CUPTI_REFUSED(cuptiActivityGetStructSize)
LUPINE_CUPTI_REFUSED(cuptiActivityPopExternalCorrelationId)
LUPINE_CUPTI_REFUSED(cuptiActivityPopExternalCorrelationId_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityPushExternalCorrelationId)
LUPINE_CUPTI_REFUSED(cuptiActivityPushExternalCorrelationId_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityRegisterCallbacks)
LUPINE_CUPTI_REFUSED(cuptiActivityRegisterCallbacks_v2)
LUPINE_CUPTI_REFUSED(cuptiActivityRegisterTimestampCallback)
LUPINE_CUPTI_REFUSED(cuptiActivitySetAttribute)
LUPINE_CUPTI_REFUSED(cuptiActivitySetAttribute_v2)
LUPINE_CUPTI_REFUSED(cuptiComputeCapabilitySupported)
LUPINE_CUPTI_REFUSED(cuptiDeviceEnumEventDomains)
LUPINE_CUPTI_REFUSED(cuptiDeviceEnumMetrics)
LUPINE_CUPTI_REFUSED(cuptiDeviceGetAttribute)
LUPINE_CUPTI_REFUSED(cuptiDeviceGetChipName)
LUPINE_CUPTI_REFUSED(cuptiDeviceGetEventDomainAttribute)
LUPINE_CUPTI_REFUSED(cuptiDeviceGetNumEventDomains)
LUPINE_CUPTI_REFUSED(cuptiDeviceGetNumMetrics)
LUPINE_CUPTI_REFUSED(cuptiDeviceGetTimestamp)
LUPINE_CUPTI_REFUSED(cuptiDeviceSupported)
LUPINE_CUPTI_REFUSED(cuptiDeviceVirtualizationMode)
LUPINE_CUPTI_REFUSED(cuptiDisableKernelReplayMode)
LUPINE_CUPTI_REFUSED(cuptiDisableLibcuda)
LUPINE_CUPTI_REFUSED(cuptiDisableNonOverlappingMode)
LUPINE_CUPTI_REFUSED(cuptiEnableAllDomains)
LUPINE_CUPTI_REFUSED(cuptiEnableCallback)
LUPINE_CUPTI_REFUSED(cuptiEnableCigMode)
LUPINE_CUPTI_REFUSED(cuptiEnableDomain)
LUPINE_CUPTI_REFUSED(cuptiEnableKernelReplayMode)
LUPINE_CUPTI_REFUSED(cuptiEnableNonOverlappingMode)
LUPINE_CUPTI_REFUSED(cuptiEnumEventDomains)
LUPINE_CUPTI_REFUSED(cuptiEnumMetrics)
LUPINE_CUPTI_REFUSED(cuptiEventDomainEnumEvents)
LUPINE_CUPTI_REFUSED(cuptiEventDomainGetAttribute)
LUPINE_CUPTI_REFUSED(cuptiEventDomainGetNumEvents)
LUPINE_CUPTI_REFUSED(cuptiEventGetAttribute)
LUPINE_CUPTI_REFUSED(cuptiEventGetIdFromName)
LUPINE_CUPTI_REFUSED(cuptiEventGroupAddEvent)
LUPINE_CUPTI_REFUSED(cuptiEventGroupCreate)
LUPINE_CUPTI_REFUSED(cuptiEventGroupDestroy)
LUPINE_CUPTI_REFUSED(cuptiEventGroupDisable)
LUPINE_CUPTI_REFUSED(cuptiEventGroupEnable)
LUPINE_CUPTI_REFUSED(cuptiEventGroupGetAttribute)
LUPINE_CUPTI_REFUSED(cuptiEventGroupReadAllEvents)
LUPINE_CUPTI_REFUSED(cuptiEventGroupReadEvent)
LUPINE_CUPTI_REFUSED(cuptiEventGroupRemoveAllEvents)
LUPINE_CUPTI_REFUSED(cuptiEventGroupRemoveEvent)
LUPINE_CUPTI_REFUSED(cuptiEventGroupResetAllEvents)
LUPINE_CUPTI_REFUSED(cuptiEventGroupSetAttribute)
LUPINE_CUPTI_REFUSED(cuptiEventGroupSetDisable)
LUPINE_CUPTI_REFUSED(cuptiEventGroupSetEnable)
LUPINE_CUPTI_REFUSED(cuptiEventGroupSetsCreate)
LUPINE_CUPTI_REFUSED(cuptiEventGroupSetsDestroy)
LUPINE_CUPTI_REFUSED(cuptiFinalize)
LUPINE_CUPTI_REFUSED(cuptiGetAutoBoostState)
LUPINE_CUPTI_REFUSED(cuptiGetCallbackName)
LUPINE_CUPTI_REFUSED(cuptiGetCallbackState)
LUPINE_CUPTI_REFUSED(cuptiGetContextId)
LUPINE_CUPTI_REFUSED(cuptiGetCubinCrc)
LUPINE_CUPTI_REFUSED(cuptiGetDeviceId)
LUPINE_CUPTI_REFUSED(cuptiGetEnabledCallbacks)
LUPINE_CUPTI_REFUSED(cuptiGetGlobalCallbackState)
LUPINE_CUPTI_REFUSED(cuptiGetGraphExecId)
LUPINE_CUPTI_REFUSED(cuptiGetGraphId)
LUPINE_CUPTI_REFUSED(cuptiGetGraphNodeId)
LUPINE_CUPTI_REFUSED(cuptiGetNumContexts)
LUPINE_CUPTI_REFUSED(cuptiGetNumEventDomains)
LUPINE_CUPTI_REFUSED(cuptiGetNumMetrics)
LUPINE_CUPTI_REFUSED(cuptiGetRecommendedBufferSize)
LUPINE_CUPTI_REFUSED(cuptiGetSassToSourceCorrelation)
LUPINE_CUPTI_REFUSED(cuptiGetStreamId)
LUPINE_CUPTI_REFUSED(cuptiGetStreamIdEx)
LUPINE_CUPTI_REFUSED(cuptiGetThreadIdType)
LUPINE_CUPTI_REFUSED(cuptiGetTimestamp)
LUPINE_CUPTI_REFUSED(cuptiGetTimestamp_v2)
LUPINE_CUPTI_REFUSED(cuptiIsTracingSessionRunning)
LUPINE_CUPTI_REFUSED(cuptiKernelReplaySubscribeUpdate)
LUPINE_CUPTI_REFUSED(cuptiMetricCreateEventGroupSets)
LUPINE_CUPTI_REFUSED(cuptiMetricEnumEvents)
LUPINE_CUPTI_REFUSED(cuptiMetricEnumProperties)
LUPINE_CUPTI_REFUSED(cuptiMetricGetAttribute)
LUPINE_CUPTI_REFUSED(cuptiMetricGetIdFromName)
LUPINE_CUPTI_REFUSED(cuptiMetricGetNumEvents)
LUPINE_CUPTI_REFUSED(cuptiMetricGetNumProperties)
LUPINE_CUPTI_REFUSED(cuptiMetricGetRequiredEventGroupSets)
LUPINE_CUPTI_REFUSED(cuptiMetricGetValue)
LUPINE_CUPTI_REFUSED(cuptiMetricGetValue2)
LUPINE_CUPTI_REFUSED(cuptiNvtxExtensionInitialize)
LUPINE_CUPTI_REFUSED(cuptiNvtxInitialize)
LUPINE_CUPTI_REFUSED(cuptiNvtxInitialize2)
LUPINE_CUPTI_REFUSED(cuptiOpenACCInitialize)
LUPINE_CUPTI_REFUSED(cuptiOpenMpInitialize)
LUPINE_CUPTI_REFUSED(cuptiOpenMpInitialize_v2)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingDisable)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingEnable)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingGetConfigurationAttribute)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingGetData)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingGetNumStallReasons)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingGetStallReasons)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingSetConfigurationAttribute)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingStart)
LUPINE_CUPTI_REFUSED(cuptiPCSamplingStop)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingCounterDataGetSampleInfo)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingCounterDataImageInitialize)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingDecodeData)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingDisable)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingEnable)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingGetCounterAvailability)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingGetCounterDataInfo)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingGetCounterDataSize)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingSetConfig)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingStart)
LUPINE_CUPTI_REFUSED(cuptiPmSamplingStop)
LUPINE_CUPTI_REFUSED(cuptiProfilerBeginPass)
LUPINE_CUPTI_REFUSED(cuptiProfilerBeginSession)
LUPINE_CUPTI_REFUSED(cuptiProfilerCounterDataImageCalculateScratchBufferSize)
LUPINE_CUPTI_REFUSED(cuptiProfilerCounterDataImageCalculateSize)
LUPINE_CUPTI_REFUSED(cuptiProfilerCounterDataImageInitialize)
LUPINE_CUPTI_REFUSED(cuptiProfilerCounterDataImageInitializeScratchBuffer)
LUPINE_CUPTI_REFUSED(cuptiProfilerDeInitialize)
LUPINE_CUPTI_REFUSED(cuptiProfilerDeviceSupported)
LUPINE_CUPTI_REFUSED(cuptiProfilerDisableProfiling)
LUPINE_CUPTI_REFUSED(cuptiProfilerEnableProfiling)
LUPINE_CUPTI_REFUSED(cuptiProfilerEndPass)
LUPINE_CUPTI_REFUSED(cuptiProfilerEndSession)
LUPINE_CUPTI_REFUSED(cuptiProfilerFlushCounterData)
LUPINE_CUPTI_REFUSED(cuptiProfilerGetCounterAvailability)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostConfigAddMetrics)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostDeinitialize)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostEvaluateToGpuValues)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetBaseMetrics)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetConfigImage)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetConfigImageSize)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetMaxNumHardwareMetricsPerPass)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetMetricProperties)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetMetricsInSinglePassSet)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetNumOfPasses)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetRangeName)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetSinglePassSets)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetSubMetrics)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostGetSupportedChips)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostInitialize)
LUPINE_CUPTI_REFUSED(cuptiProfilerHostSetDevicePartitionInfo)
LUPINE_CUPTI_REFUSED(cuptiProfilerInitialize)
LUPINE_CUPTI_REFUSED(cuptiProfilerIsPassCollected)
LUPINE_CUPTI_REFUSED(cuptiProfilerPopRange)
LUPINE_CUPTI_REFUSED(cuptiProfilerPushRange)
LUPINE_CUPTI_REFUSED(cuptiProfilerSetConfig)
LUPINE_CUPTI_REFUSED(cuptiProfilerUnsetConfig)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerCounterDataGetRangeInfo)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerCounterDataImageInitialize)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerDecodeData)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerDisable)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerEnable)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerGetCounterDataInfo)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerGetCounterDataSize)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerGetDevicePartitionInfo)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerPopRange)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerPushRange)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerSetConfig)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerStart)
LUPINE_CUPTI_REFUSED(cuptiRangeProfilerStop)
LUPINE_CUPTI_REFUSED(cuptiRegisterComputeCrcCallback)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsDisable)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsEnable)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsFlushData)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsGetDataProperties)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsGetMetrics)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsGetNumOfMetrics)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsGetProperties)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsSetConfig)
LUPINE_CUPTI_REFUSED(cuptiSassMetricsUnsetConfig)
LUPINE_CUPTI_REFUSED(cuptiSetEventCollectionMode)
LUPINE_CUPTI_REFUSED(cuptiSetThreadIdType)
LUPINE_CUPTI_REFUSED(cuptiStateQuery)
LUPINE_CUPTI_REFUSED(cuptiSubscribe)
LUPINE_CUPTI_REFUSED(cuptiSubscribe_v2)
LUPINE_CUPTI_REFUSED(cuptiSupportedDomains)
LUPINE_CUPTI_REFUSED(cuptiUnsubscribe)
