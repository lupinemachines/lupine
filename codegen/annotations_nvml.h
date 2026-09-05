#include <nvml.h>


typedef struct {
  unsigned int version;
  nvmlTemperatureSensors_t sensorType;
  int temperature;
} lupine_nvmlTemperature_t;

/**
 * @disabled client - manually initialized on every server connection
 */
nvmlReturn_t nvmlInit_v2();
/**
 * @disabled client - manually initialized on every server connection
 * @param flags SEND_ONLY
 */
nvmlReturn_t nvmlInitWithFlags(unsigned int flags);
/**
 * @disabled client - manually shut down on every server connection
 */
nvmlReturn_t nvmlShutdown();
/**
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length);
/**
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlSystemGetNVMLVersion(char *version, unsigned int length);
/**
 * @param cudaDriverVersion RECV_ONLY
 */
nvmlReturn_t nvmlSystemGetCudaDriverVersion(int *cudaDriverVersion);
/**
 * @param cudaDriverVersion RECV_ONLY
 */
nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int *cudaDriverVersion);
/**
 * @param pid SEND_ONLY
 * @param length SEND_ONLY
 * @param name RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char *name,
                                      unsigned int length);
/**
 * @param unitCount RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetCount(unsigned int *unitCount);
/**
 * @param index SEND_ONLY
 * @param unit RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t *unit);
/**
 * @param unit SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t *info);
/**
 * @param unit SEND_ONLY
 * @param state RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t *state);
/**
 * @param unit SEND_ONLY
 * @param psu RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t *psu);
/**
 * @param unit SEND_ONLY
 * @param type SEND_ONLY
 * @param temp RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type,
                                    unsigned int *temp);
/**
 * @param unit SEND_ONLY
 * @param fanSpeeds RECV_ONLY
 */
nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit,
                                     nvmlUnitFanSpeeds_t *fanSpeeds);
/**
 * @param unit SEND_ONLY
 * @param deviceCount SEND_RECV
 * @param devices RECV_ONLY LENGTH:deviceCount
 */
nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int *deviceCount,
                                nvmlDevice_t *devices);
/**
 * @param hwbcCount SEND_RECV
 * @param hwbcEntries RECV_ONLY LENGTH:hwbcCount
 */
nvmlReturn_t nvmlSystemGetHicVersion(unsigned int *hwbcCount,
                                     nvmlHwbcEntry_t *hwbcEntries);
/**
 * @disabled client - manual client aggregates every server's device count
 * @param deviceCount RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount);
/**
 * @param device SEND_ONLY
 * @param attributes RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device,
                                        nvmlDeviceAttributes_t *attributes);
/**
 * @disabled client - manual client maps the global ordinal to a virtual device handle
 * @param index SEND_ONLY
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index,
                                           nvmlDevice_t *device);
/**
 * @param serial SEND_ONLY NULL_TERMINATED
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleBySerial(const char *serial,
                                         nvmlDevice_t *device);
/**
 * @routingkey ALL
 * @recordowner DEVICE device
 * @param uuid SEND_ONLY NULL_TERMINATED
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char *uuid, nvmlDevice_t *device);
/**
 * @routingkey ALL
 * @recordowner DEVICE device
 * @param pciBusId SEND_ONLY NULL_TERMINATED
 * @param device RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId,
                                              nvmlDevice_t *device);
/**
 * @disabled client - manual client appends the server label to disambiguate devices
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param name RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name,
                               unsigned int length);
/**
 * @param device SEND_ONLY
 * @param type RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t *type);
/**
 * @disabled client - manual client returns the virtual global device index
 * @param device SEND_ONLY
 * @param index RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param serial RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char *serial,
                                 unsigned int length);
/**
 * @param device SEND_ONLY
 * @param nodeSetSize SEND_ONLY
 * @param nodeSet RECV_ONLY LENGTH:nodeSetSize
 * @param scope SEND_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device,
                                         unsigned int nodeSetSize,
                                         unsigned long *nodeSet,
                                         nvmlAffinityScope_t scope);
/**
 * @param device SEND_ONLY
 * @param cpuSetSize SEND_ONLY
 * @param cpuSet RECV_ONLY LENGTH:cpuSetSize
 * @param scope SEND_ONLY
 */
nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device,
                                                 unsigned int cpuSetSize,
                                                 unsigned long *cpuSet,
                                                 nvmlAffinityScope_t scope);
/**
 * @param device SEND_ONLY
 * @param cpuSetSize SEND_ONLY
 * @param cpuSet RECV_ONLY LENGTH:cpuSetSize
 */
nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device,
                                      unsigned int cpuSetSize,
                                      unsigned long *cpuSet);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param pathInfo RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2,
                                    nvmlGpuTopologyLevel_t *pathInfo);
/**
 * @param device SEND_ONLY
 * @param level SEND_ONLY
 * @param count SEND_RECV
 * @param deviceArray RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device,
                                              nvmlGpuTopologyLevel_t level,
                                              unsigned int *count,
                                              nvmlDevice_t *deviceArray);
/**
 * @param cpuNumber SEND_ONLY
 * @param count SEND_RECV
 * @param deviceArray RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber,
                                         unsigned int *count,
                                         nvmlDevice_t *deviceArray);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param p2pIndex SEND_ONLY
 * @param p2pStatus RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2,
                                    nvmlGpuP2PCapsIndex_t p2pIndex,
                                    nvmlGpuP2PStatus_t *p2pStatus);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param uuid RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid,
                               unsigned int length);
/**
 * @param vgpuInstance SEND_ONLY
 * @param size SEND_ONLY
 * @param mdevUuid RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance,
                                         char *mdevUuid, unsigned int size);
/**
 * @param device SEND_ONLY
 * @param minorNumber RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device,
                                      unsigned int *minorNumber);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param partNumber RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char *partNumber,
                                          unsigned int length);
/**
 * @param device SEND_ONLY
 * @param object SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device,
                                         nvmlInforomObject_t object,
                                         char *version, unsigned int length);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device,
                                              char *version,
                                              unsigned int length);
/**
 * @param device SEND_ONLY
 * @param checksum RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device,
                                                       unsigned int *checksum);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param display RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device,
                                      nvmlEnableState_t *display);
/**
 * @param device SEND_ONLY
 * @param isActive RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device,
                                        nvmlEnableState_t *isActive);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device,
                                          nvmlEnableState_t *mode);
/**
 * @param device SEND_ONLY
 * @param pci RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci);
/**
 * @param device SEND_ONLY
 * @param maxLinkGen RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device,
                                                unsigned int *maxLinkGen);
/**
 * @param device SEND_ONLY
 * @param maxLinkGenDevice RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device,
                                      unsigned int *maxLinkGenDevice);
/**
 * @param device SEND_ONLY
 * @param maxLinkWidth RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device,
                                           unsigned int *maxLinkWidth);
/**
 * @param device SEND_ONLY
 * @param currLinkGen RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device,
                                                 unsigned int *currLinkGen);
/**
 * @param device SEND_ONLY
 * @param currLinkWidth RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device,
                                            unsigned int *currLinkWidth);
/**
 * @param device SEND_ONLY
 * @param counter SEND_ONLY
 * @param value RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device,
                                         nvmlPcieUtilCounter_t counter,
                                         unsigned int *value);
/**
 * @param device SEND_ONLY
 * @param value RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device,
                                            unsigned int *value);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param clock RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type,
                                    unsigned int *clock);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param clock RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device,
                                       nvmlClockType_t type,
                                       unsigned int *clock);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device,
                                            nvmlClockType_t clockType,
                                            unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device,
                                                   nvmlClockType_t clockType,
                                                   unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockId SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType,
                                nvmlClockId_t clockId, unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 * @param clockType SEND_ONLY
 * @param clockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device,
                                                nvmlClockType_t clockType,
                                                unsigned int *clockMHz);
/**
 * @param device SEND_ONLY
 * @param count SEND_RECV
 * @param clocksMHz RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device,
                                                unsigned int *count,
                                                unsigned int *clocksMHz);
/**
 * @param device SEND_ONLY
 * @param memoryClockMHz SEND_ONLY
 * @param count SEND_RECV
 * @param clocksMHz RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device,
                                                  unsigned int memoryClockMHz,
                                                  unsigned int *count,
                                                  unsigned int *clocksMHz);
/**
 * @param device SEND_ONLY
 * @param isEnabled RECV_ONLY
 * @param defaultIsEnabled RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device,
                                      nvmlEnableState_t *isEnabled,
                                      nvmlEnableState_t *defaultIsEnabled);
/**
 * @param device SEND_ONLY
 * @param enabled SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device,
                                                   nvmlEnableState_t enabled);
/**
 * @param device SEND_ONLY
 * @param enabled SEND_ONLY
 * @param flags SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(
    nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags);
/**
 * @param device SEND_ONLY
 * @param speed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int *speed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param speed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan,
                                      unsigned int *speed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param targetSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan,
                                         unsigned int *targetSpeed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device,
                                             unsigned int fan);
/**
 * @param device SEND_ONLY
 * @param minSpeed RECV_ONLY
 * @param maxSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device,
                                         unsigned int *minSpeed,
                                         unsigned int *maxSpeed);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param policy RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device,
                                              unsigned int fan,
                                              nvmlFanControlPolicy_t *policy);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param policy SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device,
                                           unsigned int fan,
                                           nvmlFanControlPolicy_t policy);
/**
 * @param device SEND_ONLY
 * @param numFans RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int *numFans);
/**
 * @param device SEND_ONLY
 * @param sensorType SEND_ONLY
 * @param temp RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device,
                                      nvmlTemperatureSensors_t sensorType,
                                      unsigned int *temp);
/**
 * @param device SEND_ONLY
 * @param temperature SEND_RECV
 */
nvmlReturn_t nvmlDeviceGetTemperatureV(
    nvmlDevice_t device, lupine_nvmlTemperature_t *temperature);
/**
 * @param device SEND_ONLY
 * @param thresholdType SEND_ONLY
 * @param temp RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device,
                                  nvmlTemperatureThresholds_t thresholdType,
                                  unsigned int *temp);
/**
 * @param device SEND_ONLY
 * @param thresholdType SEND_ONLY
 * @param temp SEND_RECV
 */
nvmlReturn_t nvmlDeviceSetTemperatureThreshold(
    nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int *temp);
/**
 * @param device SEND_ONLY
 * @param sensorIndex SEND_ONLY
 * @param pThermalSettings RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex,
                             nvmlGpuThermalSettings_t *pThermalSettings);
/**
 * @param device SEND_ONLY
 * @param pState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device,
                                           nvmlPstates_t *pState);
/**
 * @param device SEND_ONLY
 * @param clocksThrottleReasons RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(
    nvmlDevice_t device, unsigned long long *clocksThrottleReasons);
/**
 * @param device SEND_ONLY
 * @param supportedClocksThrottleReasons RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(
    nvmlDevice_t device, unsigned long long *supportedClocksThrottleReasons);
/**
 * @param device SEND_ONLY
 * @param pState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device,
                                     nvmlPstates_t *pState);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device,
                                              nvmlEnableState_t *mode);
/**
 * @param device SEND_ONLY
 * @param limit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device,
                                               unsigned int *limit);
/**
 * @param device SEND_ONLY
 * @param minLimit RECV_ONLY
 * @param maxLimit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(
    nvmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit);
/**
 * @param device SEND_ONLY
 * @param defaultLimit RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device,
                                         unsigned int *defaultLimit);
/**
 * @param device SEND_ONLY
 * @param power RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power);
/**
 * @param device SEND_ONLY
 * @param energy RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device,
                                                 unsigned long long *energy);
/**
 * @param device SEND_ONLY
 * @param limit RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device,
                                             unsigned int *limit);
/**
 * @param device SEND_ONLY
 * @param current RECV_ONLY
 * @param pending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device,
                                           nvmlGpuOperationMode_t *current,
                                           nvmlGpuOperationMode_t *pending);
/**
 * @param device SEND_ONLY
 * @param memory RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
/**
 * @param device SEND_ONLY
 * @param memory SEND_RECV
 */
nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device,
                                        nvmlMemory_v2_t *memory);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device,
                                      nvmlComputeMode_t *mode);
/**
 * @param device SEND_ONLY
 * @param major RECV_ONLY
 * @param minor RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major,
                                                int *minor);
/**
 * @param device SEND_ONLY
 * @param current RECV_ONLY
 * @param pending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device,
                                  nvmlEnableState_t *current,
                                  nvmlEnableState_t *pending);
/**
 * @param device SEND_ONLY
 * @param defaultMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device,
                                         nvmlEnableState_t *defaultMode);
/**
 * @param device SEND_ONLY
 * @param boardId RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int *boardId);
/**
 * @param device SEND_ONLY
 * @param multiGpuBool RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device,
                                        unsigned int *multiGpuBool);
/**
 * @param device SEND_ONLY
 * @param errorType SEND_ONLY
 * @param counterType SEND_ONLY
 * @param eccCounts RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device,
                                         nvmlMemoryErrorType_t errorType,
                                         nvmlEccCounterType_t counterType,
                                         unsigned long long *eccCounts);
/**
 * @param device SEND_ONLY
 * @param errorType SEND_ONLY
 * @param counterType SEND_ONLY
 * @param eccCounts RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device,
                                            nvmlMemoryErrorType_t errorType,
                                            nvmlEccCounterType_t counterType,
                                            nvmlEccErrorCounts_t *eccCounts);
/**
 * @param device SEND_ONLY
 * @param errorType SEND_ONLY
 * @param counterType SEND_ONLY
 * @param locationType SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device,
                                             nvmlMemoryErrorType_t errorType,
                                             nvmlEccCounterType_t counterType,
                                             nvmlMemoryLocation_t locationType,
                                             unsigned long long *count);
/**
 * @param device SEND_ONLY
 * @param utilization RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device,
                                           nvmlUtilization_t *utilization);
/**
 * @param device SEND_ONLY
 * @param utilization RECV_ONLY
 * @param samplingPeriodUs RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device,
                                             unsigned int *utilization,
                                             unsigned int *samplingPeriodUs);
/**
 * @param device SEND_ONLY
 * @param encoderQueryType SEND_ONLY
 * @param encoderCapacity RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device,
                                          nvmlEncoderType_t encoderQueryType,
                                          unsigned int *encoderCapacity);
/**
 * @param device SEND_ONLY
 * @param sessionCount RECV_ONLY
 * @param averageFps RECV_ONLY
 * @param averageLatency RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device,
                                       unsigned int *sessionCount,
                                       unsigned int *averageFps,
                                       unsigned int *averageLatency);
/**
 * @param device SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfos RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t
nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int *sessionCount,
                             nvmlEncoderSessionInfo_t *sessionInfos);
/**
 * @param device SEND_ONLY
 * @param utilization RECV_ONLY
 * @param samplingPeriodUs RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device,
                                             unsigned int *utilization,
                                             unsigned int *samplingPeriodUs);
/**
 * @param device SEND_ONLY
 * @param fbcStats RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device,
                                   nvmlFBCStats_t *fbcStats);
/**
 * @param device SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfo RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device,
                                      unsigned int *sessionCount,
                                      nvmlFBCSessionInfo_t *sessionInfo);
/**
 * @param device SEND_ONLY
 * @param current RECV_ONLY
 * @param pending RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device,
                                      nvmlDriverModel_t *current,
                                      nvmlDriverModel_t *pending);
/**
 * @param device SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char *version,
                                       unsigned int length);
/**
 * @param device SEND_ONLY
 * @param bridgeHierarchy RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device,
                            nvmlBridgeChipHierarchy_t *bridgeHierarchy);
/**
 * @param device SEND_ONLY
 * @param infoCount SEND_RECV
 * @param infos RECV_ONLY LENGTH:infoCount
 */
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device,
                                                     unsigned int *infoCount,
                                                     nvmlProcessInfo_t *infos);
/**
 * @param device SEND_ONLY
 * @param infoCount SEND_RECV
 * @param infos RECV_ONLY LENGTH:infoCount
 */
nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device,
                                                      unsigned int *infoCount,
                                                      nvmlProcessInfo_t *infos);
/**
 * @param device SEND_ONLY
 * @param infoCount SEND_RECV
 * @param infos RECV_ONLY LENGTH:infoCount
 */
nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(
    nvmlDevice_t device, unsigned int *infoCount, nvmlProcessInfo_t *infos);
/**
 * @param device1 SEND_ONLY
 * @param device2 SEND_ONLY
 * @param onSameBoard RECV_ONLY
 */
nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2,
                                   int *onSameBoard);
/**
 * @param device SEND_ONLY
 * @param apiType SEND_ONLY
 * @param isRestricted RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device,
                                         nvmlRestrictedAPI_t apiType,
                                         nvmlEnableState_t *isRestricted);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param lastSeenTimeStamp SEND_ONLY
 * @param sampleValType RECV_ONLY
 * @param sampleCount SEND_RECV
 * @param samples RECV_ONLY LENGTH:sampleCount
 */
nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type,
                                  unsigned long long lastSeenTimeStamp,
                                  nvmlValueType_t *sampleValType,
                                  unsigned int *sampleCount,
                                  nvmlSample_t *samples);
/**
 * @param device SEND_ONLY
 * @param bar1Memory RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device,
                                         nvmlBAR1Memory_t *bar1Memory);
/**
 * @param device SEND_ONLY
 * @param perfPolicyType SEND_ONLY
 * @param violTime RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device,
                                          nvmlPerfPolicyType_t perfPolicyType,
                                          nvmlViolationTime_t *violTime);
/**
 * @param device SEND_ONLY
 * @param irqNum RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int *irqNum);
/**
 * @param device SEND_ONLY
 * @param numCores RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device,
                                      unsigned int *numCores);
/**
 * @param device SEND_ONLY
 * @param powerSource RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device,
                                      nvmlPowerSource_t *powerSource);
/**
 * @param device SEND_ONLY
 * @param busWidth RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device,
                                         unsigned int *busWidth);
/**
 * @param device SEND_ONLY
 * @param maxSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device,
                                           unsigned int *maxSpeed);
/**
 * @param device SEND_ONLY
 * @param pcieSpeed RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device,
                                    unsigned int *pcieSpeed);
/**
 * @param device SEND_ONLY
 * @param adaptiveClockStatus RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device,
                                     unsigned int *adaptiveClockStatus);
/**
 * @param device SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device,
                                         nvmlEnableState_t *mode);
/**
 * @param device SEND_ONLY
 * @param pid SEND_ONLY
 * @param stats RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid,
                                          nvmlAccountingStats_t *stats);
/**
 * @param device SEND_ONLY
 * @param count SEND_RECV
 * @param pids RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device,
                                         unsigned int *count,
                                         unsigned int *pids);
/**
 * @param device SEND_ONLY
 * @param bufferSize RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device,
                                               unsigned int *bufferSize);
/**
 * @param device SEND_ONLY
 * @param cause SEND_ONLY
 * @param pageCount SEND_RECV
 * @param addresses RECV_ONLY LENGTH:pageCount
 */
nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device,
                                       nvmlPageRetirementCause_t cause,
                                       unsigned int *pageCount,
                                       unsigned long long *addresses);
/**
 * @param device SEND_ONLY
 * @param cause SEND_ONLY
 * @param pageCount SEND_RECV
 * @param addresses RECV_ONLY LENGTH:pageCount
 * @param timestamps RECV_ONLY LENGTH:pageCount
 */
nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device,
                                          nvmlPageRetirementCause_t cause,
                                          unsigned int *pageCount,
                                          unsigned long long *addresses,
                                          unsigned long long *timestamps);
/**
 * @param device SEND_ONLY
 * @param isPending RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device,
                                       nvmlEnableState_t *isPending);
/**
 * @param device SEND_ONLY
 * @param corrRows RECV_ONLY
 * @param uncRows RECV_ONLY
 * @param isPending RECV_ONLY
 * @param failureOccurred RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device,
                                       unsigned int *corrRows,
                                       unsigned int *uncRows,
                                       unsigned int *isPending,
                                       unsigned int *failureOccurred);
/**
 * @param device SEND_ONLY
 * @param values RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device,
                                  nvmlRowRemapperHistogramValues_t *values);
/**
 * @param device SEND_ONLY
 * @param arch RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device,
                                       nvmlDeviceArchitecture_t *arch);
/**
 * @param unit SEND_ONLY
 * @param color SEND_ONLY
 */
nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device,
                                          nvmlEnableState_t mode);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device,
                                      nvmlComputeMode_t mode);
/**
 * @param device SEND_ONLY
 * @param ecc SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc);
/**
 * @param device SEND_ONLY
 * @param counterType SEND_ONLY
 */
nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device,
                                           nvmlEccCounterType_t counterType);
/**
 * @param device SEND_ONLY
 * @param driverModel SEND_ONLY
 * @param flags SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device,
                                      nvmlDriverModel_t driverModel,
                                      unsigned int flags);
/**
 * @param device SEND_ONLY
 * @param minGpuClockMHz SEND_ONLY
 * @param maxGpuClockMHz SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device,
                                          unsigned int minGpuClockMHz,
                                          unsigned int maxGpuClockMHz);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param minMemClockMHz SEND_ONLY
 * @param maxMemClockMHz SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device,
                                             unsigned int minMemClockMHz,
                                             unsigned int maxMemClockMHz);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param memClockMHz SEND_ONLY
 * @param graphicsClockMHz SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device,
                                             unsigned int memClockMHz,
                                             unsigned int graphicsClockMHz);
/**
 * @param device SEND_ONLY
 * @param status RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device,
                                       nvmlClkMonStatus_t *status);
/**
 * @param device SEND_ONLY
 * @param limit SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device,
                                               unsigned int limit);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device,
                                           nvmlGpuOperationMode_t mode);
/**
 * @param device SEND_ONLY
 * @param apiType SEND_ONLY
 * @param isRestricted SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device,
                                         nvmlRestrictedAPI_t apiType,
                                         nvmlEnableState_t isRestricted);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device,
                                         nvmlEnableState_t mode);
/**
 * @param device SEND_ONLY
 */
nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param isActive RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link,
                                      nvmlEnableState_t *isActive);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param version RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link,
                                        unsigned int *version);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device,
                                           unsigned int link,
                                           nvmlNvLinkCapability_t capability,
                                           unsigned int *capResult);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param pci RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device,
                                                 unsigned int link,
                                                 nvmlPciInfo_t *pci);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param counterValue RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device,
                                             unsigned int link,
                                             nvmlNvLinkErrorCounter_t counter,
                                             unsigned long long *counterValue);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device,
                                                unsigned int link);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param control SEND_ONLY DEREFERENCE
 * @param reset SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(
    nvmlDevice_t device, unsigned int link, unsigned int counter,
    nvmlNvLinkUtilizationControl_t *control, unsigned int reset);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param control RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link,
                                      unsigned int counter,
                                      nvmlNvLinkUtilizationControl_t *control);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param rxcounter RECV_ONLY
 * @param txcounter RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(
    nvmlDevice_t device, unsigned int link, unsigned int counter,
    unsigned long long *rxcounter, unsigned long long *txcounter);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 * @param freeze SEND_ONLY
 */
nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device,
                                                      unsigned int link,
                                                      unsigned int counter,
                                                      nvmlEnableState_t freeze);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param counter SEND_ONLY
 */
nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device,
                                                     unsigned int link,
                                                     unsigned int counter);
/**
 * @param device SEND_ONLY
 * @param link SEND_ONLY
 * @param pNvLinkDeviceType RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(
    nvmlDevice_t device, unsigned int link,
    nvmlIntNvLinkDeviceType_t *pNvLinkDeviceType);
/**
 * @disabled client - manual client tracks the server owning the event set
 * @param set RECV_ONLY
 */
nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t *set);
/**
 * @disabled client - manual client routes through the device owning the registration
 * @param device SEND_ONLY
 * @param eventTypes SEND_ONLY
 * @param set SEND_ONLY
 */
nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device,
                                      unsigned long long eventTypes,
                                      nvmlEventSet_t set);
/**
 * @param device SEND_ONLY
 * @param eventTypes RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device,
                                              unsigned long long *eventTypes);
/**
 * @disabled client - manual client tracks the server owning the event set
 * @param set SEND_ONLY
 * @param data RECV_ONLY
 * @param timeoutms SEND_ONLY
 */
nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data,
                                 unsigned int timeoutms);
/**
 * @disabled client - manual client tracks the server owning the event set
 * @param set SEND_ONLY
 */
nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set);
/**
 * @param pciInfo SEND_RECV
 * @param newState SEND_ONLY
 */
nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t *pciInfo,
                                        nvmlEnableState_t newState);
/**
 * @param pciInfo SEND_RECV
 * @param currentState RECV_ONLY
 */
nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t *pciInfo,
                                       nvmlEnableState_t *currentState);
/**
 * @param pciInfo SEND_RECV
 * @param gpuState SEND_ONLY
 * @param linkState SEND_ONLY
 */
nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t *pciInfo,
                                    nvmlDetachGpuState_t gpuState,
                                    nvmlPcieLinkState_t linkState);
/**
 * @param pciInfo SEND_RECV
 */
nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t *pciInfo);
/**
 * @param device SEND_ONLY
 * @param valuesCount SEND_ONLY
 * @param values RECV_ONLY LENGTH:valuesCount
 */
nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount,
                                      nvmlFieldValue_t *values);
/**
 * @param device SEND_ONLY
 * @param valuesCount SEND_ONLY
 * @param values RECV_ONLY LENGTH:valuesCount
 */
nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount,
                                        nvmlFieldValue_t *values);
/**
 * @param device SEND_ONLY
 * @param pVirtualMode RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetVirtualizationMode(nvmlDevice_t device,
                                nvmlGpuVirtualizationMode_t *pVirtualMode);
/**
 * @param device SEND_ONLY
 * @param pHostVgpuMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device,
                                       nvmlHostVgpuMode_t *pHostVgpuMode);
/**
 * @param device SEND_ONLY
 * @param virtualMode SEND_ONLY
 */
nvmlReturn_t
nvmlDeviceSetVirtualizationMode(nvmlDevice_t device,
                                nvmlGpuVirtualizationMode_t virtualMode);
/**
 * @param device SEND_ONLY
 * @param pGridLicensableFeatures RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(
    nvmlDevice_t device, nvmlGridLicensableFeatures_t *pGridLicensableFeatures);
/**
 * @param device SEND_ONLY
 * @param processSamplesCount SEND_RECV
 * @param utilization RECV_ONLY LENGTH:processSamplesCount
 * @param lastSeenTimeStamp SEND_ONLY
 */
nvmlReturn_t nvmlDeviceGetProcessUtilization(
    nvmlDevice_t device, nvmlProcessUtilizationSample_t *utilization,
    unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp);
/**
 * @param device SEND_ONLY
 * @param version RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device,
                                             char *version);
/**
 * @param device SEND_ONLY
 * @param isEnabled RECV_ONLY
 * @param defaultMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device,
                                          unsigned int *isEnabled,
                                          unsigned int *defaultMode);
/**
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t
nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability,
                              unsigned int *capResult);
/**
 * @param device SEND_ONLY
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device,
                              nvmlDeviceVgpuCapability_t capability,
                              unsigned int *capResult);
/**
 * @param device SEND_ONLY
 * @param vgpuCount SEND_RECV
 * @param vgpuTypeIds RECV_ONLY LENGTH:vgpuCount
 */
nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device,
                                         unsigned int *vgpuCount,
                                         nvmlVgpuTypeId_t *vgpuTypeIds);
/**
 * @param device SEND_ONLY
 * @param vgpuCount SEND_RECV
 * @param vgpuTypeIds RECV_ONLY LENGTH:vgpuCount
 */
nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device,
                                         unsigned int *vgpuCount,
                                         nvmlVgpuTypeId_t *vgpuTypeIds);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param size RECV_ONLY
 * @param vgpuTypeClass RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId,
                                  char *vgpuTypeClass, unsigned int *size);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param size SEND_RECV
 * @param vgpuTypeName RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId,
                                 char *vgpuTypeName, unsigned int *size);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param gpuInstanceProfileId RECV_ONLY
 */
nvmlReturn_t
nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId,
                                    unsigned int *gpuInstanceProfileId);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param deviceID RECV_ONLY
 * @param subsystemID RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId,
                                     unsigned long long *deviceID,
                                     unsigned long long *subsystemID);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param fbSize RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId,
                                            unsigned long long *fbSize);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param numDisplayHeads RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId,
                                            unsigned int *numDisplayHeads);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param displayIndex SEND_ONLY
 * @param xdim RECV_ONLY
 * @param ydim RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId,
                                       unsigned int displayIndex,
                                       unsigned int *xdim, unsigned int *ydim);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param size SEND_ONLY
 * @param vgpuTypeLicenseString RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId,
                                    char *vgpuTypeLicenseString,
                                    unsigned int size);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param frameRateLimit RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId,
                                           unsigned int *frameRateLimit);
/**
 * @param device SEND_ONLY
 * @param vgpuTypeId SEND_ONLY
 * @param vgpuInstanceCount RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device,
                                         nvmlVgpuTypeId_t vgpuTypeId,
                                         unsigned int *vgpuInstanceCount);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param vgpuInstanceCountPerVm RECV_ONLY
 */
nvmlReturn_t
nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId,
                                 unsigned int *vgpuInstanceCountPerVm);
/**
 * @param device SEND_ONLY
 * @param vgpuCount SEND_RECV
 * @param vgpuInstances RECV_ONLY LENGTH:vgpuCount
 */
nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device,
                                      unsigned int *vgpuCount,
                                      nvmlVgpuInstance_t *vgpuInstances);
/**
 * @param vgpuInstance SEND_ONLY
 * @param size SEND_ONLY
 * @param vmId RECV_ONLY LENGTH:size
 * @param vmIdType RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance,
                                     char *vmId, unsigned int size,
                                     nvmlVgpuVmIdType_t *vmIdType);
/**
 * @param vgpuInstance SEND_ONLY
 * @param size SEND_ONLY
 * @param uuid RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance,
                                     char *uuid, unsigned int size);
/**
 * @param vgpuInstance SEND_ONLY
 * @param length SEND_ONLY
 * @param version RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance,
                                                char *version,
                                                unsigned int length);
/**
 * @param vgpuInstance SEND_ONLY
 * @param fbUsage RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance,
                                        unsigned long long *fbUsage);
/**
 * @param vgpuInstance SEND_ONLY
 * @param licensed RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance,
                                              unsigned int *licensed);
/**
 * @param vgpuInstance SEND_ONLY
 * @param vgpuTypeId RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance,
                                     nvmlVgpuTypeId_t *vgpuTypeId);
/**
 * @param vgpuInstance SEND_ONLY
 * @param frameRateLimit RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance,
                                               unsigned int *frameRateLimit);
/**
 * @param vgpuInstance SEND_ONLY
 * @param eccMode RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance,
                                        nvmlEnableState_t *eccMode);
/**
 * @param vgpuInstance SEND_ONLY
 * @param encoderCapacity RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance,
                                                unsigned int *encoderCapacity);
/**
 * @param vgpuInstance SEND_ONLY
 * @param encoderCapacity SEND_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance,
                                                unsigned int encoderCapacity);
/**
 * @param vgpuInstance SEND_ONLY
 * @param sessionCount RECV_ONLY
 * @param averageFps RECV_ONLY
 * @param averageLatency RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance,
                                             unsigned int *sessionCount,
                                             unsigned int *averageFps,
                                             unsigned int *averageLatency);
/**
 * @param vgpuInstance SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfo RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t
nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance,
                                   unsigned int *sessionCount,
                                   nvmlEncoderSessionInfo_t *sessionInfo);
/**
 * @param vgpuInstance SEND_ONLY
 * @param fbcStats RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance,
                                         nvmlFBCStats_t *fbcStats);
/**
 * @param vgpuInstance SEND_ONLY
 * @param sessionCount SEND_RECV
 * @param sessionInfo RECV_ONLY LENGTH:sessionCount
 */
nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance,
                                            unsigned int *sessionCount,
                                            nvmlFBCSessionInfo_t *sessionInfo);
/**
 * @param vgpuInstance SEND_ONLY
 * @param gpuInstanceId RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance,
                                              unsigned int *gpuInstanceId);
/**
 * @param vgpuInstance SEND_ONLY
 * @param length SEND_RECV
 * @param vgpuPciId RECV_ONLY LENGTH:length
 */
nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance,
                                         char *vgpuPciId, unsigned int *length);
/**
 * @param vgpuTypeId SEND_ONLY
 * @param capability SEND_ONLY
 * @param capResult RECV_ONLY
 */
nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId,
                                         nvmlVgpuCapability_t capability,
                                         unsigned int *capResult);
/**
 * @param vgpuInstance SEND_ONLY
 * @param bufferSize SEND_RECV
 * @param vgpuMetadata RECV_ONLY LENGTH:bufferSize
 */
nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance,
                                         nvmlVgpuMetadata_t *vgpuMetadata,
                                         unsigned int *bufferSize);
/**
 * @param device SEND_ONLY
 * @param bufferSize SEND_RECV
 * @param pgpuMetadata RECV_ONLY LENGTH:bufferSize
 */
nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device,
                                       nvmlVgpuPgpuMetadata_t *pgpuMetadata,
                                       unsigned int *bufferSize);
/**
 * @param vgpuMetadata SEND_RECV
 * @param pgpuMetadata RECV_ONLY
 * @param compatibilityInfo RECV_ONLY
 */
nvmlReturn_t
nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t *vgpuMetadata,
                         nvmlVgpuPgpuMetadata_t *pgpuMetadata,
                         nvmlVgpuPgpuCompatibility_t *compatibilityInfo);
/**
 * @param device SEND_ONLY
 * @param bufferSize SEND_RECV
 * @param pgpuMetadata RECV_ONLY LENGTH:bufferSize
 */
nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device,
                                             char *pgpuMetadata,
                                             unsigned int *bufferSize);
/**
 * @param device SEND_ONLY
 * @param pSchedulerLog RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device,
                              nvmlVgpuSchedulerLog_t *pSchedulerLog);
/**
 * @param device SEND_ONLY
 * @param pSchedulerState RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device,
                                nvmlVgpuSchedulerGetState_t *pSchedulerState);
/**
 * @param device SEND_ONLY
 * @param pCapabilities RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(
    nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t *pCapabilities);
/**
 * @param supported RECV_ONLY
 * @param current RECV_ONLY
 */
nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t *supported,
                                nvmlVgpuVersion_t *current);
/**
 * @param vgpuVersion RECV_ONLY
 */
nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t *vgpuVersion);
/**
 * @param device SEND_ONLY
 * @param lastSeenTimeStamp SEND_ONLY
 * @param sampleValType SEND_RECV
 * @param vgpuInstanceSamplesCount SEND_RECV
 * @param utilizationSamples RECV_ONLY LENGTH:vgpuInstanceSamplesCount
 */
nvmlReturn_t nvmlDeviceGetVgpuUtilization(
    nvmlDevice_t device, unsigned long long lastSeenTimeStamp,
    nvmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount,
    nvmlVgpuInstanceUtilizationSample_t *utilizationSamples);
/**
 * @param device SEND_ONLY
 * @param lastSeenTimeStamp SEND_ONLY
 * @param vgpuProcessSamplesCount SEND_RECV
 * @param utilizationSamples RECV_ONLY LENGTH:vgpuProcessSamplesCount
 */
nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(
    nvmlDevice_t device, unsigned long long lastSeenTimeStamp,
    unsigned int *vgpuProcessSamplesCount,
    nvmlVgpuProcessUtilizationSample_t *utilizationSamples);
/**
 * @param vgpuInstance SEND_ONLY
 * @param mode RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance,
                                               nvmlEnableState_t *mode);
/**
 * @param vgpuInstance SEND_ONLY
 * @param count SEND_RECV
 * @param pids RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance,
                                               unsigned int *count,
                                               unsigned int *pids);
/**
 * @param vgpuInstance SEND_ONLY
 * @param pid SEND_ONLY
 * @param stats RECV_ONLY
 */
nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance,
                                                unsigned int pid,
                                                nvmlAccountingStats_t *stats);
/**
 * @param vgpuInstance SEND_ONLY
 */
nvmlReturn_t
nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance);
/**
 * @param vgpuInstance SEND_ONLY
 * @param licenseInfo RECV_ONLY
 */
nvmlReturn_t
nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance,
                                  nvmlVgpuLicenseInfo_t *licenseInfo);
/**
 * @param deviceCount RECV_ONLY
 */
nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int *deviceCount);
/**
 * @param index SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index,
                                              nvmlExcludedDeviceInfo_t *info);
/**
 * @param device SEND_ONLY
 * @param mode SEND_ONLY
 * @param activationStatus RECV_ONLY
 */
nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode,
                                  nvmlReturn_t *activationStatus);
/**
 * @param device SEND_ONLY
 * @param currentMode RECV_ONLY
 * @param pendingMode RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device,
                                  unsigned int *currentMode,
                                  unsigned int *pendingMode);
/**
 * @param device SEND_ONLY
 * @param profile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile,
                                    nvmlGpuInstanceProfileInfo_t *info);
/**
 * @param device SEND_ONLY
 * @param profile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile,
                                     nvmlGpuInstanceProfileInfo_v2_t *info);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param placements RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(
    nvmlDevice_t device, unsigned int profileId,
    nvmlGpuInstancePlacement_t *placements, unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device,
                                                       unsigned int profileId,
                                                       unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param gpuInstance RECV_ONLY
 */
nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device,
                                         unsigned int profileId,
                                         nvmlGpuInstance_t *gpuInstance);
/**
 * @param gpuInstance SEND_ONLY
 */
nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance);
/**
 * @param device SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param gpuInstances RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device,
                                       unsigned int profileId,
                                       nvmlGpuInstance_t *gpuInstances,
                                       unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param id SEND_ONLY
 * @param gpuInstance RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id,
                                          nvmlGpuInstance_t *gpuInstance);
/**
 * @param gpuInstance SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance,
                                    nvmlGpuInstanceInfo_t *info);
/**
 * @param gpuInstance SEND_ONLY
 * @param profile SEND_ONLY
 * @param engProfile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(
    nvmlGpuInstance_t gpuInstance, unsigned int profile,
    unsigned int engProfile, nvmlComputeInstanceProfileInfo_t *info);
/**
 * @param gpuInstance SEND_ONLY
 * @param profile SEND_ONLY
 * @param engProfile SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(
    nvmlGpuInstance_t gpuInstance, unsigned int profile,
    unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t *info);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(
    nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int *count);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param placements RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(
    nvmlGpuInstance_t gpuInstance, unsigned int profileId,
    nvmlComputeInstancePlacement_t *placements, unsigned int *count);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param computeInstance RECV_ONLY
 */
nvmlReturn_t
nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance,
                                     unsigned int profileId,
                                     nvmlComputeInstance_t *computeInstance);
/**
 * @param computeInstance SEND_ONLY
 */
nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance);
/**
 * @param gpuInstance SEND_ONLY
 * @param profileId SEND_ONLY
 * @param count SEND_RECV
 * @param computeInstances RECV_ONLY LENGTH:count
 */
nvmlReturn_t nvmlGpuInstanceGetComputeInstances(
    nvmlGpuInstance_t gpuInstance, unsigned int profileId,
    nvmlComputeInstance_t *computeInstances, unsigned int *count);
/**
 * @param gpuInstance SEND_ONLY
 * @param id SEND_ONLY
 * @param computeInstance RECV_ONLY
 */
nvmlReturn_t
nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance,
                                      unsigned int id,
                                      nvmlComputeInstance_t *computeInstance);
/**
 * @param computeInstance SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t
nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance,
                              nvmlComputeInstanceInfo_t *info);
/**
 * @param device SEND_ONLY
 * @param isMigDevice RECV_ONLY
 */
nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device,
                                         unsigned int *isMigDevice);
/**
 * @param device SEND_ONLY
 * @param id RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int *id);
/**
 * @param device SEND_ONLY
 * @param id RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device,
                                            unsigned int *id);
/**
 * @param device SEND_ONLY
 * @param count RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device,
                                            unsigned int *count);
/**
 * @param device SEND_ONLY
 * @param index SEND_ONLY
 * @param migDevice RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device,
                                                 unsigned int index,
                                                 nvmlDevice_t *migDevice);
/**
 * @param migDevice SEND_ONLY
 * @param device RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice,
                                             nvmlDevice_t *device);
/**
 * @param device SEND_ONLY
 * @param type RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t *type);
/**
 * @param device SEND_ONLY
 * @param pDynamicPstatesInfo RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(
    nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t *pDynamicPstatesInfo);
/**
 * @param device SEND_ONLY
 * @param fan SEND_ONLY
 * @param speed SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan,
                                      unsigned int speed);
/**
 * @param device SEND_ONLY
 * @param offset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int *offset);
/**
 * @param device SEND_ONLY
 * @param offset SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset);
/**
 * @param device SEND_ONLY
 * @param offset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int *offset);
/**
 * @param device SEND_ONLY
 * @param offset SEND_ONLY
 */
nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset);
/**
 * @param device SEND_ONLY
 * @param type SEND_ONLY
 * @param pstate SEND_ONLY
 * @param minClockMHz RECV_ONLY
 * @param maxClockMHz RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device,
                                              nvmlClockType_t type,
                                              nvmlPstates_t pstate,
                                              unsigned int *minClockMHz,
                                              unsigned int *maxClockMHz);
/**
 * @param device SEND_ONLY
 * @param size SEND_ONLY
 * @param pstates RECV_ONLY LENGTH:size
 */
nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device,
                                                     nvmlPstates_t *pstates,
                                                     unsigned int size);
/**
 * @param device SEND_ONLY
 * @param minOffset RECV_ONLY
 * @param maxOffset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device,
                                               int *minOffset, int *maxOffset);
/**
 * @param device SEND_ONLY
 * @param minOffset RECV_ONLY
 * @param maxOffset RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device,
                                               int *minOffset, int *maxOffset);
/**
 * @param device SEND_ONLY
 * @param gpuFabricInfo RECV_ONLY
 */
nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device,
                                        nvmlGpuFabricInfo_t *gpuFabricInfo);
/**
 * @param metricsGet RECV_ONLY
 */
nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t *metricsGet);
/**
 * @param gpmSample SEND_ONLY
 */
nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample);
/**
 * @param gpmSample RECV_ONLY
 */
nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t *gpmSample);
/**
 * @param device SEND_ONLY
 * @param gpmSample SEND_ONLY
 */
nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample);
/**
 * @param device SEND_ONLY
 * @param gpuInstanceId SEND_ONLY
 * @param gpmSample SEND_ONLY
 */
nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device,
                                 unsigned int gpuInstanceId,
                                 nvmlGpmSample_t gpmSample);
/**
 * @param device SEND_ONLY
 * @param gpmSupport RECV_ONLY
 */
nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device,
                                       nvmlGpmSupport_t *gpmSupport);
/**
 * @param device SEND_ONLY
 * @param info RECV_ONLY
 */
nvmlReturn_t
nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device,
                                           nvmlNvLinkPowerThres_t *info);

// Registry-only operations without API declarations above. The code generator
// reads these annotations directly; the C++ parser intentionally ignores them.
#if 0
/** @disabled */
void nvmlDeviceGetComputeRunningProcesses();
/** @disabled */
void nvmlDeviceGetComputeRunningProcesses_v2();
/** @disabled */
void nvmlDeviceGetGraphicsRunningProcesses();
/** @disabled */
void nvmlDeviceGetGraphicsRunningProcesses_v2();
/** @disabled */
void nvmlDeviceGetMPSComputeRunningProcesses();
/** @disabled */
void nvmlDeviceGetMPSComputeRunningProcesses_v2();
#endif
