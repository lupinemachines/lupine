#include <hip/hip_runtime_api.h>

// HIP runtime API. The initial generated surface covers device discovery and
// properties.

/**
 * @disabled client - manual client initializes every configured route
 * @param flags SEND_ONLY
 */
hipError_t hipInit(unsigned int flags);
/**
 * @disabled client - manual client reports the virtual device table size
 * @param count RECV_ONLY
 */
hipError_t hipGetDeviceCount(int *count);
/**
 * @disabled client - manual client maps the virtual device ordinal
 * @param device RECV_ONLY
 * @param ordinal SEND_ONLY
 */
hipError_t hipDeviceGet(int *device, int ordinal);
/**
 * @param prop RECV_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 * @clearfields prop hdpMemFlushCntl hdpRegFlushCntl
 */
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600 *prop,
                                       int deviceId);
/**
 * @param name RECV_ONLY LENGTH:len
 * @param len SEND_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceGetName(char *name, int len, int deviceId);
/**
 * @param bytes RECV_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceTotalMem(size_t *bytes, int deviceId);
/**
 * @param pi RECV_ONLY
 * @param attr SEND_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId);
/**
 * @param driverVersion RECV_ONLY
 */
hipError_t hipDriverGetVersion(int *driverVersion);
/**
 * @param runtimeVersion RECV_ONLY
 */
hipError_t hipRuntimeGetVersion(int *runtimeVersion);
