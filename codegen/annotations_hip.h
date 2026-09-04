#include <hip/hip_runtime_api.h>

// HIP runtime API. The initial generated surface covers device discovery and
// properties.

/**
 * @server HIP
 * @disabled client - manual client initializes every configured route
 * @param flags SEND_ONLY
 */
hipError_t hipInit(unsigned int flags);
/**
 * @server HIP
 * @disabled client - manual client reports the virtual device table size
 * @param count RECV_ONLY
 */
hipError_t hipGetDeviceCount(int *count);
/**
 * @server HIP
 * @disabled client - manual client maps the virtual device ordinal
 * @param device RECV_ONLY
 * @param ordinal SEND_ONLY
 */
hipError_t hipDeviceGet(int *device, int ordinal);
/**
 * @server HIP
 * @param prop RECV_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_tR0600 *prop,
                                       int deviceId);
/**
 * @server HIP
 * @param name RECV_ONLY LENGTH:len
 * @param len SEND_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceGetName(char *name, int len, int deviceId);
/**
 * @server HIP
 * @param bytes RECV_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceTotalMem(size_t *bytes, int deviceId);
/**
 * @server HIP
 * @param pi RECV_ONLY
 * @param attr SEND_ONLY
 * @param deviceId SEND_ONLY
 * @routingkey HIP_DEVICE deviceId
 */
hipError_t hipDeviceGetAttribute(int *pi, hipDeviceAttribute_t attr,
                                 int deviceId);
/**
 * @server HIP
 * @param driverVersion RECV_ONLY
 */
hipError_t hipDriverGetVersion(int *driverVersion);
/**
 * @server HIP
 * @param runtimeVersion RECV_ONLY
 */
hipError_t hipRuntimeGetVersion(int *runtimeVersion);
