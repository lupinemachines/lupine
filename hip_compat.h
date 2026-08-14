#ifndef LUPINE_HIP_COMPAT_H
#define LUPINE_HIP_COMPAT_H

// HIP version-compat shims, analogous to cuda_compat.h. The generated HIP
// client/server units (gen_hip_client.inc / gen_hip_server.inc) and the
// hand-written hip_client.cpp / hip_server.cpp pull in the HIP types through
// this header.
//
// Rather than depending on a ROCm/HIP SDK install at build time, the types the
// v1 surface marshals are vendored here directly from the ROCm 6.0 (R0600)
// headers. The R0600 suffix is HIP's versioned-ABI tag: the struct layout is
// frozen for that version, so a faithful copy is wire- and ABI-compatible with
// any ROCm 6.x runtime the server later dlopen's (and any real libamdhip64 a
// consumer links the shim against). The server loads amdhip64 at runtime via
// LoadLibrary/dlopen, so no HIP SDK is required to *build* lupine -- the AMD
// backend is compiled unconditionally and simply binds at runtime if a ROCm
// install is present, mirroring how the CUDA/NVML clients get their types from
// the always-required CUDA toolkit.
//
// Only the three types the generated code references by name are vendored
// (hipError_t, hipDeviceProp_tR0600, hipDeviceAttribute_t) plus the two they
// are composed of (hipUUID, hipDeviceArch_t). The v1 surface marshals whole
// structs/opaque values and never reaches into struct members from generated
// code, so no field shims are needed yet. Add them here as compute-path
// functions land.
//
// __HIP_PLATFORM_AMD__ is the conventional marker that these are the
// AMD-platform definitions; it is defined here once for every TU that needs the
// HIP types, matching what hip_runtime.h would have required.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <cstddef>  // size_t

// ---------------------------------------------------------------------------
// hipUUID (hip_runtime_api.h)
// ---------------------------------------------------------------------------
typedef struct hipUUID_t {
  char bytes[16];
} hipUUID;

// ---------------------------------------------------------------------------
// hipDeviceArch_t (hip_runtime_api.h) -- architectural feature flags.
// ---------------------------------------------------------------------------
typedef struct {
  unsigned hasGlobalInt32Atomics : 1;
  unsigned hasGlobalFloatAtomicExch : 1;
  unsigned hasSharedInt32Atomics : 1;
  unsigned hasSharedFloatAtomicExch : 1;
  unsigned hasFloatAtomicAdd : 1;
  unsigned hasGlobalInt64Atomics : 1;
  unsigned hasSharedInt64Atomics : 1;
  unsigned hasDoubles : 1;
  unsigned hasWarpVote : 1;
  unsigned hasWarpBallot : 1;
  unsigned hasWarpShuffle : 1;
  unsigned hasFunnelShift : 1;
  unsigned hasThreadFenceSystem : 1;
  unsigned hasSyncThreadsExt : 1;
  unsigned hasSurfaceFuncs : 1;
  unsigned has3dGrid : 1;
  unsigned hasDynamicParallelism : 1;
} hipDeviceArch_t;

// ---------------------------------------------------------------------------
// hipDeviceProp_tR0600 (hip_runtime_api.h, the R0600 versioned struct).
// Layout-faithful copy; do not reorder or resize fields.
// ---------------------------------------------------------------------------
typedef struct hipDeviceProp_tR0600 {
  char name[256];                   ///< Device name.
  hipUUID uuid;                     ///< UUID of a device.
  char luid[8];                     ///< 8-byte unique identifier (Windows only).
  unsigned int luidDeviceNodeMask;  ///< LUID node mask.
  size_t totalGlobalMem;            ///< Size of global memory region (bytes).
  size_t sharedMemPerBlock;         ///< Size of shared memory per block (bytes).
  int regsPerBlock;                 ///< Registers per block.
  int warpSize;                     ///< Warp size.
  size_t memPitch;                  ///< Max pitch in bytes for pitched memory copies.
  int maxThreadsPerBlock;           ///< Max work items per work group.
  int maxThreadsDim[3];             ///< Max threads per dimension (XYZ) of a block.
  int maxGridSize[3];               ///< Max grid dimensions (XYZ).
  int clockRate;                    ///< Max clock frequency of the multiProcessors (kHz).
  size_t totalConstMem;             ///< Size of constant memory region (bytes).
  int major;                       ///< Major compute capability.
  int minor;                       ///< Minor compute capability.
  size_t textureAlignment;          ///< Alignment requirement for textures.
  size_t texturePitchAlignment;    ///< Pitch alignment for texture references.
  int deviceOverlap;               ///< Deprecated. Use asyncEngineCount.
  int multiProcessorCount;         ///< Number of multi-processors (compute units).
  int kernelExecTimeoutEnabled;    ///< Run time limit for kernels on the device.
  int integrated;                  ///< APU vs dGPU.
  int canMapHostMemory;            ///< Whether HIP can map host memory.
  int computeMode;                 ///< Compute mode.
  int maxTexture1D;                ///< Max elements in 1D images.
  int maxTexture1DMipmap;          ///< Max 1D mipmap texture size.
  int maxTexture1DLinear;          ///< Max size for 1D textures bound to linear memory.
  int maxTexture2D[2];             ///< Max dimensions (w,h) of 2D images.
  int maxTexture2DMipmap[2];        ///< Max elements in 2D array mipmap images.
  int maxTexture2DLinear[3];        ///< Max 2D tex dims for pitched-memory textures.
  int maxTexture2DGather[2];        ///< Max 2D tex dims if gather is performed.
  int maxTexture3D[3];             ///< Max dimensions (w,h,d) of 3D images.
  int maxTexture3DAlt[3];           ///< Max alternate 3D texture dims.
  int maxTextureCubemap;            ///< Max cubemap texture dims.
  int maxTexture1DLayered[2];      ///< Max elements in 1D array images.
  int maxTexture2DLayered[3];      ///< Max elements in 2D array images.
  int maxTextureCubemapLayered[2]; ///< Max cubemaps layered texture dims.
  int maxSurface1D;                ///< Max 1D surface size.
  int maxSurface2D[2];             ///< Max 2D surface size.
  int maxSurface3D[3];             ///< Max 3D surface size.
  int maxSurface1DLayered[2];      ///< Max 1D layered surface size.
  int maxSurface2DLayered[3];      ///< Max 2D layered surface size.
  int maxSurfaceCubemap;           ///< Max cubemap surface size.
  int maxSurfaceCubemapLayered[2]; ///< Max cubemap layered surface size.
  size_t surfaceAlignment;         ///< Alignment requirement for surfaces.
  int concurrentKernels;          ///< Device can execute multiple kernels concurrently.
  int ECCEnabled;                 ///< Device has ECC support enabled.
  int pciBusID;                   ///< PCI Bus ID.
  int pciDeviceID;                ///< PCI Device ID.
  int pciDomainID;                ///< PCI Domain ID.
  int tccDriver;                  ///< 1 if Tesla TCC driver, else 0.
  int asyncEngineCount;           ///< Number of async engines.
  int unifiedAddressing;          ///< Device and host share unified address space.
  int memoryClockRate;            ///< Max global memory clock frequency (kHz).
  int memoryBusWidth;             ///< Global memory bus width (bits).
  int l2CacheSize;                ///< L2 cache size.
  int persistingL2CacheMaxSize;   ///< Device's max L2 persisting lines (bytes).
  int maxThreadsPerMultiProcessor;     ///< Max resident threads per multi-processor.
  int streamPrioritiesSupported;       ///< Device supports stream priority.
  int globalL1CacheSupported;          ///< Globals are cached in L1.
  int localL1CacheSupported;           ///< Locals are cached in L1.
  size_t sharedMemPerMultiprocessor;   ///< Shared memory per multiprocessor.
  int regsPerMultiprocessor;           ///< Registers available per multiprocessor.
  int managedMemory;              ///< Device supports allocating managed memory.
  int isMultiGpuBoard;            ///< 1 if on a multi-GPU board.
  int multiGpuBoardGroupID;       ///< Group ID for devices on the same multiboard GPU.
  int hostNativeAtomicSupported;         ///< Host-device link supports native atomics.
  int singleToDoublePrecisionPerfRatio;  ///< Deprecated. CUDA only.
  int pageableMemoryAccess;              ///< Coherent access to pageable memory.
  int concurrentManagedAccess;           ///< Coherent access to managed memory.
  int computePreemptionSupported;       ///< Compute preemption supported.
  int canUseHostPointerForRegisteredMem; ///< Host-registered memory at same address.
  int cooperativeLaunch;                 ///< Supports cooperative launch.
  int cooperativeMultiDeviceLaunch;      ///< Supports multi-device cooperative launch.
  size_t sharedMemPerBlockOptin;         ///< Opt-in shared mem per block.
  int pageableMemoryAccessUsesHostPageTables;  ///< Pageable access via host page tables.
  int directManagedMemAccessFromHost;          ///< Host direct access to managed memory.
  int maxBlocksPerMultiProcessor;              ///< Max blocks per CU.
  int accessPolicyMaxWindowSize;               ///< Max access policy window value.
  size_t reservedSharedMemPerBlock;            ///< Shared mem reserved by driver per block.
  int hostRegisterSupported;           ///< Device supports hipHostRegister.
  int sparseHipArraySupported;         ///< Device supports sparse hip arrays.
  int hostRegisterReadOnlySupported;   ///< Supports hipHostRegisterReadOnly.
  int timelineSemaphoreInteropSupported;  ///< External timeline semaphore support.
  int memoryPoolsSupported;            ///< Supports hipMallocAsync and hipMemPool APIs.
  int gpuDirectRDMASupported;          ///< Supports RDMA APIs.
  unsigned int gpuDirectRDMAFlushWritesOptions;  ///< hipFlushGPUDirectRDMAWritesOptions bitmask.
  int gpuDirectRDMAWritesOrdering;               ///< hipGPUDirectRDMAWritesOrdering value.
  unsigned int memoryPoolSupportedHandleTypes;   ///< Mempool IPC handle-type bitmask.
  int deferredMappingHipArraySupported;  ///< Supports deferred-mapping HIP arrays.
  int ipcEventSupported;                 ///< Supports IPC events.
  int clusterLaunch;                     ///< Supports cluster launch.
  int unifiedFunctionPointers;           ///< Supports unified function pointers.
  int reserved[63];                      ///< CUDA Reserved.
  int hipReserved[32];                   ///< Reserved for new HIP/CUDA entries.

  /* HIP-only struct members. */
  char gcnArchName[256];                    ///< AMD GCN arch name (HIP only).
  size_t maxSharedMemoryPerMultiProcessor;  ///< Max shared memory per CU (HIP only).
  int clockInstructionRate;  ///< Frequency (kHz) of the device-side clock* instructions.
  hipDeviceArch_t arch;      ///< Architectural feature flags (HIP only).
  unsigned int *hdpMemFlushCntl;  ///< Address of HDP_MEM_COHERENCY_FLUSH_CNTL register.
  unsigned int *hdpRegFlushCntl;  ///< Address of HDP_REG_COHERENCY_FLUSH_CNTL register.
  int cooperativeMultiDeviceUnmatchedFunc;        ///< Multi-device cooperative launch, unmatched funcs.
  int cooperativeMultiDeviceUnmatchedGridDim;      ///< ...unmatched grid dimensions.
  int cooperativeMultiDeviceUnmatchedBlockDim;     ///< ...unmatched block dimensions.
  int cooperativeMultiDeviceUnmatchedSharedMem;    ///< ...unmatched shared memories.
  int isLargeBar;  ///< 1 if large PCI bar device, else 0.
  int asicRevision;  ///< Revision of the GPU in this device.
} hipDeviceProp_tR0600;

// ---------------------------------------------------------------------------
// hipError_t (hip_runtime_api.h). Enum values are forwarded to the real
// runtime by the server, so they must match ROCm exactly.
// ---------------------------------------------------------------------------
typedef enum hipError_t {
  hipSuccess = 0,
  hipErrorInvalidValue = 1,
  hipErrorOutOfMemory = 2,
  hipErrorMemoryAllocation = 2,  // deprecated alias
  hipErrorNotInitialized = 3,
  hipErrorInitializationError = 3,  // deprecated alias
  hipErrorDeinitialized = 4,
  hipErrorProfilerDisabled = 5,
  hipErrorProfilerNotInitialized = 6,
  hipErrorProfilerAlreadyStarted = 7,
  hipErrorProfilerAlreadyStopped = 8,
  hipErrorInvalidConfiguration = 9,
  hipErrorInvalidPitchValue = 12,
  hipErrorInvalidSymbol = 13,
  hipErrorInvalidDevicePointer = 17,
  hipErrorInvalidMemcpyDirection = 21,
  hipErrorInsufficientDriver = 35,
  hipErrorMissingConfiguration = 52,
  hipErrorPriorLaunchFailure = 53,
  hipErrorInvalidDeviceFunction = 98,
  hipErrorNoDevice = 100,
  hipErrorInvalidDevice = 101,
  hipErrorInvalidImage = 200,
  hipErrorInvalidContext = 201,
  hipErrorContextAlreadyCurrent = 202,
  hipErrorMapFailed = 205,
  hipErrorMapBufferObjectFailed = 205,  // deprecated alias
  hipErrorUnmapFailed = 206,
  hipErrorArrayIsMapped = 207,
  hipErrorAlreadyMapped = 208,
  hipErrorNoBinaryForGpu = 209,
  hipErrorAlreadyAcquired = 210,
  hipErrorNotMapped = 211,
  hipErrorNotMappedAsArray = 212,
  hipErrorNotMappedAsPointer = 213,
  hipErrorECCNotCorrectable = 214,
  hipErrorUnsupportedLimit = 215,
  hipErrorContextAlreadyInUse = 216,
  hipErrorPeerAccessUnsupported = 217,
  hipErrorInvalidKernelFile = 218,
  hipErrorInvalidGraphicsContext = 219,
  hipErrorInvalidSource = 300,
  hipErrorFileNotFound = 301,
  hipErrorSharedObjectSymbolNotFound = 302,
  hipErrorSharedObjectInitFailed = 303,
  hipErrorOperatingSystem = 304,
  hipErrorInvalidHandle = 400,
  hipErrorInvalidResourceHandle = 400,  // deprecated alias
  hipErrorIllegalState = 401,
  hipErrorNotFound = 500,
  hipErrorNotReady = 600,
  hipErrorIllegalAddress = 700,
  hipErrorLaunchOutOfResources = 701,
  hipErrorLaunchTimeOut = 702,
  hipErrorPeerAccessAlreadyEnabled = 704,
  hipErrorPeerAccessNotEnabled = 705,
  hipErrorSetOnActiveProcess = 708,
  hipErrorContextIsDestroyed = 709,
  hipErrorAssert = 710,
  hipErrorHostMemoryAlreadyRegistered = 712,
  hipErrorHostMemoryNotRegistered = 713,
  hipErrorLaunchFailure = 719,
  hipErrorCooperativeLaunchTooLarge = 720,
  hipErrorNotSupported = 801,
  hipErrorStreamCaptureUnsupported = 900,
  hipErrorStreamCaptureInvalidated = 901,
  hipErrorStreamCaptureMerge = 902,
  hipErrorStreamCaptureUnmatched = 903,
  hipErrorStreamCaptureUnjoined = 904,
  hipErrorStreamCaptureIsolation = 905,
  hipErrorStreamCaptureImplicit = 906,
  hipErrorCapturedEvent = 907,
  hipErrorStreamCaptureWrongThread = 908,
  hipErrorGraphExecUpdateFailure = 910,
  hipErrorInvalidChannelDescriptor = 911,
  hipErrorInvalidTexture = 912,
  hipErrorUnknown = 999,
  hipErrorRuntimeMemory = 1052,
  hipErrorRuntimeOther = 1053,
  hipErrorTbd  // marker that more error codes are needed.
} hipError_t;

// ---------------------------------------------------------------------------
// hipDeviceAttribute_t (hip_runtime_api.h). Enum values are forwarded to the
// real runtime by the server, so they must match ROCm exactly.
// ---------------------------------------------------------------------------
typedef enum hipDeviceAttribute_t {
  hipDeviceAttributeCudaCompatibleBegin = 0,

  hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin,
  hipDeviceAttributeAccessPolicyMaxWindowSize,
  hipDeviceAttributeAsyncEngineCount,
  hipDeviceAttributeCanMapHostMemory,
  hipDeviceAttributeCanUseHostPointerForRegisteredMem,
  hipDeviceAttributeClockRate,
  hipDeviceAttributeComputeMode,
  hipDeviceAttributeComputePreemptionSupported,
  hipDeviceAttributeConcurrentKernels,
  hipDeviceAttributeConcurrentManagedAccess,
  hipDeviceAttributeCooperativeLaunch,
  hipDeviceAttributeCooperativeMultiDeviceLaunch,
  hipDeviceAttributeDeviceOverlap,
  hipDeviceAttributeDirectManagedMemAccessFromHost,
  hipDeviceAttributeGlobalL1CacheSupported,
  hipDeviceAttributeHostNativeAtomicSupported,
  hipDeviceAttributeIntegrated,
  hipDeviceAttributeIsMultiGpuBoard,
  hipDeviceAttributeKernelExecTimeout,
  hipDeviceAttributeL2CacheSize,
  hipDeviceAttributeLocalL1CacheSupported,
  hipDeviceAttributeLuid,
  hipDeviceAttributeLuidDeviceNodeMask,
  hipDeviceAttributeComputeCapabilityMajor,
  hipDeviceAttributeManagedMemory,
  hipDeviceAttributeMaxBlocksPerMultiProcessor,
  hipDeviceAttributeMaxBlockDimX,
  hipDeviceAttributeMaxBlockDimY,
  hipDeviceAttributeMaxBlockDimZ,
  hipDeviceAttributeMaxGridDimX,
  hipDeviceAttributeMaxGridDimY,
  hipDeviceAttributeMaxGridDimZ,
  hipDeviceAttributeMaxSurface1D,
  hipDeviceAttributeMaxSurface1DLayered,
  hipDeviceAttributeMaxSurface2D,
  hipDeviceAttributeMaxSurface2DLayered,
  hipDeviceAttributeMaxSurface3D,
  hipDeviceAttributeMaxSurfaceCubemap,
  hipDeviceAttributeMaxSurfaceCubemapLayered,
  hipDeviceAttributeMaxTexture1DWidth,
  hipDeviceAttributeMaxTexture1DLayered,
  hipDeviceAttributeMaxTexture1DLinear,
  hipDeviceAttributeMaxTexture1DMipmap,
  hipDeviceAttributeMaxTexture2DWidth,
  hipDeviceAttributeMaxTexture2DHeight,
  hipDeviceAttributeMaxTexture2DGather,
  hipDeviceAttributeMaxTexture2DLayered,
  hipDeviceAttributeMaxTexture2DLinear,
  hipDeviceAttributeMaxTexture2DMipmap,
  hipDeviceAttributeMaxTexture3DWidth,
  hipDeviceAttributeMaxTexture3DHeight,
  hipDeviceAttributeMaxTexture3DDepth,
  hipDeviceAttributeMaxTexture3DAlt,
  hipDeviceAttributeMaxTextureCubemap,
  hipDeviceAttributeMaxTextureCubemapLayered,
  hipDeviceAttributeMaxThreadsDim,
  hipDeviceAttributeMaxThreadsPerBlock,
  hipDeviceAttributeMaxThreadsPerMultiProcessor,
  hipDeviceAttributeMaxPitch,
  hipDeviceAttributeMemoryBusWidth,
  hipDeviceAttributeMemoryClockRate,
  hipDeviceAttributeComputeCapabilityMinor,
  hipDeviceAttributeMultiGpuBoardGroupID,
  hipDeviceAttributeMultiprocessorCount,
  hipDeviceAttributeUnused1,
  hipDeviceAttributePageableMemoryAccess,
  hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
  hipDeviceAttributePciBusId,
  hipDeviceAttributePciDeviceId,
  hipDeviceAttributePciDomainID,
  hipDeviceAttributePersistingL2CacheMaxSize,
  hipDeviceAttributeMaxRegistersPerBlock,
  hipDeviceAttributeMaxRegistersPerMultiprocessor,
  hipDeviceAttributeReservedSharedMemPerBlock,
  hipDeviceAttributeMaxSharedMemoryPerBlock,
  hipDeviceAttributeSharedMemPerBlockOptin,
  hipDeviceAttributeSharedMemPerMultiprocessor,
  hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
  hipDeviceAttributeStreamPrioritiesSupported,
  hipDeviceAttributeSurfaceAlignment,
  hipDeviceAttributeTccDriver,
  hipDeviceAttributeTextureAlignment,
  hipDeviceAttributeTexturePitchAlignment,
  hipDeviceAttributeTotalConstantMemory,
  hipDeviceAttributeTotalGlobalMem,
  hipDeviceAttributeUnifiedAddressing,
  hipDeviceAttributeUnused2,
  hipDeviceAttributeWarpSize,
  hipDeviceAttributeMemoryPoolsSupported,
  hipDeviceAttributeVirtualMemoryManagementSupported,
  hipDeviceAttributeHostRegisterSupported,
  hipDeviceAttributeMemoryPoolSupportedHandleTypes,

  hipDeviceAttributeCudaCompatibleEnd = 9999,
  hipDeviceAttributeAmdSpecificBegin = 10000,

  hipDeviceAttributeClockInstructionRate = hipDeviceAttributeAmdSpecificBegin,
  hipDeviceAttributeUnused3,
  hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
  hipDeviceAttributeUnused4,
  hipDeviceAttributeUnused5,
  hipDeviceAttributeHdpMemFlushCntl,
  hipDeviceAttributeHdpRegFlushCntl,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
  hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
  hipDeviceAttributeIsLargeBar,
  hipDeviceAttributeAsicRevision,
  hipDeviceAttributeCanUseStreamWaitValue,
  hipDeviceAttributeImageSupport,
  hipDeviceAttributePhysicalMultiProcessorCount,
  hipDeviceAttributeFineGrainSupport,
  hipDeviceAttributeWallClockRate,

  hipDeviceAttributeAmdSpecificEnd = 19999,
  hipDeviceAttributeVendorSpecificBegin = 20000,
} hipDeviceAttribute_t;

#endif  // LUPINE_HIP_COMPAT_H