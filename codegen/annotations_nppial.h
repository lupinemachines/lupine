#include <npp.h>

// NPP image arithmetic and logical operations (libnppial).
//
// Images, masks and device constants are device memory and travel as
// addresses. C arrays of constants are host memory and travel by contents.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppial by name at runtime. A
// call with a stream context goes to the server that owns the context's stream,
// or to the runtime's current device when the context names the default stream;
// a call without one goes to the current device.
//
// Each @guard names the NPP releases that export the call: the calls without a
// stream context end with NPP 12, and later calls start at the release that
// added them. A call NPP exports without declaring it in any release is left
// out.

/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_C1IRSfs_Ctx(Npp8u nConstant, Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_C1IRSfs(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_C1IRSfs_Ctx(const Npp8u *pConstant, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[3], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_C3IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                        const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_AC4IRSfs_Ctx(const Npp8u *pConstants,
                                         Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[4], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_8u_C4IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_C1IRSfs_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_C1IRSfs(Npp16u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_C1IRSfs_Ctx(const Npp16u *pConstant,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_C3IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                         const Npp16u *pConstants, Npp16u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_AC4IRSfs_Ctx(const Npp16u *pConstants,
                                          Npp16u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16u_C4IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_C1IRSfs_Ctx(Npp16s nConstant, Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_C1IRSfs(Npp16s nConstant, Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_C1IRSfs_Ctx(const Npp16s *pConstant,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[3], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_C3IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                               const Npp16s aConstants[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                         const Npp16s *pConstants, Npp16s *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_AC4IRSfs_Ctx(const Npp16s *pConstants,
                                          Npp16s *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[4], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16s_C4IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   Npp16sc nConstant, Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               Npp16sc nConstant, Npp16sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16sc_C1IRSfs_Ctx(Npp16sc nConstant, Npp16sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16sc_C1IRSfs(Npp16sc nConstant, Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc aConstants[3], Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc aConstants[3], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3],
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                    const Npp16sc aConstants[3], Npp16sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                                const Npp16sc aConstants[3], Npp16sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3],
                                     Npp16sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32s_C1IRSfs_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32s_C1IRSfs(Npp32s nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32s_C1IRSfs_Ctx(const Npp32s *pConstant,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32s aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstants, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32s_C3IRSfs_Ctx(const Npp32s *pConstants,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   Npp32sc nConstant, Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               Npp32sc nConstant, Npp32sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32sc_C1IRSfs_Ctx(Npp32sc nConstant, Npp32sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32sc_C1IRSfs(Npp32sc nConstant, Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc aConstants[3], Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc aConstants[3], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3],
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                    const Npp32sc aConstants[3], Npp32sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                                const Npp32sc aConstants[3], Npp32sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3],
                                     Npp32sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAddC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_16f_C1R(const Npp16f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp16f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16f_C1IR_Ctx(Npp32f nConstant, Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_16f_C1IR(Npp32f nConstant, Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16f_C1IR_Ctx(const Npp32f *pConstant, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_16f_C3IR(const Npp32f aConstants[3], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16f_C3IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_16f_C4IR(const Npp32f aConstants[4], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_16f_C4IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_C1IR_Ctx(Npp32f nConstant, Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_C1IR(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_C1IR_Ctx(const Npp32f *pConstant, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_C3IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_C3IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                const Npp32f aConstants[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                            const Npp32f aConstants[3], Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_AC4IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_AC4IR_Ctx(const Npp32f *pConstants,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32f_C4IR(const Npp32f aConstants[4], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddDeviceC_32f_C4IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                            Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_C1IR_Ctx(Npp32fc nConstant, Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_C1IR(Npp32fc nConstant, Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[3], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[3], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_C3IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_C3IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                 const Npp32fc aConstants[3], Npp32fc *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                             const Npp32fc aConstants[3], Npp32fc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_AC4IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_AC4IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[4], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[4], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddC_32fc_C4IR_Ctx(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddC_32fc_C4IR(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_C1IRSfs_Ctx(Npp8u nConstant, Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_C1IRSfs(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_C1IRSfs_Ctx(const Npp8u *pConstant, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[3], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_C3IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                        const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_AC4IRSfs_Ctx(const Npp8u *pConstants,
                                         Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[4], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_8u_C4IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_C1IRSfs_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_C1IRSfs(Npp16u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_C1IRSfs_Ctx(const Npp16u *pConstant,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_C3IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                         const Npp16u *pConstants, Npp16u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_AC4IRSfs_Ctx(const Npp16u *pConstants,
                                          Npp16u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16u_C4IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_C1IRSfs_Ctx(Npp16s nConstant, Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_C1IRSfs(Npp16s nConstant, Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_C1IRSfs_Ctx(const Npp16s *pConstant,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[3], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_C3IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                               const Npp16s aConstants[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                         const Npp16s *pConstants, Npp16s *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_AC4IRSfs_Ctx(const Npp16s *pConstants,
                                          Npp16s *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[4], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16s_C4IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   Npp16sc nConstant, Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               Npp16sc nConstant, Npp16sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16sc_C1IRSfs_Ctx(Npp16sc nConstant, Npp16sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16sc_C1IRSfs(Npp16sc nConstant, Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc aConstants[3], Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc aConstants[3], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3],
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                    const Npp16sc aConstants[3], Npp16sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                                const Npp16sc aConstants[3], Npp16sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3],
                                     Npp16sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32s_C1IRSfs_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32s_C1IRSfs(Npp32s nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32s_C1IRSfs_Ctx(const Npp32s *pConstant,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32s aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstants, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32s_C3IRSfs_Ctx(const Npp32s *pConstants,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   Npp32sc nConstant, Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               Npp32sc nConstant, Npp32sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32sc_C1IRSfs_Ctx(Npp32sc nConstant, Npp32sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32sc_C1IRSfs(Npp32sc nConstant, Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc aConstants[3], Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc aConstants[3], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3],
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                    const Npp32sc aConstants[3], Npp32sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                                const Npp32sc aConstants[3], Npp32sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3],
                                     Npp32sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMulC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_16f_C1R(const Npp16f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp16f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16f_C1IR_Ctx(Npp32f nConstant, Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_16f_C1IR(Npp32f nConstant, Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16f_C1IR_Ctx(const Npp32f *pConstant, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_16f_C3IR(const Npp32f aConstants[3], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16f_C3IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_16f_C4IR(const Npp32f aConstants[4], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_16f_C4IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_C1IR_Ctx(Npp32f nConstant, Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_C1IR(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_C1IR_Ctx(const Npp32f *pConstant, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_C3IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_C3IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                const Npp32f aConstants[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                            const Npp32f aConstants[3], Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_AC4IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_AC4IR_Ctx(const Npp32f *pConstants,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32f_C4IR(const Npp32f aConstants[4], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceC_32f_C4IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                            Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_C1IR_Ctx(Npp32fc nConstant, Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_C1IR(Npp32fc nConstant, Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[3], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[3], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_C3IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_C3IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                 const Npp32fc aConstants[3], Npp32fc *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                             const Npp32fc aConstants[3], Npp32fc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_AC4IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_AC4IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[4], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[4], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulC_32fc_C4IR_Ctx(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulC_32fc_C4IR(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                   Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                               Npp8u nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                         const Npp8u *pConstant, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_C1IR_Ctx(const Npp8u *pConstant,
                                          Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                   const Npp8u aConstants[3], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                               const Npp8u aConstants[3], Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                         const Npp8u *pConstants, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_C3IR_Ctx(const Npp8u *pConstants,
                                          Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                    const Npp8u aConstants[3], Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u aConstants[3], Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                          const Npp8u *pConstants, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                     int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_AC4IR_Ctx(const Npp8u *pConstants,
                                           Npp8u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                   const Npp8u aConstants[4], Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                               const Npp8u aConstants[4], Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                         const Npp8u *pConstants, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_8u_C4IR_Ctx(const Npp8u *pConstants,
                                          Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                    Npp16u nConstant, Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                                Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                          const Npp16u *pConstant, Npp16u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                     int nSrcDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_C1IR_Ctx(const Npp16u *pConstant,
                                           Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                    const Npp16u aConstants[3], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                                const Npp16u aConstants[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                          const Npp16u *pConstants,
                                          Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C3IR_Ctx(const Npp16u aConstants[3],
                                     Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_C3IR_Ctx(const Npp16u *pConstants,
                                           Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                     const Npp16u aConstants[3], Npp16u *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u aConstants[3], Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                           const Npp16u *pConstants,
                                           Npp16u *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_AC4IR_Ctx(const Npp16u aConstants[3],
                                      Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_AC4IR_Ctx(const Npp16u *pConstants,
                                            Npp16u *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                    const Npp16u aConstants[4], Npp16u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                                const Npp16u aConstants[4], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                          const Npp16u *pConstants,
                                          Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C4IR_Ctx(const Npp16u aConstants[4],
                                     Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulCScale_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulDeviceCScale_16u_C4IR_Ctx(const Npp16u *pConstants,
                                           Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_C1IRSfs_Ctx(Npp8u nConstant, Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_C1IRSfs(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_C1IRSfs_Ctx(const Npp8u *pConstant, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[3], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_C3IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                        const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_AC4IRSfs_Ctx(const Npp8u *pConstants,
                                         Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[4], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_8u_C4IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_C1IRSfs_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_C1IRSfs(Npp16u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_C1IRSfs_Ctx(const Npp16u *pConstant,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_C3IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                         const Npp16u *pConstants, Npp16u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_AC4IRSfs_Ctx(const Npp16u *pConstants,
                                          Npp16u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16u_C4IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_C1IRSfs_Ctx(Npp16s nConstant, Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_C1IRSfs(Npp16s nConstant, Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_C1IRSfs_Ctx(const Npp16s *pConstant,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[3], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_C3IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                               const Npp16s aConstants[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                         const Npp16s *pConstants, Npp16s *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_AC4IRSfs_Ctx(const Npp16s *pConstants,
                                          Npp16s *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[4], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16s_C4IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   Npp16sc nConstant, Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               Npp16sc nConstant, Npp16sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16sc_C1IRSfs_Ctx(Npp16sc nConstant, Npp16sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16sc_C1IRSfs(Npp16sc nConstant, Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc aConstants[3], Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc aConstants[3], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3],
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                    const Npp16sc aConstants[3], Npp16sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                                const Npp16sc aConstants[3], Npp16sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3],
                                     Npp16sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32s_C1IRSfs_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32s_C1IRSfs(Npp32s nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32s_C1IRSfs_Ctx(const Npp32s *pConstant,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32s aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstants, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32s_C3IRSfs_Ctx(const Npp32s *pConstants,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   Npp32sc nConstant, Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               Npp32sc nConstant, Npp32sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32sc_C1IRSfs_Ctx(Npp32sc nConstant, Npp32sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32sc_C1IRSfs(Npp32sc nConstant, Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc aConstants[3], Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc aConstants[3], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3],
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                    const Npp32sc aConstants[3], Npp32sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                                const Npp32sc aConstants[3], Npp32sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3],
                                     Npp32sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSubC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_16f_C1R(const Npp16f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp16f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16f_C1IR_Ctx(Npp32f nConstant, Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_16f_C1IR(Npp32f nConstant, Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16f_C1IR_Ctx(const Npp32f *pConstant, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_16f_C3IR(const Npp32f aConstants[3], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16f_C3IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_16f_C4IR(const Npp32f aConstants[4], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_16f_C4IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_C1IR_Ctx(Npp32f nConstant, Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_C1IR(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_C1IR_Ctx(const Npp32f *pConstant, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_C3IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_C3IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                const Npp32f aConstants[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                            const Npp32f aConstants[3], Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_AC4IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_AC4IR_Ctx(const Npp32f *pConstants,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32f_C4IR(const Npp32f aConstants[4], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubDeviceC_32f_C4IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                            Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_C1IR_Ctx(Npp32fc nConstant, Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_C1IR(Npp32fc nConstant, Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[3], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[3], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_C3IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_C3IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                 const Npp32fc aConstants[3], Npp32fc *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                             const Npp32fc aConstants[3], Npp32fc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_AC4IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_AC4IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[4], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[4], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSubC_32fc_C4IR_Ctx(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSubC_32fc_C4IR(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_C1IRSfs_Ctx(Npp8u nConstant, Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_C1IRSfs(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_C1IRSfs_Ctx(const Npp8u *pConstant, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[3], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_C3IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_C3IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_C3IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                        const Npp8u *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_AC4IRSfs_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_AC4IRSfs(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_AC4IRSfs_Ctx(const Npp8u *pConstants,
                                         Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u aConstants[4], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_8u_C4IRSfs_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_8u_C4IRSfs(const Npp8u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_8u_C4IRSfs_Ctx(const Npp8u *pConstants, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp16u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstant, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_C1IRSfs_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_C1IRSfs(Npp16u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_C1IRSfs_Ctx(const Npp16u *pConstant,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_C3IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_C3IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_C3IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                         const Npp16u *pConstants, Npp16u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_AC4IRSfs_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_AC4IRSfs(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_AC4IRSfs_Ctx(const Npp16u *pConstants,
                                          Npp16u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pConstants, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16u_C4IRSfs_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16u_C4IRSfs(const Npp16u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16u_C4IRSfs_Ctx(const Npp16u *pConstants,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  Npp16s nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              Npp16s nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstant, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_C1IRSfs_Ctx(Npp16s nConstant, Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_C1IRSfs(Npp16s nConstant, Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_C1IRSfs_Ctx(const Npp16s *pConstant,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[3], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_C3IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_C3IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_C3IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                               const Npp16s aConstants[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                         const Npp16s *pConstants, Npp16s *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_AC4IRSfs_Ctx(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_AC4IRSfs(const Npp16s aConstants[3], Npp16s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_AC4IRSfs_Ctx(const Npp16s *pConstants,
                                          Npp16s *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, int nScaleFactor,
                                          NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s aConstants[4], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pConstants, Npp16s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16s_C4IRSfs_Ctx(const Npp16s aConstants[4], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16s_C4IRSfs(const Npp16s aConstants[4], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16s_C4IRSfs_Ctx(const Npp16s *pConstants,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   Npp16sc nConstant, Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               Npp16sc nConstant, Npp16sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16sc_C1IRSfs_Ctx(Npp16sc nConstant, Npp16sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16sc_C1IRSfs(Npp16sc nConstant, Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc aConstants[3], Npp16sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc aConstants[3], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16sc_C3IRSfs_Ctx(const Npp16sc aConstants[3],
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16sc_C3IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                    const Npp16sc aConstants[3], Npp16sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                                const Npp16sc aConstants[3], Npp16sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16sc_AC4IRSfs_Ctx(const Npp16sc aConstants[3],
                                     Npp16sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_16sc_AC4IRSfs(const Npp16sc aConstants[3], Npp16sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  Npp32s nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstant, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32s_C1IRSfs_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32s_C1IRSfs(Npp32s nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32s_C1IRSfs_Ctx(const Npp32s *pConstant,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32s aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                        const Npp32s *pConstants, Npp32s *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32s_C3IRSfs_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32s_C3IRSfs(const Npp32s aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32s_C3IRSfs_Ctx(const Npp32s *pConstants,
                                         Npp32s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   Npp32sc nConstant, Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               Npp32sc nConstant, Npp32sc *pDst, int nDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32sc_C1IRSfs_Ctx(Npp32sc nConstant, Npp32sc *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32sc_C1IRSfs(Npp32sc nConstant, Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc aConstants[3], Npp32sc *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc aConstants[3], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32sc_C3IRSfs_Ctx(const Npp32sc aConstants[3],
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32sc_C3IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                    const Npp32sc aConstants[3], Npp32sc *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                                const Npp32sc aConstants[3], Npp32sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32sc_AC4IRSfs_Ctx(const Npp32sc aConstants[3],
                                     Npp32sc *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDivC_32sc_AC4IRSfs(const Npp32sc aConstants[3], Npp32sc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp16f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_16f_C1R(const Npp16f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp16f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16f_C1IR_Ctx(Npp32f nConstant, Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_16f_C1IR(Npp32f nConstant, Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16f_C1IR_Ctx(const Npp32f *pConstant, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16f_C3IR_Ctx(const Npp32f aConstants[3], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_16f_C3IR(const Npp32f aConstants[3], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16f_C3IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp16f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_16f_C4IR_Ctx(const Npp32f aConstants[4], Npp16f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_16f_C4IR(const Npp32f aConstants[4], Npp16f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_16f_C4IR_Ctx(const Npp32f *pConstants, Npp16f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               Npp32f nConstant, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, Npp32f nConstant,
                           Npp32f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstant, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_C1IR_Ctx(Npp32f nConstant, Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_C1IR(Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_C1IR_Ctx(const Npp32f *pConstant, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[3], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[3], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_C3IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_C3IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_C3IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                const Npp32f aConstants[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                            const Npp32f aConstants[3], Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pConstants, Npp32f *pDst,
                                      int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_AC4IR_Ctx(const Npp32f aConstants[3], Npp32f *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_AC4IR(const Npp32f aConstants[3], Npp32f *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_AC4IR_Ctx(const Npp32f *pConstants,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f aConstants[4], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f aConstants[4], Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pConstants, Npp32f *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32f_C4IR_Ctx(const Npp32f aConstants[4], Npp32f *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32f_C4IR(const Npp32f aConstants[4], Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pConstants SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivDeviceC_32f_C4IR_Ctx(const Npp32f *pConstants, Npp32f *pSrcDst,
                                      int nSrcDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                            Npp32fc nConstant, Npp32fc *pDst, int nDstStep,
                            NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_C1IR_Ctx(Npp32fc nConstant, Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_C1IR(Npp32fc nConstant, Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[3], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[3], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_C3IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_C3IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                 const Npp32fc aConstants[3], Npp32fc *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                             const Npp32fc aConstants[3], Npp32fc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_AC4IR_Ctx(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_AC4IR(const Npp32fc aConstants[3], Npp32fc *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc aConstants[4], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc aConstants[4], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDivC_32fc_C4IR_Ctx(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDivC_32fc_C4IR(const Npp32fc aConstants[4], Npp32fc *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiffC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nConstant,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nConstant SEND_ONLY
 */
NppStatus nppiAbsDiffC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp8u nConstant);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiffDeviceC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp8u *pConstant,
                                        NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiffC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp16u nConstant,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nConstant SEND_ONLY
 */
NppStatus nppiAbsDiffC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               Npp16u nConstant);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiffDeviceC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                         Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp16u *pConstant,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiffC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp32f nConstant,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nConstant SEND_ONLY
 */
NppStatus nppiAbsDiffC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               Npp32f nConstant);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pConstant SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiffDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                         Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp32f *pConstant,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep,
                               Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16sc_C1IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16sc_C1IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16sc_C3IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16sc_C3IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc *pSrc2, int nSrc2Step,
                                   Npp16sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc *pSrc2, int nSrc2Step,
                               Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16sc_AC4IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_16sc_AC4IRSfs(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32s_C1IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32s_C1IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32s_C3IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32s_C3IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32sc_C1IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32sc_C1IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32sc_C3IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32sc_C3IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc *pSrc2, int nSrc2Step,
                                   Npp32sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc *pSrc2, int nSrc2Step,
                               Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32sc_AC4IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiAdd_32sc_AC4IRSfs(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_16f_C1R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16f_C1IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_16f_C1IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16f_C3IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_16f_C3IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_16f_C4IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_16f_C4IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_C3IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_C3IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_AC4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_AC4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32f_C4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32f_C4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_C1IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_C1IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_C3IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_C3IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc *pSrc2, int nSrc2Step,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_AC4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                 Npp32fc *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_AC4IR(const Npp32fc *pSrc, int nSrcStep,
                             Npp32fc *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAdd_32fc_C4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAdd_32fc_C4IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddSquare_8u32f_C1IMR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        const Npp8u *pMask, int nMaskStep,
                                        Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddSquare_8u32f_C1IMR(const Npp8u *pSrc, int nSrcStep,
                                    const Npp8u *pMask, int nMaskStep,
                                    Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddSquare_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddSquare_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep,
                                   Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddSquare_16u32f_C1IMR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         const Npp8u *pMask, int nMaskStep,
                                         Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddSquare_16u32f_C1IMR(const Npp16u *pSrc, int nSrcStep,
                                     const Npp8u *pMask, int nMaskStep,
                                     Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddSquare_16u32f_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddSquare_16u32f_C1IR(const Npp16u *pSrc, int nSrcStep,
                                    Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddSquare_32f_C1IMR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      const Npp8u *pMask, int nMaskStep,
                                      Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddSquare_32f_C1IMR(const Npp32f *pSrc, int nSrcStep,
                                  const Npp8u *pMask, int nMaskStep,
                                  Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddSquare_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddSquare_32f_C1IR(const Npp32f *pSrc, int nSrcStep,
                                 Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_8u32f_C1IMR_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                         const Npp8u *pSrc2, int nSrc2Step,
                                         const Npp8u *pMask, int nMaskStep,
                                         Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_8u32f_C1IMR(const Npp8u *pSrc1, int nSrc1Step,
                                     const Npp8u *pSrc2, int nSrc2Step,
                                     const Npp8u *pMask, int nMaskStep,
                                     Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_8u32f_C1IR_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                        const Npp8u *pSrc2, int nSrc2Step,
                                        Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_8u32f_C1IR(const Npp8u *pSrc1, int nSrc1Step,
                                    const Npp8u *pSrc2, int nSrc2Step,
                                    Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_16u32f_C1IMR_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                          const Npp16u *pSrc2, int nSrc2Step,
                                          const Npp8u *pMask, int nMaskStep,
                                          Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_16u32f_C1IMR(const Npp16u *pSrc1, int nSrc1Step,
                                      const Npp16u *pSrc2, int nSrc2Step,
                                      const Npp8u *pMask, int nMaskStep,
                                      Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_16u32f_C1IR_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                         const Npp16u *pSrc2, int nSrc2Step,
                                         Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_16u32f_C1IR(const Npp16u *pSrc1, int nSrc1Step,
                                     const Npp16u *pSrc2, int nSrc2Step,
                                     Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_32f_C1IMR_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                       const Npp32f *pSrc2, int nSrc2Step,
                                       const Npp8u *pMask, int nMaskStep,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_32f_C1IMR(const Npp32f *pSrc1, int nSrc1Step,
                                   const Npp32f *pSrc2, int nSrc2Step,
                                   const Npp8u *pMask, int nMaskStep,
                                   Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_32f_C1IR_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pSrc2, int nSrc2Step,
                                      Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_32f_C1IR(const Npp32f *pSrc1, int nSrc1Step,
                                  const Npp32f *pSrc2, int nSrc2Step,
                                  Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddProduct_16f_C1IR_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                      const Npp16f *pSrc2, int nSrc2Step,
                                      Npp16f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAddProduct_16f_C1IR(const Npp16f *pSrc1, int nSrc1Step,
                                  const Npp16f *pSrc2, int nSrc2Step,
                                  Npp16f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddWeighted_8u32f_C1IMR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          const Npp8u *pMask, int nMaskStep,
                                          Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, Npp32f nAlpha,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiAddWeighted_8u32f_C1IMR(const Npp8u *pSrc, int nSrcStep,
                                      const Npp8u *pMask, int nMaskStep,
                                      Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, Npp32f nAlpha);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddWeighted_8u32f_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, Npp32f nAlpha,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiAddWeighted_8u32f_C1IR(const Npp8u *pSrc, int nSrcStep,
                                     Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp32f nAlpha);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddWeighted_16u32f_C1IMR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           const Npp8u *pMask, int nMaskStep,
                                           Npp32f *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp32f nAlpha,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiAddWeighted_16u32f_C1IMR(const Npp16u *pSrc, int nSrcStep,
                                       const Npp8u *pMask, int nMaskStep,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp32f nAlpha);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddWeighted_16u32f_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, Npp32f nAlpha,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiAddWeighted_16u32f_C1IR(const Npp16u *pSrc, int nSrcStep,
                                      Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, Npp32f nAlpha);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddWeighted_32f_C1IMR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        const Npp8u *pMask, int nMaskStep,
                                        Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp32f nAlpha,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiAddWeighted_32f_C1IMR(const Npp32f *pSrc, int nSrcStep,
                                    const Npp8u *pMask, int nMaskStep,
                                    Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp32f nAlpha);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAddWeighted_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp32f nAlpha,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiAddWeighted_32f_C1IR(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, Npp32f nAlpha);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep,
                               Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16sc_C1IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16sc_C1IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16sc_C3IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16sc_C3IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc *pSrc2, int nSrc2Step,
                                   Npp16sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc *pSrc2, int nSrc2Step,
                               Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16sc_AC4IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_16sc_AC4IRSfs(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32s_C1IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32s_C1IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32s_C3IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32s_C3IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32sc_C1IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32sc_C1IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32sc_C3IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32sc_C3IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc *pSrc2, int nSrc2Step,
                                   Npp32sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc *pSrc2, int nSrc2Step,
                               Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32sc_AC4IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiMul_32sc_AC4IRSfs(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_16f_C1R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16f_C1IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_16f_C1IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16f_C3IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_16f_C3IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_16f_C4IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_16f_C4IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_C3IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_C3IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_AC4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_AC4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32f_C4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32f_C4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_C1IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_C1IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_C3IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_C3IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc *pSrc2, int nSrc2Step,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_AC4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                 Npp32fc *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_AC4IR(const Npp32fc *pSrc, int nSrcStep,
                             Npp32fc *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMul_32fc_C4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMul_32fc_C4IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                   const Npp8u *pSrc2, int nSrc2Step,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                               const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_AC4IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_AC4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_8u_C4IR_Ctx(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_8u_C4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_C1IR(const Npp16u *pSrc, int nSrcStep,
                                Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_C3IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_C3IR(const Npp16u *pSrc, int nSrcStep,
                                Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                    const Npp16u *pSrc2, int nSrc2Step,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                                const Npp16u *pSrc2, int nSrc2Step,
                                Npp16u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_AC4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_AC4IR(const Npp16u *pSrc, int nSrcStep,
                                 Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMulScale_16u_C4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiMulScale_16u_C4IR(const Npp16u *pSrc, int nSrcStep,
                                Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep,
                               Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16sc_C1IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16sc_C1IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16sc_C3IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16sc_C3IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc *pSrc2, int nSrc2Step,
                                   Npp16sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc *pSrc2, int nSrc2Step,
                               Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16sc_AC4IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_16sc_AC4IRSfs(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32s_C1IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32s_C1IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32s_C3IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32s_C3IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32sc_C1IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32sc_C1IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32sc_C3IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32sc_C3IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc *pSrc2, int nSrc2Step,
                                   Npp32sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc *pSrc2, int nSrc2Step,
                               Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32sc_AC4IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSub_32sc_AC4IRSfs(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_16f_C1R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16f_C1IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_16f_C1IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16f_C3IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_16f_C3IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_16f_C4IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_16f_C4IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_C3IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_C3IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_AC4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_AC4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32f_C4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32f_C4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_C1IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_C1IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_C3IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_C3IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc *pSrc2, int nSrc2Step,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_AC4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                 Npp32fc *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_AC4IR(const Npp32fc *pSrc, int nSrcStep,
                             Npp32fc *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSub_32fc_C4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSub_32fc_C4IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep,
                               Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                             const Npp16s *pSrc2, int nSrc2Step, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16sc_C1RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16sc_C1RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16sc_C1IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16sc_C1IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16sc_C3RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                  const Npp16sc *pSrc2, int nSrc2Step,
                                  Npp16sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16sc_C3RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                              const Npp16sc *pSrc2, int nSrc2Step,
                              Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16sc_C3IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                   Npp16sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16sc_C3IRSfs(const Npp16sc *pSrc, int nSrcStep,
                               Npp16sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16sc_AC4RSfs_Ctx(const Npp16sc *pSrc1, int nSrc1Step,
                                   const Npp16sc *pSrc2, int nSrc2Step,
                                   Npp16sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16sc_AC4RSfs(const Npp16sc *pSrc1, int nSrc1Step,
                               const Npp16sc *pSrc2, int nSrc2Step,
                               Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16sc_AC4IRSfs_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                    Npp16sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_16sc_AC4IRSfs(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32s_C1RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32s_C1RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32s_C1IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32s_C1IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32s_C3RSfs_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32s_C3RSfs(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32s_C3IRSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32s_C3IRSfs(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32sc_C1RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32sc_C1RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32sc_C1IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32sc_C1IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32sc_C3RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                  const Npp32sc *pSrc2, int nSrc2Step,
                                  Npp32sc *pDst, int nDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32sc_C3RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                              const Npp32sc *pSrc2, int nSrc2Step,
                              Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32sc_C3IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                   Npp32sc *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32sc_C3IRSfs(const Npp32sc *pSrc, int nSrcStep,
                               Npp32sc *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32sc_AC4RSfs_Ctx(const Npp32sc *pSrc1, int nSrc1Step,
                                   const Npp32sc *pSrc2, int nSrc2Step,
                                   Npp32sc *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32sc_AC4RSfs(const Npp32sc *pSrc1, int nSrc1Step,
                               const Npp32sc *pSrc2, int nSrc2Step,
                               Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32sc_AC4IRSfs_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                    Npp32sc *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_32sc_AC4IRSfs(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_16f_C1R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16f_C1IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_16f_C1IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16f_C3R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_16f_C3R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16f_C3IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_16f_C3IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16f_C4R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_16f_C4R(const Npp16f *pSrc1, int nSrc1Step,
                          const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_16f_C4IR_Ctx(const Npp16f *pSrc, int nSrcStep,
                               Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_16f_C4IR(const Npp16f *pSrc, int nSrcStep, Npp16f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_C1IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_C1IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_C3IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_C3IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                           const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_AC4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                                Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_AC4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                          const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32f_C4IR_Ctx(const Npp32f *pSrc, int nSrcStep,
                               Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32f_C4IR(const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_C1R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_C1R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_C1IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_C1IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_C3R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_C3R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_C3IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_C3IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_AC4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                                const Npp32fc *pSrc2, int nSrc2Step,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_AC4R(const Npp32fc *pSrc1, int nSrc1Step,
                            const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_AC4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                 Npp32fc *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_AC4IR(const Npp32fc *pSrc, int nSrcStep,
                             Npp32fc *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_C4R_Ctx(const Npp32fc *pSrc1, int nSrc1Step,
                               const Npp32fc *pSrc2, int nSrc2Step,
                               Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_C4R(const Npp32fc *pSrc1, int nSrc1Step,
                           const Npp32fc *pSrc2, int nSrc2Step, Npp32fc *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_32fc_C4IR_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiDiv_32fc_C4IR(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppRoundMode rndMode,
                                      int nScaleFactor,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C1RSfs(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode rndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C1IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C1IRSfs(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C3RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppRoundMode rndMode,
                                      int nScaleFactor,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C3RSfs(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode rndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C3IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C3IRSfs(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_AC4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       const Npp8u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_AC4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                                   const Npp8u *pSrc2, int nSrc2Step,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode rndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_AC4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_AC4IRSfs(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C4RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppRoundMode rndMode,
                                      int nScaleFactor,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C4RSfs(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode rndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C4IRSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_8u_C4IRSfs(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C1RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C1RSfs(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C1IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C1IRSfs(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C3RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C3RSfs(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C3IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C3IRSfs(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_AC4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        const Npp16u *pSrc2, int nSrc2Step,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_AC4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                                    const Npp16u *pSrc2, int nSrc2Step,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_AC4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         NppRoundMode rndMode, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_AC4IRSfs(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppRoundMode rndMode,
                                     int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C4RSfs_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       const Npp16u *pSrc2, int nSrc2Step,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C4RSfs(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C4IRSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16u_C4IRSfs(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C1RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                       const Npp16s *pSrc2, int nSrc2Step,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C1RSfs(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s *pSrc2, int nSrc2Step,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C1IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C1IRSfs(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C3RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                       const Npp16s *pSrc2, int nSrc2Step,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C3RSfs(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s *pSrc2, int nSrc2Step,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C3IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C3IRSfs(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_AC4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                        const Npp16s *pSrc2, int nSrc2Step,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_AC4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                                    const Npp16s *pSrc2, int nSrc2Step,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_AC4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         NppRoundMode rndMode, int nScaleFactor,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_AC4IRSfs(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, NppRoundMode rndMode,
                                     int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C4RSfs_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                       const Npp16s *pSrc2, int nSrc2Step,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppRoundMode rndMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C4RSfs(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s *pSrc2, int nSrc2Step,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, NppRoundMode rndMode,
                                   int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C4IRSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, NppRoundMode rndMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiDiv_Round_16s_C4IRSfs(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppRoundMode rndMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16f_C4R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbs_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbs_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiff_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbsDiff_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiff_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbsDiff_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiff_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbsDiff_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiff_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbsDiff_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiff_16f_C1R_Ctx(const Npp16f *pSrc1, int nSrc1Step,
                                  const Npp16f *pSrc2, int nSrc2Step,
                                  Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbsDiff_16f_C1R(const Npp16f *pSrc1, int nSrc1Step,
                              const Npp16f *pSrc2, int nSrc2Step, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAbsDiff_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                  const Npp32f *pSrc2, int nSrc2Step,
                                  Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAbsDiff_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_C3IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_C3IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_AC4IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_AC4IRSfs(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_C4RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_8u_C4IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_8u_C4IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_C3IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_C3IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_AC4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_AC4RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_AC4IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_AC4IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_C4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_C4RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16u_C4IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16u_C4IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_C3IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_C3IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_AC4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_AC4RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_AC4IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_AC4IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_C4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_C4RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16s_C4IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqr_16s_C4IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_16f_C4R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqr_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqr_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_8u_C3IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_8u_C3IRSfs(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_8u_AC4IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_8u_AC4IRSfs(Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16u_C3IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16u_C3IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16u_AC4RSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16u_AC4RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16u_AC4IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16u_AC4IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16s_C3IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16s_C3IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16s_AC4RSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16s_AC4RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16s_AC4IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiSqrt_16s_AC4IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_16f_C4R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSqrt_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSqrt_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                            int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_8u_C3IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_8u_C3IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                            int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16u_C3IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16u_C3IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16s_C3IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiLn_16s_C3IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                             NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLn_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLn_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_8u_C1IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_8u_C1IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_8u_C3IRSfs_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_8u_C3IRSfs(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                             int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16u_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16u_C1RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16u_C1IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16u_C1IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16u_C3RSfs_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16u_C3RSfs(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16u_C3IRSfs_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16u_C3IRSfs(Npp16u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16s_C1RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16s_C1IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16s_C1IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16s_C3RSfs_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16s_C3RSfs(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_16s_C3IRSfs_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiExp_16s_C3IRSfs(Npp16s *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiExp_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiExp_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiExp_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiExp_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiExp_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                          Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                           NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                          const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                               const Npp8u aConstants[3], Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                           const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[4], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                          const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               Npp16u nConstant, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nConstant,
                           Npp16u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u aConstants[3], Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_C3IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                const Npp16u aConstants[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                            const Npp16u aConstants[3], Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_AC4IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[4], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u aConstants[4], Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_16u_C4IR_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               Npp32s nConstant, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step, Npp32s nConstant,
                           Npp32s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_C1IR_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_C1IR(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s aConstants[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s aConstants[3], Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_C3IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_C3IR(const Npp32s aConstants[3], Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                const Npp32s aConstants[3], Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                            const Npp32s aConstants[3], Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_AC4IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_AC4IR(const Npp32s aConstants[3], Npp32s *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s aConstants[4], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s aConstants[4], Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAndC_32s_C4IR_Ctx(const Npp32s aConstants[4], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAndC_32s_C4IR(const Npp32s aConstants[4], Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                          NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                         const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                          const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                         const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              Npp16u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nConstant,
                          Npp16u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep,
                           NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u aConstants[3], Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_C3IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u aConstants[3], Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_AC4IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u aConstants[4], Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_16u_C4IR_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32s nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step, Npp32s nConstant,
                          Npp32s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_C1IR_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_C1IR(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep,
                           NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s aConstants[3], Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_C3IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_C3IR(const Npp32s aConstants[3], Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s aConstants[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s aConstants[3], Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_AC4IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_AC4IR(const Npp32s aConstants[3], Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s aConstants[4], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s aConstants[4], Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOrC_32s_C4IR_Ctx(const Npp32s aConstants[4], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOrC_32s_C4IR(const Npp32s aConstants[4], Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              Npp8u nConstant, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nConstant,
                          Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_C1IR_Ctx(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_C1IR(Npp8u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                           NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                          const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_C3IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_C3IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                               const Npp8u aConstants[3], Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                           const Npp8u aConstants[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_AC4IR_Ctx(const Npp8u aConstants[3], Npp8u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_AC4IR(const Npp8u aConstants[3], Npp8u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u aConstants[4], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                          const Npp8u aConstants[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_8u_C4IR_Ctx(const Npp8u aConstants[4], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_8u_C4IR(const Npp8u aConstants[4], Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               Npp16u nConstant, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, Npp16u nConstant,
                           Npp16u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_C1IR_Ctx(Npp16u nConstant, Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_C1IR(Npp16u nConstant, Npp16u *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u aConstants[3], Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_C3IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_C3IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                const Npp16u aConstants[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                            const Npp16u aConstants[3], Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_AC4IR_Ctx(const Npp16u aConstants[3], Npp16u *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_AC4IR(const Npp16u aConstants[3], Npp16u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u aConstants[4], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u aConstants[4], Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_16u_C4IR_Ctx(const Npp16u aConstants[4], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_16u_C4IR(const Npp16u aConstants[4], Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               Npp32s nConstant, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step, Npp32s nConstant,
                           Npp32s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_C1IR_Ctx(Npp32s nConstant, Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_C1IR(Npp32s nConstant, Npp32s *pSrcDst, int nSrcDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s aConstants[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s aConstants[3], Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_C3IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_C3IR(const Npp32s aConstants[3], Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                const Npp32s aConstants[3], Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                            const Npp32s aConstants[3], Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_AC4IR_Ctx(const Npp32s aConstants[3], Npp32s *pSrcDst,
                                 int nSrcDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_AC4IR(const Npp32s aConstants[3], Npp32s *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s aConstants[4], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s aConstants[4], Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXorC_32s_C4IR_Ctx(const Npp32s aConstants[4], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXorC_32s_C4IR(const Npp32s aConstants[4], Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp32u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                             Npp32u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C1IR_Ctx(Npp32u nConstant, Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C1IR(Npp32u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp32u aConstants[3], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp32u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C3IR_Ctx(const Npp32u aConstants[3], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C3IR(const Npp32u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_AC4IR_Ctx(const Npp32u aConstants[3], Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_AC4IR(const Npp32u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp32u aConstants[4], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp32u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C4IR_Ctx(const Npp32u aConstants[4], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8u_C4IR(const Npp32u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C1R_Ctx(const Npp8s *pSrc1, int nSrc1Step,
                                 Npp32u nConstant, Npp8s *pDst, int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C1R(const Npp8s *pSrc1, int nSrc1Step,
                             Npp32u nConstant, Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C1IR_Ctx(Npp32u nConstant, Npp8s *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C1IR(Npp32u nConstant, Npp8s *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C3R_Ctx(const Npp8s *pSrc1, int nSrc1Step,
                                 const Npp32u aConstants[3], Npp8s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C3R(const Npp8s *pSrc1, int nSrc1Step,
                             const Npp32u aConstants[3], Npp8s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C3IR_Ctx(const Npp32u aConstants[3], Npp8s *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C3IR(const Npp32u aConstants[3], Npp8s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_AC4R_Ctx(const Npp8s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp8s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_AC4R(const Npp8s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp8s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_AC4IR_Ctx(const Npp32u aConstants[3], Npp8s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_AC4IR(const Npp32u aConstants[3], Npp8s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C4R_Ctx(const Npp8s *pSrc1, int nSrc1Step,
                                 const Npp32u aConstants[4], Npp8s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C4R(const Npp8s *pSrc1, int nSrc1Step,
                             const Npp32u aConstants[4], Npp8s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C4IR_Ctx(const Npp32u aConstants[4], Npp8s *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_8s_C4IR(const Npp32u aConstants[4], Npp8s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp32u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                              Npp32u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C1IR_Ctx(Npp32u nConstant, Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C1IR(Npp32u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C3IR_Ctx(const Npp32u aConstants[3], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C3IR(const Npp32u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp32u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp32u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_AC4IR_Ctx(const Npp32u aConstants[3], Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_AC4IR(const Npp32u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C4IR_Ctx(const Npp32u aConstants[4], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16u_C4IR(const Npp32u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C1R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  Npp32u nConstant, Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C1R(const Npp16s *pSrc1, int nSrc1Step,
                              Npp32u nConstant, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C1IR_Ctx(Npp32u nConstant, Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C1IR(Npp32u nConstant, Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C3R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C3R(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C3IR_Ctx(const Npp32u aConstants[3], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C3IR(const Npp32u aConstants[3], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_AC4R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp32u aConstants[3], Npp16s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_AC4R(const Npp16s *pSrc1, int nSrc1Step,
                               const Npp32u aConstants[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_AC4IR_Ctx(const Npp32u aConstants[3], Npp16s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_AC4IR(const Npp32u aConstants[3], Npp16s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C4R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[4], Npp16s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C4R(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[4], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C4IR_Ctx(const Npp32u aConstants[4], Npp16s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_16s_C4IR(const Npp32u aConstants[4], Npp16s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  Npp32u nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32u nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C1IR_Ctx(Npp32u nConstant, Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C1IR(Npp32u nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C3IR_Ctx(const Npp32u aConstants[3], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C3IR(const Npp32u aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                   const Npp32u aConstants[3], Npp32s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32u aConstants[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_AC4IR_Ctx(const Npp32u aConstants[3], Npp32s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_AC4IR(const Npp32u aConstants[3], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[4], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[4], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C4IR_Ctx(const Npp32u aConstants[4], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRShiftC_32s_C4IR(const Npp32u aConstants[4], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp32u nConstant, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                             Npp32u nConstant, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C1IR_Ctx(Npp32u nConstant, Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C1IR(Npp32u nConstant, Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp32u aConstants[3], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp32u aConstants[3], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C3IR_Ctx(const Npp32u aConstants[3], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C3IR(const Npp32u aConstants[3], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_AC4IR_Ctx(const Npp32u aConstants[3], Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_AC4IR(const Npp32u aConstants[3], Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp32u aConstants[4], Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp32u aConstants[4], Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C4IR_Ctx(const Npp32u aConstants[4], Npp8u *pSrcDst,
                                  int nSrcDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_8u_C4IR(const Npp32u aConstants[4], Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp32u nConstant, Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                              Npp32u nConstant, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C1IR_Ctx(Npp32u nConstant, Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C1IR(Npp32u nConstant, Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C3IR_Ctx(const Npp32u aConstants[3], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C3IR(const Npp32u aConstants[3], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp32u aConstants[3], Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp32u aConstants[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_AC4IR_Ctx(const Npp32u aConstants[3], Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_AC4IR(const Npp32u aConstants[3], Npp16u *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[4], Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C4IR_Ctx(const Npp32u aConstants[4], Npp16u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_16u_C4IR(const Npp32u aConstants[4], Npp16u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  Npp32u nConstant, Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                              Npp32u nConstant, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI);
/**
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C1IR_Ctx(Npp32u nConstant, Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nConstant SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C1IR(Npp32u nConstant, Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[3], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C3IR_Ctx(const Npp32u aConstants[3], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C3IR(const Npp32u aConstants[3], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                   const Npp32u aConstants[3], Npp32s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32u aConstants[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_AC4IR_Ctx(const Npp32u aConstants[3], Npp32s *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_AC4IR(const Npp32u aConstants[3], Npp32s *pSrcDst,
                                int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                  const Npp32u aConstants[4], Npp32s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param aConstants SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32u aConstants[4], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C4IR_Ctx(const Npp32u aConstants[4], Npp32s *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aConstants SEND_ONLY DEREF
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiLShiftC_32s_C4IR(const Npp32u aConstants[4], Npp32s *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                          int nSrc2Step, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_AC4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_AC4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_8u_C4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_8u_C4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_C3IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_C3IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_AC4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_AC4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_16u_C4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_16u_C4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_C1IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_C1IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_C3IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_C3IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_AC4IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                                Npp32s *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_AC4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAnd_32s_C4IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAnd_32s_C4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                        int nSrc2Step, Npp8u *pDst, int nDstStep,
                        NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                         int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                        int nSrc2Step, Npp8u *pDst, int nDstStep,
                        NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                         int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_AC4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_AC4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                            const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                        int nSrc2Step, Npp8u *pDst, int nDstStep,
                        NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_8u_C4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                             int nSrcDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_8u_C4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                         int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                         const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                         const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_C3IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_C3IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_AC4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_AC4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                             const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                         const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_16u_C4IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_16u_C4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                         const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_C1IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_C1IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                         const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_C3IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_C3IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_AC4IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_AC4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                             const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                         const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiOr_32s_C4IR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiOr_32s_C4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_C3IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_C3IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                          int nSrc2Step, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_AC4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                               int nSrcDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_AC4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_C4R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2,
                         int nSrc2Step, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_8u_C4IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                              int nSrcDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_8u_C4IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst,
                          int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_C3IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_C3IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                           const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_AC4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                                Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_AC4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                          const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_16u_C4IR_Ctx(const Npp16u *pSrc, int nSrcStep,
                               Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_16u_C4IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_C1IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_C1IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_C3R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_C3R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_C3IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_C3IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                               const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                           const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_AC4IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                                Npp32s *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_AC4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                            int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_C4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                              const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_C4R(const Npp32s *pSrc1, int nSrc1Step,
                          const Npp32s *pSrc2, int nSrc2Step, Npp32s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXor_32s_C4IR_Ctx(const Npp32s *pSrc, int nSrcStep,
                               Npp32s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiXor_32s_C4IR(const Npp32s *pSrc, int nSrcStep, Npp32s *pSrcDst,
                           int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNot_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNot_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                    Npp8u nAlpha1, const Npp8u *pSrc2,
                                    int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                                Npp8u nAlpha1, const Npp8u *pSrc2,
                                int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                    Npp8u nAlpha1, const Npp8u *pSrc2,
                                    int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                                Npp8u nAlpha1, const Npp8u *pSrc2,
                                int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                    Npp8u nAlpha1, const Npp8u *pSrc2,
                                    int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                                Npp8u nAlpha1, const Npp8u *pSrc2,
                                int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                     Npp8u nAlpha1, const Npp8u *pSrc2,
                                     int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                     int nDstStep, NppiSize oSizeROI,
                                     NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                                 Npp8u nAlpha1, const Npp8u *pSrc2,
                                 int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_8s_C1R_Ctx(const Npp8s *pSrc1, int nSrc1Step,
                                    Npp8s nAlpha1, const Npp8s *pSrc2,
                                    int nSrc2Step, Npp8s nAlpha2, Npp8s *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_8s_C1R(const Npp8s *pSrc1, int nSrc1Step,
                                Npp8s nAlpha1, const Npp8s *pSrc2,
                                int nSrc2Step, Npp8s nAlpha2, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                     Npp16u nAlpha1, const Npp16u *pSrc2,
                                     int nSrc2Step, Npp16u nAlpha2,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                                 Npp16u nAlpha1, const Npp16u *pSrc2,
                                 int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                     Npp16u nAlpha1, const Npp16u *pSrc2,
                                     int nSrc2Step, Npp16u nAlpha2,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                                 Npp16u nAlpha1, const Npp16u *pSrc2,
                                 int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                     Npp16u nAlpha1, const Npp16u *pSrc2,
                                     int nSrc2Step, Npp16u nAlpha2,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                                 Npp16u nAlpha1, const Npp16u *pSrc2,
                                 int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                      Npp16u nAlpha1, const Npp16u *pSrc2,
                                      int nSrc2Step, Npp16u nAlpha2,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                                  Npp16u nAlpha1, const Npp16u *pSrc2,
                                  int nSrc2Step, Npp16u nAlpha2, Npp16u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_16s_C1R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                     Npp16s nAlpha1, const Npp16s *pSrc2,
                                     int nSrc2Step, Npp16s nAlpha2,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_16s_C1R(const Npp16s *pSrc1, int nSrc1Step,
                                 Npp16s nAlpha1, const Npp16s *pSrc2,
                                 int nSrc2Step, Npp16s nAlpha2, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_32u_C1R_Ctx(const Npp32u *pSrc1, int nSrc1Step,
                                     Npp32u nAlpha1, const Npp32u *pSrc2,
                                     int nSrc2Step, Npp32u nAlpha2,
                                     Npp32u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_32u_C1R(const Npp32u *pSrc1, int nSrc1Step,
                                 Npp32u nAlpha1, const Npp32u *pSrc2,
                                 int nSrc2Step, Npp32u nAlpha2, Npp32u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_32s_C1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                     Npp32s nAlpha1, const Npp32s *pSrc2,
                                     int nSrc2Step, Npp32s nAlpha2,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_32s_C1R(const Npp32s *pSrc1, int nSrc1Step,
                                 Npp32s nAlpha1, const Npp32s *pSrc2,
                                 int nSrc2Step, Npp32s nAlpha2, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     Npp32f nAlpha1, const Npp32f *pSrc2,
                                     int nSrc2Step, Npp32f nAlpha2,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param nAlpha2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaCompC_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                                 Npp32f nAlpha1, const Npp32f *pSrc2,
                                 int nSrc2Step, Npp32f nAlpha2, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                                  Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C1IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst,
                                       int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C1IR(Npp8u nAlpha1, Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                                  Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C3IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst,
                                       int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C3IR(Npp8u nAlpha1, Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                                  Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C4IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst,
                                       int nSrcDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_C4IR(Npp8u nAlpha1, Npp8u *pSrcDst,
                                   int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                       Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                                   Npp8u nAlpha1, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_AC4IR_Ctx(Npp8u nAlpha1, Npp8u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_8u_AC4IR(Npp8u nAlpha1, Npp8u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       Npp16u nAlpha1, Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                                   Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C1IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C1IR(Npp16u nAlpha1, Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       Npp16u nAlpha1, Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                                   Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C3IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C3IR(Npp16u nAlpha1, Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       Npp16u nAlpha1, Npp16u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                                   Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C4IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst,
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_C4IR(Npp16u nAlpha1, Npp16u *pSrcDst,
                                    int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                        Npp16u nAlpha1, Npp16u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param nAlpha1 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                                    Npp16u nAlpha1, Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI);
/**
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_AC4IR_Ctx(Npp16u nAlpha1, Npp16u *pSrcDst,
                                         int nSrcDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nAlpha1 SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremulC_16u_AC4IR(Npp16u nAlpha1, Npp16u *pSrcDst,
                                     int nSrcDstStep, NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_8u_AC1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                    const Npp8u *pSrc2, int nSrc2Step,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_8u_AC1R(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                    const Npp8u *pSrc2, int nSrc2Step,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                                const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_8s_AC1R_Ctx(const Npp8s *pSrc1, int nSrc1Step,
                                    const Npp8s *pSrc2, int nSrc2Step,
                                    Npp8s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_8s_AC1R(const Npp8s *pSrc1, int nSrc1Step,
                                const Npp8s *pSrc2, int nSrc2Step, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_16u_AC1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                     const Npp16u *pSrc2, int nSrc2Step,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_16u_AC1R(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                     const Npp16u *pSrc2, int nSrc2Step,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                                 const Npp16u *pSrc2, int nSrc2Step,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_16s_AC1R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                     const Npp16s *pSrc2, int nSrc2Step,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_16s_AC1R(const Npp16s *pSrc1, int nSrc1Step,
                                 const Npp16s *pSrc2, int nSrc2Step,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_32u_AC1R_Ctx(const Npp32u *pSrc1, int nSrc1Step,
                                     const Npp32u *pSrc2, int nSrc2Step,
                                     Npp32u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_32u_AC1R(const Npp32u *pSrc1, int nSrc1Step,
                                 const Npp32u *pSrc2, int nSrc2Step,
                                 Npp32u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_32u_AC4R_Ctx(const Npp32u *pSrc1, int nSrc1Step,
                                     const Npp32u *pSrc2, int nSrc2Step,
                                     Npp32u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_32u_AC4R(const Npp32u *pSrc1, int nSrc1Step,
                                 const Npp32u *pSrc2, int nSrc2Step,
                                 Npp32u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_32s_AC1R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                     const Npp32s *pSrc2, int nSrc2Step,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_32s_AC1R(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_32s_AC4R_Ctx(const Npp32s *pSrc1, int nSrc1Step,
                                     const Npp32s *pSrc2, int nSrc2Step,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_32s_AC4R(const Npp32s *pSrc1, int nSrc1Step,
                                 const Npp32s *pSrc2, int nSrc2Step,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_32f_AC1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pSrc2, int nSrc2Step,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_32f_AC1R(const Npp32f *pSrc1, int nSrc1Step,
                                 const Npp32f *pSrc2, int nSrc2Step,
                                 Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaComp_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                     const Npp32f *pSrc2, int nSrc2Step,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiAlphaOp eAlphaOp,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eAlphaOp SEND_ONLY
 */
NppStatus nppiAlphaComp_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                                 const Npp32f *pSrc2, int nSrc2Step,
                                 Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppiAlphaOp eAlphaOp);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremul_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremul_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremul_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremul_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremul_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremul_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaPremul_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiAlphaPremul_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI);
