#include <npp.h>

// NPP image geometry transforms (libnppig).
//
// Images and batch descriptor lists are device memory and travel as addresses.
// C arrays of transform coefficients and quadrangles and per-plane image
// addresses are host memory and travel by contents. The transform helpers
// (nppiGetAffineTransform and the like) compute on the host, so their outputs
// travel both ways, and scratch buffer sizes come back from host memory.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppig by name at runtime. A call
// with a stream context goes to the server that owns the context's stream, or
// to the runtime's current device when the context names the default stream;
// a call without one goes to the current device.
//
// Each @guard names the NPP releases that export the call: the calls without a
// stream context end with NPP 12, and later calls start at the release that
// added them. A call NPP exports without declaring it in any release is left
// out.

/**
 * @param oSrcROI SEND_ONLY
 * @param pDstRect SEND_RECV
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiGetResizeRect(NppiRect oSrcROI, NppiRect *pDstRect,
                            double nXFactor, double nYFactor, double nXShift,
                            double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                        int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst, int nDstStep,
                                        NppiRect oDstROI, double nXFactor,
                                        double nYFactor, double nXShift,
                                        double nYShift, int eInterpolation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    double nXFactor, double nYFactor,
                                    double nXShift, double nYShift,
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                        int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst, int nDstStep,
                                        NppiRect oDstROI, double nXFactor,
                                        double nYFactor, double nXShift,
                                        double nYShift, int eInterpolation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    double nXFactor, double nYFactor,
                                    double nXShift, double nYShift,
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                        int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst, int nDstStep,
                                        NppiRect oDstROI, double nXFactor,
                                        double nYFactor, double nXShift,
                                        double nYShift, int eInterpolation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    double nXFactor, double nYFactor,
                                    double nXShift, double nYShift,
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_P3R_Ctx(
    const Npp8u *const pSrc[3], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_P3R(const Npp8u *const pSrc[3],
                                    NppiSize oSrcSize, int nSrcStep,
                                    NppiRect oSrcROI, Npp8u *pDst[3],
                                    int nDstStep, NppiRect oDstROI,
                                    double nXFactor, double nYFactor,
                                    double nXShift, double nYShift,
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_P4R_Ctx(
    const Npp8u *const pSrc[4], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp8u *pDst[4], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_P4R(const Npp8u *const pSrc[4],
                                    NppiSize oSrcSize, int nSrcStep,
                                    NppiRect oSrcROI, Npp8u *pDst[4],
                                    int nDstStep, NppiRect oDstROI,
                                    double nXFactor, double nYFactor,
                                    double nXShift, double nYShift,
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep,
                                          NppiRect oDstROI, double nXFactor,
                                          double nYFactor, double nXShift,
                                          double nYShift, int eInterpolation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep,
                                      NppiRect oDstROI, double nXFactor,
                                      double nYFactor, double nXShift,
                                      double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_P3R_Ctx(
    const Npp16u *const pSrc[3], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp16u *pDst[3], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_P3R(const Npp16u *const pSrc[3],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp16u *pDst[3],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_P4R_Ctx(
    const Npp16u *const pSrc[4], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp16u *pDst[4], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16u_P4R(const Npp16u *const pSrc[4],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp16u *pDst[4],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_C1R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp16s *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_C1R(const Npp16s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16s *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_C3R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp16s *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_C3R(const Npp16s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16s *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_C4R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp16s *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_C4R(const Npp16s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16s *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_AC4R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp16s *pDst, int nDstStep,
                                          NppiRect oDstROI, double nXFactor,
                                          double nYFactor, double nXShift,
                                          double nYShift, int eInterpolation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_AC4R(const Npp16s *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16s *pDst, int nDstStep,
                                      NppiRect oDstROI, double nXFactor,
                                      double nYFactor, double nXShift,
                                      double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_P3R_Ctx(
    const Npp16s *const pSrc[3], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp16s *pDst[3], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_P3R(const Npp16s *const pSrc[3],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp16s *pDst[3],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_P4R_Ctx(
    const Npp16s *const pSrc[4], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp16s *pDst[4], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_16s_P4R(const Npp16s *const pSrc[4],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp16s *pDst[4],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep,
                                          NppiRect oDstROI, double nXFactor,
                                          double nYFactor, double nXShift,
                                          double nYShift, int eInterpolation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep,
                                      NppiRect oDstROI, double nXFactor,
                                      double nYFactor, double nXShift,
                                      double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_P3R_Ctx(
    const Npp32f *const pSrc[3], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp32f *pDst[3], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_P3R(const Npp32f *const pSrc[3],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp32f *pDst[3],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_P4R_Ctx(
    const Npp32f *const pSrc[4], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp32f *pDst[4], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_32f_P4R(const Npp32f *const pSrc[4],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp32f *pDst[4],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_C1R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp64f *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_C1R(const Npp64f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_C3R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp64f *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_C3R(const Npp64f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_C4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp64f *pDst, int nDstStep,
                                         NppiRect oDstROI, double nXFactor,
                                         double nYFactor, double nXShift,
                                         double nYShift, int eInterpolation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_C4R(const Npp64f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *pDst, int nDstStep,
                                     NppiRect oDstROI, double nXFactor,
                                     double nYFactor, double nXShift,
                                     double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_AC4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp64f *pDst, int nDstStep,
                                          NppiRect oDstROI, double nXFactor,
                                          double nYFactor, double nXShift,
                                          double nYShift, int eInterpolation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_AC4R(const Npp64f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp64f *pDst, int nDstStep,
                                      NppiRect oDstROI, double nXFactor,
                                      double nYFactor, double nXShift,
                                      double nYShift, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_P3R_Ctx(
    const Npp64f *const pSrc[3], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp64f *pDst[3], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_P3R(const Npp64f *const pSrc[3],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp64f *pDst[3],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_P4R_Ctx(
    const Npp64f *const pSrc[4], NppiSize oSrcSize, int nSrcStep,
    NppiRect oSrcROI, Npp64f *pDst[4], int nDstStep, NppiRect oDstROI,
    double nXFactor, double nYFactor, double nXShift, double nYShift,
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param nXShift SEND_ONLY
 * @param nYShift SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_64f_P4R(const Npp64f *const pSrc[4],
                                     NppiSize oSrcSize, int nSrcStep,
                                     NppiRect oSrcROI, Npp64f *pDst[4],
                                     int nDstStep, NppiRect oDstROI,
                                     double nXFactor, double nYFactor,
                                     double nXShift, double nYShift,
                                     int eInterpolation);
/**
 * @param oSrcROI SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param eInterpolationMode SEND_ONLY
 */
NppStatus nppiResizeAdvancedGetBufferHostSize_8u_C1R(NppiSize oSrcROI,
                                                     NppiSize oDstROI,
                                                     int *hpBufferSize,
                                                     int eInterpolationMode);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eInterpolationMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C1R_Advanced_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, double nXFactor,
    double nYFactor, Npp8u *pBuffer, int eInterpolationMode,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nXFactor SEND_ONLY
 * @param nYFactor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eInterpolationMode SEND_ONLY
 */
NppStatus nppiResizeSqrPixel_8u_C1R_Advanced(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, double nXFactor,
    double nYFactor, Npp8u *pBuffer, int eInterpolationMode);
/**
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param pNewSrcRectOffset SEND_RECV
 */
NppStatus nppiGetResizeTiledSourceOffset(NppiRect oSrcRectROI,
                                         NppiRect oDstRectROI,
                                         NppiPoint *pNewSrcRectOffset);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                            NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep,
                            NppiSize oDstSize, NppiRect oDstRectROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                            NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep,
                            NppiSize oDstSize, NppiRect oDstRectROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_8u_C4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                            NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep,
                            NppiSize oDstSize, NppiRect oDstRectROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp8u *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_8u_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize,
                             NppiRect oSrcRectROI, Npp8u *pDst, int nDstStep,
                             NppiSize oDstSize, NppiRect oDstRectROI,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_8u_P3R_Ctx(const Npp8u *pSrc[3], int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst[3], int nDstStep, NppiSize oDstSize,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_8u_P3R(const Npp8u *pSrc[3], int nSrcStep,
                            NppiSize oSrcSize, NppiRect oSrcRectROI,
                            Npp8u *pDst[3], int nDstStep, NppiSize oDstSize,
                            NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_8u_P4R_Ctx(const Npp8u *pSrc[4], int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcRectROI,
                                Npp8u *pDst[4], int nDstStep, NppiSize oDstSize,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_8u_P4R(const Npp8u *pSrc[4], int nSrcStep,
                            NppiSize oSrcSize, NppiRect oSrcRectROI,
                            Npp8u *pDst[4], int nDstStep, NppiSize oDstSize,
                            NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiRect oSrcRectROI,
                                  Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                              NppiSize oSrcSize, NppiRect oSrcRectROI,
                              Npp16u *pDst, int nDstStep, NppiSize oDstSize,
                              NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16u_P3R_Ctx(const Npp16u *pSrc[3], int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst[3], int nDstStep,
                                 NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16u_P3R(const Npp16u *pSrc[3], int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst[3], int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16u_P4R_Ctx(const Npp16u *pSrc[4], int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst[4], int nDstStep,
                                 NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16u_P4R(const Npp16u *pSrc[4], int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16u *pDst[4], int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiRect oSrcRectROI,
                                  Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                              NppiSize oSrcSize, NppiRect oSrcRectROI,
                              Npp16s *pDst, int nDstStep, NppiSize oDstSize,
                              NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16s_P3R_Ctx(const Npp16s *pSrc[3], int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16s *pDst[3], int nDstStep,
                                 NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16s_P3R(const Npp16s *pSrc[3], int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16s *pDst[3], int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16s_P4R_Ctx(const Npp16s *pSrc[4], int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16s *pDst[4], int nDstStep,
                                 NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16s_P4R(const Npp16s *pSrc[4], int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16s *pDst[4], int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16f *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16f_C1R(const Npp16f *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16f *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16f *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16f_C3R(const Npp16f *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16f *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16f *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_16f_C4R(const Npp16f *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp16f *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiRect oSrcRectROI,
                                  Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                              NppiSize oSrcSize, NppiRect oSrcRectROI,
                              Npp32f *pDst, int nDstStep, NppiSize oDstSize,
                              NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_32f_P3R_Ctx(const Npp32f *pSrc[3], int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst[3], int nDstStep,
                                 NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_32f_P3R(const Npp32f *pSrc[3], int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst[3], int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResize_32f_P4R_Ctx(const Npp32f *pSrc[4], int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst[4], int nDstStep,
                                 NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResize_32f_P4R(const Npp32f *pSrc[4], int nSrcStep,
                             NppiSize oSrcSize, NppiRect oSrcRectROI,
                             Npp32f *pDst[4], int nDstStep, NppiSize oDstSize,
                             NppiRect oDstRectROI, int eInterpolation);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C1R(NppiSize oSmallestSrcSize,
                                 NppiRect oSrcRectROI,
                                 NppiSize oSmallestDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppiResizeBatchCXR *pBatchList,
                                 unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C3R(NppiSize oSmallestSrcSize,
                                 NppiRect oSrcRectROI,
                                 NppiSize oSmallestDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppiResizeBatchCXR *pBatchList,
                                 unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C4R(NppiSize oSmallestSrcSize,
                                 NppiRect oSrcRectROI,
                                 NppiSize oSmallestDstSize,
                                 NppiRect oDstRectROI, int eInterpolation,
                                 NppiResizeBatchCXR *pBatchList,
                                 unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_AC4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_AC4R(NppiSize oSmallestSrcSize,
                                  NppiRect oSrcRectROI,
                                  NppiSize oSmallestDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppiResizeBatchCXR *pBatchList,
                                  unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C1R(NppiSize oSmallestSrcSize,
                                  NppiRect oSrcRectROI,
                                  NppiSize oSmallestDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppiResizeBatchCXR *pBatchList,
                                  unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C3R(NppiSize oSmallestSrcSize,
                                  NppiRect oSrcRectROI,
                                  NppiSize oSmallestDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppiResizeBatchCXR *pBatchList,
                                  unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C4R(NppiSize oSmallestSrcSize,
                                  NppiRect oSrcRectROI,
                                  NppiSize oSmallestDstSize,
                                  NppiRect oDstRectROI, int eInterpolation,
                                  NppiResizeBatchCXR *pBatchList,
                                  unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_AC4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiSize oSmallestDstSize,
    NppiRect oDstRectROI, int eInterpolation, NppiResizeBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oSmallestDstSize SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_AC4R(NppiSize oSmallestSrcSize,
                                   NppiRect oSrcRectROI,
                                   NppiSize oSmallestDstSize,
                                   NppiRect oDstRectROI, int eInterpolation,
                                   NppiResizeBatchCXR *pBatchList,
                                   unsigned int nBatchSize);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C1R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C1R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C3R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C3R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C4R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_C4R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_AC4R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_8u_AC4R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_16f_C1R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_16f_C1R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_16f_C3R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_16f_C3R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_16f_C4R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_16f_C4R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C1R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C1R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C3R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C3R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C4R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_C4R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_AC4R_Advanced_Ctx(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMaxWidth SEND_ONLY
 * @param nMaxHeight SEND_ONLY
 * @param pBatchSrc SEND_ONLY
 * @param pBatchDst SEND_ONLY
 * @param pBatchROI SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiResizeBatch_32f_AC4R_Advanced(
    int nMaxWidth, int nMaxHeight, NppiImageDescriptor *pBatchSrc,
    NppiImageDescriptor *pBatchDst, NppiResizeBatchROI_Advanced *pBatchROI,
    unsigned int nBatchSize, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                               int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep,
                               const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI,
                               int eInterpolation,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, const Npp32f *pXMap, int nXMapStep,
                           const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                           int nDstStep, NppiSize oDstSizeROI,
                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                               int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep,
                               const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI,
                               int eInterpolation,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, const Npp32f *pXMap, int nXMapStep,
                           const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                           int nDstStep, NppiSize oDstSizeROI,
                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                               int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep,
                               const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI,
                               int eInterpolation,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, const Npp32f *pXMap, int nXMapStep,
                           const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                           int nDstStep, NppiSize oDstSizeROI,
                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp8u *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_8u_P3R_Ctx(const Npp8u *const pSrc[3], NppiSize oSrcSize,
                               int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep,
                               const Npp32f *pYMap, int nYMapStep,
                               Npp8u *pDst[3], int nDstStep,
                               NppiSize oDstSizeROI, int eInterpolation,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_8u_P3R(const Npp8u *const pSrc[3], NppiSize oSrcSize,
                           int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                           int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                           Npp8u *pDst[3], int nDstStep, NppiSize oDstSizeROI,
                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_8u_P4R_Ctx(const Npp8u *const pSrc[4], NppiSize oSrcSize,
                               int nSrcStep, NppiRect oSrcROI,
                               const Npp32f *pXMap, int nXMapStep,
                               const Npp32f *pYMap, int nYMapStep,
                               Npp8u *pDst[4], int nDstStep,
                               NppiSize oDstSizeROI, int eInterpolation,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_8u_P4R(const Npp8u *const pSrc[4], NppiSize oSrcSize,
                           int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                           int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                           Npp8u *pDst[4], int nDstStep, NppiSize oDstSizeROI,
                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16u *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep,
                                 const Npp32f *pYMap, int nYMapStep,
                                 Npp16u *pDst, int nDstStep,
                                 NppiSize oDstSizeROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI,
                             const Npp32f *pXMap, int nXMapStep,
                             const Npp32f *pYMap, int nYMapStep, Npp16u *pDst,
                             int nDstStep, NppiSize oDstSizeROI,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16u_P3R_Ctx(const Npp16u *const pSrc[3], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst[3], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16u_P3R(const Npp16u *const pSrc[3], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16u *pDst[3], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16u_P4R_Ctx(const Npp16u *const pSrc[4], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16u *pDst[4], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16u_P4R(const Npp16u *const pSrc[4], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16u *pDst[4], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16s_C1R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16s_C1R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16s_C3R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16s_C3R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16s_C4R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16s_C4R(const Npp16s *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16s *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16s_AC4R_Ctx(const Npp16s *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep,
                                 const Npp32f *pYMap, int nYMapStep,
                                 Npp16s *pDst, int nDstStep,
                                 NppiSize oDstSizeROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16s_AC4R(const Npp16s *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI,
                             const Npp32f *pXMap, int nXMapStep,
                             const Npp32f *pYMap, int nYMapStep, Npp16s *pDst,
                             int nDstStep, NppiSize oDstSizeROI,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16s_P3R_Ctx(const Npp16s *const pSrc[3], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst[3], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16s_P3R(const Npp16s *const pSrc[3], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16s *pDst[3], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_16s_P4R_Ctx(const Npp16s *const pSrc[4], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp16s *pDst[4], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_16s_P4R(const Npp16s *const pSrc[4], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp16s *pDst[4], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 const Npp32f *pXMap, int nXMapStep,
                                 const Npp32f *pYMap, int nYMapStep,
                                 Npp32f *pDst, int nDstStep,
                                 NppiSize oDstSizeROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI,
                             const Npp32f *pXMap, int nXMapStep,
                             const Npp32f *pYMap, int nYMapStep, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSizeROI,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_32f_P3R_Ctx(const Npp32f *const pSrc[3], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst[3], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_32f_P3R(const Npp32f *const pSrc[3], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp32f *pDst[3], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_32f_P4R_Ctx(const Npp32f *const pSrc[4], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp32f *pXMap, int nXMapStep,
                                const Npp32f *pYMap, int nYMapStep,
                                Npp32f *pDst[4], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_32f_P4R(const Npp32f *const pSrc[4], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap,
                            int nXMapStep, const Npp32f *pYMap, int nYMapStep,
                            Npp32f *pDst[4], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_64f_C1R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep,
                                const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_64f_C1R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                            Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_64f_C3R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep,
                                const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_64f_C3R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                            Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_64f_C4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep,
                                const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst, int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_64f_C4R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                            Npp64f *pDst, int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_64f_AC4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 const Npp64f *pXMap, int nXMapStep,
                                 const Npp64f *pYMap, int nYMapStep,
                                 Npp64f *pDst, int nDstStep,
                                 NppiSize oDstSizeROI, int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_64f_AC4R(const Npp64f *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI,
                             const Npp64f *pXMap, int nXMapStep,
                             const Npp64f *pYMap, int nYMapStep, Npp64f *pDst,
                             int nDstStep, NppiSize oDstSizeROI,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_64f_P3R_Ctx(const Npp64f *const pSrc[3], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep,
                                const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst[3], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_64f_P3R(const Npp64f *const pSrc[3], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                            Npp64f *pDst[3], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRemap_64f_P4R_Ctx(const Npp64f *const pSrc[4], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI,
                                const Npp64f *pXMap, int nXMapStep,
                                const Npp64f *pYMap, int nYMapStep,
                                Npp64f *pDst[4], int nDstStep,
                                NppiSize oDstSizeROI, int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pXMap SEND_ONLY
 * @param nXMapStep SEND_ONLY
 * @param pYMap SEND_ONLY
 * @param nYMapStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRemap_64f_P4R(const Npp64f *const pSrc[4], NppiSize oSrcSize,
                            int nSrcStep, NppiRect oSrcROI, const Npp64f *pXMap,
                            int nXMapStep, const Npp64f *pYMap, int nYMapStep,
                            Npp64f *pDst[4], int nDstStep, NppiSize oDstSizeROI,
                            int eInterpolation);
/**
 * @param oSrcROI SEND_ONLY
 * @param aQuad SEND_RECV DEREF
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 */
NppStatus nppiGetRotateQuad(NppiRect oSrcROI, double aQuad[4][2], double nAngle,
                            double nShiftX, double nShiftY);
/**
 * @param oSrcROI SEND_ONLY
 * @param aBoundingBox SEND_RECV DEREF
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 */
NppStatus nppiGetRotateBound(NppiRect oSrcROI, double aBoundingBox[2][2],
                             double nAngle, double nShiftX, double nShiftY);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, double nAngle,
                                double nShiftX, double nShiftY,
                                int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp8u *pDst, int nDstStep,
                            NppiRect oDstROI, double nAngle, double nShiftX,
                            double nShiftY, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, double nAngle,
                                double nShiftX, double nShiftY,
                                int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp8u *pDst, int nDstStep,
                            NppiRect oDstROI, double nAngle, double nShiftX,
                            double nShiftY, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, double nAngle,
                                double nShiftX, double nShiftY,
                                int eInterpolation,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp8u *pDst, int nDstStep,
                            NppiRect oDstROI, double nAngle, double nShiftX,
                            double nShiftY, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                             NppiRect oSrcROI, Npp8u *pDst, int nDstStep,
                             NppiRect oDstROI, double nAngle, double nShiftX,
                             double nShiftY, int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle,
                             double nShiftX, double nShiftY,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle,
                             double nShiftX, double nShiftY,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle,
                             double nShiftX, double nShiftY,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize,
                                  int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                  int nDstStep, NppiRect oDstROI, double nAngle,
                                  double nShiftX, double nShiftY,
                                  int eInterpolation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                              int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                              int nDstStep, NppiRect oDstROI, double nAngle,
                              double nShiftX, double nShiftY,
                              int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle,
                             double nShiftX, double nShiftY,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle,
                             double nShiftX, double nShiftY,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle,
                                 double nShiftX, double nShiftY,
                                 int eInterpolation,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                             int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle,
                             double nShiftX, double nShiftY,
                             int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRotate_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize,
                                  int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                  int nDstStep, NppiRect oDstROI, double nAngle,
                                  double nShiftX, double nShiftY,
                                  int eInterpolation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param nAngle SEND_ONLY
 * @param nShiftX SEND_ONLY
 * @param nShiftY SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiRotate_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                              int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                              int nDstStep, NppiRect oDstROI, double nAngle,
                              double nShiftX, double nShiftY,
                              int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oROI, NppiAxis flip,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                             NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oROI, NppiAxis flip,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                             NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oROI, NppiAxis flip,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                             NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_C4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI,
                               NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                               NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_C1IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_C1IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_C3IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_C3IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_C4IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_C4IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_AC4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                              int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32s_AC4IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep,
                                   NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32s_AC4IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI,
                               NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oROI, NppiAxis flip,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI,
                              NppiAxis flip);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oROI, NppiAxis flip);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirror_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oROI SEND_ONLY
 * @param flip SEND_ONLY
 */
NppStatus nppiMirror_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI,
                               NppiAxis flip);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C1R_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                      NppiMirrorBatchCXR *pBatchList,
                                      int nBatchSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C1R(NppiSize oSizeROI, NppiAxis flip,
                                  NppiMirrorBatchCXR *pBatchList,
                                  int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C1IR_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                       NppiMirrorBatchCXR *pBatchList,
                                       int nBatchSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C1IR(NppiSize oSizeROI, NppiAxis flip,
                                   NppiMirrorBatchCXR *pBatchList,
                                   int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C3R_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                      NppiMirrorBatchCXR *pBatchList,
                                      int nBatchSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C3R(NppiSize oSizeROI, NppiAxis flip,
                                  NppiMirrorBatchCXR *pBatchList,
                                  int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C3IR_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                       NppiMirrorBatchCXR *pBatchList,
                                       int nBatchSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C3IR(NppiSize oSizeROI, NppiAxis flip,
                                   NppiMirrorBatchCXR *pBatchList,
                                   int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C4R_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                      NppiMirrorBatchCXR *pBatchList,
                                      int nBatchSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C4R(NppiSize oSizeROI, NppiAxis flip,
                                  NppiMirrorBatchCXR *pBatchList,
                                  int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C4IR_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                       NppiMirrorBatchCXR *pBatchList,
                                       int nBatchSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_C4IR(NppiSize oSizeROI, NppiAxis flip,
                                   NppiMirrorBatchCXR *pBatchList,
                                   int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_AC4R_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                       NppiMirrorBatchCXR *pBatchList,
                                       int nBatchSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_AC4R(NppiSize oSizeROI, NppiAxis flip,
                                   NppiMirrorBatchCXR *pBatchList,
                                   int nBatchSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_AC4IR_Ctx(NppiSize oSizeROI, NppiAxis flip,
                                        NppiMirrorBatchCXR *pBatchList,
                                        int nBatchSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSizeROI SEND_ONLY
 * @param flip SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiMirrorBatch_32f_AC4IR(NppiSize oSizeROI, NppiAxis flip,
                                    NppiMirrorBatchCXR *pBatchList,
                                    int nBatchSize);
/**
 * @param oSrcROI SEND_ONLY
 * @param aQuad SEND_ONLY DEREF
 * @param aCoeffs SEND_RECV DEREF
 */
NppStatus nppiGetAffineTransform(NppiRect oSrcROI, const double aQuad[4][2],
                                 double aCoeffs[2][3]);
/**
 * @param oSrcROI SEND_ONLY
 * @param aQuad SEND_RECV DEREF
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiGetAffineQuad(NppiRect oSrcROI, double aQuad[4][2],
                            const double aCoeffs[2][3]);
/**
 * @param oSrcROI SEND_ONLY
 * @param aBound SEND_RECV DEREF
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiGetAffineBound(NppiRect oSrcROI, double aBound[2][2],
                             const double aCoeffs[2][3]);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI,
                                const double aCoeffs[2][3], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI,
                                const double aCoeffs[2][3], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI,
                                const double aCoeffs[2][3], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp8u *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep,
                          NppiRect oSrcROI, Npp8u *pDst[3], int nDstStep,
                          NppiRect oDstROI, const double aCoeffs[2][3],
                          int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3],
                                int nDstStep, NppiRect oDstROI,
                                const double aCoeffs[2][3], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_8u_P4R_Ctx(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep,
                          NppiRect oSrcROI, Npp8u *pDst[4], int nDstStep,
                          NppiRect oDstROI, const double aCoeffs[2][3],
                          int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize,
                                int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[4],
                                int nDstStep, NppiRect oDstROI,
                                const double aCoeffs[2][3], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp16u *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp16u *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp16u *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp16u *pDst, int nDstStep,
                            NppiRect oDstROI, const double aCoeffs[2][3],
                            int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                  int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                  int nDstStep, NppiRect oDstROI,
                                  const double aCoeffs[2][3],
                                  int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_P3R_Ctx(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp16u *pDst[3], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_P4R_Ctx(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp16u *pDst[4], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32s_C1R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp32s *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32s_C3R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp32s *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32s_C4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp32s *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32s_AC4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp32s *pDst, int nDstStep,
                            NppiRect oDstROI, const double aCoeffs[2][3],
                            int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                  int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                  int nDstStep, NppiRect oDstROI,
                                  const double aCoeffs[2][3],
                                  int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_P3R_Ctx(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp32s *pDst[3], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_P4R_Ctx(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp32s *pDst[4], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16f_C1R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp16f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16f_C1R(const Npp16f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16f_C3R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp16f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16f_C3R(const Npp16f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_16f_C4R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp16f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_16f_C4R(const Npp16f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp16f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp32f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp32f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp32f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp32f *pDst, int nDstStep,
                            NppiRect oDstROI, const double aCoeffs[2][3],
                            int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                  int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                  int nDstStep, NppiRect oDstROI,
                                  const double aCoeffs[2][3],
                                  int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_P3R_Ctx(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp32f *pDst[3], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_P4R_Ctx(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp32f *pDst[4], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_64f_C1R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp64f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_C1R(const Npp64f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_64f_C3R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp64f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_C3R(const Npp64f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_64f_C4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                           NppiRect oSrcROI, Npp64f *pDst, int nDstStep,
                           NppiRect oDstROI, const double aCoeffs[2][3],
                           int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_C4R(const Npp64f *pSrc, NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                 int nDstStep, NppiRect oDstROI,
                                 const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpAffine_64f_AC4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep,
                            NppiRect oSrcROI, Npp64f *pDst, int nDstStep,
                            NppiRect oDstROI, const double aCoeffs[2][3],
                            int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_AC4R(const Npp64f *pSrc, NppiSize oSrcSize,
                                  int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                  int nDstStep, NppiRect oDstROI,
                                  const double aCoeffs[2][3],
                                  int eInterpolation);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_P3R_Ctx(
    const Npp64f *aSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp64f *aDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_P3R(const Npp64f *aSrc[3], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp64f *aDst[3], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_P4R_Ctx(
    const Npp64f *aSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp64f *aDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffine_64f_P4R(const Npp64f *aSrc[4], NppiSize oSrcSize,
                                 int nSrcStep, NppiRect oSrcROI,
                                 Npp64f *aDst[4], int nDstStep,
                                 NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation);
/**
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatchInit_Ctx(NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatchInit(NppiWarpAffineBatchCXR *pBatchList,
                                  unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_C1R(NppiSize oSmallestSrcSize,
                                     NppiRect oSrcRectROI, NppiRect oDstRectROI,
                                     int eInterpolation,
                                     NppiWarpAffineBatchCXR *pBatchList,
                                     unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_C3R(NppiSize oSmallestSrcSize,
                                     NppiRect oSrcRectROI, NppiRect oDstRectROI,
                                     int eInterpolation,
                                     NppiWarpAffineBatchCXR *pBatchList,
                                     unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_C4R(NppiSize oSmallestSrcSize,
                                     NppiRect oSrcRectROI, NppiRect oDstRectROI,
                                     int eInterpolation,
                                     NppiWarpAffineBatchCXR *pBatchList,
                                     unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_AC4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_8u_AC4R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_16f_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_16f_C1R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_16f_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_16f_C3R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_16f_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_16f_C4R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_C1R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_C3R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_C4R(NppiSize oSmallestSrcSize,
                                      NppiRect oSrcRectROI,
                                      NppiRect oDstRectROI, int eInterpolation,
                                      NppiWarpAffineBatchCXR *pBatchList,
                                      unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_AC4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpAffineBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpAffineBatch_32f_AC4R(NppiSize oSmallestSrcSize,
                                       NppiRect oSrcRectROI,
                                       NppiRect oDstRectROI, int eInterpolation,
                                       NppiWarpAffineBatchCXR *pBatchList,
                                       unsigned int nBatchSize);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_C1R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_C3R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_C4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_AC4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_P3R_Ctx(
    const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_P4R_Ctx(
    const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst[4], int nDstStep,
                                    NppiRect oDstROI,
                                    const double aCoeffs[2][3],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_C1R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_C3R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_C4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_AC4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[2][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_P3R_Ctx(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst[3], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_P4R_Ctx(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst[4], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_C1R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_C3R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_C4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_AC4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[2][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_P3R_Ctx(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst[3], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_P4R_Ctx(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst[4], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_C1R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_C3R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_C4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_AC4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[2][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_P3R_Ctx(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst[3], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_P4R_Ctx(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineBack_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst[4], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[2][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_C1R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    const double aSrcQuad[4][2], Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aDstQuad[4][2],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_C3R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    const double aSrcQuad[4][2], Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aDstQuad[4][2],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_C4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    const double aSrcQuad[4][2], Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI,
                                    const double aDstQuad[4][2],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_AC4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_P3R_Ctx(
    const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst[3], int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    const double aSrcQuad[4][2], Npp8u *pDst[3],
                                    int nDstStep, NppiRect oDstROI,
                                    const double aDstQuad[4][2],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_P4R_Ctx(
    const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst[4], int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize,
                                    int nSrcStep, NppiRect oSrcROI,
                                    const double aSrcQuad[4][2], Npp8u *pDst[4],
                                    int nDstStep, NppiRect oDstROI,
                                    const double aDstQuad[4][2],
                                    int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_C1R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp16u *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_C3R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp16u *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_C4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp16u *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_AC4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      const double aSrcQuad[4][2], Npp16u *pDst,
                                      int nDstStep, NppiRect oDstROI,
                                      const double aDstQuad[4][2],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_P3R_Ctx(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_P3R(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_P4R_Ctx(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_16u_P4R(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_C1R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp32s *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_C3R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp32s *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_C4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp32s *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_AC4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      const double aSrcQuad[4][2], Npp32s *pDst,
                                      int nDstStep, NppiRect oDstROI,
                                      const double aDstQuad[4][2],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_P3R_Ctx(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_P3R(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_P4R_Ctx(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32s_P4R(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_C1R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp32f *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_C3R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp32f *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_C4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     const double aSrcQuad[4][2], Npp32f *pDst,
                                     int nDstStep, NppiRect oDstROI,
                                     const double aDstQuad[4][2],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_AC4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      const double aSrcQuad[4][2], Npp32f *pDst,
                                      int nDstStep, NppiRect oDstROI,
                                      const double aDstQuad[4][2],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_P3R_Ctx(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_P3R(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_P4R_Ctx(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpAffineQuad_32f_P4R(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param oSrcROI SEND_ONLY
 * @param quad SEND_ONLY DEREF
 * @param aCoeffs SEND_RECV DEREF
 */
NppStatus nppiGetPerspectiveTransform(NppiRect oSrcROI, const double quad[4][2],
                                      double aCoeffs[3][3]);
/**
 * @param oSrcROI SEND_ONLY
 * @param quad SEND_RECV DEREF
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiGetPerspectiveQuad(NppiRect oSrcROI, double quad[4][2],
                                 const double aCoeffs[3][3]);
/**
 * @param oSrcROI SEND_ONLY
 * @param bound SEND_RECV DEREF
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiGetPerspectiveBound(NppiRect oSrcROI, double bound[2][2],
                                  const double aCoeffs[3][3]);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_C1R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[3][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_C3R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[3][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_C4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst, int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[3][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_AC4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp8u *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_P3R_Ctx(
    const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst[3], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[3][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_P4R_Ctx(
    const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize,
                                     int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst[4], int nDstStep,
                                     NppiRect oDstROI,
                                     const double aCoeffs[3][3],
                                     int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_C1R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_C3R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_C4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_AC4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                       int nSrcStep, NppiRect oSrcROI,
                                       Npp16u *pDst, int nDstStep,
                                       NppiRect oDstROI,
                                       const double aCoeffs[3][3],
                                       int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_P3R_Ctx(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst[3], int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_P4R_Ctx(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst[4], int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_C1R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_C3R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_C4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_AC4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                       int nSrcStep, NppiRect oSrcROI,
                                       Npp32s *pDst, int nDstStep,
                                       NppiRect oDstROI,
                                       const double aCoeffs[3][3],
                                       int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_P3R_Ctx(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst[3], int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_P4R_Ctx(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst[4], int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16f_C1R_Ctx(
    const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16f_C1R(const Npp16f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16f_C3R_Ctx(
    const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16f_C3R(const Npp16f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_16f_C4R_Ctx(
    const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_16f_C4R(const Npp16f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_C1R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_C3R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_C4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_AC4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                       int nSrcStep, NppiRect oSrcROI,
                                       Npp32f *pDst, int nDstStep,
                                       NppiRect oDstROI,
                                       const double aCoeffs[3][3],
                                       int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_P3R_Ctx(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst[3], int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_P4R_Ctx(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspective_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize,
                                      int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst[4], int nDstStep,
                                      NppiRect oDstROI,
                                      const double aCoeffs[3][3],
                                      int eInterpolation);
/**
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiWarpPerspectiveBatchInit_Ctx(NppiWarpPerspectiveBatchCXR *pBatchList,
                                 unsigned int nBatchSize,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatchInit(NppiWarpPerspectiveBatchCXR *pBatchList,
                                       unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_8u_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus
nppiWarpPerspectiveBatch_8u_C1R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppiWarpPerspectiveBatchCXR *pBatchList,
                                unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_8u_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus
nppiWarpPerspectiveBatch_8u_C3R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppiWarpPerspectiveBatchCXR *pBatchList,
                                unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_8u_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus
nppiWarpPerspectiveBatch_8u_C4R(NppiSize oSmallestSrcSize, NppiRect oSrcRectROI,
                                NppiRect oDstRectROI, int eInterpolation,
                                NppiWarpPerspectiveBatchCXR *pBatchList,
                                unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_8u_AC4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_8u_AC4R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_16f_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_16f_C1R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_16f_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_16f_C3R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_16f_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_16f_C4R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_C1R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_C1R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_C3R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_C3R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_C4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_C4R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_AC4R_Ctx(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oSmallestSrcSize SEND_ONLY
 * @param oSrcRectROI SEND_ONLY
 * @param oDstRectROI SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBatch_32f_AC4R(
    NppiSize oSmallestSrcSize, NppiRect oSrcRectROI, NppiRect oDstRectROI,
    int eInterpolation, NppiWarpPerspectiveBatchCXR *pBatchList,
    unsigned int nBatchSize);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_C1R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep,
                                         NppiRect oDstROI,
                                         const double aCoeffs[3][3],
                                         int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_C3R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep,
                                         NppiRect oDstROI,
                                         const double aCoeffs[3][3],
                                         int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_C4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                         int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep,
                                         NppiRect oDstROI,
                                         const double aCoeffs[3][3],
                                         int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_AC4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp8u *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_P3R_Ctx(
    const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_P3R(const Npp8u *pSrc[3],
                                         NppiSize oSrcSize, int nSrcStep,
                                         NppiRect oSrcROI, Npp8u *pDst[3],
                                         int nDstStep, NppiRect oDstROI,
                                         const double aCoeffs[3][3],
                                         int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_P4R_Ctx(
    const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_8u_P4R(const Npp8u *pSrc[4],
                                         NppiSize oSrcSize, int nSrcStep,
                                         NppiRect oSrcROI, Npp8u *pDst[4],
                                         int nDstStep, NppiRect oDstROI,
                                         const double aCoeffs[3][3],
                                         int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_C1R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_C3R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_C4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_AC4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_AC4R(const Npp16u *pSrc,
                                           NppiSize oSrcSize, int nSrcStep,
                                           NppiRect oSrcROI, Npp16u *pDst,
                                           int nDstStep, NppiRect oDstROI,
                                           const double aCoeffs[3][3],
                                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_P3R_Ctx(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_P3R(const Npp16u *pSrc[3],
                                          NppiSize oSrcSize, int nSrcStep,
                                          NppiRect oSrcROI, Npp16u *pDst[3],
                                          int nDstStep, NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_P4R_Ctx(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_16u_P4R(const Npp16u *pSrc[4],
                                          NppiSize oSrcSize, int nSrcStep,
                                          NppiRect oSrcROI, Npp16u *pDst[4],
                                          int nDstStep, NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_C1R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_C3R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_C4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_AC4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_AC4R(const Npp32s *pSrc,
                                           NppiSize oSrcSize, int nSrcStep,
                                           NppiRect oSrcROI, Npp32s *pDst,
                                           int nDstStep, NppiRect oDstROI,
                                           const double aCoeffs[3][3],
                                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_P3R_Ctx(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_P3R(const Npp32s *pSrc[3],
                                          NppiSize oSrcSize, int nSrcStep,
                                          NppiRect oSrcROI, Npp32s *pDst[3],
                                          int nDstStep, NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_P4R_Ctx(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32s_P4R(const Npp32s *pSrc[4],
                                          NppiSize oSrcSize, int nSrcStep,
                                          NppiRect oSrcROI, Npp32s *pDst[4],
                                          int nDstStep, NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_C1R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_C3R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_C4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize,
                                          int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep,
                                          NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_AC4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_AC4R(const Npp32f *pSrc,
                                           NppiSize oSrcSize, int nSrcStep,
                                           NppiRect oSrcROI, Npp32f *pDst,
                                           int nDstStep, NppiRect oDstROI,
                                           const double aCoeffs[3][3],
                                           int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_P3R_Ctx(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_P3R(const Npp32f *pSrc[3],
                                          NppiSize oSrcSize, int nSrcStep,
                                          NppiRect oSrcROI, Npp32f *pDst[3],
                                          int nDstStep, NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_P4R_Ctx(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
    int eInterpolation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveBack_32f_P4R(const Npp32f *pSrc[4],
                                          NppiSize oSrcSize, int nSrcStep,
                                          NppiRect oSrcROI, Npp32f *pDst[4],
                                          int nDstStep, NppiRect oDstROI,
                                          const double aCoeffs[3][3],
                                          int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_C1R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_C1R(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_C3R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_C3R(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_C4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_C4R(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_AC4R_Ctx(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_AC4R(
    const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_P3R_Ctx(
    const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst[3], int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_P3R(
    const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst[3], int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_P4R_Ctx(
    const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst[4], int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_8u_P4R(
    const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp8u *pDst[4], int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_C1R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_C1R(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_C3R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_C3R(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_C4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_C4R(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_AC4R_Ctx(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_AC4R(
    const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_P3R_Ctx(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_P3R(
    const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_P4R_Ctx(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_16u_P4R(
    const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp16u *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_C1R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_C1R(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_C3R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_C3R(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_C4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_C4R(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_AC4R_Ctx(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_AC4R(
    const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_P3R_Ctx(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_P3R(
    const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_P4R_Ctx(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32s_P4R(
    const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32s *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_C1R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_C1R(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_C3R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_C3R(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_C4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_C4R(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_AC4R_Ctx(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_AC4R(
    const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst, int nDstStep, NppiRect oDstROI,
    const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_P3R_Ctx(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_P3R(
    const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[3], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_P4R_Ctx(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param oSrcSize SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param aSrcQuad SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oDstROI SEND_ONLY
 * @param aDstQuad SEND_ONLY DEREF
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiWarpPerspectiveQuad_32f_P4R(
    const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    const double aSrcQuad[4][2], Npp32f *pDst[4], int nDstStep,
    NppiRect oDstROI, const double aDstQuad[4][2], int eInterpolation);
