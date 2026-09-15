#include <npp.h>

// NPP image color conversion (libnppicc).
//
// Images, lookup tables, palettes and batch descriptor lists are device memory
// and travel as addresses. C arrays of coefficients, color twists, per-plane
// steps, levels, color keys and per-plane image or table addresses are host
// memory and travel by contents.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppicc by name at runtime. A
// call with a stream context goes to the server that owns the context's stream,
// or to the runtime's current device when the context names the default stream;
// a call without one goes to the current device.
//
// Each @guard names the NPP releases that export the call: the calls without a
// stream context end with NPP 12, and later calls start at the release that
// added them. A call NPP exports without declaring it in any release is left
// out.

/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToYUV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToYUV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3],
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiBGRToYUV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiBGRToYUV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3],
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYUV_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiYUVToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiYUVToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUVToRGB_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUVToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep,
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGBBatch_8u_C3R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                       NppiImageDescriptor *pDstBatchList,
                                       int nBatchSize, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUVToRGBBatch_8u_C3R(const NppiImageDescriptor *pSrcBatchList,
                                   NppiImageDescriptor *pDstBatchList,
                                   int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGBBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYUVToRGBBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                           NppiImageDescriptor *pDstBatchList, int nBatchSize,
                           NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiYUVToRGBBatch_8u_C3R_Advanced_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                      NppiImageDescriptor *pDstBatchList,
                                      int nBatchSize, NppiSize oMaxSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus
nppiYUVToRGBBatch_8u_C3R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                  NppiImageDescriptor *pDstBatchList,
                                  int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToRGBBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYUVToRGBBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGR_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiYUVToBGR_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGR_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiYUVToBGR_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGR_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUVToBGR_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUVToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep,
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGRBatch_8u_C3R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                       NppiImageDescriptor *pDstBatchList,
                                       int nBatchSize, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUVToBGRBatch_8u_C3R(const NppiImageDescriptor *pSrcBatchList,
                                   NppiImageDescriptor *pDstBatchList,
                                   int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGRBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYUVToBGRBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                           NppiImageDescriptor *pDstBatchList, int nBatchSize,
                           NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiYUVToBGRBatch_8u_C3R_Advanced_Ctx(const NppiImageDescriptor *pSrcBatchList,
                                      NppiImageDescriptor *pDstBatchList,
                                      int nBatchSize, NppiSize oMaxSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus
nppiYUVToBGRBatch_8u_C3R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                  NppiImageDescriptor *pDstBatchList,
                                  int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUVToBGRBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYUVToBGRBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                                 Npp8u *pDst[3], int rDstStep[3],
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst[3],
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                 Npp8u *pDst[3], int nDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                       int rSrcStep[3], Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                   Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_P3AC4R_Ctx(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_P3AC4R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGBBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYUV422ToRGBBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                              NppiImageDescriptor *pDstBatchList,
                              int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGBBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYUV422ToRGBBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToBGRBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYUV422ToBGRBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                              NppiImageDescriptor *pDstBatchList,
                              int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToBGRBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYUV422ToBGRBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV420_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV420_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                                 Npp8u *pDst[3], int rDstStep[3],
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYUV420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst[3],
                                     int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                 Npp8u *pDst[3], int nDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                       int rSrcStep[3], Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                   Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                       int rSrcStep[3], Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3C4R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                   Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3AC4R_Ctx(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_P3AC4R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGBBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYUV420ToRGBBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                              NppiImageDescriptor *pDstBatchList,
                              int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGBBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToRGBBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToRGB_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                             int rSrcStep, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R(const Npp8u *const pSrc[2],
                                         int rSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                            int rSrcStep, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R(const Npp8u *const pSrc[2],
                                        int rSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_601_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                         int rSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV21ToRGB_8u_P2C4R_Ctx(const Npp8u *const pSrc[2], int rSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV21ToRGB_8u_P2C4R(const Npp8u *const pSrc[2], int rSrcStep,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYUV420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst[3], int rDstStep[3],
                                        NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYUV420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                       int rSrcStep[3], Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                   Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToBGR_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                       int rSrcStep[3], Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToBGR_8u_P3C4R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                   Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToBGRBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYUV420ToBGRBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                              NppiImageDescriptor *pDstBatchList,
                              int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToBGRBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYUV420ToBGRBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToBGR_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToBGR_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                             int rSrcStep, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R(const Npp8u *const pSrc[2],
                                         int rSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                            int rSrcStep, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToBGR_709CSC_8u_P2C3R(const Npp8u *const pSrc[2],
                                        int rSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToBGR_601_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                         int rSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV21ToBGR_8u_P2C4R_Ctx(const Npp8u *const pSrc[2], int rSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV21ToBGR_8u_P2C4R(const Npp8u *const pSrc[2], int rSrcStep,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                                Npp8u *pDst[3], int nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pDst[3], int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                                Npp8u *pDst[3], int nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nAval,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCrToRGB_8u_P3C4R(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nAval);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiYCbCrToRGBBatch_8u_C3R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                               NppiImageDescriptor *pDstBatchList,
                               int nBatchSize, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToRGBBatch_8u_C3R(const NppiImageDescriptor *pSrcBatchList,
                                     NppiImageDescriptor *pDstBatchList,
                                     int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGBBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYCbCrToRGBBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                             NppiImageDescriptor *pDstBatchList, int nBatchSize,
                             NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGBBatch_8u_C3R_Advanced_Ctx(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus
nppiYCbCrToRGBBatch_8u_C3R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                    NppiImageDescriptor *pDstBatchList,
                                    int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToRGBBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToRGBBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nAval,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_8u_P3C4R(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nAval);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiYCbCrToBGRBatch_8u_C3R_Ctx(const NppiImageDescriptor *pSrcBatchList,
                               NppiImageDescriptor *pDstBatchList,
                               int nBatchSize, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToBGRBatch_8u_C3R(const NppiImageDescriptor *pSrcBatchList,
                                     NppiImageDescriptor *pDstBatchList,
                                     int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGRBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus
nppiYCbCrToBGRBatch_8u_P3C3R(const NppiImageDescriptor *const pSrcBatchList[3],
                             NppiImageDescriptor *pDstBatchList, int nBatchSize,
                             NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGRBatch_8u_C3R_Advanced_Ctx(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus
nppiYCbCrToBGRBatch_8u_C3R_Advanced(const NppiImageDescriptor *pSrcBatchList,
                                    NppiImageDescriptor *pDstBatchList,
                                    int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGRBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToBGRBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_709CSC_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                             int nSrcStep, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_709CSC_8u_P3C3R(const Npp8u *const pSrc[3],
                                         int nSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_709CSC_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                             int nSrcStep, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             Npp8u nAval,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCrToBGR_709CSC_8u_P3C4R(const Npp8u *const pSrc[3],
                                         int nSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         Npp8u nAval);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                         int nSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_8u_P3C2R(const Npp8u *const pSrc[3], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_8u_P3C3R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGBBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGBBatch_8u_P3C3R(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGBBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCrCb422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCrCb422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCrCb422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                         int nSrcStep, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCrCb422_8u_P3C2R(const Npp8u *const pSrc[3], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb422ToRGB_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb422ToRGB_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_AC4C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_AC4C2R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_8u_C2C3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_8u_C2C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nAval,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_8u_C2C4R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_8u_P3C3R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGRBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGRBatch_8u_P3C3R(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGRBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToCbYCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToCbYCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToCbYCr422Gamma_8u_C3C2R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToCbYCr422_8u_AC4C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToCbYCr422_8u_AC4C2R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToCbYCr422_709HDTV_8u_C3C2R_Ctx(const Npp8u *pSrc,
                                                 int nSrcStep, Npp8u *pDst,
                                                 int nDstStep,
                                                 NppiSize oSizeROI,
                                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToCbYCr422_709HDTV_8u_C3C2R(const Npp8u *pSrc, int nSrcStep,
                                             Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToCbYCr422_709HDTV_8u_AC4C2R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToCbYCr422_709HDTV_8u_AC4C2R(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToBGR_8u_C2C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nAval,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiCbYCr422ToBGR_8u_C2C4R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C3R_Ctx(const Npp8u *pSrc,
                                                 int nSrcStep, Npp8u *pDst,
                                                 int nDstStep,
                                                 NppiSize oSizeROI,
                                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C3R(const Npp8u *pSrc, int nSrcStep,
                                             Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C4R_Ctx(const Npp8u *pSrc,
                                                 int nSrcStep, Npp8u *pDst,
                                                 int nDstStep,
                                                 NppiSize oSizeROI, Npp8u nAval,
                                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C4R(const Npp8u *pSrc, int nSrcStep,
                                             Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGB_8u_P3C3R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGBBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGBBatch_8u_P3C3R(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGBBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCrCb420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCrCb420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb420ToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         Npp8u nAval,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCrCb420ToRGB_8u_P3C4R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_709CSC_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                                Npp8u *pDst[3], int rDstStep[3],
                                                NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_709CSC_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst[3], int rDstStep[3],
                                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_709CSC_8u_AC4P3R_Ctx(const Npp8u *pSrc,
                                                 int nSrcStep, Npp8u *pDst[3],
                                                 int rDstStep[3],
                                                 NppiSize oSizeROI,
                                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_709CSC_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                             Npp8u *pDst[3], int rDstStep[3],
                                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
    NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_709HDTV_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_709CSC_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                                Npp8u *pDst[3], int rDstStep[3],
                                                NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_709CSC_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst[3], int rDstStep[3],
                                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_709CSC_8u_AC4P3R_Ctx(const Npp8u *pSrc,
                                                 int nSrcStep, Npp8u *pDst[3],
                                                 int rDstStep[3],
                                                 NppiSize oSizeROI,
                                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_709CSC_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                             Npp8u *pDst[3], int rDstStep[3],
                                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_8u_P3C3R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         Npp8u nAval,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_8u_P3C4R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGRBatch_8u_P3C3R_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGRBatch_8u_P3C3R(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oSizeROI);
/**
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced_Ctx(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcBatchList SEND_ONLY DEREF
 * @param pDstBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param oMaxSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGRBatch_8u_P3C3R_Advanced(
    const NppiImageDescriptor *const pSrcBatchList[3],
    NppiImageDescriptor *pDstBatchList, int nBatchSize, NppiSize oMaxSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_709CSC_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                                int rSrcStep[3], Npp8u *pDst,
                                                int nDstStep, NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_709CSC_8u_P3C3R(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_709HDTV_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                                 int rSrcStep[3], Npp8u *pDst,
                                                 int nDstStep,
                                                 NppiSize oSizeROI, Npp8u nAval,
                                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_709HDTV_8u_P3C4R(const Npp8u *const pSrc[3],
                                             int rSrcStep[3], Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             Npp8u nAval);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCrCb420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst[3], int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pDst[3], int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst[4], int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep,
                                   Npp8u *pDst[4], int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_8u_P3C3R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         Npp8u nAval,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_8u_P3C4R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_8u_P3C3R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3],
                                         int rSrcStep[3], Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         Npp8u nAval,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nAval SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_8u_P3C4R(const Npp8u *const pSrc[3],
                                     int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToXYZ_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToXYZ_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToXYZ_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToXYZ_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXYZToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiXYZToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiXYZToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiXYZToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToLUV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToLUV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToLUV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToLUV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUVToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiLUVToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUVToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiLUVToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToLab_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiBGRToLab_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLabToBGR_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiLabToBGR_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToYCC_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToYCC_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCCToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiYCCToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCCToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiYCCToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCCKToCMYK_JPEG_601_8u_P4R_Ctx(const Npp8u *pSrc[4], int nSrcStep,
                                             Npp8u *pDst[4], int nDstStep,
                                             NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCCKToCMYK_JPEG_601_8u_P4R(const Npp8u *pSrc[4], int nSrcStep,
                                         Npp8u *pDst[4], int nDstStep,
                                         NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R_Ctx(const Npp8u *pSrc[4],
                                                int nSrcStep, Npp8u *pDst[3],
                                                int nDstStep, NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToRGB_JPEG_8u_P4P3R(const Npp8u *pSrc[4], int nSrcStep,
                                            Npp8u *pDst[3], int nDstStep,
                                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R_Ctx(const Npp8u *pSrc[4],
                                                int nSrcStep, Npp8u *pDst,
                                                int nDstStep, NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToRGB_JPEG_8u_P4C3R(const Npp8u *pSrc[4], int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R_Ctx(const Npp8u *pSrc[4],
                                                int nSrcStep, Npp8u *pDst[3],
                                                int nDstStep, NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToBGR_JPEG_8u_P4P3R(const Npp8u *pSrc[4], int nSrcStep,
                                            Npp8u *pDst[3], int nDstStep,
                                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R_Ctx(const Npp8u *pSrc[4],
                                                int nSrcStep, Npp8u *pDst,
                                                int nDstStep, NppiSize oSizeROI,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCMYKOrYCCKToBGR_JPEG_8u_P4C3R(const Npp8u *pSrc[4], int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToHLS_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToHLS_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToHLS_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToHLS_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiHLSToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiHLSToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiBGRToHLS_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3],
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep,
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AP4C4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AP4C4R(const Npp8u *const pSrc[4], int nSrcStep,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AP4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep,
                                   Npp8u *pDst[4], int nDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToHLS_8u_AP4R(const Npp8u *const pSrc[4], int nSrcStep,
                               Npp8u *pDst[4], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3],
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep,
                                 Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_AP4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep,
                                   Npp8u *pDst[4], int nDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_AP4R(const Npp8u *const pSrc[4], int nSrcStep,
                               Npp8u *pDst[4], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep,
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_AP4C4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiHLSToBGR_8u_AP4C4R(const Npp8u *const pSrc[4], int nSrcStep,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToHSV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToHSV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToHSV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiRGBToHSV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHSVToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiHSVToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiHSVToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiHSVToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr420_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr420_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr444_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr444_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int aDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr444_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr444_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGB_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int aSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGB_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int aSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int aSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int aSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int aSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int aSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr444ToRGB_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr444ToRGB_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int aSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int aSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int aSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int aSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int aSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int aSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr444ToBGR_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int nSrcStep, Npp8u *pDst[3],
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr444ToBGR_JPEG_8u_P3R(const Npp8u *const pSrc[3],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr420_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int aDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr420_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int aDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int aDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr422_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int aDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int aDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr411_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int aDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYCbCr444_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int nDstStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToYCbCr444_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int nDstStep,
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int aDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr420_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int aDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int aDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr422_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int aDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int aDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr411_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int aDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiBGRToYCbCr444_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int nDstStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiBGRToYCbCr444_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int nDstStep,
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGB_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int aSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToRGB_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int aSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int aSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToRGB_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int aSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int aSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int aSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr444ToRGB_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int nSrcStep, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr444ToRGB_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int nSrcStep, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int aSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToBGR_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int aSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int aSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToBGR_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int aSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int aSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int aSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr444ToBGR_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                              int nSrcStep, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr444ToBGR_JPEG_8u_P3C3R(const Npp8u *const pSrc[3],
                                          int nSrcStep, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_8u_AC4C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_8u_AC4C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_16u_C3C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_16u_C3C1R(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_16u_AC4C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_16u_AC4C1R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_16s_C3C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_16s_C3C1R(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_16s_AC4C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_16s_AC4C1R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToGray_32f_AC4C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiRGBToGray_32f_AC4C1R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aCoeffs[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_8u_AC4C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_8u_AC4C1R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_8u_C4C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aCoeffs[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_8u_C4C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aCoeffs[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_16u_C3C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_16u_C3C1R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_16u_AC4C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aCoeffs[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_16u_AC4C1R(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_16u_C4C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_16u_C4C1R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_16s_C3C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_16s_C3C1R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_16s_AC4C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         Npp16s *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aCoeffs[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_16s_AC4C1R(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_16s_C4C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_16s_C4C1R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_32f_AC4C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aCoeffs[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_32f_AC4C1R(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aCoeffs[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorToGray_32f_C4C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aCoeffs[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aCoeffs SEND_ONLY DEREF
 */
NppStatus nppiColorToGray_32f_C4C1R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f aCoeffs[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGradientColorToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                               Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI,
                                               NppiNorm eNorm,
                                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 */
NppStatus nppiGradientColorToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep,
                                           Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppiNorm eNorm);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGradientColorToGray_16u_C3C1R_Ctx(const Npp16u *pSrc,
                                                int nSrcStep, Npp16u *pDst,
                                                int nDstStep, NppiSize oSizeROI,
                                                NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 */
NppStatus nppiGradientColorToGray_16u_C3C1R(const Npp16u *pSrc, int nSrcStep,
                                            Npp16u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGradientColorToGray_16s_C3C1R_Ctx(const Npp16s *pSrc,
                                                int nSrcStep, Npp16s *pDst,
                                                int nDstStep, NppiSize oSizeROI,
                                                NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 */
NppStatus nppiGradientColorToGray_16s_C3C1R(const Npp16s *pSrc, int nSrcStep,
                                            Npp16s *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGradientColorToGray_32f_C3C1R_Ctx(const Npp32f *pSrc,
                                                int nSrcStep, Npp32f *pDst,
                                                int nDstStep, NppiSize oSizeROI,
                                                NppiNorm eNorm,
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eNorm SEND_ONLY
 */
NppStatus nppiGradientColorToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep,
                                            Npp32f *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppiNorm eNorm);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCFAToRGB_8u_C1C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    NppiSize oSrcSize, NppiRect oSrcROI,
                                    Npp8u *pDst, int nDstStep,
                                    NppiBayerGridPosition eGrid,
                                    NppiInterpolationMode eInterpolation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiCFAToRGB_8u_C1C3R(const Npp8u *pSrc, int nSrcStep,
                                NppiSize oSrcSize, NppiRect oSrcROI,
                                Npp8u *pDst, int nDstStep,
                                NppiBayerGridPosition eGrid,
                                NppiInterpolationMode eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCFAToRGBA_8u_C1AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiRect oSrcROI,
                                      Npp8u *pDst, int nDstStep,
                                      NppiBayerGridPosition eGrid,
                                      NppiInterpolationMode eInterpolation,
                                      Npp8u nAlpha,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiCFAToRGBA_8u_C1AC4R(const Npp8u *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiRect oSrcROI,
                                  Npp8u *pDst, int nDstStep,
                                  NppiBayerGridPosition eGrid,
                                  NppiInterpolationMode eInterpolation,
                                  Npp8u nAlpha);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCFAToRGB_16u_C1C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep,
                                     NppiBayerGridPosition eGrid,
                                     NppiInterpolationMode eInterpolation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 */
NppStatus nppiCFAToRGB_16u_C1C3R(const Npp16u *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiRect oSrcROI,
                                 Npp16u *pDst, int nDstStep,
                                 NppiBayerGridPosition eGrid,
                                 NppiInterpolationMode eInterpolation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCFAToRGBA_16u_C1AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiRect oSrcROI,
                                       Npp16u *pDst, int nDstStep,
                                       NppiBayerGridPosition eGrid,
                                       NppiInterpolationMode eInterpolation,
                                       Npp16u nAlpha,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nAlpha SEND_ONLY
 */
NppStatus nppiCFAToRGBA_16u_C1AC4R(const Npp16u *pSrc, int nSrcStep,
                                   NppiSize oSrcSize, NppiRect oSrcROI,
                                   Npp16u *pDst, int nDstStep,
                                   NppiBayerGridPosition eGrid,
                                   NppiInterpolationMode eInterpolation,
                                   Npp16u nAlpha);
/**
 * @guard NPP_VERSION >= 12300
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCFAToRGB_32u_C1C3R_Ctx(const Npp32u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiRect oSrcROI,
                                     Npp32u *pDst, int nDstStep,
                                     NppiBayerGridPosition eGrid,
                                     NppiInterpolationMode eInterpolation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12300
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param eGrid SEND_ONLY
 * @param eInterpolation SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCFAToRGBA_32u_C1AC4R_Ctx(const Npp32u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiRect oSrcROI,
                                       Npp32u *pDst, int nDstStep,
                                       NppiBayerGridPosition eGrid,
                                       NppiInterpolationMode eInterpolation,
                                       Npp32u nAlpha,
                                       NppStreamContext nppStreamCtx);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr411_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr,
                                              int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr411_8u_P3P2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDstY,
                                          int nDstYStep, Npp8u *pDstCbCr,
                                          int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr411_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3],
                                              int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr411_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3],
                                          int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422_8u_C2P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3],
                                int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3],
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422_8u_P3C2R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCrCb422_8u_C2R(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToCbYCr422_8u_C2R(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr411_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr411_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr,
                                              int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDstY,
                                          int nDstYStep, Npp8u *pDstCbCr,
                                          int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_C2P2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCbCr, int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr420_8u_C2P2R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDstY, int nDstYStep,
                                          Npp8u *pDstCbCr, int nDstCbCrStep,
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr422_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int rDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr422_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr422_8u_P3C2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCrCb420_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCrCb420_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int rDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr,
                                              int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_P3P2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDstY,
                                          int nDstYStep, Npp8u *pDstCbCr,
                                          int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_C2P2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCbCr, int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr422ToYCbCr411_8u_C2P2R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDstY, int nDstYStep,
                                          Npp8u *pDstCbCr, int nDstCbCrStep,
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb422ToYCbCr422_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb422ToYCbCr420_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb422ToYCbCr411_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb422ToYCbCr411_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr422_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr422_8u_C2R(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr422_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr422_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr420_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr420_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr420_8u_C2P2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCbCr, int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCbCr420_8u_C2P2R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDstY, int nDstYStep,
                                          Npp8u *pDstCbCr, int nDstCbCrStep,
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCrCb420_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                              Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCbYCr422ToYCrCb420_8u_C2P3R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3],
                                    Npp8u *pDstY, int nDstYStep,
                                    Npp8u *pDstCbCr, int nDstCbCrStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                Npp8u *pDstY, int nDstYStep, Npp8u *pDstCbCr,
                                int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                    const Npp8u *pSrcCbCr, int nSrcCbCrStep,
                                    Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                const Npp8u *pSrcCbCr, int nSrcCbCrStep,
                                Npp8u *pDst[3], int rDstStep[3],
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr422_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr422_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr422_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3],
                                              int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr422_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3],
                                          int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr422_8u_P2C2R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCbCr422_8u_P2C2R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToCbYCr422_8u_P2C2R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToCbYCr422_8u_P2C2R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3],
                                              int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3],
                                          int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb420ToCbYCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb420ToCbYCr422_8u_P3C2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr,
                                              int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDstY,
                                          int nDstYStep, Npp8u *pDstCbCr,
                                          int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr411_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr,
                                              int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCrCb420ToYCbCr411_8u_P3P2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDstY,
                                          int nDstYStep, Npp8u *pDstCbCr,
                                          int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3],
                                    Npp8u *pDstY, int nDstYStep,
                                    Npp8u *pDstCbCr, int nDstCbCrStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3],
                                Npp8u *pDstY, int nDstYStep, Npp8u *pDstCbCr,
                                int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                    const Npp8u *pSrcCbCr, int nSrcCbCrStep,
                                    Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                const Npp8u *pSrcCbCr, int nSrcCbCrStep,
                                Npp8u *pDst[3], int rDstStep[3],
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P3C2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3],
                                              int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3],
                                          int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P2C2R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr422_8u_P2C2R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCrCb422_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCrCb422_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCrCb422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCrCb422_8u_P3C2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr420_8u_P3R_Ctx(const Npp8u *const pSrc[3],
                                            int rSrcStep[3], Npp8u *pDst[3],
                                            int nDstStep[3], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr420_8u_P3R(const Npp8u *const pSrc[3],
                                        int rSrcStep[3], Npp8u *pDst[3],
                                        int nDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                              int rSrcStep[3], Npp8u *pDstY,
                                              int nDstYStep, Npp8u *pDstCbCr,
                                              int nDstCbCrStep,
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param rSrcStep SEND_ONLY DEREF
 * @param pDstY SEND_ONLY
 * @param nDstYStep SEND_ONLY
 * @param pDstCbCr SEND_ONLY
 * @param nDstCbCrStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3],
                                          int rSrcStep[3], Npp8u *pDstY,
                                          int nDstYStep, Npp8u *pDstCbCr,
                                          int nDstCbCrStep, NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr420_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3],
                                              int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCbCr420_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3],
                                          int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep,
                                              const Npp8u *pSrcCbCr,
                                              int nSrcCbCrStep, Npp8u *pDst[3],
                                              int rDstStep[3],
                                              NppiSize oSizeROI,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcY SEND_ONLY
 * @param nSrcYStep SEND_ONLY
 * @param pSrcCbCr SEND_ONLY
 * @param nSrcCbCrStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param rDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiYCbCr411ToYCrCb420_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep,
                                          const Npp8u *pSrcCbCr,
                                          int nSrcCbCrStep, Npp8u *pDst[3],
                                          int rDstStep[3], NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToYUV420_8u_P2P3R_Ctx(const Npp8u *const pSrc[2],
                                        int nSrcStep, Npp8u *pDst[3],
                                        int aDstStep[3], NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiNV12ToYUV420_8u_P2P3R(const Npp8u *const pSrc[2], int nSrcStep,
                                    Npp8u *pDst[3], int aDstStep[3],
                                    NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiGammaFwd_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiGammaFwd_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaFwd_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaInv_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiGammaInv_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaInv_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaInv_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaInv_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiGammaInv_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaInv_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaInv_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaInv_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                  Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaInv_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                              Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGammaInv_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiGammaInv_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nColorKeyConst SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompColorKey_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nColorKeyConst,
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
 * @param nColorKeyConst SEND_ONLY
 */
NppStatus nppiCompColorKey_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nColorKeyConst);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nColorKeyConst SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompColorKey_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      Npp8u nColorKeyConst[3],
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
 * @param nColorKeyConst SEND_ONLY DEREF
 */
NppStatus nppiCompColorKey_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nColorKeyConst[3]);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nColorKeyConst SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompColorKey_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                      const Npp8u *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      Npp8u nColorKeyConst[4],
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
 * @param nColorKeyConst SEND_ONLY DEREF
 */
NppStatus nppiCompColorKey_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  Npp8u nColorKeyConst[4]);
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
 * @param nColorKeyConst SEND_ONLY DEREF
 * @param nppAlphaOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiAlphaCompColorKey_8u_AC4R_Ctx(
    const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1, const Npp8u *pSrc2,
    int nSrc2Step, Npp8u nAlpha2, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
    Npp8u nColorKeyConst[4], NppiAlphaOp nppAlphaOp,
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
 * @param nColorKeyConst SEND_ONLY DEREF
 * @param nppAlphaOp SEND_ONLY
 */
NppStatus
nppiAlphaCompColorKey_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step, Npp8u nAlpha1,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u nAlpha2,
                              Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                              Npp8u nColorKeyConst[4], NppiAlphaOp nppAlphaOp);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C2IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C2IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32fC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[4][4],
                                        const Npp32f aConstants[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32fC_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[4][4],
                                    const Npp32f aConstants[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32fC_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[4][4],
                                         const Npp32f aConstants[4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32fC_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[4][4],
                                     const Npp32f aConstants[4]);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                       Npp8u *const pDst[3], int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                                   Npp8u *const pDst[3], int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8u_IP3R_Ctx(Npp8u *const pSrcDst[3],
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C1R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C1IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C1IR(Npp8s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C2R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C2R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C2IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C2IR(Npp8s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C3R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C3R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C3IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C3IR(Npp8s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C4R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C4R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_C4IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_C4IR(Npp8s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_AC4R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                        Npp8s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_AC4R(const Npp8s *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_AC4IR_Ctx(Npp8s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_AC4IR(Npp8s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_P3R_Ctx(const Npp8s *const pSrc[3], int nSrcStep,
                                       Npp8s *const pDst[3], int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_P3R(const Npp8s *const pSrc[3], int nSrcStep,
                                   Npp8s *const pDst[3], int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_8s_IP3R_Ctx(Npp8s *const pSrcDst[3],
                                        int nSrcDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_8s_IP3R(Npp8s *const pSrcDst[3], int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_C2R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_C2R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_C2IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_C2IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp32f aTwist[3][4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_P3R_Ctx(const Npp16u *const pSrc[3],
                                        int nSrcStep, Npp16u *const pDst[3],
                                        int nDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_P3R(const Npp16u *const pSrc[3], int nSrcStep,
                                    Npp16u *const pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16u_IP3R_Ctx(Npp16u *const pSrcDst[3],
                                         int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16u_IP3R(Npp16u *const pSrcDst[3], int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_C2R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_C2R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_C2IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_C2IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         Npp16s *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp32f aTwist[3][4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_P3R_Ctx(const Npp16s *const pSrc[3],
                                        int nSrcStep, Npp16s *const pDst[3],
                                        int nDstStep, NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_P3R(const Npp16s *const pSrc[3], int nSrcStep,
                                    Npp16s *const pDst[3], int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16s_IP3R_Ctx(Npp16s *const pSrcDst[3],
                                         int nSrcDstStep, NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16s_IP3R(Npp16s *const pSrcDst[3], int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                        Npp16f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C1R(const Npp16f *pSrc, int nSrcStep,
                                    Npp16f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C1IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C1IR(Npp16f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C2R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                        Npp16f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C2R(const Npp16f *pSrc, int nSrcStep,
                                    Npp16f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C2IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C2IR(Npp16f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                        Npp16f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C3R(const Npp16f *pSrc, int nSrcStep,
                                    Npp16f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C3IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C3IR(Npp16f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                        Npp16f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f aTwist[3][4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C4R(const Npp16f *pSrc, int nSrcStep,
                                    Npp16f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32f_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[3][4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32f_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32fC_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                         Npp16f *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f aTwist[4][4],
                                         const Npp32f aConstants[4],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32fC_16f_C4R(const Npp16f *pSrc, int nSrcStep,
                                     Npp16f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[4][4],
                                     const Npp32f aConstants[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist32fC_16f_C4IR_Ctx(Npp16f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp32f aTwist[4][4],
                                          const Npp32f aConstants[4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 */
NppStatus nppiColorTwist32fC_16f_C4IR(Npp16f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[4][4],
                                      const Npp32f aConstants[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C2R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C2R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C2IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C2IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32fC_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[4][4],
                                      const Npp32f aConstants[4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32fC_C4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp32f aTwist[4][4],
                                  const Npp32f aConstants[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32fC_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f aTwist[4][4],
                                       const Npp32f aConstants[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param aConstants SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32fC_C4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, const Npp32f aTwist[4][4],
                                   const Npp32f aConstants[4]);
/**
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_P3R_Ctx(const Npp32f *const pSrc[3], int nSrcStep,
                                     Npp32f *const pDst[3], int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f aTwist[3][4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_P3R(const Npp32f *const pSrc[3], int nSrcStep,
                                 Npp32f *const pDst[3], int nDstStep,
                                 NppiSize oSizeROI, const Npp32f aTwist[3][4]);
/**
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwist_32f_IP3R_Ctx(Npp32f *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY DEREF
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 */
NppStatus nppiColorTwist_32f_IP3R(Npp32f *const pSrcDst[3], int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f aTwist[3][4]);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C1R_Ctx(Npp32f nMin, Npp32f nMax,
                                            NppiSize oSizeROI,
                                            NppiColorTwistBatchCXR *pBatchList,
                                            int nBatchSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C1R(Npp32f nMin, Npp32f nMax,
                                        NppiSize oSizeROI,
                                        NppiColorTwistBatchCXR *pBatchList,
                                        int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C1IR_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C1IR(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C3R_Ctx(Npp32f nMin, Npp32f nMax,
                                            NppiSize oSizeROI,
                                            NppiColorTwistBatchCXR *pBatchList,
                                            int nBatchSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C3R(Npp32f nMin, Npp32f nMax,
                                        NppiSize oSizeROI,
                                        NppiColorTwistBatchCXR *pBatchList,
                                        int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C3IR_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C3IR(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C4R_Ctx(Npp32f nMin, Npp32f nMax,
                                            NppiSize oSizeROI,
                                            NppiColorTwistBatchCXR *pBatchList,
                                            int nBatchSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C4R(Npp32f nMin, Npp32f nMax,
                                        NppiSize oSizeROI,
                                        NppiColorTwistBatchCXR *pBatchList,
                                        int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C4IR_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_C4IR(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_AC4R_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_AC4R(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiColorTwistBatch32f_8u_AC4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
                                    NppiColorTwistBatchCXR *pBatchList,
                                    int nBatchSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_8u_AC4IR(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32fC_8u_C4R_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32fC_8u_C4R(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiColorTwistBatch32fC_8u_C4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
                                    NppiColorTwistBatchCXR *pBatchList,
                                    int nBatchSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32fC_8u_C4IR(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C1R_Ctx(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C1R(Npp32f nMin, Npp32f nMax,
                                      NppiSize oSizeROI,
                                      NppiColorTwistBatchCXR *pBatchList,
                                      int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C1IR_Ctx(Npp32f nMin, Npp32f nMax,
                                           NppiSize oSizeROI,
                                           NppiColorTwistBatchCXR *pBatchList,
                                           int nBatchSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C1IR(Npp32f nMin, Npp32f nMax,
                                       NppiSize oSizeROI,
                                       NppiColorTwistBatchCXR *pBatchList,
                                       int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C3R_Ctx(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C3R(Npp32f nMin, Npp32f nMax,
                                      NppiSize oSizeROI,
                                      NppiColorTwistBatchCXR *pBatchList,
                                      int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C3IR_Ctx(Npp32f nMin, Npp32f nMax,
                                           NppiSize oSizeROI,
                                           NppiColorTwistBatchCXR *pBatchList,
                                           int nBatchSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C3IR(Npp32f nMin, Npp32f nMax,
                                       NppiSize oSizeROI,
                                       NppiColorTwistBatchCXR *pBatchList,
                                       int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C4R_Ctx(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C4R(Npp32f nMin, Npp32f nMax,
                                      NppiSize oSizeROI,
                                      NppiColorTwistBatchCXR *pBatchList,
                                      int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C4IR_Ctx(Npp32f nMin, Npp32f nMax,
                                           NppiSize oSizeROI,
                                           NppiColorTwistBatchCXR *pBatchList,
                                           int nBatchSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_C4IR(Npp32f nMin, Npp32f nMax,
                                       NppiSize oSizeROI,
                                       NppiColorTwistBatchCXR *pBatchList,
                                       int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_AC4R_Ctx(Npp32f nMin, Npp32f nMax,
                                           NppiSize oSizeROI,
                                           NppiColorTwistBatchCXR *pBatchList,
                                           int nBatchSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_AC4R(Npp32f nMin, Npp32f nMax,
                                       NppiSize oSizeROI,
                                       NppiColorTwistBatchCXR *pBatchList,
                                       int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_AC4IR_Ctx(Npp32f nMin, Npp32f nMax,
                                            NppiSize oSizeROI,
                                            NppiColorTwistBatchCXR *pBatchList,
                                            int nBatchSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32f_AC4IR(Npp32f nMin, Npp32f nMax,
                                        NppiSize oSizeROI,
                                        NppiColorTwistBatchCXR *pBatchList,
                                        int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32fC_C4R_Ctx(Npp32f nMin, Npp32f nMax,
                                           NppiSize oSizeROI,
                                           NppiColorTwistBatchCXR *pBatchList,
                                           int nBatchSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32fC_C4R(Npp32f nMin, Npp32f nMax,
                                       NppiSize oSizeROI,
                                       NppiColorTwistBatchCXR *pBatchList,
                                       int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32fC_C4IR_Ctx(Npp32f nMin, Npp32f nMax,
                                            NppiSize oSizeROI,
                                            NppiColorTwistBatchCXR *pBatchList,
                                            int nBatchSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch_32fC_C4IR(Npp32f nMin, Npp32f nMax,
                                        NppiSize oSizeROI,
                                        NppiColorTwistBatchCXR *pBatchList,
                                        int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C1R_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C1R(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiColorTwistBatch32f_16f_C1IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
                                    NppiColorTwistBatchCXR *pBatchList,
                                    int nBatchSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C1IR(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C3R_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C3R(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiColorTwistBatch32f_16f_C3IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
                                    NppiColorTwistBatchCXR *pBatchList,
                                    int nBatchSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C3IR(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C4R_Ctx(Npp32f nMin, Npp32f nMax,
                                             NppiSize oSizeROI,
                                             NppiColorTwistBatchCXR *pBatchList,
                                             int nBatchSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C4R(Npp32f nMin, Npp32f nMax,
                                         NppiSize oSizeROI,
                                         NppiColorTwistBatchCXR *pBatchList,
                                         int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiColorTwistBatch32f_16f_C4IR_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
                                    NppiColorTwistBatchCXR *pBatchList,
                                    int nBatchSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32f_16f_C4IR(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppiColorTwistBatch32fC_16f_C4R_Ctx(Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
                                    NppiColorTwistBatchCXR *pBatchList,
                                    int nBatchSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32fC_16f_C4R(Npp32f nMin, Npp32f nMax,
                                          NppiSize oSizeROI,
                                          NppiColorTwistBatchCXR *pBatchList,
                                          int nBatchSize);
/**
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiColorTwistBatch32fC_16f_C4IR_Ctx(
    Npp32f nMin, Npp32f nMax, NppiSize oSizeROI,
    NppiColorTwistBatchCXR *pBatchList, int nBatchSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pBatchList SEND_ONLY
 * @param nBatchSize SEND_ONLY
 */
NppStatus nppiColorTwistBatch32fC_16f_C4IR(Npp32f nMin, Npp32f nMax,
                                           NppiSize oSizeROI,
                                           NppiColorTwistBatchCXR *pBatchList,
                                           int nBatchSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp32s *pValues, const Npp32s *pLevels,
                             int nLevels, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI, const Npp32s *pValues,
                         const Npp32s *pLevels, int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, const Npp32s *pValues,
                              const Npp32s *pLevels, int nLevels,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues, const Npp32s *pLevels,
                          int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp32s *pValues[3], const Npp32s *pLevels[3],
                             int nLevels[3], NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI,
                         const Npp32s *pValues[3], const Npp32s *pLevels[3],
                         int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, const Npp32s *pValues[3],
                              const Npp32s *pLevels[3], int nLevels[3],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[3], const Npp32s *pLevels[3],
                          int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp32s *pValues[4], const Npp32s *pLevels[4],
                             int nLevels[4], NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                         int nDstStep, NppiSize oSizeROI,
                         const Npp32s *pValues[4], const Npp32s *pLevels[4],
                         int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                              NppiSize oSizeROI, const Npp32s *pValues[4],
                              const Npp32s *pLevels[4], int nLevels[4],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[4], const Npp32s *pLevels[4],
                          int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues[3],
                              const Npp32s *pLevels[3], int nLevels[3],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[3], const Npp32s *pLevels[3],
                          int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues[3],
                               const Npp32s *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[3], const Npp32s *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues, const Npp32s *pLevels,
                              int nLevels, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues, const Npp32s *pLevels,
                          int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues,
                               const Npp32s *pLevels, int nLevels,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues, const Npp32s *pLevels,
                           int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues[3],
                              const Npp32s *pLevels[3], int nLevels[3],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[3], const Npp32s *pLevels[3],
                          int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues[3],
                               const Npp32s *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[3], const Npp32s *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues[4],
                              const Npp32s *pLevels[4], int nLevels[4],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[4], const Npp32s *pLevels[4],
                          int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_C4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues[4],
                               const Npp32s *pLevels[4], int nLevels[4],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[4], const Npp32s *pLevels[4],
                           int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp32s *pValues[3],
                               const Npp32s *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[3], const Npp32s *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3],
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                            const Npp32s *pValues[3], const Npp32s *pLevels[3],
                            int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues, const Npp32s *pLevels,
                              int nLevels, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues, const Npp32s *pLevels,
                          int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues,
                               const Npp32s *pLevels, int nLevels,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues, const Npp32s *pLevels,
                           int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues[3],
                              const Npp32s *pLevels[3], int nLevels[3],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[3], const Npp32s *pLevels[3],
                          int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues[3],
                               const Npp32s *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[3], const Npp32s *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32s *pValues[4],
                              const Npp32s *pLevels[4], int nLevels[4],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32s *pValues[4], const Npp32s *pLevels[4],
                          int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32s *pValues[4],
                               const Npp32s *pLevels[4], int nLevels[4],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[4], const Npp32s *pLevels[4],
                           int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp32s *pValues[3],
                               const Npp32s *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                           int nDstStep, NppiSize oSizeROI,
                           const Npp32s *pValues[3], const Npp32s *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3],
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                            const Npp32s *pValues[3], const Npp32s *pLevels[3],
                            int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32f *pValues, const Npp32f *pLevels,
                              int nLevels, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32f *pValues, const Npp32f *pLevels,
                          int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32f *pValues,
                               const Npp32f *pLevels, int nLevels,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32f *pValues, const Npp32f *pLevels,
                           int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32f *pValues[3],
                              const Npp32f *pLevels[3], int nLevels[3],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32f *pValues[3], const Npp32f *pLevels[3],
                          int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32f *pValues[3],
                               const Npp32f *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32f *pValues[3], const Npp32f *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp32f *pValues[4],
                              const Npp32f *pLevels[4], int nLevels[4],
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                          int nDstStep, NppiSize oSizeROI,
                          const Npp32f *pValues[4], const Npp32f *pLevels[4],
                          int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                               NppiSize oSizeROI, const Npp32f *pValues[4],
                               const Npp32f *pLevels[4], int nLevels[4],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                           const Npp32f *pValues[4], const Npp32f *pLevels[4],
                           int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp32f *pValues[3],
                               const Npp32f *pLevels[3], int nLevels[3],
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI,
                           const Npp32f *pValues[3], const Npp32f *pLevels[3],
                           int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp32f *pValues[3],
                                const Npp32f *pLevels[3], int nLevels[3],
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                            const Npp32f *pValues[3], const Npp32f *pLevels[3],
                            int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues,
                                    const Npp32s *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues, const Npp32s *pLevels,
                                int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, const Npp32s *pValues,
                                     const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues,
                                 const Npp32s *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[3],
                                    const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[4],
                                    const Npp32s *pLevels[4], int nLevels[4],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[4],
                                const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[4],
                                     const Npp32s *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[4],
                                 const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32s *pValues,
                                     const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels,
                                 int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32s *pValues,
                                      const Npp32s *pLevels, int nLevels,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues,
                                  const Npp32s *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[4],
                                     const Npp32s *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[4],
                                 const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_C4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[4],
                                      const Npp32s *pLevels[4], int nLevels[4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[4],
                                  const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32s *pValues[3],
                                       const Npp32s *pLevels[3], int nLevels[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues[3],
                                   const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32s *pValues,
                                     const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues, const Npp32s *pLevels,
                                 int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32s *pValues,
                                      const Npp32s *pLevels, int nLevels,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues,
                                  const Npp32s *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[4],
                                     const Npp32s *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[4],
                                 const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[4],
                                      const Npp32s *pLevels[4], int nLevels[4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[4],
                                  const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32s *pValues[3],
                                       const Npp32s *pLevels[3], int nLevels[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, const Npp32s *pValues[3],
                                   const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp32f *pValues,
                                     const Npp32f *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f *pValues, const Npp32f *pLevels,
                                 int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f *pValues,
                                      const Npp32f *pLevels, int nLevels,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f *pValues,
                                  const Npp32f *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f *pValues[3],
                                     const Npp32f *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f *pValues[3],
                                 const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f *pValues[3],
                                      const Npp32f *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f *pValues[3],
                                  const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f *pValues[4],
                                     const Npp32f *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f *pValues[4],
                                 const Npp32f *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f *pValues[4],
                                      const Npp32f *pLevels[4], int nLevels[4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f *pValues[4],
                                  const Npp32f *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f *pValues[3],
                                      const Npp32f *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp32f *pValues[3],
                                  const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Linear_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f *pValues[3],
                                       const Npp32f *pLevels[3], int nLevels[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Linear_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, const Npp32f *pValues[3],
                                   const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32s *pValues, const Npp32s *pLevels,
                                   int nLevels, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp32s *pValues, const Npp32s *pLevels,
                               int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues,
                                    const Npp32s *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp32s *pValues,
                                const Npp32s *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32s *pValues[3],
                                   const Npp32s *pLevels[3], int nLevels[3],
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp32s *pValues[3],
                               const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[3],
                                    const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp32s *pValues[4],
                                   const Npp32s *pLevels[4], int nLevels[4],
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp32s *pValues[4],
                               const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[4],
                                    const Npp32s *pLevels[4], int nLevels[4],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp32s *pValues[4],
                                const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[3],
                                    const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues,
                                    const Npp32s *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues, const Npp32s *pLevels,
                                int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, const Npp32s *pValues,
                                     const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues,
                                 const Npp32s *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[3],
                                    const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[4],
                                    const Npp32s *pLevels[4], int nLevels[4],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[4],
                                const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_C4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[4],
                                     const Npp32s *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[4],
                                 const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues,
                                    const Npp32s *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues, const Npp32s *pLevels,
                                int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, const Npp32s *pValues,
                                     const Npp32s *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues,
                                 const Npp32s *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[3],
                                    const Npp32s *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[3],
                                const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32s *pValues[4],
                                    const Npp32s *pLevels[4], int nLevels[4],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32s *pValues[4],
                                const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[4],
                                     const Npp32s *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32s *pValues[4],
                                 const Npp32s *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32s *pValues[3],
                                     const Npp32s *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32s *pValues[3],
                                 const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32s *pValues[3],
                                      const Npp32s *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32s *pValues[3],
                                  const Npp32s *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f *pValues,
                                    const Npp32f *pLevels, int nLevels,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32f *pValues, const Npp32f *pLevels,
                                int nLevels);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, const Npp32f *pValues,
                                     const Npp32f *pLevels, int nLevels,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY
 * @param nLevels SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32f *pValues,
                                 const Npp32f *pLevels, int nLevels);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f *pValues[3],
                                    const Npp32f *pLevels[3], int nLevels[3],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32f *pValues[3],
                                const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f *pValues[3],
                                     const Npp32f *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32f *pValues[3],
                                 const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32f *pValues[4],
                                    const Npp32f *pLevels[4], int nLevels[4],
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32f *pValues[4],
                                const Npp32f *pLevels[4], int nLevels[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f *pValues[4],
                                     const Npp32f *pLevels[4], int nLevels[4],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32f *pValues[4],
                                 const Npp32f *pLevels[4], int nLevels[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f *pValues[3],
                                     const Npp32f *pLevels[3], int nLevels[3],
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f *pValues[3],
                                 const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Cubic_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f *pValues[3],
                                      const Npp32f *pLevels[3], int nLevels[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY DEREF
 * @param pLevels SEND_ONLY DEREF
 * @param nLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Cubic_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI, const Npp32f *pValues[3],
                                  const Npp32f *pLevels[3], int nLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY DEREF
 * @param aLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Trilinear_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp32u *pValues,
                                       Npp8u *pLevels[3], int aLevels[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY DEREF
 * @param aLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Trilinear_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   Npp32u *pValues, Npp8u *pLevels[3],
                                   int aLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY DEREF
 * @param aLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Trilinear_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp32u *pValues,
                                        Npp8u *pLevels[3], int aLevels[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY DEREF
 * @param aLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Trilinear_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp32u *pValues,
                                    Npp8u *pLevels[3], int aLevels[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY DEREF
 * @param aLevels SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUT_Trilinear_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, Npp32u *pValues,
                                         Npp8u *pLevels[3], int aLevels[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pValues SEND_ONLY
 * @param pLevels SEND_ONLY DEREF
 * @param aLevels SEND_ONLY DEREF
 */
NppStatus nppiLUT_Trilinear_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp32u *pValues,
                                     Npp8u *pLevels[3], int aLevels[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u *pTable,
                                    int nBitSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pTable, int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_8u24u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pTable,
                                       int nBitSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_8u24u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp8u *pTable, int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_8u32u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp32u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp32u *pTable,
                                       int nBitSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_8u32u_C1R(const Npp8u *pSrc, int nSrcStep,
                                   Npp32u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp32u *pTable,
                                   int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u *pTables[3],
                                    int nBitSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pTables[3], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u *pTables[4],
                                    int nBitSize,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pTables[4], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pTables[3],
                                     int nBitSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pTables[3], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp16u *pTable,
                                     int nBitSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp16u *pTable, int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u8u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pTable,
                                       int nBitSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u8u_C1R(const Npp16u *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp8u *pTable, int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u24u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pTable,
                                        int nBitSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u24u_C1R(const Npp16u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u *pTable,
                                    int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u32u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp32u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp32u *pTable,
                                        int nBitSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTable SEND_ONLY
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u32u_C1R(const Npp16u *pSrc, int nSrcStep,
                                    Npp32u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp32u *pTable,
                                    int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u *pTables[3], int nBitSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp16u *pTables[3], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u *pTables[4], int nBitSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp16u *pTables[4], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16u *pTables[3], int nBitSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPalette_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp16u *pTables[3], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nAlphaValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPaletteSwap_8u_C3A0C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                            int nAlphaValue, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            const Npp8u *pTables[3],
                                            int nBitSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nAlphaValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPaletteSwap_8u_C3A0C4R(const Npp8u *pSrc, int nSrcStep,
                                        int nAlphaValue, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        const Npp8u *pTables[3], int nBitSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nAlphaValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiLUTPaletteSwap_16u_C3A0C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                             int nAlphaValue, Npp16u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             const Npp16u *pTables[3],
                                             int nBitSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nAlphaValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pTables SEND_ONLY DEREF
 * @param nBitSize SEND_ONLY
 */
NppStatus nppiLUTPaletteSwap_16u_C3A0C4R(const Npp16u *pSrc, int nSrcStep,
                                         int nAlphaValue, Npp16u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp16u *pTables[3],
                                         int nBitSize);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV420_8u_ColorTwist32f_P3R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV420_16u_ColorTwist32f_P3R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV420_8u_ColorTwist32f_C3P3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int aDstStep[3],
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV420_16u_ColorTwist32f_C3P3R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[3], int aDstStep[3],
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_ColorTwist32f_P3R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_16u_ColorTwist32f_P3R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_ColorTwist32f_C3C2R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_16u_ColorTwist32f_C3C2R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_8u_ColorTwist32f_C3P3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int aDstStep[3],
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToYUV422_16u_ColorTwist32f_C3P3R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[3], int aDstStep[3],
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[2], int aDstStep[2],
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToNV12_16u_ColorTwist32f_C3P2R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst[2], int aDstStep[2],
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToNV12_8u_ColorTwist32f_P3P2R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst[2],
    int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToNV12_16u_ColorTwist32f_P3P2R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst[2],
    int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3C3R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3C4R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4], Npp8u nAlpha,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4], Npp16u nAlpha,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_C2C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_C2C3R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_P3R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param aDstStep SEND_ONLY DEREF
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_P3R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst[3],
    int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_P3C3R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_P3C3R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(
    const Npp8u *const pSrc[3], int aSrcStep[3], Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4], Npp8u nAlpha,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV422ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(
    const Npp16u *const pSrc[3], int aSrcStep[3], Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4], Npp16u nAlpha,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
    const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aTwist SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(
    const Npp16u *const pSrc[2], int aSrcStep[2], Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f aTwist[3][4],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiPoint oSrcOffset, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_8u_ToUYVP_10u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                          int nSrcStep, NppiPoint oSrcOffset,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_8u_ToUYVP_10u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiPoint oSrcOffset, Npp8u *pDst[3],
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBA_8u_ToUYVP_10u_AC4C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                            NppiPoint oSrcOffset, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_16u_ToUYVP_10u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_16u_ToUYVP_10u_P3C3R_Ctx(const Npp16u *const pSrc[3],
                                           int nSrcStep, NppiPoint oSrcOffset,
                                           Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBA_16u_ToUYVP_10u_AC4C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                             NppiPoint oSrcOffset, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_16f_ToUYVP_10u_C3C3R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_16f_ToUYVP_10u_P3C3R_Ctx(const Npp16f *const pSrc[3],
                                           int nSrcStep, NppiPoint oSrcOffset,
                                           Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBA_16f_ToUYVP_10u_AC4C3R_Ctx(const Npp16f *pSrc, int nSrcStep,
                                             NppiPoint oSrcOffset, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param srcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param dstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_8u_C3C3R_Ctx(const Npp8u *pSrc, int srcStep,
                                          NppiPoint oSrcOffset, Npp8u *pDst,
                                          int dstStep, NppiSize oSizeROI,
                                          NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param alpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_8u_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           Npp8u alpha,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiPoint oSrcOffset, Npp8u *pDst[3],
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param srcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param dstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_16u_C3C3R_Ctx(const Npp8u *pSrc, int srcStep,
                                           NppiPoint oSrcOffset, Npp16u *pDst,
                                           int dstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13003
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param alpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_16u_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                            NppiPoint oSrcOffset, Npp16u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            NppiColorSpace eColorSpace,
                                            Npp16u alpha,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_16u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset,
                                           Npp16u *pDst[3], int nDstStep,
                                           NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_16f_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset, Npp16f *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlphaValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGBA_16f_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                             NppiPoint oSrcOffset, Npp16f *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             Npp32f nAlphaValue,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVP_10u_ToRGB_16f_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset,
                                           Npp16f *const pDst[3], int nDstStep,
                                           NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_8u_ToUYVY_8u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiColorSpace eColorSpace,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_8u_ToUYVY_8u_P3C3R_Ctx(const Npp8u *const pSrc[3],
                                         int nSrcStep, NppiPoint oSrcOffset,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         NppiColorSpace eColorSpace,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBA_8u_ToUYVY_8u_AC4C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                           NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_16u_ToUYVY16_16u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                             NppiPoint oSrcOffset, Npp16u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGB_16u_ToUYVY16_16u_P3C3R_Ctx(const Npp16u *const pSrc[3],
                                             int nSrcStep, NppiPoint oSrcOffset,
                                             Npp16u *pDst, int nDstStep,
                                             NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBA_16u_ToUYVY16_16u_AC4C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                               NppiPoint oSrcOffset,
                                               Npp16u *pDst, int nDstStep,
                                               NppiSize oSizeROI,
                                               NppiColorSpace eColorSpace,
                                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVY_8u_ToRGB_8u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiColorSpace eColorSpace,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVY_8u_ToRGB_8u_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiPoint oSrcOffset, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiColorSpace eColorSpace,
                                          Npp8u nAlpha,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVY_8u_ToRGB_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiPoint oSrcOffset, Npp8u *pDst[3],
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiColorSpace eColorSpace,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVY16_16u_ToRGB_16u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                             NppiPoint oSrcOffset, Npp16u *pDst,
                                             int nDstStep, NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVY16_16u_ToRGB_16u_C3AC4R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst,
    int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u nAlpha,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiUYVY16_16u_ToRGB_16u_C3P3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                             NppiPoint oSrcOffset,
                                             Npp16u *pDst[3], int nDstStep,
                                             NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_8u_P2C4R_Ctx(const Npp8u *const pSrc[2], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace, Npp8u nAlpha,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToBGR_8u_P2C4R_Ctx(const Npp8u *const pSrc[2], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace, Npp8u nAlpha,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToRGB_BT2020_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                            int nSrcStep, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx(const Npp8u *const pSrc[2],
                                            int nSrcStep, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiP010ToRGB_8u_P2C3R_Ctx(const Npp16u *const pSrc[2], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiP010ToRGB_8u_P2C4R_Ctx(const Npp16u *const pSrc[2], int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace, Npp8u nAlpha,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiP010ToRGB_16u_P2C3R_Ctx(const Npp16u *const pSrc[2], int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiColorSpace eColorSpace,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiP010ToRGB_16u_P2C4R_Ctx(const Npp16u *const pSrc[2], int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiColorSpace eColorSpace, Npp16u nAlpha,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiP016ToRGB_16u_P2C3R_Ctx(const Npp16u *const pSrc[2], int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiColorSpace eColorSpace,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nAlpha SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiP016ToRGB_16u_P2C4R_Ctx(const Npp16u *const pSrc[2], int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiColorSpace eColorSpace, Npp16u nAlpha,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToNV12_8u_C3P2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst[2], int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eColorSpace SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiRGBToNV12_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                     Npp8u *pDst[2], int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION >= 13200
 * @param pSrc SEND_ONLY DEREF
 * @param aSrcStep SEND_ONLY DEREF
 * @param pDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiYUV420ToNV12_8u_P3P2R_Ctx(const Npp8u *const pSrc[3],
                                        const int aSrcStep[3], Npp8u *pDst[2],
                                        int nDstStep, NppiSize oSizeROI,
                                        NppStreamContext nppStreamCtx);
