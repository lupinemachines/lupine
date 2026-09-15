#include <npp.h>

// NPP image morphological operations (libnppim).
//
// Images, masks and scratch buffers are device memory and travel as addresses;
// scratch buffer sizes come back from host memory.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppim by name at runtime. A call
// with a stream context goes to the server that owns the context's stream, or
// to the runtime's current device when the context names the default stream;
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
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                                Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize,
                                NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                            Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize,
                            NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                                Npp32s nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize,
                                NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                            Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize,
                            NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize,
                                NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            NppiSize oMaskSize, NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize,
                                 NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                 Npp16u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, const Npp8u *pMask,
                                 NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst,
                             Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                 Npp16u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, const Npp8u *pMask,
                                 NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst,
                             Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize,
                                 NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp8u *pMask, NppiSize oMaskSize,
                                  NppiPoint oAnchor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp8u *pMask, NppiSize oMaskSize,
                              NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                 Npp32f *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, const Npp8u *pMask,
                                 NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst,
                             Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                 Npp32f *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, const Npp8u *pMask,
                                 NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst,
                             Npp32s nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize,
                                 NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp8u *pMask, NppiSize oMaskSize,
                                  NppiPoint oAnchor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiDilate_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              const Npp8u *pMask, NppiSize oMaskSize,
                              NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp8u *pDst, Npp32s nDstStep,
                                  NppiSize oSizeROI, const Npp8u *pMask,
                                  NppiSize oMaskSize, NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp8u *pDst, Npp32s nDstStep,
                                  NppiSize oSizeROI, const Npp8u *pMask,
                                  NppiSize oMaskSize, NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp8u *pMask, NppiSize oMaskSize,
                                  NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp8u *pMask, NppiSize oMaskSize,
                                   NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16u *pDst, Npp32s nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp16u *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16u *pDst, Npp32s nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp16u *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        NppiBorderType eBorderType,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                    NppiSize oSrcSize, NppiPoint oSrcOffset,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u *pMask,
                                    NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, Npp32s nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp32f *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, Npp32s nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp32f *pDst, Npp32s nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        NppiBorderType eBorderType,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilateBorder_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                    NppiSize oSrcSize, NppiPoint oSrcOffset,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const Npp8u *pMask,
                                    NppiSize oMaskSize, NppiPoint oAnchor,
                                    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGrayDilateBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiGrayDilateBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp32s *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGrayDilateBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, Npp32s nSrcStep, NppiSize oSrcSize,
    NppiPoint oSrcOffset, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
    const Npp32f *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiGrayDilateBorder_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, Npp32s nDstStep,
                                       NppiSize oSizeROI, const Npp32f *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                   Npp8u *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                               Npp32s nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                   Npp8u *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                               Npp32s nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiDilate3x3_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiDilate3x3_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                    Npp16u *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep,
                                Npp16u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                    Npp16u *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep,
                                Npp16u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiDilate3x3_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiDilate3x3_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                    Npp32f *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                Npp32f *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                    Npp32f *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep,
                                Npp32f *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
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
NppStatus nppiDilate3x3_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
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
NppStatus nppiDilate3x3_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3_64f_C1R_Ctx(const Npp64f *pSrc, Npp32s nSrcStep,
                                    Npp64f *pDst, Npp32s nDstStep,
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
NppStatus nppiDilate3x3_64f_C1R(const Npp64f *pSrc, Npp32s nSrcStep,
                                Npp64f *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp16u *pDst,
                                          Npp32s nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp16u *pDst,
                                          Npp32s nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp16u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           NppiSize oSrcSize,
                                           NppiPoint oSrcOffset, Npp16u *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiBorderType eBorderType,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp32f *pDst,
                                          Npp32s nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp32f *pDst,
                                          Npp32s nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp32f *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           NppiSize oSrcSize,
                                           NppiPoint oSrcOffset, Npp32f *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           NppiBorderType eBorderType,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiDilate3x3Border_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                               Npp32s nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize,
                               NppiPoint oAnchor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                           Npp32s nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize,
                           NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                               Npp32s nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize,
                               NppiPoint oAnchor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                           Npp32s nDstStep, NppiSize oSizeROI,
                           const Npp8u *pMask, NppiSize oMaskSize,
                           NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, NppiSize oMaskSize,
                               NppiPoint oAnchor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                           NppiSize oMaskSize, NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize,
                                NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            NppiSize oMaskSize, NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                Npp16u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, const Npp8u *pMask,
                                NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst,
                            Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize,
                            NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                Npp16u *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, const Npp8u *pMask,
                                NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep, Npp16u *pDst,
                            Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize,
                            NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize,
                                NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            NppiSize oMaskSize, NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize,
                                 NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                Npp32f *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, const Npp8u *pMask,
                                NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst,
                            Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize,
                            NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                Npp32f *pDst, Npp32s nDstStep,
                                NppiSize oSizeROI, const Npp8u *pMask,
                                NppiSize oMaskSize, NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst,
                            Npp32s nDstStep, NppiSize oSizeROI,
                            const Npp8u *pMask, NppiSize oMaskSize,
                            NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, NppiSize oMaskSize,
                                NppiPoint oAnchor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            NppiSize oMaskSize, NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize,
                                 NppiPoint oAnchor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 */
NppStatus nppiErode_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, NppiSize oMaskSize,
                             NppiPoint oAnchor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppiBorderType eBorderType,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep,
                                 NppiSize oSrcSize, NppiPoint oSrcOffset,
                                 Npp8u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, const Npp8u *pMask,
                                 NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppiBorderType eBorderType,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep,
                                 NppiSize oSrcSize, NppiPoint oSrcOffset,
                                 Npp8u *pDst, Npp32s nDstStep,
                                 NppiSize oSizeROI, const Npp8u *pMask,
                                 NppiSize oMaskSize, NppiPoint oAnchor,
                                 NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppiBorderType eBorderType,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                 NppiSize oSrcSize, NppiPoint oSrcOffset,
                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, NppiSize oMaskSize,
                                 NppiPoint oAnchor, NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp8u *pMask, NppiSize oMaskSize,
                                  NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp16u *pDst, Npp32s nDstStep,
                                  NppiSize oSizeROI, const Npp8u *pMask,
                                  NppiSize oMaskSize, NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp16u *pDst, Npp32s nDstStep,
                                  NppiSize oSizeROI, const Npp8u *pMask,
                                  NppiSize oMaskSize, NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp8u *pMask, NppiSize oMaskSize,
                                  NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp32f *pDst, Npp32s nDstStep,
                                  NppiSize oSizeROI, const Npp8u *pMask,
                                  NppiSize oMaskSize, NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp32f *pDst, Npp32s nDstStep,
                                  NppiSize oSizeROI, const Npp8u *pMask,
                                  NppiSize oMaskSize, NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                  NppiSize oSrcSize, NppiPoint oSrcOffset,
                                  Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                  const Npp8u *pMask, NppiSize oMaskSize,
                                  NppiPoint oAnchor,
                                  NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       NppiBorderType eBorderType,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErodeBorder_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                   NppiSize oSrcSize, NppiPoint oSrcOffset,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, const Npp8u *pMask,
                                   NppiSize oMaskSize, NppiPoint oAnchor,
                                   NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGrayErodeBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiGrayErodeBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, const Npp32s *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiGrayErodeBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, Npp32s nSrcStep, NppiSize oSrcSize,
    NppiPoint oSrcOffset, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
    const Npp32f *pMask, NppiSize oMaskSize, NppiPoint oAnchor,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiGrayErodeBorder_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, const Npp32f *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                  Npp8u *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                              Npp32s nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                  Npp8u *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst,
                              Npp32s nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiErode3x3_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiErode3x3_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                   Npp16u *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep,
                               Npp16u *pDst, Npp32s nDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                   Npp16u *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep,
                               Npp16u *pDst, Npp32s nDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiErode3x3_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiErode3x3_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                   Npp32f *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                               Npp32f *pDst, Npp32s nDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                   Npp32f *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep,
                               Npp32f *pDst, Npp32s nDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
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
NppStatus nppiErode3x3_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
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
NppStatus nppiErode3x3_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3_64f_C1R_Ctx(const Npp64f *pSrc, Npp32s nSrcStep,
                                   Npp64f *pDst, Npp32s nDstStep,
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
NppStatus nppiErode3x3_64f_C1R(const Npp64f *pSrc, Npp32s nSrcStep,
                               Npp64f *pDst, Npp32s nDstStep,
                               NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI,
                                        NppiBorderType eBorderType,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep,
                                    NppiSize oSrcSize, NppiPoint oSrcOffset,
                                    Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI,
                                    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI,
                                        NppiBorderType eBorderType,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep,
                                    NppiSize oSrcSize, NppiPoint oSrcOffset,
                                    Npp8u *pDst, Npp32s nDstStep,
                                    NppiSize oSizeROI,
                                    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppiBorderType eBorderType,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                    NppiSize oSrcSize, NppiPoint oSrcOffset,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_C1R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp16u *pDst,
                                         Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_C1R(const Npp16u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp16u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_C3R_Ctx(const Npp16u *pSrc, Npp32s nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp16u *pDst,
                                         Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_C3R(const Npp16u *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp16u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp16u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp16u *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp32f *pDst,
                                         Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp32f *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_C3R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp32f *pDst,
                                         Npp32s nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_C3R(const Npp32f *pSrc, Npp32s nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp32f *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp32f *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         NppiBorderType eBorderType,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSize,
                                          NppiPoint oSrcOffset, Npp32f *pDst,
                                          int nDstStep, NppiSize oSizeROI,
                                          NppiBorderType eBorderType,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiErode3x3Border_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppiBorderType eBorderType);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_8u_C1R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_8u_C3R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_8u_C4R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_16u_C1R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_16s_C1R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_32f_C1R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_32f_C3R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param oSizeROI SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppiMorphGetBufferSize_32f_C4R(NppiSize oSizeROI, int *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_8u_C4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_32f_C4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphCloseBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     Npp8u *pBuffer,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     Npp8u *pBuffer,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_8u_C4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSize, NppiPoint oSrcOffset,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const Npp8u *pMask,
                                     NppiSize oMaskSize, NppiPoint oAnchor,
                                     Npp8u *pBuffer,
                                     NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_32f_C4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphOpenBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSize, NppiPoint oSrcOffset,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, const Npp8u *pMask,
                                      NppiSize oMaskSize, NppiPoint oAnchor,
                                      Npp8u *pBuffer,
                                      NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_8u_C4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                       NppiSize oSrcSize, NppiPoint oSrcOffset,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, const Npp8u *pMask,
                                       NppiSize oMaskSize, NppiPoint oAnchor,
                                       Npp8u *pBuffer,
                                       NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        Npp8u *pBuffer,
                                        NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        Npp8u *pBuffer,
                                        NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        Npp8u *pBuffer,
                                        NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        Npp8u *pBuffer,
                                        NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_32f_C4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphTopHatBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                        NppiSize oSrcSize, NppiPoint oSrcOffset,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, const Npp8u *pMask,
                                        NppiSize oMaskSize, NppiPoint oAnchor,
                                        Npp8u *pBuffer,
                                        NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp8u *pMask, NppiSize oMaskSize,
                                         NppiPoint oAnchor, Npp8u *pBuffer,
                                         NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp8u *pMask, NppiSize oMaskSize,
                                         NppiPoint oAnchor, Npp8u *pBuffer,
                                         NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_8u_C4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp8u *pMask, NppiSize oMaskSize,
                                         NppiPoint oAnchor, Npp8u *pBuffer,
                                         NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_16u_C1R(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_16s_C1R(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_32f_C1R(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_32f_C3R(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_32f_C4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphBlackHatBorder_32f_C4R(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_8u_C1R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp8u *pMask, NppiSize oMaskSize,
                                         NppiPoint oAnchor, Npp8u *pBuffer,
                                         NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp8u *pMask, NppiSize oMaskSize,
                                         NppiPoint oAnchor, Npp8u *pBuffer,
                                         NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_8u_C4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp8u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSize,
                                         NppiPoint oSrcOffset, Npp8u *pDst,
                                         int nDstStep, NppiSize oSizeROI,
                                         const Npp8u *pMask, NppiSize oMaskSize,
                                         NppiPoint oAnchor, Npp8u *pBuffer,
                                         NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_16u_C1R(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16u *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_16s_C1R(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp16s *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_32f_C1R(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_32f_C3R(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_32f_C4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSize SEND_ONLY
 * @param oSrcOffset SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param oMaskSize SEND_ONLY
 * @param oAnchor SEND_ONLY
 * @param pBuffer SEND_ONLY
 * @param eBorderType SEND_ONLY
 */
NppStatus nppiMorphGradientBorder_32f_C4R(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset,
    Npp32f *pDst, int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
    NppiSize oMaskSize, NppiPoint oAnchor, Npp8u *pBuffer,
    NppiBorderType eBorderType);
