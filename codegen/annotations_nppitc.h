#include <npp.h>

// NPP image threshold and compare operations (libnppitc).
//
// Images and device thresholds are device memory and travel as addresses. C
// arrays of thresholds and replacement values are host memory and travel by
// contents, as do nppiFusedAbsDiff_Threshold_GTVal's threshold and value: one
// element of eSrcDstType for each channel it thresholds.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppitc by name at runtime. A
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
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   Npp8u nThreshold,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               Npp8u nThreshold, NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp8u nThreshold,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, Npp8u nThreshold,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp16u nThreshold,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                Npp16u nThreshold,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp16u nThreshold,
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, Npp16u nThreshold,
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp16s nThreshold,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                Npp16s nThreshold,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp16s nThreshold,
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, Npp16s nThreshold,
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp32f nThreshold,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                Npp32f nThreshold,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp32f nThreshold,
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, Npp32f nThreshold,
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp8u rThresholds[3],
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u rThresholds[3],
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp8u rThresholds[3],
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                NppiSize oSizeROI, const Npp8u rThresholds[3],
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16u rThresholds[3],
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp16u rThresholds[3],
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp16u rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16s rThresholds[3],
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp16s rThresholds[3],
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16s rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp16s rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f rThresholds[3],
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp32f rThresholds[3],
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp32f rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp8u rThresholds[3],
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u rThresholds[3],
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp8u rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                 NppiSize oSizeROI, const Npp8u rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp16u rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16u rThresholds[3],
                                      NppCmpOp eComparisonOperation,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI,
                                  const Npp16u rThresholds[3],
                                  NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16s rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp16s rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16s rThresholds[3],
                                      NppCmpOp eComparisonOperation,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI,
                                  const Npp16s rThresholds[3],
                                  NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f rThresholds[3],
                                     NppCmpOp eComparisonOperation,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp32f rThresholds[3],
                                 NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f rThresholds[3],
                                      NppCmpOp eComparisonOperation,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                  NppiSize oSizeROI,
                                  const Npp32f rThresholds[3],
                                  NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nThreshold,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  Npp8u nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp8u nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, Npp8u nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp16u nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp16u nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp16u nThreshold,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp16u nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp16s nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp16s nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp16s nThreshold,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp16s nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp32f nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp32f nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp32f nThreshold,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp32f nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp8u rThresholds[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  const Npp8u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp8u rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   const Npp8u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16u rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp16u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16u rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16s rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp16s rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16s rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16s rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp32f rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp8u rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp8u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp8u rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp8u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16u rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp16u rThresholds[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16s rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16s rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp16s rThresholds[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16s rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GT_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f rThresholds[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GT_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nThreshold,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  Npp8u nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp8u nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, Npp8u nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp16u nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp16u nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp16u nThreshold,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp16u nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp16s nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp16s nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp16s nThreshold,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp16s nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp32f nThreshold,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp32f nThreshold);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp32f nThreshold,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp32f nThreshold);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp8u rThresholds[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  const Npp8u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp8u rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI,
                                   const Npp8u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16u rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp16u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16u rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16s rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp16s rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16s rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16s rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   const Npp32f rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp8u rThresholds[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp8u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp8u rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp8u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16u rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16u rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp16u rThresholds[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16s rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16s rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp16s rThresholds[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16s rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f rThresholds[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f rThresholds[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LT_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f rThresholds[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LT_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f rThresholds[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp8u nThreshold,
                                       Npp8u nValue,
                                       NppCmpOp eComparisonOperation,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   Npp8u nThreshold, Npp8u nValue,
                                   NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI, Npp8u nThreshold,
                                        Npp8u nValue,
                                        NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, Npp8u nThreshold,
                                    Npp8u nValue,
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp16u nThreshold,
                                        Npp16u nValue,
                                        NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp16u nThreshold,
                                    Npp16u nValue,
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, Npp16u nThreshold,
                                         Npp16u nValue,
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp16u nThreshold,
                                     Npp16u nValue,
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp16s nThreshold,
                                        Npp16s nValue,
                                        NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp16s nThreshold,
                                    Npp16s nValue,
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, Npp16s nThreshold,
                                         Npp16s nValue,
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp16s nThreshold,
                                     Npp16s nValue,
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, Npp32f nThreshold,
                                        Npp32f nValue,
                                        NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, Npp32f nThreshold,
                                    Npp32f nValue,
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI, Npp32f nThreshold,
                                         Npp32f nValue,
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI, Npp32f nThreshold,
                                     Npp32f nValue,
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp8u rThresholds[3], const Npp8u rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const Npp8u rThresholds[3],
                                   const Npp8u rValues[3],
                                   NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp8u rThresholds[3],
                                        const Npp8u rValues[3],
                                        NppCmpOp eComparisonOperation,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI,
                                    const Npp8u rThresholds[3],
                                    const Npp8u rValues[3],
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C3R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16u rThresholds[3], const Npp16u rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16u rThresholds[3],
                                    const Npp16u rValues[3],
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp16u rThresholds[3],
                                         const Npp16u rValues[3],
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u rThresholds[3],
                                     const Npp16u rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C3R_Ctx(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16s rThresholds[3], const Npp16s rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp16s rThresholds[3],
                                    const Npp16s rValues[3],
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp16s rThresholds[3],
                                         const Npp16s rValues[3],
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16s rThresholds[3],
                                     const Npp16s rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f rThresholds[3], const Npp32f rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp32f rThresholds[3],
                                    const Npp32f rValues[3],
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp32f rThresholds[3],
                                         const Npp32f rValues[3],
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f rThresholds[3],
                                     const Npp32f rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_AC4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp8u rThresholds[3], const Npp8u rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI,
                                    const Npp8u rThresholds[3],
                                    const Npp8u rValues[3],
                                    NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                         NppiSize oSizeROI,
                                         const Npp8u rThresholds[3],
                                         const Npp8u rValues[3],
                                         NppCmpOp eComparisonOperation,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                     NppiSize oSizeROI,
                                     const Npp8u rThresholds[3],
                                     const Npp8u rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_AC4R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16u rThresholds[3], const Npp16u rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16u rThresholds[3],
                                     const Npp16u rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp16u rThresholds[3],
                                          const Npp16u rValues[3],
                                          NppCmpOp eComparisonOperation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16u rThresholds[3],
                                      const Npp16u rValues[3],
                                      NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_AC4R_Ctx(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16s rThresholds[3], const Npp16s rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp16s rThresholds[3],
                                     const Npp16s rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp16s rThresholds[3],
                                          const Npp16s rValues[3],
                                          NppCmpOp eComparisonOperation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16s rThresholds[3],
                                      const Npp16s rValues[3],
                                      NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_AC4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f rThresholds[3], const Npp32f rValues[3],
    NppCmpOp eComparisonOperation, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp32f rThresholds[3],
                                     const Npp32f rValues[3],
                                     NppCmpOp eComparisonOperation);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp32f rThresholds[3],
                                          const Npp32f rValues[3],
                                          NppCmpOp eComparisonOperation,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiThreshold_Val_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f rThresholds[3],
                                      const Npp32f rValues[3],
                                      NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nThreshold,
                                         Npp8u nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nThreshold,
                                     Npp8u nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, Npp8u nThreshold,
                                          Npp8u nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, Npp8u nThreshold,
                                      Npp8u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp16u nThreshold,
                                          Npp16u nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp16u nThreshold,
                                      Npp16u nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp16u nThreshold,
                                           Npp16u nValue,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp16u nThreshold,
                                       Npp16u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          Npp16s *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp16s nThreshold,
                                          Npp16s nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp16s nThreshold,
                                      Npp16s nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp16s nThreshold,
                                           Npp16s nValue,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp16s nThreshold,
                                       Npp16s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp32f nThreshold,
                                          Npp32f nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp32f nThreshold,
                                      Npp32f nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp32f nThreshold,
                                           Npp32f nValue,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp32f nThreshold,
                                       Npp32f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp8u rThresholds[3],
                                         const Npp8u rValues[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp8u rThresholds[3],
                                     const Npp8u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp8u rThresholds[3],
                                          const Npp8u rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp8u rThresholds[3],
                                      const Npp8u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp16u rThresholds[3],
                                          const Npp16u rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16u rThresholds[3],
                                      const Npp16u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16u rThresholds[3],
                                           const Npp16u rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16u rThresholds[3],
                                       const Npp16u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          Npp16s *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp16s rThresholds[3],
                                          const Npp16s rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16s rThresholds[3],
                                      const Npp16s rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16s rThresholds[3],
                                           const Npp16s rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16s rThresholds[3],
                                       const Npp16s rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp32f rThresholds[3],
                                          const Npp32f rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f rThresholds[3],
                                      const Npp32f rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp32f rThresholds[3],
                                           const Npp32f rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f rThresholds[3],
                                       const Npp32f rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp8u rThresholds[3],
                                          const Npp8u rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp8u rThresholds[3],
                                      const Npp8u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp8u rThresholds[3],
                                           const Npp8u rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp8u rThresholds[3],
                                       const Npp8u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           Npp16u *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16u rThresholds[3],
                                           const Npp16u rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16u rThresholds[3],
                                       const Npp16u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp16u rThresholds[3],
                                            const Npp16u rValues[3],
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16u rThresholds[3],
                                        const Npp16u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                           Npp16s *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16s rThresholds[3],
                                           const Npp16s rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16s rThresholds[3],
                                       const Npp16s rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp16s rThresholds[3],
                                            const Npp16s rValues[3],
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16s rThresholds[3],
                                        const Npp16s rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           const Npp32f rThresholds[3],
                                           const Npp32f rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f rThresholds[3],
                                       const Npp32f rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_GTVal_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp32f rThresholds[3],
                                            const Npp32f rValues[3],
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_GTVal_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f rThresholds[3],
                                        const Npp32f rValues[3]);
// clang-format off
/**
 * @guard NPP_VERSION >= 12002
 * @param eSrcDstType SEND_ONLY
 * @param eSrcDstChannels SEND_ONLY
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pThreshold SEND_ONLY LENGTH:npp_threshold_bytes(eSrcDstType,eSrcDstChannels)
 * @param pvalue SEND_ONLY LENGTH:npp_threshold_bytes(eSrcDstType,eSrcDstChannels)
 * @param nppStreamCtx SEND_ONLY
 */
// clang-format on
NppStatus nppiFusedAbsDiff_Threshold_GTVal_Ctx(
    NppDataType eSrcDstType, NppiChannels eSrcDstChannels, const void *pSrc1,
    int nSrc1Step, const void *pSrc2, int nSrc2Step, void *pDst, int nDstStep,
    NppiSize oSizeROI, const void *pThreshold, const void *pvalue,
    NppStreamContext nppStreamCtx);
// clang-format off
/**
 * @guard NPP_VERSION >= 12002
 * @param eSrcDstType SEND_ONLY
 * @param eSrcDstChannels SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pThreshold SEND_ONLY LENGTH:npp_threshold_bytes(eSrcDstType,eSrcDstChannels)
 * @param pvalue SEND_ONLY LENGTH:npp_threshold_bytes(eSrcDstType,eSrcDstChannels)
 * @param nppStreamCtx SEND_ONLY
 */
// clang-format on
NppStatus nppiFusedAbsDiff_Threshold_GTVal_I_Ctx(
    NppDataType eSrcDstType, NppiChannels eSrcDstChannels, void *pSrcDst,
    int nSrcDstStep, const void *pSrc2, int nSrc2Step, NppiSize oSizeROI,
    const void *pThreshold, const void *pvalue, NppStreamContext nppStreamCtx);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nThreshold,
                                         Npp8u nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nThreshold,
                                     Npp8u nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI, Npp8u nThreshold,
                                          Npp8u nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, Npp8u nThreshold,
                                      Npp8u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp16u nThreshold,
                                          Npp16u nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp16u nThreshold,
                                      Npp16u nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp16u nThreshold,
                                           Npp16u nValue,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp16u nThreshold,
                                       Npp16u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          Npp16s *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp16s nThreshold,
                                          Npp16s nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp16s nThreshold,
                                      Npp16s nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp16s nThreshold,
                                           Npp16s nValue,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp16s nThreshold,
                                       Npp16s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp32f nThreshold,
                                          Npp32f nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp32f nThreshold,
                                      Npp32f nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_C1IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI, Npp32f nThreshold,
                                           Npp32f nValue,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThreshold SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_C1IR(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI, Npp32f nThreshold,
                                       Npp32f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const Npp8u rThresholds[3],
                                         const Npp8u rValues[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI,
                                     const Npp8u rThresholds[3],
                                     const Npp8u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                          NppiSize oSizeROI,
                                          const Npp8u rThresholds[3],
                                          const Npp8u rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI,
                                      const Npp8u rThresholds[3],
                                      const Npp8u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          Npp16u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp16u rThresholds[3],
                                          const Npp16u rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16u rThresholds[3],
                                      const Npp16u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16u rThresholds[3],
                                           const Npp16u rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16u rThresholds[3],
                                       const Npp16u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          Npp16s *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp16s rThresholds[3],
                                          const Npp16s rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp16s rThresholds[3],
                                      const Npp16s rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16s rThresholds[3],
                                           const Npp16s rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16s rThresholds[3],
                                       const Npp16s rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp32f rThresholds[3],
                                          const Npp32f rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp32f rThresholds[3],
                                      const Npp32f rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp32f rThresholds[3],
                                           const Npp32f rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f rThresholds[3],
                                       const Npp32f rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI,
                                          const Npp8u rThresholds[3],
                                          const Npp8u rValues[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const Npp8u rThresholds[3],
                                      const Npp8u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp8u rThresholds[3],
                                           const Npp8u rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const Npp8u rThresholds[3],
                                       const Npp8u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           Npp16u *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16u rThresholds[3],
                                           const Npp16u rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16u rThresholds[3],
                                       const Npp16u rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp16u rThresholds[3],
                                            const Npp16u rValues[3],
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16u rThresholds[3],
                                        const Npp16u rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                           Npp16s *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           const Npp16s rThresholds[3],
                                           const Npp16s rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp16s rThresholds[3],
                                       const Npp16s rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp16s rThresholds[3],
                                            const Npp16s rValues[3],
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp16s rThresholds[3],
                                        const Npp16s rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           const Npp32f rThresholds[3],
                                           const Npp32f rValues[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const Npp32f rThresholds[3],
                                       const Npp32f rValues[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTVal_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp32f rThresholds[3],
                                            const Npp32f rValues[3],
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholds SEND_ONLY DEREF
 * @param rValues SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTVal_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const Npp32f rThresholds[3],
                                        const Npp32f rValues[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, Npp8u nThresholdLT, Npp8u nValueLT, Npp8u nThresholdGT,
    Npp8u nValueGT, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp8u nThresholdLT,
                                          Npp8u nValueLT, Npp8u nThresholdGT,
                                          Npp8u nValueGT);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1IR_Ctx(
    Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp8u nThresholdLT,
    Npp8u nValueLT, Npp8u nThresholdGT, Npp8u nValueGT,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           Npp8u nThresholdLT, Npp8u nValueLT,
                                           Npp8u nThresholdGT, Npp8u nValueGT);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, Npp16u nThresholdLT, Npp16u nValueLT,
    Npp16u nThresholdGT, Npp16u nValueGT, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                           Npp16u *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           Npp16u nThresholdLT, Npp16u nValueLT,
                                           Npp16u nThresholdGT,
                                           Npp16u nValueGT);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1IR_Ctx(
    Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp16u nThresholdLT,
    Npp16u nValueLT, Npp16u nThresholdGT, Npp16u nValueGT,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_C1IR(
    Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp16u nThresholdLT,
    Npp16u nValueLT, Npp16u nThresholdGT, Npp16u nValueGT);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, Npp16s nThresholdLT, Npp16s nValueLT,
    Npp16s nThresholdGT, Npp16s nValueGT, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                           Npp16s *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           Npp16s nThresholdLT, Npp16s nValueLT,
                                           Npp16s nThresholdGT,
                                           Npp16s nValueGT);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1IR_Ctx(
    Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp16s nThresholdLT,
    Npp16s nValueLT, Npp16s nThresholdGT, Npp16s nValueGT,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_C1IR(
    Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp16s nThresholdLT,
    Npp16s nValueLT, Npp16s nThresholdGT, Npp16s nValueGT);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, Npp32f nThresholdLT, Npp32f nValueLT,
    Npp32f nThresholdGT, Npp32f nValueGT, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                           Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI,
                                           Npp32f nThresholdLT, Npp32f nValueLT,
                                           Npp32f nThresholdGT,
                                           Npp32f nValueGT);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1IR_Ctx(
    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nThresholdLT,
    Npp32f nValueLT, Npp32f nThresholdGT, Npp32f nValueGT,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nThresholdLT SEND_ONLY
 * @param nValueLT SEND_ONLY
 * @param nThresholdGT SEND_ONLY
 * @param nValueGT SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_C1IR(
    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, Npp32f nThresholdLT,
    Npp32f nValueLT, Npp32f nThresholdGT, Npp32f nValueGT);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3],
    const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3R(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3],
    const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                               NppiSize oSizeROI,
                                               const Npp8u rThresholdsLT[3],
                                               const Npp8u rValuesLT[3],
                                               const Npp8u rThresholdsGT[3],
                                               const Npp8u rValuesGT[3],
                                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                           NppiSize oSizeROI,
                                           const Npp8u rThresholdsLT[3],
                                           const Npp8u rValuesLT[3],
                                           const Npp8u rThresholdsGT[3],
                                           const Npp8u rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3],
    const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3R(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3],
    const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3IR_Ctx(
    Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
    const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3],
    const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp16u rThresholdsLT[3],
                                            const Npp16u rValuesLT[3],
                                            const Npp16u rThresholdsGT[3],
                                            const Npp16u rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3R_Ctx(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3],
    const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3R(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3],
    const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3IR_Ctx(
    Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
    const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3],
    const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp16s rThresholdsLT[3],
                                            const Npp16s rValuesLT[3],
                                            const Npp16s rThresholdsGT[3],
                                            const Npp16s rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3],
    const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3R(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3],
    const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3IR_Ctx(
    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
    const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3],
    const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp32f rThresholdsLT[3],
                                            const Npp32f rValuesLT[3],
                                            const Npp32f rThresholdsGT[3],
                                            const Npp32f rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3],
    const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4R(
    const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp8u rThresholdsLT[3], const Npp8u rValuesLT[3],
    const Npp8u rThresholdsGT[3], const Npp8u rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                                NppiSize oSizeROI,
                                                const Npp8u rThresholdsLT[3],
                                                const Npp8u rValuesLT[3],
                                                const Npp8u rThresholdsGT[3],
                                                const Npp8u rValuesGT[3],
                                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                            NppiSize oSizeROI,
                                            const Npp8u rThresholdsLT[3],
                                            const Npp8u rValuesLT[3],
                                            const Npp8u rThresholdsGT[3],
                                            const Npp8u rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4R_Ctx(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3],
    const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4R(
    const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3],
    const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4IR_Ctx(
    Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
    const Npp16u rThresholdsLT[3], const Npp16u rValuesLT[3],
    const Npp16u rThresholdsGT[3], const Npp16u rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                             NppiSize oSizeROI,
                                             const Npp16u rThresholdsLT[3],
                                             const Npp16u rValuesLT[3],
                                             const Npp16u rThresholdsGT[3],
                                             const Npp16u rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4R_Ctx(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3],
    const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4R(
    const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3],
    const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4IR_Ctx(
    Npp16s *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
    const Npp16s rThresholdsLT[3], const Npp16s rValuesLT[3],
    const Npp16s rThresholdsGT[3], const Npp16s rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                             NppiSize oSizeROI,
                                             const Npp16s rThresholdsLT[3],
                                             const Npp16s rValuesLT[3],
                                             const Npp16s rThresholdsGT[3],
                                             const Npp16s rValuesGT[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3],
    const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4R(
    const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
    NppiSize oSizeROI, const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3],
    const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4IR_Ctx(
    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
    const Npp32f rThresholdsLT[3], const Npp32f rValuesLT[3],
    const Npp32f rThresholdsGT[3], const Npp32f rValuesGT[3],
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param rThresholdsLT SEND_ONLY DEREF
 * @param rValuesLT SEND_ONLY DEREF
 * @param rThresholdsGT SEND_ONLY DEREF
 * @param rValuesGT SEND_ONLY DEREF
 */
NppStatus nppiThreshold_LTValGTVal_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                             NppiSize oSizeROI,
                                             const Npp32f rThresholdsLT[3],
                                             const Npp32f rValuesLT[3],
                                             const Npp32f rThresholdsGT[3],
                                             const Npp32f rValuesGT[3]);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_8u_C1R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_8u_C3R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_8u_C4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                 const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_8u_C4R(const Npp8u *pSrc1, int nSrc1Step,
                             const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_8u_AC4R_Ctx(const Npp8u *pSrc1, int nSrc1Step,
                                  const Npp8u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_8u_AC4R(const Npp8u *pSrc1, int nSrc1Step,
                              const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16u_C1R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16u_C3R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16u_C3R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16u_C4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                  const Npp16u *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16u_C4R(const Npp16u *pSrc1, int nSrc1Step,
                              const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16u_AC4R_Ctx(const Npp16u *pSrc1, int nSrc1Step,
                                   const Npp16u *pSrc2, int nSrc2Step,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16u_AC4R(const Npp16u *pSrc1, int nSrc1Step,
                               const Npp16u *pSrc2, int nSrc2Step, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16s_C1R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16s_C1R(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16s_C3R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16s_C3R(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16s_C4R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                  const Npp16s *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16s_C4R(const Npp16s *pSrc1, int nSrc1Step,
                              const Npp16s *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_16s_AC4R_Ctx(const Npp16s *pSrc1, int nSrc1Step,
                                   const Npp16s *pSrc2, int nSrc2Step,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_16s_AC4R(const Npp16s *pSrc1, int nSrc1Step,
                               const Npp16s *pSrc2, int nSrc2Step, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                  const Npp32f *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                  const Npp32f *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                  const Npp32f *pSrc2, int nSrc2Step,
                                  Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                              const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompare_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                   const Npp32f *pSrc2, int nSrc2Step,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
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
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompare_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                               const Npp32f *pSrc2, int nSrc2Step, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  Npp8u nConstant, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u nConstant,
                              Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  const Npp8u *pConstants, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                              const Npp8u *pConstants, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                  const Npp8u *pConstants, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppCmpOp eComparisonOperation,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                              const Npp8u *pConstants, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                   const Npp8u *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                               const Npp8u *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                               Npp16u nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   const Npp16u *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                               const Npp16u *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   const Npp16u *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                               const Npp16u *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    const Npp16u *pConstants, Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                const Npp16u *pConstants, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                               Npp16s nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   const Npp16s *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                               const Npp16s *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   const Npp16s *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                               const Npp16s *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    const Npp16s *pConstants, Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                const Npp16s *pConstants, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f nConstant, Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                               Npp32f nConstant, Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                   const Npp32f *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                               const Npp32f *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                   const Npp32f *pConstants, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   NppCmpOp eComparisonOperation,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                               const Npp32f *pConstants, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppCmpOp eComparisonOperation);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareC_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    const Npp32f *pConstants, Npp8u *pDst,
                                    int nDstStep, NppiSize oSizeROI,
                                    NppCmpOp eComparisonOperation,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eComparisonOperation SEND_ONLY
 */
NppStatus nppiCompareC_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                const Npp32f *pConstants, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppCmpOp eComparisonOperation);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                          const Npp32f *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp32f nEpsilon,
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
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_C1R(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_C3R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                          const Npp32f *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp32f nEpsilon,
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
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_C3R(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_C4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                          const Npp32f *pSrc2, int nSrc2Step,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, Npp32f nEpsilon,
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
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_C4R(const Npp32f *pSrc1, int nSrc1Step,
                                      const Npp32f *pSrc2, int nSrc2Step,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp32f nEpsilon);
/**
 * @param pSrc1 SEND_ONLY
 * @param nSrc1Step SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nSrc2Step SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_AC4R_Ctx(const Npp32f *pSrc1, int nSrc1Step,
                                           const Npp32f *pSrc2, int nSrc2Step,
                                           Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon,
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
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEps_32f_AC4R(const Npp32f *pSrc1, int nSrc1Step,
                                       const Npp32f *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, Npp32f nEpsilon);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           Npp32f nConstant, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI,
                                           Npp32f nEpsilon,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param nConstant SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f nConstant, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       Npp32f nEpsilon);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           const Npp32f *pConstants,
                                           Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                       const Npp32f *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       Npp32f nEpsilon);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           const Npp32f *pConstants,
                                           Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, Npp32f nEpsilon,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                       const Npp32f *pConstants, Npp8u *pDst,
                                       int nDstStep, NppiSize oSizeROI,
                                       Npp32f nEpsilon);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                            const Npp32f *pConstants,
                                            Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, Npp32f nEpsilon,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pConstants SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nEpsilon SEND_ONLY
 */
NppStatus nppiCompareEqualEpsC_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                        const Npp32f *pConstants, Npp8u *pDst,
                                        int nDstStep, NppiSize oSizeROI,
                                        Npp32f nEpsilon);
