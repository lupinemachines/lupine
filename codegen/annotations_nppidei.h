#include <npp.h>

// NPP image data exchange and initialization (libnppidei).
//
// Images are device memory and travel as addresses. C arrays of fill and
// border values, channel orders and per-plane image addresses are host memory
// and travel by contents.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnppidei by name at runtime. A
// call with a stream context goes to the server that owns the context's stream,
// or to the runtime's current device when the context names the default stream;
// a call without one goes to the current device.
//
// Each @guard names the NPP releases that export the call: the calls without a
// stream context end with NPP 12, and later calls start at the release that
// added them. A call NPP exports without declaring it in any release is left
// out.

/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8s_C1R_Ctx(Npp8s nValue, Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8s_C1R(Npp8s nValue, Npp8s *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8s_C2R_Ctx(const Npp8s aValue[2], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8s_C2R(const Npp8s aValue[2], Npp8s *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8s_C3R_Ctx(const Npp8s aValue[3], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8s_C3R(const Npp8s aValue[3], Npp8s *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8s_C4R_Ctx(const Npp8s aValue[4], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8s_C4R(const Npp8s aValue[4], Npp8s *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8s_AC4R_Ctx(const Npp8s aValue[3], Npp8s *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8s_AC4R(const Npp8s aValue[3], Npp8s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C1R_Ctx(Npp8u nValue, Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_C1R(Npp8u nValue, Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C2R_Ctx(const Npp8u aValue[2], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_C2R(const Npp8u aValue[2], Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C3R_Ctx(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_C3R(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C4R_Ctx(const Npp8u aValue[4], Npp8u *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_C4R(const Npp8u aValue[4], Npp8u *pDst, int nDstStep,
                         NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_AC4R_Ctx(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_AC4R(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C1R_Ctx(Npp16u nValue, Npp16u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_C1R(Npp16u nValue, Npp16u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C2R_Ctx(const Npp16u aValue[2], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_C2R(const Npp16u aValue[2], Npp16u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C3R_Ctx(const Npp16u aValue[3], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_C3R(const Npp16u aValue[3], Npp16u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C4R_Ctx(const Npp16u aValue[4], Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_C4R(const Npp16u aValue[4], Npp16u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_AC4R_Ctx(const Npp16u aValue[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_AC4R(const Npp16u aValue[3], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C1R_Ctx(Npp16s nValue, Npp16s *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_C1R(Npp16s nValue, Npp16s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C2R_Ctx(const Npp16s aValue[2], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_C2R(const Npp16s aValue[2], Npp16s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C3R_Ctx(const Npp16s aValue[3], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_C3R(const Npp16s aValue[3], Npp16s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C4R_Ctx(const Npp16s aValue[4], Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_C4R(const Npp16s aValue[4], Npp16s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_AC4R_Ctx(const Npp16s aValue[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_AC4R(const Npp16s aValue[3], Npp16s *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param oValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16sc_C1R_Ctx(Npp16sc oValue, Npp16sc *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16sc_C1R(Npp16sc oValue, Npp16sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16sc_C2R_Ctx(const Npp16sc aValue[2], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16sc_C2R(const Npp16sc aValue[2], Npp16sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16sc_C3R_Ctx(const Npp16sc aValue[3], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16sc_C3R(const Npp16sc aValue[3], Npp16sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16sc_C4R_Ctx(const Npp16sc aValue[4], Npp16sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16sc_C4R(const Npp16sc aValue[4], Npp16sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16sc_AC4R_Ctx(const Npp16sc aValue[3], Npp16sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16sc_AC4R(const Npp16sc aValue[3], Npp16sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C1R_Ctx(Npp32s nValue, Npp32s *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_C1R(Npp32s nValue, Npp32s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C2R_Ctx(const Npp32s aValue[2], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_C2R(const Npp32s aValue[2], Npp32s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C3R_Ctx(const Npp32s aValue[3], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_C3R(const Npp32s aValue[3], Npp32s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C4R_Ctx(const Npp32s aValue[4], Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_C4R(const Npp32s aValue[4], Npp32s *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_AC4R_Ctx(const Npp32s aValue[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_AC4R(const Npp32s aValue[3], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32u_C1R_Ctx(Npp32u nValue, Npp32u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32u_C1R(Npp32u nValue, Npp32u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32u_C2R_Ctx(const Npp32u aValue[2], Npp32u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32u_C2R(const Npp32u aValue[2], Npp32u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32u_C3R_Ctx(const Npp32u aValue[3], Npp32u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32u_C3R(const Npp32u aValue[3], Npp32u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32u_C4R_Ctx(const Npp32u aValue[4], Npp32u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32u_C4R(const Npp32u aValue[4], Npp32u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32u_AC4R_Ctx(const Npp32u aValue[3], Npp32u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32u_AC4R(const Npp32u aValue[3], Npp32u *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param oValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32sc_C1R_Ctx(Npp32sc oValue, Npp32sc *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32sc_C1R(Npp32sc oValue, Npp32sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32sc_C2R_Ctx(const Npp32sc aValue[2], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32sc_C2R(const Npp32sc aValue[2], Npp32sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32sc_C3R_Ctx(const Npp32sc aValue[3], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32sc_C3R(const Npp32sc aValue[3], Npp32sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32sc_C4R_Ctx(const Npp32sc aValue[4], Npp32sc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32sc_C4R(const Npp32sc aValue[4], Npp32sc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32sc_AC4R_Ctx(const Npp32sc aValue[3], Npp32sc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32sc_AC4R(const Npp32sc aValue[3], Npp32sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16f_C1R_Ctx(Npp32f nValue, Npp16f *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16f_C1R(Npp32f nValue, Npp16f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValues SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16f_C2R_Ctx(const Npp32f aValues[2], Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValues SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16f_C2R(const Npp32f aValues[2], Npp16f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValues SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16f_C3R_Ctx(const Npp32f aValues[3], Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValues SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16f_C3R(const Npp32f aValues[3], Npp16f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValues SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16f_C4R_Ctx(const Npp32f aValues[4], Npp16f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValues SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16f_C4R(const Npp32f aValues[4], Npp16f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C1R_Ctx(Npp32f nValue, Npp32f *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_C1R(Npp32f nValue, Npp32f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C2R_Ctx(const Npp32f aValue[2], Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_C2R(const Npp32f aValue[2], Npp32f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C3R_Ctx(const Npp32f aValue[3], Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_C3R(const Npp32f aValue[3], Npp32f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C4R_Ctx(const Npp32f aValue[4], Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_C4R(const Npp32f aValue[4], Npp32f *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_AC4R_Ctx(const Npp32f aValue[3], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_AC4R(const Npp32f aValue[3], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param oValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32fc_C1R_Ctx(Npp32fc oValue, Npp32fc *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param oValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32fc_C1R(Npp32fc oValue, Npp32fc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32fc_C2R_Ctx(const Npp32fc aValue[2], Npp32fc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32fc_C2R(const Npp32fc aValue[2], Npp32fc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32fc_C3R_Ctx(const Npp32fc aValue[3], Npp32fc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32fc_C3R(const Npp32fc aValue[3], Npp32fc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32fc_C4R_Ctx(const Npp32fc aValue[4], Npp32fc *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32fc_C4R(const Npp32fc aValue[4], Npp32fc *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32fc_AC4R_Ctx(const Npp32fc aValue[3], Npp32fc *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32fc_AC4R(const Npp32fc aValue[3], Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C1MR_Ctx(Npp8u nValue, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp8u *pMask,
                              int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_8u_C1MR(Npp8u nValue, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp8u *pMask, int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C3MR_Ctx(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp8u *pMask,
                              int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_8u_C3MR(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp8u *pMask, int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C4MR_Ctx(const Npp8u aValue[4], Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, const Npp8u *pMask,
                              int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_8u_C4MR(const Npp8u aValue[4], Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI, const Npp8u *pMask, int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_AC4MR_Ctx(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp8u *pMask,
                               int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_8u_AC4MR(const Npp8u aValue[3], Npp8u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C1MR_Ctx(Npp16u nValue, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp8u *pMask,
                               int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16u_C1MR(Npp16u nValue, Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C3MR_Ctx(const Npp16u aValue[3], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16u_C3MR(const Npp16u aValue[3], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C4MR_Ctx(const Npp16u aValue[4], Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16u_C4MR(const Npp16u aValue[4], Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_AC4MR_Ctx(const Npp16u aValue[3], Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16u_AC4MR(const Npp16u aValue[3], Npp16u *pDst, int nDstStep,
                            NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C1MR_Ctx(Npp16s nValue, Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp8u *pMask,
                               int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16s_C1MR(Npp16s nValue, Npp16s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C3MR_Ctx(const Npp16s aValue[3], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16s_C3MR(const Npp16s aValue[3], Npp16s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C4MR_Ctx(const Npp16s aValue[4], Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16s_C4MR(const Npp16s aValue[4], Npp16s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_AC4MR_Ctx(const Npp16s aValue[3], Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_16s_AC4MR(const Npp16s aValue[3], Npp16s *pDst, int nDstStep,
                            NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C1MR_Ctx(Npp32s nValue, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp8u *pMask,
                               int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32s_C1MR(Npp32s nValue, Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C3MR_Ctx(const Npp32s aValue[3], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32s_C3MR(const Npp32s aValue[3], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C4MR_Ctx(const Npp32s aValue[4], Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32s_C4MR(const Npp32s aValue[4], Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_AC4MR_Ctx(const Npp32s aValue[3], Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32s_AC4MR(const Npp32s aValue[3], Npp32s *pDst, int nDstStep,
                            NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C1MR_Ctx(Npp32f nValue, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI, const Npp8u *pMask,
                               int nMaskStep, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32f_C1MR(Npp32f nValue, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C3MR_Ctx(const Npp32f aValue[3], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32f_C3MR(const Npp32f aValue[3], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C4MR_Ctx(const Npp32f aValue[4], Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32f_C4MR(const Npp32f aValue[4], Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_AC4MR_Ctx(const Npp32f aValue[3], Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aValue SEND_ONLY DEREF
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiSet_32f_AC4MR(const Npp32f aValue[3], Npp32f *pDst, int nDstStep,
                            NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C3CR_Ctx(Npp8u nValue, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_C3CR(Npp8u nValue, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_8u_C4CR_Ctx(Npp8u nValue, Npp8u *pDst, int nDstStep,
                              NppiSize oSizeROI, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_8u_C4CR(Npp8u nValue, Npp8u *pDst, int nDstStep,
                          NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C3CR_Ctx(Npp16u nValue, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_C3CR(Npp16u nValue, Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16u_C4CR_Ctx(Npp16u nValue, Npp16u *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16u_C4CR(Npp16u nValue, Npp16u *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C3CR_Ctx(Npp16s nValue, Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_C3CR(Npp16s nValue, Npp16s *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_16s_C4CR_Ctx(Npp16s nValue, Npp16s *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_16s_C4CR(Npp16s nValue, Npp16s *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C3CR_Ctx(Npp32s nValue, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_C3CR(Npp32s nValue, Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32s_C4CR_Ctx(Npp32s nValue, Npp32s *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32s_C4CR(Npp32s nValue, Npp32s *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C3CR_Ctx(Npp32f nValue, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_C3CR(Npp32f nValue, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSet_32f_C4CR_Ctx(Npp32f nValue, Npp32f *pDst, int nDstStep,
                               NppiSize oSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiSet_32f_C4CR(Npp32f nValue, Npp32f *pDst, int nDstStep,
                           NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
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
NppStatus nppiCopy_8s_C1R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8s_C2R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
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
NppStatus nppiCopy_8s_C2R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8s_C3R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
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
NppStatus nppiCopy_8s_C3R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8s_C4R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
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
NppStatus nppiCopy_8s_C4R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8s_AC4R_Ctx(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
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
NppStatus nppiCopy_8s_AC4R(const Npp8s *pSrc, int nSrcStep, Npp8s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                          int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16sc_C1R_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16sc_C1R(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16sc_C2R_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16sc_C2R(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16sc_C3R_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16sc_C3R(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16sc_C4R_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16sc_C4R(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16sc_AC4R_Ctx(const Npp16sc *pSrc, int nSrcStep,
                                 Npp16sc *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16sc_AC4R(const Npp16sc *pSrc, int nSrcStep, Npp16sc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_AC4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32sc_C1R_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32sc_C1R(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32sc_C2R_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32sc_C2R(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32sc_C3R_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32sc_C3R(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32sc_C4R_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32sc_C4R(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32sc_AC4R_Ctx(const Npp32sc *pSrc, int nSrcStep,
                                 Npp32sc *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32sc_AC4R(const Npp32sc *pSrc, int nSrcStep, Npp32sc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
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
NppStatus nppiCopy_16f_C1R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
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
NppStatus nppiCopy_16f_C3R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
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
NppStatus nppiCopy_16f_C4R(const Npp16f *pSrc, int nSrcStep, Npp16f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32fc_C1R_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32fc_C1R(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32fc_C2R_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32fc_C2R(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32fc_C3R_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32fc_C3R(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32fc_C4R_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32fc_C4R(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32fc_AC4R_Ctx(const Npp32fc *pSrc, int nSrcStep,
                                 Npp32fc *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32fc_AC4R(const Npp32fc *pSrc, int nSrcStep, Npp32fc *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C1MR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_8u_C1MR(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C3MR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_8u_C3MR(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C4MR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               const Npp8u *pMask, int nMaskStep,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_8u_C4MR(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                           int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_AC4MR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_8u_AC4MR(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C1MR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16u_C1MR(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C3MR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16u_C3MR(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C4MR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16u_C4MR(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_AC4MR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, int nMaskStep,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16u_AC4MR(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C1MR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16s_C1MR(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C3MR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16s_C3MR(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C4MR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16s_C4MR(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_AC4MR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, int nMaskStep,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_16s_AC4MR(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C1MR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32s_C1MR(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C3MR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32s_C3MR(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C4MR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32s_C4MR(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_AC4MR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, int nMaskStep,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32s_AC4MR(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C1MR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32f_C1MR(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C3MR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32f_C3MR(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C4MR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                const Npp8u *pMask, int nMaskStep,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32f_C4MR(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI, const Npp8u *pMask,
                            int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_AC4MR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 const Npp8u *pMask, int nMaskStep,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param pMask SEND_ONLY
 * @param nMaskStep SEND_ONLY
 */
NppStatus nppiCopy_32f_AC4MR(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI,
                             const Npp8u *pMask, int nMaskStep);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C3CR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C3CR(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C4CR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C4CR(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C3CR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C3CR(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C4CR_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C4CR(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C3CR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C3CR(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C4CR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C4CR(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C3CR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C3CR(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C4CR_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C4CR(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C3CR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C3CR(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C4CR_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C4CR(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C4C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C4C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C3C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C3C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C4C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C4C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C3C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C3C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C4C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C4C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C3C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C3C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C4C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C4C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C2C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C2C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C3C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C4C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C4C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C1C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C1C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C1C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
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
NppStatus nppiCopy_8u_C1C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C1C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C1C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C1C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiCopy_16s_C1C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C1C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C1C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C1C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiCopy_16u_C1C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C1C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C1C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C1C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiCopy_32s_C1C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C1C2R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C1C2R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C1C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C1C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C1C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
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
NppStatus nppiCopy_32f_C1C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                Npp8u *const aDst[3], int nDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_8u_C3P3R(const Npp8u *pSrc, int nSrcStep,
                            Npp8u *const aDst[3], int nDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_C4P4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                Npp8u *const aDst[4], int nDstStep,
                                NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_8u_C4P4R(const Npp8u *pSrc, int nSrcStep,
                            Npp8u *const aDst[4], int nDstStep,
                            NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C3P3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                 Npp16s *const aDst[3], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16s_C3P3R(const Npp16s *pSrc, int nSrcStep,
                             Npp16s *const aDst[3], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_C4P4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                 Npp16s *const aDst[4], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16s_C4P4R(const Npp16s *pSrc, int nSrcStep,
                             Npp16s *const aDst[4], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C3P3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                 Npp16u *const aDst[3], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16u_C3P3R(const Npp16u *pSrc, int nSrcStep,
                             Npp16u *const aDst[3], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_C4P4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                 Npp16u *const aDst[4], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16u_C4P4R(const Npp16u *pSrc, int nSrcStep,
                             Npp16u *const aDst[4], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C3P3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                 Npp32s *const aDst[3], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32s_C3P3R(const Npp32s *pSrc, int nSrcStep,
                             Npp32s *const aDst[3], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_C4P4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                 Npp32s *const aDst[4], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32s_C4P4R(const Npp32s *pSrc, int nSrcStep,
                             Npp32s *const aDst[4], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C3P3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                 Npp32f *const aDst[3], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32f_C3P3R(const Npp32f *pSrc, int nSrcStep,
                             Npp32f *const aDst[3], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_C4P4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                 Npp32f *const aDst[4], int nDstStep,
                                 NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param aDst SEND_ONLY DEREF
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32f_C4P4R(const Npp32f *pSrc, int nSrcStep,
                             Npp32f *const aDst[4], int nDstStep,
                             NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_P3C3R_Ctx(const Npp8u *const aSrc[3], int nSrcStep,
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_8u_P3C3R(const Npp8u *const aSrc[3], int nSrcStep,
                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_8u_P4C4R_Ctx(const Npp8u *const aSrc[4], int nSrcStep,
                                Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_8u_P4C4R(const Npp8u *const aSrc[4], int nSrcStep,
                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_P3C3R_Ctx(const Npp16u *const aSrc[3], int nSrcStep,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16u_P3C3R(const Npp16u *const aSrc[3], int nSrcStep,
                             Npp16u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16u_P4C4R_Ctx(const Npp16u *const aSrc[4], int nSrcStep,
                                 Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16u_P4C4R(const Npp16u *const aSrc[4], int nSrcStep,
                             Npp16u *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_P3C3R_Ctx(const Npp16s *const aSrc[3], int nSrcStep,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16s_P3C3R(const Npp16s *const aSrc[3], int nSrcStep,
                             Npp16s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_16s_P4C4R_Ctx(const Npp16s *const aSrc[4], int nSrcStep,
                                 Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_16s_P4C4R(const Npp16s *const aSrc[4], int nSrcStep,
                             Npp16s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_P3C3R_Ctx(const Npp32s *const aSrc[3], int nSrcStep,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32s_P3C3R(const Npp32s *const aSrc[3], int nSrcStep,
                             Npp32s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32s_P4C4R_Ctx(const Npp32s *const aSrc[4], int nSrcStep,
                                 Npp32s *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32s_P4C4R(const Npp32s *const aSrc[4], int nSrcStep,
                             Npp32s *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_P3C3R_Ctx(const Npp32f *const aSrc[3], int nSrcStep,
                                 Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32f_P3C3R(const Npp32f *const aSrc[3], int nSrcStep,
                             Npp32f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopy_32f_P4C4R_Ctx(const Npp32f *const aSrc[4], int nSrcStep,
                                 Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param aSrc SEND_ONLY DEREF
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 */
NppStatus nppiCopy_32f_P4C4R(const Npp32f *const aSrc[4], int nSrcStep,
                             Npp32f *pDst, int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp8u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth, Npp8u nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiCopyConstBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp8u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     Npp8u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_8u_C3R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, const Npp8u aValue[3], NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp8u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Npp8u aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_8u_C4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, const Npp8u aValue[4], NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp8u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight, int nLeftBorderWidth,
                                     const Npp8u aValue[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_8u_AC4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, const Npp8u aValue[3], NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp8u *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp8u aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth, Npp16u nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16u *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth, Npp16u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp16u aValue[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16u *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp16u aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp16u aValue[4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16u *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp16u aValue[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp16u *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth,
                                           const Npp16u aValue[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                       NppiSize oSrcSizeROI, Npp16u *pDst,
                                       int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight,
                                       int nLeftBorderWidth,
                                       const Npp16u aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth, Npp16s nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth, Npp16s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp16s aValue[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp16s aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp16s aValue[4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp16s aValue[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp16s *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth,
                                           const Npp16s aValue[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                       NppiSize oSrcSizeROI, Npp16s *pDst,
                                       int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight,
                                       int nLeftBorderWidth,
                                       const Npp16s aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth, Npp32s nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32s_C1R(const Npp32s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth, Npp32s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp32s aValue[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_32s_C3R(const Npp32s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp32s aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp32s aValue[4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_32s_C4R(const Npp32s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp32s aValue[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp32s *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth,
                                           const Npp32s aValue[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_32s_AC4R(const Npp32s *pSrc, int nSrcStep,
                                       NppiSize oSrcSizeROI, Npp32s *pDst,
                                       int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight,
                                       int nLeftBorderWidth,
                                       const Npp32s aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth, Npp32f nValue,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32f *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth, Npp32f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp32f aValue[3],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32f *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp32f aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          const Npp32f aValue[4],
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32f *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth,
                                      const Npp32f aValue[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyConstBorder_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp32f *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth,
                                           const Npp32f aValue[3],
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param aValue SEND_ONLY DEREF
 */
NppStatus nppiCopyConstBorder_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                       NppiSize oSrcSizeROI, Npp32f *pDst,
                                       int nDstStep, NppiSize oDstSizeROI,
                                       int nTopBorderHeight,
                                       int nLeftBorderWidth,
                                       const Npp32f aValue[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                             NppiSize oSrcSizeROI, Npp8u *pDst,
                                             int nDstStep, NppiSize oDstSizeROI,
                                             int nTopBorderHeight,
                                             int nLeftBorderWidth,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp8u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                             NppiSize oSrcSizeROI, Npp8u *pDst,
                                             int nDstStep, NppiSize oDstSizeROI,
                                             int nTopBorderHeight,
                                             int nLeftBorderWidth,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp8u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                             NppiSize oSrcSizeROI, Npp8u *pDst,
                                             int nDstStep, NppiSize oDstSizeROI,
                                             int nTopBorderHeight,
                                             int nLeftBorderWidth,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp8u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_AC4R_Ctx(
    const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp8u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp8u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_C1R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_C3R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_C4R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_AC4R_Ctx(
    const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16u *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp16u *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_C1R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_C3R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_C4R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_AC4R_Ctx(
    const Npp16s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp16s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp16s *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_C1R_Ctx(
    const Npp32s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_C1R(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_C3R_Ctx(
    const Npp32s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_C3R(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_C4R_Ctx(
    const Npp32s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_C4R(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_AC4R_Ctx(
    const Npp32s *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32s *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32s_AC4R(const Npp32s *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp32s *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_C1R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32f *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_C3R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32f *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_C4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32f *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_AC4R_Ctx(
    const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSizeROI, Npp32f *pDst,
    int nDstStep, NppiSize oDstSizeROI, int nTopBorderHeight,
    int nLeftBorderWidth, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyReplicateBorder_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                           NppiSize oSrcSizeROI, Npp32f *pDst,
                                           int nDstStep, NppiSize oDstSizeROI,
                                           int nTopBorderHeight,
                                           int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        NppiSize oSrcSizeROI, Npp8u *pDst,
                                        int nDstStep, NppiSize oDstSizeROI,
                                        int nTopBorderHeight,
                                        int nLeftBorderWidth,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                    NppiSize oSrcSizeROI, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        NppiSize oSrcSizeROI, Npp8u *pDst,
                                        int nDstStep, NppiSize oDstSizeROI,
                                        int nTopBorderHeight,
                                        int nLeftBorderWidth,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                    NppiSize oSrcSizeROI, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        NppiSize oSrcSizeROI, Npp8u *pDst,
                                        int nDstStep, NppiSize oDstSizeROI,
                                        int nTopBorderHeight,
                                        int nLeftBorderWidth,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                    NppiSize oSrcSizeROI, Npp8u *pDst,
                                    int nDstStep, NppiSize oDstSizeROI,
                                    int nTopBorderHeight, int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp8u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp8u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp16u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp16u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp16u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp16u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp16u *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp16u *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16u *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16u *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp16s *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp16s *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp16s *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp16s *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp16s *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp16s *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp16s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp16s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp32s *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_C1R(const Npp32s *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp32s *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp32s *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_C3R(const Npp32s *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp32s *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp32s *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_C4R(const Npp32s *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp32s *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32s *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32s_AC4R(const Npp32s *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32s *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp32f *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_C1R(const Npp32f *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp32f *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp32f *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp32f *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         NppiSize oSrcSizeROI, Npp32f *pDst,
                                         int nDstStep, NppiSize oDstSizeROI,
                                         int nTopBorderHeight,
                                         int nLeftBorderWidth,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                     NppiSize oSrcSizeROI, Npp32f *pDst,
                                     int nDstStep, NppiSize oDstSizeROI,
                                     int nTopBorderHeight,
                                     int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI, Npp32f *pDst,
                                          int nDstStep, NppiSize oDstSizeROI,
                                          int nTopBorderHeight,
                                          int nLeftBorderWidth,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param oSrcSizeROI SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nTopBorderHeight SEND_ONLY
 * @param nLeftBorderWidth SEND_ONLY
 */
NppStatus nppiCopyWrapBorder_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI, Npp32f *pDst,
                                      int nDstStep, NppiSize oDstSizeROI,
                                      int nTopBorderHeight,
                                      int nLeftBorderWidth);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oDstSizeROI, Npp32f nDx,
                                    Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oDstSizeROI, Npp32f nDx,
                                    Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oDstSizeROI, Npp32f nDx,
                                    Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oDstSizeROI, Npp32f nDx,
                                      Npp32f nDy,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep,
                                  NppiSize oDstSizeROI, Npp32f nDx, Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oDstSizeROI, Npp32f nDx,
                                      Npp32f nDy,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep,
                                  NppiSize oDstSizeROI, Npp32f nDx, Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                      Npp32s *pDst, int nDstStep,
                                      NppiSize oDstSizeROI, Npp32f nDx,
                                      Npp32f nDy,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32s_AC4R(const Npp32s *pSrc, int nSrcStep,
                                  Npp32s *pDst, int nDstStep,
                                  NppiSize oDstSizeROI, Npp32f nDx, Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oDstSizeROI, Npp32f nDx,
                                     Npp32f nDy, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oDstSizeROI, Npp32f nDx,
                                 Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp32f *pDst, int nDstStep,
                                      NppiSize oDstSizeROI, Npp32f nDx,
                                      Npp32f nDy,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nDx SEND_ONLY
 * @param nDy SEND_ONLY
 */
NppStatus nppiCopySubpix_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp32f *pDst, int nDstStep,
                                  NppiSize oDstSizeROI, Npp32f nDx, Npp32f nDy);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16u_C1R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16u_C3R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16u_C4R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16s_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16s_C1R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16s_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16s_C3R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16s_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16s_C4R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u16s_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u16s_AC4R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32s_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8u32s_C1R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32s_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8u32s_C3R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32s_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8u32s_C4R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32s_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8u32s_AC4R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32f_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u32f_C3R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32f_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u32f_C4R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u32f_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8u32f_AC4R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8s32s_C1R(const Npp8s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32s_C3R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8s32s_C3R(const Npp8s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32s_C4R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8s32s_C4R(const Npp8s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32s_AC4R_Ctx(const Npp8s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_8s32s_AC4R(const Npp8s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32f_C1R_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s32f_C1R(const Npp8s *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32f_C3R_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s32f_C3R(const Npp8s *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32f_C4R_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s32f_C4R(const Npp8s *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32f_AC4R_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s32f_AC4R(const Npp8s *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32s_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16u32s_C1R(const Npp16u *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32s_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16u32s_C3R(const Npp16u *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32s_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16u32s_C4R(const Npp16u *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32s_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16u32s_AC4R(const Npp16u *pSrc, int nSrcStep,
                                  Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32f_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u32f_C1R(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32f_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u32f_C3R(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32f_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u32f_C4R(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32f_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u32f_AC4R(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s32s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16s32s_C1R(const Npp16s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16s32s_C3R(const Npp16s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16s32s_C4R(const Npp16s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp32s *pDst, int nDstStep,
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
NppStatus nppiConvert_16s32s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                  Npp32s *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32f_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s32f_C1R(const Npp16s *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32f_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s32f_C3R(const Npp16s *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32f_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s32f_C4R(const Npp16s *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s32f_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s32f_AC4R(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16f32f_C1R_Ctx(const Npp16f *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16f32f_C1R(const Npp16f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16f32f_C3R_Ctx(const Npp16f *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16f32f_C3R(const Npp16f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16f32f_C4R_Ctx(const Npp16f *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16f32f_C4R(const Npp16f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16f32f_AC4R_Ctx(const Npp16f *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16f32f_AC4R(const Npp16f *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s8u_C1Rs_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s8u_C1Rs(const Npp8s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s16u_C1Rs_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s16u_C1Rs(const Npp8s *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s16s_C1R_Ctx(const Npp8s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_8s16s_C1R(const Npp8s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8s32u_C1Rs_Ctx(const Npp8s *pSrc, int nSrcStep,
                                     Npp32u *pDst, int nDstStep,
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
NppStatus nppiConvert_8s32u_C1Rs(const Npp8s *pSrc, int nSrcStep, Npp32u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s16u_C1Rs_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s16u_C1Rs(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s32u_C1Rs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp32u *pDst, int nDstStep,
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
NppStatus nppiConvert_16s32u_C1Rs(const Npp16s *pSrc, int nSrcStep,
                                  Npp32u *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u32u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                     Npp32u *pDst, int nDstStep,
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
NppStatus nppiConvert_16u32u_C1R(const Npp16u *pSrc, int nSrcStep, Npp32u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s32u_C1Rs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                      Npp32u *pDst, int nDstStep,
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
NppStatus nppiConvert_32s32u_C1Rs(const Npp32s *pSrc, int nSrcStep,
                                  Npp32u *pDst, int nDstStep,
                                  NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s32f_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_32s32f_C1R(const Npp32s *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32u32f_C1R_Ctx(const Npp32u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_32u32f_C1R(const Npp32u *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u8u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u8u_C1R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u8u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u8u_C3R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u8u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u8u_C4R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u8u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16u8u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s8u_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s8u_C1R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s8u_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s8u_C3R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s8u_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s8u_C4R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s8u_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_16s8u_AC4R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8u_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_32s8u_C1R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8u_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_32s8u_C3R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8u_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_32s8u_C4R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8u_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
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
NppStatus nppiConvert_32s8u_AC4R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
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
NppStatus nppiConvert_32s8s_C1R(const Npp32s *pSrc, int nSrcStep, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
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
NppStatus nppiConvert_32s8s_C3R(const Npp32s *pSrc, int nSrcStep, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
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
NppStatus nppiConvert_32s8s_C4R(const Npp32s *pSrc, int nSrcStep, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s8s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                     Npp8s *pDst, int nDstStep,
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
NppStatus nppiConvert_32s8s_AC4R(const Npp32s *pSrc, int nSrcStep, Npp8s *pDst,
                                 int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_8u8s_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8s *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppRoundMode eRoundMode, int nScaleFactor,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_8u8s_C1RSfs(const Npp8u *pSrc, int nSrcStep, Npp8s *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u8s_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppRoundMode eRoundMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_16u8s_C1RSfs(const Npp16u *pSrc, int nSrcStep,
                                   Npp8s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16s8s_C1RSfs_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppRoundMode eRoundMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_16s8s_C1RSfs(const Npp16s *pSrc, int nSrcStep,
                                   Npp8s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_16u16s_C1RSfs_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_16u16s_C1RSfs(const Npp16u *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32u8u_C1RSfs_Ctx(const Npp32u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppRoundMode eRoundMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32u8u_C1RSfs(const Npp32u *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32u8s_C1RSfs_Ctx(const Npp32u *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppRoundMode eRoundMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32u8s_C1RSfs(const Npp32u *pSrc, int nSrcStep,
                                   Npp8s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32u16u_C1RSfs_Ctx(const Npp32u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32u16u_C1RSfs(const Npp32u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32u16s_C1RSfs_Ctx(const Npp32u *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32u16s_C1RSfs(const Npp32u *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32u32s_C1RSfs_Ctx(const Npp32u *pSrc, int nSrcStep,
                                        Npp32s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32u32s_C1RSfs(const Npp32u *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s16u_C1RSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32s16u_C1RSfs(const Npp32s *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32s16s_C1RSfs_Ctx(const Npp32s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32s16s_C1RSfs(const Npp32s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C1R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C3R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C4R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8u_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8u_AC4R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C1R(const Npp32f *pSrc, int nSrcStep, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C3R(const Npp32f *pSrc, int nSrcStep, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp8s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C4R(const Npp32f *pSrc, int nSrcStep, Npp8s *pDst,
                                int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8s_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp8s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f8s_AC4R(const Npp32f *pSrc, int nSrcStep, Npp8s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C1R(const Npp32f *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C3R(const Npp32f *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C4R(const Npp32f *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16u_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppRoundMode eRoundMode,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16u_AC4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C1R(const Npp32f *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C3R(const Npp32f *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C4R(const Npp32f *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16s_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppRoundMode eRoundMode,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16s_AC4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16f *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16f_C1R(const Npp32f *pSrc, int nSrcStep, Npp16f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16f *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16f_C3R(const Npp32f *pSrc, int nSrcStep, Npp16f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                     Npp16f *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppRoundMode eRoundMode,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16f_C4R(const Npp32f *pSrc, int nSrcStep, Npp16f *pDst,
                                 int nDstStep, NppiSize oSizeROI,
                                 NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                      Npp16f *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      NppRoundMode eRoundMode,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 */
NppStatus nppiConvert_32f16f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                  Npp16f *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppRoundMode eRoundMode);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C1RSfs_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppRoundMode eRoundMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32f8u_C1RSfs(const Npp32f *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C1RSfs_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp8s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       NppRoundMode eRoundMode,
                                       int nScaleFactor,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32f8s_C1RSfs(const Npp32f *pSrc, int nSrcStep,
                                   Npp8s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C1RSfs_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32f16u_C1RSfs(const Npp32f *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C1RSfs_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32f16s_C1RSfs(const Npp32f *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f32u_C1RSfs_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32f32u_C1RSfs(const Npp32f *pSrc, int nSrcStep,
                                    Npp32u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiConvert_32f32s_C1RSfs_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        NppRoundMode eRoundMode,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppiConvert_32f32s_C1RSfs(const Npp32f *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppRoundMode eRoundMode,
                                    int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiScale_8u16u_C1R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiScale_8u16u_C3R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
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
NppStatus nppiScale_8u16u_C4R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiScale_8u16u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16s_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiScale_8u16s_C1R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16s_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiScale_8u16s_C3R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16s_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
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
NppStatus nppiScale_8u16s_C4R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u16s_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
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
NppStatus nppiScale_8u16s_AC4R(const Npp8u *pSrc, int nSrcStep, Npp16s *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32s_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiScale_8u32s_C1R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32s_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiScale_8u32s_C3R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32s_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
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
NppStatus nppiScale_8u32s_C4R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                              int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32s_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                   Npp32s *pDst, int nDstStep,
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
NppStatus nppiScale_8u32s_AC4R(const Npp8u *pSrc, int nSrcStep, Npp32s *pDst,
                               int nDstStep, NppiSize oSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                  int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                                  Npp32f nMax, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                              Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32f_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                  int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                                  Npp32f nMax, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_8u32f_C3R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                              Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32f_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                                  int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                                  Npp32f nMax, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_8u32f_C4R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                              Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_8u32f_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, Npp32f nMin, Npp32f nMax,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_8u32f_AC4R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst,
                               int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                               Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16u8u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16u8u_C1R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16u8u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16u8u_C3R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16u8u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16u8u_C4R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16u8u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppHintAlgorithm hint,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16u8u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16s8u_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16s8u_C1R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16s8u_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16s8u_C3R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16s8u_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16s8u_C4R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_16s8u_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppHintAlgorithm hint,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_16s8u_AC4R(const Npp16s *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32s8u_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_32s8u_C1R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32s8u_C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_32s8u_C3R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32s8u_C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  NppHintAlgorithm hint,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_32s8u_C4R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI,
                              NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32s8u_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppHintAlgorithm hint,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param hint SEND_ONLY
 */
NppStatus nppiScale_32s8u_AC4R(const Npp32s *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI,
                               NppHintAlgorithm hint);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32f8u_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                                  Npp32f nMax, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_32f8u_C1R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                              Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32f8u_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                                  Npp32f nMax, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_32f8u_C3R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                              Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32f8u_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                                  Npp32f nMax, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_32f8u_C4R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                              Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiScale_32f8u_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                   Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   Npp32f nMin, Npp32f nMax,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param nMin SEND_ONLY
 * @param nMax SEND_ONLY
 */
NppStatus nppiScale_32f8u_AC4R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSizeROI, Npp32f nMin,
                               Npp32f nMax);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_8u_C1C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_8u_C1C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_8u_C1C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oDstSizeROI,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_8u_C1C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                           int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_8u_C1AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_8u_C1AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_16u_C1C3R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_16u_C1C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_16u_C1C4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_16u_C1C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_16u_C1AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                 int nDstStep, NppiSize oDstSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_16u_C1AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                             int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_16s_C1C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_16s_C1C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_16s_C1C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_16s_C1C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_16s_C1AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_16s_C1AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                             int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_32s_C1C3R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_32s_C1C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_32s_C1C4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_32s_C1C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_32s_C1AC4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                 int nDstStep, NppiSize oDstSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_32s_C1AC4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                             int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_32f_C1C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_32f_C1C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_32f_C1C4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oDstSizeROI,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_32f_C1C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                            int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiDup_32f_C1AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                 int nDstStep, NppiSize oDstSizeROI,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oDstSizeROI SEND_ONLY
 */
NppStatus nppiDup_32f_C1AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSizeROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSrcROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSrcROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSrcROI,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                               int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_16u_C1R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_16u_C3R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_16u_C4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_32s_C1R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_32s_C1R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_32s_C3R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_32s_C4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_32f_C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiTranspose_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSrcROI,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSrcROI SEND_ONLY
 */
NppStatus nppiTranspose_32f_C4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,
                                int nDstStep, NppiSize oSrcROI);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const int aDstOrder[3],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  const int aDstOrder[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C4C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_8u_C4C3R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, const int aDstOrder[4],
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI,
                                  const int aDstOrder[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C3C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                        Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[4], Npp8u nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_C3C4R(const Npp8u *pSrc, int nSrcStep,
                                    Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[4],
                                    Npp8u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                       Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                   int nDstStep, NppiSize oSizeROI,
                                   const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C4C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16u_C4C3R(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                       Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16u_C4R(const Npp16u *pSrc, int nSrcStep,
                                   Npp16u *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16u_C4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C3C4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                         Npp16u *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[4], Npp16u nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_C3C4R(const Npp16u *pSrc, int nSrcStep,
                                     Npp16u *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[4],
                                     Npp16u nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                        Npp16u *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                    Npp16u *pDst, int nDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C4C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         Npp16s *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16s_C4C3R(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                       Npp16s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16s_C4R(const Npp16s *pSrc, int nSrcStep,
                                   Npp16s *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C3C4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                         Npp16s *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[4], Npp16s nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_C3C4R(const Npp16s *pSrc, int nSrcStep,
                                     Npp16s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[4],
                                     Npp16s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                        Npp16s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                    Npp16s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                       Npp32s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32s_C3R(const Npp32s *pSrc, int nSrcStep,
                                   Npp32s *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C3IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32s_C3IR(Npp32s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C4C3R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                         Npp32s *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32s_C4C3R(const Npp32s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                       Npp32s *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32s_C4R(const Npp32s *pSrc, int nSrcStep,
                                   Npp32s *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C4IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32s_C4IR(Npp32s *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C3C4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                         Npp32s *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[4], Npp32s nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_C3C4R(const Npp32s *pSrc, int nSrcStep,
                                     Npp32s *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[4],
                                     Npp32s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep,
                                        Npp32s *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32s_AC4R(const Npp32s *pSrc, int nSrcStep,
                                    Npp32s *pDst, int nDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[3],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32f_C3R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C3IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32f_C3IR(Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C4C3R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[3],
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32f_C4C3R(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[3]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                       Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI,
                                       const int aDstOrder[4],
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32f_C4R(const Npp32f *pSrc, int nSrcStep,
                                   Npp32f *pDst, int nDstStep,
                                   NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[4],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nSrcDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32f_C4IR(Npp32f *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[4]);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C3C4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                         Npp32f *pDst, int nDstStep,
                                         NppiSize oSizeROI,
                                         const int aDstOrder[4], Npp32f nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nValue SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_C3C4R(const Npp32f *pSrc, int nSrcStep,
                                     Npp32f *pDst, int nDstStep,
                                     NppiSize oSizeROI, const int aDstOrder[4],
                                     Npp32f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppiSwapChannels_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep,
                                        Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI,
                                        const int aDstOrder[3],
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nSrcStep SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nDstStep SEND_ONLY
 * @param oSizeROI SEND_ONLY
 * @param aDstOrder SEND_ONLY DEREF
 */
NppStatus nppiSwapChannels_32f_AC4R(const Npp32f *pSrc, int nSrcStep,
                                    Npp32f *pDst, int nDstStep,
                                    NppiSize oSizeROI, const int aDstOrder[3]);
