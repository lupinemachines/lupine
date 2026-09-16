#include <npp.h>

// NPP signal processing (libnpps).
//
// Signals, scratch buffers and every result a call computes are device memory
// and travel as addresses; scratch buffer sizes come back from host memory.
// nppsMalloc returns device memory the client records against the server that
// allocated it, so copies to and from it and nppsFree go to that server.
//
// The client forwards each call over the CUDA driver shim's connections, so the
// server child that owns the driver and runtime state runs the library too;
// the server handlers resolve the machine's libnpps by name at runtime. A call
// with a stream context goes to the server that owns the context's stream, or
// to the runtime's current device when the context names the default stream;
// a call without one goes to the current device.
//
// Each @guard names the NPP releases that export the call: the calls without a
// stream context end with NPP 12, and later calls start at the release that
// added them. A call NPP exports without declaring it in any release is left
// out.
//
// lupine_npp_length_t is the int that NPP 12.2.5 widened to size_t.

/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_8u_ISfs_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_8u_ISfs(Npp8u nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_8u_Sfs(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_16u_ISfs_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_16u_ISfs(Npp16u nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_16u_Sfs(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_16s_ISfs_Ctx(Npp16s nValue, Npp16s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_16s_ISfs(Npp16s nValue, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_16s_Sfs(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_16sc_ISfs(Npp16sc nValue, Npp16sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc nValue,
                                Npp16sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_16sc_Sfs(const Npp16sc *pSrc, Npp16sc nValue, Npp16sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32s_ISfs_Ctx(Npp32s nValue, Npp32s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_32s_ISfs(Npp32s nValue, Npp32s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_32s_Sfs(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_32sc_ISfs(Npp32sc nValue, Npp32sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32sc_Sfs_Ctx(const Npp32sc *pSrc, Npp32sc nValue,
                                Npp32sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddC_32sc_Sfs(const Npp32sc *pSrc, Npp32sc nValue, Npp32sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32f_I_Ctx(Npp32f nValue, Npp32f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_32f_I(Npp32f nValue, Npp32f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32f_Ctx(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32fc_I_Ctx(Npp32fc nValue, Npp32fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_32fc_I(Npp32fc nValue, Npp32fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_32fc_Ctx(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_32fc(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_64f_I_Ctx(Npp64f nValue, Npp64f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_64f_I(Npp64f nValue, Npp64f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_64f_Ctx(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_64f(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_64fc_I_Ctx(Npp64fc nValue, Npp64fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_64fc_I(Npp64fc nValue, Npp64fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddC_64fc_Ctx(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddC_64fc(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProductC_32f_Ctx(const Npp32f *pSrc, Npp32f nValue,
                                  Npp32f *pDst, lupine_npp_length_t nLength,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddProductC_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                              lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_8u_ISfs_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_8u_ISfs(Npp8u nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_8u_Sfs(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_16u_ISfs_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_16u_ISfs(Npp16u nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_16u_Sfs(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_16s_ISfs_Ctx(Npp16s nValue, Npp16s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_16s_ISfs(Npp16s nValue, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_16s_Sfs(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_16sc_ISfs(Npp16sc nValue, Npp16sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc nValue,
                                Npp16sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_16sc_Sfs(const Npp16sc *pSrc, Npp16sc nValue, Npp16sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32s_ISfs_Ctx(Npp32s nValue, Npp32s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_32s_ISfs(Npp32s nValue, Npp32s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_32s_Sfs(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_32sc_ISfs(Npp32sc nValue, Npp32sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32sc_Sfs_Ctx(const Npp32sc *pSrc, Npp32sc nValue,
                                Npp32sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_32sc_Sfs(const Npp32sc *pSrc, Npp32sc nValue, Npp32sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32f_I_Ctx(Npp32f nValue, Npp32f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_32f_I(Npp32f nValue, Npp32f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32f_Ctx(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_Low_32f16s_Ctx(const Npp32f *pSrc, Npp32f nValue,
                                  Npp16s *pDst, lupine_npp_length_t nLength,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_Low_32f16s(const Npp32f *pSrc, Npp32f nValue, Npp16s *pDst,
                              lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32f16s_Sfs_Ctx(const Npp32f *pSrc, Npp32f nValue,
                                  Npp16s *pDst, lupine_npp_length_t nLength,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_32f16s_Sfs(const Npp32f *pSrc, Npp32f nValue, Npp16s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32fc_I_Ctx(Npp32fc nValue, Npp32fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_32fc_I(Npp32fc nValue, Npp32fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_32fc_Ctx(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_32fc(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_64f_I_Ctx(Npp64f nValue, Npp64f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_64f_I(Npp64f nValue, Npp64f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_64f_Ctx(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_64f(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_64f64s_ISfs_Ctx(Npp64f nValue, Npp64s *pDst,
                                   lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMulC_64f64s_ISfs(Npp64f nValue, Npp64s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_64fc_I_Ctx(Npp64fc nValue, Npp64fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_64fc_I(Npp64fc nValue, Npp64fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMulC_64fc_Ctx(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMulC_64fc(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_8u_ISfs_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_8u_ISfs(Npp8u nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_8u_Sfs(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_16u_ISfs_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_16u_ISfs(Npp16u nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_16u_Sfs(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_16s_ISfs_Ctx(Npp16s nValue, Npp16s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_16s_ISfs(Npp16s nValue, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_16s_Sfs(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_16sc_ISfs(Npp16sc nValue, Npp16sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc nValue,
                                Npp16sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_16sc_Sfs(const Npp16sc *pSrc, Npp16sc nValue, Npp16sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32s_ISfs_Ctx(Npp32s nValue, Npp32s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_32s_ISfs(Npp32s nValue, Npp32s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_32s_Sfs(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_32sc_ISfs(Npp32sc nValue, Npp32sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32sc_Sfs_Ctx(const Npp32sc *pSrc, Npp32sc nValue,
                                Npp32sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubC_32sc_Sfs(const Npp32sc *pSrc, Npp32sc nValue, Npp32sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32f_I_Ctx(Npp32f nValue, Npp32f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_32f_I(Npp32f nValue, Npp32f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32f_Ctx(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32fc_I_Ctx(Npp32fc nValue, Npp32fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_32fc_I(Npp32fc nValue, Npp32fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_32fc_Ctx(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_32fc(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_64f_I_Ctx(Npp64f nValue, Npp64f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_64f_I(Npp64f nValue, Npp64f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_64f_Ctx(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_64f(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_64fc_I_Ctx(Npp64fc nValue, Npp64fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_64fc_I(Npp64fc nValue, Npp64fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubC_64fc_Ctx(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubC_64fc(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_8u_ISfs_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                                  lupine_npp_length_t nLength, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_8u_ISfs(Npp8u nValue, Npp8u *pSrcDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_8u_Sfs(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_16u_ISfs_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                                   lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_16u_ISfs(Npp16u nValue, Npp16u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u nValue,
                                  Npp16u *pDst, lupine_npp_length_t nLength,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_16u_Sfs(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_16s_ISfs_Ctx(Npp16s nValue, Npp16s *pSrcDst,
                                   lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_16s_ISfs(Npp16s nValue, Npp16s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s nValue,
                                  Npp16s *pDst, lupine_npp_length_t nLength,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_16s_Sfs(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc *pSrcDst,
                                    lupine_npp_length_t nLength,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_16sc_ISfs(Npp16sc nValue, Npp16sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc nValue,
                                   Npp16sc *pDst, lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_16sc_Sfs(const Npp16sc *pSrc, Npp16sc nValue,
                               Npp16sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32s_ISfs_Ctx(Npp32s nValue, Npp32s *pSrcDst,
                                   lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_32s_ISfs(Npp32s nValue, Npp32s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s nValue,
                                  Npp32s *pDst, lupine_npp_length_t nLength,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_32s_Sfs(const Npp32s *pSrc, Npp32s nValue, Npp32s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32sc_ISfs_Ctx(Npp32sc nValue, Npp32sc *pSrcDst,
                                    lupine_npp_length_t nLength,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_32sc_ISfs(Npp32sc nValue, Npp32sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32sc_Sfs_Ctx(const Npp32sc *pSrc, Npp32sc nValue,
                                   Npp32sc *pDst, lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSubCRev_32sc_Sfs(const Npp32sc *pSrc, Npp32sc nValue,
                               Npp32sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32f_I_Ctx(Npp32f nValue, Npp32f *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_32f_I(Npp32f nValue, Npp32f *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32f_Ctx(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32fc_I_Ctx(Npp32fc nValue, Npp32fc *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_32fc_I(Npp32fc nValue, Npp32fc *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_32fc_Ctx(const Npp32fc *pSrc, Npp32fc nValue,
                               Npp32fc *pDst, lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_32fc(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                           lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_64f_I_Ctx(Npp64f nValue, Npp64f *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_64f_I(Npp64f nValue, Npp64f *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_64f_Ctx(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_64f(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_64fc_I_Ctx(Npp64fc nValue, Npp64fc *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_64fc_I(Npp64fc nValue, Npp64fc *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSubCRev_64fc_Ctx(const Npp64fc *pSrc, Npp64fc nValue,
                               Npp64fc *pDst, lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSubCRev_64fc(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                           lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_8u_ISfs_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_8u_ISfs(Npp8u nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_8u_Sfs(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_16u_ISfs_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_16u_ISfs(Npp16u nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_16u_Sfs(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_16s_ISfs_Ctx(Npp16s nValue, Npp16s *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_16s_ISfs(Npp16s nValue, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_16s_Sfs(const Npp16s *pSrc, Npp16s nValue, Npp16s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_16sc_ISfs_Ctx(Npp16sc nValue, Npp16sc *pSrcDst,
                                 lupine_npp_length_t nLength, int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_16sc_ISfs(Npp16sc nValue, Npp16sc *pSrcDst,
                             lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc nValue,
                                Npp16sc *pDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDivC_16sc_Sfs(const Npp16sc *pSrc, Npp16sc nValue, Npp16sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_32f_I_Ctx(Npp32f nValue, Npp32f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_32f_I(Npp32f nValue, Npp32f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_32f_Ctx(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_32fc_I_Ctx(Npp32fc nValue, Npp32fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_32fc_I(Npp32fc nValue, Npp32fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_32fc_Ctx(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_32fc(const Npp32fc *pSrc, Npp32fc nValue, Npp32fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_64f_I_Ctx(Npp64f nValue, Npp64f *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_64f_I(Npp64f nValue, Npp64f *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_64f_Ctx(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_64f(const Npp64f *pSrc, Npp64f nValue, Npp64f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_64fc_I_Ctx(Npp64fc nValue, Npp64fc *pSrcDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_64fc_I(Npp64fc nValue, Npp64fc *pSrcDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivC_64fc_Ctx(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivC_64fc(const Npp64fc *pSrc, Npp64fc nValue, Npp64fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivCRev_16u_I_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivCRev_16u_I(Npp16u nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivCRev_16u_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivCRev_16u(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivCRev_32f_I_Ctx(Npp32f nValue, Npp32f *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivCRev_32f_I(Npp32f nValue, Npp32f *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDivCRev_32f_Ctx(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDivCRev_32f(const Npp32f *pSrc, Npp32f nValue, Npp32f *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_16s(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16u_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_16u(const Npp16u *pSrc1, const Npp16u *pSrc2, Npp16u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32u_Ctx(const Npp32u *pSrc1, const Npp32u *pSrc2,
                          Npp32u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_32u(const Npp32u *pSrc1, const Npp32u *pSrc2, Npp32u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                          Npp32f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_32f(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                          Npp64f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_64f(const Npp64f *pSrc1, const Npp64f *pSrc2, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                           Npp32fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                       Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                           Npp64fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                       Npp64fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_8u16u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                            Npp16u *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_8u16u(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp16u *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                             Npp32f *pDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp32f *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_8u_Sfs_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                             Npp8u *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_8u_Sfs(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16u_Sfs_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                              Npp16u *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_16u_Sfs(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                              Npp16s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                              Npp32s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                          Npp32s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_64s_Sfs_Ctx(const Npp64s *pSrc1, const Npp64s *pSrc2,
                              Npp64s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_64s_Sfs(const Npp64s *pSrc1, const Npp64s *pSrc2,
                          Npp64s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16sc_Sfs_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               Npp16sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_16sc_Sfs(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                           Npp16sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32sc_Sfs_Ctx(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                               Npp32sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_32sc_Sfs(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                           Npp32sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16s_I_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_16s_I(const Npp16s *pSrc, Npp16s *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32f_I_Ctx(const Npp32f *pSrc, Npp32f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_32f_I(const Npp32f *pSrc, Npp32f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_64f_I_Ctx(const Npp64f *pSrc, Npp64f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_64f_I(const Npp64f *pSrc, Npp64f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32fc_I_Ctx(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_32fc_I(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_64fc_I_Ctx(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_64fc_I(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16s32s_I_Ctx(const Npp16s *pSrc, Npp32s *pSrcDst,
                               lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAdd_16s32s_I(const Npp16s *pSrc, Npp32s *pSrcDst,
                           lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_8u_ISfs_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_8u_ISfs(const Npp8u *pSrc, Npp8u *pSrcDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16u_ISfs_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_16u_ISfs(const Npp16u *pSrc, Npp16u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16s_ISfs_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_16s_ISfs(const Npp16s *pSrc, Npp16s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32s_ISfs_Ctx(const Npp32s *pSrc, Npp32s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_32s_ISfs(const Npp32s *pSrc, Npp32s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_16sc_ISfs_Ctx(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_16sc_ISfs(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAdd_32sc_ISfs_Ctx(const Npp32sc *pSrc, Npp32sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAdd_32sc_ISfs(const Npp32sc *pSrc, Npp32sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                 Npp32f *pDst, lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddProduct_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                             Npp32f *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                 Npp64f *pDst, lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddProduct_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                             Npp64f *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                  Npp32fc *pDst, lupine_npp_length_t nLength,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddProduct_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                              Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                  Npp64fc *pDst, lupine_npp_length_t nLength,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAddProduct_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                              Npp64fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     Npp16s *pDst, lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddProduct_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 Npp16s *pDst, lupine_npp_length_t nLength,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                     Npp32s *pDst, lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddProduct_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                 Npp32s *pDst, lupine_npp_length_t nLength,
                                 int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAddProduct_16s32s_Sfs_Ctx(const Npp16s *pSrc1,
                                        const Npp16s *pSrc2, Npp32s *pDst,
                                        lupine_npp_length_t nLength,
                                        int nScaleFactor,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsAddProduct_16s32s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                    Npp32s *pDst, lupine_npp_length_t nLength,
                                    int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_16s(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                          Npp32f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_32f(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                          Npp64f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_64f(const Npp64f *pSrc1, const Npp64f *pSrc2, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                           Npp32fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                       Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                           Npp64fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                       Npp64fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_8u16u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                            Npp16u *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_8u16u(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp16u *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                             Npp32f *pDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp32f *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32f32fc_Ctx(const Npp32f *pSrc1, const Npp32fc *pSrc2,
                              Npp32fc *pDst, lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_32f32fc(const Npp32f *pSrc1, const Npp32fc *pSrc2,
                          Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_8u_Sfs_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                             Npp8u *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_8u_Sfs(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16u_Sfs_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                              Npp16u *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16u_Sfs(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                              Npp16s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                              Npp32s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                          Npp32s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16sc_Sfs_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               Npp16sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16sc_Sfs(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                           Npp16sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32sc_Sfs_Ctx(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                               Npp32sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_32sc_Sfs(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                           Npp32sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16u16s_Sfs_Ctx(const Npp16u *pSrc1, const Npp16s *pSrc2,
                                 Npp16s *pDst, lupine_npp_length_t nLength,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16u16s_Sfs(const Npp16u *pSrc1, const Npp16s *pSrc2,
                             Npp16s *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16s32s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 Npp32s *pDst, lupine_npp_length_t nLength,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16s32s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                             Npp32s *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32s32sc_Sfs_Ctx(const Npp32s *pSrc1, const Npp32sc *pSrc2,
                                  Npp32sc *pDst, lupine_npp_length_t nLength,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_32s32sc_Sfs(const Npp32s *pSrc1, const Npp32sc *pSrc2,
                              Npp32sc *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_Low_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                  Npp32s *pDst, lupine_npp_length_t nLength,
                                  int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_Low_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                              Npp32s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16s_I_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_16s_I(const Npp16s *pSrc, Npp16s *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32f_I_Ctx(const Npp32f *pSrc, Npp32f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_32f_I(const Npp32f *pSrc, Npp32f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_64f_I_Ctx(const Npp64f *pSrc, Npp64f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_64f_I(const Npp64f *pSrc, Npp64f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32fc_I_Ctx(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_32fc_I(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_64fc_I_Ctx(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_64fc_I(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32f32fc_I_Ctx(const Npp32f *pSrc, Npp32fc *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMul_32f32fc_I(const Npp32f *pSrc, Npp32fc *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_8u_ISfs_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_8u_ISfs(const Npp8u *pSrc, Npp8u *pSrcDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16u_ISfs_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16u_ISfs(const Npp16u *pSrc, Npp16u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16s_ISfs_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16s_ISfs(const Npp16s *pSrc, Npp16s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32s_ISfs_Ctx(const Npp32s *pSrc, Npp32s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_32s_ISfs(const Npp32s *pSrc, Npp32s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_16sc_ISfs_Ctx(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_16sc_ISfs(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32sc_ISfs_Ctx(const Npp32sc *pSrc, Npp32sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_32sc_ISfs(const Npp32sc *pSrc, Npp32sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMul_32s32sc_ISfs_Ctx(const Npp32s *pSrc, Npp32sc *pSrcDst,
                                   lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsMul_32s32sc_ISfs(const Npp32s *pSrc, Npp32sc *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_16s(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp16s *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                          Npp32f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_32f(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                          Npp64f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_64f(const Npp64f *pSrc1, const Npp64f *pSrc2, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                           Npp32fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                       Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                           Npp64fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                       Npp64fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                             Npp32f *pDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2, Npp32f *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_8u_Sfs_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                             Npp8u *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_8u_Sfs(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16u_Sfs_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                              Npp16u *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_16u_Sfs(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                              Npp16s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                              Npp32s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                          Npp32s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16sc_Sfs_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               Npp16sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_16sc_Sfs(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                           Npp16sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32sc_Sfs_Ctx(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                               Npp32sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_32sc_Sfs(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                           Npp32sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16s_I_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_16s_I(const Npp16s *pSrc, Npp16s *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32f_I_Ctx(const Npp32f *pSrc, Npp32f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_32f_I(const Npp32f *pSrc, Npp32f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_64f_I_Ctx(const Npp64f *pSrc, Npp64f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_64f_I(const Npp64f *pSrc, Npp64f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32fc_I_Ctx(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_32fc_I(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_64fc_I_Ctx(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSub_64fc_I(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_8u_ISfs_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_8u_ISfs(const Npp8u *pSrc, Npp8u *pSrcDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16u_ISfs_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_16u_ISfs(const Npp16u *pSrc, Npp16u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16s_ISfs_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_16s_ISfs(const Npp16s *pSrc, Npp16s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32s_ISfs_Ctx(const Npp32s *pSrc, Npp32s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_32s_ISfs(const Npp32s *pSrc, Npp32s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_16sc_ISfs_Ctx(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_16sc_ISfs(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSub_32sc_ISfs_Ctx(const Npp32sc *pSrc, Npp32sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSub_32sc_ISfs(const Npp32sc *pSrc, Npp32sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_8u_Sfs_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                             Npp8u *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_8u_Sfs(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_16u_Sfs_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                              Npp16u *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_16u_Sfs(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                              Npp16s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                          Npp16s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                              Npp32s *pDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                          Npp32s *pDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_16sc_Sfs_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               Npp16sc *pDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_16sc_Sfs(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                           Npp16sc *pDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32s16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp32s *pSrc2,
                                 Npp16s *pDst, lupine_npp_length_t nLength,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_32s16s_Sfs(const Npp16s *pSrc1, const Npp32s *pSrc2,
                             Npp16s *pDst, lupine_npp_length_t nLength,
                             int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                          Npp32f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_32f(const Npp32f *pSrc1, const Npp32f *pSrc2, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                          Npp64f *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_64f(const Npp64f *pSrc1, const Npp64f *pSrc2, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                           Npp32fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                       Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                           Npp64fc *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                       Npp64fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_8u_ISfs_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_8u_ISfs(const Npp8u *pSrc, Npp8u *pSrcDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_16u_ISfs_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_16u_ISfs(const Npp16u *pSrc, Npp16u *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_16s_ISfs_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_16s_ISfs(const Npp16s *pSrc, Npp16s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_16sc_ISfs_Ctx(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_16sc_ISfs(const Npp16sc *pSrc, Npp16sc *pSrcDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32s_ISfs_Ctx(const Npp32s *pSrc, Npp32s *pSrcDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_32s_ISfs(const Npp32s *pSrc, Npp32s *pSrcDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32f_I_Ctx(const Npp32f *pSrc, Npp32f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_32f_I(const Npp32f *pSrc, Npp32f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_64f_I_Ctx(const Npp64f *pSrc, Npp64f *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_64f_I(const Npp64f *pSrc, Npp64f *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_32fc_I_Ctx(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_32fc_I(const Npp32fc *pSrc, Npp32fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_64fc_I_Ctx(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsDiv_64fc_I(const Npp64fc *pSrc, Npp64fc *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_Round_8u_Sfs_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                                   Npp8u *pDst, lupine_npp_length_t nLength,
                                   NppRoundMode nRndMode, int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_Round_8u_Sfs(const Npp8u *pSrc1, const Npp8u *pSrc2,
                               Npp8u *pDst, lupine_npp_length_t nLength,
                               NppRoundMode nRndMode, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_Round_16u_Sfs_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                                    Npp16u *pDst, lupine_npp_length_t nLength,
                                    NppRoundMode nRndMode, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_Round_16u_Sfs(const Npp16u *pSrc1, const Npp16u *pSrc2,
                                Npp16u *pDst, lupine_npp_length_t nLength,
                                NppRoundMode nRndMode, int nScaleFactor);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_Round_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                    Npp16s *pDst, lupine_npp_length_t nLength,
                                    NppRoundMode nRndMode, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_Round_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                Npp16s *pDst, lupine_npp_length_t nLength,
                                NppRoundMode nRndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_Round_8u_ISfs_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                                    lupine_npp_length_t nLength,
                                    NppRoundMode nRndMode, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_Round_8u_ISfs(const Npp8u *pSrc, Npp8u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppRoundMode nRndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_Round_16u_ISfs_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode nRndMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_Round_16u_ISfs(const Npp16u *pSrc, Npp16u *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode nRndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDiv_Round_16s_ISfs_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode nRndMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nRndMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsDiv_Round_16s_ISfs(const Npp16s *pSrc, Npp16s *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode nRndMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_16s(const Npp16s *pSrc, Npp16s *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_32s_Ctx(const Npp32s *pSrc, Npp32s *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_32s(const Npp32s *pSrc, Npp32s *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_32f(const Npp32f *pSrc, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_64f(const Npp64f *pSrc, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_16s_I_Ctx(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_16s_I(Npp16s *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_32s_I_Ctx(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_32s_I(Npp32s *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAbs_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAbs_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_32f(const Npp32f *pSrc, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_64f(const Npp64f *pSrc, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_32fc_I_Ctx(Npp32fc *pSrcDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_32fc_I(Npp32fc *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_64fc_I_Ctx(Npp64fc *pSrcDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqr_64fc_I(Npp64fc *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u *pDst,
                             lupine_npp_length_t nLength, int nScaleFactor,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_8u_Sfs(const Npp8u *pSrc, Npp8u *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_16u_Sfs(const Npp16u *pSrc, Npp16u *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_16s_Sfs(const Npp16s *pSrc, Npp16s *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_16sc_Sfs(const Npp16sc *pSrc, Npp16sc *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_8u_ISfs_Ctx(Npp8u *pSrcDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_8u_ISfs(Npp8u *pSrcDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_16u_ISfs_Ctx(Npp16u *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_16u_ISfs(Npp16u *pSrcDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_16s_ISfs_Ctx(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_16s_ISfs(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqr_16sc_ISfs_Ctx(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqr_16sc_ISfs(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                            int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_32f(const Npp32f *pSrc, Npp32f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_64f(const Npp64f *pSrc, Npp64f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_32fc_I_Ctx(Npp32fc *pSrcDst, lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_32fc_I(Npp32fc *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64fc_I_Ctx(Npp64fc *pSrcDst, lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSqrt_64fc_I(Npp64fc *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_8u_Sfs_Ctx(const Npp8u *pSrc, Npp8u *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_8u_Sfs(const Npp8u *pSrc, Npp8u *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_16u_Sfs_Ctx(const Npp16u *pSrc, Npp16u *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_16u_Sfs(const Npp16u *pSrc, Npp16u *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_16s_Sfs(const Npp16s *pSrc, Npp16s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_16sc_Sfs(const Npp16sc *pSrc, Npp16sc *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64s_Sfs_Ctx(const Npp64s *pSrc, Npp64s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_64s_Sfs(const Npp64s *pSrc, Npp64s *pDst,
                           lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_32s16s_Sfs_Ctx(const Npp32s *pSrc, Npp16s *pDst,
                                  lupine_npp_length_t nLength, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_32s16s_Sfs(const Npp32s *pSrc, Npp16s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64s16s_Sfs_Ctx(const Npp64s *pSrc, Npp16s *pDst,
                                  lupine_npp_length_t nLength, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_64s16s_Sfs(const Npp64s *pSrc, Npp16s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_8u_ISfs_Ctx(Npp8u *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_8u_ISfs(Npp8u *pSrcDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_16u_ISfs_Ctx(Npp16u *pSrcDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_16u_ISfs(Npp16u *pSrcDst, lupine_npp_length_t nLength,
                            int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_16s_ISfs_Ctx(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_16s_ISfs(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                            int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_16sc_ISfs_Ctx(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                                 int nScaleFactor,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_16sc_ISfs(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                             int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSqrt_64s_ISfs_Ctx(Npp64s *pSrcDst, lupine_npp_length_t nLength,
                                int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsSqrt_64s_ISfs(Npp64s *pSrcDst, lupine_npp_length_t nLength,
                            int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCubrt_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCubrt_32f(const Npp32f *pSrc, Npp32f *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCubrt_32s16s_Sfs_Ctx(const Npp32s *pSrc, Npp16s *pDst,
                                   lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsCubrt_32s16s_Sfs(const Npp32s *pSrc, Npp16s *pDst,
                               lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsExp_32f(const Npp32f *pSrc, Npp32f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsExp_64f(const Npp64f *pSrc, Npp64f *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_32f64f_Ctx(const Npp32f *pSrc, Npp64f *pDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsExp_32f64f(const Npp32f *pSrc, Npp64f *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsExp_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsExp_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsExp_16s_Sfs(const Npp16s *pSrc, Npp16s *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsExp_32s_Sfs(const Npp32s *pSrc, Npp32s *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_64s_Sfs_Ctx(const Npp64s *pSrc, Npp64s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsExp_64s_Sfs(const Npp64s *pSrc, Npp64s *pDst,
                          lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_16s_ISfs_Ctx(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsExp_16s_ISfs(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_32s_ISfs_Ctx(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsExp_32s_ISfs(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsExp_64s_ISfs_Ctx(Npp64s *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsExp_64s_ISfs(Npp64s *pSrcDst, lupine_npp_length_t nLength,
                           int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLn_32f(const Npp32f *pSrc, Npp32f *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLn_64f(const Npp64f *pSrc, Npp64f *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_64f32f_Ctx(const Npp64f *pSrc, Npp32f *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLn_64f32f(const Npp64f *pSrc, Npp32f *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLn_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLn_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                             lupine_npp_length_t nLength, int nScaleFactor,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsLn_16s_Sfs(const Npp16s *pSrc, Npp16s *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s *pDst,
                             lupine_npp_length_t nLength, int nScaleFactor,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsLn_32s_Sfs(const Npp32s *pSrc, Npp32s *pDst,
                         lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_32s16s_Sfs_Ctx(const Npp32s *pSrc, Npp16s *pDst,
                                lupine_npp_length_t nLength, int nScaleFactor,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsLn_32s16s_Sfs(const Npp32s *pSrc, Npp16s *pDst,
                            lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_16s_ISfs_Ctx(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsLn_16s_ISfs(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLn_32s_ISfs_Ctx(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                              int nScaleFactor, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsLn_32s_ISfs(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                          int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus npps10Log10_32s_Sfs_Ctx(const Npp32s *pSrc, Npp32s *pDst,
                                  lupine_npp_length_t nLength, int nScaleFactor,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus npps10Log10_32s_Sfs(const Npp32s *pSrc, Npp32s *pDst,
                              lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus npps10Log10_32s_ISfs_Ctx(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                                   int nScaleFactor,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus npps10Log10_32s_ISfs(Npp32s *pSrcDst, lupine_npp_length_t nLength,
                               int nScaleFactor);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLnGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumLnGetBufferSize_32f(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLn_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                            Npp32f *pDst, Npp8u *pDeviceBuffer,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSumLn_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                        Npp32f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLnGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumLnGetBufferSize_64f(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLn_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSumLn_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLnGetBufferSize_32f64f_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumLnGetBufferSize_32f64f(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLn_32f64f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                               Npp64f *pDst, Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSumLn_32f64f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                           Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLnGetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumLnGetBufferSize_16s32f(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumLn_16s32f_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                               Npp32f *pDst, Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSumLn_16s32f(const Npp16s *pSrc, lupine_npp_length_t nLength,
                           Npp32f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsArctan_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsArctan_32f(const Npp32f *pSrc, Npp32f *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsArctan_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsArctan_64f(const Npp64f *pSrc, Npp64f *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsArctan_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsArctan_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsArctan_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsArctan_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormalize_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                                lupine_npp_length_t nLength, Npp32f vSub,
                                Npp32f vDiv, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 */
NppStatus nppsNormalize_32f(const Npp32f *pSrc, Npp32f *pDst,
                            lupine_npp_length_t nLength, Npp32f vSub,
                            Npp32f vDiv);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormalize_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                                 lupine_npp_length_t nLength, Npp32fc vSub,
                                 Npp32f vDiv, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 */
NppStatus nppsNormalize_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                             lupine_npp_length_t nLength, Npp32fc vSub,
                             Npp32f vDiv);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormalize_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                                lupine_npp_length_t nLength, Npp64f vSub,
                                Npp64f vDiv, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 */
NppStatus nppsNormalize_64f(const Npp64f *pSrc, Npp64f *pDst,
                            lupine_npp_length_t nLength, Npp64f vSub,
                            Npp64f vDiv);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormalize_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                                 lupine_npp_length_t nLength, Npp64fc vSub,
                                 Npp64f vDiv, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 */
NppStatus nppsNormalize_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                             lupine_npp_length_t nLength, Npp64fc vSub,
                             Npp64f vDiv);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormalize_16s_Sfs_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                                    lupine_npp_length_t nLength, Npp16s vSub,
                                    int vDiv, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsNormalize_16s_Sfs(const Npp16s *pSrc, Npp16s *pDst,
                                lupine_npp_length_t nLength, Npp16s vSub,
                                int vDiv, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormalize_16sc_Sfs_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                     lupine_npp_length_t nLength, Npp16sc vSub,
                                     int vDiv, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param vSub SEND_ONLY
 * @param vDiv SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsNormalize_16sc_Sfs(const Npp16sc *pSrc, Npp16sc *pDst,
                                 lupine_npp_length_t nLength, Npp16sc vSub,
                                 int vDiv, int nScaleFactor);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nParam SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCauchy_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                               Npp32f nParam, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nParam SEND_ONLY
 */
NppStatus nppsCauchy_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                           Npp32f nParam);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nParam SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCauchyD_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                                Npp32f nParam, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nParam SEND_ONLY
 */
NppStatus nppsCauchyD_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                            Npp32f nParam);
/**
 * @param pSrcDst SEND_ONLY
 * @param pD2FVal SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nParam SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCauchyDD2_32f_I_Ctx(Npp32f *pSrcDst, Npp32f *pD2FVal,
                                  lupine_npp_length_t nLength, Npp32f nParam,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param pD2FVal SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nParam SEND_ONLY
 */
NppStatus nppsCauchyDD2_32f_I(Npp32f *pSrcDst, Npp32f *pD2FVal,
                              lupine_npp_length_t nLength, Npp32f nParam);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAndC_8u_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAndC_8u(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAndC_16u_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAndC_16u(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAndC_32u_Ctx(const Npp32u *pSrc, Npp32u nValue, Npp32u *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAndC_32u(const Npp32u *pSrc, Npp32u nValue, Npp32u *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAndC_8u_I_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAndC_8u_I(Npp8u nValue, Npp8u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAndC_16u_I_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAndC_16u_I(Npp16u nValue, Npp16u *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAndC_32u_I_Ctx(Npp32u nValue, Npp32u *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAndC_32u_I(Npp32u nValue, Npp32u *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAnd_8u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAnd_8u(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAnd_16u_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAnd_16u(const Npp16u *pSrc1, const Npp16u *pSrc2, Npp16u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAnd_32u_Ctx(const Npp32u *pSrc1, const Npp32u *pSrc2,
                          Npp32u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAnd_32u(const Npp32u *pSrc1, const Npp32u *pSrc2, Npp32u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAnd_8u_I_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAnd_8u_I(const Npp8u *pSrc, Npp8u *pSrcDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAnd_16u_I_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAnd_16u_I(const Npp16u *pSrc, Npp16u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAnd_32u_I_Ctx(const Npp32u *pSrc, Npp32u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsAnd_32u_I(const Npp32u *pSrc, Npp32u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOrC_8u_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOrC_8u(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOrC_16u_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOrC_16u(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOrC_32u_Ctx(const Npp32u *pSrc, Npp32u nValue, Npp32u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOrC_32u(const Npp32u *pSrc, Npp32u nValue, Npp32u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOrC_8u_I_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOrC_8u_I(Npp8u nValue, Npp8u *pSrcDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOrC_16u_I_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOrC_16u_I(Npp16u nValue, Npp16u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOrC_32u_I_Ctx(Npp32u nValue, Npp32u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOrC_32u_I(Npp32u nValue, Npp32u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOr_8u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                        lupine_npp_length_t nLength,
                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOr_8u(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                    lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOr_16u_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2, Npp16u *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOr_16u(const Npp16u *pSrc1, const Npp16u *pSrc2, Npp16u *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOr_32u_Ctx(const Npp32u *pSrc1, const Npp32u *pSrc2, Npp32u *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOr_32u(const Npp32u *pSrc1, const Npp32u *pSrc2, Npp32u *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOr_8u_I_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOr_8u_I(const Npp8u *pSrc, Npp8u *pSrcDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOr_16u_I_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOr_16u_I(const Npp16u *pSrc, Npp16u *pSrcDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsOr_32u_I_Ctx(const Npp32u *pSrc, Npp32u *pSrcDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsOr_32u_I(const Npp32u *pSrc, Npp32u *pSrcDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXorC_8u_Ctx(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXorC_8u(const Npp8u *pSrc, Npp8u nValue, Npp8u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXorC_16u_Ctx(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXorC_16u(const Npp16u *pSrc, Npp16u nValue, Npp16u *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXorC_32u_Ctx(const Npp32u *pSrc, Npp32u nValue, Npp32u *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXorC_32u(const Npp32u *pSrc, Npp32u nValue, Npp32u *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXorC_8u_I_Ctx(Npp8u nValue, Npp8u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXorC_8u_I(Npp8u nValue, Npp8u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXorC_16u_I_Ctx(Npp16u nValue, Npp16u *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXorC_16u_I(Npp16u nValue, Npp16u *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXorC_32u_I_Ctx(Npp32u nValue, Npp32u *pSrcDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXorC_32u_I(Npp32u nValue, Npp32u *pSrcDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXor_8u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXor_8u(const Npp8u *pSrc1, const Npp8u *pSrc2, Npp8u *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXor_16u_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                          Npp16u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXor_16u(const Npp16u *pSrc1, const Npp16u *pSrc2, Npp16u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXor_32u_Ctx(const Npp32u *pSrc1, const Npp32u *pSrc2,
                          Npp32u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXor_32u(const Npp32u *pSrc1, const Npp32u *pSrc2, Npp32u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXor_8u_I_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXor_8u_I(const Npp8u *pSrc, Npp8u *pSrcDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXor_16u_I_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXor_16u_I(const Npp16u *pSrc, Npp16u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsXor_32u_I_Ctx(const Npp32u *pSrc, Npp32u *pSrcDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsXor_32u_I(const Npp32u *pSrc, Npp32u *pSrcDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNot_8u_Ctx(const Npp8u *pSrc, Npp8u *pDst,
                         lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsNot_8u(const Npp8u *pSrc, Npp8u *pDst,
                     lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNot_16u_Ctx(const Npp16u *pSrc, Npp16u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsNot_16u(const Npp16u *pSrc, Npp16u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNot_32u_Ctx(const Npp32u *pSrc, Npp32u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsNot_32u(const Npp32u *pSrc, Npp32u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNot_8u_I_Ctx(Npp8u *pSrcDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsNot_8u_I(Npp8u *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNot_16u_I_Ctx(Npp16u *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsNot_16u_I(Npp16u *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNot_32u_I_Ctx(Npp32u *pSrcDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsNot_32u_I(Npp32u *pSrcDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_8u_Ctx(const Npp8u *pSrc, int nValue, Npp8u *pDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_8u(const Npp8u *pSrc, int nValue, Npp8u *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_16u_Ctx(const Npp16u *pSrc, int nValue, Npp16u *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_16u(const Npp16u *pSrc, int nValue, Npp16u *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_16s_Ctx(const Npp16s *pSrc, int nValue, Npp16s *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_16s(const Npp16s *pSrc, int nValue, Npp16s *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_32u_Ctx(const Npp32u *pSrc, int nValue, Npp32u *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_32u(const Npp32u *pSrc, int nValue, Npp32u *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_32s_Ctx(const Npp32s *pSrc, int nValue, Npp32s *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_32s(const Npp32s *pSrc, int nValue, Npp32s *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_8u_I_Ctx(int nValue, Npp8u *pSrcDst,
                               lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_8u_I(int nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_16u_I_Ctx(int nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_16u_I(int nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_16s_I_Ctx(int nValue, Npp16s *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_16s_I(int nValue, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_32u_I_Ctx(int nValue, Npp32u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_32u_I(int nValue, Npp32u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsLShiftC_32s_I_Ctx(int nValue, Npp32s *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsLShiftC_32s_I(int nValue, Npp32s *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_8u_Ctx(const Npp8u *pSrc, int nValue, Npp8u *pDst,
                             lupine_npp_length_t nLength,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_8u(const Npp8u *pSrc, int nValue, Npp8u *pDst,
                         lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_16u_Ctx(const Npp16u *pSrc, int nValue, Npp16u *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_16u(const Npp16u *pSrc, int nValue, Npp16u *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_16s_Ctx(const Npp16s *pSrc, int nValue, Npp16s *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_16s(const Npp16s *pSrc, int nValue, Npp16s *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_32u_Ctx(const Npp32u *pSrc, int nValue, Npp32u *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_32u(const Npp32u *pSrc, int nValue, Npp32u *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_32s_Ctx(const Npp32s *pSrc, int nValue, Npp32s *pDst,
                              lupine_npp_length_t nLength,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_32s(const Npp32s *pSrc, int nValue, Npp32s *pDst,
                          lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_8u_I_Ctx(int nValue, Npp8u *pSrcDst,
                               lupine_npp_length_t nLength,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_8u_I(int nValue, Npp8u *pSrcDst,
                           lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_16u_I_Ctx(int nValue, Npp16u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_16u_I(int nValue, Npp16u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_16s_I_Ctx(int nValue, Npp16s *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_16s_I(int nValue, Npp16s *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_32u_I_Ctx(int nValue, Npp32u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_32u_I(int nValue, Npp32u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsRShiftC_32s_I_Ctx(int nValue, Npp32s *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsRShiftC_32s_I(int nValue, Npp32s *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_8s16s_Ctx(const Npp8s *pSrc, Npp16s *pDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_8s16s(const Npp8s *pSrc, Npp16s *pDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_8s32f_Ctx(const Npp8s *pSrc, Npp32f *pDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_8s32f(const Npp8s *pSrc, Npp32f *pDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_8u32f_Ctx(const Npp8u *pSrc, Npp32f *pDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_8u32f(const Npp8u *pSrc, Npp32f *pDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_16s8s_Sfs_Ctx(const Npp16s *pSrc, Npp8s *pDst,
                                    Npp32u nLength, NppRoundMode eRoundMode,
                                    int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_16s8s_Sfs(const Npp16s *pSrc, Npp8s *pDst, Npp32u nLength,
                                NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_16s32s_Ctx(const Npp16s *pSrc, Npp32s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_16s32s(const Npp16s *pSrc, Npp32s *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_16s32f_Ctx(const Npp16s *pSrc, Npp32f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_16s32f(const Npp16s *pSrc, Npp32f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_16u32f_Ctx(const Npp16u *pSrc, Npp32f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_16u32f(const Npp16u *pSrc, Npp32f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32s16s_Ctx(const Npp32s *pSrc, Npp16s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_32s16s(const Npp32s *pSrc, Npp16s *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32s32f_Ctx(const Npp32s *pSrc, Npp32f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_32s32f(const Npp32s *pSrc, Npp32f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32s64f_Ctx(const Npp32s *pSrc, Npp64f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_32s64f(const Npp32s *pSrc, Npp64f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32f64f_Ctx(const Npp32f *pSrc, Npp64f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_32f64f(const Npp32f *pSrc, Npp64f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_64s64f_Ctx(const Npp64s *pSrc, Npp64f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_64s64f(const Npp64s *pSrc, Npp64f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_64f32f_Ctx(const Npp64f *pSrc, Npp32f *pDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsConvert_64f32f(const Npp64f *pSrc, Npp32f *pDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_16s32f_Sfs_Ctx(const Npp16s *pSrc, Npp32f *pDst,
                                     lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_16s32f_Sfs(const Npp16s *pSrc, Npp32f *pDst,
                                 lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_16s64f_Sfs_Ctx(const Npp16s *pSrc, Npp64f *pDst,
                                     lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_16s64f_Sfs(const Npp16s *pSrc, Npp64f *pDst,
                                 lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32s16s_Sfs_Ctx(const Npp32s *pSrc, Npp16s *pDst,
                                     lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32s16s_Sfs(const Npp32s *pSrc, Npp16s *pDst,
                                 lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32s32f_Sfs_Ctx(const Npp32s *pSrc, Npp32f *pDst,
                                     lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32s32f_Sfs(const Npp32s *pSrc, Npp32f *pDst,
                                 lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32s64f_Sfs_Ctx(const Npp32s *pSrc, Npp64f *pDst,
                                     lupine_npp_length_t nLength,
                                     int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32s64f_Sfs(const Npp32s *pSrc, Npp64f *pDst,
                                 lupine_npp_length_t nLength, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32f8s_Sfs_Ctx(const Npp32f *pSrc, Npp8s *pDst,
                                    lupine_npp_length_t nLength,
                                    NppRoundMode eRoundMode, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32f8s_Sfs(const Npp32f *pSrc, Npp8s *pDst,
                                lupine_npp_length_t nLength,
                                NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32f8u_Sfs_Ctx(const Npp32f *pSrc, Npp8u *pDst,
                                    lupine_npp_length_t nLength,
                                    NppRoundMode eRoundMode, int nScaleFactor,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32f8u_Sfs(const Npp32f *pSrc, Npp8u *pDst,
                                lupine_npp_length_t nLength,
                                NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32f16s_Sfs_Ctx(const Npp32f *pSrc, Npp16s *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32f16s_Sfs(const Npp32f *pSrc, Npp16s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32f16u_Sfs_Ctx(const Npp32f *pSrc, Npp16u *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32f16u_Sfs(const Npp32f *pSrc, Npp16u *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_32f32s_Sfs_Ctx(const Npp32f *pSrc, Npp32s *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_32f32s_Sfs(const Npp32f *pSrc, Npp32s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_64s32s_Sfs_Ctx(const Npp64s *pSrc, Npp32s *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_64s32s_Sfs(const Npp64s *pSrc, Npp32s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_64f16s_Sfs_Ctx(const Npp64f *pSrc, Npp16s *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_64f16s_Sfs(const Npp64f *pSrc, Npp16s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_64f32s_Sfs_Ctx(const Npp64f *pSrc, Npp32s *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_64f32s_Sfs(const Npp64f *pSrc, Npp32s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsConvert_64f64s_Sfs_Ctx(const Npp64f *pSrc, Npp64s *pDst,
                                     lupine_npp_length_t nLength,
                                     NppRoundMode eRoundMode, int nScaleFactor,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param eRoundMode SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 */
NppStatus nppsConvert_64f64s_Sfs(const Npp64f *pSrc, Npp64s *pDst,
                                 lupine_npp_length_t nLength,
                                 NppRoundMode eRoundMode, int nScaleFactor);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                                lupine_npp_length_t nLength, Npp16s nLevel,
                                NppCmpOp nRelOp, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_16s(const Npp16s *pSrc, Npp16s *pDst,
                            lupine_npp_length_t nLength, Npp16s nLevel,
                            NppCmpOp nRelOp);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_16s_I_Ctx(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                                  Npp16s nLevel, NppCmpOp nRelOp,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_16s_I(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                              Npp16s nLevel, NppCmpOp nRelOp);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_16sc_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                 lupine_npp_length_t nLength, Npp16s nLevel,
                                 NppCmpOp nRelOp,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_16sc(const Npp16sc *pSrc, Npp16sc *pDst,
                             lupine_npp_length_t nLength, Npp16s nLevel,
                             NppCmpOp nRelOp);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_16sc_I_Ctx(Npp16sc *pSrcDst,
                                   lupine_npp_length_t nLength, Npp16s nLevel,
                                   NppCmpOp nRelOp,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_16sc_I(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                               Npp16s nLevel, NppCmpOp nRelOp);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                                lupine_npp_length_t nLength, Npp32f nLevel,
                                NppCmpOp nRelOp, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_32f(const Npp32f *pSrc, Npp32f *pDst,
                            lupine_npp_length_t nLength, Npp32f nLevel,
                            NppCmpOp nRelOp);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_32f_I_Ctx(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                                  Npp32f nLevel, NppCmpOp nRelOp,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                              Npp32f nLevel, NppCmpOp nRelOp);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                                 lupine_npp_length_t nLength, Npp32f nLevel,
                                 NppCmpOp nRelOp,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                             lupine_npp_length_t nLength, Npp32f nLevel,
                             NppCmpOp nRelOp);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_32fc_I_Ctx(Npp32fc *pSrcDst,
                                   lupine_npp_length_t nLength, Npp32f nLevel,
                                   NppCmpOp nRelOp,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_32fc_I(Npp32fc *pSrcDst, lupine_npp_length_t nLength,
                               Npp32f nLevel, NppCmpOp nRelOp);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                                lupine_npp_length_t nLength, Npp64f nLevel,
                                NppCmpOp nRelOp, NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_64f(const Npp64f *pSrc, Npp64f *pDst,
                            lupine_npp_length_t nLength, Npp64f nLevel,
                            NppCmpOp nRelOp);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_64f_I_Ctx(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                                  Npp64f nLevel, NppCmpOp nRelOp,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                              Npp64f nLevel, NppCmpOp nRelOp);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                                 lupine_npp_length_t nLength, Npp64f nLevel,
                                 NppCmpOp nRelOp,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                             lupine_npp_length_t nLength, Npp64f nLevel,
                             NppCmpOp nRelOp);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_64fc_I_Ctx(Npp64fc *pSrcDst,
                                   lupine_npp_length_t nLength, Npp64f nLevel,
                                   NppCmpOp nRelOp,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nRelOp SEND_ONLY
 */
NppStatus nppsThreshold_64fc_I(Npp64fc *pSrcDst, lupine_npp_length_t nLength,
                               Npp64f nLevel, NppCmpOp nRelOp);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                                   lupine_npp_length_t nLength, Npp16s nLevel,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_16s(const Npp16s *pSrc, Npp16s *pDst,
                               lupine_npp_length_t nLength, Npp16s nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_16s_I_Ctx(Npp16s *pSrcDst,
                                     lupine_npp_length_t nLength, Npp16s nLevel,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_16s_I(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                                 Npp16s nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_16sc_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                    lupine_npp_length_t nLength, Npp16s nLevel,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_16sc(const Npp16sc *pSrc, Npp16sc *pDst,
                                lupine_npp_length_t nLength, Npp16s nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_16sc_I_Ctx(Npp16sc *pSrcDst,
                                      lupine_npp_length_t nLength,
                                      Npp16s nLevel,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_16sc_I(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                                  Npp16s nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                                   lupine_npp_length_t nLength, Npp32f nLevel,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_32f(const Npp32f *pSrc, Npp32f *pDst,
                               lupine_npp_length_t nLength, Npp32f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_32f_I_Ctx(Npp32f *pSrcDst,
                                     lupine_npp_length_t nLength, Npp32f nLevel,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                                 Npp32f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                                    lupine_npp_length_t nLength, Npp32f nLevel,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                                lupine_npp_length_t nLength, Npp32f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_32fc_I_Ctx(Npp32fc *pSrcDst,
                                      lupine_npp_length_t nLength,
                                      Npp32f nLevel,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_32fc_I(Npp32fc *pSrcDst, lupine_npp_length_t nLength,
                                  Npp32f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                                   lupine_npp_length_t nLength, Npp64f nLevel,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_64f(const Npp64f *pSrc, Npp64f *pDst,
                               lupine_npp_length_t nLength, Npp64f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_64f_I_Ctx(Npp64f *pSrcDst,
                                     lupine_npp_length_t nLength, Npp64f nLevel,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                                 Npp64f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                                    lupine_npp_length_t nLength, Npp64f nLevel,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                                lupine_npp_length_t nLength, Npp64f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LT_64fc_I_Ctx(Npp64fc *pSrcDst,
                                      lupine_npp_length_t nLength,
                                      Npp64f nLevel,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_LT_64fc_I(Npp64fc *pSrcDst, lupine_npp_length_t nLength,
                                  Npp64f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                                   lupine_npp_length_t nLength, Npp16s nLevel,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_16s(const Npp16s *pSrc, Npp16s *pDst,
                               lupine_npp_length_t nLength, Npp16s nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_16s_I_Ctx(Npp16s *pSrcDst,
                                     lupine_npp_length_t nLength, Npp16s nLevel,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_16s_I(Npp16s *pSrcDst, lupine_npp_length_t nLength,
                                 Npp16s nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_16sc_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                    lupine_npp_length_t nLength, Npp16s nLevel,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_16sc(const Npp16sc *pSrc, Npp16sc *pDst,
                                lupine_npp_length_t nLength, Npp16s nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_16sc_I_Ctx(Npp16sc *pSrcDst,
                                      lupine_npp_length_t nLength,
                                      Npp16s nLevel,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_16sc_I(Npp16sc *pSrcDst, lupine_npp_length_t nLength,
                                  Npp16s nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                                   lupine_npp_length_t nLength, Npp32f nLevel,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_32f(const Npp32f *pSrc, Npp32f *pDst,
                               lupine_npp_length_t nLength, Npp32f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_32f_I_Ctx(Npp32f *pSrcDst,
                                     lupine_npp_length_t nLength, Npp32f nLevel,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_32f_I(Npp32f *pSrcDst, lupine_npp_length_t nLength,
                                 Npp32f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                                    lupine_npp_length_t nLength, Npp32f nLevel,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                                lupine_npp_length_t nLength, Npp32f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_32fc_I_Ctx(Npp32fc *pSrcDst,
                                      lupine_npp_length_t nLength,
                                      Npp32f nLevel,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_32fc_I(Npp32fc *pSrcDst, lupine_npp_length_t nLength,
                                  Npp32f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                                   lupine_npp_length_t nLength, Npp64f nLevel,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_64f(const Npp64f *pSrc, Npp64f *pDst,
                               lupine_npp_length_t nLength, Npp64f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_64f_I_Ctx(Npp64f *pSrcDst,
                                     lupine_npp_length_t nLength, Npp64f nLevel,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_64f_I(Npp64f *pSrcDst, lupine_npp_length_t nLength,
                                 Npp64f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                                    lupine_npp_length_t nLength, Npp64f nLevel,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                                lupine_npp_length_t nLength, Npp64f nLevel);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GT_64fc_I_Ctx(Npp64fc *pSrcDst,
                                      lupine_npp_length_t nLength,
                                      Npp64f nLevel,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 */
NppStatus nppsThreshold_GT_64fc_I(Npp64fc *pSrcDst, lupine_npp_length_t nLength,
                                  Npp64f nLevel);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                                      lupine_npp_length_t nLength,
                                      Npp16s nLevel, Npp16s nValue,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16s(const Npp16s *pSrc, Npp16s *pDst,
                                  lupine_npp_length_t nLength, Npp16s nLevel,
                                  Npp16s nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16s_I_Ctx(Npp16s *pSrcDst,
                                        lupine_npp_length_t nLength,
                                        Npp16s nLevel, Npp16s nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16s_I(Npp16s *pSrcDst,
                                    lupine_npp_length_t nLength, Npp16s nLevel,
                                    Npp16s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16sc_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                       lupine_npp_length_t nLength,
                                       Npp16s nLevel, Npp16sc nValue,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16sc(const Npp16sc *pSrc, Npp16sc *pDst,
                                   lupine_npp_length_t nLength, Npp16s nLevel,
                                   Npp16sc nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16sc_I_Ctx(Npp16sc *pSrcDst,
                                         lupine_npp_length_t nLength,
                                         Npp16s nLevel, Npp16sc nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_16sc_I(Npp16sc *pSrcDst,
                                     lupine_npp_length_t nLength, Npp16s nLevel,
                                     Npp16sc nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                                      lupine_npp_length_t nLength,
                                      Npp32f nLevel, Npp32f nValue,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32f(const Npp32f *pSrc, Npp32f *pDst,
                                  lupine_npp_length_t nLength, Npp32f nLevel,
                                  Npp32f nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32f_I_Ctx(Npp32f *pSrcDst,
                                        lupine_npp_length_t nLength,
                                        Npp32f nLevel, Npp32f nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32f_I(Npp32f *pSrcDst,
                                    lupine_npp_length_t nLength, Npp32f nLevel,
                                    Npp32f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                                       lupine_npp_length_t nLength,
                                       Npp32f nLevel, Npp32fc nValue,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                                   lupine_npp_length_t nLength, Npp32f nLevel,
                                   Npp32fc nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32fc_I_Ctx(Npp32fc *pSrcDst,
                                         lupine_npp_length_t nLength,
                                         Npp32f nLevel, Npp32fc nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_32fc_I(Npp32fc *pSrcDst,
                                     lupine_npp_length_t nLength, Npp32f nLevel,
                                     Npp32fc nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                                      lupine_npp_length_t nLength,
                                      Npp64f nLevel, Npp64f nValue,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64f(const Npp64f *pSrc, Npp64f *pDst,
                                  lupine_npp_length_t nLength, Npp64f nLevel,
                                  Npp64f nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64f_I_Ctx(Npp64f *pSrcDst,
                                        lupine_npp_length_t nLength,
                                        Npp64f nLevel, Npp64f nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64f_I(Npp64f *pSrcDst,
                                    lupine_npp_length_t nLength, Npp64f nLevel,
                                    Npp64f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                                       lupine_npp_length_t nLength,
                                       Npp64f nLevel, Npp64fc nValue,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                                   lupine_npp_length_t nLength, Npp64f nLevel,
                                   Npp64fc nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64fc_I_Ctx(Npp64fc *pSrcDst,
                                         lupine_npp_length_t nLength,
                                         Npp64f nLevel, Npp64fc nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_LTVal_64fc_I(Npp64fc *pSrcDst,
                                     lupine_npp_length_t nLength, Npp64f nLevel,
                                     Npp64fc nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                                      lupine_npp_length_t nLength,
                                      Npp16s nLevel, Npp16s nValue,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16s(const Npp16s *pSrc, Npp16s *pDst,
                                  lupine_npp_length_t nLength, Npp16s nLevel,
                                  Npp16s nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16s_I_Ctx(Npp16s *pSrcDst,
                                        lupine_npp_length_t nLength,
                                        Npp16s nLevel, Npp16s nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16s_I(Npp16s *pSrcDst,
                                    lupine_npp_length_t nLength, Npp16s nLevel,
                                    Npp16s nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16sc_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                                       lupine_npp_length_t nLength,
                                       Npp16s nLevel, Npp16sc nValue,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16sc(const Npp16sc *pSrc, Npp16sc *pDst,
                                   lupine_npp_length_t nLength, Npp16s nLevel,
                                   Npp16sc nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16sc_I_Ctx(Npp16sc *pSrcDst,
                                         lupine_npp_length_t nLength,
                                         Npp16s nLevel, Npp16sc nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_16sc_I(Npp16sc *pSrcDst,
                                     lupine_npp_length_t nLength, Npp16s nLevel,
                                     Npp16sc nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                                      lupine_npp_length_t nLength,
                                      Npp32f nLevel, Npp32f nValue,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32f(const Npp32f *pSrc, Npp32f *pDst,
                                  lupine_npp_length_t nLength, Npp32f nLevel,
                                  Npp32f nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32f_I_Ctx(Npp32f *pSrcDst,
                                        lupine_npp_length_t nLength,
                                        Npp32f nLevel, Npp32f nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32f_I(Npp32f *pSrcDst,
                                    lupine_npp_length_t nLength, Npp32f nLevel,
                                    Npp32f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                                       lupine_npp_length_t nLength,
                                       Npp32f nLevel, Npp32fc nValue,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                                   lupine_npp_length_t nLength, Npp32f nLevel,
                                   Npp32fc nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32fc_I_Ctx(Npp32fc *pSrcDst,
                                         lupine_npp_length_t nLength,
                                         Npp32f nLevel, Npp32fc nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_32fc_I(Npp32fc *pSrcDst,
                                     lupine_npp_length_t nLength, Npp32f nLevel,
                                     Npp32fc nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64f_Ctx(const Npp64f *pSrc, Npp64f *pDst,
                                      lupine_npp_length_t nLength,
                                      Npp64f nLevel, Npp64f nValue,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64f(const Npp64f *pSrc, Npp64f *pDst,
                                  lupine_npp_length_t nLength, Npp64f nLevel,
                                  Npp64f nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64f_I_Ctx(Npp64f *pSrcDst,
                                        lupine_npp_length_t nLength,
                                        Npp64f nLevel, Npp64f nValue,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64f_I(Npp64f *pSrcDst,
                                    lupine_npp_length_t nLength, Npp64f nLevel,
                                    Npp64f nValue);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                                       lupine_npp_length_t nLength,
                                       Npp64f nLevel, Npp64fc nValue,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                                   lupine_npp_length_t nLength, Npp64f nLevel,
                                   Npp64fc nValue);
/**
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64fc_I_Ctx(Npp64fc *pSrcDst,
                                         lupine_npp_length_t nLength,
                                         Npp64f nLevel, Npp64fc nValue,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nLevel SEND_ONLY
 * @param nValue SEND_ONLY
 */
NppStatus nppsThreshold_GTVal_64fc_I(Npp64fc *pSrcDst,
                                     lupine_npp_length_t nLength, Npp64f nLevel,
                                     Npp64fc nValue);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsIntegralGetBufferSize_32s(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @guard NPP_VERSION >= 12200
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsIntegral_32s_Ctx(const Npp32s *pSrc, Npp32s *pDst,
                               lupine_npp_length_t nLength,
                               Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsIntegral_32s(const Npp32s *pSrc, Npp32s *pDst,
                           lupine_npp_length_t nLength, Npp8u *pDeviceBuffer);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_8u_Ctx(Npp8u nValue, Npp8u *pDst, lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_8u(Npp8u nValue, Npp8u *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_8s_Ctx(Npp8s nValue, Npp8s *pDst, lupine_npp_length_t nLength,
                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_8s(Npp8s nValue, Npp8s *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_16u_Ctx(Npp16u nValue, Npp16u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_16u(Npp16u nValue, Npp16u *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_16s_Ctx(Npp16s nValue, Npp16s *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_16s(Npp16s nValue, Npp16s *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_16sc_Ctx(Npp16sc nValue, Npp16sc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_16sc(Npp16sc nValue, Npp16sc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_32u_Ctx(Npp32u nValue, Npp32u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_32u(Npp32u nValue, Npp32u *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_32s_Ctx(Npp32s nValue, Npp32s *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_32s(Npp32s nValue, Npp32s *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_32sc_Ctx(Npp32sc nValue, Npp32sc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_32sc(Npp32sc nValue, Npp32sc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_32f_Ctx(Npp32f nValue, Npp32f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_32f(Npp32f nValue, Npp32f *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_32fc_Ctx(Npp32fc nValue, Npp32fc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_32fc(Npp32fc nValue, Npp32fc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_64s_Ctx(Npp64s nValue, Npp64s *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_64s(Npp64s nValue, Npp64s *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_64sc_Ctx(Npp64sc nValue, Npp64sc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_64sc(Npp64sc nValue, Npp64sc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_64f_Ctx(Npp64f nValue, Npp64f *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_64f(Npp64f nValue, Npp64f *pDst, lupine_npp_length_t nLength);
/**
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSet_64fc_Ctx(Npp64fc nValue, Npp64fc *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nValue SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsSet_64fc(Npp64fc nValue, Npp64fc *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_8u_Ctx(Npp8u *pDst, lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_8u(Npp8u *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_16s_Ctx(Npp16s *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_16s(Npp16s *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_16sc_Ctx(Npp16sc *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_16sc(Npp16sc *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_32s_Ctx(Npp32s *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_32s(Npp32s *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_32sc_Ctx(Npp32sc *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_32sc(Npp32sc *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_32f_Ctx(Npp32f *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_32f(Npp32f *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_32fc_Ctx(Npp32fc *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_32fc(Npp32fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_64s_Ctx(Npp64s *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_64s(Npp64s *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_64sc_Ctx(Npp64sc *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_64sc(Npp64sc *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_64f_Ctx(Npp64f *pDst, lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_64f(Npp64f *pDst, lupine_npp_length_t nLength);
/**
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZero_64fc_Ctx(Npp64fc *pDst, lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsZero_64fc(Npp64fc *pDst, lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_8u_Ctx(const Npp8u *pSrc, Npp8u *pDst,
                          lupine_npp_length_t nLength,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_8u(const Npp8u *pSrc, Npp8u *pDst,
                      lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_16s_Ctx(const Npp16s *pSrc, Npp16s *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_16s(const Npp16s *pSrc, Npp16s *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_32s_Ctx(const Npp32s *pSrc, Npp32s *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_32s(const Npp32s *pSrc, Npp32s *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_32f_Ctx(const Npp32f *pSrc, Npp32f *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_32f(const Npp32f *pSrc, Npp32f *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_64s_Ctx(const Npp64s *pSrc, Npp64s *pDst,
                           lupine_npp_length_t nLength,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_64s(const Npp64s *pSrc, Npp64s *pDst,
                       lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_16sc_Ctx(const Npp16sc *pSrc, Npp16sc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_16sc(const Npp16sc *pSrc, Npp16sc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_32sc_Ctx(const Npp32sc *pSrc, Npp32sc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_32sc(const Npp32sc *pSrc, Npp32sc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_32fc_Ctx(const Npp32fc *pSrc, Npp32fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_32fc(const Npp32fc *pSrc, Npp32fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_64sc_Ctx(const Npp64sc *pSrc, Npp64sc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_64sc(const Npp64sc *pSrc, Npp64sc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCopy_64fc_Ctx(const Npp64fc *pSrc, Npp64fc *pDst,
                            lupine_npp_length_t nLength,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsCopy_64fc(const Npp64fc *pSrc, Npp64fc *pDst,
                        lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinEvery_8u_I_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMinEvery_8u_I(const Npp8u *pSrc, Npp8u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinEvery_16u_I_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMinEvery_16u_I(const Npp16u *pSrc, Npp16u *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinEvery_16s_I_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMinEvery_16s_I(const Npp16s *pSrc, Npp16s *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinEvery_32s_I_Ctx(const Npp32s *pSrc, Npp32s *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMinEvery_32s_I(const Npp32s *pSrc, Npp32s *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinEvery_32f_I_Ctx(const Npp32f *pSrc, Npp32f *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMinEvery_32f_I(const Npp32f *pSrc, Npp32f *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinEvery_64f_I_Ctx(const Npp64f *pSrc, Npp64f *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMinEvery_64f_I(const Npp64f *pSrc, Npp64f *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxEvery_8u_I_Ctx(const Npp8u *pSrc, Npp8u *pSrcDst,
                                lupine_npp_length_t nLength,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMaxEvery_8u_I(const Npp8u *pSrc, Npp8u *pSrcDst,
                            lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxEvery_16u_I_Ctx(const Npp16u *pSrc, Npp16u *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMaxEvery_16u_I(const Npp16u *pSrc, Npp16u *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxEvery_16s_I_Ctx(const Npp16s *pSrc, Npp16s *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMaxEvery_16s_I(const Npp16s *pSrc, Npp16s *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxEvery_32s_I_Ctx(const Npp32s *pSrc, Npp32s *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMaxEvery_32s_I(const Npp32s *pSrc, Npp32s *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxEvery_32f_I_Ctx(const Npp32f *pSrc, Npp32f *pSrcDst,
                                 lupine_npp_length_t nLength,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param pSrcDst SEND_ONLY
 * @param nLength SEND_ONLY
 */
NppStatus nppsMaxEvery_32f_I(const Npp32f *pSrc, Npp32f *pSrcDst,
                             lupine_npp_length_t nLength);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_32f(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_32fc_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_32fc(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_64f(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_64fc_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_64fc(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_16s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_16s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_16sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_16sc_Sfs(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsSumGetBufferSize_16sc32sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_16sc32sc_Sfs(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_32s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSumGetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsSumGetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pSum, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                      Npp32f *pSum, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_32fc_Ctx(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                           Npp32fc *pSum, Npp8u *pDeviceBuffer,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_32fc(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                       Npp32fc *pSum, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pSum, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                      Npp64f *pSum, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_64fc_Ctx(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                           Npp64fc *pSum, Npp8u *pDeviceBuffer,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_64fc(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                       Npp64fc *pSum, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_16s_Sfs_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                              Npp16s *pSum, int nScaleFactor,
                              Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_16s_Sfs(const Npp16s *pSrc, lupine_npp_length_t nLength,
                          Npp16s *pSum, int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_32s_Sfs_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                              Npp32s *pSum, int nScaleFactor,
                              Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_32s_Sfs(const Npp32s *pSrc, lupine_npp_length_t nLength,
                          Npp32s *pSum, int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_16sc_Sfs_Ctx(const Npp16sc *pSrc, lupine_npp_length_t nLength,
                               Npp16sc *pSum, int nScaleFactor,
                               Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_16sc_Sfs(const Npp16sc *pSrc, lupine_npp_length_t nLength,
                           Npp16sc *pSum, int nScaleFactor,
                           Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_16sc32sc_Sfs_Ctx(const Npp16sc *pSrc,
                                   lupine_npp_length_t nLength, Npp32sc *pSum,
                                   int nScaleFactor, Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_16sc32sc_Sfs(const Npp16sc *pSrc, lupine_npp_length_t nLength,
                               Npp32sc *pSum, int nScaleFactor,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsSum_16s32s_Sfs_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp32s *pSum,
                                 int nScaleFactor, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pSum SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsSum_16s32s_Sfs(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pSum, int nScaleFactor,
                             Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxGetBufferSize_16s(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxGetBufferSize_32s(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxGetBufferSize_32f(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxGetBufferSize_64f(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMax_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                          Npp16s *pMax, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMax_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                      Npp16s *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMax_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                          Npp32s *pMax, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMax_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                      Npp32s *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMax_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pMax, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMax_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                      Npp32f *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMax_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pMax, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMax_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                      Npp64f *pMax, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxIndxGetBufferSize_16s(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxIndxGetBufferSize_32s(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndxGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxIndxGetBufferSize_32f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndxGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxIndxGetBufferSize_64f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndx_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                              Npp16s *pMax, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxIndx_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                          Npp16s *pMax, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndx_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                              Npp32s *pMax, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxIndx_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                          Npp32s *pMax, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndx_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                              Npp32f *pMax, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxIndx_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pMax, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxIndx_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pMax, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxIndx_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pMax, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbsGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxAbsGetBufferSize_16s(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbsGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxAbsGetBufferSize_32s(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbs_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pMaxAbs, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxAbs_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                         Npp16s *pMaxAbs, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbs_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pMaxAbs, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxAbs_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                         Npp32s *pMaxAbs, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbsIndxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxAbsIndxGetBufferSize_16s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbsIndxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaxAbsIndxGetBufferSize_32s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbsIndx_16s_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp16s *pMaxAbs,
                                 int *pIndx, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxAbsIndx_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pMaxAbs, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaxAbsIndx_32s_Ctx(const Npp32s *pSrc,
                                 lupine_npp_length_t nLength, Npp32s *pMaxAbs,
                                 int *pIndx, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMaxAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaxAbsIndx_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pMaxAbs, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinGetBufferSize_16s(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinGetBufferSize_32s(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinGetBufferSize_32f(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinGetBufferSize_64f(lupine_npp_length_t nLength,
                                   lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMin_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                          Npp16s *pMin, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMin_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                      Npp16s *pMin, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMin_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                          Npp32s *pMin, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMin_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                      Npp32s *pMin, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMin_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pMin, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMin_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                      Npp32f *pMin, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMin_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pMin, Npp8u *pDeviceBuffer,
                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMin_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                      Npp64f *pMin, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinIndxGetBufferSize_16s(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinIndxGetBufferSize_32s(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndxGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinIndxGetBufferSize_32f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndxGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinIndxGetBufferSize_64f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndx_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                              Npp16s *pMin, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinIndx_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                          Npp16s *pMin, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndx_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                              Npp32s *pMin, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinIndx_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                          Npp32s *pMin, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndx_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                              Npp32f *pMin, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinIndx_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pMin, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinIndx_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pMin, int *pIndx, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinIndx_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pMin, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbsGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinAbsGetBufferSize_16s(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbsGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinAbsGetBufferSize_32s(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbs_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pMinAbs, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinAbs_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                         Npp16s *pMinAbs, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbs_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pMinAbs, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinAbs_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                         Npp32s *pMinAbs, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbsIndxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinAbsIndxGetBufferSize_16s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbsIndxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinAbsIndxGetBufferSize_32s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbsIndx_16s_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp16s *pMinAbs,
                                 int *pIndx, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinAbsIndx_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pMinAbs, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinAbsIndx_32s_Ctx(const Npp32s *pSrc,
                                 lupine_npp_length_t nLength, Npp32s *pMinAbs,
                                 int *pIndx, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMinAbs SEND_ONLY
 * @param pIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinAbsIndx_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pMinAbs, int *pIndx, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_32f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_32fc_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_32fc(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_64f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_64fc_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_64fc(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_16s_Sfs_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_16s_Sfs(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_32s_Sfs(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanGetBufferSize_16sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanGetBufferSize_16sc_Sfs(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                           Npp32f *pMean, Npp8u *pDeviceBuffer,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                       Npp32f *pMean, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_32fc_Ctx(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                            Npp32fc *pMean, Npp8u *pDeviceBuffer,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_32fc(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                        Npp32fc *pMean, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                           Npp64f *pMean, Npp8u *pDeviceBuffer,
                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                       Npp64f *pMean, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_64fc_Ctx(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                            Npp64fc *pMean, Npp8u *pDeviceBuffer,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_64fc(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                        Npp64fc *pMean, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_16s_Sfs_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                               Npp16s *pMean, int nScaleFactor,
                               Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_16s_Sfs(const Npp16s *pSrc, lupine_npp_length_t nLength,
                           Npp16s *pMean, int nScaleFactor,
                           Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_32s_Sfs_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                               Npp32s *pMean, int nScaleFactor,
                               Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_32s_Sfs(const Npp32s *pSrc, lupine_npp_length_t nLength,
                           Npp32s *pMean, int nScaleFactor,
                           Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMean_16sc_Sfs_Ctx(const Npp16sc *pSrc,
                                lupine_npp_length_t nLength, Npp16sc *pMean,
                                int nScaleFactor, Npp8u *pDeviceBuffer,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMean_16sc_Sfs(const Npp16sc *pSrc, lupine_npp_length_t nLength,
                            Npp16sc *pMean, int nScaleFactor,
                            Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDevGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsStdDevGetBufferSize_32f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDevGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsStdDevGetBufferSize_64f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsStdDevGetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsStdDevGetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDevGetBufferSize_16s_Sfs_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsStdDevGetBufferSize_16s_Sfs(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDev_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                             Npp32f *pStdDev, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsStdDev_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                         Npp32f *pStdDev, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDev_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                             Npp64f *pStdDev, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsStdDev_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                         Npp64f *pStdDev, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDev_16s32s_Sfs_Ctx(const Npp16s *pSrc,
                                    lupine_npp_length_t nLength,
                                    Npp32s *pStdDev, int nScaleFactor,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsStdDev_16s32s_Sfs(const Npp16s *pSrc, lupine_npp_length_t nLength,
                                Npp32s *pStdDev, int nScaleFactor,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsStdDev_16s_Sfs_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp16s *pStdDev,
                                 int nScaleFactor, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsStdDev_16s_Sfs(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pStdDev, int nScaleFactor,
                             Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanStdDevGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanStdDevGetBufferSize_32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanStdDevGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMeanStdDevGetBufferSize_64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMeanStdDevGetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMeanStdDevGetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMeanStdDevGetBufferSize_16s_Sfs_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMeanStdDevGetBufferSize_16s_Sfs(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanStdDev_32f_Ctx(const Npp32f *pSrc,
                                 lupine_npp_length_t nLength, Npp32f *pMean,
                                 Npp32f *pStdDev, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMeanStdDev_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                             Npp32f *pMean, Npp32f *pStdDev,
                             Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanStdDev_64f_Ctx(const Npp64f *pSrc,
                                 lupine_npp_length_t nLength, Npp64f *pMean,
                                 Npp64f *pStdDev, Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMeanStdDev_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                             Npp64f *pMean, Npp64f *pStdDev,
                             Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanStdDev_16s32s_Sfs_Ctx(const Npp16s *pSrc,
                                        lupine_npp_length_t nLength,
                                        Npp32s *pMean, Npp32s *pStdDev,
                                        int nScaleFactor, Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMeanStdDev_16s32s_Sfs(const Npp16s *pSrc,
                                    lupine_npp_length_t nLength, Npp32s *pMean,
                                    Npp32s *pStdDev, int nScaleFactor,
                                    Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMeanStdDev_16s_Sfs_Ctx(const Npp16s *pSrc,
                                     lupine_npp_length_t nLength, Npp16s *pMean,
                                     Npp16s *pStdDev, int nScaleFactor,
                                     Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMean SEND_ONLY
 * @param pStdDev SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMeanStdDev_16s_Sfs(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp16s *pMean,
                                 Npp16s *pStdDev, int nScaleFactor,
                                 Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_8u_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_8u(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_16s(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_16u_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_16u(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_32s(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_32u_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_32u(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_32f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxGetBufferSize_64f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_8u_Ctx(const Npp8u *pSrc, lupine_npp_length_t nLength,
                            Npp8u *pMin, Npp8u *pMax, Npp8u *pDeviceBuffer,
                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_8u(const Npp8u *pSrc, lupine_npp_length_t nLength,
                        Npp8u *pMin, Npp8u *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_16s_Ctx(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pMin, Npp16s *pMax, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                         Npp16s *pMin, Npp16s *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_16u_Ctx(const Npp16u *pSrc, lupine_npp_length_t nLength,
                             Npp16u *pMin, Npp16u *pMax, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_16u(const Npp16u *pSrc, lupine_npp_length_t nLength,
                         Npp16u *pMin, Npp16u *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_32u_Ctx(const Npp32u *pSrc, lupine_npp_length_t nLength,
                             Npp32u *pMin, Npp32u *pMax, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_32u(const Npp32u *pSrc, lupine_npp_length_t nLength,
                         Npp32u *pMin, Npp32u *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_32s_Ctx(const Npp32s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pMin, Npp32s *pMax, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                         Npp32s *pMin, Npp32s *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                             Npp32f *pMin, Npp32f *pMax, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                         Npp32f *pMin, Npp32f *pMax, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMax_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                             Npp64f *pMin, Npp64f *pMax, Npp8u *pDeviceBuffer,
                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMax_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                         Npp64f *pMin, Npp64f *pMax, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_8u_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_8u(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_16s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_16u_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_16u(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_32s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_32u_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_32u(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndxGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMinMaxIndxGetBufferSize_64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_8u_Ctx(const Npp8u *pSrc, lupine_npp_length_t nLength,
                                Npp8u *pMin, int *pMinIndx, Npp8u *pMax,
                                int *pMaxIndx, Npp8u *pDeviceBuffer,
                                NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_8u(const Npp8u *pSrc, lupine_npp_length_t nLength,
                            Npp8u *pMin, int *pMinIndx, Npp8u *pMax,
                            int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_16s_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp16s *pMin,
                                 int *pMinIndx, Npp16s *pMax, int *pMaxIndx,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_16s(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp16s *pMin, int *pMinIndx, Npp16s *pMax,
                             int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_16u_Ctx(const Npp16u *pSrc,
                                 lupine_npp_length_t nLength, Npp16u *pMin,
                                 int *pMinIndx, Npp16u *pMax, int *pMaxIndx,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_16u(const Npp16u *pSrc, lupine_npp_length_t nLength,
                             Npp16u *pMin, int *pMinIndx, Npp16u *pMax,
                             int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_32s_Ctx(const Npp32s *pSrc,
                                 lupine_npp_length_t nLength, Npp32s *pMin,
                                 int *pMinIndx, Npp32s *pMax, int *pMaxIndx,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                             Npp32s *pMin, int *pMinIndx, Npp32s *pMax,
                             int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_32u_Ctx(const Npp32u *pSrc,
                                 lupine_npp_length_t nLength, Npp32u *pMin,
                                 int *pMinIndx, Npp32u *pMax, int *pMaxIndx,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_32u(const Npp32u *pSrc, lupine_npp_length_t nLength,
                             Npp32u *pMin, int *pMinIndx, Npp32u *pMax,
                             int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_32f_Ctx(const Npp32f *pSrc,
                                 lupine_npp_length_t nLength, Npp32f *pMin,
                                 int *pMinIndx, Npp32f *pMax, int *pMaxIndx,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                             Npp32f *pMin, int *pMinIndx, Npp32f *pMax,
                             int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMinMaxIndx_64f_Ctx(const Npp64f *pSrc,
                                 lupine_npp_length_t nLength, Npp64f *pMin,
                                 int *pMinIndx, Npp64f *pMax, int *pMaxIndx,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pMin SEND_ONLY
 * @param pMinIndx SEND_ONLY
 * @param pMax SEND_ONLY
 * @param pMaxIndx SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMinMaxIndx_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                             Npp64f *pMin, int *pMinIndx, Npp64f *pMax,
                             int *pMaxIndx, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormInfGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormInfGetBufferSize_32f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_Inf_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                               Npp32f *pNorm, Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_Inf_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                           Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormInfGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormInfGetBufferSize_64f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_Inf_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                               Npp64f *pNorm, Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_Inf_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                           Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormInfGetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormInfGetBufferSize_16s32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_Inf_16s32f_Ctx(const Npp16s *pSrc,
                                  lupine_npp_length_t nLength, Npp32f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_Inf_16s32f(const Npp16s *pSrc, lupine_npp_length_t nLength,
                              Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormInfGetBufferSize_32fc32f_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormInfGetBufferSize_32fc32f(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_Inf_32fc32f_Ctx(const Npp32fc *pSrc,
                                   lupine_npp_length_t nLength, Npp32f *pNorm,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_Inf_32fc32f(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                               Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormInfGetBufferSize_64fc64f_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormInfGetBufferSize_64fc64f(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_Inf_64fc64f_Ctx(const Npp64fc *pSrc,
                                   lupine_npp_length_t nLength, Npp64f *pNorm,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_Inf_64fc64f(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                               Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormInfGetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormInfGetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_Inf_16s32s_Sfs_Ctx(const Npp16s *pSrc,
                                      lupine_npp_length_t nLength,
                                      Npp32s *pNorm, int nScaleFactor,
                                      Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_Inf_16s32s_Sfs(const Npp16s *pSrc,
                                  lupine_npp_length_t nLength, Npp32s *pNorm,
                                  int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL1GetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_32f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                              Npp32f *pNorm, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL1GetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_64f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pNorm, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL1GetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_16s32f(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_16s32f_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp32f *pNorm,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_16s32f(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL1GetBufferSize_32fc64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_32fc64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_32fc64f_Ctx(const Npp32fc *pSrc,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_32fc64f(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL1GetBufferSize_64fc64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_64fc64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_64fc64f_Ctx(const Npp64fc *pSrc,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_64fc64f(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormL1GetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_16s32s_Sfs_Ctx(const Npp16s *pSrc,
                                     lupine_npp_length_t nLength, Npp32s *pNorm,
                                     int nScaleFactor, Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_16s32s_Sfs(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp32s *pNorm,
                                 int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormL1GetBufferSize_16s64s_Sfs_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL1GetBufferSize_16s64s_Sfs(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L1_16s64s_Sfs_Ctx(const Npp16s *pSrc,
                                     lupine_npp_length_t nLength, Npp64s *pNorm,
                                     int nScaleFactor, Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L1_16s64s_Sfs(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp64s *pNorm,
                                 int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL2GetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL2GetBufferSize_32f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2_32f_Ctx(const Npp32f *pSrc, lupine_npp_length_t nLength,
                              Npp32f *pNorm, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                          Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL2GetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL2GetBufferSize_64f(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2_64f_Ctx(const Npp64f *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pNorm, Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2_64f(const Npp64f *pSrc, lupine_npp_length_t nLength,
                          Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL2GetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL2GetBufferSize_16s32f(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2_16s32f_Ctx(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp32f *pNorm,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2_16s32f(const Npp16s *pSrc, lupine_npp_length_t nLength,
                             Npp32f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL2GetBufferSize_32fc64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL2GetBufferSize_32fc64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2_32fc64f_Ctx(const Npp32fc *pSrc,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2_32fc64f(const Npp32fc *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormL2GetBufferSize_64fc64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL2GetBufferSize_64fc64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2_64fc64f_Ctx(const Npp64fc *pSrc,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2_64fc64f(const Npp64fc *pSrc, lupine_npp_length_t nLength,
                              Npp64f *pNorm, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormL2GetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormL2GetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2_16s32s_Sfs_Ctx(const Npp16s *pSrc,
                                     lupine_npp_length_t nLength, Npp32s *pNorm,
                                     int nScaleFactor, Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2_16s32s_Sfs(const Npp16s *pSrc,
                                 lupine_npp_length_t nLength, Npp32s *pNorm,
                                 int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormL2SqrGetBufferSize_16s64s_Sfs_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormL2SqrGetBufferSize_16s64s_Sfs(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNorm_L2Sqr_16s64s_Sfs_Ctx(const Npp16s *pSrc,
                                        lupine_npp_length_t nLength,
                                        Npp64s *pNorm, int nScaleFactor,
                                        Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNorm_L2Sqr_16s64s_Sfs(const Npp16s *pSrc,
                                    lupine_npp_length_t nLength, Npp64s *pNorm,
                                    int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffInfGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffInfGetBufferSize_32f(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                   lupine_npp_length_t nLength, Npp32f *pNorm,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                               lupine_npp_length_t nLength, Npp32f *pNorm,
                               Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffInfGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffInfGetBufferSize_64f(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pNorm,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pNorm,
                               Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffInfGetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffInfGetBufferSize_16s32f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                      lupine_npp_length_t nLength,
                                      Npp32f *pNorm, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                  lupine_npp_length_t nLength, Npp32f *pNorm,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffInfGetBufferSize_32fc32f_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffInfGetBufferSize_32fc32f(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_32fc32f_Ctx(const Npp32fc *pSrc1,
                                       const Npp32fc *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp32f *pNorm, Npp8u *pDeviceBuffer,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_32fc32f(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                   lupine_npp_length_t nLength, Npp32f *pNorm,
                                   Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffInfGetBufferSize_64fc64f_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffInfGetBufferSize_64fc64f(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_64fc64f_Ctx(const Npp64fc *pSrc1,
                                       const Npp64fc *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pNorm, Npp8u *pDeviceBuffer,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_64fc64f(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pNorm,
                                   Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffInfGetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffInfGetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_16s32s_Sfs_Ctx(const Npp16s *pSrc1,
                                          const Npp16s *pSrc2,
                                          lupine_npp_length_t nLength,
                                          Npp32s *pNorm, int nScaleFactor,
                                          Npp8u *pDeviceBuffer,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_Inf_16s32s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                      lupine_npp_length_t nLength,
                                      Npp32s *pNorm, int nScaleFactor,
                                      Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiffL1GetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffL1GetBufferSize_32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                  lupine_npp_length_t nLength, Npp32f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                              lupine_npp_length_t nLength, Npp32f *pNorm,
                              Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiffL1GetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffL1GetBufferSize_64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pNorm,
                              Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffL1GetBufferSize_16s32f(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     lupine_npp_length_t nLength, Npp32f *pNorm,
                                     Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 lupine_npp_length_t nLength, Npp32f *pNorm,
                                 Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL1GetBufferSize_32fc64f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL1GetBufferSize_32fc64f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_32fc64f_Ctx(const Npp32fc *pSrc1,
                                      const Npp32fc *pSrc2,
                                      lupine_npp_length_t nLength,
                                      Npp64f *pNorm, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_32fc64f(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL1GetBufferSize_64fc64f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL1GetBufferSize_64fc64f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_64fc64f_Ctx(const Npp64fc *pSrc1,
                                      const Npp64fc *pSrc2,
                                      lupine_npp_length_t nLength,
                                      Npp64f *pNorm, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_64fc64f(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_16s32s_Sfs_Ctx(const Npp16s *pSrc1,
                                         const Npp16s *pSrc2,
                                         lupine_npp_length_t nLength,
                                         Npp32s *pNorm, int nScaleFactor,
                                         Npp8u *pDeviceBuffer,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_16s32s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     lupine_npp_length_t nLength, Npp32s *pNorm,
                                     int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s64s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL1GetBufferSize_16s64s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L1_16s64s_Sfs_Ctx(const Npp16s *pSrc1,
                                         const Npp16s *pSrc2,
                                         lupine_npp_length_t nLength,
                                         Npp64s *pNorm, int nScaleFactor,
                                         Npp8u *pDeviceBuffer,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L1_16s64s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     lupine_npp_length_t nLength, Npp64s *pNorm,
                                     int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiffL2GetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffL2GetBufferSize_32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                  lupine_npp_length_t nLength, Npp32f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                              lupine_npp_length_t nLength, Npp32f *pNorm,
                              Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiffL2GetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffL2GetBufferSize_64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pNorm,
                              Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL2GetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsNormDiffL2GetBufferSize_16s32f(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     lupine_npp_length_t nLength, Npp32f *pNorm,
                                     Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 lupine_npp_length_t nLength, Npp32f *pNorm,
                                 Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL2GetBufferSize_32fc64f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL2GetBufferSize_32fc64f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2_32fc64f_Ctx(const Npp32fc *pSrc1,
                                      const Npp32fc *pSrc2,
                                      lupine_npp_length_t nLength,
                                      Npp64f *pNorm, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2_32fc64f(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL2GetBufferSize_64fc64f_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL2GetBufferSize_64fc64f(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2_64fc64f_Ctx(const Npp64fc *pSrc1,
                                      const Npp64fc *pSrc2,
                                      lupine_npp_length_t nLength,
                                      Npp64f *pNorm, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2_64fc64f(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pNorm,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL2GetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL2GetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2_16s32s_Sfs_Ctx(const Npp16s *pSrc1,
                                         const Npp16s *pSrc2,
                                         lupine_npp_length_t nLength,
                                         Npp32s *pNorm, int nScaleFactor,
                                         Npp8u *pDeviceBuffer,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2_16s32s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     lupine_npp_length_t nLength, Npp32s *pNorm,
                                     int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsNormDiff_L2Sqr_16s64s_Sfs_Ctx(const Npp16s *pSrc1,
                                            const Npp16s *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64s *pNorm, int nScaleFactor,
                                            Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pNorm SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsNormDiff_L2Sqr_16s64s_Sfs(const Npp16s *pSrc1,
                                        const Npp16s *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64s *pNorm, int nScaleFactor,
                                        Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                              lupine_npp_length_t nLength, Npp32f *pDp,
                              Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                          lupine_npp_length_t nLength, Npp32f *pDp,
                          Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_32fc_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32fc(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                               lupine_npp_length_t nLength, Npp32fc *pDp,
                               Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                           lupine_npp_length_t nLength, Npp32fc *pDp,
                           Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_32f32fc_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32f32fc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32f32fc_Ctx(const Npp32f *pSrc1, const Npp32fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp32fc *pDp,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32f32fc(const Npp32f *pSrc1, const Npp32fc *pSrc2,
                              lupine_npp_length_t nLength, Npp32fc *pDp,
                              Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_32f64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32f64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32f64f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                 lupine_npp_length_t nLength, Npp64f *pDp,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32f64f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                             lupine_npp_length_t nLength, Npp64f *pDp,
                             Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_32fc64fc_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32fc64fc(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32fc64fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                   lupine_npp_length_t nLength, Npp64fc *pDp,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32fc64fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                               lupine_npp_length_t nLength, Npp64fc *pDp,
                               Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_32f32fc64fc_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_32f32fc64fc(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32f32fc64fc_Ctx(const Npp32f *pSrc1, const Npp32fc *pSrc2,
                                      lupine_npp_length_t nLength, Npp64fc *pDp,
                                      Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32f32fc64fc(const Npp32f *pSrc1, const Npp32fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64fc *pDp,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_64f(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pDp,
                              Npp8u *pDeviceBuffer,
                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                          lupine_npp_length_t nLength, Npp64f *pDp,
                          Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_64fc_Ctx(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_64fc(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                               lupine_npp_length_t nLength, Npp64fc *pDp,
                               Npp8u *pDeviceBuffer,
                               NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                           lupine_npp_length_t nLength, Npp64fc *pDp,
                           Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_64f64fc_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_64f64fc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_64f64fc_Ctx(const Npp64f *pSrc1, const Npp64fc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64fc *pDp,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_64f64fc(const Npp64f *pSrc1, const Npp64fc *pSrc2,
                              lupine_npp_length_t nLength, Npp64fc *pDp,
                              Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_16s64s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_16s64s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s64s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 lupine_npp_length_t nLength, Npp64s *pDp,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s64s(const Npp16s *pSrc1, const Npp16s *pSrc2,
                             lupine_npp_length_t nLength, Npp64s *pDp,
                             Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16sc64sc_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_16sc64sc(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16sc64sc_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                   lupine_npp_length_t nLength, Npp64sc *pDp,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16sc64sc(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               lupine_npp_length_t nLength, Npp64sc *pDp,
                               Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc64sc_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc64sc(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc64sc_Ctx(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                      lupine_npp_length_t nLength, Npp64sc *pDp,
                                      Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc64sc(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                  lupine_npp_length_t nLength, Npp64sc *pDp,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProdGetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_16s32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s32f_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 lupine_npp_length_t nLength, Npp32f *pDp,
                                 Npp8u *pDeviceBuffer,
                                 NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s32f(const Npp16s *pSrc1, const Npp16s *pSrc2,
                             lupine_npp_length_t nLength, Npp32f *pDp,
                             Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16sc32fc_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_16sc32fc(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16sc32fc_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                   lupine_npp_length_t nLength, Npp32fc *pDp,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16sc32fc(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               lupine_npp_length_t nLength, Npp32fc *pDp,
                               Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc32fc_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc32fc(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc32fc_Ctx(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                      lupine_npp_length_t nLength, Npp32fc *pDp,
                                      Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc32fc(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                  lupine_npp_length_t nLength, Npp32fc *pDp,
                                  Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s_Sfs_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_16s_Sfs(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                  lupine_npp_length_t nLength, Npp16s *pDp,
                                  int nScaleFactor, Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                              lupine_npp_length_t nLength, Npp16s *pDp,
                              int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_16sc_Sfs(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16sc_Sfs_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                   lupine_npp_length_t nLength, Npp16sc *pDp,
                                   int nScaleFactor, Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16sc_Sfs(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                               lupine_npp_length_t nLength, Npp16sc *pDp,
                               int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32s_Sfs(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32s_Sfs_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                  lupine_npp_length_t nLength, Npp32s *pDp,
                                  int nScaleFactor, Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32s_Sfs(const Npp32s *pSrc1, const Npp32s *pSrc2,
                              lupine_npp_length_t nLength, Npp32s *pDp,
                              int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_32sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsDotProdGetBufferSize_32sc_Sfs(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32sc_Sfs_Ctx(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                                   lupine_npp_length_t nLength, Npp32sc *pDp,
                                   int nScaleFactor, Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32sc_Sfs(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                               lupine_npp_length_t nLength, Npp32sc *pDp,
                               int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                        lupine_npp_length_t *hpBufferSize,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16s32s_Sfs(lupine_npp_length_t nLength,
                                    lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s32s_Sfs_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                     lupine_npp_length_t nLength, Npp32s *pDp,
                                     int nScaleFactor, Npp8u *pDeviceBuffer,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s32s_Sfs(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                 lupine_npp_length_t nLength, Npp32s *pDp,
                                 int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc32sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc32sc_Sfs(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc32sc_Sfs_Ctx(const Npp16s *pSrc1,
                                          const Npp16sc *pSrc2,
                                          lupine_npp_length_t nLength,
                                          Npp32sc *pDp, int nScaleFactor,
                                          Npp8u *pDeviceBuffer,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc32sc_Sfs(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                      lupine_npp_length_t nLength, Npp32sc *pDp,
                                      int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s32s32s_Sfs_Ctx(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16s32s32s_Sfs(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s32s32s_Sfs_Ctx(const Npp16s *pSrc1,
                                        const Npp32s *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp32s *pDp, int nScaleFactor,
                                        Npp8u *pDeviceBuffer,
                                        NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s32s32s_Sfs(const Npp16s *pSrc1, const Npp32s *pSrc2,
                                    lupine_npp_length_t nLength, Npp32s *pDp,
                                    int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16s16sc_Sfs(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc_Sfs_Ctx(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                      lupine_npp_length_t nLength, Npp16sc *pDp,
                                      int nScaleFactor, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16s16sc_Sfs(const Npp16s *pSrc1, const Npp16sc *pSrc2,
                                  lupine_npp_length_t nLength, Npp16sc *pDp,
                                  int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_16sc32sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_16sc32sc_Sfs(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_16sc32sc_Sfs_Ctx(const Npp16sc *pSrc1,
                                       const Npp16sc *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp32sc *pDp, int nScaleFactor,
                                       Npp8u *pDeviceBuffer,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_16sc32sc_Sfs(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                   lupine_npp_length_t nLength, Npp32sc *pDp,
                                   int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsDotProdGetBufferSize_32s32sc_Sfs_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsDotProdGetBufferSize_32s32sc_Sfs(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsDotProd_32s32sc_Sfs_Ctx(const Npp32s *pSrc1, const Npp32sc *pSrc2,
                                      lupine_npp_length_t nLength, Npp32sc *pDp,
                                      int nScaleFactor, Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDp SEND_ONLY
 * @param nScaleFactor SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsDotProd_32s32sc_Sfs(const Npp32s *pSrc1, const Npp32sc *pSrc2,
                                  lupine_npp_length_t nLength, Npp32sc *pDp,
                                  int nScaleFactor, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsCountInRangeGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsCountInRangeGetBufferSize_32s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pCounts SEND_ONLY
 * @param nLowerBound SEND_ONLY
 * @param nUpperBound SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsCountInRange_32s_Ctx(const Npp32s *pSrc,
                                   lupine_npp_length_t nLength, int *pCounts,
                                   Npp32s nLowerBound, Npp32s nUpperBound,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pCounts SEND_ONLY
 * @param nLowerBound SEND_ONLY
 * @param nUpperBound SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsCountInRange_32s(const Npp32s *pSrc, lupine_npp_length_t nLength,
                               int *pCounts, Npp32s nLowerBound,
                               Npp32s nUpperBound, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsZeroCrossingGetBufferSize_16s32f_Ctx(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize,
                                         NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsZeroCrossingGetBufferSize_16s32f(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pValZC SEND_ONLY
 * @param tZCType SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZeroCrossing_16s32f_Ctx(const Npp16s *pSrc,
                                      lupine_npp_length_t nLength,
                                      Npp32f *pValZC, NppsZCType tZCType,
                                      Npp8u *pDeviceBuffer,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pValZC SEND_ONLY
 * @param tZCType SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsZeroCrossing_16s32f(const Npp16s *pSrc,
                                  lupine_npp_length_t nLength, Npp32f *pValZC,
                                  NppsZCType tZCType, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsZeroCrossingGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsZeroCrossingGetBufferSize_32f(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pValZC SEND_ONLY
 * @param tZCType SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsZeroCrossing_32f_Ctx(const Npp32f *pSrc,
                                   lupine_npp_length_t nLength, Npp32f *pValZC,
                                   NppsZCType tZCType, Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pValZC SEND_ONLY
 * @param tZCType SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsZeroCrossing_32f(const Npp32f *pSrc, lupine_npp_length_t nLength,
                               Npp32f *pValZC, NppsZCType tZCType,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_8u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pDst,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_8u(const Npp8u *pSrc1, const Npp8u *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pDst,
                              Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_8s_Ctx(const Npp8s *pSrc1, const Npp8s *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pDst,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_8s(const Npp8s *pSrc1, const Npp8s *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pDst,
                              Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_16u_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_16u(const Npp16u *pSrc1, const Npp16u *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_16s(const Npp16s *pSrc1, const Npp16s *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_16sc_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_16sc(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_32u_Ctx(const Npp32u *pSrc1, const Npp32u *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_32u(const Npp32u *pSrc1, const Npp32u *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_32s_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_32s(const Npp32s *pSrc1, const Npp32s *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_32sc_Ctx(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_32sc(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_64s_Ctx(const Npp64s *pSrc1, const Npp64s *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_64s(const Npp64s *pSrc1, const Npp64s *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_64sc_Ctx(const Npp64sc *pSrc1, const Npp64sc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_64sc(const Npp64sc *pSrc1, const Npp64sc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumError_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumError_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_8u_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_8u(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_8s_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_8s(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_16u_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_16u(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_16s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_16sc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_16sc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_32u_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_32u(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_32s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_32sc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_32sc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_64s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_64s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_64sc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_64sc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_32f(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_32fc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_32fc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_64f(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumErrorGetBufferSize_64fc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsMaximumErrorGetBufferSize_64fc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_8u_Ctx(const Npp8u *pSrc1, const Npp8u *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pDst,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_8u(const Npp8u *pSrc1, const Npp8u *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pDst,
                              Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_8s_Ctx(const Npp8s *pSrc1, const Npp8s *pSrc2,
                                  lupine_npp_length_t nLength, Npp64f *pDst,
                                  Npp8u *pDeviceBuffer,
                                  NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_8s(const Npp8s *pSrc1, const Npp8s *pSrc2,
                              lupine_npp_length_t nLength, Npp64f *pDst,
                              Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_16u_Ctx(const Npp16u *pSrc1, const Npp16u *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_16u(const Npp16u *pSrc1, const Npp16u *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_16s_Ctx(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_16s(const Npp16s *pSrc1, const Npp16s *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_16sc_Ctx(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_16sc(const Npp16sc *pSrc1, const Npp16sc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_32u_Ctx(const Npp32u *pSrc1, const Npp32u *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_32u(const Npp32u *pSrc1, const Npp32u *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_32s_Ctx(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_32s(const Npp32s *pSrc1, const Npp32s *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_32sc_Ctx(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_32sc(const Npp32sc *pSrc1, const Npp32sc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_64s_Ctx(const Npp64s *pSrc1, const Npp64s *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_64s(const Npp64s *pSrc1, const Npp64s *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_64sc_Ctx(const Npp64sc *pSrc1, const Npp64sc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_64sc(const Npp64sc *pSrc1, const Npp64sc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_32f_Ctx(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_32fc_Ctx(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_32fc(const Npp32fc *pSrc1, const Npp32fc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_64f_Ctx(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                   lupine_npp_length_t nLength, Npp64f *pDst,
                                   Npp8u *pDeviceBuffer,
                                   NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                               lupine_npp_length_t nLength, Npp64f *pDst,
                               Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageError_64fc_Ctx(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                    lupine_npp_length_t nLength, Npp64f *pDst,
                                    Npp8u *pDeviceBuffer,
                                    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageError_64fc(const Npp64fc *pSrc1, const Npp64fc *pSrc2,
                                lupine_npp_length_t nLength, Npp64f *pDst,
                                Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_8u_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_8u(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_8s_Ctx(lupine_npp_length_t nLength,
                                     lupine_npp_length_t *hpBufferSize,
                                     NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_8s(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_16u_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_16u(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_16s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_16sc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_16sc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_32u_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_32u(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_32s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_32sc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_32sc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_64s_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_64s(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_64sc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_64sc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_32f(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_32fc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_32fc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                      lupine_npp_length_t *hpBufferSize,
                                      NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_64f(lupine_npp_length_t nLength,
                                            lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageErrorGetBufferSize_64fc_Ctx(lupine_npp_length_t nLength,
                                       lupine_npp_length_t *hpBufferSize,
                                       NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus nppsAverageErrorGetBufferSize_64fc(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_8u_Ctx(const Npp8u *pSrc1,
                                          const Npp8u *pSrc2,
                                          lupine_npp_length_t nLength,
                                          Npp64f *pDst, Npp8u *pDeviceBuffer,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_8u(const Npp8u *pSrc1, const Npp8u *pSrc2,
                                      lupine_npp_length_t nLength, Npp64f *pDst,
                                      Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_8s_Ctx(const Npp8s *pSrc1,
                                          const Npp8s *pSrc2,
                                          lupine_npp_length_t nLength,
                                          Npp64f *pDst, Npp8u *pDeviceBuffer,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_8s(const Npp8s *pSrc1, const Npp8s *pSrc2,
                                      lupine_npp_length_t nLength, Npp64f *pDst,
                                      Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_16u_Ctx(const Npp16u *pSrc1,
                                           const Npp16u *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_16u(const Npp16u *pSrc1, const Npp16u *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_16s_Ctx(const Npp16s *pSrc1,
                                           const Npp16s *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_16s(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_16sc_Ctx(const Npp16sc *pSrc1,
                                            const Npp16sc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_16sc(const Npp16sc *pSrc1,
                                        const Npp16sc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32u_Ctx(const Npp32u *pSrc1,
                                           const Npp32u *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32u(const Npp32u *pSrc1, const Npp32u *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32s_Ctx(const Npp32s *pSrc1,
                                           const Npp32s *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32s(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32sc_Ctx(const Npp32sc *pSrc1,
                                            const Npp32sc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32sc(const Npp32sc *pSrc1,
                                        const Npp32sc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64s_Ctx(const Npp64s *pSrc1,
                                           const Npp64s *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64s(const Npp64s *pSrc1, const Npp64s *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64sc_Ctx(const Npp64sc *pSrc1,
                                            const Npp64sc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64sc(const Npp64sc *pSrc1,
                                        const Npp64sc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32f_Ctx(const Npp32f *pSrc1,
                                           const Npp32f *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32fc_Ctx(const Npp32fc *pSrc1,
                                            const Npp32fc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_32fc(const Npp32fc *pSrc1,
                                        const Npp32fc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64f_Ctx(const Npp64f *pSrc1,
                                           const Npp64f *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64fc_Ctx(const Npp64fc *pSrc1,
                                            const Npp64fc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsMaximumRelativeError_64fc(const Npp64fc *pSrc1,
                                        const Npp64fc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_8u_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_8u(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_8s_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_8s(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16u_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16u(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeErrorGetBufferSize_16sc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_16sc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32u_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32u(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeErrorGetBufferSize_32sc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32sc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeErrorGetBufferSize_64sc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64sc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeErrorGetBufferSize_32fc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_32fc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsMaximumRelativeErrorGetBufferSize_64fc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsMaximumRelativeErrorGetBufferSize_64fc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_8u_Ctx(const Npp8u *pSrc1,
                                          const Npp8u *pSrc2,
                                          lupine_npp_length_t nLength,
                                          Npp64f *pDst, Npp8u *pDeviceBuffer,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_8u(const Npp8u *pSrc1, const Npp8u *pSrc2,
                                      lupine_npp_length_t nLength, Npp64f *pDst,
                                      Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_8s_Ctx(const Npp8s *pSrc1,
                                          const Npp8s *pSrc2,
                                          lupine_npp_length_t nLength,
                                          Npp64f *pDst, Npp8u *pDeviceBuffer,
                                          NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_8s(const Npp8s *pSrc1, const Npp8s *pSrc2,
                                      lupine_npp_length_t nLength, Npp64f *pDst,
                                      Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_16u_Ctx(const Npp16u *pSrc1,
                                           const Npp16u *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_16u(const Npp16u *pSrc1, const Npp16u *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_16s_Ctx(const Npp16s *pSrc1,
                                           const Npp16s *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_16s(const Npp16s *pSrc1, const Npp16s *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_16sc_Ctx(const Npp16sc *pSrc1,
                                            const Npp16sc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_16sc(const Npp16sc *pSrc1,
                                        const Npp16sc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32u_Ctx(const Npp32u *pSrc1,
                                           const Npp32u *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32u(const Npp32u *pSrc1, const Npp32u *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32s_Ctx(const Npp32s *pSrc1,
                                           const Npp32s *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32s(const Npp32s *pSrc1, const Npp32s *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32sc_Ctx(const Npp32sc *pSrc1,
                                            const Npp32sc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32sc(const Npp32sc *pSrc1,
                                        const Npp32sc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64s_Ctx(const Npp64s *pSrc1,
                                           const Npp64s *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64s(const Npp64s *pSrc1, const Npp64s *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64sc_Ctx(const Npp64sc *pSrc1,
                                            const Npp64sc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64sc(const Npp64sc *pSrc1,
                                        const Npp64sc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32f_Ctx(const Npp32f *pSrc1,
                                           const Npp32f *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32f(const Npp32f *pSrc1, const Npp32f *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32fc_Ctx(const Npp32fc *pSrc1,
                                            const Npp32fc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_32fc(const Npp32fc *pSrc1,
                                        const Npp32fc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64f_Ctx(const Npp64f *pSrc1,
                                           const Npp64f *pSrc2,
                                           lupine_npp_length_t nLength,
                                           Npp64f *pDst, Npp8u *pDeviceBuffer,
                                           NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64f(const Npp64f *pSrc1, const Npp64f *pSrc2,
                                       lupine_npp_length_t nLength,
                                       Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64fc_Ctx(const Npp64fc *pSrc1,
                                            const Npp64fc *pSrc2,
                                            lupine_npp_length_t nLength,
                                            Npp64f *pDst, Npp8u *pDeviceBuffer,
                                            NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param pSrc1 SEND_ONLY
 * @param pSrc2 SEND_ONLY
 * @param nLength SEND_ONLY
 * @param pDst SEND_ONLY
 * @param pDeviceBuffer SEND_ONLY
 */
NppStatus nppsAverageRelativeError_64fc(const Npp64fc *pSrc1,
                                        const Npp64fc *pSrc2,
                                        lupine_npp_length_t nLength,
                                        Npp64f *pDst, Npp8u *pDeviceBuffer);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_8u_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_8u(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_8s_Ctx(lupine_npp_length_t nLength,
                                             lupine_npp_length_t *hpBufferSize,
                                             NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_8s(lupine_npp_length_t nLength,
                                         lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16u_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16u(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeErrorGetBufferSize_16sc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_16sc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32u_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32u(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeErrorGetBufferSize_32sc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32sc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64s_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64s(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeErrorGetBufferSize_64sc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64sc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeErrorGetBufferSize_32fc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_32fc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64f_Ctx(lupine_npp_length_t nLength,
                                              lupine_npp_length_t *hpBufferSize,
                                              NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64f(lupine_npp_length_t nLength,
                                          lupine_npp_length_t *hpBufferSize);
/**
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 * @param nppStreamCtx SEND_ONLY
 */
NppStatus nppsAverageRelativeErrorGetBufferSize_64fc_Ctx(
    lupine_npp_length_t nLength, lupine_npp_length_t *hpBufferSize,
    NppStreamContext nppStreamCtx);
/**
 * @guard NPP_VERSION < 13000
 * @param nLength SEND_ONLY
 * @param hpBufferSize SEND_RECV
 */
NppStatus
nppsAverageRelativeErrorGetBufferSize_64fc(lupine_npp_length_t nLength,
                                           lupine_npp_length_t *hpBufferSize);
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp8u* nppsMalloc_8u(lupine_npp_length_t nSize) {
  Npp8u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp8u), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp8s* nppsMalloc_8s(lupine_npp_length_t nSize) {
  Npp8s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp8s), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp16u* nppsMalloc_16u(lupine_npp_length_t nSize) {
  Npp16u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp16u), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp16s* nppsMalloc_16s(lupine_npp_length_t nSize) {
  Npp16s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp16s), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp16sc* nppsMalloc_16sc(lupine_npp_length_t nSize) {
  Npp16sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp16sc), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp32u* nppsMalloc_32u(lupine_npp_length_t nSize) {
  Npp32u* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp32u), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp32s* nppsMalloc_32s(lupine_npp_length_t nSize) {
  Npp32s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp32s), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp32sc* nppsMalloc_32sc(lupine_npp_length_t nSize) {
  Npp32sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp32sc), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp32f* nppsMalloc_32f(lupine_npp_length_t nSize) {
  Npp32f* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp32f), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp32fc* nppsMalloc_32fc(lupine_npp_length_t nSize) {
  Npp32fc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp32fc), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp64s* nppsMalloc_64s(lupine_npp_length_t nSize) {
  Npp64s* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp64s), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp64sc* nppsMalloc_64sc(lupine_npp_length_t nSize) {
  Npp64sc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp64sc), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp64f* nppsMalloc_64f(lupine_npp_length_t nSize) {
  Npp64f* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp64f), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @param nSize SEND_ONLY
 */
// clang-format off
Npp64fc* nppsMalloc_64fc(lupine_npp_length_t nSize) {
  Npp64fc* return_value = LUPINE_GENERATED_CALL();
  if (return_value != nullptr) {
    lupine_note_deviceptr_allocation(
        reinterpret_cast<unsigned long long>(return_value), nSize * sizeof(Npp64fc), conn);
  }
  return return_value;
}
// clang-format on
/**
 * @routingkey HANDLE pValues
 * @param pValues SEND_ONLY
 */
void nppsFree(void *pValues) {
  LUPINE_GENERATED_CALL();
  lupine_forget_deviceptr_owner(reinterpret_cast<unsigned long long>(pValues));
}
