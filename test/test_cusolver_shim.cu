// Exercises the cuSOLVER and cuSOLVERMg shims end to end against the remote
// devices: dense LU, Cholesky, QR, symmetric eigenvalues through the legacy
// and generic APIs with a host workspace, Jacobi SVD with its info handle,
// iterative refinement with its residual history, a cusolverSp host solve and
// reordering, the low-level host LU with its extracted factors, cusolverRf
// refactorization single and batched, and a cuSOLVERMg eigensolve across the
// devices present, each result checked against a CPU or cross-API reference.
#define DISABLE_CUSOLVER_DEPRECATED
#define DISABLE_CUSOLVERMG_DEPRECATED
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverMg.h>
#include <cusolverRf.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUSOLVER(call)                                                   \
  do {                                                                         \
    cusolverStatus_t status = (call);                                          \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUSPARSE(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = (call);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

const int kN = 4;

template <typename T> static T *upload(const std::vector<T> &host) {
  T *device = nullptr;
  if (cudaMalloc(&device, host.size() * sizeof(T)) != cudaSuccess ||
      cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    return nullptr;
  }
  return device;
}

template <typename T>
static std::vector<T> download(const T *device, size_t count) {
  std::vector<T> host(count);
  cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost);
  return host;
}

static bool near(double got, double want, const char *what) {
  if (std::fabs(got - want) > 1e-6 * (1.0 + std::fabs(want))) {
    fprintf(stderr, "%s = %.12g, expected %.12g\n", what, got, want);
    return true;
  }
  return false;
}

// Column-major dense matrices.
static std::vector<double> general_matrix() {
  return {4, 1, 0, 2, 1, 5, 1, 0, 0, 2, 6, 1, 1, 0, 1, 7};
}

static std::vector<double> symmetric_matrix() {
  const std::vector<double> a = general_matrix();
  std::vector<double> s(kN * kN);
  for (int j = 0; j < kN; ++j) {
    for (int i = 0; i < kN; ++i) {
      s[i + j * kN] = a[i + j * kN] + a[j + i * kN];
    }
  }
  return s;
}

static std::vector<double> multiply(const std::vector<double> &a, int rows,
                                    int cols, const std::vector<double> &x) {
  std::vector<double> y(rows, 0.0);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      y[i] += a[i + j * rows] * x[j];
    }
  }
  return y;
}

const std::vector<double> kSolution = {1, -2, 3, 0.5};

static int test_getrf(cusolverDnHandle_t handle) {
  const std::vector<double> a = general_matrix();
  double *da = upload(a);
  double *db = upload(multiply(a, kN, kN, kSolution));
  int *ipiv = upload(std::vector<int>(kN));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(da && db && ipiv && dinfo);
  int lwork = -1;
  CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(handle, kN, kN, da, kN, &lwork));
  EXPECT(lwork >= 0);
  double *work = upload(std::vector<double>(lwork + 1));
  CHECK_CUSOLVER(cusolverDnDgetrf(handle, kN, kN, da, kN, work, ipiv, dinfo));
  CHECK_CUSOLVER(cusolverDnDgetrs(handle, CUBLAS_OP_N, kN, 1, da, kN, ipiv, db,
                                  kN, dinfo));
  EXPECT(download(dinfo, 1)[0] == 0);
  const std::vector<double> x = download(db, kN);
  for (int i = 0; i < kN; ++i) {
    if (near(x[i], kSolution[i], "getrs")) {
      return 1;
    }
  }
  // The server's library answers the error.
  EXPECT(cusolverDnDpotrf_bufferSize(handle, static_cast<cublasFillMode_t>(7),
                                     kN, da, kN,
                                     &lwork) == CUSOLVER_STATUS_INVALID_VALUE);
  for (void *p :
       {(void *)da, (void *)db, (void *)ipiv, (void *)dinfo, (void *)work}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("getrf/getrs: passed\n");
  return 0;
}

static int test_potrf(cusolverDnHandle_t handle) {
  std::vector<double> spd = symmetric_matrix();
  for (int i = 0; i < kN; ++i) {
    spd[i + i * kN] += 10;
  }
  double *da = upload(spd);
  double *db = upload(multiply(spd, kN, kN, kSolution));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(da && db && dinfo);
  int lwork = -1;
  CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, kN,
                                             da, kN, &lwork));
  double *work = upload(std::vector<double>(lwork + 1));
  CHECK_CUSOLVER(cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, kN, da, kN,
                                  work, lwork, dinfo));
  CHECK_CUSOLVER(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, kN, 1, da, kN,
                                  db, kN, dinfo));
  EXPECT(download(dinfo, 1)[0] == 0);
  const std::vector<double> x = download(db, kN);
  for (int i = 0; i < kN; ++i) {
    if (near(x[i], kSolution[i], "potrs")) {
      return 1;
    }
  }
  for (void *p : {(void *)da, (void *)db, (void *)dinfo, (void *)work}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("potrf/potrs: passed\n");
  return 0;
}

// Q^T A through ormqr reproduces the R that geqrf left in its upper triangle.
static int test_geqrf(cusolverDnHandle_t handle) {
  const std::vector<double> a = general_matrix();
  double *dqr = upload(a);
  double *dc = upload(a);
  double *tau = upload(std::vector<double>(kN));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(dqr && dc && tau && dinfo);
  int lwork = -1;
  CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(handle, kN, kN, dqr, kN, &lwork));
  int lwork_ormqr = -1;
  CHECK_CUSOLVER(cusolverDnDormqr_bufferSize(handle, CUBLAS_SIDE_LEFT,
                                             CUBLAS_OP_T, kN, kN, kN, dqr, kN,
                                             tau, dc, kN, &lwork_ormqr));
  lwork = std::max(lwork, lwork_ormqr);
  double *work = upload(std::vector<double>(lwork + 1));
  CHECK_CUSOLVER(
      cusolverDnDgeqrf(handle, kN, kN, dqr, kN, tau, work, lwork, dinfo));
  CHECK_CUSOLVER(cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, kN, kN,
                                  kN, dqr, kN, tau, dc, kN, work, lwork,
                                  dinfo));
  EXPECT(download(dinfo, 1)[0] == 0);
  const std::vector<double> qr = download(dqr, kN * kN);
  const std::vector<double> r = download(dc, kN * kN);
  for (int j = 0; j < kN; ++j) {
    for (int i = 0; i < kN; ++i) {
      if (near(r[i + j * kN], i <= j ? qr[i + j * kN] : 0.0, "ormqr")) {
        return 1;
      }
    }
  }
  for (void *p :
       {(void *)dqr, (void *)dc, (void *)tau, (void *)dinfo, (void *)work}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("geqrf/ormqr: passed\n");
  return 0;
}

static std::vector<double> legacy_eigenvalues;

// The generic call's host workspace travels both ways; its eigenvalues must
// match the legacy call's and sum to the trace.
static int test_syevd(cusolverDnHandle_t handle) {
  const std::vector<double> s = symmetric_matrix();
  double *da = upload(s);
  double *dw = upload(std::vector<double>(kN));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(da && dw && dinfo);
  int lwork = -1;
  CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER, kN, da, kN,
                                             dw, &lwork));
  double *work = upload(std::vector<double>(lwork + 1));
  CHECK_CUSOLVER(cusolverDnDsyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                  CUBLAS_FILL_MODE_LOWER, kN, da, kN, dw, work,
                                  lwork, dinfo));
  EXPECT(download(dinfo, 1)[0] == 0);
  legacy_eigenvalues = download(dw, kN);
  double trace = 0;
  for (int i = 0; i < kN; ++i) {
    trace += s[i + i * kN];
  }
  if (near(std::accumulate(legacy_eigenvalues.begin(), legacy_eigenvalues.end(),
                           0.0),
           trace, "syevd trace")) {
    return 1;
  }

  cusolverDnParams_t params = nullptr;
  CHECK_CUSOLVER(cusolverDnCreateParams(&params));
  double *da2 = upload(s);
  double *dw2 = upload(std::vector<double>(kN));
  EXPECT(da2 && dw2);
  size_t device_bytes = 0;
  size_t host_bytes = 0;
  CHECK_CUSOLVER(cusolverDnXsyevd_bufferSize(
      handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, kN,
      CUDA_R_64F, da2, kN, CUDA_R_64F, dw2, CUDA_R_64F, &device_bytes,
      &host_bytes));
  void *device_work = nullptr;
  CHECK_CUDA(cudaMalloc(&device_work, device_bytes + 1));
  std::vector<char> host_work(host_bytes + 1);
  CHECK_CUSOLVER(cusolverDnXsyevd(
      handle, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, kN,
      CUDA_R_64F, da2, kN, CUDA_R_64F, dw2, CUDA_R_64F, device_work,
      device_bytes, host_bytes == 0 ? nullptr : host_work.data(), host_bytes,
      dinfo));
  EXPECT(download(dinfo, 1)[0] == 0);
  const std::vector<double> w = download(dw2, kN);
  for (int i = 0; i < kN; ++i) {
    if (near(w[i], legacy_eigenvalues[i], "Xsyevd")) {
      return 1;
    }
  }
  CHECK_CUSOLVER(cusolverDnDestroyParams(params));
  for (void *p : {(void *)da, (void *)dw, (void *)dinfo, (void *)work,
                  (void *)da2, (void *)dw2, device_work}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("syevd and Xsyevd: passed\n");
  return 0;
}

// The squared singular values sum to the squared Frobenius norm, and the
// info handle reports sweeps and a residual as host values.
static int test_gesvdj(cusolverDnHandle_t handle) {
  const std::vector<double> a = general_matrix();
  double *da = upload(a);
  double *ds = upload(std::vector<double>(kN));
  double *du = upload(std::vector<double>(kN * kN));
  double *dv = upload(std::vector<double>(kN * kN));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(da && ds && du && dv && dinfo);
  gesvdjInfo_t info = nullptr;
  CHECK_CUSOLVER(cusolverDnCreateGesvdjInfo(&info));
  CHECK_CUSOLVER(cusolverDnXgesvdjSetTolerance(info, 1e-12));
  CHECK_CUSOLVER(cusolverDnXgesvdjSetMaxSweeps(info, 50));
  int lwork = -1;
  CHECK_CUSOLVER(cusolverDnDgesvdj_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
                                              0, kN, kN, da, kN, ds, du, kN, dv,
                                              kN, &lwork, info));
  double *work = upload(std::vector<double>(lwork + 1));
  CHECK_CUSOLVER(cusolverDnDgesvdj(handle, CUSOLVER_EIG_MODE_VECTOR, 0, kN, kN,
                                   da, kN, ds, du, kN, dv, kN, work, lwork,
                                   dinfo, info));
  EXPECT(download(dinfo, 1)[0] == 0);
  int sweeps = -1;
  double residual = -1;
  CHECK_CUSOLVER(cusolverDnXgesvdjGetSweeps(handle, info, &sweeps));
  CHECK_CUSOLVER(cusolverDnXgesvdjGetResidual(handle, info, &residual));
  EXPECT(sweeps > 0 && residual >= 0);
  const std::vector<double> s = download(ds, kN);
  double norm = 0;
  double sum = 0;
  for (double v : a) {
    norm += v * v;
  }
  for (double v : s) {
    sum += v * v;
  }
  if (near(sum, norm, "gesvdj")) {
    return 1;
  }
  CHECK_CUSOLVER(cusolverDnDestroyGesvdjInfo(info));
  for (void *p : {(void *)da, (void *)ds, (void *)du, (void *)dv, (void *)dinfo,
                  (void *)work}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("gesvdj: passed (%d sweeps)\n", sweeps);
  return 0;
}

// The library keeps the residual history host-side as (maxiters + 1) by 2
// doubles: iteration numbers, then residual norms.
static int test_irs(cusolverDnHandle_t handle) {
  const int n = 64;
  std::vector<double> a(n * n);
  std::vector<double> xref(n);
  for (int j = 0; j < n; ++j) {
    xref[j] = std::sin(j + 1.0);
    for (int i = 0; i < n; ++i) {
      a[i + j * n] = (i == j ? n : 0) + std::cos(i * 3.0 + j) / 4;
    }
  }
  const std::vector<double> b = multiply(a, n, n, xref);
  double *da = upload(a);
  double *db = upload(b);
  double *dx = upload(std::vector<double>(n));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(da && db && dx && dinfo);
  cusolverDnIRSParams_t params = nullptr;
  cusolverDnIRSInfos_t infos = nullptr;
  CHECK_CUSOLVER(cusolverDnIRSParamsCreate(&params));
  CHECK_CUSOLVER(cusolverDnIRSParamsSetSolverPrecisions(params, CUSOLVER_R_64F,
                                                        CUSOLVER_R_32F));
  CHECK_CUSOLVER(cusolverDnIRSParamsSetRefinementSolver(
      params, CUSOLVER_IRS_REFINE_CLASSICAL));
  CHECK_CUSOLVER(cusolverDnIRSParamsSetMaxIters(params, 10));
  CHECK_CUSOLVER(cusolverDnIRSInfosCreate(&infos));
  CHECK_CUSOLVER(cusolverDnIRSInfosRequestResidual(infos));
  size_t lwork = 0;
  CHECK_CUSOLVER(cusolverDnIRSXgesv_bufferSize(handle, params, n, 1, &lwork));
  void *work = nullptr;
  CHECK_CUDA(cudaMalloc(&work, lwork + 1));
  cusolver_int_t niters = 0;
  CHECK_CUSOLVER(cusolverDnIRSXgesv(handle, params, infos, n, 1, da, n, db, n,
                                    dx, n, work, lwork, &niters, dinfo));
  EXPECT(download(dinfo, 1)[0] == 0);
  const std::vector<double> x = download(dx, n);
  for (int i = 0; i < n; ++i) {
    if (near(x[i], xref[i], "IRSXgesv")) {
      return 1;
    }
  }
  cusolver_int_t max_iters = 0;
  cusolver_int_t outer = 0;
  CHECK_CUSOLVER(cusolverDnIRSInfosGetMaxIters(infos, &max_iters));
  CHECK_CUSOLVER(cusolverDnIRSInfosGetOuterNiters(infos, &outer));
  void *history = nullptr;
  CHECK_CUSOLVER(cusolverDnIRSInfosGetResidualHistory(infos, &history));
  EXPECT(history != nullptr && max_iters == 10 && outer >= 1);
  const double *h = static_cast<const double *>(history);
  EXPECT(h[0] == 0 && h[1] == 1 && h[max_iters + 1] >= 0);
  CHECK_CUSOLVER(cusolverDnIRSInfosDestroy(infos));
  CHECK_CUSOLVER(cusolverDnIRSParamsDestroy(params));
  for (void *p : {(void *)da, (void *)db, (void *)dx, (void *)dinfo, work}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("IRS gesv: passed (%d outer iterations)\n", static_cast<int>(outer));
  return 0;
}

// A 4x4 sparse matrix, zero-based CSR:
//   4 1 0 0
//   1 4 1 0
//   0 1 4 1
//   0 0 1 4
const std::vector<int> kRowPtr = {0, 2, 5, 8, 10};
const std::vector<int> kColInd = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
const std::vector<double> kValues = {4, 1, 1, 4, 1, 1, 4, 1, 1, 4};
const int kNnz = 10;

static std::vector<double> sparse_multiply(const std::vector<double> &x) {
  std::vector<double> y(kN, 0.0);
  for (int i = 0; i < kN; ++i) {
    for (int k = kRowPtr[i]; k < kRowPtr[i + 1]; ++k) {
      y[i] += kValues[k] * x[kColInd[k]];
    }
  }
  return y;
}

static int test_sp_host(cusolverSpHandle_t handle, cusparseMatDescr_t descr) {
  const std::vector<double> b = sparse_multiply(kSolution);
  std::vector<double> x(kN, -99);
  int singularity = -2;
  CHECK_CUSOLVER(cusolverSpDcsrlsvqrHost(
      handle, kN, kNnz, descr, kValues.data(), kRowPtr.data(), kColInd.data(),
      b.data(), 1e-12, 0, x.data(), &singularity));
  EXPECT(singularity == -1);
  for (int i = 0; i < kN; ++i) {
    if (near(x[i], kSolution[i], "csrlsvqrHost")) {
      return 1;
    }
  }
  std::vector<int> p(kN, -1);
  CHECK_CUSOLVER(cusolverSpXcsrsymrcmHost(
      handle, kN, kNnz, descr, kRowPtr.data(), kColInd.data(), p.data()));
  std::vector<int> sorted = p;
  std::sort(sorted.begin(), sorted.end());
  for (int i = 0; i < kN; ++i) {
    EXPECT(sorted[i] == i);
  }
  printf("cusolverSp host: passed\n");
  return 0;
}

struct lu_factors {
  std::vector<int> p, q, row_l, col_l, row_u, col_u;
  std::vector<double> val_l, val_u;
};

// The low-level host LU sizes its workspace and extracted factors from what
// analysis, buffer info and nnz reported for the info structure.
static int test_sp_lu_host(cusolverSpHandle_t handle, cusparseMatDescr_t descr,
                           lu_factors *factors) {
  csrluInfoHost_t info = nullptr;
  CHECK_CUSOLVER(cusolverSpCreateCsrluInfoHost(&info));
  CHECK_CUSOLVER(cusolverSpXcsrluAnalysisHost(
      handle, kN, kNnz, descr, kRowPtr.data(), kColInd.data(), info));
  size_t internal_bytes = 0;
  size_t workspace_bytes = 0;
  CHECK_CUSOLVER(cusolverSpDcsrluBufferInfoHost(
      handle, kN, kNnz, descr, kValues.data(), kRowPtr.data(), kColInd.data(),
      info, &internal_bytes, &workspace_bytes));
  std::vector<char> buffer(workspace_bytes + 1);
  CHECK_CUSOLVER(cusolverSpDcsrluFactorHost(
      handle, kN, kNnz, descr, kValues.data(), kRowPtr.data(), kColInd.data(),
      info, 1.0, buffer.data()));
  int position = -2;
  CHECK_CUSOLVER(cusolverSpDcsrluZeroPivotHost(handle, info, 1e-14, &position));
  EXPECT(position == -1);
  const std::vector<double> b = sparse_multiply(kSolution);
  std::vector<double> x(kN, -99);
  CHECK_CUSOLVER(cusolverSpDcsrluSolveHost(handle, kN, b.data(), x.data(), info,
                                           buffer.data()));
  for (int i = 0; i < kN; ++i) {
    if (near(x[i], kSolution[i], "csrluSolveHost")) {
      return 1;
    }
  }
  int nnz_l = -1;
  int nnz_u = -1;
  CHECK_CUSOLVER(cusolverSpXcsrluNnzHost(handle, &nnz_l, &nnz_u, info));
  EXPECT(nnz_l > 0 && nnz_u > 0);
  cusparseMatDescr_t descr_l = nullptr;
  cusparseMatDescr_t descr_u = nullptr;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_l));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_u));
  CHECK_CUSPARSE(cusparseSetMatFillMode(descr_l, CUSPARSE_FILL_MODE_LOWER));
  CHECK_CUSPARSE(cusparseSetMatDiagType(descr_l, CUSPARSE_DIAG_TYPE_UNIT));
  CHECK_CUSPARSE(cusparseSetMatFillMode(descr_u, CUSPARSE_FILL_MODE_UPPER));
  CHECK_CUSPARSE(cusparseSetMatDiagType(descr_u, CUSPARSE_DIAG_TYPE_NON_UNIT));
  factors->p.assign(kN, -1);
  factors->q.assign(kN, -1);
  factors->row_l.assign(kN + 1, -1);
  factors->col_l.assign(nnz_l, -1);
  factors->val_l.assign(nnz_l, 0);
  factors->row_u.assign(kN + 1, -1);
  factors->col_u.assign(nnz_u, -1);
  factors->val_u.assign(nnz_u, 0);
  CHECK_CUSOLVER(cusolverSpDcsrluExtractHost(
      handle, factors->p.data(), factors->q.data(), descr_l,
      factors->val_l.data(), factors->row_l.data(), factors->col_l.data(),
      descr_u, factors->val_u.data(), factors->row_u.data(),
      factors->col_u.data(), info, buffer.data()));
  EXPECT(factors->row_l[kN] == nnz_l && factors->row_u[kN] == nnz_u);
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_l));
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_u));
  CHECK_CUSOLVER(cusolverSpDestroyCsrluInfoHost(info));
  printf("csrlu host: passed (nnz L %d, U %d)\n", nnz_l, nnz_u);
  return 0;
}

// cusolverRf refactors the same pattern from the host LU's factors, solves on
// the device, and hands back malloc'd copies of the factors it extracts.
static int test_rf(const lu_factors &f) {
  const int nnz_l = f.row_l[kN];
  const int nnz_u = f.row_u[kN];
  std::vector<int> row_a = kRowPtr, col_a = kColInd, row_l = f.row_l,
                   col_l = f.col_l, row_u = f.row_u, col_u = f.col_u, p = f.p,
                   q = f.q;
  std::vector<double> val_a = kValues, val_l = f.val_l, val_u = f.val_u;
  // The host LU leaves L's unit diagonal out.
  cusolverRfHandle_t rf = nullptr;
  CHECK_CUSOLVER(cusolverRfCreate(&rf));
  CHECK_CUSOLVER(cusolverRfSetMatrixFormat(rf, CUSOLVERRF_MATRIX_FORMAT_CSR,
                                           CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));
  CHECK_CUSOLVER(cusolverRfSetupHost(
      kN, kNnz, row_a.data(), col_a.data(), val_a.data(), nnz_l, row_l.data(),
      col_l.data(), val_l.data(), nnz_u, row_u.data(), col_u.data(),
      val_u.data(), p.data(), q.data(), rf));
  CHECK_CUSOLVER(cusolverRfAnalyze(rf));
  CHECK_CUSOLVER(cusolverRfRefactor(rf));
  cusolverRfMatrixFormat_t format = CUSOLVERRF_MATRIX_FORMAT_CSC;
  cusolverRfUnitDiagonal_t diag = CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U;
  CHECK_CUSOLVER(cusolverRfGetMatrixFormat(rf, &format, &diag));
  EXPECT(format == CUSOLVERRF_MATRIX_FORMAT_CSR &&
         diag == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L);

  int *dp = upload(p);
  int *dq = upload(q);
  double *dtemp = upload(std::vector<double>(kN));
  double *dx = upload(sparse_multiply(kSolution));
  EXPECT(dp && dq && dtemp && dx);
  CHECK_CUSOLVER(cusolverRfSolve(rf, dp, dq, 1, dtemp, kN, dx, kN));
  std::vector<double> x = download(dx, kN);
  for (int i = 0; i < kN; ++i) {
    if (near(x[i], kSolution[i], "RfSolve")) {
      return 1;
    }
  }
  int got_l = -1;
  int got_u = -1;
  int *h_row_l = nullptr, *h_col_l = nullptr, *h_row_u = nullptr,
      *h_col_u = nullptr;
  double *h_val_l = nullptr, *h_val_u = nullptr;
  CHECK_CUSOLVER(cusolverRfExtractSplitFactorsHost(
      rf, &got_l, &h_row_l, &h_col_l, &h_val_l, &got_u, &h_row_u, &h_col_u,
      &h_val_u));
  EXPECT(got_l == h_row_l[kN] && got_u == h_row_u[kN]);
  free(h_row_l);
  free(h_col_l);
  free(h_val_l);
  free(h_row_u);
  free(h_col_u);
  free(h_val_u);
  CHECK_CUSOLVER(cusolverRfDestroy(rf));

  // Batched setup copies every batch entry's values; native cuSOLVER 12
  // faults in cusolverRfBatchAnalyze on this system, so the batch is only set
  // up.
  const int batch = 2;
  std::vector<double> values(kValues);
  double *batch_values[batch] = {values.data(), values.data()};
  CHECK_CUSOLVER(cusolverRfCreate(&rf));
  CHECK_CUSOLVER(cusolverRfSetMatrixFormat(rf, CUSOLVERRF_MATRIX_FORMAT_CSR,
                                           CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));
  CHECK_CUSOLVER(cusolverRfBatchSetupHost(
      batch, kN, kNnz, row_a.data(), col_a.data(), batch_values, nnz_l,
      row_l.data(), col_l.data(), val_l.data(), nnz_u, row_u.data(),
      col_u.data(), val_u.data(), p.data(), q.data(), rf));
  CHECK_CUSOLVER(cusolverRfDestroy(rf));
  for (void *d : {(void *)dp, (void *)dq, (void *)dtemp, (void *)dx}) {
    CHECK_CUDA(cudaFree(d));
  }
  printf("cusolverRf: passed\n");
  return 0;
}

// A symmetric 8x8 matrix split column-wise across the devices of a 1 by D
// grid; its eigenvalues must match the dense solver's.
static int test_mg() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  devices = std::min(devices, 2);
  const int n = 8;
  std::vector<double> a(n * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[i + j * n] = (i == j ? 10.0 : 0.0) + 1.0 / (1 + i + j);
    }
  }
  const int block = (n + devices - 1) / devices;
  cusolverMgHandle_t handle = nullptr;
  CHECK_CUSOLVER(cusolverMgCreate(&handle));
  std::vector<int> ids(devices);
  std::iota(ids.begin(), ids.end(), 0);
  CHECK_CUSOLVER(cusolverMgDeviceSelect(handle, devices, ids.data()));
  cudaLibMgGrid_t grid = nullptr;
  CHECK_CUSOLVER(cusolverMgCreateDeviceGrid(&grid, 1, devices, ids.data(),
                                            CUDALIBMG_GRID_MAPPING_COL_MAJOR));
  cudaLibMgMatrixDesc_t descr = nullptr;
  CHECK_CUSOLVER(
      cusolverMgCreateMatrixDesc(&descr, n, n, n, block, CUDA_R_64F, grid));
  std::vector<void *> columns(devices);
  for (int d = 0; d < devices; ++d) {
    CHECK_CUDA(cudaSetDevice(d));
    const int first = d * block;
    const int count = std::max(0, std::min(block, n - first));
    CHECK_CUDA(cudaMalloc(&columns[d], n * block * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(columns[d], a.data() + first * n,
                          n * count * sizeof(double), cudaMemcpyHostToDevice));
  }
  CHECK_CUDA(cudaSetDevice(0));
  std::vector<double> w(n, 0);
  int64_t lwork = -1;
  CHECK_CUSOLVER(cusolverMgSyevd_bufferSize(
      handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
      columns.data(), 1, 1, descr, w.data(), CUDA_R_64F, CUDA_R_64F, &lwork));
  EXPECT(lwork >= 0);
  std::vector<void *> work(devices);
  for (int d = 0; d < devices; ++d) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&work[d], (lwork + 1) * sizeof(double)));
  }
  CHECK_CUDA(cudaSetDevice(0));
  int info = -1;
  CHECK_CUSOLVER(cusolverMgSyevd(handle, CUSOLVER_EIG_MODE_VECTOR,
                                 CUBLAS_FILL_MODE_LOWER, n, columns.data(), 1,
                                 1, descr, w.data(), CUDA_R_64F, CUDA_R_64F,
                                 work.data(), lwork, &info));
  EXPECT(info == 0);

  cusolverDnHandle_t dense = nullptr;
  CHECK_CUSOLVER(cusolverDnCreate(&dense));
  double *da = upload(a);
  double *dw = upload(std::vector<double>(n));
  int *dinfo = upload(std::vector<int>(1));
  EXPECT(da && dw && dinfo);
  int dense_lwork = -1;
  CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(dense, CUSOLVER_EIG_MODE_NOVECTOR,
                                             CUBLAS_FILL_MODE_LOWER, n, da, n,
                                             dw, &dense_lwork));
  double *dwork = upload(std::vector<double>(dense_lwork + 1));
  CHECK_CUSOLVER(cusolverDnDsyevd(dense, CUSOLVER_EIG_MODE_NOVECTOR,
                                  CUBLAS_FILL_MODE_LOWER, n, da, n, dw, dwork,
                                  dense_lwork, dinfo));
  const std::vector<double> reference = download(dw, n);
  for (int i = 0; i < n; ++i) {
    if (near(w[i], reference[i], "MgSyevd")) {
      return 1;
    }
  }
  CHECK_CUSOLVER(cusolverDnDestroy(dense));
  for (void *p : {(void *)da, (void *)dw, (void *)dinfo, (void *)dwork}) {
    CHECK_CUDA(cudaFree(p));
  }
  for (int d = 0; d < devices; ++d) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaFree(columns[d]));
    CHECK_CUDA(cudaFree(work[d]));
  }
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUSOLVER(cusolverMgDestroyMatrixDesc(descr));
  CHECK_CUSOLVER(cusolverMgDestroyGrid(grid));
  CHECK_CUSOLVER(cusolverMgDestroy(handle));
  printf("cusolverMg syevd: passed on %d device(s)\n", devices);
  return 0;
}

int main() {
  int version = 0;
  CHECK_CUSOLVER(cusolverGetVersion(&version));
  int major = 0;
  CHECK_CUSOLVER(cusolverGetProperty(MAJOR_VERSION, &major));
  EXPECT(version > 0 && major == version / 1000);

  cusolverDnHandle_t dn = nullptr;
  CHECK_CUSOLVER(cusolverDnCreate(&dn));
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUSOLVER(cusolverDnSetStream(dn, stream));
  cudaStream_t got_stream = nullptr;
  CHECK_CUSOLVER(cusolverDnGetStream(dn, &got_stream));
  EXPECT(got_stream == stream);
  if (test_getrf(dn) || test_potrf(dn) || test_geqrf(dn) || test_syevd(dn) ||
      test_gesvdj(dn) || test_irs(dn)) {
    return 1;
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUSOLVER(cusolverDnDestroy(dn));
  CHECK_CUDA(cudaStreamDestroy(stream));

  cusolverSpHandle_t sp = nullptr;
  CHECK_CUSOLVER(cusolverSpCreate(&sp));
  cusparseMatDescr_t descr = nullptr;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
  lu_factors factors;
  if (test_sp_host(sp, descr) || test_sp_lu_host(sp, descr, &factors) ||
      test_rf(factors)) {
    return 1;
  }
  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
  CHECK_CUSOLVER(cusolverSpDestroy(sp));
  if (test_mg()) {
    return 1;
  }
  printf("cusolver shim: all checks passed (cuSOLVER %d)\n", version);
  return 0;
}
