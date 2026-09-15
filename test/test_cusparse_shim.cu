// Exercises the cuSPARSE shim end to end against the remote device: handles
// and pointer modes, generic SpMV, SpMM, SpVV, Axpby and SpGEMM, the legacy
// csrgeam2 and Csr2cscEx2 paths, descriptor getters and attributes, status
// strings, error results and, when two GPUs are present, one handle per
// device, each result checked against a CPU reference.
#define DISABLE_CUSPARSE_DEPRECATED
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_CUSPARSE(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = (call);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "%s failed: %s\n", #call, cusparseGetErrorName(status)); \
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

// A 4x4 matrix with two entries per row:
//   1 0 2 0
//   0 3 0 4
//   5 0 6 0
//   0 7 0 8
const int kRows = 4;
const int kNnz = 8;
const std::vector<int> kRowPtr = {0, 2, 4, 6, 8};
const std::vector<int> kColInd = {0, 2, 1, 3, 0, 2, 1, 3};
const std::vector<double> kValues = {1, 2, 3, 4, 5, 6, 7, 8};

template <typename T>
static std::vector<double> dense(const std::vector<T> &v) {
  std::vector<double> a(kRows * kRows, 0.0);
  for (int i = 0; i < kRows; ++i) {
    for (int k = kRowPtr[i]; k < kRowPtr[i + 1]; ++k) {
      a[i * kRows + kColInd[k]] = static_cast<double>(v[k]);
    }
  }
  return a;
}

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
  if (count != 0) {
    cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  return host;
}

static int near(double got, double want, const char *what) {
  if (std::fabs(got - want) > 1e-4 * (1.0 + std::fabs(want))) {
    fprintf(stderr, "%s = %g, expected %g\n", what, got, want);
    return 1;
  }
  return 0;
}

// y = alpha * A * x + beta * y in double precision, with the scalars first on
// the host and then in device memory.
static int test_spmv(cusparseHandle_t handle) {
  const std::vector<double> x = {1, -1, 2, 0.5};
  const std::vector<double> y0 = {4, 3, 2, 1};
  const std::vector<double> a = dense(kValues);
  const double alpha = 2.0, beta = 0.5;

  int *row = upload(kRowPtr);
  int *col = upload(kColInd);
  double *val = upload(kValues);
  double *dx = upload(x);
  double *dy = upload(y0);
  EXPECT(row && col && val && dx && dy);

  cusparseSpMatDescr_t mat = nullptr;
  CHECK_CUSPARSE(cusparseCreateCsr(&mat, kRows, kRows, kNnz, row, col, val,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
  CHECK_CUSPARSE(cusparseCreateDnVec(&vx, kRows, dx, CUDA_R_64F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vy, kRows, dy, CUDA_R_64F));

  for (int device_scalars = 0; device_scalars < 2; ++device_scalars) {
    const double *alpha_ptr = &alpha, *beta_ptr = &beta;
    double *dalpha = nullptr, *dbeta = nullptr;
    if (device_scalars) {
      dalpha = upload(std::vector<double>{alpha});
      dbeta = upload(std::vector<double>{beta});
      EXPECT(dalpha && dbeta);
      alpha_ptr = dalpha;
      beta_ptr = dbeta;
      CHECK_CUSPARSE(
          cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    }
    CHECK_CUDA(cudaMemcpy(dy, y0.data(), kRows * sizeof(double),
                          cudaMemcpyHostToDevice));
    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_ptr, mat, vx, beta_ptr,
        vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));
    void *buffer = nullptr;
    CHECK_CUDA(cudaMalloc(&buffer, buffer_size + 1));
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                alpha_ptr, mat, vx, beta_ptr, vy, CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    const std::vector<double> y = download(dy, kRows);
    // cuSPARSE 12.3.1 through 12.5.1 compute a wrong device-scalar CSR SpMV
    // natively; 12.5.4 and 11.x are correct.
    int version = 0;
    CHECK_CUSPARSE(cusparseGetVersion(handle, &version));
    const bool native_bug =
        device_scalars && version >= 12000 && version < 12504;
    for (int i = 0; i < kRows && !native_bug; ++i) {
      double want = beta * y0[i];
      for (int j = 0; j < kRows; ++j) {
        want += alpha * a[i * kRows + j] * x[j];
      }
      if (near(y[i], want, device_scalars ? "SpMV device scalars" : "SpMV")) {
        return 1;
      }
    }
    CHECK_CUDA(cudaFree(buffer));
    if (device_scalars) {
      cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_HOST;
      CHECK_CUSPARSE(cusparseGetPointerMode(handle, &mode));
      EXPECT(mode == CUSPARSE_POINTER_MODE_DEVICE);
      CHECK_CUSPARSE(
          cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
      CHECK_CUDA(cudaFree(dalpha));
      CHECK_CUDA(cudaFree(dbeta));
    }
  }

  // The descriptor reports what it was created with, and the fill mode
  // attribute round-trips through a host buffer.
  int64_t rows = 0, cols = 0, nnz = 0;
  CHECK_CUSPARSE(cusparseSpMatGetSize(mat, &rows, &cols, &nnz));
  EXPECT(rows == kRows && cols == kRows && nnz == kNnz);
  void *got_row = nullptr, *got_col = nullptr, *got_val = nullptr;
  cusparseIndexType_t row_type, col_type;
  cusparseIndexBase_t base;
  cudaDataType value_type;
  CHECK_CUSPARSE(cusparseCsrGet(mat, &rows, &cols, &nnz, &got_row, &got_col,
                                &got_val, &row_type, &col_type, &base,
                                &value_type));
  EXPECT(got_row == row && got_col == col && got_val == val);
  EXPECT(row_type == CUSPARSE_INDEX_32I && value_type == CUDA_R_64F);
  cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(mat, CUSPARSE_SPMAT_FILL_MODE, &fill,
                                           sizeof(fill)));
  fill = CUSPARSE_FILL_MODE_LOWER;
  CHECK_CUSPARSE(cusparseSpMatGetAttribute(mat, CUSPARSE_SPMAT_FILL_MODE, &fill,
                                           sizeof(fill)));
  EXPECT(fill == CUSPARSE_FILL_MODE_UPPER);
  void *values = nullptr;
  CHECK_CUSPARSE(cusparseDnVecGetValues(vy, &values));
  EXPECT(values == dy);

  CHECK_CUSPARSE(cusparseDestroyDnVec(vx));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vy));
  CHECK_CUSPARSE(cusparseDestroySpMat(mat));
  for (void *p :
       {(void *)row, (void *)col, (void *)val, (void *)dx, (void *)dy}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("SpMV: passed\n");
  return 0;
}

// C = alpha * A * B + beta * C in single precision, B dense 4x2.
static int test_spmm(cusparseHandle_t handle) {
  std::vector<float> values(kValues.begin(), kValues.end());
  const std::vector<float> b = {1, 2, 3, 4, 5, 6, 7, 8};
  const int b_cols = 2;
  const std::vector<double> a = dense(values);
  const float alpha = 1.5f, beta = 0.0f;

  int *row = upload(kRowPtr);
  int *col = upload(kColInd);
  float *val = upload(values);
  float *db = upload(b);
  float *dc = upload(std::vector<float>(kRows * b_cols, 0.0f));
  EXPECT(row && col && val && db && dc);

  cusparseSpMatDescr_t mat = nullptr;
  CHECK_CUSPARSE(cusparseCreateCsr(&mat, kRows, kRows, kNnz, row, col, val,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnMatDescr_t mb = nullptr, mc = nullptr;
  CHECK_CUSPARSE(cusparseCreateDnMat(&mb, kRows, b_cols, kRows, db, CUDA_R_32F,
                                     CUSPARSE_ORDER_COL));
  CHECK_CUSPARSE(cusparseCreateDnMat(&mc, kRows, b_cols, kRows, dc, CUDA_R_32F,
                                     CUSPARSE_ORDER_COL));
  size_t buffer_size = 0;
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, mb, &beta, mc, CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));
  void *buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size + 1));
  CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, mb,
                              &beta, mc, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                              buffer));
  const std::vector<float> c = download(dc, kRows * b_cols);
  for (int j = 0; j < b_cols; ++j) {
    for (int i = 0; i < kRows; ++i) {
      double want = 0.0;
      for (int k = 0; k < kRows; ++k) {
        want += alpha * a[i * kRows + k] * b[j * kRows + k];
      }
      if (near(c[j * kRows + i], want, "SpMM")) {
        return 1;
      }
    }
  }
  CHECK_CUSPARSE(cusparseDestroyDnMat(mb));
  CHECK_CUSPARSE(cusparseDestroyDnMat(mc));
  CHECK_CUSPARSE(cusparseDestroySpMat(mat));
  for (void *p : {(void *)row, (void *)col, (void *)val, (void *)db, (void *)dc,
                  buffer}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("SpMM: passed\n");
  return 0;
}

// Axpby's scalars are sized by the vector's value type, and SpVV returns its
// result through a pointer-mode scalar.
static int test_vectors(cusparseHandle_t handle) {
  const std::vector<int> indices = {0, 3, 5};
  const std::vector<float> sparse = {2, -1, 4};
  const std::vector<float> y0 = {1, 1, 1, 1, 1, 1};
  int *di = upload(indices);
  float *dv = upload(sparse);
  float *dy = upload(y0);
  EXPECT(di && dv && dy);
  cusparseSpVecDescr_t vx = nullptr;
  CHECK_CUSPARSE(cusparseCreateSpVec(&vx, 6, 3, di, dv, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnVecDescr_t vy = nullptr;
  CHECK_CUSPARSE(cusparseCreateDnVec(&vy, 6, dy, CUDA_R_32F));

  const float alpha = 3.0f, beta = 2.0f;
  CHECK_CUSPARSE(cusparseAxpby(handle, &alpha, vx, &beta, vy));
  std::vector<float> want = {8, 2, 2, -1, 2, 14};
  const std::vector<float> y = download(dy, 6);
  for (int i = 0; i < 6; ++i) {
    if (near(y[i], want[i], "Axpby")) {
      return 1;
    }
  }

  float dot = 0.0f;
  size_t buffer_size = 0;
  CHECK_CUSPARSE(cusparseSpVV_bufferSize(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE, vx,
                                         vy, &dot, CUDA_R_32F, &buffer_size));
  void *buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size + 1));
  CHECK_CUSPARSE(cusparseSpVV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, vx, vy,
                              &dot, CUDA_R_32F, buffer));
  if (near(dot, 2 * 8 + -1 * -1 + 4 * 14, "SpVV host result")) {
    return 1;
  }
  float *ddot = upload(std::vector<float>{0.0f});
  EXPECT(ddot);
  CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
  CHECK_CUSPARSE(cusparseSpVV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, vx, vy,
                              ddot, CUDA_R_32F, buffer));
  CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  if (near(download(ddot, 1)[0], 2 * 8 + -1 * -1 + 4 * 14,
           "SpVV device result")) {
    return 1;
  }
  CHECK_CUSPARSE(cusparseDestroySpVec(vx));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vy));
  for (void *p : {(void *)di, (void *)dv, (void *)dy, (void *)ddot, buffer}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("Axpby and SpVV: passed\n");
  return 0;
}

// C = A * A: the work estimation and compute steps first ask for their buffer
// sizes with a null buffer, then run with one.
static int test_spgemm(cusparseHandle_t handle) {
  std::vector<float> values(kValues.begin(), kValues.end());
  const std::vector<double> a = dense(values);
  const float alpha = 1.0f, beta = 0.0f;
  int *row = upload(kRowPtr);
  int *col = upload(kColInd);
  float *val = upload(values);
  int *row_c = upload(std::vector<int>(kRows + 1, 0));
  EXPECT(row && col && val && row_c);

  cusparseSpMatDescr_t ma = nullptr, mc = nullptr;
  CHECK_CUSPARSE(cusparseCreateCsr(&ma, kRows, kRows, kNnz, row, col, val,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateCsr(
      &mc, kRows, kRows, 0, nullptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseSpGEMMDescr_t descr = nullptr;
  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&descr));
  const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t size1 = 0, size2 = 0;
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
      handle, op, op, &alpha, ma, ma, &beta, mc, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, descr, &size1, nullptr));
  void *buffer1 = nullptr;
  CHECK_CUDA(cudaMalloc(&buffer1, size1 + 1));
  CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
      handle, op, op, &alpha, ma, ma, &beta, mc, CUDA_R_32F,
      CUSPARSE_SPGEMM_DEFAULT, descr, &size1, buffer1));
  CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, ma, ma, &beta,
                                        mc, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                        descr, &size2, nullptr));
  void *buffer2 = nullptr;
  CHECK_CUDA(cudaMalloc(&buffer2, size2 + 1));
  CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, ma, ma, &beta,
                                        mc, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                        descr, &size2, buffer2));
  int64_t rows = 0, cols = 0, nnz_c = 0;
  CHECK_CUSPARSE(cusparseSpMatGetSize(mc, &rows, &cols, &nnz_c));
  int *col_c = nullptr;
  float *val_c = nullptr;
  CHECK_CUDA(cudaMalloc(&col_c, nnz_c * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&val_c, nnz_c * sizeof(float)));
  CHECK_CUSPARSE(cusparseCsrSetPointers(mc, row_c, col_c, val_c));
  CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, op, op, &alpha, ma, ma, &beta, mc,
                                     CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
                                     descr));

  const std::vector<int> rc = download(row_c, kRows + 1);
  const std::vector<int> cc = download(col_c, nnz_c);
  const std::vector<float> vc = download(val_c, nnz_c);
  std::vector<double> got(kRows * kRows, 0.0);
  for (int i = 0; i < kRows; ++i) {
    for (int k = rc[i]; k < rc[i + 1]; ++k) {
      got[i * kRows + cc[k]] += vc[k];
    }
  }
  for (int i = 0; i < kRows; ++i) {
    for (int j = 0; j < kRows; ++j) {
      double want = 0.0;
      for (int k = 0; k < kRows; ++k) {
        want += a[i * kRows + k] * a[k * kRows + j];
      }
      if (near(got[i * kRows + j], want, "SpGEMM")) {
        return 1;
      }
    }
  }
  CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(descr));
  CHECK_CUSPARSE(cusparseDestroySpMat(ma));
  CHECK_CUSPARSE(cusparseDestroySpMat(mc));
  for (void *p : {(void *)row, (void *)col, (void *)val, (void *)row_c,
                  (void *)col_c, (void *)val_c, buffer1, buffer2}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("SpGEMM: passed (nnz %lld)\n", static_cast<long long>(nnz_c));
  return 0;
}

// C = alpha * A + beta * I through the legacy csrgeam2 calls, whose nnz comes
// back through a pointer-mode scalar, then C's transpose through Csr2cscEx2.
static int test_legacy(cusparseHandle_t handle) {
  std::vector<float> values(kValues.begin(), kValues.end());
  const std::vector<int> id_row = {0, 1, 2, 3, 4};
  const std::vector<int> id_col = {0, 1, 2, 3};
  const std::vector<float> id_val = {1, 1, 1, 1};
  const float alpha = 2.0f, beta = -1.0f;

  cusparseMatDescr_t descr = nullptr;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
  CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  EXPECT(cusparseGetMatType(descr) == CUSPARSE_MATRIX_TYPE_GENERAL);
  EXPECT(cusparseGetMatIndexBase(descr) == CUSPARSE_INDEX_BASE_ZERO);

  int *row_a = upload(kRowPtr);
  int *col_a = upload(kColInd);
  float *val_a = upload(values);
  int *row_b = upload(id_row);
  int *col_b = upload(id_col);
  float *val_b = upload(id_val);
  int *row_c = upload(std::vector<int>(kRows + 1, 0));
  EXPECT(row_a && col_a && val_a && row_b && col_b && val_b && row_c);

  size_t buffer_size = 0;
  CHECK_CUSPARSE(cusparseScsrgeam2_bufferSizeExt(
      handle, kRows, kRows, &alpha, descr, kNnz, val_a, row_a, col_a, &beta,
      descr, kRows, val_b, row_b, col_b, descr, nullptr, row_c, nullptr,
      &buffer_size));
  void *buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&buffer, buffer_size + 1));
  int nnz_c = -1;
  CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, kRows, kRows, descr, kNnz, row_a,
                                      col_a, descr, kRows, row_b, col_b, descr,
                                      row_c, &nnz_c, buffer));
  EXPECT(nnz_c == kNnz);
  int *col_c = nullptr;
  float *val_c = nullptr;
  CHECK_CUDA(cudaMalloc(&col_c, nnz_c * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&val_c, nnz_c * sizeof(float)));
  CHECK_CUSPARSE(cusparseScsrgeam2(
      handle, kRows, kRows, &alpha, descr, kNnz, val_a, row_a, col_a, &beta,
      descr, kRows, val_b, row_b, col_b, descr, val_c, row_c, col_c, buffer));

  std::vector<double> want = dense(values);
  for (double &v : want) {
    v *= alpha;
  }
  for (int i = 0; i < kRows; ++i) {
    want[i * kRows + i] += beta;
  }
  const std::vector<int> rc = download(row_c, kRows + 1);
  const std::vector<int> cc = download(col_c, nnz_c);
  const std::vector<float> vc = download(val_c, nnz_c);
  std::vector<double> got(kRows * kRows, 0.0);
  for (int i = 0; i < kRows; ++i) {
    for (int k = rc[i]; k < rc[i + 1]; ++k) {
      got[i * kRows + cc[k]] = vc[k];
    }
  }
  for (int i = 0; i < kRows * kRows; ++i) {
    if (near(got[i], want[i], "csrgeam2")) {
      return 1;
    }
  }

  int *csc_col = upload(std::vector<int>(kRows + 1, 0));
  int *csc_row = nullptr;
  float *csc_val = nullptr;
  CHECK_CUDA(cudaMalloc(&csc_row, nnz_c * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&csc_val, nnz_c * sizeof(float)));
  size_t csc_buffer_size = 0;
  CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
      handle, kRows, kRows, nnz_c, val_c, row_c, col_c, csc_val, csc_col,
      csc_row, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1, &csc_buffer_size));
  void *csc_buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&csc_buffer, csc_buffer_size + 1));
  CHECK_CUSPARSE(cusparseCsr2cscEx2(
      handle, kRows, kRows, nnz_c, val_c, row_c, col_c, csc_val, csc_col,
      csc_row, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG1, csc_buffer));
  const std::vector<int> ccol = download(csc_col, kRows + 1);
  const std::vector<int> crow = download(csc_row, nnz_c);
  const std::vector<float> cval = download(csc_val, nnz_c);
  std::vector<double> from_csc(kRows * kRows, 0.0);
  for (int j = 0; j < kRows; ++j) {
    for (int k = ccol[j]; k < ccol[j + 1]; ++k) {
      from_csc[crow[k] * kRows + j] = cval[k];
    }
  }
  for (int i = 0; i < kRows * kRows; ++i) {
    if (near(from_csc[i], want[i], "Csr2cscEx2")) {
      return 1;
    }
  }

  CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
  for (void *p : {(void *)row_a, (void *)col_a, (void *)val_a, (void *)row_b,
                  (void *)col_b, (void *)val_b, (void *)row_c, (void *)col_c,
                  (void *)val_c, buffer, (void *)csc_col, (void *)csc_row,
                  (void *)csc_val, csc_buffer}) {
    CHECK_CUDA(cudaFree(p));
  }
  printf("csrgeam2 and Csr2cscEx2: passed\n");
  return 0;
}

// One handle per device; a descriptor made on one device answers its getters
// after the current device moves away.
static int test_multi_device() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    printf("multi-device: skipped, one device\n");
    return 0;
  }
  for (int device = 0; device < 2; ++device) {
    CHECK_CUDA(cudaSetDevice(device));
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    if (test_spmv(handle)) {
      return 1;
    }
    int *row = upload(kRowPtr);
    EXPECT(row);
    cusparseSpMatDescr_t mat = nullptr;
    CHECK_CUSPARSE(cusparseCreateCsr(&mat, kRows, kRows, kNnz, row, row, row,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUDA(cudaSetDevice(1 - device));
    int64_t rows = 0, cols = 0, nnz = 0;
    CHECK_CUSPARSE(cusparseSpMatGetSize(mat, &rows, &cols, &nnz));
    EXPECT(nnz == kNnz);
    CHECK_CUSPARSE(cusparseDestroySpMat(mat));
    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaFree(row));
    CHECK_CUSPARSE(cusparseDestroy(handle));
  }
  CHECK_CUDA(cudaSetDevice(0));
  printf("multi-device: passed\n");
  return 0;
}

int main() {
  cusparseHandle_t handle = nullptr;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  int version = 0;
  CHECK_CUSPARSE(cusparseGetVersion(handle, &version));
  int major = 0;
  CHECK_CUSPARSE(cusparseGetProperty(MAJOR_VERSION, &major));
  EXPECT(version > 0 && major == version / 1000);
  EXPECT(strcmp(cusparseGetErrorName(CUSPARSE_STATUS_INVALID_VALUE),
                "CUSPARSE_STATUS_INVALID_VALUE") == 0);
  EXPECT(strlen(cusparseGetErrorString(CUSPARSE_STATUS_INVALID_VALUE)) > 0);
  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUSPARSE(cusparseSetStream(handle, stream));
  cudaStream_t got_stream = nullptr;
  CHECK_CUSPARSE(cusparseGetStream(handle, &got_stream));
  EXPECT(got_stream == stream);

  // The server's library answers the error, and a client callback cannot
  // cross to it.
  cusparseSpMatDescr_t bad = nullptr;
  EXPECT(cusparseCreateCsr(&bad, -1, kRows, kNnz, nullptr, nullptr, nullptr,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F) == CUSPARSE_STATUS_INVALID_VALUE);
  // The native library accepts the callback; a remote library cannot call a
  // client function, so the shim refuses it.
  const cusparseStatus_t logger = cusparseLoggerSetCallback(nullptr);
  EXPECT(logger == CUSPARSE_STATUS_SUCCESS ||
         logger == CUSPARSE_STATUS_NOT_SUPPORTED);

  if (test_spmv(handle) || test_spmm(handle) || test_vectors(handle) ||
      test_spgemm(handle) || test_legacy(handle)) {
    return 1;
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (test_multi_device()) {
    return 1;
  }
  printf("cusparse shim: all checks passed (cuSPARSE %d)\n", version);
  return 0;
}
