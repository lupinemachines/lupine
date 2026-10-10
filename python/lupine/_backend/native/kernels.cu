// Prototype kernels. All tensor arithmetic executes on the GPU; host code
// supplies only sizes, strides, and scalar arguments. PTX is built on Linux.
#include <stdint.h>

__device__ int64_t offset(int64_t index, const int64_t *desc, int dims) {
  int64_t result = 0;
  for (int axis = dims - 1; axis >= 0; --axis) {
    result += (index % desc[axis]) * desc[8 + axis];
    index /= desc[axis];
  }
  return result;
}
__device__ float read(const void *p, int64_t i, int dtype) {
  return dtype == 0 ? ((const float *)p)[i] : (float)((const int64_t *)p)[i];
}
__device__ void write(void *p, int64_t i, int dtype, float value) {
  if (dtype == 0)
    ((float *)p)[i] = value;
  else
    ((int64_t *)p)[i] = (int64_t)value;
}
extern "C" __global__ void copy_tensor(void *out, const void *in,
                                       const int64_t *od, const int64_t *id,
                                       int dims, int64_t n, int ot, int it) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t oi = offset(i, od, dims), ii = offset(i, id, dims);
  if (ot == 1 && it == 1)
    ((int64_t *)out)[oi] = ((const int64_t *)in)[ii];
  else
    write(out, oi, ot, read(in, ii, it));
}
extern "C" __global__ void fill_tensor(void *out, const int64_t *desc, int dims,
                                       int64_t n, int dtype, float value,
                                       int64_t integer) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t oi = offset(i, desc, dims);
  if (dtype == 1)
    ((int64_t *)out)[oi] = integer;
  else
    ((float *)out)[oi] = value;
}
extern "C" __global__ void arange_tensor(void *out, int64_t n, int dtype,
                                         float start, float step,
                                         int64_t istart, int64_t istep) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  if (dtype == 1)
    ((int64_t *)out)[i] = istart + i * istep;
  else
    ((float *)out)[i] = start + (float)i * step;
}
extern "C" __global__ void binary_tensor(float *out, const float *a,
                                         const float *b, const int64_t *ad,
                                         const int64_t *bd, int dims, int64_t n,
                                         int op, int scalar, float value,
                                         float alpha) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = a[offset(i, ad, dims)];
  float y = scalar ? value : b[offset(i, bd, dims)];
  out[i] = op == 0   ? x + alpha * y
           : op == 1 ? x - alpha * y
           : op == 2 ? x * y
                     : x / y;
}
extern "C" __global__ void unary_tensor(float *out, const float *in,
                                        const int64_t *desc, int dims,
                                        int64_t n, int op) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = in[offset(i, desc, dims)];
  out[i] = op == 0 ? -x : (x > 0.f ? x : 0.f);
}
extern "C" __global__ void matmul_tensor(float *out, const float *a,
                                         const float *b, int64_t m, int64_t n,
                                         int64_t k, int64_t as0, int64_t as1,
                                         int64_t bs0, int64_t bs1) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m * n)
    return;
  int64_t row = i / n, col = i % n;
  float value = 0.f;
  for (int64_t j = 0; j < k; ++j)
    value += a[row * as0 + j * as1] * b[j * bs0 + col * bs1];
  out[i] = value;
}
extern "C" __global__ void sum_tensor(float *out, const float *in,
                                      const int64_t *outer,
                                      const int64_t *inner, int odims,
                                      int idims, int64_t on, int64_t rn,
                                      int mean) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= on)
    return;
  float value = 0.f;
  int64_t base = offset(i, outer, odims);
  for (int64_t j = 0; j < rn; ++j)
    value += in[base + offset(j, inner, idims)];
  out[i] = mean ? value / (float)rn : value;
}
