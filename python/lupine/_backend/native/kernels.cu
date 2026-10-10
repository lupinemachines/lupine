// Prototype kernels. All tensor arithmetic executes on the GPU; host code
// supplies only sizes, strides, and scalar arguments. Converted NVRTC compiles
// this source on macOS; checked-in PTX supports older runtime bundles.
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
           : op == 3 ? x / y
                     : powf(x, y);
}
extern "C" __global__ void unary_tensor(float *out, const float *in,
                                        const int64_t *desc, int dims,
                                        int64_t n, int op) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float x = in[offset(i, desc, dims)];
  switch (op) {
  case 0:
    out[i] = -x;
    break;
  case 1:
    out[i] = x > 0.f ? x : 0.f;
    break;
  case 2:
    out[i] = expf(x);
    break;
  case 3:
    out[i] = logf(x);
    break;
  case 4:
    out[i] = sqrtf(x);
    break;
  case 5:
    out[i] = rsqrtf(x);
    break;
  case 6:
    out[i] = sinf(x);
    break;
  case 7:
    out[i] = cosf(x);
    break;
  case 8:
    out[i] = tanhf(x);
    break;
  case 9:
    out[i] = 1.f / (1.f + expf(-x));
    break;
  case 10:
    out[i] = fabsf(x);
    break;
  case 11:
    out[i] = 1.f / x;
    break;
  }
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

// Counter-based deterministic RNG. No host tensor values are generated.
__device__ uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
extern "C" __global__ void random_tensor(float *out, const int64_t *desc,
                                         int dims, int64_t n, uint64_t seed,
                                         uint64_t counter, int normal,
                                         float first, float second) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float u = ((mix(seed ^ (counter + 2 * i)) >> 40) + 0.5f) * 0x1p-24f;
  float value;
  if (normal) {
    float v = ((mix(seed ^ (counter + 2 * i + 1)) >> 40) + 0.5f) * 0x1p-24f;
    value =
        first + second * sqrtf(-2.f * logf(u)) * cosf(6.283185307179586f * v);
  } else
    value = first + (second - first) * u;
  out[offset(i, desc, dims)] = value;
}
extern "C" __global__ void
activation_backward(float *out, const float *grad, const float *value,
                    const int64_t *gd, const int64_t *vd, int dims, int64_t n,
                    int op, float threshold) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float g = grad[offset(i, gd, dims)], x = value[offset(i, vd, dims)];
  out[i] = op == 0   ? (x > threshold ? g : 0.f)
           : op == 1 ? g * x * (1.f - x)
                     : g * (1.f - x * x);
}
extern "C" __global__ void
softmax_tensor(float *out, const float *in, const int64_t *desc, int dims,
               int64_t n, int64_t width, int64_t axis_stride, int logarithmic) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t base = i - (i / axis_stride % width) * axis_stride;
  float maximum = -__int_as_float(0x7f800000), total = 0.f;
  for (int64_t j = 0; j < width; ++j)
    maximum = fmaxf(maximum, in[offset(base + j * axis_stride, desc, dims)]);
  if (logarithmic == 2 && maximum == -__int_as_float(0x7f800000)) {
    out[i] = 0.f;
    return;
  }
  for (int64_t j = 0; j < width; ++j)
    total += expf(in[offset(base + j * axis_stride, desc, dims)] - maximum);
  float x = in[offset(i, desc, dims)] - maximum;
  out[i] = logarithmic == 1 ? x - logf(total) : expf(x) / total;
}
extern "C" __global__ void
softmax_backward(float *out, const float *grad, const float *value,
                 const int64_t *gd, const int64_t *vd, int dims, int64_t n,
                 int64_t width, int64_t axis_stride, int logarithmic) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int64_t base = i - (i / axis_stride % width) * axis_stride;
  float total = 0.f;
  for (int64_t j = 0; j < width; ++j) {
    int64_t k = base + j * axis_stride;
    float g = grad[offset(k, gd, dims)], v = value[offset(k, vd, dims)];
    total += logarithmic ? g : g * v;
  }
  float g = grad[offset(i, gd, dims)], v = value[offset(i, vd, dims)];
  out[i] = logarithmic ? g - expf(v) * total : v * (g - total);
}

// NCHW grouped convolution and its two gradients. One thread owns each output
// element, so every accumulation is deterministic and needs no GPU atomics.
extern "C" __global__ void convolution2d(float *out, const float *input,
                                         const float *weight, const float *bias,
                                         int64_t count, int ci, int h, int w,
                                         int co, int kh, int kw, int oh, int ow,
                                         int sh, int sw, int ph, int pw, int dh,
                                         int dw, int groups) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int x = i % ow, y = (i / ow) % oh, c = (i / ow / oh) % co;
  int64_t batch = i / ow / oh / co;
  int first = (c / (co / groups)) * (ci / groups);
  float sum = bias ? bias[c] : 0.0f;
  for (int ic = 0; ic < ci / groups; ++ic)
    for (int ky = 0; ky < kh; ++ky)
      for (int kx = 0; kx < kw; ++kx) {
        int iy = y * sh - ph + ky * dh, ix = x * sw - pw + kx * dw;
        if (iy >= 0 && iy < h && ix >= 0 && ix < w)
          sum += input[((batch * ci + first + ic) * h + iy) * w + ix] *
                 weight[((c * (ci / groups) + ic) * kh + ky) * kw + kx];
      }
  out[i] = sum;
}
extern "C" __global__ void
convolution2d_input_backward(float *out, const float *grad, const float *weight,
                             int64_t count, int ci, int h, int w, int co,
                             int kh, int kw, int oh, int ow, int sh, int sw,
                             int ph, int pw, int dh, int dw, int groups) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int x = i % w, y = (i / w) % h, c = (i / w / h) % ci;
  int64_t batch = i / w / h / ci;
  int first = (c / (ci / groups)) * (co / groups);
  float sum = 0.0f;
  for (int oc = first; oc < first + co / groups; ++oc)
    for (int ky = 0; ky < kh; ++ky)
      for (int kx = 0; kx < kw; ++kx) {
        int gy = y + ph - ky * dh, gx = x + pw - kx * dw;
        if (gy >= 0 && gx >= 0 && gy % sh == 0 && gx % sw == 0 &&
            gy / sh < oh && gx / sw < ow)
          sum +=
              grad[((batch * co + oc) * oh + gy / sh) * ow + gx / sw] *
              weight[((oc * (ci / groups) + c % (ci / groups)) * kh + ky) * kw +
                     kx];
      }
  out[i] = sum;
}
extern "C" __global__ void convolution2d_weight_backward(
    float *out, const float *input, const float *grad, int64_t count,
    int batches, int ci, int h, int w, int co, int kh, int kw, int oh, int ow,
    int sh, int sw, int ph, int pw, int dh, int dw, int groups) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  int kx = i % kw, ky = (i / kw) % kh, ic = (i / kw / kh) % (ci / groups);
  int oc = i / kw / kh / (ci / groups);
  int c = (oc / (co / groups)) * (ci / groups) + ic;
  float sum = 0.0f;
  for (int batch = 0; batch < batches; ++batch)
    for (int y = 0; y < oh; ++y)
      for (int x = 0; x < ow; ++x) {
        int iy = y * sh - ph + ky * dh, ix = x * sw - pw + kx * dw;
        if (iy >= 0 && iy < h && ix >= 0 && ix < w)
          sum += input[((batch * ci + c) * h + iy) * w + ix] *
                 grad[((batch * co + oc) * oh + y) * ow + x];
      }
  out[i] = sum;
}
