extern "C" __global__ void burn(float *a, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = a[i] + 1.0f, y = x * 0.5f, z = x * 0.25f, w = x * 0.125f;
  for (int k = 0; k < iters; k++) {
    x = fmaf(x, 0.9999f, 0.0001f);
    y = fmaf(y, 1.0001f, -0.0001f);
    z = fmaf(z, 0.9998f, 0.0002f);
    w = fmaf(w, 1.0002f, -0.0002f);
  }
  a[i] = x + y + z + w;
}
