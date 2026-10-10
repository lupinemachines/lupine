extern "C" __global__ void topology_kernel(unsigned int *output) {
  unsigned int i = threadIdx.x;
  output[i] = i * 17 + 3;
}
