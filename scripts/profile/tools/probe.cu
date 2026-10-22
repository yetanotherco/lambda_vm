// Preflight probe: a few kernels, copies and one memset, so the profile checks nsys export + nsys_extract.py on this
// machine's nsys before its real captures. Expected: 50 kernels, 100 memcpys, 1 memset.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void profile_probe_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 2.0f + 1.0f;
}
int main() {
  const int n = 1 << 20;
  float* h = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) h[i] = (float)i;
  float* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return 2;
  cudaMemset(d, 0, n * sizeof(float));
  for (int r = 0; r < 50; r++) {
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
    profile_probe_kernel<<<(n + 255) / 256, 256>>>(d, n);
    cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  printf("probe ok %f\n", h[1]);
  cudaFree(d);
  free(h);
  return 0;
}
