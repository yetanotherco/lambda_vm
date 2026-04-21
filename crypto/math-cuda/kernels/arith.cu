// Element-wise Goldilocks kernels used by the Phase-2 parity tests. These mirror
// the CPU reference in `crypto/math/src/field/goldilocks.rs` so raw u64 outputs
// are bit-identical to the CPU path.

#include "goldilocks.cuh"

using goldilocks::add;
using goldilocks::sub;
using goldilocks::mul;
using goldilocks::neg;

extern "C" __global__ void vector_add_u64(const uint64_t *a,
                                          const uint64_t *b,
                                          uint64_t *c,
                                          uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];  // plain wrapping u64 add — toolchain sanity only.
}

extern "C" __global__ void gl_add_kernel(const uint64_t *a,
                                         const uint64_t *b,
                                         uint64_t *c,
                                         uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = add(a[tid], b[tid]);
}

extern "C" __global__ void gl_sub_kernel(const uint64_t *a,
                                         const uint64_t *b,
                                         uint64_t *c,
                                         uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = sub(a[tid], b[tid]);
}

extern "C" __global__ void gl_mul_kernel(const uint64_t *a,
                                         const uint64_t *b,
                                         uint64_t *c,
                                         uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = mul(a[tid], b[tid]);
}

extern "C" __global__ void gl_neg_kernel(const uint64_t *a,
                                         uint64_t *c,
                                         uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = neg(a[tid]);
}
