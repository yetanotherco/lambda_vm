// Element-wise Goldilocks kernels used by the parity tests. These mirror
// the CPU reference in `crypto/math/src/field/goldilocks.rs` so raw u64 outputs
// are bit-identical to the CPU path.

#include "goldilocks.cuh"
#include "ext3.cuh"

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

// ---------------------------------------------------------------------------
// Ext3 (Goldilocks cubic extension) test kernels.
// Input/output arrays are interleaved [a_0, b_0, c_0, a_1, b_1, c_1, ...].
// ---------------------------------------------------------------------------

extern "C" __global__ void ext3_mul_kernel(const uint64_t *a_int,
                                           const uint64_t *b_int,
                                           uint64_t *c_int,
                                           uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    ext3::Fe3 a = ext3::make(a_int[tid*3 + 0], a_int[tid*3 + 1], a_int[tid*3 + 2]);
    ext3::Fe3 b = ext3::make(b_int[tid*3 + 0], b_int[tid*3 + 1], b_int[tid*3 + 2]);
    ext3::Fe3 r = ext3::mul(a, b);
    c_int[tid*3 + 0] = r.a;
    c_int[tid*3 + 1] = r.b;
    c_int[tid*3 + 2] = r.c;
}

extern "C" __global__ void ext3_add_kernel(const uint64_t *a_int,
                                           const uint64_t *b_int,
                                           uint64_t *c_int,
                                           uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    ext3::Fe3 a = ext3::make(a_int[tid*3 + 0], a_int[tid*3 + 1], a_int[tid*3 + 2]);
    ext3::Fe3 b = ext3::make(b_int[tid*3 + 0], b_int[tid*3 + 1], b_int[tid*3 + 2]);
    ext3::Fe3 r = ext3::add(a, b);
    c_int[tid*3 + 0] = r.a;
    c_int[tid*3 + 1] = r.b;
    c_int[tid*3 + 2] = r.c;
}

extern "C" __global__ void ext3_sub_kernel(const uint64_t *a_int,
                                           const uint64_t *b_int,
                                           uint64_t *c_int,
                                           uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    ext3::Fe3 a = ext3::make(a_int[tid*3 + 0], a_int[tid*3 + 1], a_int[tid*3 + 2]);
    ext3::Fe3 b = ext3::make(b_int[tid*3 + 0], b_int[tid*3 + 1], b_int[tid*3 + 2]);
    ext3::Fe3 r = ext3::sub(a, b);
    c_int[tid*3 + 0] = r.a;
    c_int[tid*3 + 1] = r.b;
    c_int[tid*3 + 2] = r.c;
}
