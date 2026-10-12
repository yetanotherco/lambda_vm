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

// Widen a packed main trace (crypto/stark/src/narrow.rs): column c holds `rows`
// little-endian words of widths[c] bytes from byte offsets[c] of `data`; `out`
// is the row-major rows x cols trace of the same words, zero-extended. One
// thread per output word, so the writes coalesce.
extern "C" __global__ void widen_narrow_row_major(const uint8_t *data,
                                                  const uint64_t *offsets,
                                                  const uint8_t *widths,
                                                  uint64_t *out,
                                                  uint64_t rows,
                                                  uint64_t cols) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows * cols) return;
    uint64_t r = tid / cols;
    uint64_t c = tid - r * cols;
    uint32_t w = widths[c];
    const uint8_t *p = data + offsets[c] + r * w;
    uint64_t v = 0;
    for (uint32_t i = 0; i < w; i++) v |= (uint64_t)p[i] << (8 * i);
    out[tid] = v;
}

// Column maxima of a column-major rows x cols trace (`src[c*rows + r]`), the
// widths of its packed form (crypto/stark/src/narrow.rs): blockIdx.y is the
// column, the x blocks stride its rows, and each block folds its maximum into
// max_out[c]. blockDim.x must be 256.
extern "C" __global__ void column_max_col_major(const uint64_t *src,
                                                uint64_t rows,
                                                unsigned long long *max_out) {
    __shared__ unsigned long long s[256];
    const uint64_t *col = src + (uint64_t)blockIdx.y * rows;
    unsigned long long m = 0;
    for (uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows;
         r += (uint64_t)gridDim.x * blockDim.x) {
        unsigned long long v = col[r];
        m = v > m ? v : m;
    }
    s[threadIdx.x] = m;
    __syncthreads();
    for (unsigned int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k && s[threadIdx.x + k] > s[threadIdx.x]) s[threadIdx.x] = s[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(&max_out[blockIdx.y], s[0]);
}

// Pack a column-major rows x cols trace at per-column widths: column c's word
// r goes to out[offsets[c] + r*widths[c]], little-endian (the layout
// widen_narrow_row_major reads). One thread per word.
extern "C" __global__ void pack_col_major(const uint64_t *src,
                                          uint64_t rows,
                                          uint64_t cols,
                                          const uint64_t *offsets,
                                          const uint8_t *widths,
                                          uint8_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rows * cols) return;
    uint64_t c = tid / rows;
    uint64_t r = tid - c * rows;
    uint32_t w = widths[c];
    uint64_t v = src[tid];
    uint8_t *p = out + offsets[c] + r * w;
    for (uint32_t i = 0; i < w; i++) p[i] = (uint8_t)(v >> (8 * i));
}
