// A measurement kernel, on no proving path (`src/gkr4_bench.rs`): one GKR step over two tree
// layers at once. Today a step proves a layer against the layer below it, a sumcheck of
// `eq · (p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi)` (degree 3, `gkr_round_gruen` in `sumcheck.cu`).
// This step proves a layer against its grandchildren: the four quarters of the layer two below
// (`f_j` at offset `j·stride`, `j = 0..3`) combine as the two folds would, `(f_0, f_2)` into the
// middle layer's lower half and `(f_1, f_3)` into its upper half, then those two into the parent.
// The polynomial under `eq` is degree 4, so a round sends `H(1)..H(4)` (and `H(0)`) where today's
// sends `H(1), H(2)` (and `H(0)`).
//
// The layout and the round are today's, with eight factors where today's has four: `p` and `q`
// each hold four quarters of `stride` cells; with `fold`, each quarter still holds the previous
// round's cube, twice as long, and is bound to `s` on the way in and written back where the next
// round reads it. The weight is Gruen's `e_hi[x >> low]·e_lo[x mod 2^low]`. Block partials go to
// `d_partials[(row·gridDim.x + block)·3]`, rows `H(1), H(2), H(3), H(4), H(0)`.

#include "goldilocks.cuh"
#include "ext3.cuh"

using ext3::Fe3;

__device__ __forceinline__ Fe3 b2_load(const uint64_t *at) { return ext3::make(at[0], at[1], at[2]); }

__device__ __forceinline__ void b2_store(uint64_t *at, const Fe3 &v) {
    at[0] = v.a;
    at[1] = v.b;
    at[2] = v.c;
}

// The parent at one node from the grandchildren there: `v[0..4]` the numerators at offsets 0..3,
// `v[4..8]` the denominators.
__device__ __forceinline__ Fe3 b2_h(const Fe3 *v, const Fe3 &lambda) {
    Fe3 lo_q = ext3::mul(v[4], v[6]);
    Fe3 lo_p = ext3::add(ext3::mul(v[0], v[6]), ext3::mul(v[2], v[4]));
    Fe3 hi_q = ext3::mul(v[5], v[7]);
    Fe3 hi_p = ext3::add(ext3::mul(v[1], v[7]), ext3::mul(v[3], v[5]));
    Fe3 num = ext3::add(ext3::mul(lo_p, hi_q), ext3::mul(hi_p, lo_q));
    return ext3::add(num, ext3::mul(lambda, ext3::mul(lo_q, hi_q)));
}

extern "C" __global__ __launch_bounds__(256) void gkr4_round_gruen(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t want_h0, uint32_t fault, uint64_t *__restrict__ d_partials) {
    Fe3 s = ext3::make(s0, s1, s2);
    Fe3 lambda = ext3::make(l0, l1, l2);
    uint64_t mask = ((uint64_t)1 << low) - 1;
    Fe3 acc[5];
#pragma unroll
    for (int r = 0; r < 5; r++) acc[r] = ext3::zero();
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < quarter;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 lo[8], hi[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            uint64_t *f = (k < 4 ? p : q) + (uint64_t)(k & 3) * stride * 3;
            if (fold) {
                Fe3 a = b2_load(f + x * 3);
                Fe3 b = b2_load(f + (x + quarter) * 3);
                Fe3 c = b2_load(f + (x + 2 * quarter) * 3);
                Fe3 d = b2_load(f + (x + 3 * quarter) * 3);
                lo[k] = ext3::add(a, ext3::mul(s, ext3::sub(c, a)));
                hi[k] = ext3::add(b, ext3::mul(s, ext3::sub(d, b)));
                b2_store(f + x * 3, lo[k]);
                b2_store(f + (x + quarter) * 3, hi[k]);
            } else {
                // ⛔ A negative control's fault only: the last quarter's lower cell one off.
                uint64_t at = (fault && k == 7) ? ((x + 1) & (quarter - 1)) : x;
                lo[k] = b2_load(f + at * 3);
                hi[k] = b2_load(f + (x + quarter) * 3);
            }
        }
        Fe3 w = ext3::mul(b2_load(e_hi + (x >> low) * 3), b2_load(e_lo + (x & mask) * 3));
        if (want_h0) acc[4] = ext3::add(acc[4], ext3::mul(w, b2_h(lo, lambda)));
        // `lo` becomes the step `hi − lo`; `hi` walks t = 1, 2, 3, 4.
#pragma unroll
        for (int k = 0; k < 8; k++) lo[k] = ext3::sub(hi[k], lo[k]);
        acc[0] = ext3::add(acc[0], ext3::mul(w, b2_h(hi, lambda)));
#pragma unroll
        for (int t = 1; t < 4; t++) {
#pragma unroll
            for (int k = 0; k < 8; k++) hi[k] = ext3::add(hi[k], lo[k]);
            acc[t] = ext3::add(acc[t], ext3::mul(w, b2_h(hi, lambda)));
        }
    }

    extern __shared__ uint64_t shared[];
    uint32_t rows = want_h0 ? 5 : 4;
    for (uint32_t row = 0; row < rows; row++) {
        Fe3 mine = acc[0];
#pragma unroll
        for (int r = 1; r < 5; r++)
            if (row == (uint32_t)r) mine = acc[r];
        b2_store(shared + threadIdx.x * 3, mine);
        __syncthreads();
        for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
            if (threadIdx.x < width) {
                Fe3 sum = ext3::add(b2_load(shared + threadIdx.x * 3),
                                    b2_load(shared + (threadIdx.x + width) * 3));
                b2_store(shared + threadIdx.x * 3, sum);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            b2_store(d_partials + ((uint64_t)row * gridDim.x + blockIdx.x) * 3, b2_load(shared));
        __syncthreads();
    }
}
