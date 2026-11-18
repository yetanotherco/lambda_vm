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

// The value-form round: one thread a cube index, the eight factors' values at each `t`, the
// composition evaluated at every point. `gkr4_round_gruen` (the first bench's kernel, compiled as
// it was) and `gkr4_round_gruen_lb2` (the same body held to two 256-thread blocks an SM) share it.
__device__ __forceinline__ void gkr4_value_body(
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

extern "C" __global__ __launch_bounds__(256) void gkr4_round_gruen(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t want_h0, uint32_t fault, uint64_t *__restrict__ d_partials) {
    gkr4_value_body(p, q, stride, quarter, fold, s0, s1, s2, l0, l1, l2, e_hi, e_lo, low, want_h0,
                    fault, d_partials);
}

// The same round held to two 256-thread blocks an SM: does a register cap alone help, or spill?
extern "C" __global__ __launch_bounds__(256, 2) void gkr4_round_gruen_lb2(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t want_h0, uint32_t fault, uint64_t *__restrict__ d_partials) {
    gkr4_value_body(p, q, stride, quarter, fold, s0, s1, s2, l0, l1, l2, e_hi, e_lo, low, want_h0,
                    fault, d_partials);
}

// ── The coefficient form ───────────────────────────────────────────────────────────────────────
//
// Write the middle layer's two halves at one node as `M_r = (X_r, Y_r)`, `r = 0, 1`, each the fold
// of quarters `r` (`b₁ = 0`) and `r + 2` (`b₁ = 1`):
//
//   X_r = p_r·q_{r+2} + p_{r+2}·q_r,   Y_r = q_r·q_{r+2}
//
// and the parent is `N + λQ = X_0·Y_1 + X_1·Y_0 + λ·Y_0·Y_1 = (X_0 + λY_0)·Y_1 + X_1·Y_0`. Every
// factor is linear in the round's `t`, so `X_r` and `Y_r` are quadratics, held by their values at
// `t = 0` and `t = 1` and their `t²` coefficient: a product of two linear factors costs three
// multiplications (`lo·lo`, `hi·hi`, `Δ·Δ`) where the value form costs one a point. The rows walk
// `t = 1..4` by finite differences, additions only.

// A quadratic in `t`: its values at 0 and 1 and its `t²` coefficient.
struct B2Quad {
    Fe3 v0, v1, c2;
};

// The product of two linear factors given at `t = 0` (`lo`) and `t = 1` (`hi`).
__device__ __forceinline__ B2Quad b2_prod(const Fe3 &lo_a, const Fe3 &hi_a, const Fe3 &lo_b,
                                          const Fe3 &hi_b) {
    B2Quad r;
    r.v0 = ext3::mul(lo_a, lo_b);
    r.v1 = ext3::mul(hi_a, hi_b);
    r.c2 = ext3::mul(ext3::sub(hi_a, lo_a), ext3::sub(hi_b, lo_b));
    return r;
}

__device__ __forceinline__ B2Quad b2_qadd(const B2Quad &a, const B2Quad &b) {
    B2Quad r;
    r.v0 = ext3::add(a.v0, b.v0);
    r.v1 = ext3::add(a.v1, b.v1);
    r.c2 = ext3::add(a.c2, b.c2);
    return r;
}

__device__ __forceinline__ B2Quad b2_qscale(const Fe3 &s, const B2Quad &a) {
    B2Quad r;
    r.v0 = ext3::mul(s, a.v0);
    r.v1 = ext3::mul(s, a.v1);
    r.c2 = ext3::mul(s, a.c2);
    return r;
}

// A quadratic walked from `t = 1` by finite differences: `val` is its value at the current `t`,
// `d` the step to the next, `dd` the constant second difference `2·c2`.
struct B2Walk {
    Fe3 val, d, dd;
};

__device__ __forceinline__ B2Walk b2_walk(const B2Quad &a) {
    B2Walk w;
    w.dd = ext3::add(a.c2, a.c2);
    w.d = ext3::add(ext3::sub(a.v1, a.v0), w.dd);
    w.val = a.v1;
    return w;
}

__device__ __forceinline__ void b2_step(B2Walk &w) {
    w.val = ext3::add(w.val, w.d);
    w.d = ext3::add(w.d, w.dd);
}

// One factor's cells at cube index `x`, bound to `s` on the way in when the round folds (and
// written back), as the value-form round reads them. `faulty` reads the lower cell one index off
// in a round without a fold: a negative control's fault only.
__device__ __forceinline__ void b2_factor(uint64_t *f, uint64_t x, uint64_t quarter, bool fold,
                                          const Fe3 &s, bool faulty, Fe3 &lo, Fe3 &hi) {
    if (fold) {
        Fe3 a = b2_load(f + x * 3);
        Fe3 b = b2_load(f + (x + quarter) * 3);
        Fe3 c = b2_load(f + (x + 2 * quarter) * 3);
        Fe3 d = b2_load(f + (x + 3 * quarter) * 3);
        lo = ext3::add(a, ext3::mul(s, ext3::sub(c, a)));
        hi = ext3::add(b, ext3::mul(s, ext3::sub(d, b)));
        b2_store(f + x * 3, lo);
        b2_store(f + (x + quarter) * 3, hi);
    } else {
        uint64_t at = faulty ? ((x + 1) & (quarter - 1)) : x;
        lo = b2_load(f + at * 3);
        hi = b2_load(f + (x + quarter) * 3);
    }
}

__device__ __forceinline__ Fe3 b2_shfl(const Fe3 &v, int src) {
    return ext3::make(__shfl_sync(0xffffffffu, v.a, src), __shfl_sync(0xffffffffu, v.b, src),
                      __shfl_sync(0xffffffffu, v.c, src));
}

// The block's rows by warp shuffles: each warp sums its lanes in five levels, lane 0 of each warp
// puts its sums in shared memory (`rows × warps` ext3), and thread 0 adds the warps' sums into
// `d_partials[(row·gridDim.x + block)·3]`. Every lane of the block must reach it. `rows` is a
// constant at every call, so the loops unroll and `acc` stays in registers.
__device__ __forceinline__ void b2_block_rows(const Fe3 *acc, int rows,
                                              uint64_t *__restrict__ d_partials) {
    extern __shared__ uint64_t shared[];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (int row = 0; row < 5; row++) {
        if (row >= rows) break;
        Fe3 v = acc[row];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = ext3::add(v, b2_shfl(v, (int)(lane ^ off)));
        if (lane == 0) b2_store(shared + ((uint64_t)row * warps + warp) * 3, v);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int row = 0; row < 5; row++) {
            if (row >= rows) break;
            Fe3 sum = ext3::zero();
            for (unsigned w = 0; w < warps; w++)
                sum = ext3::add(sum, b2_load(shared + ((uint64_t)row * warps + w) * 3));
            b2_store(d_partials + ((uint64_t)row * gridDim.x + blockIdx.x) * 3, sum);
        }
    }
}

// The rows `H(1)..H(4)` (and `H(0)` as row 4 with `H0`) of `w·(Z(t)·W(t))`, `Z` already weighted.
template <bool H0>
__device__ __forceinline__ void b2_accumulate(const B2Quad &Z, const B2Quad &W, Fe3 (&acc)[5]) {
    if (H0) acc[4] = ext3::add(acc[4], ext3::mul(Z.v0, W.v0));
    B2Walk z = b2_walk(Z), u = b2_walk(W);
    acc[0] = ext3::add(acc[0], ext3::mul(z.val, u.val));
#pragma unroll
    for (int t = 1; t < 4; t++) {
        b2_step(z);
        b2_step(u);
        acc[t] = ext3::add(acc[t], ext3::mul(z.val, u.val));
    }
}

// ── V2: one thread a cube index, the coefficient form ───────────────────────────────────────────
//
// The thread folds and combines quarters `0, 2` (`X_0, Y_0`), then quarters `1, 3` (`X_1, Y_1`), so
// only four factors' cells need be in flight at once, then sums `w·[(X_0 + λY_0)(t)·Y_1(t) +
// X_1(t)·Y_0(t)]` at `t = 1..4`. Same layout, rows and partials as the value-form round.
template <bool H0>
__device__ __forceinline__ void gkr2_v2_body(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t fault, uint64_t *__restrict__ d_partials) {
    Fe3 s = ext3::make(s0, s1, s2);
    Fe3 lambda = ext3::make(l0, l1, l2);
    uint64_t mask = ((uint64_t)1 << low) - 1;
    Fe3 acc[5];
#pragma unroll
    for (int r = 0; r < 5; r++) acc[r] = ext3::zero();
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < quarter;
         x += (uint64_t)gridDim.x * blockDim.x) {
        B2Quad X[2], Y[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            // p_r, p_{r+2}, q_r, q_{r+2}
            Fe3 a_lo, a_hi, b_lo, b_hi, c_lo, c_hi, d_lo, d_hi;
            b2_factor(p + (uint64_t)r * stride * 3, x, quarter, fold, s, false, a_lo, a_hi);
            b2_factor(p + (uint64_t)(r + 2) * stride * 3, x, quarter, fold, s, false, b_lo, b_hi);
            b2_factor(q + (uint64_t)r * stride * 3, x, quarter, fold, s, false, c_lo, c_hi);
            b2_factor(q + (uint64_t)(r + 2) * stride * 3, x, quarter, fold, s,
                      fault == 1 && r == 1, d_lo, d_hi);
            X[r] = b2_qadd(b2_prod(a_lo, a_hi, d_lo, d_hi), b2_prod(b_lo, b_hi, c_lo, c_hi));
            Y[r] = b2_prod(c_lo, c_hi, d_lo, d_hi);
        }
        Fe3 w = ext3::mul(b2_load(e_hi + (x >> low) * 3), b2_load(e_lo + (x & mask) * 3));
        B2Quad Z = b2_qscale(w, b2_qadd(X[0], b2_qscale(lambda, Y[0])));
        B2Quad V = b2_qscale(w, X[1]);
        b2_accumulate<H0>(Z, Y[1], acc);
        b2_accumulate<H0>(V, Y[0], acc);
    }
    b2_block_rows(acc, H0 ? 5 : 4, d_partials);
}

#define GKR2_V2(NAME, H0, BOUNDS)                                                                 \
    extern "C" __global__ BOUNDS void NAME(                                                       \
        uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,    \
        uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1,           \
        uint64_t l2, const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo,        \
        uint32_t low, uint32_t fault, uint64_t *__restrict__ d_partials) {                        \
        gkr2_v2_body<H0>(p, q, stride, quarter, fold, s0, s1, s2, l0, l1, l2, e_hi, e_lo, low,    \
                         fault, d_partials);                                                      \
    }
GKR2_V2(gkr2_round_v2, false, __launch_bounds__(256, 2))
GKR2_V2(gkr2_round_v2_h0, true, __launch_bounds__(256, 2))

// ── V3: a thread pair a cube index ──────────────────────────────────────────────────────────────
//
// Lanes `2i, 2i + 1` take cube index `x`; lane `r = lane & 1` owns quarters `r` and `r + 2` of both
// `p` and `q`, so it loads, folds and writes back four factors, as one thread of today's round
// does. It makes `X_r, Y_r`, then `Z_r = X_r + λ_r·Y_r` with `λ_0 = λ`, `λ_1 = 0` (one code path),
// trades `Y` with its partner (`W = Y_{1−r}`, nine limbs by `__shfl_sync`) and sums
// `w·Z_r(t)·W(t)`: lane 0 adds `(X_0 + λY_0)·Y_1` and lane 1 `X_1·Y_0`, which is `N + λQ`.
//
// The lane loop runs whole warps: a warp past the last cube index still reaches every shuffle,
// with zero values and a zero weight. `fault` 1 reads lane 1's last quarter one cell off in a round
// without a fold; `fault` 2 trades with lane `^ 2` (the next pair) instead of the partner.
template <bool H0>
__device__ __forceinline__ void gkr2_v3_body(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t fault, uint64_t *__restrict__ d_partials) {
    Fe3 s = ext3::make(s0, s1, s2);
    const unsigned lane = threadIdx.x & 31u;
    const uint32_t r = lane & 1u;
    Fe3 lambda_r = r == 0 ? ext3::make(l0, l1, l2) : ext3::zero();
    const int partner = (int)(lane ^ (fault == 2 ? 2u : 1u));
    uint64_t mask = ((uint64_t)1 << low) - 1;
    uint64_t lanes = 2 * quarter;
    uint64_t *pa = p + (uint64_t)r * stride * 3, *pb = p + (uint64_t)(r + 2) * stride * 3;
    uint64_t *qa = q + (uint64_t)r * stride * 3, *qb = q + (uint64_t)(r + 2) * stride * 3;
    Fe3 acc[5];
#pragma unroll
    for (int k = 0; k < 5; k++) acc[k] = ext3::zero();
    for (uint64_t base = (uint64_t)blockIdx.x * blockDim.x + (threadIdx.x & ~31u); base < lanes;
         base += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t g = base + lane;
        bool live = g < lanes;
        uint64_t x = g >> 1;
        Fe3 a_lo = ext3::zero(), a_hi = ext3::zero(), b_lo = ext3::zero(), b_hi = ext3::zero();
        Fe3 c_lo = ext3::zero(), c_hi = ext3::zero(), d_lo = ext3::zero(), d_hi = ext3::zero();
        Fe3 w = ext3::zero();
        if (live) {
            b2_factor(pa, x, quarter, fold, s, false, a_lo, a_hi);
            b2_factor(pb, x, quarter, fold, s, false, b_lo, b_hi);
            b2_factor(qa, x, quarter, fold, s, false, c_lo, c_hi);
            b2_factor(qb, x, quarter, fold, s, fault == 1 && r == 1, d_lo, d_hi);
            w = ext3::mul(b2_load(e_hi + (x >> low) * 3), b2_load(e_lo + (x & mask) * 3));
        }
        B2Quad X = b2_qadd(b2_prod(a_lo, a_hi, d_lo, d_hi), b2_prod(b_lo, b_hi, c_lo, c_hi));
        B2Quad Y = b2_prod(c_lo, c_hi, d_lo, d_hi);
        B2Quad Z = b2_qscale(w, b2_qadd(X, b2_qscale(lambda_r, Y)));
        B2Quad W;
        W.v0 = b2_shfl(Y.v0, partner);
        W.v1 = b2_shfl(Y.v1, partner);
        W.c2 = b2_shfl(Y.c2, partner);
        b2_accumulate<H0>(Z, W, acc);
    }
    b2_block_rows(acc, H0 ? 5 : 4, d_partials);
}

#define GKR2_V3(NAME, H0, BOUNDS)                                                                 \
    extern "C" __global__ BOUNDS void NAME(                                                       \
        uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,    \
        uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1,           \
        uint64_t l2, const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo,        \
        uint32_t low, uint32_t fault, uint64_t *__restrict__ d_partials) {                        \
        gkr2_v3_body<H0>(p, q, stride, quarter, fold, s0, s1, s2, l0, l1, l2, e_hi, e_lo, low,    \
                         fault, d_partials);                                                      \
    }
GKR2_V3(gkr2_round_v3, false, __launch_bounds__(256, 2))
GKR2_V3(gkr2_round_v3_h0, true, __launch_bounds__(256, 2))
// V3b: the same pair kernel at three blocks an SM (≤ 85 registers). Information only.
GKR2_V3(gkr2_round_v3b, false, __launch_bounds__(256, 3))
GKR2_V3(gkr2_round_v3b_h0, true, __launch_bounds__(256, 3))

// ── Spill-free attempts at the pair kernel ──────────────────────────────────────────────────────
//
// V3 under (256, 2) used its 128 registers and spilled 140–232 B. These two take state out of the
// registers in two different ways; both leave `H(0)` to its own pass (`gkr2_round_h0_pass`), so the
// round kernel never carries a fifth accumulator. `H(0)` is wanted only where `1 − u_j = 0`.

// The block's rows by warp shuffles as `b2_block_rows`, its per-warp sums at `scratch` in shared
// memory, written to `d_partials` from row `first`.
__device__ __forceinline__ void b2_block_rows_at(const Fe3 *acc, int rows, int first,
                                                 uint64_t *scratch,
                                                 uint64_t *__restrict__ d_partials) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (int row = 0; row < 4; row++) {
        if (row >= rows) break;
        Fe3 v = acc[row];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = ext3::add(v, b2_shfl(v, (int)(lane ^ off)));
        if (lane == 0) b2_store(scratch + ((uint64_t)row * warps + warp) * 3, v);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int row = 0; row < 4; row++) {
            if (row >= rows) break;
            Fe3 sum = ext3::zero();
            for (unsigned w = 0; w < warps; w++)
                sum = ext3::add(sum, b2_load(scratch + ((uint64_t)row * warps + w) * 3));
            b2_store(d_partials + ((uint64_t)(first + row) * gridDim.x + blockIdx.x) * 3, sum);
        }
    }
}

// ── V4: the pair kernel with its accumulators in shared memory ──────────────────────────────────
//
// V3's body without `H(0)`, but each lane's four row sums live in shared memory (limb-major, one
// `volatile` load and store a limb a point a cube index) instead of 24 registers across the loop.
// Shared memory: `12·blockDim` u64 of sums, then the reduction's `4·warps·3`.
__device__ __forceinline__ void gkr2_v4_body(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t fault, uint64_t *__restrict__ d_partials) {
    extern __shared__ uint64_t shared[];
    volatile uint64_t *sacc = shared;
    const unsigned nb = blockDim.x, tid = threadIdx.x;
    Fe3 s = ext3::make(s0, s1, s2);
    const unsigned lane = tid & 31u;
    const uint32_t r = lane & 1u;
    Fe3 lambda_r = r == 0 ? ext3::make(l0, l1, l2) : ext3::zero();
    const int partner = (int)(lane ^ (fault == 2 ? 2u : 1u));
    uint64_t mask = ((uint64_t)1 << low) - 1;
    uint64_t lanes = 2 * quarter;
    uint64_t *pa = p + (uint64_t)r * stride * 3, *pb = p + (uint64_t)(r + 2) * stride * 3;
    uint64_t *qa = q + (uint64_t)r * stride * 3, *qb = q + (uint64_t)(r + 2) * stride * 3;
#pragma unroll
    for (int k = 0; k < 12; k++) sacc[k * nb + tid] = 0;
    for (uint64_t base = (uint64_t)blockIdx.x * nb + (tid & ~31u); base < lanes;
         base += (uint64_t)gridDim.x * nb) {
        uint64_t g = base + lane;
        bool live = g < lanes;
        uint64_t x = g >> 1;
        Fe3 a_lo = ext3::zero(), a_hi = ext3::zero(), b_lo = ext3::zero(), b_hi = ext3::zero();
        Fe3 c_lo = ext3::zero(), c_hi = ext3::zero(), d_lo = ext3::zero(), d_hi = ext3::zero();
        Fe3 w = ext3::zero();
        if (live) {
            b2_factor(pa, x, quarter, fold, s, false, a_lo, a_hi);
            b2_factor(pb, x, quarter, fold, s, false, b_lo, b_hi);
            b2_factor(qa, x, quarter, fold, s, false, c_lo, c_hi);
            b2_factor(qb, x, quarter, fold, s, fault == 1 && r == 1, d_lo, d_hi);
            w = ext3::mul(b2_load(e_hi + (x >> low) * 3), b2_load(e_lo + (x & mask) * 3));
        }
        B2Quad X = b2_qadd(b2_prod(a_lo, a_hi, d_lo, d_hi), b2_prod(b_lo, b_hi, c_lo, c_hi));
        B2Quad Y = b2_prod(c_lo, c_hi, d_lo, d_hi);
        B2Quad Z = b2_qscale(w, b2_qadd(X, b2_qscale(lambda_r, Y)));
        B2Quad W;
        W.v0 = b2_shfl(Y.v0, partner);
        W.v1 = b2_shfl(Y.v1, partner);
        W.c2 = b2_shfl(Y.c2, partner);
        B2Walk z = b2_walk(Z), u = b2_walk(W);
#pragma unroll
        for (int t = 0; t < 4; t++) {
            if (t > 0) {
                b2_step(z);
                b2_step(u);
            }
            Fe3 m = ext3::mul(z.val, u.val);
            volatile uint64_t *at = sacc + (uint64_t)(t * 3) * nb + tid;
            at[0] = goldilocks::add(at[0], m.a);
            at[nb] = goldilocks::add(at[nb], m.b);
            at[2 * nb] = goldilocks::add(at[2 * nb], m.c);
        }
    }
    Fe3 acc[4];
#pragma unroll
    for (int t = 0; t < 4; t++)
        acc[t] = ext3::make(sacc[(t * 3) * nb + tid], sacc[(t * 3 + 1) * nb + tid],
                            sacc[(t * 3 + 2) * nb + tid]);
    b2_block_rows_at(acc, 4, 0, shared + 12 * (uint64_t)nb, d_partials);
}

extern "C" __global__ __launch_bounds__(256, 2) void gkr2_round_v4(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t fault, uint64_t *__restrict__ d_partials) {
    gkr2_v4_body(p, q, stride, quarter, fold, s0, s1, s2, l0, l1, l2, e_hi, e_lo, low, fault,
                 d_partials);
}

// ── V5: the pair kernel at 128-thread blocks, three an SM ───────────────────────────────────────
//
// V3's body without `H(0)` under `__launch_bounds__(128, 3)`: up to 170 registers a thread, 12 warps
// an SM where V3 had 16. Launched with blocks of at most 128 lanes.
GKR2_V3(gkr2_round_v5, false, __launch_bounds__(128, 3))

// ── H(0) in its own pass, for V4 and V5 ─────────────────────────────────────────────────────────
//
// Run after the round kernel on the same stream, grid and block: it reads the cells the round
// left at `t = 0` (the folded `lo` halves after a fold, the layer's own before one), one thread a
// cube index, and writes row 4 (`H(0)`) of the same partials. `fault` 1 reads the last quarter's
// lower cell one off in a round without a fold, as the round kernels do.
__device__ __forceinline__ void gkr2_h0_pass_body(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t l0, uint64_t l1, uint64_t l2, const uint64_t *__restrict__ e_hi,
    const uint64_t *__restrict__ e_lo, uint32_t low, uint32_t fault,
    uint64_t *__restrict__ d_partials) {
    extern __shared__ uint64_t shared[];
    Fe3 lambda = ext3::make(l0, l1, l2);
    uint64_t mask = ((uint64_t)1 << low) - 1;
    Fe3 acc[1] = {ext3::zero()};
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < quarter;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 v[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint64_t *f = (k < 4 ? p : q) + (uint64_t)(k & 3) * stride * 3;
            uint64_t at = (!fold && fault == 1 && k == 7) ? ((x + 1) & (quarter - 1)) : x;
            v[k] = b2_load(f + at * 3);
        }
        Fe3 w = ext3::mul(b2_load(e_hi + (x >> low) * 3), b2_load(e_lo + (x & mask) * 3));
        acc[0] = ext3::add(acc[0], ext3::mul(w, b2_h(v, lambda)));
    }
    b2_block_rows_at(acc, 1, 4, shared, d_partials);
}

extern "C" __global__ __launch_bounds__(256) void gkr2_round_h0_pass(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t stride, uint64_t quarter,
    uint32_t fold, uint64_t l0, uint64_t l1, uint64_t l2, const uint64_t *__restrict__ e_hi,
    const uint64_t *__restrict__ e_lo, uint32_t low, uint32_t fault,
    uint64_t *__restrict__ d_partials) {
    gkr2_h0_pass_body(p, q, stride, quarter, fold, l0, l1, l2, e_hi, e_lo, low, fault, d_partials);
}
