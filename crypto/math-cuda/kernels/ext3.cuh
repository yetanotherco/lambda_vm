// Goldilocks cubic extension on device: Fp3 = Fp[w] / (w^3 - 2)
// where Fp is Goldilocks (2^64 - 2^32 + 1).
//
// Layout matches the CPU `Degree3GoldilocksExtensionField` (see
// `crypto/math/src/field/extensions_goldilocks.rs`): an element is a
// 3-tuple `(a, b, c)` representing `a + b*w + c*w^2`.
//
// The reducible `w^3 = 2` means cross-term products get a factor of 2:
//   (a0 + a1*w + a2*w^2) * (b0 + b1*w + b2*w^2)
//     = (a0*b0 + 2*(a1*b2 + a2*b1))
//     + (a0*b1 + a1*b0 + 2*a2*b2) * w
//     + (a0*b2 + a1*b1 + a2*b0) * w^2
//
// We use the same dot-product-of-three folding as the CPU (which saves
// reductions by summing u128 products before `reduce128`). CUDA has
// `__umul64hi` so we implement `dot_product_3` inline.

#pragma once
#include "goldilocks.cuh"

namespace ext3 {

struct Fe3 {
    uint64_t a, b, c;
};

__device__ __forceinline__ Fe3 make(uint64_t a, uint64_t b, uint64_t c) {
    Fe3 r = {a, b, c};
    return r;
}

__device__ __forceinline__ Fe3 zero() { return make(0, 0, 0); }
__device__ __forceinline__ Fe3 one()  { return make(1, 0, 0); }

__device__ __forceinline__ Fe3 add(const Fe3 &x, const Fe3 &y) {
    return make(goldilocks::add(x.a, y.a),
                goldilocks::add(x.b, y.b),
                goldilocks::add(x.c, y.c));
}

__device__ __forceinline__ Fe3 sub(const Fe3 &x, const Fe3 &y) {
    return make(goldilocks::sub(x.a, y.a),
                goldilocks::sub(x.b, y.b),
                goldilocks::sub(x.c, y.c));
}

__device__ __forceinline__ Fe3 neg(const Fe3 &x) {
    return make(goldilocks::neg(x.a),
                goldilocks::neg(x.b),
                goldilocks::neg(x.c));
}

/// Mixed: base * ext3 → ext3 (componentwise).
__device__ __forceinline__ Fe3 mul_base(const Fe3 &x, uint64_t s) {
    return make(goldilocks::mul(x.a, s),
                goldilocks::mul(x.b, s),
                goldilocks::mul(x.c, s));
}

/// Dot-product of three (a0*b0 + a1*b1 + a2*b2) mod p, with one reduce128
/// on the sum of three u128 products. Matches CPU `dot_product_3`.
__device__ __forceinline__ uint64_t dot3(uint64_t a0, uint64_t b0,
                                         uint64_t a1, uint64_t b1,
                                         uint64_t a2, uint64_t b2) {
    // Split the sum of three u128 products into hi/lo u128 halves, then
    // reduce once. We track overflow-count (at most 2) and add EPSILON^2
    // per overflow, matching the CPU path.
    // prod_i = a_i * b_i (u128)
    uint64_t lo0 = a0 * b0, hi0 = __umul64hi(a0, b0);
    uint64_t lo1 = a1 * b1, hi1 = __umul64hi(a1, b1);
    uint64_t lo2 = a2 * b2, hi2 = __umul64hi(a2, b2);

    // sum01 = prod0 + prod1 (in u128 lanes)
    uint64_t s01_lo = lo0 + lo1;
    uint64_t carry01 = (s01_lo < lo0) ? 1ULL : 0ULL;
    uint64_t s01_hi = hi0 + hi1 + carry01;
    uint32_t over1 = (s01_hi < hi0 + carry01) ? 1u : 0u; // low-pass overflow

    // sum012 = sum01 + prod2
    uint64_t s012_lo = s01_lo + lo2;
    uint64_t carry012 = (s012_lo < s01_lo) ? 1ULL : 0ULL;
    uint64_t s012_hi = s01_hi + hi2 + carry012;
    uint32_t over2 = (s012_hi < hi2 + carry012) ? 1u : 0u;

    uint64_t reduced = goldilocks::reduce128(s012_lo, s012_hi);

    uint32_t overflow_count = over1 + over2;
    if (overflow_count > 0) {
        // 2^128 mod p = EPSILON^2 (= (2^32 - 1)^2).
        uint64_t eps = goldilocks::EPSILON;
        uint64_t eps_sq = eps * eps;
        reduced = goldilocks::add_no_canonicalize(reduced, eps_sq);
        if (overflow_count > 1) {
            reduced = goldilocks::add_no_canonicalize(reduced, eps_sq);
        }
    }
    return reduced;
}

/// Full ext3 × ext3 multiplication (matches CPU
/// `Degree3GoldilocksExtensionField::mul`).
__device__ __forceinline__ Fe3 mul(const Fe3 &x, const Fe3 &y) {
    // c0 = x.a*y.a + x.b*(2*y.c) + x.c*(2*y.b)
    // c1 = x.a*y.b + x.b*y.a     + x.c*(2*y.c)
    // c2 = x.a*y.c + x.b*y.b     + x.c*y.a
    uint64_t b1_2 = goldilocks::add(y.b, y.b);
    uint64_t b2_2 = goldilocks::add(y.c, y.c);

    uint64_t c0 = dot3(x.a, y.a, x.b, b2_2, x.c, b1_2);
    uint64_t c1 = dot3(x.a, y.b, x.b, y.a, x.c, b2_2);
    uint64_t c2 = dot3(x.a, y.c, x.b, y.b, x.c, y.a);
    return make(c0, c1, c2);
}

__device__ __forceinline__ Fe3 canonical(const Fe3 &x) {
    return make(goldilocks::canonical(x.a),
                goldilocks::canonical(x.b),
                goldilocks::canonical(x.c));
}

}  // namespace ext3
