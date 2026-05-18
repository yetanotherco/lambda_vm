// Goldilocks field on device. Ports `crypto/math/src/field/goldilocks.rs` one-to-one:
// - Representation: non-canonical u64 in [0, 2^64). Canonicalise only at boundaries.
// - Prime: 2^64 - 2^32 + 1.
// - Reduction: exploits 2^64 ≡ EPSILON (mod p) and 2^96 ≡ -1 (mod p).
//
// The arithmetic here must produce bit-identical u64 outputs to the CPU path so
// LDE parity tests can assert raw equality.

#pragma once
#include <cstdint>

namespace goldilocks {

__device__ constexpr uint64_t PRIME   = 0xFFFFFFFF00000001ULL;
__device__ constexpr uint64_t EPSILON = 0xFFFFFFFFULL;  // 2^32 - 1

__device__ __forceinline__ uint64_t add_no_canonicalize(uint64_t x, uint64_t y) {
    // Mirror of `add_no_canonicalize_trashing_input`: one add, one EPSILON bump on carry.
    uint64_t sum = x + y;
    return sum + (sum < x ? EPSILON : 0ULL);
}

__device__ __forceinline__ uint64_t add(uint64_t a, uint64_t b) {
    uint64_t sum  = a + b;
    uint64_t over1 = (sum < a) ? EPSILON : 0ULL;
    uint64_t sum2 = sum + over1;
    uint64_t over2 = (sum2 < sum) ? EPSILON : 0ULL;
    return sum2 + over2;
}

__device__ __forceinline__ uint64_t sub(uint64_t a, uint64_t b) {
    uint64_t diff  = a - b;
    uint64_t under1 = (a < b) ? EPSILON : 0ULL;
    uint64_t diff2 = diff - under1;
    uint64_t under2 = (diff2 > diff) ? EPSILON : 0ULL;
    return diff2 - under2;
}

__device__ __forceinline__ uint64_t reduce128(uint64_t lo, uint64_t hi) {
    uint64_t x_hi_hi = hi >> 32;
    uint64_t x_hi_lo = hi & EPSILON;

    // 2^96 ≡ -1 (mod p): subtract x_hi_hi from lo, EPSILON-correct on borrow.
    uint64_t t0 = lo - x_hi_hi;
    if (lo < x_hi_hi) t0 -= EPSILON;

    // 2^64 ≡ EPSILON (mod p): x_hi_lo * EPSILON = (x_hi_lo << 32) - x_hi_lo.
    uint64_t t1 = (x_hi_lo << 32) - x_hi_lo;

    return add_no_canonicalize(t0, t1);
}

__device__ __forceinline__ uint64_t mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return reduce128(lo, hi);
}

__device__ __forceinline__ uint64_t neg(uint64_t a) {
    // `a` may be non-canonical. Canonicalise first, then p - a (or 0).
    uint64_t canon = (a >= PRIME) ? (a - PRIME) : a;
    return canon == 0 ? 0 : (PRIME - canon);
}

__device__ __forceinline__ uint64_t canonical(uint64_t a) {
    return (a >= PRIME) ? (a - PRIME) : a;
}

}  // namespace goldilocks
