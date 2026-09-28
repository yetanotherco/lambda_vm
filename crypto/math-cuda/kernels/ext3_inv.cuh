// Row-local inversion of a handful of ext3 elements with ONE base-field
// inversion.
//
// The DEEP and OOD denominators are `x - z` (or `z - x`): `x` a base-field
// domain point, `z` one ext3 opening point per term. The scan path
// (`inverse.cu`) inverts a whole domain-sized array with prefix and suffix
// products and one Fermat inversion for the batch: six kernels, each a full
// pass over the array. A row only ever needs its own few denominators, so here
// each thread inverts them in registers:
//
//   e^{-1} = adj(e) / N(e),   N(e) = e · adj(e) in Fp,
//
// for e = a + b·w + c·w^2 in Fp[w]/(w^3 - 2):
//
//   adj(e) = (a^2 - 2bc) + (2c^2 - ab)·w + (b^2 - ac)·w^2
//   N(e)   = a·adj_0 + 2·(b·adj_2 + c·adj_1)
//
// (the w and w^2 components of e·adj(e) cancel identically), and the row's norms
// are inverted together by Montgomery's trick in the base field. One Fermat
// inversion per row, the rest a few dozen multiplications.
//
// The result is THE inverse, so it equals the scan's as a field element. Its raw
// u64 may differ by p — the device representation is non-canonical everywhere —
// which nothing downstream observes: every consumer is field arithmetic, the
// Merkle leaves absorb raw limbs into a representation-independent permutation,
// and the host canonicalises before it serialises or absorbs.
//
// A zero element (x equal to an opening point, which Fiat-Shamir makes
// negligible) has norm zero, which zeroes the row's product and so every
// inverse in that row. The scan path has the same release behaviour for its
// whole batch; its debug-build guard is mirrored by the kernels' zero flag.

#pragma once
#include "goldilocks.cuh"
#include "ext3.cuh"

namespace ext3_inv {

using ext3::Fe3;

// x^(p-2), which is x^{-1} for x != 0 and 0 for x == 0 (Fermat).
//
// p - 2 = 2^64 - 2^32 - 1 = (2^31 - 1)·2^33 + (2^32 - 1), built from
// t_k = x^(2^k - 1) by t_{j+k} = t_j^(2^k)·t_k: 64 squarings, 9 products.
__device__ __forceinline__ uint64_t sq_n(uint64_t v, int n) {
    for (int i = 0; i < n; ++i) v = goldilocks::mul(v, v);
    return v;
}

__device__ __forceinline__ uint64_t gl_inv(uint64_t x) {
    using goldilocks::mul;
    uint64_t t2 = mul(mul(x, x), x);           // x^(2^2 - 1)
    uint64_t t3 = mul(mul(t2, t2), x);         // x^(2^3 - 1)
    uint64_t t6 = mul(sq_n(t3, 3), t3);        // x^(2^6 - 1)
    uint64_t t12 = mul(sq_n(t6, 6), t6);       // x^(2^12 - 1)
    uint64_t t24 = mul(sq_n(t12, 12), t12);    // x^(2^24 - 1)
    uint64_t t30 = mul(sq_n(t24, 6), t6);      // x^(2^30 - 1)
    uint64_t t31 = mul(mul(t30, t30), x);      // x^(2^31 - 1)
    uint64_t t32 = mul(mul(t31, t31), x);      // x^(2^32 - 1)
    return mul(sq_n(t31, 33), t32);            // x^((2^31 - 1)·2^33 + 2^32 - 1)
}

// The adjugate of `e` and its norm N(e) = e·adj(e) (a base-field value).
__device__ __forceinline__ void adj_norm(const Fe3 &e, Fe3 &adj, uint64_t &norm) {
    using goldilocks::add;
    using goldilocks::mul;
    using goldilocks::sub;
    uint64_t bc = mul(e.b, e.c);
    uint64_t cc = mul(e.c, e.c);
    uint64_t a0 = sub(mul(e.a, e.a), add(bc, bc));  // a^2 - 2bc
    uint64_t a1 = sub(add(cc, cc), mul(e.a, e.b));  // 2c^2 - ab
    uint64_t a2 = sub(mul(e.b, e.b), mul(e.a, e.c));  // b^2 - ac
    uint64_t t = add(mul(e.b, a2), mul(e.c, a1));
    norm = add(mul(e.a, a0), add(t, t));
    adj = ext3::make(a0, a1, a2);
}

// Invert d[0..M) in place with one base-field inversion. M is a compile-time
// count so every array below stays in registers. Returns the product of the
// norms, which is zero (possibly as the raw value p) exactly when some d[j] was.
template <int M>
__device__ __forceinline__ uint64_t batch_inv(Fe3 (&d)[M]) {
    Fe3 adj[M];
    uint64_t norm[M];
    uint64_t prefix[M];
#pragma unroll
    for (int j = 0; j < M; ++j) {
        adj_norm(d[j], adj[j], norm[j]);
        prefix[j] = j == 0 ? norm[0] : goldilocks::mul(prefix[j - 1], norm[j]);
    }
    // inv = 1 / (norm[0]·…·norm[j]) as j walks down.
    uint64_t inv = gl_inv(prefix[M - 1]);
#pragma unroll
    for (int j = M - 1; j > 0; --j) {
        uint64_t norm_inv = goldilocks::mul(inv, prefix[j - 1]);
        inv = goldilocks::mul(inv, norm[j]);
        d[j] = ext3::mul_base(adj[j], norm_inv);
    }
    d[0] = ext3::mul_base(adj[0], inv);
    return prefix[M - 1];
}

// `x - z` for base `x` and ext3 `z` (DenomSign::XMinusZ), exactly as the scan
// path's `compute_denoms_ext3` builds it.
__device__ __forceinline__ Fe3 x_minus_z(uint64_t x, const uint64_t *z) {
    return ext3::make(goldilocks::sub(x, z[0]), goldilocks::neg(z[1]), goldilocks::neg(z[2]));
}

// `z - x` (DenomSign::ZMinusX), likewise.
__device__ __forceinline__ Fe3 z_minus_x(uint64_t x, const uint64_t *z) {
    return ext3::make(goldilocks::sub(z[0], x), z[1], z[2]);
}

}  // namespace ext3_inv
