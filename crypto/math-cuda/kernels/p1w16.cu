// Poseidon1 over Goldilocks at width 16 on device — the D-HASH stage-1
// MEASUREMENT kernels: the permutation, the rate-12 leaf sponge over WHIR fold
// cosets, the 4-ary Merkle node, and the grind search. Nothing on a proving
// path loads this module (`src/p1w16.rs`).
//
// CLEAN-ROOM. Written from the Poseidon paper (eprint 2019/458) and this
// repository's host reference `crypto/crypto/src/hash/poseidon1_w16/mod.rs`,
// with the field arithmetic of `goldilocks.cuh` and the limb multiply of our
// own `rpx.cu`. No third-party GPU Poseidon code was read (pil2-stark's is
// AGPL-3.0).
//
// THE ORACLE is the host reference, lane for lane:
//   permute        4 full rounds, 22 partial (S-box on lane 0), 4 full; each
//                  round adds RC[r], applies x^7, then state <- M·state with the
//                  circulant M[i][j] = MDS_ROW[(j − i) mod 16];
//   sponge leaf    rate 12 overwrite duplex; capacity lane 12 = felts mod 12,
//                  lane 13 = DOMAIN_LEAF ("P1WL"), lanes 14, 15 = 0; the tail
//                  block zero-padded; an empty leaf never permutes;
//   4-ary node     the four child digests fill all 16 lanes, permute, keep 0..4;
//   grind head     lane 0 of sponge_leaf([inner0..3, nonce]) — five felts, one
//                  permutation — the RPX grind's construction (`rpx.cu`) with
//                  this sponge.
// `tests/host_kat/p1w16_host_kat.cpp` checks all four on the host against the
// vectors the Python reference, Plonky3's generator and Plonky3's Rust
// permutation agree on; `tests/p1w16_device.rs` checks the device against the
// Rust host reference.
//
// DIGESTS are four canonical u64s (32 bytes, native order) — a measurement
// format, not the RPX kernels' big-endian commitment bytes.
//
// COST MODEL (one permutation): 150 S-boxes × 4 multiplications = 600 field
// multiplications; 30 MDS products of 256 small-constant 32×32→64 MACs on each
// 32-bit half (15,360 MACs) plus 16 reductions. Against RPX's 2,736 field
// multiplications + 2,016 MACs.
//
// VARIANTS (template `V`; every entry below is instantiated so one job
// measures them all):
//   V = 0  every product through `goldilocks::mul`;
//   V = 1  products and squares through the 32-bit-limb multiply (`rpx.cu`'s
//          `RPX_V_LIMB_MUL | RPX_V_LIMB_SQR`, reproduced below);
//   V = 2  V = 1's multiply with the partial rounds in the Fourier domain of
//          the circulant MDS (`permute_fourier`);
//   V = 3  the Grain Cauchy MDS alternative (`permute_cauchy`: dense full
//          rounds, the paper's sparse partial rounds, V = 1's multiply).
// Every variant computes the same field values and `permute` canonicalises.
//
// ZISK'S INSTANCE (stage P1 of the Poseidon1 base STARK, at the end of the
// file): the same permutation with ZisK's leaf hash (`zisk_leaf`: zero
// capacity, the previous digest carried into it) over cosets and the STARK's
// LDE rows, and ZisK's width-8 grinding permutation (`p1w8`, its own Fourier
// form). Oracle: `crypto::hash::poseidon1_stark` / `poseidon1_w8`, whose
// vectors come from ZisK's own code (`tests/host_kat/p1_zisk_kat_vectors.h`).

#include <cstdint>
#include "goldilocks.cuh"
#include "p1w16_constants.cuh"

namespace p1w16 {

enum : int {
    WIDTH = 16,
    RATE = 12,
    DIGEST = 4,
    HALF_FULL = 4,
    PARTIAL = 22,
    ROUNDS = 30,
    CAP_PAD_LANE = RATE + 0,
    CAP_DOMAIN_LANE = RATE + 1,
};

// `u32::from_le_bytes(*b"P1WL")`.
__device__ constexpr uint64_t DOMAIN_LEAF = 0x4C573150ull;
// Five felts: the inner hash's four and the nonce.
__device__ constexpr uint64_t GRIND_FELTS = 5;

// ---------------------------------------------------------------------------
// The 32-bit-limb multiply and square — `rpx.cu`'s `reduce_limbs`, `mul_limb`,
// `sqr_limb` (see the derivation there), reproduced so this module has no
// dependency on the RPX kernel file.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t reduce_limbs(uint32_t t0, uint32_t t1, uint32_t t2,
                                                 uint32_t t3) {
#if defined(__CUDA_ARCH__)
    const uint32_t w = 0xFFFFFFFFu;
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1; addc.u32 %2, 0, 0;"
        : "+r"(t0), "+r"(t1), "+r"(t2)
        : "r"(w));
    asm("sub.cc.u32 %0, %0, %3; subc.cc.u32 %1, %1, 0; subc.u32 %2, %2, 0;"
        : "+r"(t0), "+r"(t1), "+r"(t2)
        : "r"(t3));
    const uint32_t hi_sub = (t2 == 1u) ? 0xFFFFFFFFu : 0u;
    asm("sub.cc.u32 %0, %0, %2; subc.u32 %1, %1, %3;" : "+r"(t0), "+r"(t1) : "r"(t2), "r"(hi_sub));
    return ((uint64_t)t1 << 32) | t0;
#else
    const uint64_t s = ((uint64_t)t1 << 32) | t0;
    const uint64_t s1 = s + (uint64_t)t2 * 0xFFFFFFFFull;
    int c = (s1 < s) ? 1 : 0;
    const uint64_t s2 = s1 - t3;
    c -= (s1 < t3) ? 1 : 0;
    if (c == 1) return s2 + 0xFFFFFFFFull;
    if (c == -1) return s2 - 0xFFFFFFFFull;
    return s2;
#endif
}

__device__ __forceinline__ uint64_t mul_limb(uint64_t a, uint64_t b) {
    const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
    const uint32_t b0 = (uint32_t)b, b1 = (uint32_t)(b >> 32);
#if defined(__CUDA_ARCH__)
    uint32_t t0, t1, t2, t3, c;
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t0), "=r"(t1) : "r"(a0), "r"(b0));
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t2), "=r"(t3) : "r"(a1), "r"(b1));
    asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
        : "+r"(t1), "+r"(t2), "=r"(c)
        : "r"(a0), "r"(b1));
    asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, %5;"
        : "+r"(t1), "+r"(t2), "+r"(t3)
        : "r"(a1), "r"(b0), "r"(c));
    return reduce_limbs(t0, t1, t2, t3);
#else
    const uint64_t p00 = (uint64_t)a0 * b0, p01 = (uint64_t)a0 * b1;
    const uint64_t p10 = (uint64_t)a1 * b0, p11 = (uint64_t)a1 * b1;
    const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    const uint64_t top = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return reduce_limbs((uint32_t)p00, (uint32_t)mid, (uint32_t)top, (uint32_t)(top >> 32));
#endif
}

__device__ __forceinline__ uint64_t sqr_limb(uint64_t a) {
    const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
#if defined(__CUDA_ARCH__)
    uint32_t t0, t1, t2, t3, x0, x1;
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t0), "=r"(t1) : "r"(a0), "r"(a0));
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t2), "=r"(t3) : "r"(a1), "r"(a1));
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(x0), "=r"(x1) : "r"(a0), "r"(a1));
    asm("add.cc.u32 %0, %0, %3; addc.cc.u32 %1, %1, %4; addc.u32 %2, %2, 0;"
        : "+r"(t1), "+r"(t2), "+r"(t3)
        : "r"(x0), "r"(x1));
    asm("add.cc.u32 %0, %0, %3; addc.cc.u32 %1, %1, %4; addc.u32 %2, %2, 0;"
        : "+r"(t1), "+r"(t2), "+r"(t3)
        : "r"(x0), "r"(x1));
    return reduce_limbs(t0, t1, t2, t3);
#else
    const uint64_t p00 = (uint64_t)a0 * a0, x = (uint64_t)a0 * a1, p11 = (uint64_t)a1 * a1;
    const uint64_t mid = (p00 >> 32) + 2 * (uint64_t)(uint32_t)x;
    const uint64_t top = p11 + 2 * (x >> 32) + (mid >> 32);
    return reduce_limbs((uint32_t)p00, (uint32_t)mid, (uint32_t)top, (uint32_t)(top >> 32));
#endif
}

template <int V>
__device__ __forceinline__ uint64_t fmul(uint64_t a, uint64_t b) {
    if constexpr (V >= 1) {
        return mul_limb(a, b);
    } else {
        return goldilocks::mul(a, b);
    }
}

template <int V>
__device__ __forceinline__ uint64_t fsqr(uint64_t a) {
    if constexpr (V >= 1) {
        return sqr_limb(a);
    } else {
        return goldilocks::mul(a, a);
    }
}

// x^7 as x², x³ = x²·x, x⁶ = (x³)², x⁷ = x⁶·x — the host's association.
template <int V>
__device__ __forceinline__ uint64_t sbox(uint64_t x) {
    const uint64_t x2 = fsqr<V>(x);
    const uint64_t x3 = fmul<V>(x2, x);
    const uint64_t x6 = fsqr<V>(x3);
    return fmul<V>(x6, x);
}

// The circulant MDS, fully unrolled over compile-time entries: each lane is
// Σ_j c·lo_j and Σ_j c·hi_j over the 32-bit halves (each < 371·2^32 < 2^41),
// recombined to `hi·2^64 + lo` with `hi < 2^10` and folded by
// `2^64 ≡ EPSILON` — `rpx.cu`'s `mds` at width 16. Inputs may be raw
// `[0, 2^64)` storage; the output is congruent, not canonical.
__device__ __forceinline__ void mds(uint64_t s[WIDTH]) {
    uint32_t lo32[WIDTH], hi32[WIDTH];
#pragma unroll
    for (int j = 0; j < WIDTH; ++j) {
        lo32[j] = (uint32_t)s[j];
        hi32[j] = (uint32_t)(s[j] >> 32);
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) {
        uint64_t acc_lo = 0, acc_hi = 0;
#pragma unroll
        for (int j = 0; j < WIDTH; ++j) {
            const uint64_t c = MDS_ROW[(j + WIDTH - i) % WIDTH];
            acc_lo += c * (uint64_t)lo32[j];
            acc_hi += c * (uint64_t)hi32[j];
        }
        const uint64_t lo = (acc_hi << 32) + acc_lo;
        const uint64_t carry = (lo < acc_lo) ? 1ull : 0ull;
        const uint64_t hi = (acc_hi >> 32) + carry;
        s[i] = goldilocks::add(lo, hi * goldilocks::EPSILON);
    }
}

// One full round: constants, the S-box on every lane, the circulant MDS.
template <int V>
__device__ __forceinline__ void full_round(uint64_t s[WIDTH], int r) {
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = sbox<V>(goldilocks::add(s[i], RC[r][i]));
    mds(s);
}

// The 16-point DFT in registers, radix-2 decimation in time over a
// bit-reversed copy: X[k] = sum_n x[n]·omega^(±nk), omega = 2^12 (order 16).
// `INV` takes omega^-1 and does NOT scale by 1/16 (the caller folds it into
// the last partial round's eigenvalues). Every twiddle index is a
// compile-time constant after unrolling, so each product is by an immediate.
template <int V, bool INV>
__device__ __forceinline__ void dft16(uint64_t x[WIDTH]) {
    constexpr int BITREV[WIDTH] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    uint64_t y[WIDTH];
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) y[i] = x[BITREV[i]];
#pragma unroll
    for (int h = 1; h < WIDTH; h <<= 1) {
#pragma unroll
        for (int start = 0; start < WIDTH; start += 2 * h) {
#pragma unroll
            for (int j = 0; j < h; ++j) {
                const int e = j * (8 / h);  // omega_{2h}^j = omega^(j·16/(2h))
                const int ei = INV ? ((WIDTH - e) & (WIDTH - 1)) : e;
                const uint64_t a = y[start + j];
                const uint64_t t = (e == 0) ? y[start + j + h] : fmul<V>(y[start + j + h], OMEGA_POW[ei]);
                y[start + j] = goldilocks::add(a, t);
                y[start + j + h] = goldilocks::sub(a, t);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) x[i] = y[i];
}

// ★ The permutation with its 22 partial rounds in the Fourier domain of the
// circulant MDS (`scripts/poseidon1/p1_fourier.py`, checked equal to the
// textbook form there): the last initial full round's MDS becomes D·F, each
// partial round is s0 = (1/16)·Σŝ, δ = (s0 + c0)^7 − (s0 + c0),
// ŝ_j ← d_j·(ŝ_j + ĉ_j + δ), and one inverse DFT precedes the terminal full
// rounds. 21 products a partial round instead of a dense MDS.
template <int V>
__device__ __forceinline__ void permute_fourier(uint64_t s[WIDTH]) {
#pragma unroll 1
    for (int r = 0; r < HALF_FULL - 1; ++r) full_round<V>(s, r);
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = sbox<V>(goldilocks::add(s[i], RC[HALF_FULL - 1][i]));
    dft16<V, false>(s);
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = fmul<V>(s[i], FD[i]);
#pragma unroll 1
    for (int k = 0; k < PARTIAL; ++k) {
        uint64_t sum = s[0];
#pragma unroll
        for (int i = 1; i < WIDTH; ++i) sum = goldilocks::add(sum, s[i]);
        const uint64_t a0 = goldilocks::add(fmul<V>(sum, INV16), RC[HALF_FULL + k][0]);
        const uint64_t delta = goldilocks::sub(sbox<V>(a0), a0);
        if (k + 1 < PARTIAL) {
#pragma unroll
            for (int i = 0; i < WIDTH; ++i)
                s[i] = fmul<V>(goldilocks::add(goldilocks::add(s[i], FC[k][i]), delta), FD[i]);
        } else {
#pragma unroll
            for (int i = 0; i < WIDTH; ++i)
                s[i] = fmul<V>(goldilocks::add(goldilocks::add(s[i], FC[k][i]), delta), FD_LAST[i]);
        }
    }
    dft16<V, true>(s);
#pragma unroll 1
    for (int r = HALF_FULL + PARTIAL; r < ROUNDS; ++r) full_round<V>(s, r);
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = goldilocks::canonical(s[i]);
}

// A dot product of full-field constants with raw state words, accumulated
// as a 132-bit sum (lo, hi, top) and reduced once: 2^128 ≡ −2^32 (mod p),
// because 2^96 ≡ −1. `N` terms, the constants at compile-time indices.
template <int N>
__device__ __forceinline__ uint64_t dot_lazy(const uint64_t *c, const uint64_t *x) {
    uint64_t lo = 0, hi = 0, top = 0;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const uint64_t pl = c[j] * x[j];
        const uint64_t ph = __umul64hi(c[j], x[j]);
        lo += pl;
        const uint64_t c1 = (lo < pl) ? 1ull : 0ull;
        const uint64_t h2 = hi + ph;
        uint64_t c2 = (h2 < hi) ? 1ull : 0ull;
        hi = h2 + c1;
        c2 += (hi < h2) ? 1ull : 0ull;
        top += c2;
    }
    return goldilocks::sub(goldilocks::reduce128(lo, hi), top << 32);
}

// The dense product by the Grain Cauchy matrix — the MDS alternative. 256
// full products, where the circulant needs 512 small-constant MACs.
__device__ __forceinline__ void mds_cauchy(uint64_t s[WIDTH]) {
    uint64_t out[WIDTH];
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) out[i] = dot_lazy<WIDTH>(CAUCHY_M[i], s);
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = out[i];
}

template <int V>
__device__ __forceinline__ void full_round_cauchy(uint64_t s[WIDTH], const uint64_t *rc) {
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = sbox<V>(goldilocks::add(s[i], rc[i]));
    mds_cauchy(s);
}

// ★ The Cauchy alternative, with the paper's Appendix B sparse partial rounds
// (`scripts/poseidon1/p1_sparse.py`, checked equal to the textbook form there
// for this matrix): dense full rounds; CAUCHY_B0 on lanes 1..16 once; per
// partial round x0 = (s0 + C0[k])^7, new0 = N00[k]·x0 + W[k]·s[1..],
// s_i += V[k][i-1]·x0; the folded carry rides in CAUCHY_RC_T0. The Fourier
// trick does not apply: a Cauchy matrix is not circulant.
template <int V>
__device__ __forceinline__ void permute_cauchy(uint64_t s[WIDTH]) {
#pragma unroll 1
    for (int r = 0; r < HALF_FULL; ++r) full_round_cauchy<V>(s, RC[r]);
    {
        uint64_t t[WIDTH - 1];
#pragma unroll
        for (int i = 0; i < WIDTH - 1; ++i) t[i] = dot_lazy<WIDTH - 1>(CAUCHY_B0[i], s + 1);
#pragma unroll
        for (int i = 0; i < WIDTH - 1; ++i) s[i + 1] = t[i];
    }
#pragma unroll 1
    for (int k = 0; k < PARTIAL; ++k) {
        const uint64_t x0 = sbox<V>(goldilocks::add(s[0], CAUCHY_C0[k]));
        const uint64_t new0 =
            goldilocks::add(fmul<V>(CAUCHY_N00[k], x0), dot_lazy<WIDTH - 1>(CAUCHY_W[k], s + 1));
#pragma unroll
        for (int i = 1; i < WIDTH; ++i) s[i] = goldilocks::add(s[i], fmul<V>(CAUCHY_V[k][i - 1], x0));
        s[0] = new0;
    }
    full_round_cauchy<V>(s, CAUCHY_RC_T0);
#pragma unroll 1
    for (int r = HALF_FULL + PARTIAL + 1; r < ROUNDS; ++r) full_round_cauchy<V>(s, RC[r]);
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = goldilocks::canonical(s[i]);
}

// ★ The permutation. V = 2 takes the Fourier-domain partial rounds above,
// V = 3 the Cauchy alternative; otherwise one rolled round loop with a
// warp-uniform branch on the round kind, so the MDS body is emitted once.
template <int V>
__device__ __forceinline__ void permute(uint64_t s[WIDTH]) {
    if constexpr (V == 2) {
        permute_fourier<V>(s);
        return;
    }
    if constexpr (V == 3) {
        permute_cauchy<V>(s);
        return;
    }
#pragma unroll 1
    for (int r = 0; r < ROUNDS; ++r) {
#pragma unroll
        for (int i = 0; i < WIDTH; ++i) s[i] = goldilocks::add(s[i], RC[r][i]);
        if (r < HALF_FULL || r >= HALF_FULL + PARTIAL) {
#pragma unroll
            for (int i = 0; i < WIDTH; ++i) s[i] = sbox<V>(s[i]);
        } else {
            s[0] = sbox<V>(s[0]);
        }
        mds(s);
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = goldilocks::canonical(s[i]);
}

// The leaf sponge over `num_felts` felts read through `load(i)`. Blocks are
// written with compile-time lane indices (no dynamic state indexing).
template <int V, typename Load>
__device__ __forceinline__ void sponge_leaf(uint64_t num_felts, Load load, uint64_t digest[DIGEST]) {
    uint64_t s[WIDTH];
#pragma unroll
    for (int i = 0; i < RATE; ++i) s[i] = 0;
    s[CAP_PAD_LANE] = num_felts % RATE;
    s[CAP_DOMAIN_LANE] = DOMAIN_LEAF;
    s[CAP_DOMAIN_LANE + 1] = 0;
    s[CAP_DOMAIN_LANE + 2] = 0;
    uint64_t t = 0;
#pragma unroll 1
    for (; t + RATE <= num_felts; t += RATE) {
#pragma unroll
        for (int k = 0; k < RATE; ++k) s[k] = load(t + k);
        permute<V>(s);
    }
    if (t < num_felts) {
#pragma unroll
        for (int k = 0; k < RATE; ++k) s[k] = (t + k < num_felts) ? load(t + k) : 0;
        permute<V>(s);
    }
#pragma unroll
    for (int i = 0; i < DIGEST; ++i) digest[i] = s[i];
}

template <int V>
__device__ __forceinline__ void compress4(const uint64_t *children, uint64_t out[DIGEST]) {
    uint64_t s[WIDTH];
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = children[i];
    permute<V>(s);
#pragma unroll
    for (int i = 0; i < DIGEST; ++i) out[i] = s[i];
}

// ZisK's leaf hash (`linear_hash_seq` at width 16; host oracle
// `crypto::hash::poseidon1_stark::linear_hash`): each block of up to 12 felts
// overwrites lanes 0..12, zero-filled past its end; the capacity lanes 12..16
// are zero for the first block and the previous output's lanes 0..4 for every
// later one. No length, flag or domain. An empty leaf never permutes.
template <int V, typename Load>
__device__ __forceinline__ void zisk_leaf(uint64_t num_felts, Load load, uint64_t digest[DIGEST]) {
    uint64_t s[WIDTH];
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = 0;
    uint64_t t = 0;
#pragma unroll 1
    for (; t + RATE <= num_felts; t += RATE) {
        if (t > 0) {
#pragma unroll
            for (int i = 0; i < DIGEST; ++i) s[RATE + i] = s[i];
        }
#pragma unroll
        for (int k = 0; k < RATE; ++k) s[k] = load(t + k);
        permute<V>(s);
    }
    if (t < num_felts) {
        if (t > 0) {
#pragma unroll
            for (int i = 0; i < DIGEST; ++i) s[RATE + i] = s[i];
        }
#pragma unroll
        for (int k = 0; k < RATE; ++k) s[k] = (t + k < num_felts) ? load(t + k) : 0;
        permute<V>(s);
    }
#pragma unroll
    for (int i = 0; i < DIGEST; ++i) digest[i] = s[i];
}

}  // namespace p1w16

// ---------------------------------------------------------------------------
// Width 8: ZisK's grinding permutation (`crypto::hash::poseidon1_w8`). The same
// recipe at t = 8: x^7, R_F 8, R_P 22, Grain constants, the circulant
// `[7, 1, 3, 8, 8, 3, 4, 9]`; the 8-point DFT (omega = 2^24) diagonalises it.
//   V = 1  textbook rounds (rolled loop, dense circulant every round);
//   V = 2  the partial rounds in the Fourier domain (`p1_fourier.py` at T = 8).
// ---------------------------------------------------------------------------
#include "p1w8_constants.cuh"

namespace p1w8 {

enum : int { WIDTH = 8 };
using p1w16::HALF_FULL;
using p1w16::PARTIAL;
using p1w16::ROUNDS;

// The circulant over 32-bit halves, as `p1w16::mds`: each half-sum is below
// 43·2^32 < 2^38, `hi < 2^7`.
__device__ __forceinline__ void mds(uint64_t s[WIDTH]) {
    uint32_t lo32[WIDTH], hi32[WIDTH];
#pragma unroll
    for (int j = 0; j < WIDTH; ++j) {
        lo32[j] = (uint32_t)s[j];
        hi32[j] = (uint32_t)(s[j] >> 32);
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) {
        uint64_t acc_lo = 0, acc_hi = 0;
#pragma unroll
        for (int j = 0; j < WIDTH; ++j) {
            const uint64_t c = MDS_ROW[(j + WIDTH - i) % WIDTH];
            acc_lo += c * (uint64_t)lo32[j];
            acc_hi += c * (uint64_t)hi32[j];
        }
        const uint64_t lo = (acc_hi << 32) + acc_lo;
        const uint64_t carry = (lo < acc_lo) ? 1ull : 0ull;
        const uint64_t hi = (acc_hi >> 32) + carry;
        s[i] = goldilocks::add(lo, hi * goldilocks::EPSILON);
    }
}

template <int V>
__device__ __forceinline__ void full_round(uint64_t s[WIDTH], int r) {
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = p1w16::sbox<V>(goldilocks::add(s[i], RC[r][i]));
    mds(s);
}

// The 8-point DFT, `p1w16::dft16`'s radix-2 form at width 8.
template <int V, bool INV>
__device__ __forceinline__ void dft8(uint64_t x[WIDTH]) {
    constexpr int BITREV[WIDTH] = {0, 4, 2, 6, 1, 5, 3, 7};
    uint64_t y[WIDTH];
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) y[i] = x[BITREV[i]];
#pragma unroll
    for (int h = 1; h < WIDTH; h <<= 1) {
#pragma unroll
        for (int start = 0; start < WIDTH; start += 2 * h) {
#pragma unroll
            for (int j = 0; j < h; ++j) {
                const int e = j * (4 / h);  // omega_{2h}^j = omega^(j·8/(2h))
                const int ei = INV ? ((WIDTH - e) & (WIDTH - 1)) : e;
                const uint64_t a = y[start + j];
                const uint64_t t =
                    (e == 0) ? y[start + j + h] : p1w16::fmul<V>(y[start + j + h], OMEGA_POW[ei]);
                y[start + j] = goldilocks::add(a, t);
                y[start + j + h] = goldilocks::sub(a, t);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) x[i] = y[i];
}

// ★ The permutation; the output is canonical.
template <int V>
__device__ __forceinline__ void permute(uint64_t s[WIDTH]) {
    if constexpr (V == 2) {
#pragma unroll 1
        for (int r = 0; r < HALF_FULL - 1; ++r) full_round<V>(s, r);
#pragma unroll
        for (int i = 0; i < WIDTH; ++i)
            s[i] = p1w16::sbox<V>(goldilocks::add(s[i], RC[HALF_FULL - 1][i]));
        dft8<V, false>(s);
#pragma unroll
        for (int i = 0; i < WIDTH; ++i) s[i] = p1w16::fmul<V>(s[i], FD[i]);
#pragma unroll 1
        for (int k = 0; k < PARTIAL; ++k) {
            uint64_t sum = s[0];
#pragma unroll
            for (int i = 1; i < WIDTH; ++i) sum = goldilocks::add(sum, s[i]);
            const uint64_t a0 = goldilocks::add(p1w16::fmul<V>(sum, INV8), RC[HALF_FULL + k][0]);
            const uint64_t delta = goldilocks::sub(p1w16::sbox<V>(a0), a0);
            if (k + 1 < PARTIAL) {
#pragma unroll
                for (int i = 0; i < WIDTH; ++i)
                    s[i] = p1w16::fmul<V>(goldilocks::add(goldilocks::add(s[i], FC[k][i]), delta), FD[i]);
            } else {
#pragma unroll
                for (int i = 0; i < WIDTH; ++i)
                    s[i] = p1w16::fmul<V>(goldilocks::add(goldilocks::add(s[i], FC[k][i]), delta),
                                          FD_LAST[i]);
            }
        }
        dft8<V, true>(s);
#pragma unroll 1
        for (int r = HALF_FULL + PARTIAL; r < ROUNDS; ++r) full_round<V>(s, r);
    } else {
#pragma unroll 1
        for (int r = 0; r < ROUNDS; ++r) {
#pragma unroll
            for (int i = 0; i < WIDTH; ++i) s[i] = goldilocks::add(s[i], RC[r][i]);
            if (r < HALF_FULL || r >= HALF_FULL + PARTIAL) {
#pragma unroll
                for (int i = 0; i < WIDTH; ++i) s[i] = p1w16::sbox<V>(s[i]);
            } else {
                s[0] = p1w16::sbox<V>(s[0]);
            }
            mds(s);
        }
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) s[i] = goldilocks::canonical(s[i]);
}

}  // namespace p1w8

// ---------------------------------------------------------------------------
// The `extern "C"` surface, one entry per variant.
// ---------------------------------------------------------------------------

// Parity probe: `n` independent permutations, one thread each.
template <int V>
__device__ __forceinline__ void permute_probe(const uint64_t *states, uint64_t n, uint64_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t s[p1w16::WIDTH];
#pragma unroll
    for (int i = 0; i < p1w16::WIDTH; ++i) s[i] = states[tid * p1w16::WIDTH + i];
    p1w16::permute<V>(s);
#pragma unroll
    for (int i = 0; i < p1w16::WIDTH; ++i) out[tid * p1w16::WIDTH + i] = s[i];
}

// Base-field coset leaves: leaf `tid` hashes `codeword[tid + t·num_leaves]`,
// `t` in `[0, block)` — `rpx_leaves_base_coset`'s geometry.
template <int V>
__device__ __forceinline__ void leaves_base_coset(const uint64_t *__restrict__ codeword,
                                                  uint64_t num_leaves, uint64_t block,
                                                  uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    uint64_t d[p1w16::DIGEST];
    p1w16::sponge_leaf<V>(
        block, [&](uint64_t i) { return codeword[tid + i * num_leaves]; }, d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) out[tid * p1w16::DIGEST + i] = d[i];
}

// Ext3 coset leaves: each element's three components in order.
template <int V>
__device__ __forceinline__ void leaves_ext3_coset(const uint64_t *__restrict__ codeword,
                                                  uint64_t num_leaves, uint64_t block,
                                                  uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    uint64_t d[p1w16::DIGEST];
    p1w16::sponge_leaf<V>(
        3 * block,
        [&](uint64_t i) { return codeword[(tid + (i / 3) * num_leaves) * 3 + i % 3]; }, d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) out[tid * p1w16::DIGEST + i] = d[i];
}

// One 4-ary level: parent `tid` = compress4(children[4·tid .. 4·tid + 4]).
template <int V>
__device__ __forceinline__ void merkle_level4(const uint64_t *__restrict__ children,
                                              uint64_t *__restrict__ parents, uint64_t n_parents) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_parents) return;
    uint64_t d[p1w16::DIGEST];
    p1w16::compress4<V>(children + tid * p1w16::WIDTH, d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) parents[tid * p1w16::DIGEST + i] = d[i];
}

// The grind search: `rpx_grind_search`'s loop, first-hit `atomicMin` and poll,
// with this sponge. The head is lane 0, canonical after `permute`.
template <int V>
__device__ __forceinline__ void grind_search(const uint64_t *inner, uint64_t limit, uint64_t base,
                                             uint64_t count, volatile unsigned long long *result) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t f[4] = {inner[0], inner[1], inner[2], inner[3]};
    for (uint64_t i = tid; i < count; i += stride) {
        uint64_t nonce = base + i;
        if (nonce < base) break;
        if (nonce >= (uint64_t)*result) break;
        uint64_t s[p1w16::WIDTH];
#pragma unroll
        for (int k = 0; k < 4; ++k) s[k] = f[k];
        s[4] = goldilocks::canonical(nonce);
#pragma unroll
        for (int k = 5; k < p1w16::RATE; ++k) s[k] = 0;
        s[p1w16::CAP_PAD_LANE] = p1w16::GRIND_FELTS % p1w16::RATE;
        s[p1w16::CAP_DOMAIN_LANE] = p1w16::DOMAIN_LEAF;
        s[p1w16::CAP_DOMAIN_LANE + 1] = 0;
        s[p1w16::CAP_DOMAIN_LANE + 2] = 0;
        p1w16::permute<V>(s);
        if (s[0] < limit) {
            atomicMin((unsigned long long *)result, (unsigned long long)nonce);
        }
    }
}

// Register caps for the occupancy sweep: `__launch_bounds__(128, minb)` asks for
// `minb` resident 128-thread blocks per SM, i.e. at most 65536 / (128·minb)
// registers per thread (128 at 4, 102 at 5, 85 at 6). The host shim has none.
#if defined(__CUDACC__)
#define P1W16_LB(minb) __launch_bounds__(128, minb)
#else
#define P1W16_LB(minb)
#endif

#define P1W16_ENTRIES(NAME, V, BOUND)                                                             \
    extern "C" __global__ void BOUND p1w16_permute_probe_##NAME(const uint64_t *states,          \
                                                                uint64_t n, uint64_t *out) {      \
        permute_probe<V>(states, n, out);                                                          \
    }                                                                                              \
    extern "C" __global__ void BOUND p1w16_leaves_base_coset_##NAME(                              \
        const uint64_t *__restrict__ codeword, uint64_t num_leaves, uint64_t block,               \
        uint64_t *__restrict__ out) {                                                              \
        leaves_base_coset<V>(codeword, num_leaves, block, out);                                    \
    }                                                                                              \
    extern "C" __global__ void BOUND p1w16_leaves_ext3_coset_##NAME(                              \
        const uint64_t *__restrict__ codeword, uint64_t num_leaves, uint64_t block,               \
        uint64_t *__restrict__ out) {                                                              \
        leaves_ext3_coset<V>(codeword, num_leaves, block, out);                                    \
    }                                                                                              \
    extern "C" __global__ void BOUND p1w16_merkle_level4_##NAME(                                  \
        const uint64_t *__restrict__ children, uint64_t *__restrict__ parents,                    \
        uint64_t n_parents) {                                                                      \
        merkle_level4<V>(children, parents, n_parents);                                            \
    }                                                                                              \
    extern "C" __global__ void BOUND p1w16_grind_search_##NAME(                                   \
        const uint64_t *inner, uint64_t limit, uint64_t base, uint64_t count,                     \
        volatile unsigned long long *result) {                                                     \
        grind_search<V>(inner, limit, base, count, result);                                        \
    }

// The two multiply variants, the Fourier-domain partial rounds, and the
// Cauchy alternative (`src/p1w16.rs` VARIANTS lists the same names in the
// same order). FAST job
// 281 measured the register caps (4/5/6 blocks of 128) SLOWER than uncapped
// (+6 %, +63 %, ×4.6), so no capped variant is instantiated; `P1W16_LB` stays
// for the next sweep.
P1W16_ENTRIES(v0, 0, )
P1W16_ENTRIES(v1, 1, )
P1W16_ENTRIES(v2, 2, )
P1W16_ENTRIES(c1, 3, )

// ---------------------------------------------------------------------------
// ZisK's instance (stage P1 of the Poseidon1 base STARK): its leaf hash over
// the coset geometry above and over the STARK's column-major LDE rows, and its
// width-8 grind. The Fourier-domain permutation (V = 2) only; the 4-ary node
// is `p1w16_merkle_level4_v2` unchanged. Digests are four canonical u64s.
// ---------------------------------------------------------------------------

// Coset leaves under ZisK's leaf hash: leaf `tid` hashes
// `codeword[tid + t·num_leaves]` (base) or that element's three components
// (ext3), `t` in `[0, block)`.
extern "C" __global__ void p1w16_zleaves_base_coset_v2(const uint64_t *__restrict__ codeword,
                                                       uint64_t num_leaves, uint64_t block,
                                                       uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        block, [&](uint64_t i) { return codeword[tid + i * num_leaves]; }, d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) out[tid * p1w16::DIGEST + i] = d[i];
}

extern "C" __global__ void p1w16_zleaves_ext3_coset_v2(const uint64_t *__restrict__ codeword,
                                                       uint64_t num_leaves, uint64_t block,
                                                       uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        3 * block,
        [&](uint64_t i) { return codeword[(tid + (i / 3) * num_leaves) * 3 + i % 3]; }, d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) out[tid * p1w16::DIGEST + i] = d[i];
}

// One leaf per bit-reversed LDE row: column `c` of row `br` at
// `columns[c * col_stride + br]`, absorbed column by column —
// `rpx_leaves_base_batched`'s geometry. An ext3 matrix stored as three base
// slabs per column (`rpx_leaves_ext3_batched`) is this kernel over
// `num_cols = 3 · ext3 columns`: felt `3c + k` is slab `3c + k`.
extern "C" __global__ void p1w16_zleaves_rows_v2(const uint64_t *__restrict__ columns,
                                                 uint64_t col_stride, uint64_t num_cols,
                                                 uint64_t num_rows, uint64_t log_num_rows,
                                                 uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    const uint64_t br = __brevll(tid) >> (64 - log_num_rows);
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        num_cols, [&](uint64_t c) { return columns[c * col_stride + br]; }, d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) out[tid * p1w16::DIGEST + i] = d[i];
}

// Row-pair leaves: leaf `tid` hashes bit-reversed rows `2·tid` then `2·tid + 1`,
// each column by column — `rpx_leaves_base_row_pair_batched`'s geometry.
extern "C" __global__ void p1w16_zleaves_row_pair_v2(const uint64_t *__restrict__ columns,
                                                     uint64_t col_stride, uint64_t num_cols,
                                                     uint64_t num_rows, uint64_t log_num_rows,
                                                     uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows / 2) return;
    const uint64_t br0 = __brevll(2 * tid) >> (64 - log_num_rows);
    const uint64_t br1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        2 * num_cols,
        [&](uint64_t i) {
            return i < num_cols ? columns[i * col_stride + br0]
                                : columns[(i - num_cols) * col_stride + br1];
        },
        d);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) out[tid * p1w16::DIGEST + i] = d[i];
}

// Width-8 parity probe: `n` independent permutations.
template <int V>
__device__ __forceinline__ void p1w8_probe(const uint64_t *states, uint64_t n, uint64_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t s[p1w8::WIDTH];
#pragma unroll
    for (int i = 0; i < p1w8::WIDTH; ++i) s[i] = states[tid * p1w8::WIDTH + i];
    p1w8::permute<V>(s);
#pragma unroll
    for (int i = 0; i < p1w8::WIDTH; ++i) out[tid * p1w8::WIDTH + i] = s[i];
}

// ZisK's grind: the smallest nonce in `[base, base + count)` whose width-8
// permutation of `[c0, c1, c2, nonce, 0, 0, 0, 0]` has lane 0 below `limit`
// (`crypto::hash::poseidon1_stark::grinding_lane0`). The grid-stride loop,
// first-hit `atomicMin` and poll of `grind_search` above.
template <int V>
__device__ __forceinline__ void p1w8_grind(const uint64_t *challenge, uint64_t limit, uint64_t base,
                                           uint64_t count, volatile unsigned long long *result) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t c0 = challenge[0], c1 = challenge[1], c2 = challenge[2];
    for (uint64_t i = tid; i < count; i += stride) {
        uint64_t nonce = base + i;
        if (nonce < base) break;
        if (nonce >= (uint64_t)*result) break;
        uint64_t s[p1w8::WIDTH] = {c0, c1, c2, goldilocks::canonical(nonce), 0, 0, 0, 0};
        p1w8::permute<V>(s);
        if (s[0] < limit) {
            atomicMin((unsigned long long *)result, (unsigned long long)nonce);
        }
    }
}

#define P1W8_ENTRIES(NAME, V)                                                                      \
    extern "C" __global__ void p1w8_permute_probe_##NAME(const uint64_t *states, uint64_t n,       \
                                                         uint64_t *out) {                          \
        p1w8_probe<V>(states, n, out);                                                             \
    }                                                                                              \
    extern "C" __global__ void p1w8_grind_search_##NAME(const uint64_t *challenge, uint64_t limit, \
                                                        uint64_t base, uint64_t count,             \
                                                        volatile unsigned long long *result) {     \
        p1w8_grind<V>(challenge, limit, base, count, result);                                      \
    }

P1W8_ENTRIES(v1, 1)
P1W8_ENTRIES(v2, 2)

// Bench fill: `out[i]` = a SplitMix64 of `i`, canonicalised — a codeword the
// microbench hashes without a host upload.
extern "C" __global__ void p1w16_fill(uint64_t *out, uint64_t n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t z = tid + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    out[tid] = goldilocks::canonical(z ^ (z >> 31));
}

// ===========================================================================
// PRODUCTION (`p1/*` P2): ZisK's instance on the STARK's device commit paths,
// launched by `src/p1_stark.rs` for `DeviceHash::Poseidon1`.
//
// NODE BYTES are the host's commitment bytes, as `rpx.cu`'s: four canonical
// felts, each eight BIG-endian bytes (`algebraic_commit::digest_to_commitment`),
// so a device node buffer is the host's byte for byte and a parent reads its
// children back with `commitment_to_digest`'s decoding.
//
// TREES are 4-ary in the host's arity-4 layout (`crypto::merkle_tree::utils::
// level_offsets4`): the levels top-down, root at node 0, leaves last; a level
// holds ⌈below / 4⌉ nodes, so the level above the one at `off` with `n` nodes
// starts at `off − ⌈n / 4⌉`. A short group's missing children are the zero
// digest (ZisK's rule; `P1BatchBackend::padding_node`) and are never stored.
// A parent is `compress4`: the four child digests fill the sixteen lanes,
// one permutation, lanes 0..4.
//
// LEAVES are `zisk_leaf` over the felt sequence the host leaf hashes: the read
// patterns of `rpx.cu`'s leaf kernels (bit-reversed rows, column by column; a
// row pair is the first row then the second; an ext3 element as its three
// components), which is what `element_felts` / `felts_from_bytes` give on the
// host.
// ===========================================================================

namespace p1s {

// Byte-swap a u64 (`rpx::bswap64`).
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    x = ((x & 0x00FF00FF00FF00FFull) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFull);
    x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFull);
    return (x << 32) | (x >> 32);
}

__device__ __forceinline__ void store_be(const uint64_t d[p1w16::DIGEST], uint8_t *node) {
    uint64_t *dst = reinterpret_cast<uint64_t *>(node);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) dst[i] = bswap64(d[i]);
}

__device__ __forceinline__ void load_be(const uint8_t *node, uint64_t *d) {
    const uint64_t *src = reinterpret_cast<const uint64_t *>(node);
#pragma unroll
    for (int i = 0; i < p1w16::DIGEST; ++i) d[i] = bswap64(src[i]);
}

// Parent `p` of the level at `child_off` (`n_children` nodes), written at
// `parent_off + p`.
__device__ __forceinline__ void parent4(uint8_t *nodes, uint64_t child_off, uint64_t n_children,
                                        uint64_t parent_off, uint64_t p) {
    uint64_t s[p1w16::WIDTH];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint64_t idx = 4 * p + (uint64_t)c;
        if (idx < n_children) {
            load_be(nodes + (child_off + idx) * 32, s + 4 * c);
        } else {
#pragma unroll
            for (int i = 0; i < p1w16::DIGEST; ++i) s[4 * c + i] = 0;
        }
    }
    p1w16::permute<2>(s);
    store_be(s, nodes + (parent_off + p) * 32);
}

}  // namespace p1s

// One leaf per bit-reversed row of a column-major matrix: column `c` of row
// `br` at `cols[c * col_stride + br]` (`rpx_leaves_base_batched`'s geometry).
// An ext3 matrix stored as three base slabs per column
// (`rpx_leaves_ext3_batched`) is this kernel over `num_cols = 3 · columns`.
extern "C" __global__ void p1s_leaves_cols_row(const uint64_t *__restrict__ cols, uint64_t col_stride,
                                               uint64_t num_cols, uint64_t num_rows,
                                               uint64_t log_num_rows, uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    const uint64_t br = __brevll(tid) >> (64 - log_num_rows);
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        num_cols, [&](uint64_t c) { return cols[c * col_stride + br]; }, d);
    p1s::store_be(d, out + tid * 32);
}

// Row-pair leaves of a column-major matrix: leaf `tid` hashes bit-reversed rows
// `2·tid` then `2·tid + 1`, each column by column
// (`rpx_leaves_base_row_pair_batched`). Over `num_cols = 3 · parts` slabs it is
// the composition tree's ext3 row-pair leaf (`rpx_comp_poly_leaves_ext3`).
extern "C" __global__ void p1s_leaves_cols_pair(const uint64_t *__restrict__ cols, uint64_t col_stride,
                                                uint64_t num_cols, uint64_t num_rows,
                                                uint64_t log_num_rows, uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows / 2) return;
    const uint64_t br0 = __brevll(2 * tid) >> (64 - log_num_rows);
    const uint64_t br1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        2 * num_cols,
        [&](uint64_t i) {
            return i < num_cols ? cols[i * col_stride + br0] : cols[(i - num_cols) * col_stride + br1];
        },
        d);
    p1s::store_be(d, out + tid * 32);
}

// Row-major row-pair leaves over columns `[col_start, col_end)` of rows of
// stride `m` (`rpx_leaves_base_row_major_row_pair[_range]`; the full row is
// the range `[0, m)`).
extern "C" __global__ void p1s_leaves_rm_pair(const uint64_t *__restrict__ data, uint64_t m,
                                              uint64_t col_start, uint64_t col_end,
                                              uint64_t num_rows, uint64_t log_num_rows,
                                              uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows / 2) return;
    const uint64_t *row0 = data + (__brevll(2 * tid) >> (64 - log_num_rows)) * m + col_start;
    const uint64_t *row1 = data + (__brevll(2 * tid + 1) >> (64 - log_num_rows)) * m + col_start;
    const uint64_t w = col_end - col_start;
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        2 * w, [&](uint64_t i) { return i < w ? row0[i] : row1[i - w]; }, d);
    p1s::store_be(d, out + tid * 32);
}

// Row-major one-row leaves over columns `[col_start, col_end)`
// (`rpx_leaves_base_row_major_row_range`).
extern "C" __global__ void p1s_leaves_rm_row(const uint64_t *__restrict__ data, uint64_t m,
                                             uint64_t col_start, uint64_t col_end,
                                             uint64_t num_rows, uint64_t log_num_rows,
                                             uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    const uint64_t *row = data + (__brevll(tid) >> (64 - log_num_rows)) * m + col_start;
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        col_end - col_start, [&](uint64_t i) { return row[i]; }, d);
    p1s::store_be(d, out + tid * 32);
}

// FRI leaves: leaf `tid` hashes the `3 · group` contiguous felts of the `group`
// consecutive ext3 values from `tid · group` of an interleaved eval vector
// (`rpx_fri_group_leaves_ext3`; the pair leaf is `group = 2`, six felts, as
// `P1PairBackend::hash_data`).
extern "C" __global__ void p1s_fri_group_leaves(const uint64_t *__restrict__ evals, uint64_t num_leaves,
                                                uint64_t group, uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    const uint64_t *g = evals + tid * group * 3;
    uint64_t d[p1w16::DIGEST];
    p1w16::zisk_leaf<2>(
        3 * group, [&](uint64_t i) { return g[i]; }, d);
    p1s::store_be(d, out + tid * 32);
}

// One 4-ary level in place: parent `tid` of the level at `child_off`
// (`n_children` nodes) into the level at `parent_off` (`n_parents` nodes).
extern "C" __global__ void p1s_merkle_level4(uint8_t *nodes, uint64_t child_off, uint64_t n_children,
                                             uint64_t parent_off, uint64_t n_parents) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_parents) return;
    p1s::parent4(nodes, child_off, n_children, parent_off, tid);
}

// Every level from the one at `child_off` (`n_children` nodes) up to the root,
// in one block, a barrier between levels.
extern "C" __global__ void p1s_merkle_tail4(uint8_t *nodes, uint64_t child_off, uint64_t n_children) {
    while (n_children > 1) {
        const uint64_t n_parents = (n_children + 3) / 4;
        const uint64_t parent_off = child_off - n_parents;
        for (uint64_t p = threadIdx.x; p < n_parents; p += blockDim.x) {
            p1s::parent4(nodes, child_off, n_children, parent_off, p);
        }
        __syncthreads();
        child_off = parent_off;
        n_children = n_parents;
    }
}

// Authentication paths: query `tid`'s path is `depth4` levels from the leaves
// up, each the three other children of its group in child order, the zero
// digest where the group is short — `MerkleTree::get_proof_by_pos` at arity 4.
// `total_nodes` is the tree's node count (the leaves start at
// `total_nodes − leaves_len`).
extern "C" __global__ void p1s_gather_paths4(const uint8_t *__restrict__ nodes,
                                             const uint32_t *__restrict__ positions, uint32_t nq,
                                             uint64_t leaves_len, uint64_t total_nodes,
                                             uint32_t depth4, uint8_t *__restrict__ out) {
    uint64_t q = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nq) return;
    uint64_t i = positions[q];
    uint64_t n = leaves_len;
    uint64_t off = total_nodes - leaves_len;
    uint64_t *dst = reinterpret_cast<uint64_t *>(out + q * (uint64_t)depth4 * 3 * 32);
    for (uint32_t level = 0; level < depth4; ++level) {
        const uint64_t first = i & ~3ull;
        for (uint64_t c = first; c < first + 4; ++c) {
            if (c == i) continue;
            if (c < n) {
                const uint64_t *src = reinterpret_cast<const uint64_t *>(nodes + (off + c) * 32);
#pragma unroll
                for (int k = 0; k < 4; ++k) dst[k] = src[k];
            } else {
#pragma unroll
                for (int k = 0; k < 4; ++k) dst[k] = 0;
            }
            dst += 4;
        }
        const uint64_t parents = (n + 3) / 4;
        off -= parents;
        n = parents;
        i >>= 2;
    }
}

// The STARK grind (`P1GrindDigest`, `crypto::grinding`'s two-level form): the
// smallest nonce in `[base, base + count)` whose width-8 permutation of
// `[inner0, inner1, inner2, inner3, nonce, 0, 0, 0]` has lane 0 below `limit`.
// `inner` is the inner hash's four big-endian felts (`inner_hash_felts`). The
// grid-stride loop, first-hit `atomicMin` and poll of `p1w8_grind`.
extern "C" __global__ void p1s_grind_w8(const uint64_t *inner, uint64_t limit, uint64_t base,
                                        uint64_t count, volatile unsigned long long *result) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t i0 = inner[0], i1 = inner[1], i2 = inner[2], i3 = inner[3];
    for (uint64_t i = tid; i < count; i += stride) {
        uint64_t nonce = base + i;
        if (nonce < base) break;
        if (nonce >= (uint64_t)*result) break;
        uint64_t s[p1w8::WIDTH] = {i0, i1, i2, i3, goldilocks::canonical(nonce), 0, 0, 0};
        p1w8::permute<2>(s);
        if (s[0] < limit) {
            atomicMin((unsigned long long *)result, (unsigned long long)nonce);
        }
    }
}
