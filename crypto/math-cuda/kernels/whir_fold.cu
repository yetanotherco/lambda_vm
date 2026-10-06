// Folding a WHIR codeword on device.
//
// One fold halves the codeword, pairing `j` with `j + half` — the two points
// `x` and `−x`, which is half a period apart on a multiplicative domain:
//
//   out[j] = two_inv·(a + b) + (two_inv·g^{-j})·(a − b)·alpha
//
// where `a = in[j]`, `b = in[j + half]`. A transliteration of `fold_codeword`
// in `crypto/multilinear/src/whir.rs`, and `tests/whir_fold.rs` pins it there.
//
// `g^{-j}` is computed per thread by squaring rather than read from a table:
// the fold is bound by the codeword it streams, and a table would be another
// pass over memory the exponentiation does not need.
//
// Two entry points: the first fold of a chain reads the committed base-field
// codeword and lifts it, every later one is ext3 throughout. Ext3 values are
// interleaved (`[a0,b0,c0, a1,b1,c1, ...]`), the layout the rest of the crate
// uses.

#include "goldilocks.cuh"
#include "ext3.cuh"

using ext3::Fe3;

__device__ __forceinline__ uint64_t pow_base(uint64_t base, uint64_t exp) {
    uint64_t acc = 1;
    uint64_t square = base;
    while (exp > 0) {
        if (exp & 1ull) {
            acc = goldilocks::mul(acc, square);
        }
        square = goldilocks::mul(square, square);
        exp >>= 1;
    }
    return acc;
}

extern "C" __global__ void whir_fold_base_ext3(const uint64_t *__restrict__ in, uint64_t half,
                                               uint64_t two_inv, uint64_t g_inv,
                                               const uint64_t *__restrict__ alpha,
                                               uint64_t *__restrict__ out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;

    uint64_t a = in[j];
    uint64_t b = in[j + half];
    uint64_t even = goldilocks::mul(two_inv, goldilocks::add(a, b));
    uint64_t scale = goldilocks::mul(two_inv, pow_base(g_inv, j));
    uint64_t odd = goldilocks::mul(scale, goldilocks::sub(a, b));

    // `odd·alpha` with `odd` in the base field, then `even +` it: the two
    // mixed-field shortcuts the tower gives, both bit-identical to the full
    // ext3 ops on the embedded operand.
    Fe3 term = ext3::mul_base(ext3::make(alpha[0], alpha[1], alpha[2]), odd);
    uint64_t *at = out + j * 3;
    at[0] = goldilocks::add(even, term.a);
    at[1] = term.b;
    at[2] = term.c;
}

extern "C" __global__ void whir_fold_ext3(const uint64_t *__restrict__ in, uint64_t half,
                                          uint64_t two_inv, uint64_t g_inv,
                                          const uint64_t *__restrict__ alpha,
                                          uint64_t *__restrict__ out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;

    const uint64_t *lo = in + j * 3;
    const uint64_t *hi = in + (j + half) * 3;
    Fe3 a = ext3::make(lo[0], lo[1], lo[2]);
    Fe3 b = ext3::make(hi[0], hi[1], hi[2]);
    Fe3 even = ext3::mul_base(ext3::add(a, b), two_inv);
    uint64_t scale = goldilocks::mul(two_inv, pow_base(g_inv, j));
    Fe3 odd = ext3::mul_base(ext3::sub(a, b), scale);
    Fe3 res = ext3::add(even, ext3::mul(odd, ext3::make(alpha[0], alpha[1], alpha[2])));

    uint64_t *at = out + j * 3;
    at[0] = res.a;
    at[1] = res.b;
    at[2] = res.c;
}

// The fold blocks a round's queries open, gathered into one buffer:
// `out[(q*block + t)*limbs ..]` is `codeword[(index[q] + t*num_leaves)*limbs]`.
// One launch and one copy back, against one of each per value.
extern "C" __global__ void gather_cosets(const uint64_t *__restrict__ codeword,
                                         const uint64_t *__restrict__ indices, uint64_t queries,
                                         uint64_t num_leaves, uint64_t block, uint64_t limbs,
                                         uint64_t *__restrict__ out) {
    uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (task >= queries * block) return;
    uint64_t q = task / block;
    uint64_t t = task - q * block;
    const uint64_t *from = codeword + (indices[q] + t * num_leaves) * limbs;
    uint64_t *at = out + task * limbs;
    for (uint64_t k = 0; k < limbs; ++k) {
        at[k] = from[k];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EVERY LEVEL OF A ROUND'S FOLD IN ONE PASS.
//
// Folding `k` levels one launch at a time writes every intermediate codeword:
// the first level alone is `2^(n+1)` extension values — 1.5× the committed
// base codeword — and it is live beside the second level's (0.75×) while the
// committed codeword still is. But output `j` depends only on the strided
// coset `{j + t·n_out : t < 2^k}` (`n_out = N / 2^k`), so one thread folds its
// coset to the end in registers and the only thing written is the output.
//
// Inside a coset, level `L` pairs `(t, t + 2^(k−1−L))` — the TOP bit of that
// level's index — so the coset is a binary tree over its index bits from the
// top down. The thread walks the leaf pairs in bit-reversed order, which makes
// the two halves of every later pair adjacent in the walk, and merges them on a
// stack holding one pending value per level (a binary counter): `k` extension
// values of state rather than `2^(k−1)`.
//
// ★ RAW-IDENTICAL, not merely equal. A proof serializes raw limbs, which may be
// non-canonical, so every pair is folded with the exact operation sequence of
// `whir_fold_base_ext3` / `whir_fold_ext3` above — including `pow_base`'s
// square-and-multiply for `g^{-pos}`, replayed from a table of the same squares
// in the same ascending-bit order — on the same raw inputs. The output is the
// level-by-level result bit for bit (`tests/whir_fold.rs`,
// `tests/host_kat/whir_host_kat.cpp`).
// ─────────────────────────────────────────────────────────────────────────────

#define WHIR_MAX_FOLD 6
// Positions stay below 2^32: a codeword longer than that is refused on the host
// (`commit_from`), so 32 squares cover every exponent `pow_base` could see.
#define WHIR_POW_BITS 32

__device__ __forceinline__ uint32_t whir_reverse_low_bits(uint32_t x, uint32_t bits) {
    uint32_t r = 0;
    for (uint32_t i = 0; i < bits; ++i) {
        r = (r << 1) | ((x >> i) & 1u);
    }
    return r;
}

// `pow_base(base, exp)` from the squares `pow_base` would compute, in the order
// it would multiply them: bit for bit its result.
__device__ __forceinline__ uint64_t whir_pow_from_squares(const uint64_t *squares, uint64_t exp) {
    uint64_t acc = 1;
    uint32_t i = 0;
    while (exp > 0) {
        if (exp & 1ull) {
            acc = goldilocks::mul(acc, squares[i]);
        }
        exp >>= 1;
        ++i;
    }
    return acc;
}

// `whir_fold_base_ext3`'s arithmetic for one pair, `x_pow = g^{-pos}`.
__device__ __forceinline__ Fe3 whir_fold_pair_base(uint64_t a, uint64_t b, uint64_t two_inv,
                                                   uint64_t x_pow, const Fe3 &alpha) {
    uint64_t even = goldilocks::mul(two_inv, goldilocks::add(a, b));
    uint64_t scale = goldilocks::mul(two_inv, x_pow);
    uint64_t odd = goldilocks::mul(scale, goldilocks::sub(a, b));
    Fe3 term = ext3::mul_base(alpha, odd);
    return ext3::make(goldilocks::add(even, term.a), term.b, term.c);
}

// `whir_fold_ext3`'s arithmetic for one pair.
__device__ __forceinline__ Fe3 whir_fold_pair_ext3(const Fe3 &a, const Fe3 &b, uint64_t two_inv,
                                                   uint64_t x_pow, const Fe3 &alpha) {
    Fe3 even = ext3::mul_base(ext3::add(a, b), two_inv);
    uint64_t scale = goldilocks::mul(two_inv, x_pow);
    Fe3 odd = ext3::mul_base(ext3::sub(a, b), scale);
    return ext3::add(even, ext3::mul(odd, alpha));
}

// The squares `pow_base(g_inv, ·)` walks for level `level`'s inverse
// generator: `squares[level·WHIR_POW_BITS + i] = g_inv^(2^i)`, each the square
// of the one before, exactly as `pow_base` computes them.
__device__ __forceinline__ void whir_fold_squares(uint64_t *squares, uint32_t level,
                                                  uint64_t g_inv) {
    uint64_t square = g_inv;
    for (uint32_t i = 0; i < WHIR_POW_BITS; ++i) {
        squares[level * WHIR_POW_BITS + i] = square;
        square = goldilocks::mul(square, square);
    }
}

// Output `j` of a `k`-level fold of `in` (`n_out · 2^k` values, base-field when
// `base`), level `L` at challenge `alphas[3L..3L+3]` and at the inverse
// generator whose squares are `squares[L·WHIR_POW_BITS ..]`.
__device__ Fe3 whir_fold_coset(const uint64_t *__restrict__ in, bool base, uint64_t j,
                               uint64_t n_out, uint32_t k, uint64_t two_inv,
                               const uint64_t *squares, const uint64_t *__restrict__ alphas) {
    Fe3 pending[WHIR_MAX_FOLD];
    const uint32_t leaf_half = 1u << (k - 1);
    Fe3 v = ext3::zero();
    for (uint32_t m = 0; m < leaf_half; ++m) {
        // Level 0: the pair `(t0, t0 + 2^(k−1))` of the coset.
        uint32_t t0 = whir_reverse_low_bits(m, k - 1);
        uint64_t lo = j + (uint64_t)t0 * n_out;
        uint64_t hi = j + (uint64_t)(t0 + leaf_half) * n_out;
        uint64_t x_pow = whir_pow_from_squares(squares, lo);
        Fe3 alpha0 = ext3::make(alphas[0], alphas[1], alphas[2]);
        if (base) {
            v = whir_fold_pair_base(in[lo], in[hi], two_inv, x_pow, alpha0);
        } else {
            v = whir_fold_pair_ext3(ext3::make(in[lo * 3], in[lo * 3 + 1], in[lo * 3 + 2]),
                                    ext3::make(in[hi * 3], in[hi * 3 + 1], in[hi * 3 + 2]),
                                    two_inv, x_pow, alpha0);
        }
        // Merge up while `v` is the right half of a pair whose left is pending:
        // at level `L` the left half is level-(L−1) node `2·(m >> L)` in walk
        // order, whose index in that level's array is its bit reversal.
        uint32_t level = 1;
        uint32_t counter = m;
        while (level < k && (counter & 1u)) {
            uint32_t t_left = whir_reverse_low_bits(2u * (m >> level), k - level);
            uint64_t pos = j + (uint64_t)t_left * n_out;
            uint64_t xl = whir_pow_from_squares(squares + level * WHIR_POW_BITS, pos);
            Fe3 alpha = ext3::make(alphas[3 * level], alphas[3 * level + 1], alphas[3 * level + 2]);
            v = whir_fold_pair_ext3(pending[level], v, two_inv, xl, alpha);
            counter >>= 1;
            ++level;
        }
        if (level < k) {
            pending[level] = v;
        }
    }
    return v;
}

// The squares are the same for every thread, so a block builds them once in
// shared memory — one thread per level — before any thread folds.
__device__ __forceinline__ void whir_fold_k_kernel(const uint64_t *__restrict__ in, bool base,
                                                   uint64_t n_out, uint32_t k, uint64_t two_inv,
                                                   const uint64_t *__restrict__ g_invs,
                                                   const uint64_t *__restrict__ alphas,
                                                   uint64_t *__restrict__ out) {
    __shared__ uint64_t squares[WHIR_MAX_FOLD * WHIR_POW_BITS];
    if (threadIdx.x < k) {
        whir_fold_squares(squares, threadIdx.x, g_invs[threadIdx.x]);
    }
    __syncthreads();
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    Fe3 v = whir_fold_coset(in, base, j, n_out, k, two_inv, squares, alphas);
    out[j * 3] = v.a;
    out[j * 3 + 1] = v.b;
    out[j * 3 + 2] = v.c;
}

extern "C" __global__ void whir_fold_k_base_ext3(const uint64_t *__restrict__ in, uint64_t n_out,
                                                 uint32_t k, uint64_t two_inv,
                                                 const uint64_t *__restrict__ g_invs,
                                                 const uint64_t *__restrict__ alphas,
                                                 uint64_t *__restrict__ out) {
    whir_fold_k_kernel(in, true, n_out, k, two_inv, g_invs, alphas, out);
}

extern "C" __global__ void whir_fold_k_ext3(const uint64_t *__restrict__ in, uint64_t n_out,
                                            uint32_t k, uint64_t two_inv,
                                            const uint64_t *__restrict__ g_invs,
                                            const uint64_t *__restrict__ alphas,
                                            uint64_t *__restrict__ out) {
    whir_fold_k_kernel(in, false, n_out, k, two_inv, g_invs, alphas, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// A WHIR CHAIN'S FIRST ROUNDS OVER ITS FACTORS IN SHARE FORM.
//
// The opening's factors are the weight `w` and the message `f`, each `2^n`
// extension values once materialised — 1.5× the committed base codeword
// together — and the first group of rounds reads them at full width before
// binding anything. They need not exist at that width:
//
//   f(x) = the column holding `x`, at its row;
//   w(x) = scale_c · eq(z_c, row)   for the column `c` holding `x`, else 0,
//
// where a column is a SHARE: a stack offset, its data, its point and its scale.
// Rounds bind the index's top bits first (`Mle::fix_first_variable` pairs `j`
// with `j + half`), so after `s − 1` of them the factors at `(bit s, rest)` are
// `Σ_p eq(α_{<s}, p) · factor(p ‖ bit s ‖ rest)` over the `2^(s−1)` bound
// prefixes `p` — computed here from the shares, per round, and summed into the
// round's evaluations. The factors are materialised only once the group's
// variables are bound (`whir_lean_materialize`), at `2^(n − bound)`.
//
// ★ EQUAL, not raw-identical: the evaluations and the bound tables are the
// values the materialised path computes, by a different sequence of the same
// field operations. Their raw limbs agree whenever both are canonical — every
// value but one in ~2^32 — which is the parity the device already has with the
// host path. The transcript absorbs canonical bytes either way.
//
// A share row is nine u64: stack offset, data offset, `lo_bits`, the offsets of
// its `eq` tables' high and low halves in `eqbuf` (in extension elements), and
// its scale (three limbs), then one spare. `eq(z, row) = hi[row >> lo_bits] ·
// lo[row & (2^lo_bits − 1)]`: the two halves of the point's table, each a few
// kilobytes, instead of the table.
//
// An OVERLAY is one more row past the shares, at stack offset 0 over all `n`
// variables: `w(x) += scale_o · eq(p, x)` at EVERY position, the gaps between
// columns included — a commit-time out-of-domain claim batched into the
// chain's first claim (I-WOOD). The shares tile disjoint ranges and the
// overlay covers all of them, so it is added on top rather than given a slot
// in the column map. `WHIR_LEAN_NO_OVERLAY` when there is none.
// ─────────────────────────────────────────────────────────────────────────────

#define WHIR_LEAN_NONE 0xFFFFu
#define WHIR_LEAN_NO_OVERLAY 0xFFFFFFFFu
#define WHIR_LEAN_SHARE_WORDS 9
#define WHIR_LEAN_MAX_T 4

// The share holding each stack position, by binary search over the shares'
// offsets (sorted, disjoint): `NONE` for a position no column covers.
extern "C" __global__ void whir_lean_colmap(uint16_t *__restrict__ colmap, uint64_t len,
                                            const uint64_t *__restrict__ starts,
                                            const uint64_t *__restrict__ ends, uint32_t count) {
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < len; x += stride) {
        uint32_t lo = 0, hi = count;
        while (lo < hi) {
            uint32_t mid = (lo + hi) / 2;
            if (starts[mid] <= x) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        uint16_t id = WHIR_LEAN_NONE;
        if (lo > 0 && x < ends[lo - 1]) {
            id = (uint16_t)(lo - 1);
        }
        colmap[x] = id;
    }
}

// `scale · eq(z, row)` for one share row, `row` counted from its offset.
__device__ __forceinline__ Fe3 whir_lean_share_weight(const uint64_t *__restrict__ share,
                                                      const uint64_t *__restrict__ eqbuf,
                                                      uint64_t row) {
    uint64_t lo_bits = share[2];
    uint64_t hi_at = (share[3] + (row >> lo_bits)) * 3;
    uint64_t lo_at = (share[4] + (row & ((1ull << lo_bits) - 1))) * 3;
    Fe3 eq = ext3::mul(ext3::make(eqbuf[hi_at], eqbuf[hi_at + 1], eqbuf[hi_at + 2]),
                       ext3::make(eqbuf[lo_at], eqbuf[lo_at + 1], eqbuf[lo_at + 2]));
    return ext3::mul(ext3::make(share[5], share[6], share[7]), eq);
}

// `(w(x), f(x))` from the shares, plus the overlay row's weight when
// `overlay` names one.
__device__ __forceinline__ void whir_lean_value(const uint16_t *__restrict__ colmap,
                                                const uint64_t *__restrict__ shares,
                                                const uint64_t *__restrict__ eqbuf,
                                                const uint64_t *__restrict__ data, uint64_t x,
                                                uint32_t overlay, Fe3 &w, uint64_t &f) {
    uint16_t id = colmap[x];
    if (id == WHIR_LEAN_NONE) {
        w = ext3::zero();
        f = 0;
    } else {
        const uint64_t *share = shares + (uint64_t)id * WHIR_LEAN_SHARE_WORDS;
        uint64_t row = x - share[0];
        f = data[share[1] + row];
        w = whir_lean_share_weight(share, eqbuf, row);
    }
    if (overlay != WHIR_LEAN_NO_OVERLAY) {
        // Offset 0 over the whole stack: the row is the position.
        w = ext3::add(
            w, whir_lean_share_weight(shares + (uint64_t)overlay * WHIR_LEAN_SHARE_WORDS, eqbuf, x));
    }
}

// One rest-position's share of round `s` (1-based) over `n` variables: the
// factors at `(p ‖ b ‖ rest)` for the `2^(s−1)` bound prefixes `p` (weights
// `eqc[p]`) and `b ∈ {0, 1}`, evaluated at each node `t` as `lo + t·(hi − lo)`
// and multiplied — the product rule the opening's program is.
__device__ void whir_lean_round_at(const uint16_t *__restrict__ colmap,
                                   const uint64_t *__restrict__ shares,
                                   const uint64_t *__restrict__ eqbuf,
                                   const uint64_t *__restrict__ data, uint32_t n, uint32_t s,
                                   const uint64_t *__restrict__ eqc,
                                   const uint64_t *__restrict__ d_t, uint32_t num_t,
                                   uint64_t rest, uint32_t overlay, Fe3 *acc) {
    uint32_t below = n - s;  // bits under the round's variable
    uint64_t half = 1ull << below;
    Fe3 w_lo = ext3::zero(), w_hi = ext3::zero(), f_lo = ext3::zero(), f_hi = ext3::zero();
    uint64_t prefixes = 1ull << (s - 1);
    for (uint64_t p = 0; p < prefixes; ++p) {
        Fe3 c = ext3::make(eqc[p * 3], eqc[p * 3 + 1], eqc[p * 3 + 2]);
        uint64_t x_lo = (p << (below + 1)) | rest;
        Fe3 w;
        uint64_t f;
        whir_lean_value(colmap, shares, eqbuf, data, x_lo, overlay, w, f);
        w_lo = ext3::add(w_lo, ext3::mul(c, w));
        f_lo = ext3::add(f_lo, ext3::mul_base(c, f));
        whir_lean_value(colmap, shares, eqbuf, data, x_lo | half, overlay, w, f);
        w_hi = ext3::add(w_hi, ext3::mul(c, w));
        f_hi = ext3::add(f_hi, ext3::mul_base(c, f));
    }
    Fe3 w_d = ext3::sub(w_hi, w_lo), f_d = ext3::sub(f_hi, f_lo);
    for (uint32_t ti = 0; ti < num_t; ++ti) {
        Fe3 t = ext3::make(d_t[ti * 3], d_t[ti * 3 + 1], d_t[ti * 3 + 2]);
        Fe3 w_t = ext3::add(w_lo, ext3::mul(t, w_d));
        Fe3 f_t = ext3::add(f_lo, ext3::mul(t, f_d));
        acc[ti] = ext3::add(acc[ti], ext3::mul(w_t, f_t));
    }
}

// Round `s`'s evaluations at the `num_t` nodes, one partial per (node, block) in
// `sum_partials_ext3`'s layout. Grid-stride over the `2^(n−s)` rest positions.
extern "C" __global__ void whir_lean_round(const uint16_t *__restrict__ colmap,
                                           const uint64_t *__restrict__ shares,
                                           const uint64_t *__restrict__ eqbuf,
                                           const uint64_t *__restrict__ data, uint32_t n,
                                           uint32_t s, const uint64_t *__restrict__ eqc,
                                           const uint64_t *__restrict__ d_t, uint32_t num_t,
                                           uint32_t overlay, uint64_t *__restrict__ d_partials) {
    Fe3 acc[WHIR_LEAN_MAX_T];
    for (uint32_t ti = 0; ti < num_t; ++ti) acc[ti] = ext3::zero();
    uint64_t rests = 1ull << (n - s);
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t rest = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; rest < rests;
         rest += stride) {
        whir_lean_round_at(colmap, shares, eqbuf, data, n, s, eqc, d_t, num_t, rest, overlay,
                           acc);
    }
    extern __shared__ uint64_t shared[];
    for (uint32_t ti = 0; ti < num_t; ++ti) {
        shared[threadIdx.x * 3 + 0] = acc[ti].a;
        shared[threadIdx.x * 3 + 1] = acc[ti].b;
        shared[threadIdx.x * 3 + 2] = acc[ti].c;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
            if (threadIdx.x < width) {
                Fe3 x = ext3::make(shared[threadIdx.x * 3], shared[threadIdx.x * 3 + 1],
                                   shared[threadIdx.x * 3 + 2]);
                Fe3 y = ext3::make(shared[(threadIdx.x + width) * 3],
                                   shared[(threadIdx.x + width) * 3 + 1],
                                   shared[(threadIdx.x + width) * 3 + 2]);
                Fe3 sum = ext3::add(x, y);
                shared[threadIdx.x * 3 + 0] = sum.a;
                shared[threadIdx.x * 3 + 1] = sum.b;
                shared[threadIdx.x * 3 + 2] = sum.c;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            uint64_t at = ((uint64_t)ti * gridDim.x + blockIdx.x) * 3;
            d_partials[at + 0] = shared[0];
            d_partials[at + 1] = shared[1];
            d_partials[at + 2] = shared[2];
        }
        __syncthreads();
    }
}

// The factors bound at the `bound` challenges so far, materialised:
// `out[y] = Σ_b eqfull[b] · factor(b ‖ y)` over the `2^bound` prefixes, for the
// weight and the (lifted) message, `2^(n − bound)` extension values each.
__device__ void whir_lean_materialize_at(const uint16_t *__restrict__ colmap,
                                         const uint64_t *__restrict__ shares,
                                         const uint64_t *__restrict__ eqbuf,
                                         const uint64_t *__restrict__ data, uint32_t n,
                                         uint32_t bound, const uint64_t *__restrict__ eqfull,
                                         uint64_t y, uint32_t overlay, Fe3 &w_out, Fe3 &f_out) {
    uint32_t below = n - bound;
    Fe3 w_acc = ext3::zero(), f_acc = ext3::zero();
    uint64_t prefixes = 1ull << bound;
    for (uint64_t p = 0; p < prefixes; ++p) {
        Fe3 c = ext3::make(eqfull[p * 3], eqfull[p * 3 + 1], eqfull[p * 3 + 2]);
        Fe3 w;
        uint64_t f;
        whir_lean_value(colmap, shares, eqbuf, data, (p << below) | y, overlay, w, f);
        w_acc = ext3::add(w_acc, ext3::mul(c, w));
        f_acc = ext3::add(f_acc, ext3::mul_base(c, f));
    }
    w_out = w_acc;
    f_out = f_acc;
}

extern "C" __global__ void whir_lean_materialize(const uint16_t *__restrict__ colmap,
                                                 const uint64_t *__restrict__ shares,
                                                 const uint64_t *__restrict__ eqbuf,
                                                 const uint64_t *__restrict__ data, uint32_t n,
                                                 uint32_t bound,
                                                 const uint64_t *__restrict__ eqfull,
                                                 uint32_t overlay,
                                                 uint64_t *__restrict__ out_w,
                                                 uint64_t *__restrict__ out_f) {
    uint64_t len = 1ull << (n - bound);
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t y = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; y < len; y += stride) {
        Fe3 w, f;
        whir_lean_materialize_at(colmap, shares, eqbuf, data, n, bound, eqfull, y, overlay, w, f);
        out_w[y * 3] = w.a;
        out_w[y * 3 + 1] = w.b;
        out_w[y * 3 + 2] = w.c;
        out_f[y * 3] = f.a;
        out_f[y * 3 + 1] = f.b;
        out_f[y * 3 + 2] = f.c;
    }
}
