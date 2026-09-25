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
