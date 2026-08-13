// BLAKE3 compression on device, round-count parameterized.
//
// THE PARITY REFERENCE is the host `blake3_compress_rounds(h, m, t, block_len,
// flags, rounds)` in `prover/src/lfm/blake3.rs:125` — one function whose ONLY
// parameter is the round count. `blake3_compress<ROUNDS>` below is a
// transcription of it and must agree bit-for-bit at both 6 and 7 rounds; at 7
// rounds both are standard BLAKE3, so the `blake3` crate anchors the pair from
// outside this tree. (P-a Stage 1 moves the host reference down into
// `crypto/crypto`; nothing here changes when it does.)
//
// ROUND COUNT is a compile-time knob with the same polarity as the host's
// `BLAKE3_ROUNDS`: 7 (standard BLAKE3) by default, 6 when build.rs passes
// `-DBLAKE3_ROUNDS=6` under math-cuda's `blake3-6round` feature. The two knobs
// are separate crates' features and nothing forces them equal — see
// `blake3_rounds_probe`, which exports this cubin's round count so a caller can
// assert the match rather than discover it as a wrong commitment.

#include <cstdint>

// 7 = standard BLAKE3. Overridden to 6 by build.rs; see the header comment.
#ifndef BLAKE3_ROUNDS
#define BLAKE3_ROUNDS 7
#endif

// The BLAKE3 IV (= SHA-256's initial state). Mirror of `BLAKE3_IV`
// (`blake3.rs:46`). `IV[0..4]` also seeds `v[8..12]` of the working state.
__device__ __constant__ uint32_t BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    // Every call site passes 16, 12, 8 or 7, so the 32-n shift is never a
    // shift-by-32. Kept as an explicit expression rather than __funnelshift_r
    // so the transcription against the host `rotate_right` is readable.
    return (x >> n) | (x << (32 - n));
}

// The BLAKE3 quarter-round G (spec §2.1). Mirror of `blake3_g`
// (`blake3.rs:89`); uint32_t arithmetic wraps, matching `wrapping_add`.
__device__ __forceinline__ void blake3_g(uint32_t *v, int a, int b, int c, int d,
                                         uint32_t mx, uint32_t my) {
    v[a] = v[a] + v[b] + mx;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + my;
    v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7);
}

// The message-schedule permutation `m'[i] = m[PERM[i]]`, written out. The
// indices are `BLAKE3_MSG_PERMUTATION` (`blake3.rs:52`):
//   [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
// Spelled as literals rather than read from a constant array so it stays a
// register shuffle after unrolling; the parity tests are what check the
// transcription.
__device__ __forceinline__ void blake3_permute(uint32_t *m) {
    uint32_t p[16] = {m[2], m[6],  m[3],  m[10], m[7], m[0],  m[4],  m[13],
                      m[1], m[11], m[12], m[5],  m[9], m[14], m[15], m[8]};
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = p[i];
}

// The BLAKE3 compression function `f` at `ROUNDS` rounds, full 16-word output.
//
// State init: `v[0..8] = h`, `v[8..12] = IV[0..4]`, `v[12] = t as u32`,
// `v[13] = (t >> 32) as u32`, `v[14] = block_len`, `v[15] = flags`. Each round
// is 8 G-calls (4 columns then 4 diagonals); the schedule is permuted between
// rounds only (`r < ROUNDS - 1` — the trailing permute is never consumed).
// Feed-forward: `out[i] = v[i] ^ v[i+8]`, `out[i+8] = v[i+8] ^ h[i]`; the
// truncated chaining value is `out[0..8]`.
//
// Callers that need only the chaining value still get the full 16 words: the
// second half is 8 XORs the compiler drops when they are unused, and one
// function is one place to be wrong.
template <int ROUNDS>
__device__ __forceinline__ void blake3_compress(const uint32_t *h, const uint32_t *m_in,
                                                uint64_t t, uint32_t block_len, uint32_t flags,
                                                uint32_t *out) {
    uint32_t v[16] = {
        h[0],         h[1],         h[2],         h[3],
        h[4],         h[5],         h[6],         h[7],
        BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3],
        (uint32_t)t,  (uint32_t)(t >> 32), block_len, flags,
    };

    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = m_in[i];

    #pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        // Mix the columns.
        blake3_g(v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(v, 3, 7, 11, 15, m[6], m[7]);
        // Mix the diagonals.
        blake3_g(v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(v, 3, 4, 9, 14, m[14], m[15]);
        if (r < ROUNDS - 1) blake3_permute(m);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ h[i];
    }
}

// ---------------------------------------------------------------------------
// Parity-harness entry points.
//
// The device compression function is not otherwise reachable from host code, so
// there would be nothing to check it against the host reference with. These are
// that oracle — the same role `build_fri_layer_tree_from_evals_ext3` plays for
// the keccak tree. Not on any production path.
// ---------------------------------------------------------------------------

// `n` independent compressions, full 16-word outputs. One thread per vector.
template <int ROUNDS>
__device__ __forceinline__ void compress_probe_body(const uint32_t *h, const uint32_t *m,
                                                    const uint64_t *t, const uint32_t *block_len,
                                                    const uint32_t *flags, uint64_t n,
                                                    uint32_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    blake3_compress<ROUNDS>(h + tid * 8, m + tid * 16, t[tid], block_len[tid], flags[tid],
                            out + tid * 16);
}

extern "C" __global__ void blake3_compress_probe_6r(const uint32_t *h, const uint32_t *m,
                                                    const uint64_t *t, const uint32_t *block_len,
                                                    const uint32_t *flags, uint64_t n,
                                                    uint32_t *out) {
    compress_probe_body<6>(h, m, t, block_len, flags, n, out);
}

extern "C" __global__ void blake3_compress_probe_7r(const uint32_t *h, const uint32_t *m,
                                                    const uint64_t *t, const uint32_t *block_len,
                                                    const uint32_t *flags, uint64_t n,
                                                    uint32_t *out) {
    compress_probe_body<7>(h, m, t, block_len, flags, n, out);
}

// The same probe at the round count this cubin's PRODUCTION kernels are built
// for. Not redundant with the two above: it is the only way to observe from host
// code which of them the production kernels use.
extern "C" __global__ void blake3_compress_probe_default(const uint32_t *h, const uint32_t *m,
                                                         const uint64_t *t,
                                                         const uint32_t *block_len,
                                                         const uint32_t *flags, uint64_t n,
                                                         uint32_t *out) {
    compress_probe_body<BLAKE3_ROUNDS>(h, m, t, block_len, flags, n, out);
}

// This cubin's compiled-in round count, so a caller can assert it against the
// host's `BLAKE3_ROUNDS` instead of discovering a mismatch as a wrong root.
extern "C" __global__ void blake3_rounds_probe(uint32_t *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (uint32_t)BLAKE3_ROUNDS;
}
