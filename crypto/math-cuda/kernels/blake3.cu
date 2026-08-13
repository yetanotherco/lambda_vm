// BLAKE3 compression on device, round-count parameterized, plus the Merkle
// parent compressors and the field-element byte serialization the leaf kernels
// share with the CPU commit path.
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
//
// WHAT IS NOT HERE YET: the multi-block leaf kernels. A leaf spans many 64-byte
// blocks, so hashing one needs a *chaining construction* — bare cv-chain vs
// standard BLAKE3 chunk tree — and that decision is open (PA-PLAN §1.6). The
// compression function, the byte serialization and the block framing are
// identical either way and are all here; `Blake3Block` exposes the sink so the
// chaining loop drops in without touching any of it.

#include <cstdint>
#include "goldilocks.cuh"

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

// CHUNK_START | CHUNK_END | ROOT: the flags of a BLAKE3 hash whose whole message
// is one block of one chunk. At 7 rounds a compression under these flags with
// `h = IV` and `t = 0` IS `blake3::hash(message)`, which is what makes the crate
// an anchor for the framing and not just for the round function. Same framing
// the live LFM socket uses (`blake3_socket.rs:258` `FLAGS_LFMC = 0x0B`).
#define BLAKE3_FLAGS_ONE_BLOCK 0x0Bu

// A Merkle parent's message is two 32-byte child digests = exactly 64 bytes.
#define BLAKE3_PARENT_BLOCK_LEN 64u

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    // Every call site passes 16, 12, 8 or 7, so the 32-n shift is never a
    // shift-by-32. Kept as an explicit expression rather than __funnelshift_r
    // so the transcription against the host `rotate_right` is readable.
    return (x >> n) | (x << (32 - n));
}

// Reverse the four bytes of a 32-bit word. nvcc lowers this to a single PRMT.
__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0x0000FF00u) | ((x << 8) & 0x00FF0000u) | (x << 24);
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
// Byte serialization — shared with the leaf kernels and with the CPU commit.
//
// The leaf byte encoding does NOT move under P-a: `leaves_bit_reversed_grouped`
// (`crypto/stark/src/commitment.rs:55`) serializes each field element in
// canonical BIG-endian form and concatenates. BLAKE3 reads a 64-byte block as
// 16 LITTLE-endian u32 words, so one 8-byte element is two words: the
// byte-reverse of its high half, then of its low half. That transposition is
// the whole of the serialization difference from keccak, which absorbs the same
// bytes as one byte-swapped u64 lane.
// ---------------------------------------------------------------------------

// The two BLAKE3 message words covered by one Goldilocks element's canonical
// big-endian bytes. `raw` may be non-canonical; canonicalising here matches
// `canonical_u64().to_be_bytes()` on host.
__device__ __forceinline__ void blake3_words_of_felt(uint64_t raw, uint32_t &w0, uint32_t &w1) {
    uint64_t canon = goldilocks::canonical(raw);
    w0 = bswap32((uint32_t)(canon >> 32));
    w1 = bswap32((uint32_t)canon);
}

// A 64-byte BLAKE3 message block under construction.
//
// The SINK is deliberately the caller's: a leaf kernel compresses each full
// block into a chaining value, and which chaining construction that is (bare
// cv-chain vs standard chunk tree, PA-PLAN §1.6) is still open. Everything this
// struct does — word packing, block boundaries, zero-padding the tail, the byte
// count the final `block_len` comes from — is the same under either.
//
// Usage: `push_word` returns true when the block just filled, at which point the
// caller consumes `m` and calls `reset()`; pushing into a full block is the one
// way to misuse it. Field elements go in two words at a time (via
// `blake3_words_of_felt`) and straddle a block boundary whenever the element
// count is not a multiple of 8 — ext3 elements, at three felts, straddle
// routinely — which is why this works at word granularity and not element
// granularity.
struct Blake3Block {
    uint32_t m[16];
    uint32_t nwords;  // words filled in the current block, 0..15 between pushes

    __device__ __forceinline__ void init() {
        nwords = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = 0;
    }

    __device__ __forceinline__ bool push_word(uint32_t w) {
        m[nwords++] = w;
        return nwords == 16;
    }

    __device__ __forceinline__ void reset() { init(); }

    // Bytes occupied in the pending (partial) block — the `block_len` a final
    // compression over it takes. Zero exactly when no partial block is pending.
    __device__ __forceinline__ uint32_t pending_bytes() const { return nwords * 4u; }
};

// ---------------------------------------------------------------------------
// Merkle parent / level compressors.
//
// A parent is ONE compression over the 64 bytes of its two child digests:
// `h = IV`, `t = 0`, `block_len = 64`, `flags = CHUNK_START|CHUNK_END|ROOT`,
// digest = `out[0..8]` little-endian. That is `hash_bytes(left ‖ right)`, which
// is what `hash_new_parent` is on host for every existing backend
// (`hash_new_parent_bytes`, `field_element_vector.rs:74`) — so a parent needs no
// chaining and is construction-independent: with a single-block message the
// chunk tree and a bare cv-chain agree bit-for-bit.
//
// The u32 casts are byte-order-free in both directions and that is not an
// accident: a digest's 32 bytes ARE its 8 output words little-endian, and BLAKE3
// reads message bytes as little-endian words, so on a little-endian device
// (every NVIDIA GPU) reading a child digest as `uint32_t[8]` yields exactly the
// message words, and storing `out[0..8]` as u32 yields exactly the digest bytes.
// No byte swapping anywhere on this path — contrast the leaf path above, whose
// input is big-endian field bytes.
//
// Node buffer layout mirrors `keccak.cu`'s and the CPU
// `crypto/crypto/src/merkle_tree/merkle.rs`: children at
// `nodes[parent_begin + n_pairs .. parent_begin + 3*n_pairs]`, parents at
// `nodes[parent_begin .. parent_begin + n_pairs]`, 32 bytes per node.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void blake3_hash_merkle_parent(uint8_t *nodes, uint64_t parent_begin,
                                                          uint64_t n_pairs, uint64_t tid) {
    // `nodes` comes from cuMemAlloc (256-byte aligned) and every 32-byte node
    // sits at a 32-byte-aligned offset, so the u32 casts are safe.
    const uint32_t *left = reinterpret_cast<const uint32_t *>(
        nodes + (parent_begin + n_pairs + 2 * tid) * 32);
    const uint32_t *right = reinterpret_cast<const uint32_t *>(
        nodes + (parent_begin + n_pairs + 2 * tid + 1) * 32);

    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        m[i] = left[i];
        m[i + 8] = right[i];
    }

    uint32_t out[16];
    blake3_compress<BLAKE3_ROUNDS>(BLAKE3_IV, m, 0, BLAKE3_PARENT_BLOCK_LEN,
                                   BLAKE3_FLAGS_ONE_BLOCK, out);

    uint32_t *dst = reinterpret_cast<uint32_t *>(nodes + (parent_begin + tid) * 32);
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = out[i];
}

// One level of the inner Merkle tree: each thread hashes one child pair.
extern "C" __global__ void blake3_merkle_level(uint8_t *nodes,
                                               uint64_t parent_begin,  // in 32-byte nodes
                                               uint64_t n_pairs) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;
    blake3_hash_merkle_parent(nodes, parent_begin, n_pairs, tid);
}

// Build every remaining level (from `level_begin` up to the root) in ONE
// single-block launch: each level's pairs are grid-strided over the block, with
// a __syncthreads() barrier between levels. Replaces log2 launches of
// `blake3_merkle_level` for the small top levels, whose per-level work is
// dwarfed by launch overhead. Twin of `keccak_merkle_tail`.
extern "C" __global__ void blake3_merkle_tail(uint8_t *nodes, uint64_t level_begin) {
    uint64_t lb = level_begin;
    while (lb != 0) {
        uint64_t nb = lb / 2;
        uint64_t n_pairs = lb - nb;
        for (uint64_t tid = threadIdx.x; tid < n_pairs; tid += blockDim.x) {
            blake3_hash_merkle_parent(nodes, nb, n_pairs, tid);
        }
        __syncthreads();
        lb = nb;
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
// code which of them `blake3_merkle_level` actually uses.
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

// The two message words of each of `n` field elements, in order — the
// serialization contract on its own (canonicalisation, big-endian element bytes,
// little-endian word packing), with no hashing over it.
extern "C" __global__ void blake3_serialize_felts_probe(const uint64_t *vals, uint64_t n,
                                                        uint32_t *out_words) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t w0, w1;
    blake3_words_of_felt(vals[tid], w0, w1);
    out_words[tid * 2] = w0;
    out_words[tid * 2 + 1] = w1;
}

// `n` field elements streamed through `Blake3Block`, with the completed blocks
// written out instead of compressed. Single-threaded on purpose: that is the
// shape a leaf kernel has (one thread hashes one whole leaf, sequentially), so
// this exercises the block framing on the code path the chaining loop will use.
// Writes `ceil(2n/16)` blocks of 16 words; the tail block is zero-padded.
extern "C" __global__ void blake3_blocks_of_felts_probe(const uint64_t *vals, uint64_t n,
                                                        uint32_t *out_blocks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    Blake3Block b;
    b.init();
    uint64_t nblocks = 0;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t w0, w1;
        blake3_words_of_felt(vals[i], w0, w1);
        if (b.push_word(w0)) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) out_blocks[nblocks * 16 + k] = b.m[k];
            ++nblocks;
            b.reset();
        }
        if (b.push_word(w1)) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) out_blocks[nblocks * 16 + k] = b.m[k];
            ++nblocks;
            b.reset();
        }
    }
    // Flush the partial tail block, zero-padded (`init` zeroed it, and
    // `pending_bytes` is what a real final compression would pass as block_len).
    if (b.pending_bytes() != 0) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) out_blocks[nblocks * 16 + k] = b.m[k];
    }
}
