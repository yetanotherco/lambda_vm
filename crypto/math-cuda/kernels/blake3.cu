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
// THE CHAINING CONSTRUCTION is `Blake3Chain`, specified in PA-PLAN §1.7 and
// implemented on host at `crypto/crypto/src/hash/blake3/chain.rs`: standard
// BLAKE3 restricted to a single chunk that never ends. `t = 0` on every block,
// CHUNK_START on the first, CHUNK_END|ROOT and the true byte count as
// `block_len` on the last, digest = the low 8 output words little-endian. The
// device `Blake3Chain` below is a transcription of that host type, and every
// leaf kernel here streams its message through one.
//
// ⚠ The construction is a DRAFT pending ratification of forks F1-F3
// (PA-PLAN §1.7.3): `t = 0` throughout rather than a block counter, the
// three-flag schedule rather than one constant, and no leaf/parent domain
// separation. It is implemented as the working default by standing decision. If
// a fork is ratified the other way, the change lands in `Blake3Chain::finalize`
// and `compress_pending` — the leaf kernels themselves do not move.

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

// The three BLAKE3 domain flags this construction uses. Mirrors of `CHUNK_START`
// / `CHUNK_END` / `ROOT` in `crypto/crypto/src/hash/blake3/chain.rs:53-58`.
// `Blake3Chain` sets CHUNK_START on the first block only and CHUNK_END|ROOT on
// the last only; interior blocks carry no flags at all.
#define BLAKE3_FLAG_CHUNK_START 1u
#define BLAKE3_FLAG_CHUNK_END 2u
#define BLAKE3_FLAG_ROOT 8u

// CHUNK_START | CHUNK_END | ROOT: the flags of a BLAKE3 hash whose whole message
// is one block of one chunk. At 7 rounds a compression under these flags with
// `h = IV` and `t = 0` IS `blake3::hash(message)`, which is what makes the crate
// an anchor for the framing and not just for the round function. Same framing
// the live LFM socket uses (`blake3_socket.rs:258` `FLAGS_LFMC = 0x0B`).
#define BLAKE3_FLAGS_ONE_BLOCK \
    (BLAKE3_FLAG_CHUNK_START | BLAKE3_FLAG_CHUNK_END | BLAKE3_FLAG_ROOT)

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
// `Blake3Chain` — the byte hash every leaf kernel commits with.
//
// Transcription of the host `Blake3Chain` (`crypto/crypto/src/hash/blake3/
// chain.rs:98`), and the two are checked against each other by the leaf parity
// tests at the build's round count. The construction is PA-PLAN §1.7:
//
//     n      = max(1, ceil(|M| / 64))          blocks; the empty message is ONE
//     L      = |M| - 64*(n-1)                  0 when |M| = 0, else 1..=64
//     F_i    = (CHUNK_START if i = 0) | (CHUNK_END|ROOT if i = n-1)
//     cv_0   = IV
//     cv_i+1 = compress(cv_i, m_i, 0, 64, F_i)[0..8]     for i < n-1
//     digest = compress(cv_n-1, m_n-1, 0, L, F_n-1)[0..8]  little-endian
//
// ★ THE ONE SUBTLETY, and the reason this is a state machine rather than a
// loop: a FULL block is *held*, not compressed. The last block's flags and
// `block_len` differ from every other block's, and whether a block is the last
// is not known until the message ends — so a block is only folded into the
// chaining value once a further word proves it was not the last. `push_word`
// therefore compresses on the *next* push, never on filling. This mirrors the
// host `update`, which tests `block_len == BLOCK_LEN` at the top of the loop
// body and so only compresses when there is more input (`chain.rs:186-195`).
//
// Compressing eagerly on fill is the bug this shape exists to prevent: it would
// hash a 64-byte message as two blocks (one flagged CHUNK_START, one empty
// final) instead of one, breaking P2 — the property that a 64-byte message is
// exactly a Merkle parent — and with it the `StarkHash` two-element invariant.
//
// Word granularity, not byte: every message these kernels hash is a whole
// number of 8-byte field elements, so `block_len` is always a multiple of 4 and
// a partial word can never occur. `Blake3Block` is reused for the pending block
// so that the framing (packing, boundaries, zero-padding, the byte count) has
// exactly one implementation shared with `blake3_blocks_of_felts_probe`.
// ---------------------------------------------------------------------------
struct Blake3Chain {
    uint32_t cv[8];
    Blake3Block block;
    // Whether any block has been compressed — i.e. whether the pending block
    // still carries CHUNK_START. Host counterpart: `started` (`chain.rs:109`).
    bool started;

    __device__ __forceinline__ void init() {
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = BLAKE3_IV[i];
        block.init();
        started = false;
    }

    // The pending block's flags. CHUNK_START while nothing has been compressed
    // yet; CHUNK_END|ROOT when this is the message's last block. Mirror of the
    // host `flags(is_final)` (`chain.rs:160`).
    __device__ __forceinline__ uint32_t flags(bool is_final) const {
        uint32_t start = started ? 0u : BLAKE3_FLAG_CHUNK_START;
        uint32_t end = is_final ? (BLAKE3_FLAG_CHUNK_END | BLAKE3_FLAG_ROOT) : 0u;
        return start | end;
    }

    // Fold the pending block — known NOT to be the last — into the chaining
    // value, and clear it so the next block starts zero-padded.
    __device__ __forceinline__ void compress_pending() {
        uint32_t out[16];
        blake3_compress<BLAKE3_ROUNDS>(cv, block.m, 0, 64u, flags(false), out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = out[i];
        block.reset();
        started = true;
    }

    // Absorb one message word. The full-block test comes FIRST: reaching here
    // with a full block is what proves that block was not the last.
    __device__ __forceinline__ void push_word(uint32_t w) {
        if (block.nwords == 16) compress_pending();
        block.push_word(w);
    }

    // Absorb one Goldilocks field element as its two message words — the
    // canonical big-endian element bytes read back as little-endian words.
    __device__ __forceinline__ void push_felt(uint64_t raw) {
        uint32_t w0, w1;
        blake3_words_of_felt(raw, w0, w1);
        push_word(w0);
        push_word(w1);
    }

    // The 32-byte digest: one final compression over the pending block with the
    // true byte count as `block_len` and CHUNK_END|ROOT set. The empty message
    // takes this path with an all-zero block and `block_len = 0`, which is ONE
    // compression, not zero.
    //
    // `dst` is 32-byte aligned at every call site (node buffers come from
    // cuMemAlloc, 256-byte aligned, and every leaf sits at a multiple of 32), so
    // the u32 store is safe. A digest's 32 bytes ARE its 8 output words
    // little-endian and the device is little-endian, so this is a plain copy
    // with no byte swapping — contrast the leaf INPUT path, whose field bytes
    // are big-endian.
    __device__ __forceinline__ void finalize(uint8_t *dst) {
        uint32_t out[16];
        blake3_compress<BLAKE3_ROUNDS>(cv, block.m, 0, block.pending_bytes(), flags(true), out);
        uint32_t *w = reinterpret_cast<uint32_t *>(dst);
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = out[i];
    }
};

// ---------------------------------------------------------------------------
// Leaf kernels.
//
// Twins of the seven keccak leaf kernels, one for one, with the sponge replaced
// by a `Blake3Chain` and the lane byte-swap replaced by the two-word field
// serialization. THE READ PATTERN IS IDENTICAL IN EVERY CASE — same bit
// reversal, same column/component order, same row-pair ordering — because the
// leaf byte layout does not move under P-a: `leaves_bit_reversed_grouped`
// (`crypto/stark/src/commitment.rs:55`) serializes each element in canonical
// big-endian and concatenates, and only the hash over those bytes changes.
//
// So the correctness argument for each kernel below is two independent halves:
// the byte stream (identical to the keccak twin's, and checked against the CPU
// leaf helpers by the parity tests) and the hash over it (`Blake3Chain`, checked
// against the host chain by the same tests).
// ---------------------------------------------------------------------------

// Goldilocks BASE-FIELD leaf hashing, one leaf per bit-reversed row.
// Twin of `keccak256_leaves_base_batched` (`keccak.cu:152`).
extern "C" __global__ void blake3_leaves_base_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    // Read columns at the bit-reversed row, write the leaf at `tid` — matching
    // the CPU per-row `commit_bit_reversed(.., 1)`.
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);

    Blake3Chain h;
    h.init();
    for (uint64_t c = 0; c < num_cols; ++c) {
        h.push_felt(columns_base_ptr[c * col_stride + br]);
    }
    h.finalize(hashed_leaves_out + tid * 32);
}

// Goldilocks BASE-FIELD row-pair leaf hashing: leaf `tid` hashes bit-reversed
// rows `2*tid` and `2*tid+1`, each written column-by-column, first row then
// second. `num_leaves = num_rows / 2`.
// Twin of `keccak256_leaves_base_row_pair_batched` (`keccak.cu:196`).
extern "C" __global__ void blake3_leaves_base_row_pair_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;

    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);

    Blake3Chain h;
    h.init();
    for (uint64_t c = 0; c < num_cols; ++c) {
        h.push_felt(columns_base_ptr[c * col_stride + br_0]);
    }
    for (uint64_t c = 0; c < num_cols; ++c) {
        h.push_felt(columns_base_ptr[c * col_stride + br_1]);
    }
    h.finalize(hashed_leaves_out + tid * 32);
}

// Goldilocks EXT3 leaf hashing, one leaf per bit-reversed row. Components live
// in three separate base-field slabs: column `c` component `k` is at
// `columns_base_ptr[(c*3 + k)*col_stride + br]`, and per-element bytes are
// `[comp0, comp1, comp2]` each 8 big-endian bytes (matching
// `FieldElement::<Ext3>::write_bytes_be`).
// Twin of `keccak256_leaves_ext3_batched` (`keccak.cu:237`).
extern "C" __global__ void blake3_leaves_ext3_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,          // number of ext3 columns (NOT slabs)
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);

    Blake3Chain h;
    h.init();
    for (uint64_t c = 0; c < num_cols; ++c) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            h.push_felt(columns_base_ptr[(c * 3 + (uint64_t)k) * col_stride + br]);
        }
    }
    h.finalize(hashed_leaves_out + tid * 32);
}

// R2 composition-polynomial leaf hashing: each leaf hashes `2 * num_parts` ext3
// values from bit-reversed rows `2*tid` and `2*tid+1`, in (row 0: parts) then
// (row 1: parts) order, three base components per value.
// Twin of `keccak_comp_poly_leaves_ext3` (`keccak.cu:277`).
extern "C" __global__ void blake3_comp_poly_leaves_ext3(
    const uint64_t *parts_base_ptr,
    uint64_t col_stride,
    uint64_t num_parts,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;

    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);

    Blake3Chain h;
    h.init();
    for (uint64_t p = 0; p < num_parts; ++p) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            h.push_felt(parts_base_ptr[(p * 3 + (uint64_t)k) * col_stride + br_0]);
        }
    }
    for (uint64_t p = 0; p < num_parts; ++p) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            h.push_felt(parts_base_ptr[(p * 3 + (uint64_t)k) * col_stride + br_1]);
        }
    }
    h.finalize(leaves_out + tid * 32);
}

// FRI layer leaf hashing: each leaf hashes two consecutive ext3 values from an
// interleaved eval vector `[a0,a1,a2,b0,b1,b2,...]` = 48 bytes. No bit reversal
// and no slab layout.
//
// Note 48 bytes is under one block, so a FRI leaf is a SINGLE compression with
// `flags = 0x0B` and `block_len = 48` — the chain's degenerate one-block case,
// same shape as a Merkle parent but at a different length.
// Twin of `keccak_fri_leaves_ext3` (`keccak.cu:326`).
extern "C" __global__ void blake3_fri_leaves_ext3(
    const uint64_t *evals_interleaved,  // 3 * num_evals u64s (ext3 interleaved)
    uint64_t num_leaves,                 // = num_evals / 2
    uint8_t *leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;

    const uint64_t *left = evals_interleaved + 2 * tid * 3;  // 3 u64s
    const uint64_t *right = left + 3;

    Blake3Chain h;
    h.init();
    #pragma unroll
    for (int i = 0; i < 3; ++i) h.push_felt(left[i]);
    #pragma unroll
    for (int i = 0; i < 3; ++i) h.push_felt(right[i]);
    h.finalize(leaves_out + tid * 32);
}

// Row-major ROW-PAIR leaf hashing: the row-major analog of
// `blake3_leaves_base_row_pair_batched`. Leaf `tid` hashes row
// `reverse_index(2*tid)` then row `reverse_index(2*tid+1)`, each `m` lanes read
// contiguously from `data + br * m`. `m` is the row stride in u64s: base trace =
// column count, ext3 trace = 3 * column count (an ext3 element's components are
// consecutive, matching `write_bytes_be`).
// Twin of `keccak256_leaves_base_row_major_row_pair` (`keccak.cu:473`).
extern "C" __global__ void blake3_leaves_base_row_major_row_pair(
    const uint64_t *data,
    uint64_t m,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;

    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);
    const uint64_t *row_0 = data + br_0 * m;
    const uint64_t *row_1 = data + br_1 * m;

    Blake3Chain h;
    h.init();
    for (uint64_t c = 0; c < m; ++c) h.push_felt(row_0[c]);
    for (uint64_t c = 0; c < m; ++c) h.push_felt(row_1[c]);
    h.finalize(hashed_leaves_out + tid * 32);
}

// Column-range variant: each leaf hashes only columns `[col_start, col_end)` of
// the two bit-reversed rows, while `m` stays the full row stride. Byte layout
// equals the CPU `commit_rows_bit_reversed_subset` — used for preprocessed
// tables, whose precomputed and multiplicity column ranges commit to separate
// Merkle trees over the same row-major LDE.
// Twin of `keccak256_leaves_base_row_major_row_pair_range` (`keccak.cu:511`).
extern "C" __global__ void blake3_leaves_base_row_major_row_pair_range(
    const uint64_t *data,
    uint64_t m,
    uint64_t col_start,
    uint64_t col_end,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;

    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);
    const uint64_t *row_0 = data + br_0 * m;
    const uint64_t *row_1 = data + br_1 * m;

    Blake3Chain h;
    h.init();
    for (uint64_t c = col_start; c < col_end; ++c) h.push_felt(row_0[c]);
    for (uint64_t c = col_start; c < col_end; ++c) h.push_felt(row_1[c]);
    h.finalize(hashed_leaves_out + tid * 32);
}

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

// `n_words` message words streamed through the device `Blake3Chain`, digest out.
//
// ★ This is the harness that lets the device be checked against the COMMITTED
// KAT TABLE (`CHAIN_KAT_6ROUND`, `chain.rs:304`) rather than only against the
// host implementation. That distinction is the whole of risk R13: a device port
// checked solely against the Rust it was transcribed from is checked against
// nothing. The KAT digests came from a Python oracle, so asserting the device
// against them closes the loop with an artifact this tree did not produce.
//
// Word-granular, because that is all the device ever hashes: every production
// message is a whole number of 8-byte field elements. The KAT lengths that are
// not multiples of 4 are therefore unreachable from device code by construction,
// and the host tests cover them instead.
//
// Single-threaded on purpose — same shape as a leaf kernel, one thread hashing
// one whole message sequentially.
extern "C" __global__ void blake3_chain_probe(const uint32_t *words, uint64_t n_words,
                                              uint8_t *out32) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    Blake3Chain h;
    h.init();
    for (uint64_t i = 0; i < n_words; ++i) h.push_word(words[i]);
    h.finalize(out32);
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
