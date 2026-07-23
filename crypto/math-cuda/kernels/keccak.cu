// Original Keccak-256 (0x01 padding)
//
// Used by the lambda-vm prover's Merkle commit:
//   leaf = Keccak-256(concat(col_0[br_idx].to_be_bytes(), col_1[br_idx].to_be_bytes(), ...))
// where `br_idx = bit_reverse(row_idx, log_num_rows)` and each element is
// written in BIG-ENDIAN canonical form (per `FieldElement::write_bytes_be`).
//
// Keccak state is 5x5 lanes of u64, interpreted little-endian. Rate = 136 B
// (17 lanes) for 256-bit output, capacity = 64 B (8 lanes).
//
// Since every input byte is u64-aligned (each field element is 8 or 24 bytes),
// we can absorb lane-by-lane instead of byte-by-byte. Canonicalise + byte-swap
// each u64 on read to turn a BE-serialised element into its LE-interpreted
// lane value.

#include <cstdint>
#include "goldilocks.cuh"

__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

// Rotation offsets indexed by lane position x + 5*y. Standard Keccak rho.
__device__ __constant__ uint32_t KECCAK_RHO_OFFSETS[25] = {
    0,  1,  62, 28, 27,  // y=0: x=0..4
    36, 44, 6,  55, 20,  // y=1
    3,  10, 43, 25, 39,  // y=2
    41, 45, 15, 21, 8,   // y=3
    18, 2,  61, 56, 14,  // y=4
};

__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    return (n == 0) ? x : ((x << n) | (x >> (64 - n)));
}

__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    // Reverse byte order: turns a BE-serialised u64 into its LE-read lane.
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x & 0xff00ff00ff00ff00ULL) >> 8);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x & 0xffff0000ffff0000ULL) >> 16);
    return (x << 32) | (x >> 32);
}

__device__ __forceinline__ void keccak_f1600(uint64_t st[25]) {
    uint64_t C[5], D[5], B[25];
    // No outer unroll: fully unrolling the 24 rounds slowed the kernel ~7.5% on RTX 5090.
    for (int r = 0; r < 24; ++r) {
        // Theta
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        }
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                st[x + 5 * y] ^= D[x];
            }
        }

        // Rho + Pi: B[pi(x,y)] = rotl(st[x,y], rho(x,y))
        //   pi: (x', y') = (y, (2x + 3y) mod 5)
        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                int nx = y;
                int ny = (2 * x + 3 * y) % 5;
                B[nx + 5 * ny] = rotl64(st[x + 5 * y], KECCAK_RHO_OFFSETS[x + 5 * y]);
            }
        }

        // Chi
        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                st[x + 5 * y] =
                    B[x + 5 * y] ^ ((~B[((x + 1) % 5) + 5 * y]) & B[((x + 2) % 5) + 5 * y]);
            }
        }

        // Iota
        st[0] ^= KECCAK_RC[r];
    }
}

// Batched Keccak-f[1600] permutation for the KeccakPermute PRECOMPILE (Phase 6): one thread
// permutes one 25-lane state in place. `states[i*25 + j]` is lane j of permutation i. Reuses the
// same `keccak_f1600` the Merkle tree uses (already validated bit-for-bit vs the CPU permutation),
// so a precompile can compute its 25-lane output on device instead of the host `keccak_f1600`.
extern "C" __global__ void keccak_f1600_batch(uint64_t n, uint64_t *states) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    uint64_t st[25];
    uint64_t *s = states + idx * 25;
#pragma unroll
    for (int i = 0; i < 25; ++i)
        st[i] = s[i];
    keccak_f1600(st);
#pragma unroll
    for (int i = 0; i < 25; ++i)
        s[i] = st[i];
}

// ---------------------------------------------------------------------------
// Helper: absorb one 8-byte lane (already byte-swapped from BE serialisation
// into Keccak's LE lane form) into the sponge at `rate_pos` (in bytes).
// Permutes when a full 136-byte block has been absorbed.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void absorb_lane(uint64_t st[25],
                                            uint32_t &rate_pos,
                                            uint64_t lane) {
    st[rate_pos / 8] ^= lane;
    rate_pos += 8;
    if (rate_pos == 136) {
        keccak_f1600(st);
        rate_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// After all data lanes absorbed, apply Keccak (pre-SHA-3) padding: a single
// 0x01 byte at the current position, then bit 0x80 on the last rate byte
// (byte 135 = last byte of lane 16). Then permute and squeeze 32 bytes from
// the first four lanes in LE order.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void finalize_keccak256(uint64_t st[25],
                                                    uint32_t rate_pos,
                                                    uint8_t *out32) {
    // 0x01 at rate_pos
    st[rate_pos / 8] ^= ((uint64_t)0x01) << ((rate_pos & 7) * 8);
    // 0x80 at byte 135 (last byte of lane 16)
    st[16] ^= ((uint64_t)0x80) << 56;
    keccak_f1600(st);

    // Squeeze 32 bytes: 4 lanes, each LE-serialised.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t lane = st[i];
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            out32[i * 8 + b] = (uint8_t)((lane >> (b * 8)) & 0xff);
        }
    }
}

// ---------------------------------------------------------------------------
// Goldilocks BASE-FIELD leaf hashing.
//
// For output row `row_idx` (natural order), the leaf hashes the canonical BE
// byte representation of `columns[c][bit_reverse(row_idx, log_num_rows)]` for
// `c` in `[0, num_cols)`, concatenated in column order. Writes 32 bytes to
// `hashed_leaves_out[row_idx * 32 ..]`.
//
// `columns_base_ptr` points to a `num_cols * col_stride * u64` buffer; column
// `c` is the contiguous slab `[c*col_stride .. c*col_stride + num_rows]`. The
// remaining `col_stride - num_rows` entries (if any) are ignored.
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak256_leaves_base_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    // Bit-reverse the row index so we read columns at `br` but write the hashed
    // leaf at `tid` — matching the CPU per-row `commit_bit_reversed(.., 1)`.
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;

    uint32_t rate_pos = 0;
    for (uint64_t c = 0; c < num_cols; ++c) {
        uint64_t v = columns_base_ptr[c * col_stride + br];
        // Canonicalise to match `canonical_u64().to_be_bytes()` on host.
        uint64_t canon = goldilocks::canonical(v);
        // The on-disk leaf bytes are canon.to_be_bytes(). Keccak reads those
        // as a LE lane, which equals bswap64(canon).
        uint64_t lane = bswap64(canon);
        absorb_lane(st, rate_pos, lane);
    }

    finalize_keccak256(st, rate_pos, hashed_leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// Goldilocks BASE-FIELD row-pair leaf hashing.
//
// Leaf `leaf_idx` hashes TWO consecutive bit-reversed rows
//   br_0 = reverse_index(2*leaf_idx),  br_1 = reverse_index(2*leaf_idx + 1)
// each written column-by-column in canonical BE (same per-row byte layout as
// `keccak256_leaves_base_batched`), in (br_0 row: col 0..K-1) then (br_1 row:
// col 0..K-1) order. `num_leaves = num_rows / 2`; writes 32 bytes to
// `hashed_leaves_out[leaf_idx * 32 ..]`. Matches the CPU
// `keccak_leaves_row_pair_bit_reversed` (rows_per_leaf = 2) — the base-field
// analog of `keccak_comp_poly_leaves_ext3`.
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak256_leaves_base_row_pair_batched(
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

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;

    uint32_t rate_pos = 0;
    // First row (br_0): col 0..K-1.
    for (uint64_t c = 0; c < num_cols; ++c) {
        uint64_t v = columns_base_ptr[c * col_stride + br_0];
        absorb_lane(st, rate_pos, bswap64(goldilocks::canonical(v)));
    }
    // Second row (br_1): col 0..K-1.
    for (uint64_t c = 0; c < num_cols; ++c) {
        uint64_t v = columns_base_ptr[c * col_stride + br_1];
        absorb_lane(st, rate_pos, bswap64(goldilocks::canonical(v)));
    }

    finalize_keccak256(st, rate_pos, hashed_leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// Goldilocks EXT3 leaf hashing (3 base-field components per ext3 element).
//
// Components live in three separate base-field slabs (our de-interleaved
// layout). Column `c` component `k` is at `columns_base_ptr[(c*3 + k)*col_stride
// + br]`. Per-element BE bytes are `[comp0, comp1, comp2]` each 8 BE bytes
// (matches `FieldElement::<Ext3>::write_bytes_be`).
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak256_leaves_ext3_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,          // number of ext3 columns (NOT slabs)
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;

    uint32_t rate_pos = 0;
    for (uint64_t c = 0; c < num_cols; ++c) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            uint64_t v = columns_base_ptr[(c * 3 + (uint64_t)k) * col_stride + br];
            uint64_t canon = goldilocks::canonical(v);
            uint64_t lane = bswap64(canon);
            absorb_lane(st, rate_pos, lane);
        }
    }

    finalize_keccak256(st, rate_pos, hashed_leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// R2 composition-polynomial leaf hashing.
//
// Each leaf hashes `2 * num_parts` ext3 values taken from bit-reversed rows
// `br_0 = reverse_index(2*leaf_idx)` and `br_1 = reverse_index(2*leaf_idx+1)`
// across all `num_parts` parts, in (br_0 row: part 0..K-1) then (br_1 row:
// part 0..K-1) order. Each ext3 value is 3 base components × 8 BE bytes.
//
// Columns arrive in the de-interleaved 3-slab layout: part `p` component
// `k` is at `parts_base_ptr[(p*3 + k) * col_stride + row]`.
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak_comp_poly_leaves_ext3(
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

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;

    uint32_t rate_pos = 0;
    // First row (br_0): part 0..K-1 × 3 components each.
    for (uint64_t p = 0; p < num_parts; ++p) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            uint64_t v = parts_base_ptr[(p * 3 + (uint64_t)k) * col_stride + br_0];
            uint64_t canon = goldilocks::canonical(v);
            absorb_lane(st, rate_pos, bswap64(canon));
        }
    }
    // Second row (br_1).
    for (uint64_t p = 0; p < num_parts; ++p) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            uint64_t v = parts_base_ptr[(p * 3 + (uint64_t)k) * col_stride + br_1];
            uint64_t canon = goldilocks::canonical(v);
            absorb_lane(st, rate_pos, bswap64(canon));
        }
    }

    finalize_keccak256(st, rate_pos, leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// FRI layer leaf hashing.
//
// Each leaf hashes 2 consecutive ext3 values: Keccak256 over
//   evals[2j].to_bytes_be() ++ evals[2j+1].to_bytes_be()
// = 48 BE bytes = 6 u64 BE lanes. No bit reversal, no column slab layout.
// The input is a single interleaved ext3 eval vector `[a0,a1,a2,b0,b1,b2,...]`.
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak_fri_leaves_ext3(
    const uint64_t *evals_interleaved,  // 3 * num_evals u64s (ext3 interleaved)
    uint64_t num_leaves,                 // = num_evals / 2
    uint8_t *leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;
    uint32_t rate_pos = 0;

    const uint64_t *left = evals_interleaved + 2 * tid * 3;  // 3 u64s
    const uint64_t *right = left + 3;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        uint64_t canon = goldilocks::canonical(left[i]);
        absorb_lane(st, rate_pos, bswap64(canon));
    }
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        uint64_t canon = goldilocks::canonical(right[i]);
        absorb_lane(st, rate_pos, bswap64(canon));
    }

    finalize_keccak256(st, rate_pos, leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// Merkle inner-tree pair hash: one level of the inner Merkle tree.
//
// `nodes` is the full Merkle node buffer (length `2*leaves_len - 1`, each
// element 32 bytes). `parent_begin` is the node-index offset of the first
// parent slot in this level. Children live at `parent_begin + n_pairs`.
// The layout mirrors `crypto/crypto/src/merkle_tree/merkle.rs`:
//
//     children: nodes[parent_begin + n_pairs .. parent_begin + 3 * n_pairs]
//     parents:  nodes[parent_begin .. parent_begin + n_pairs]
//
// Each thread hashes one child pair → one parent. Keccak-256 of the
// concatenation of two 32-byte siblings, identical to
// `FieldElementVectorBackend::hash_new_parent` on host.
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak_merkle_level(
    uint8_t *nodes,
    uint64_t parent_begin,     // node index (counted in 32-byte nodes)
    uint64_t n_pairs) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;

    uint32_t rate_pos = 0;
    // `nodes` comes from cuMemAlloc (256-byte aligned); each 32-byte node
    // sits at a 32-byte-aligned offset, so the u64 cast is safe.
    const uint64_t *left = reinterpret_cast<const uint64_t *>(
        nodes + (parent_begin + n_pairs + 2 * tid) * 32);
    #pragma unroll
    for (int i = 0; i < 4; ++i) absorb_lane(st, rate_pos, left[i]);

    const uint64_t *right = reinterpret_cast<const uint64_t *>(
        nodes + (parent_begin + n_pairs + 2 * tid + 1) * 32);
    #pragma unroll
    for (int i = 0; i < 4; ++i) absorb_lane(st, rate_pos, right[i]);

    finalize_keccak256(st, rate_pos, nodes + (parent_begin + tid) * 32);
}

// Gather Merkle authentication paths for a batch of leaf positions, reading the
// resident tree `nodes` (32-byte nodes; layout: inner nodes [0..leaves_len-1],
// root at 0, leaves at [leaves_len-1..]). One thread per query walks leaf->root,
// writing each sibling node into the output. This mirrors the CPU
// `build_merkle_path` exactly (sibling_index / parent_index in
// crypto/crypto/src/merkle_tree/utils.rs):
//   leaf node = pos + leaves_len - 1
//   sibling   = node even ? node-1 : node+1
//   parent    = node even ? (node-1)/2 : node/2
// so `out[(q*depth + level)*32 .. +32]` is the level-th sibling for query q.
extern "C" __global__ void merkle_gather_paths(
    const uint8_t *nodes,
    const uint32_t *positions,   // leaf positions, length num_queries
    uint32_t num_queries,
    uint64_t leaves_len,
    uint32_t depth,              // = log2(leaves_len)
    uint8_t *out) {              // num_queries * depth * 32 bytes
    uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_queries) return;

    uint64_t node = (uint64_t)positions[q] + leaves_len - 1;
    for (uint32_t level = 0; level < depth; ++level) {
        uint64_t sib = (node & 1ull) ? (node + 1ull) : (node - 1ull);
        // 32-byte nodes at 32-byte-aligned offsets (cuMemAlloc 256-aligned),
        // so the u64 copy is safe.
        const uint64_t *src = reinterpret_cast<const uint64_t *>(nodes + sib * 32);
        uint64_t *dst = reinterpret_cast<uint64_t *>(
            out + ((uint64_t)q * depth + level) * 32);
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = src[i];
        node = (node & 1ull) ? (node >> 1) : ((node - 1ull) >> 1);
    }
}

// ---------------------------------------------------------------------------
// Row-major ROW-PAIR leaf hashing.
//
// Row-major analog of `keccak256_leaves_base_row_pair_batched` (which reads a
// column-major slab): each leaf hashes TWO consecutive bit-reversed rows.
// Leaf `tid` hashes row `reverse_index(2*tid)` followed by row
// `reverse_index(2*tid + 1)`, each as `m` canonical big-endian lanes read from
// the contiguous row-major buffer (`data + br * m`). `num_leaves = num_rows/2`;
// writes 32 bytes to `hashed_leaves_out[tid*32 ..]`.
//
// `m` is the row stride in u64s: base trace = num columns; ext3 trace = 3 *
// num columns (an ext3 element's components c0,c1,c2 are consecutive, matching
// the CPU `write_bytes_be`). Byte layout therefore equals the CPU
// `commit_bit_reversed(.., ROWS_PER_LEAF=2)` and the verifier's
// `verify_opening_pair` (queried row ‖ its symmetric counterpart, one leaf).
// ---------------------------------------------------------------------------
extern "C" __global__ void keccak256_leaves_base_row_major_row_pair(
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

    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;

    uint32_t rate_pos = 0;
    // First row (br_0): cols 0..m-1.
    for (uint64_t c = 0; c < m; ++c) {
        absorb_lane(st, rate_pos, bswap64(goldilocks::canonical(row_0[c])));
    }
    // Second row (br_1): cols 0..m-1.
    for (uint64_t c = 0; c < m; ++c) {
        absorb_lane(st, rate_pos, bswap64(goldilocks::canonical(row_1[c])));
    }
    finalize_keccak256(st, rate_pos, hashed_leaves_out + tid * 32);
}
