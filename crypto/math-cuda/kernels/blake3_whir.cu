// The WHIR base proof's BLAKE3 kernels (D-B3HYB S3): coset leaves over a base
// or ext3 codeword, the base leaves in the coset leaf order, and the
// proof-of-work nonce search — all at 7 ROUNDS, standard BLAKE3, whatever
// round count `blake3.cu`'s own cubin is built at.
//
// THE ROUND COUNT IS FIXED HERE, NOT BY A FEATURE. `blake3.cu` takes its round
// count from math-cuda's `blake3-6round` feature (6 in a default build), because
// its kernels commit the STARK pipeline's trees, whose host hash is the
// build's `Blake3Chain`. The WHIR arm's host hash is `Blake3Std`, 7 rounds in
// every build, so this file defines `BLAKE3_ROUNDS` before including
// `blake3.cu` and build.rs passes it no define: the compression, the device
// `Blake3Chain`, the felt serialization and the parent compressors
// (`blake3_merkle_level` / `blake3_merkle_tail`) are `blake3.cu`'s own source,
// instantiated a second time at 7. `b3w_rounds_probe` exports the count so the
// host asserts it when the module loads (`crate::blake3_whir`).
//
// THE BYTES are the host's `FieldElementVectorBackend<_, Blake3Std, 32>`: a
// leaf hashes its fold block's values in coset order, each value as canonical
// big-endian limbs (one for a base value, three for an ext3 value); a parent is
// one compression over its two children (`blake3_hash_merkle_parent`).

#define BLAKE3_ROUNDS 7
#include "blake3.cu"

static_assert(BLAKE3_ROUNDS == 7, "the WHIR BLAKE3 kernels are standard BLAKE3");

// Leaf `j` hashes `codeword[j + t * num_leaves]` for `t` in `[0, block)`: the
// fold block a WHIR query opens. Twin of `keccak256_leaves_base_coset`.
extern "C" __global__ void b3w_leaves_base_coset(const uint64_t *__restrict__ codeword,
                                                 uint64_t num_leaves, uint64_t block,
                                                 uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    Blake3Chain h;
    h.init();
    for (uint64_t t = 0; t < block; ++t) h.push_felt(codeword[tid + t * num_leaves]);
    h.finalize(out + tid * 32);
}

// `b3w_leaves_base_coset` with fold block `tid` written at tree position
// `bitrev(tid)` over the tree's `leaf_bits` levels (`LeafOrder::Coset`), so the
// leaves under an aligned subtree are one coset of the domain. The reads are
// `b3w_leaves_base_coset`'s (coalesced); only the digest's slot moves. Twin of
// `p1s_leaves_base_coset_rev`.
extern "C" __global__ void b3w_leaves_base_coset_rev(const uint64_t *__restrict__ codeword,
                                                     uint64_t num_leaves, uint64_t block,
                                                     uint32_t leaf_bits,
                                                     uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    Blake3Chain h;
    h.init();
    for (uint64_t t = 0; t < block; ++t) h.push_felt(codeword[tid + t * num_leaves]);
    uint64_t at = leaf_bits == 0 ? tid : (__brevll(tid) >> (64 - leaf_bits));
    h.finalize(out + at * 32);
}

// An ext3 codeword's fold blocks: leaf `j` hashes `codeword[j + t·num_leaves]`
// for `t` in `[0, block)`, each value as its three limbs in order — what
// `FieldElement::<Ext3>::write_bytes_be` streams. Twin of
// `keccak256_leaves_ext3_coset`.
extern "C" __global__ void b3w_leaves_ext3_coset(const uint64_t *__restrict__ codeword,
                                                 uint64_t num_leaves, uint64_t block,
                                                 uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    Blake3Chain h;
    h.init();
    for (uint64_t t = 0; t < block; ++t) {
        const uint64_t *at = codeword + (tid + t * num_leaves) * 3;
#pragma unroll
        for (int k = 0; k < 3; ++k) h.push_felt(at[k]);
    }
    h.finalize(out + tid * 32);
}

// The proof-of-work nonce search under standard BLAKE3.
//
// The host predicate (`crypto::grinding::is_valid_nonce::<Blake3Std>`) hashes
// the 40 bytes `inner_hash ‖ nonce.to_be_bytes()` — one block, first and last,
// so ONE compression with `h = IV`, `t = 0`, `block_len = 40`, flags
// CHUNK_START|CHUNK_END|ROOT — and compares the first eight digest bytes, read
// big-endian, with `limit`. `inner_lanes` are the 32 inner-hash bytes as four
// LITTLE-endian u64 lanes (`crypto::grinding::inner_hash_lanes`), so message
// word `2i` is lane `i`'s low half and `2i + 1` its high half. The nonce's
// big-endian bytes are words 8 and 9: the byte swap of its high and its low
// 32 bits. Digest bytes 0..8 are output words 0 and 1 little-endian, so the
// big-endian head is `bswap(out[0]) << 32 | bswap(out[1])`.
//
// The walk is `grind_search`'s (keccak.cu): each thread strides over
// `[base, base + count)` and `atomicMin`s the smallest valid nonce into
// `*result`, stopping once a smaller one is known.
extern "C" __global__ void b3w_grind_search(const uint64_t *inner_lanes, uint64_t limit,
                                            uint64_t base, uint64_t count,
                                            volatile unsigned long long *result) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint32_t inner[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        inner[2 * i] = (uint32_t)inner_lanes[i];
        inner[2 * i + 1] = (uint32_t)(inner_lanes[i] >> 32);
    }
    for (uint64_t i = tid; i < count; i += stride) {
        uint64_t nonce = base + i;
        if (nonce < base) break;
        if (nonce >= (uint64_t)*result) break;
        uint32_t m[16];
#pragma unroll
        for (int k = 0; k < 8; ++k) m[k] = inner[k];
        m[8] = bswap32((uint32_t)(nonce >> 32));
        m[9] = bswap32((uint32_t)nonce);
#pragma unroll
        for (int k = 10; k < 16; ++k) m[k] = 0;
        uint32_t out[16];
        blake3_compress<BLAKE3_ROUNDS>(BLAKE3_IV, m, 0, 40u, BLAKE3_FLAGS_ONE_BLOCK, out);
        uint64_t head = ((uint64_t)bswap32(out[0]) << 32) | (uint64_t)bswap32(out[1]);
        if (head < limit) {
            atomicMin((unsigned long long *)result, (unsigned long long)nonce);
        }
    }
}

// This module's compiled-in round count: the loader asserts it is 7.
extern "C" __global__ void b3w_rounds_probe(uint32_t *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = (uint32_t)BLAKE3_ROUNDS;
}
