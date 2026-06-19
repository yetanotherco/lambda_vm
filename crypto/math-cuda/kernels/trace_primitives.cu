// Trivial trace-builder primitives used by every per-table GPU port.
// All operate on raw u64 in canonical Goldilocks form. No field arithmetic
// here — these are bulk data shuffles.
//
// Kernel inventory (1 thread = 1 output element unless noted):
//   1. pad_to_pow2_u64                  src[0..src_len] + sentinel tail
//   2. decompose_u64_to_bytes           8 bytes per u64, little-endian
//   3. decompose_u64_to_halfwords       4 halfwords per u64, little-endian
//   4. fill_sequential_u64              dst[i] = start + i * stride
//   5. range_check_column_u64           dst[i] = i
//   6. extract_bits_u64                 dst[i] = (src[i] >> shift) & mask
//   7. multiplicity_count_by_index      atomicAdd(counts[keys[i]], 1)
//
// All launches use grid = ceil(n / BLOCK_SIZE), block = BLOCK_SIZE.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// 1. pad_to_pow2_u64
//
// dst[i] = src[i] for i < src_len, sentinel otherwise. Caller arranges
// dst_len >= src_len; if dst_len > src_len the tail is filled with `sentinel`.
// ---------------------------------------------------------------------------
extern "C" __global__ void pad_to_pow2_u64(
    const uint64_t *src,
    uint64_t src_len,
    uint64_t sentinel,
    uint64_t *dst,
    uint64_t dst_len
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= dst_len) return;
    dst[i] = (i < src_len) ? src[i] : sentinel;
}

// ---------------------------------------------------------------------------
// 2. decompose_u64_to_bytes
//
// For each i in 0..n: writes 8 bytes (LSB-first) to dst[i*8 .. i*8+8].
// Output is in canonical Goldilocks form (each byte fits in u64 as 0..255).
// ---------------------------------------------------------------------------
extern "C" __global__ void decompose_u64_to_bytes(
    const uint64_t *src,
    uint64_t n,
    uint64_t *dst    // 8 * n u64s
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t v = src[i];
    uint64_t *out = dst + i * 8;
    out[0] = (v >>  0) & 0xff;
    out[1] = (v >>  8) & 0xff;
    out[2] = (v >> 16) & 0xff;
    out[3] = (v >> 24) & 0xff;
    out[4] = (v >> 32) & 0xff;
    out[5] = (v >> 40) & 0xff;
    out[6] = (v >> 48) & 0xff;
    out[7] = (v >> 56) & 0xff;
}

// ---------------------------------------------------------------------------
// 3. decompose_u64_to_halfwords
//
// For each i in 0..n: writes 4 halfwords (16-bit, LSB-first) to dst[i*4..i*4+4].
// Each halfword stored as u64 in canonical Goldilocks form (0..65535).
// ---------------------------------------------------------------------------
extern "C" __global__ void decompose_u64_to_halfwords(
    const uint64_t *src,
    uint64_t n,
    uint64_t *dst    // 4 * n u64s
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t v = src[i];
    uint64_t *out = dst + i * 4;
    out[0] = (v >>  0) & 0xffff;
    out[1] = (v >> 16) & 0xffff;
    out[2] = (v >> 32) & 0xffff;
    out[3] = (v >> 48) & 0xffff;
}

// ---------------------------------------------------------------------------
// 4. fill_sequential_u64
//
// dst[i] = start + i * stride, plain u64 arithmetic (no Goldilocks reduction).
// Caller is responsible for keeping start + (n-1)*stride below the Goldilocks
// prime when the output is consumed as a field element — typical uses
// (timestamps, indices, offsets) stay well below 2^32 so this is safe.
// ---------------------------------------------------------------------------
extern "C" __global__ void fill_sequential_u64(
    uint64_t start,
    uint64_t stride,
    uint64_t n,
    uint64_t *dst
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    dst[i] = start + i * stride;
}

// ---------------------------------------------------------------------------
// 5. range_check_column_u64
//
// dst[i] = i. Special case of fill_sequential with start=0, stride=1; kept
// as a separate kernel so call sites read clearly.
// ---------------------------------------------------------------------------
extern "C" __global__ void range_check_column_u64(uint64_t n, uint64_t *dst) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    dst[i] = i;
}

// ---------------------------------------------------------------------------
// 6. extract_bits_u64
//
// dst[i] = (src[i] >> shift) & mask, where mask = (1ULL << width) - 1.
// Used by decode/branch/load/shift to extract opcode fields and flag bits.
// `width == 64` is a degenerate "extract everything from shift onward" case.
// ---------------------------------------------------------------------------
extern "C" __global__ void extract_bits_u64(
    const uint64_t *src,
    uint64_t n,
    uint32_t shift,
    uint32_t width,
    uint64_t *dst
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t mask = (width >= 64) ? ~0ULL : ((1ULL << width) - 1);
    dst[i] = (src[i] >> shift) & mask;
}

// ---------------------------------------------------------------------------
// 7. multiplicity_count_by_index
//
// For each i in 0..n: atomicAdd(counts[keys[i]], 1). Caller pre-zeroes the
// `counts` buffer (size = max_key + 1 supplied externally) and guarantees
// every keys[i] < counts_len. Used by decode.rs (PC-keyed multiplicities).
// ---------------------------------------------------------------------------
extern "C" __global__ void multiplicity_count_by_index(
    const uint64_t *keys,
    uint64_t n,
    uint64_t *counts    // pre-zeroed
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t k = keys[i];
    atomicAdd((unsigned long long *)&counts[k], 1ULL);
}
