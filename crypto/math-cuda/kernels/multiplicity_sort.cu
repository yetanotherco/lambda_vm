// Multi-field-key multiplicity counting: bit-by-bit radix sort over u128
// keys (passed as parallel (hi, lo) u64 arrays) followed by a segmented
// reduce to compact unique keys + counts.
//
// Used by branch.rs / mul.rs / dvrm.rs trace expansion ports (PR-7+).
// PR-6 ships the primitive + parity tests; the first production consumer
// arrives with the first multi-field-key table port.
//
// Stage A — radix sort: 128 bit-passes (LSB first across `lo`, then `hi`).
// Each pass does extract_bit_predicate → inclusive_scan_u64 → scatter_by_bit
// with ping-pong buffers. No host bounce inside the loop — total_ones is
// read on-device from the scan buffer.
//
// Stage B — segmented reduce: mark_boundaries → inclusive_scan_u64 →
// compact_unique_and_counts. One u64 D2H at the end to learn num_unique
// so the caller can size its output buffers.
//
// Block sizing: BLOCK_SIZE = 256, matches inverse.cu's scan pattern.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// 1. extract_bit_predicate
//
// pred[i] = (key[i] >> bit_in_half) & 1, where bit selects lo (b < 64) or
// hi (b >= 64) and bit_in_half = b % 64.
// ---------------------------------------------------------------------------
extern "C" __global__ void extract_bit_predicate(
    const uint64_t *keys_hi,
    const uint64_t *keys_lo,
    uint64_t n,
    uint32_t bit,        // 0..128
    uint64_t *pred       // 0 or 1
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t v = (bit < 64) ? keys_lo[i] : keys_hi[i];
    uint32_t b = (bit < 64) ? bit : (bit - 64);
    pred[i] = (v >> b) & 1ULL;
}

// ---------------------------------------------------------------------------
// 2. block_inclusive_scan_fwd_u64
//
// Per-block inclusive prefix sum over u64. Output: scan_out[i] = sum of
// input[block_start..=i] within each block. block_totals[bid] = sum across
// the entire block (last valid thread writes it).
//
// Partial last block: out-of-range threads load 0 (identity for add), so the
// scan stays correct. Mutually exclusive last-thread mask prevents two
// threads from racing on block_totals.
// ---------------------------------------------------------------------------
extern "C" __global__ void block_inclusive_scan_fwd_u64(
    const uint64_t *input,
    uint64_t n,
    uint64_t *scan_out,
    uint64_t *block_totals    // length = ceil(n / BLOCK_SIZE)
) {
    __shared__ uint64_t shmem[BLOCK_SIZE];
    uint64_t tid = threadIdx.x;
    uint64_t gid = (uint64_t)blockIdx.x * BLOCK_SIZE + tid;
    shmem[tid] = (gid < n) ? input[gid] : 0ULL;
    __syncthreads();

    for (uint32_t offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        uint64_t prev = (tid >= offset) ? shmem[tid - offset] : 0ULL;
        __syncthreads();
        if (tid >= offset) {
            shmem[tid] = prev + shmem[tid];
        }
        __syncthreads();
    }

    if (gid < n) {
        scan_out[gid] = shmem[tid];
    }

    // Block total: the value at the last VALID thread of this block.
    uint64_t block_end = ((uint64_t)blockIdx.x + 1) * BLOCK_SIZE;
    uint64_t last_valid = (block_end - 1 < n - 1) ? (block_end - 1) : (n - 1);
    if (gid == last_valid) {
        block_totals[blockIdx.x] = shmem[tid];
    }
}

// ---------------------------------------------------------------------------
// 3. apply_block_offsets_fwd_u64
//
// scan_inout[gid] += block_totals_scanned[blockIdx.x - 1] for block > 0.
// Block 0 has no offset.
// ---------------------------------------------------------------------------
extern "C" __global__ void apply_block_offsets_fwd_u64(
    uint64_t *scan_inout,
    uint64_t n,
    const uint64_t *block_totals_scanned
) {
    if (blockIdx.x == 0) return;
    uint64_t tid = threadIdx.x;
    uint64_t gid = (uint64_t)blockIdx.x * BLOCK_SIZE + tid;
    if (gid >= n) return;
    scan_inout[gid] += block_totals_scanned[blockIdx.x - 1];
}

// ---------------------------------------------------------------------------
// 4. scatter_by_bit
//
// Given pred[i] (0 or 1) and the inclusive scan of pred, compute each
// element's final position:
//   pred[i] == 0 → pos = i - inclusive_scan[i]              (rank among zeros)
//   pred[i] == 1 → pos = total_zeros + inclusive_scan[i]-1  (rank among ones)
// where total_zeros = n - inclusive_scan[n-1]. Reads scan_buf[n-1] on-device
// to avoid a per-bit D2H round-trip.
// ---------------------------------------------------------------------------
extern "C" __global__ void scatter_by_bit(
    const uint64_t *keys_hi_in,
    const uint64_t *keys_lo_in,
    uint64_t n,
    const uint64_t *pred,
    const uint64_t *inclusive_scan,
    uint64_t *keys_hi_out,
    uint64_t *keys_lo_out
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t total_ones = inclusive_scan[n - 1];
    uint64_t total_zeros = n - total_ones;
    uint64_t pos;
    if (pred[i] == 0ULL) {
        pos = i - inclusive_scan[i];
    } else {
        pos = total_zeros + inclusive_scan[i] - 1ULL;
    }
    keys_hi_out[pos] = keys_hi_in[i];
    keys_lo_out[pos] = keys_lo_in[i];
}

// ---------------------------------------------------------------------------
// 5. mark_boundaries
//
// is_first[i] = 1 iff i == 0 or key[i] != key[i-1]. Run AFTER the sort.
// ---------------------------------------------------------------------------
extern "C" __global__ void mark_boundaries(
    const uint64_t *keys_hi,
    const uint64_t *keys_lo,
    uint64_t n,
    uint64_t *is_first
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    if (i == 0) {
        is_first[i] = 1;
    } else {
        bool diff = (keys_hi[i] != keys_hi[i - 1]) || (keys_lo[i] != keys_lo[i - 1]);
        is_first[i] = diff ? 1ULL : 0ULL;
    }
}

// ---------------------------------------------------------------------------
// 6. compact_unique_and_counts
//
// For each sorted-array index i:
//   - if is_first[i]==1, write (keys_hi[i], keys_lo[i]) to unique buffers at
//     slot (first_inclusive_scan[i] - 1).
//   - always: atomicAdd 1 into counts[first_inclusive_scan[i] - 1].
// Caller pre-zeroes counts.
// ---------------------------------------------------------------------------
extern "C" __global__ void compact_unique_and_counts(
    const uint64_t *keys_hi,
    const uint64_t *keys_lo,
    uint64_t n,
    const uint64_t *is_first,
    const uint64_t *first_inclusive_scan,
    uint64_t *unique_hi,
    uint64_t *unique_lo,
    uint64_t *counts            // pre-zeroed
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    uint64_t u = first_inclusive_scan[i] - 1ULL;
    if (is_first[i] == 1ULL) {
        unique_hi[u] = keys_hi[i];
        unique_lo[u] = keys_lo[i];
    }
    atomicAdd((unsigned long long *)&counts[u], 1ULL);
}
