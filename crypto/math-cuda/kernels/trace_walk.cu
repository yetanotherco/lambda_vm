// On-GPU register memory-model walk.
//
// Recovers, for each register access, the predecessor `(old_value, old_ts)` at its
// register address — the device analog of the sequential CPU walk. The model is
// read-old / write-new where EVERY access advances the cell timeline (a read writes
// the cell back at its own ts), so `old` is the *previous access* at the same
// address, not the previous write. See `prover/.../trace_builder.rs::
// walk_register_accesses` (the CPU reference this must match bit-for-bit).
//
// Correctness-first design (register keyspace is tiny — bucket = register word
// address < nbins ≤ 512): a STABLE counting-sort group-by (accesses within a bucket
// stay in input order, which is ts order) followed by a per-run predecessor link.
//   1. walk_seg_hist    — per-segment per-bucket histogram
//   2. walk_seg_offsets — bucket base offsets + per-segment prefix (one block)
//   3. walk_seg_scatter — stable scatter of access indices into grouped `perm`
//   4. walk_link        — old[perm[p]] = access at perm[p-1] (or the bucket seed)
//
// Segments partition the input into `grid.x` contiguous chunks of `seg_size`; one
// block per segment. All position/index arrays are u64 (n can exceed u32).

#include <cstdint>

// --- 1. Per-segment histogram: block s counts bucket keys in its chunk. ---
// seg_hist[s*nbins + b] = count of key==b in segment s. Shared u32 histogram
// (per-segment counts ≤ seg_size fit u32), written out as u64.
extern "C" __global__ void walk_seg_hist(const uint32_t *key, uint64_t n,
                                         uint32_t nbins, uint64_t seg_size,
                                         uint64_t *seg_hist) {
    extern __shared__ uint32_t sh[]; // nbins entries
    for (uint32_t b = threadIdx.x; b < nbins; b += blockDim.x)
        sh[b] = 0u;
    __syncthreads();

    uint64_t start = (uint64_t)blockIdx.x * seg_size;
    uint64_t end = start + seg_size;
    if (end > n)
        end = n;
    for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x)
        atomicAdd(&sh[key[i]], 1u);
    __syncthreads();

    uint64_t *out = seg_hist + (uint64_t)blockIdx.x * nbins;
    for (uint32_t b = threadIdx.x; b < nbins; b += blockDim.x)
        out[b] = (uint64_t)sh[b];
}

// --- 2. Offsets (single block, blockDim >= nbins). Transforms seg_hist in place
// into per-(segment,bucket) absolute base positions, and emits global_off[b] =
// start of bucket b's run. Thread b owns bucket b. ---
extern "C" __global__ void walk_seg_offsets(uint64_t *seg_hist, uint64_t seg,
                                            uint32_t nbins, uint64_t *global_off) {
    extern __shared__ uint64_t total[]; // nbins entries
    uint32_t b = threadIdx.x;
    if (b < nbins) {
        // Exclusive prefix over segments for bucket b; leaves running total.
        uint64_t running = 0;
        for (uint64_t s = 0; s < seg; ++s) {
            uint64_t idx = s * nbins + b;
            uint64_t c = seg_hist[idx];
            seg_hist[idx] = running; // per-segment base within the bucket (pre-global)
            running += c;
        }
        total[b] = running; // total count for bucket b
    }
    __syncthreads();

    // Exclusive scan of totals across buckets → global_off (thread 0, nbins small).
    if (threadIdx.x == 0) {
        uint64_t acc = 0;
        for (uint32_t k = 0; k < nbins; ++k) {
            global_off[k] = acc;
            acc += total[k];
        }
    }
    __syncthreads();

    // Fold the bucket base into every segment's per-bucket position.
    if (b < nbins) {
        uint64_t base = global_off[b];
        for (uint64_t s = 0; s < seg; ++s)
            seg_hist[s * nbins + b] += base;
    }
}

// --- 3. Stable scatter: block s (thread 0) walks its segment in order, placing
// each access index into `perm` at its bucket's running cursor. In-order within a
// segment + segment-ordered bases ⇒ globally stable (input = ts order preserved). ---
extern "C" __global__ void walk_seg_scatter(const uint32_t *key, uint64_t n,
                                            uint32_t nbins, uint64_t seg_size,
                                            uint64_t *seg_base, uint64_t *perm) {
    if (threadIdx.x != 0)
        return;
    uint64_t start = (uint64_t)blockIdx.x * seg_size;
    uint64_t end = start + seg_size;
    if (end > n)
        end = n;
    uint64_t *cursor = seg_base + (uint64_t)blockIdx.x * nbins; // this block's row
    for (uint64_t i = start; i < end; ++i) {
        uint64_t p = cursor[key[i]]++;
        perm[p] = i;
    }
}

// --- 4. Predecessor link: one thread per grouped position p. The access at perm[p]
// links to perm[p-1] (previous access at the same bucket) unless p starts the run
// (p == global_off[bucket]), which seeds from init. Writes old_* at ORIGINAL index. ---
extern "C" __global__ void walk_link(const uint64_t *perm, const uint32_t *key,
                                     const uint64_t *ts, const uint64_t *value,
                                     const uint64_t *global_off,
                                     const uint64_t *init_value, uint64_t init_ts,
                                     uint64_t n, uint64_t *old_value,
                                     uint64_t *old_ts) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    uint64_t i = perm[p];
    uint32_t b = key[i];
    if (p == global_off[b]) {
        old_value[i] = init_value[b];
        old_ts[i] = init_ts;
    } else {
        uint64_t j = perm[p - 1];
        old_value[i] = value[j];
        old_ts[i] = ts[j];
    }
}

// =============================================================================
// Route + compact: partition the walked accesses into MEMW_R rows vs the rare
// fallback (timestamp delta out of the IS_HALFWORD range), then compact each
// partition into contiguous positions on device. Mirrors the host
// `MemwBuckets::push_reg_access` routing (`reg_ts_delta_in_range`) so the device
// MEMW_R fill + IS_HALF histogram reproduce the CPU walk bit-for-bit while only
// the small fallback subset ever returns to the host.
// =============================================================================

// Per access: MEMW_R (emitting row, delta in range) vs fallback (emitting row,
// delta out of range) vs neither (timeline-only access, e.g. the implicit PC
// write). Matches `reg_ts_delta_in_range`: same hi32, ts_lo>old_ts_lo, delta ≤ 2^16.
extern "C" __global__ void memw_route_flags(uint64_t n, const uint64_t *ts,
                                            const uint64_t *old_ts,
                                            const uint8_t *emits_row,
                                            uint32_t *flag_memw,
                                            uint32_t *flag_fb) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t memw = 0u, fb = 0u;
    if (emits_row[i]) {
        uint64_t t = ts[i], ot = old_ts[i];
        uint64_t t_lo = t & 0xFFFFFFFFull, ot_lo = ot & 0xFFFFFFFFull;
        bool in_range =
            (t >> 32) == (ot >> 32) && t_lo > ot_lo && (t_lo - ot_lo) <= 0x10000ull;
        if (in_range)
            memw = 1u;
        else
            fb = 1u;
    }
    flag_memw[i] = memw;
    flag_fb[i] = fb;
}

// Two-level exclusive prefix scan of a 0/1 flag array over `nblocks` contiguous
// blocks of `epb` elements (nblocks ≤ blockDim of the spine). Correctness-first:
// each block is scanned serially by its lane 0 (flags are cheap counters; this is
// never the bottleneck next to the walk).
//   scan_reduce     — block total into block_tot[blk]
//   scan_spine      — single block, exclusive-scan block_tot in place; grand total
//   scan_write_excl — per-element exclusive prefix seeded by block_tot[blk]
extern "C" __global__ void scan_reduce(const uint32_t *flag, uint64_t n,
                                       uint64_t epb, uint64_t *block_tot) {
    if (threadIdx.x != 0)
        return;
    uint64_t start = (uint64_t)blockIdx.x * epb;
    uint64_t end = start + epb;
    if (end > n)
        end = n;
    uint64_t s = 0;
    for (uint64_t i = start; i < end; ++i)
        s += flag[i];
    block_tot[blockIdx.x] = s;
}

extern "C" __global__ void scan_spine(uint64_t *block_tot, uint64_t nblocks,
                                      uint64_t *total_out) {
    if (threadIdx.x != 0)
        return;
    uint64_t acc = 0;
    for (uint64_t b = 0; b < nblocks; ++b) {
        uint64_t c = block_tot[b];
        block_tot[b] = acc;
        acc += c;
    }
    total_out[0] = acc;
}

extern "C" __global__ void scan_write_excl(const uint32_t *flag, uint64_t n,
                                           uint64_t epb,
                                           const uint64_t *block_base,
                                           uint64_t *excl) {
    if (threadIdx.x != 0)
        return;
    uint64_t start = (uint64_t)blockIdx.x * epb;
    uint64_t end = start + epb;
    if (end > n)
        end = n;
    uint64_t run = block_base[blockIdx.x];
    for (uint64_t i = start; i < end; ++i) {
        excl[i] = run;
        run += flag[i];
    }
}

// row_index[i] = compacted MEMW_R row for an in-range access, else -1 (the fill
// skips row < 0). `excl_memw` is the exclusive prefix of `flag_memw`.
extern "C" __global__ void memw_rowindex_from_excl(uint64_t n,
                                                   const uint32_t *flag_memw,
                                                   const uint64_t *excl_memw,
                                                   long long *row_index) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    row_index[i] = flag_memw[i] ? (long long)excl_memw[i] : -1LL;
}

// Localize the global compacted MEMW_R row index to one chunk `[row_lo, row_hi)`:
// `local[i] = global_row_index[i] - row_lo` for rows in the chunk, else -1 (skipped
// by the fill). Lets one walk feed several capped MEMW_R tables (the prover splits
// register rows into ≤2^20-row chunks) without re-walking.
extern "C" __global__ void memw_rowindex_localize(uint64_t n,
                                                  const long long *global_row_index,
                                                  long long row_lo, long long row_hi,
                                                  long long *local) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    long long g = global_row_index[i];
    local[i] = (g >= row_lo && g < row_hi) ? (g - row_lo) : -1LL;
}

// One IS_HALFWORD lookup per MEMW_R row, keyed by the delta `ts_lo-old_ts_lo-1`
// (a u16). 65536-bin histogram; the host merges these counts into the BITWISE
// histogram (`memw_register_is_half_lookup`, multiplicity +1 per row).
extern "C" __global__ void memw_is_half_hist(uint64_t n, const uint32_t *flag_memw,
                                             const uint64_t *ts,
                                             const uint64_t *old_ts,
                                             unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (!flag_memw[i])
        return;
    uint32_t t_lo = (uint32_t)(ts[i] & 0xFFFFFFFFull);
    uint32_t ot_lo = (uint32_t)(old_ts[i] & 0xFFFFFFFFull);
    uint32_t d = (t_lo - ot_lo - 1u) & 0xFFFFu;
    atomicAdd(&hist[d], 1ull);
}

// Gather the rare fallback subset into a compact array, in emit order (via the
// `excl_fb` prefix), so the host builds their MemwOperations identically to the
// sequential path. Record = 6 u64: reg_addr, ts, value, old_value, old_ts, is_read.
extern "C" __global__ void memw_fb_gather(uint64_t n, const uint32_t *flag_fb,
                                          const uint64_t *excl_fb,
                                          const uint32_t *reg_addr,
                                          const uint64_t *ts, const uint64_t *value,
                                          const uint64_t *old_value,
                                          const uint64_t *old_ts,
                                          const uint8_t *is_read, uint64_t *fb_out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (!flag_fb[i])
        return;
    uint64_t slot = excl_fb[i] * 6ull;
    fb_out[slot + 0] = (uint64_t)reg_addr[i];
    fb_out[slot + 1] = ts[i];
    fb_out[slot + 2] = value[i];
    fb_out[slot + 3] = old_value[i];
    fb_out[slot + 4] = old_ts[i];
    fb_out[slot + 5] = (uint64_t)is_read[i];
}
