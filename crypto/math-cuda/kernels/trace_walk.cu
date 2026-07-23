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

// ============================================================================
// Image-on-device: look up the initial-memory-image byte for each accessed address via binary
// search over the sorted image `(img_addr, img_val)` (ascending img_addr). `out[i]` = image byte at
// `addr[i]`, or 0 if absent (matches the host `image.get(addr).unwrap_or(0)`). One-time image upload
// replaces per-access host HashMap lookups; provides `init_value` for the resident memory walk and
// init/fini for the resident PAGE source.
// ----------------------------------------------------------------------------
extern "C" __global__ void image_lookup(uint64_t n, const uint64_t *addr, const uint64_t *img_addr,
                                        const uint64_t *img_val, uint64_t n_img, uint64_t *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t a = addr[i];
    uint64_t lo = 0, hi = n_img, found = 0;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t m = img_addr[mid];
        if (m == a) {
            found = img_val[mid];
            break;
        } else if (m < a) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    out[i] = found;
}

// ============================================================================
// P1: device REGISTER-access emission from the resident cpu_ops (packed decode +
// rv1/rv2/rvd/next_pc). Mirrors `trace_builder::emit_register_accesses`: per op emit
// M1 rs1@ts (read), M3 rs2@ts+1 (read), M5 rd@ts+2 (write) — each only when its
// read/write flag is set AND the register != x0 — plus the implicit PC write
// @ts+1 (reg PC_WORD_ADDR=510, value=next_pc, NON-emitting, row_index=-1). ts = i*4+4.
// Two passes: counts → excl_scan (access offset + emit-row base) → scatter.
// ----------------------------------------------------------------------------
#define RA_PD_READ_REG1 0
#define RA_PD_READ_REG2 1
#define RA_PD_WRITE_REG 2
#define RA_PD_RS1 10
#define RA_PD_RS2 18
#define RA_PD_RD 26
#define RA_PC_WORD_ADDR 510u
__device__ __forceinline__ bool ra_bit(uint64_t pk, int b) { return (pk >> b) & 1ull; }
__device__ __forceinline__ uint32_t ra_byte(uint64_t pk, int off) {
    return (uint32_t)((pk >> off) & 0xFFull);
}

// Per-op counts: acc_cnt = emitting accesses + 1 (PC always); emit_cnt = emitting only.
extern "C" __global__ void reg_access_counts(uint64_t n, const uint64_t *packed,
                                             uint32_t *acc_cnt, uint32_t *emit_cnt) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    uint32_t e = 0;
    if (ra_bit(pk, RA_PD_READ_REG1) && ra_byte(pk, RA_PD_RS1) != 0) e++;
    if (ra_bit(pk, RA_PD_READ_REG2) && ra_byte(pk, RA_PD_RS2) != 0) e++;
    if (ra_bit(pk, RA_PD_WRITE_REG) && ra_byte(pk, RA_PD_RD) != 0) e++;
    emit_cnt[i] = e;
    acc_cnt[i] = e + 1; // + implicit PC write
}

// Scatter the accesses (op order, within-op M1/M3/M5/PC) at acc_off[i]; emitting rows get
// row_index = row_base[i] + local emit index, the PC write gets -1.
extern "C" __global__ void reg_access_scatter(
    uint64_t n, const uint64_t *packed, const uint64_t *rv1, const uint64_t *rv2,
    const uint64_t *rvd, const uint64_t *next_pc, const uint64_t *acc_off,
    const uint64_t *row_base, uint32_t *reg_addr, uint64_t *ts_out, uint64_t *value,
    uint8_t *is_read, int64_t *row_index) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    uint64_t ts = i * 4ull + 4ull;
    uint64_t pos = acc_off[i];
    int64_t row = (int64_t)row_base[i];

    if (ra_bit(pk, RA_PD_READ_REG1) && ra_byte(pk, RA_PD_RS1) != 0) {
        reg_addr[pos] = 2u * ra_byte(pk, RA_PD_RS1);
        ts_out[pos] = ts;
        value[pos] = rv1[i];
        is_read[pos] = 1u;
        row_index[pos] = row++;
        pos++;
    }
    if (ra_bit(pk, RA_PD_READ_REG2) && ra_byte(pk, RA_PD_RS2) != 0) {
        reg_addr[pos] = 2u * ra_byte(pk, RA_PD_RS2);
        ts_out[pos] = ts + 1;
        value[pos] = rv2[i];
        is_read[pos] = 1u;
        row_index[pos] = row++;
        pos++;
    }
    if (ra_bit(pk, RA_PD_WRITE_REG) && ra_byte(pk, RA_PD_RD) != 0) {
        reg_addr[pos] = 2u * ra_byte(pk, RA_PD_RD);
        ts_out[pos] = ts + 2;
        value[pos] = rvd[i];
        is_read[pos] = 0u;
        row_index[pos] = row++;
        pos++;
    }
    // Implicit PC write (non-emitting).
    reg_addr[pos] = RA_PC_WORD_ADDR;
    ts_out[pos] = ts + 1;
    value[pos] = next_pc[i];
    is_read[pos] = 0u;
    row_index[pos] = -1;
}

// ---------------------------------------------------------------------------
// P1-ecall: INTERLEAVED register-access emission. The register walk links each access to its
// predecessor in a STABLE counting-sort by bin that PRESERVES INPUT ORDER (it does NOT ts-sort
// within a bin), so ecall accesses must be emitted at their op's timeline position, not appended.
// These kernels reserve per-op slots for the op's ecall accesses and write them right after the
// op's regular accesses + PC write (all non-emitting, row_index = -1 — their MEMW rows are produced
// on CPU per Option Z). ecall accesses arrive grouped by op index (non-decreasing).
// ---------------------------------------------------------------------------

// Per-op count of injected ecall accesses (scatter-add). `ecall_op_cnt` is pre-zeroed, size n.
extern "C" __global__ void reg_ecall_op_counts(uint64_t m, const uint32_t *ecall_op_index,
                                               uint32_t *ecall_op_cnt) {
    uint64_t k = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m)
        return;
    atomicAdd(&ecall_op_cnt[ecall_op_index[k]], 1u);
}

// Like `reg_access_counts` but reserves `ecall_op_cnt[i]` extra (non-emitting) slots per op.
extern "C" __global__ void reg_access_counts_ecall(uint64_t n, const uint64_t *packed,
                                                   const uint32_t *ecall_op_cnt, uint32_t *acc_cnt,
                                                   uint32_t *emit_cnt) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    uint32_t e = 0;
    if (ra_bit(pk, RA_PD_READ_REG1) && ra_byte(pk, RA_PD_RS1) != 0) e++;
    if (ra_bit(pk, RA_PD_READ_REG2) && ra_byte(pk, RA_PD_RS2) != 0) e++;
    if (ra_bit(pk, RA_PD_WRITE_REG) && ra_byte(pk, RA_PD_RD) != 0) e++;
    // ecall register accesses ARE emitting MEMW_R candidates: the routed histogram kernel applies the
    // register ts-delta filter (matching the CPU's is_register_op routing of the ecall MemwOperations
    // into `register_rows`), so they must reserve emit slots. Only the implicit PC write is non-emitting.
    emit_cnt[i] = e + ecall_op_cnt[i];
    acc_cnt[i] = e + 1u + ecall_op_cnt[i];       // + implicit PC write + op's ecall accesses
}

// Derive the route path's `emits_row` mask (1 = MEMW_R candidate, 0 = timeline-only PC write) from
// the emitter's compacted `row_index` (>= 0 for every emitting regular/ecall access, -1 for the PC
// write). Lets the device-emitted ecall stream feed `route_core_from_device` (which routes candidates
// into MEMW_R vs aligned/general fallback), matching the CPU's is_register_op routing.
extern "C" __global__ void rowindex_to_emits(uint64_t n, const int64_t *row_index,
                                             uint8_t *emits_row) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    emits_row[i] = (row_index[i] >= 0) ? 1u : 0u;
}

// Like `reg_access_scatter` but writes op i's ecall accesses (from the flat ecall arrays at
// `ecall_op_off[i]`) right after its regular accesses + PC write, as non-emitting timeline events.
extern "C" __global__ void reg_access_scatter_ecall(
    uint64_t n, const uint64_t *packed, const uint64_t *rv1, const uint64_t *rv2,
    const uint64_t *rvd, const uint64_t *next_pc, const uint64_t *acc_off, const uint64_t *row_base,
    const uint32_t *ecall_op_cnt, const uint64_t *ecall_op_off, const uint32_t *ecall_reg_addr,
    const uint64_t *ecall_ts, const uint64_t *ecall_value, const uint8_t *ecall_is_read,
    uint32_t *reg_addr, uint64_t *ts_out, uint64_t *value, uint8_t *is_read, int64_t *row_index) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    uint64_t ts = i * 4ull + 4ull;
    uint64_t pos = acc_off[i];
    int64_t row = (int64_t)row_base[i];

    if (ra_bit(pk, RA_PD_READ_REG1) && ra_byte(pk, RA_PD_RS1) != 0) {
        reg_addr[pos] = 2u * ra_byte(pk, RA_PD_RS1);
        ts_out[pos] = ts;
        value[pos] = rv1[i];
        is_read[pos] = 1u;
        row_index[pos] = row++;
        pos++;
    }
    if (ra_bit(pk, RA_PD_READ_REG2) && ra_byte(pk, RA_PD_RS2) != 0) {
        reg_addr[pos] = 2u * ra_byte(pk, RA_PD_RS2);
        ts_out[pos] = ts + 1;
        value[pos] = rv2[i];
        is_read[pos] = 1u;
        row_index[pos] = row++;
        pos++;
    }
    if (ra_bit(pk, RA_PD_WRITE_REG) && ra_byte(pk, RA_PD_RD) != 0) {
        reg_addr[pos] = 2u * ra_byte(pk, RA_PD_RD);
        ts_out[pos] = ts + 2;
        value[pos] = rvd[i];
        is_read[pos] = 0u;
        row_index[pos] = row++;
        pos++;
    }
    // Implicit PC write (non-emitting).
    reg_addr[pos] = RA_PC_WORD_ADDR;
    ts_out[pos] = ts + 1;
    value[pos] = next_pc[i];
    is_read[pos] = 0u;
    row_index[pos] = -1;
    pos++;
    // Interleaved ecall accesses for this op — emitting MEMW_R candidates (row_index >= 0); the routed
    // histogram kernel applies the register ts-delta filter, so fallbacks (MEMW_A/MEMW) drop naturally.
    uint32_t ec = ecall_op_cnt[i];
    uint64_t eoff = ecall_op_off[i];
    for (uint32_t k = 0; k < ec; ++k) {
        reg_addr[pos] = ecall_reg_addr[eoff + k];
        ts_out[pos] = ecall_ts[eoff + k];
        value[pos] = ecall_value[eoff + k];
        is_read[pos] = ecall_is_read[eoff + k];
        row_index[pos] = row++;
        pos++;
    }
}

// ============================================================================
// P1: device MEMORY-access emission (load/store) from the resident cpu_ops. Mirrors the host prep
// in the memw parity tests: per load/store op emit `width` byte-accesses (addr=res+j, ts, val =
// byte j of the value word) + per-op metadata (base, ts, is_read, width, signed, value_word). Two
// compactions: op-level (is_mem) and byte-level (Σ width). mem_flags: bit0=store, bit1=signed,
// bits 2/3/4 = 2B/4B/8B (default 1B); is_load value = rvd, is_store value = rv2. ts = i*4+4.
// ----------------------------------------------------------------------------
#define MA_PD_MEMORY 7
#define MA_PD_MEM_FLAGS 50
__device__ __forceinline__ uint32_t ma_width(uint64_t mf) {
    return ((mf >> 4) & 1) ? 8u : ((mf >> 3) & 1) ? 4u : ((mf >> 2) & 1) ? 2u : 1u;
}

// Per-op counts: ls_flag = 1 if memory op; byte_cnt = width (0 if not a memory op).
extern "C" __global__ void memacc_counts(uint64_t n, const uint64_t *packed, uint32_t *ls_flag,
                                         uint32_t *byte_cnt) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool is_mem = (pk >> MA_PD_MEMORY) & 1ull;
    uint64_t mf = (pk >> MA_PD_MEM_FLAGS) & 0xFFull;
    ls_flag[i] = is_mem ? 1u : 0u;
    byte_cnt[i] = is_mem ? ma_width(mf) : 0u;
}

// Scatter per-op metadata at op_off[i] and `width` byte-accesses at byte_base[i].
extern "C" __global__ void memacc_emit(
    uint64_t n, const uint64_t *packed, const uint64_t *res, const uint64_t *rvd,
    const uint64_t *rv2, const uint64_t *op_off, const uint64_t *byte_base, uint64_t *base,
    uint64_t *op_ts, uint32_t *is_read, uint32_t *width_out, uint32_t *signed_out,
    uint64_t *value_word, uint64_t *addr, uint64_t *ts_a, uint64_t *val_a, uint64_t *op_row,
    uint32_t *byte_off) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool is_mem = (pk >> MA_PD_MEMORY) & 1ull;
    if (!is_mem)
        return;
    uint64_t mf = (pk >> MA_PD_MEM_FLAGS) & 0xFFull;
    bool is_store = mf & 1ull;
    uint32_t w = ma_width(mf);
    uint32_t sgn = (mf >> 1) & 1ull;
    uint64_t ts = i * 4ull + 4ull;
    uint64_t base_addr = res[i];
    uint64_t vw = is_store ? rv2[i] : rvd[i];

    uint64_t r = op_off[i]; // compacted op row
    base[r] = base_addr;
    op_ts[r] = ts;
    is_read[r] = is_store ? 0u : 1u;
    width_out[r] = w;
    signed_out[r] = sgn;
    value_word[r] = vw;

    uint64_t b = byte_base[i];
    for (uint32_t j = 0; j < w; ++j) {
        addr[b + j] = base_addr + (uint64_t)j;
        ts_a[b + j] = ts;
        val_a[b + j] = (vw >> (8 * j)) & 0xFFull;
        op_row[b + j] = r;
        byte_off[b + j] = j;
    }
}

// ---------------------------------------------------------------------------
// P1-ecall: INTERLEAVED memory-access injection. Like the register walk, the memory walk links each
// byte-access to its predecessor in a STABLE radix sort by address that preserves INPUT ORDER (within
// an address, ts order = emit order). So ecall memory byte-accesses must be emitted at their op's
// timeline position, reserving per-op slots after the op's regular bytes. They are non-emitting: they
// get `op_row = num_ops` (a DUMP row `memw_gather` writes to but `classify`/`pack` ignore), so they
// advance the walk timeline (correct old_ts for regular LOAD/STORE) without producing MEMW rows.
// ---------------------------------------------------------------------------

// Per-op count of injected ecall memory byte-accesses (scatter-add). `ecall_byte_cnt` pre-zeroed, size n.
extern "C" __global__ void mem_ecall_byte_counts(uint64_t m, const uint32_t *mem_op_index,
                                                 uint32_t *ecall_byte_cnt) {
    uint64_t k = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= m)
        return;
    atomicAdd(&ecall_byte_cnt[mem_op_index[k]], 1u);
}

// Like `memacc_counts` but reserves `ecall_byte_cnt[i]` extra byte slots per op. `ls_flag` unchanged
// (only real is_mem LOAD/STORE ops emit MEMW rows; ecall ops are non-emitting).
extern "C" __global__ void memacc_counts_ecall(uint64_t n, const uint64_t *packed,
                                               const uint32_t *ecall_byte_cnt, uint32_t *ls_flag,
                                               uint32_t *byte_cnt) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool is_mem = (pk >> MA_PD_MEMORY) & 1ull;
    uint64_t mf = (pk >> MA_PD_MEM_FLAGS) & 0xFFull;
    ls_flag[i] = is_mem ? 1u : 0u;
    byte_cnt[i] = (is_mem ? ma_width(mf) : 0u) + ecall_byte_cnt[i];
}

// Writes op i's ecall memory bytes (from the flat ecall arrays at `ecall_op_off[i]`) into the slots
// right after its regular bytes, as non-emitting (`op_row = num_ops` DUMP, byte_off = 0).
extern "C" __global__ void memacc_emit_ecall(
    uint64_t n, const uint64_t *packed, const uint32_t *ecall_op_cnt, const uint64_t *ecall_op_off,
    const uint64_t *byte_base, const uint64_t *mem_addr, const uint64_t *mem_ts,
    const uint64_t *mem_val, uint64_t num_ops, uint64_t *addr, uint64_t *ts_a, uint64_t *val_a,
    uint64_t *op_row, uint32_t *byte_off, uint64_t *ecall_pos) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t ec = ecall_op_cnt[i];
    if (ec == 0)
        return;
    uint64_t pk = packed[i];
    bool is_mem = (pk >> MA_PD_MEMORY) & 1ull;
    uint32_t rw = is_mem ? ma_width((pk >> MA_PD_MEM_FLAGS) & 0xFFull) : 0u;
    uint64_t pos = byte_base[i] + (uint64_t)rw;
    uint64_t eoff = ecall_op_off[i];
    for (uint32_t k = 0; k < ec; ++k) {
        addr[pos] = mem_addr[eoff + k];
        ts_a[pos] = mem_ts[eoff + k];
        val_a[pos] = mem_val[eoff + k];
        op_row[pos] = num_ops; // DUMP row (ignored by classify/pack)
        byte_off[pos] = 0u;
        // Option A2: remember this ecall byte's position in the combined stream, keyed by its flat
        // ecall index, so its resolved old_ts/old_value can be read back after the walk.
        ecall_pos[eoff + k] = pos;
        pos++;
    }
}

// Option A2: gather each ecall byte's resolved old_ts/old_value (computed by the walk over the
// combined stream) back into flat per-ecall-byte arrays. `ecall_pos[e]` is the combined-stream
// position of flat ecall byte `e` (set by `memacc_emit_ecall`); `old_ts`/`old_value` are indexed
// by that same combined-stream position.
extern "C" __global__ void ecall_oldstate_gather(uint64_t m, const uint64_t *ecall_pos,
                                                 const uint64_t *old_ts, const uint64_t *old_value,
                                                 uint64_t *ecall_old_ts, uint64_t *ecall_old_val) {
    uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= m)
        return;
    uint64_t pos = ecall_pos[e];
    ecall_old_ts[e] = old_ts[pos];
    ecall_old_val[e] = old_value[pos];
}

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

// =============================================================================
// Memory memory-model walk (Phase 2). Same read-old/write-new model as the register
// walk, but the key is a sparse 64-bit BYTE ADDRESS, so the bounded direct-histogram
// group-by does not apply. Instead: a stable LSD radix sort of a permutation by the
// 64-bit address (8 passes of an 8-bit digit, each a stable counting sort reusing
// `walk_seg_offsets` with nbins=256), then a predecessor link keyed on address change.
// Accesses are emitted in ts order and the sort is stable ⇒ within an address the ts
// order is preserved, so a sort by address alone yields (address, ts) order. All arrays
// are u64. Must match the CPU `MemoryState` walk (read_bytes→old, write_bytes→new).
// =============================================================================

// MEMW table assembly (Phase-2 fill): classify each LOAD/STORE op aligned vs general, then
// pack its MEMW_A / MEMW row from the gathered walk output. Rows are compacted in program
// order (excl = exclusive prefix of the bucket flag). Constrained positions [0,width) carry
// the walk's old_ts/old_value; positions [width,8) are unconstrained (zero write-multiplicity
// on the bus) and left 0 here — a valid trace (the CPU fills them from its read-8, which does
// not affect the proof). value[8]/old[8] follow the LOAD (own value, sign-extended) / STORE
// (all 8 value bytes; old = walk) semantics of collect_load/store_op_from_cpu.

// aligned iff width==1, or (base aligned to width AND old_ts uniform over [0,width)).
extern "C" __global__ void memw_classify(uint64_t num_ops, const uint64_t *base,
                                         const uint32_t *width, const uint64_t *g_old_ts,
                                         uint32_t *flag_aligned, uint32_t *flag_general) {
    uint64_t op = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (op >= num_ops)
        return;
    uint32_t w = width[op];
    uint32_t low = (uint32_t)(base[op] & 0xFFFFFFFFu);
    bool aligned = true;
    if (w > 1u && (low & (w - 1u)) != 0u)
        aligned = false;
    if (aligned) {
        uint64_t t0 = g_old_ts[op * 8];
        for (uint32_t i = 1; i < w; ++i)
            if (g_old_ts[op * 8 + i] != t0) {
                aligned = false;
                break;
            }
    }
    flag_aligned[op] = aligned ? 1u : 0u;
    flag_general[op] = aligned ? 0u : 1u;
}

// Pack MEMW_A (stride 12) / MEMW (stride 19) rows. is_read: 1=LOAD, 0=STORE. value_word =
// loaded value (rvd) for LOAD or store value (rv2) for STORE. is_signed = mem_signed (LOAD
// sign-extension). g_old_ts/g_old_value = gathered walk output [0,width); [width,8)=0.
extern "C" __global__ void memw_pack(uint64_t num_ops, const uint64_t *base, const uint64_t *ts,
                                     const uint32_t *is_read, const uint32_t *width,
                                     const uint32_t *is_signed, const uint64_t *value_word,
                                     const uint64_t *g_old_ts, const uint64_t *g_old_value,
                                     const uint32_t *flag_aligned, const uint64_t *excl_a,
                                     const uint64_t *excl_g, uint64_t *out_aligned,
                                     uint64_t *out_general) {
    uint64_t op = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (op >= num_ops)
        return;
    uint32_t w = width[op];
    uint64_t vw = value_word[op];
    uint32_t vb[8], ob[8];
    if (is_read[op]) {
        for (uint32_t j = 0; j < 8u; ++j)
            vb[j] = (j < w) ? (uint32_t)((vw >> (8u * j)) & 0xFFu) : 0u;
        if (w < 8u) {
            uint32_t sb = (vb[w - 1] >> 7) & 1u;
            uint32_t fill = (is_signed[op] && sb) ? 0xFFu : 0u;
            for (uint32_t j = w; j < 8u; ++j)
                vb[j] = fill;
        }
        for (uint32_t j = 0; j < 8u; ++j)
            ob[j] = vb[j]; // LOAD old_value = own (res_bytes)
    } else {
        for (uint32_t j = 0; j < 8u; ++j)
            vb[j] = (uint32_t)((vw >> (8u * j)) & 0xFFu); // STORE value = all 8 bytes
        for (uint32_t j = 0; j < 8u; ++j)
            ob[j] = (j < w) ? (uint32_t)(g_old_value[op * 8 + j] & 0xFFu) : 0u;
    }
    uint64_t flags = ((uint64_t)is_read[op] << 1) | ((uint64_t)w << 8); // is_register=0
    uint64_t vlo0 = (uint64_t)vb[0] | ((uint64_t)vb[1] << 32);
    uint64_t vlo1 = (uint64_t)vb[2] | ((uint64_t)vb[3] << 32);
    uint64_t vlo2 = (uint64_t)vb[4] | ((uint64_t)vb[5] << 32);
    uint64_t vlo3 = (uint64_t)vb[6] | ((uint64_t)vb[7] << 32);
    uint64_t olo0 = (uint64_t)ob[0] | ((uint64_t)ob[1] << 32);
    uint64_t olo1 = (uint64_t)ob[2] | ((uint64_t)ob[3] << 32);
    uint64_t olo2 = (uint64_t)ob[4] | ((uint64_t)ob[5] << 32);
    uint64_t olo3 = (uint64_t)ob[6] | ((uint64_t)ob[7] << 32);
    if (flag_aligned[op]) {
        uint64_t *o = out_aligned + excl_a[op] * 12ull;
        o[0] = flags;
        o[1] = base[op];
        o[2] = ts[op];
        o[3] = g_old_ts[op * 8]; // uniform → old_timestamp[0]
        o[4] = vlo0; o[5] = vlo1; o[6] = vlo2; o[7] = vlo3;
        o[8] = olo0; o[9] = olo1; o[10] = olo2; o[11] = olo3;
    } else {
        uint64_t *o = out_general + excl_g[op] * 19ull;
        o[0] = flags;
        o[1] = base[op];
        o[2] = ts[op];
        o[3] = vlo0; o[4] = vlo1; o[5] = vlo2; o[6] = vlo3;
        o[7] = olo0; o[8] = olo1; o[9] = olo2; o[10] = olo3;
        for (uint32_t j = 0; j < 8u; ++j)
            o[11 + j] = (j < w) ? g_old_ts[op * 8 + j] : 0ull;
    }
}

// MEMW routing gather: scatter the per-byte-access walk outputs (old_ts, old_value) into
// per-op MEMW rows. Each access k belongs to op `op_row[k]` at byte position `byte_off[k]`.
// out_old_ts / out_old_value are sized num_ops*8 (pre-zeroed); positions beyond an op's
// width stay 0 (never touched). The consumer applies the LOAD old_value=own-value override
// and the is_aligned classification (uniform old_timestamp over [0,width)).
extern "C" __global__ void memw_gather(uint64_t n_acc, const uint64_t *old_ts,
                                       const uint64_t *old_value, const uint64_t *op_row,
                                       const uint32_t *byte_off, uint64_t *out_old_ts,
                                       uint64_t *out_old_value) {
    uint64_t k = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_acc)
        return;
    uint64_t slot = op_row[k] * 8ull + (uint64_t)byte_off[k];
    out_old_ts[slot] = old_ts[k];
    out_old_value[slot] = old_value[k];
}

// perm[i] = i.
extern "C" __global__ void radix_iota(uint64_t *perm, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        perm[i] = i;
}

// --- radix pass step 1: per-segment histogram of digit `(addr[perm[k]]>>shift)&0xFF`.
// Mirrors walk_seg_hist but perm-indexed + digit-extracted; nbins is fixed 256. ---
extern "C" __global__ void radix_seg_hist(const uint64_t *perm, const uint64_t *addr,
                                          uint64_t n, uint32_t shift, uint64_t seg_size,
                                          uint64_t *seg_hist) {
    __shared__ uint32_t sh[256];
    for (uint32_t b = threadIdx.x; b < 256u; b += blockDim.x)
        sh[b] = 0u;
    __syncthreads();

    uint64_t start = (uint64_t)blockIdx.x * seg_size;
    uint64_t end = start + seg_size;
    if (end > n)
        end = n;
    for (uint64_t k = start + threadIdx.x; k < end; k += blockDim.x) {
        uint8_t d = (uint8_t)((addr[perm[k]] >> shift) & 0xFFull);
        atomicAdd(&sh[d], 1u);
    }
    __syncthreads();

    uint64_t *out = seg_hist + (uint64_t)blockIdx.x * 256ull;
    for (uint32_t b = threadIdx.x; b < 256u; b += blockDim.x)
        out[b] = (uint64_t)sh[b];
}

// --- radix pass step 3: stable scatter of perm_in → perm_out by digit. (step 2 is the
// shared walk_seg_offsets with nbins=256.) Block s, lane 0 only, in segment order. ---
extern "C" __global__ void radix_seg_scatter(const uint64_t *perm_in, const uint64_t *addr,
                                             uint64_t n, uint32_t shift, uint64_t seg_size,
                                             uint64_t *seg_base, uint64_t *perm_out) {
    if (threadIdx.x != 0)
        return;
    uint64_t start = (uint64_t)blockIdx.x * seg_size;
    uint64_t end = start + seg_size;
    if (end > n)
        end = n;
    uint64_t *cursor = seg_base + (uint64_t)blockIdx.x * 256ull;
    for (uint64_t k = start; k < end; ++k) {
        uint64_t orig = perm_in[k];
        uint8_t d = (uint8_t)((addr[orig] >> shift) & 0xFFull);
        uint64_t p = cursor[d]++;
        perm_out[p] = orig;
    }
}

// =============================================================================
// Device dedup (Phase 3 resident enabler for the deduped chips: LT/SHIFT/EQ/BYTEWISE/
// MUL/DVRM/BRANCH). Sort a permutation by the full op key (3 u64 words, via the radix
// machinery above called per word — LSD), then segment-reduce: adjacent equal full-keys
// collapse to one unique row with summed multiplicity. The deduped chips' buses are
// order-independent (LogUp), so only the (unique op, mult) multiset must match the host
// HashMap — sorted order is fine. `mem_link` continues below.
// =============================================================================

// Mark the start of each equal-full-key run in the sorted permutation.
extern "C" __global__ void dedup_seg_start(uint64_t n, const uint64_t *perm, const uint64_t *k0,
                                           const uint64_t *k1, const uint64_t *k2,
                                           uint32_t *seg_start) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    if (p == 0ull) {
        seg_start[p] = 1u;
        return;
    }
    uint64_t i = perm[p], j = perm[p - 1];
    bool same = (k0[i] == k0[j]) && (k1[i] == k1[j]) && (k2[i] == k2[j]);
    seg_start[p] = same ? 0u : 1u;
}

// Emit one unique row per run: mult = run length; key written once (at the run start).
// `excl` = exclusive prefix-sum of seg_start; group id = excl[p] + seg_start[p] - 1.
// `out_mult` must be pre-zeroed.
extern "C" __global__ void dedup_emit(uint64_t n, const uint64_t *perm, const uint64_t *k0,
                                      const uint64_t *k1, const uint64_t *k2,
                                      const uint64_t *excl, const uint32_t *seg_start,
                                      uint64_t *out_k0, uint64_t *out_k1, uint64_t *out_k2,
                                      uint64_t *out_mult) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    uint64_t gid = excl[p] + (uint64_t)seg_start[p] - 1ull;
    uint64_t i = perm[p];
    atomicAdd((unsigned long long *)&out_mult[gid], 1ull);
    if (seg_start[p]) {
        out_k0[gid] = k0[i];
        out_k1[gid] = k1[i];
        out_k2[gid] = k2[i];
    }
}

// Dual-multiplicity emit (MUL: mu_lo/mu_hi; DVRM: mu_q/mu_r). Same runs as dedup_emit, but a
// per-op selector bit routes each op's +1 into out_m0 (sel=0) or out_m1 (sel=1). Key still
// dedups on (k0,k1,k2) only — the selector is NOT part of the key. Both mults pre-zeroed.
extern "C" __global__ void dedup_emit2(uint64_t n, const uint64_t *perm, const uint64_t *k0,
                                       const uint64_t *k1, const uint64_t *k2,
                                       const uint32_t *sel, const uint64_t *excl,
                                       const uint32_t *seg_start, uint64_t *out_k0,
                                       uint64_t *out_k1, uint64_t *out_k2, uint64_t *out_m0,
                                       uint64_t *out_m1) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    uint64_t gid = excl[p] + (uint64_t)seg_start[p] - 1ull;
    uint64_t i = perm[p];
    if (sel[i])
        atomicAdd((unsigned long long *)&out_m1[gid], 1ull);
    else
        atomicAdd((unsigned long long *)&out_m0[gid], 1ull);
    if (seg_start[p]) {
        out_k0[gid] = k0[i];
        out_k1[gid] = k1[i];
        out_k2[gid] = k2[i];
    }
}

// 4-key run-start marker (BRANCH: pc/offset/register/jalr).
extern "C" __global__ void dedup_seg_start4(uint64_t n, const uint64_t *perm, const uint64_t *k0,
                                            const uint64_t *k1, const uint64_t *k2,
                                            const uint64_t *k3, uint32_t *seg_start) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    if (p == 0ull) {
        seg_start[p] = 1u;
        return;
    }
    uint64_t i = perm[p], j = perm[p - 1];
    bool same = (k0[i] == k0[j]) && (k1[i] == k1[j]) && (k2[i] == k2[j]) && (k3[i] == k3[j]);
    seg_start[p] = same ? 0u : 1u;
}

// 4-key emit (single multiplicity).
extern "C" __global__ void dedup_emit4(uint64_t n, const uint64_t *perm, const uint64_t *k0,
                                       const uint64_t *k1, const uint64_t *k2, const uint64_t *k3,
                                       const uint64_t *excl, const uint32_t *seg_start,
                                       uint64_t *out_k0, uint64_t *out_k1, uint64_t *out_k2,
                                       uint64_t *out_k3, uint64_t *out_mult) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    uint64_t gid = excl[p] + (uint64_t)seg_start[p] - 1ull;
    uint64_t i = perm[p];
    atomicAdd((unsigned long long *)&out_mult[gid], 1ull);
    if (seg_start[p]) {
        out_k0[gid] = k0[i];
        out_k1[gid] = k1[i];
        out_k2[gid] = k2[i];
        out_k3[gid] = k3[i];
    }
}

// --- predecessor link over the address-sorted permutation. One thread per sorted
// position p: if p starts a new address run (p==0 or addr changed) seed from init
// (old_value = init_value[i], old_ts = 0); else old = the previous sorted access. ---
extern "C" __global__ void mem_link(const uint64_t *perm, const uint64_t *addr,
                                    const uint64_t *ts, const uint64_t *value,
                                    const uint64_t *init_value, uint64_t n,
                                    uint64_t *old_value, uint64_t *old_ts) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    uint64_t i = perm[p];
    if (p == 0ull || addr[perm[p - 1]] != addr[i]) {
        old_value[i] = init_value[i];
        old_ts[i] = 0ull;
    } else {
        uint64_t j = perm[p - 1];
        old_value[i] = value[j];
        old_ts[i] = ts[j];
    }
}

// FINAL memory snapshot: after the stable radix sort by address (`perm`), the LAST access of each
// address run holds that address's final (value, timestamp) — the memory state after the whole replay
// (regular LOAD/STORE + interleaved ecall writes, since all are in the sorted stream). `flag[p]=1` marks
// sorted-position p as the last of its address run. Feeds PAGE-FINI + ARE_BYTES without a CPU replay.
extern "C" __global__ void mem_final_flag(const uint64_t *perm, const uint64_t *addr, uint64_t n,
                                          uint32_t *flag) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    uint64_t a = addr[perm[p]];
    bool is_last = (p + 1 == n) || (addr[perm[p + 1]] != a);
    flag[p] = is_last ? 1u : 0u;
}

// Gather the final (addr, value, ts) for each flagged (last-of-run) sorted position into the compacted
// snapshot arrays at `excl[p]`.
extern "C" __global__ void mem_final_gather(const uint64_t *perm, const uint64_t *addr,
                                            const uint64_t *ts, const uint64_t *value,
                                            const uint32_t *flag, const uint64_t *excl, uint64_t n,
                                            uint64_t *out_addr, uint64_t *out_val,
                                            uint64_t *out_ts) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    if (!flag[p])
        return;
    uint64_t i = perm[p];
    uint64_t slot = excl[p];
    out_addr[slot] = addr[i];
    out_val[slot] = value[i];
    out_ts[slot] = ts[i];
}

// ---------------------------------------------------------------------------
// Device REGISTER final-state snapshot (C2-c1): per register word-address (< NADDR), the (value, ts)
// of the LAST access (max ts) — the device analog of `RegisterState::to_final_state_map`. Every
// register access (read OR write) updates the register's (value, ts) to the access's, so the max-ts
// access per address is the final state. Seeded with the init (value, ts) so never-accessed registers
// keep their init. `reg_addr`/`ts`/`value` = the resident register-access stream (emit order).
// Timestamps start at 4, so seeding max_ts = init_ts (1) means real accesses always win.
extern "C" __global__ void reg_final_seed(uint64_t naddr, const uint64_t *init_value, uint64_t init_ts,
                                          uint64_t *out_val, uint64_t *out_ts, uint64_t *max_ts) {
    uint64_t a = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= naddr)
        return;
    out_val[a] = init_value[a];
    out_ts[a] = init_ts;
    max_ts[a] = init_ts;
}
extern "C" __global__ void reg_final_maxts(uint64_t n, const uint32_t *reg_addr, const uint64_t *ts,
                                           uint64_t *max_ts) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    atomicMax((unsigned long long *)&max_ts[reg_addr[i]], (unsigned long long)ts[i]);
}
extern "C" __global__ void reg_final_gather(uint64_t n, const uint32_t *reg_addr, const uint64_t *ts,
                                            const uint64_t *value, const uint64_t *max_ts,
                                            uint64_t *out_val, uint64_t *out_ts) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t a = reg_addr[i];
    // ts is unique per address (each access has a distinct timestamp), so exactly one access matches.
    if (ts[i] == max_ts[a]) {
        out_val[a] = value[i];
        out_ts[a] = ts[i];
    }
}

// C2-c2: x254 (the commit-index register, addr 508) final state. x254 is width-1 and NOT part of the
// register access stream the walk sees (see `capture_ecall_reg_accesses`), so it is tracked separately:
// `final_index = start + Σ commit_count` over ecall-commit ops; `final_ts` = the last commit op's ts
// (op i's ts = i*4+4). `commit_flag[i]` marks ecall-commit ops; `commit_count[i]` is 0 elsewhere.
extern "C" __global__ void reg_x254_scan(uint64_t n, const uint8_t *commit_flag,
                                         const uint64_t *commit_count, unsigned long long *out_total,
                                         unsigned long long *out_last_ts) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (commit_flag[i]) {
        atomicAdd(out_total, (unsigned long long)commit_count[i]);
        atomicMax(out_last_ts, (unsigned long long)(i * 4ull + 4ull));
    }
}

// ---------------------------------------------------------------------------
// memw→lt pair generation (LT-resident-table, session 5): produce the
// timestamp-ordering LT operands from the packed MEMW rows on device — the
// device analog of `collect_lt_from_memw` / `collect_lt_from_memw_aligned`.
//   aligned row (MEMW_ALIGNED_STRIDE=12): r[2]=timestamp, r[3]=old_timestamp[0]
//     → 1 LT (lhs=old_ts[0], rhs=ts).
//   general row (MEMW_STRIDE=19): r[0] flags (width @ bits 8-15), r[2]=timestamp,
//     r[11+i]=old_timestamp[i] → `width` LTs (lhs=old_ts[i], rhs=ts), i in 0..width.
// The LT bus is order-free (LogUp), so pairs land at a compacted offset in any order.
// ---------------------------------------------------------------------------
extern "C" __global__ void memw_lt_widths(uint64_t ng, const uint64_t *pg, uint32_t *width_out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng)
        return;
    width_out[i] = (uint32_t)((pg[i * 19ull] >> 8) & 0xFFull);
}

extern "C" __global__ void memw_lt_emit_aligned(uint64_t na, const uint64_t *pa, uint64_t *lhs,
                                                uint64_t *rhs) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na)
        return;
    lhs[i] = pa[i * 12ull + 3ull]; // old_timestamp[0]
    rhs[i] = pa[i * 12ull + 2ull]; // timestamp
}

extern "C" __global__ void memw_lt_emit_general(uint64_t ng, const uint64_t *pg,
                                                const uint64_t *excl_w, uint64_t na, uint64_t *lhs,
                                                uint64_t *rhs) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ng)
        return;
    uint64_t row = i * 19ull;
    uint32_t w = (uint32_t)((pg[row] >> 8) & 0xFFull);
    uint64_t ts = pg[row + 2ull];
    uint64_t off = na + excl_w[i];
    for (uint32_t j = 0; j < w; ++j) {
        lhs[off + j] = pg[row + 11ull + (uint64_t)j]; // old_timestamp[j]
        rhs[off + j] = ts;
    }
}
