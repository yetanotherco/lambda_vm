// On-GPU BITWISE multiplicity histogram (the prover's `BitwiseHistogram`).
//
// The counter array is `hist[type_idx * num_rows + row_index(x, y, z)]` (u64), with
// `row_index(x, y, z) = x + y*256 + z*65536`. This kernel emits the per-CPU-op in-walk
// range-check lookups and atomic-adds them — the device analog of the host
// `add_ops(collect_bitwise_ops(cpu_ops))`, which alone is ~half of all BITWISE bumps.
//
// Mirrors `CpuOperation::collect_bitwise_ops` bit-for-bit: 3 ARE_BYTES (type 3) + 4
// IS_HALF (type 4). Word (`*W`) instruction rows zero rs1/rs2/rd/alu_flags/mem_flags
// and res (CPU32 emits their real range checks); half_instruction_length is never
// zeroed. `row_index(x, y, 0) = x + y*256`, and an IS_HALF halfword `h` maps to
// `(h & 0xFF) + (h >> 8)*256 == h`.

#include <cstdint>

// `num_copies` REPLICATED histograms defuse atomic contention: the ARE_BYTES lookups
// key on register indices (~1K hot bins), so naive single-copy global atomics serialize.
// Each block accumulates into copy `blockIdx.x % num_copies` (stride `copy_stride =
// num_rows * num_types`), spreading each hot address across `num_copies` addresses;
// `bitwise_hist_reduce` then sums the copies. Concurrent blocks land on distinct copies.
extern "C" __global__ void bitwise_hist_cpu_ops(
    uint64_t n, const uint8_t *rs1, const uint8_t *rs2, const uint8_t *rd,
    const uint8_t *hil, const uint8_t *alu_flags, const uint8_t *mem_flags,
    const uint64_t *res, const uint8_t *word, uint64_t num_rows, uint32_t num_copies,
    uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    bool w = word[i] != 0;
    uint64_t z_rs1 = w ? 0u : rs1[i];
    uint64_t z_rs2 = w ? 0u : rs2[i];
    uint64_t z_rd = w ? 0u : rd[i];
    uint64_t z_alu = w ? 0u : alu_flags[i];
    uint64_t z_mem = w ? 0u : mem_flags[i];
    uint64_t h = hil[i]; // half_instruction_length — NOT zeroed on word rows
    uint64_t r = w ? 0ull : res[i];

    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t are = base + 3ull * num_rows; // AreBytes lane in this copy
    const uint64_t ish = base + 4ull * num_rows; // IsHalf lane in this copy

    atomicAdd(&hist[are + z_rs1 + z_rs2 * 256ull], 1ull);
    atomicAdd(&hist[are + z_rd + h * 256ull], 1ull);
    atomicAdd(&hist[are + z_alu + z_mem * 256ull], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t half = (r >> (k * 16)) & 0xFFFFull;
        atomicAdd(&hist[ish + half], 1ull);
    }
}

// RESIDENT in-walk source: identical bumps to `bitwise_hist_cpu_ops`, but reads the packed
// decode + `res` from the DEVICE-RESIDENT cpu_ops (no host SoA rebuild / upload). Unpacks
// rs1/rs2/rd/hil/alu_flags/mem_flags/word from `packed` on device — same bit layout as
// `build_cpu_ops` (PD_* offsets). This is the seam that lets the biggest BITWISE source run
// with zero host round-trip.
#define BH_PD_WORD_INSTR 3
#define BH_PD_RS1 10
#define BH_PD_RS2 18
#define BH_PD_RD 26
#define BH_PD_HIL 34
#define BH_PD_ALU_FLAGS 42
#define BH_PD_MEM_FLAGS 50
__device__ __forceinline__ bool bh_pd_bit(uint64_t pk, int b) { return (pk >> b) & 1ull; }
__device__ __forceinline__ uint64_t bh_pd_byte(uint64_t pk, int off) {
    return (pk >> off) & 0xFFull;
}

extern "C" __global__ void bitwise_hist_cpu_ops_packed(
    uint64_t n, const uint64_t *packed, const uint64_t *res, uint64_t num_rows,
    uint32_t num_copies, uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t pk = packed[i];
    bool w = bh_pd_bit(pk, BH_PD_WORD_INSTR);
    uint64_t z_rs1 = w ? 0ull : bh_pd_byte(pk, BH_PD_RS1);
    uint64_t z_rs2 = w ? 0ull : bh_pd_byte(pk, BH_PD_RS2);
    uint64_t z_rd = w ? 0ull : bh_pd_byte(pk, BH_PD_RD);
    uint64_t z_alu = w ? 0ull : bh_pd_byte(pk, BH_PD_ALU_FLAGS);
    uint64_t z_mem = w ? 0ull : bh_pd_byte(pk, BH_PD_MEM_FLAGS);
    uint64_t h = bh_pd_byte(pk, BH_PD_HIL); // half_instruction_length — NOT zeroed on word rows
    uint64_t r = w ? 0ull : res[i];

    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t are = base + 3ull * num_rows;
    const uint64_t ish = base + 4ull * num_rows;

    atomicAdd(&hist[are + z_rs1 + z_rs2 * 256ull], 1ull);
    atomicAdd(&hist[are + z_rd + h * 256ull], 1ull);
    atomicAdd(&hist[are + z_alu + z_mem * 256ull], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t half = (r >> (k * 16)) & 0xFFFFull;
        atomicAdd(&hist[ish + half], 1ull);
    }
}

// MEMW_R source: one IS_HALF lookup per register row, keyed by the timestamp delta
// `diff = ts_lo - old_ts_lo - 1` (u16). Mirrors `memw_register_is_half_lookup`; IS_HALF
// is type 4 and `row_index(diff&0xff, diff>>8, 0) == diff`. Accumulates into the same
// replicated histogram as `bitwise_hist_cpu_ops`.
extern "C" __global__ void bitwise_hist_memw_reg(uint64_t n, const uint64_t *ts,
                                                 const uint64_t *old_ts, uint64_t num_rows,
                                                 uint32_t num_copies, uint64_t copy_stride,
                                                 unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t ts_lo = (uint32_t)(ts[i] & 0xFFFFFFFFull);
    uint32_t ot_lo = (uint32_t)(old_ts[i] & 0xFFFFFFFFull);
    uint64_t diff = (uint64_t)((ts_lo - ot_lo - 1u) & 0xFFFFu);
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 4ull * num_rows + diff], 1ull);
}

// RESIDENT MEMW_R source (P3): one IS_HALF lookup per EMITTING register row, keyed by the ts delta
// `diff = ts_lo - old_ts_lo - 1` (u16) — the same as `bitwise_hist_memw_reg`, but reads the FULL
// device access stream (incl non-emitting PC writes) straight from the resident register walk and
// skips `row_index < 0` (non-emitting). No host round-trip: ts/old_ts/row_index are resident.
extern "C" __global__ void bitwise_hist_memw_reg_masked(uint64_t n, const uint64_t *ts,
                                                        const uint64_t *old_ts,
                                                        const int64_t *row_index, uint64_t num_rows,
                                                        uint32_t num_copies, uint64_t copy_stride,
                                                        unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (row_index[i] < 0)
        return; // non-emitting access (implicit PC write) — advances timeline, no row
    uint32_t ts_lo = (uint32_t)(ts[i] & 0xFFFFFFFFull);
    uint32_t ot_lo = (uint32_t)(old_ts[i] & 0xFFFFFFFFull);
    uint64_t diff = (uint64_t)((ts_lo - ot_lo - 1u) & 0xFFFFu);
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 4ull * num_rows + diff], 1ull);
}

// RESIDENT MEMW_R source WITH the `is_register_op` FALLBACK ROUTING: counts the IS_HALF ts-delta ONLY
// for emitting rows that stay in MEMW_R (ts_hi==old_ts_hi && ts_lo>old_ts_lo && ts_lo-old_ts_lo<=2^16
// — the `reg_ts_delta_in_range` predicate). Rows that fail it route to MEMW_A/MEMW on the host (a tiny
// ~0.002% remainder on ethrex), so excluding them here keeps the device memw_reg count exactly equal
// to the sequential `memw_register_rows` histogram. Reads the resident walk's ts/old_ts/row_index.
extern "C" __global__ void bitwise_hist_memw_reg_routed(uint64_t n, const uint64_t *ts,
                                                        const uint64_t *old_ts,
                                                        const int64_t *row_index, uint64_t num_rows,
                                                        uint32_t num_copies, uint64_t copy_stride,
                                                        unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (row_index[i] < 0)
        return; // non-emitting (PC write / injected ecall) — advances timeline, no row
    uint64_t t = ts[i], o = old_ts[i];
    uint32_t ts_lo = (uint32_t)(t & 0xFFFFFFFFull);
    uint32_t ot_lo = (uint32_t)(o & 0xFFFFFFFFull);
    uint32_t ts_hi = (uint32_t)(t >> 32);
    uint32_t ot_hi = (uint32_t)(o >> 32);
    // reg_ts_delta_in_range: upper limbs equal, lower strictly increasing, delta <= 2^16.
    if (!(ts_hi == ot_hi && ts_lo > ot_lo && (ts_lo - ot_lo) <= 0x10000u))
        return; // falls back to MEMW_A/MEMW (counted on host)
    uint64_t diff = (uint64_t)((ts_lo - ot_lo - 1u) & 0xFFFFu);
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 4ull * num_rows + diff], 1ull);
}

// OP-VEC source MEMW_ALIGNED (P4): one IS_HALF[base_address_low + mask] per ALIGNED memw op, where
// mask = width-1 for width∈{2,4,8} (else 0). Reads the resident op metadata (base, width) + the
// aligned classify flag from the resident memory walk. Mirrors `collect_bitwise_from_memw_aligned`.
extern "C" __global__ void bitwise_hist_memw_aligned(uint64_t n, const uint64_t *base,
                                                     const uint32_t *width, const uint32_t *aligned,
                                                     uint64_t num_rows, uint32_t num_copies,
                                                     uint64_t copy_stride,
                                                     unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (aligned[i] == 0)
        return;
    uint32_t w = width[i];
    uint32_t mask = (w == 2) ? 1u : (w == 4) ? 3u : (w == 8) ? 7u : 0u;
    uint64_t v = ((base[i] & 0xFFFFull) + (uint64_t)mask) & 0xFFFFull;
    uint64_t b = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[b + 4ull * num_rows + v], 1ull);
}

// PAGE source: one ARE_BYTES[init, fini] lookup per byte of every touched page (~10% of all
// bumps). Mirrors `collect_bitwise_from_page`'s per-byte bump. `init[i]`/`fini[i]` are the init
// (ELF/0) and final byte values; ARE_BYTES is type 3 and row_index(init, fini, 0) = init +
// fini*256. Accumulates into the same replicated histogram as the other sources.
extern "C" __global__ void bitwise_hist_page(uint64_t n, const uint8_t *init, const uint8_t *fini,
                                             uint64_t num_rows, uint32_t num_copies,
                                             uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 3ull * num_rows + (uint64_t)init[i] + (uint64_t)fini[i] * 256ull], 1ull);
}

// FR4a — RESIDENT PAGE source: the SAME ARE_BYTES[init, fini] per-byte bumps as `bitwise_hist_page`,
// but computed on-device from the sorted initial image + the device final-memory snapshot instead of
// host-built `(init, fini)` byte arrays (which cost ~1s to assemble via a HashMap over ~4.7M cells).
// One thread per byte of every page in `page_bases` (num_pages * page_size threads):
//   addr = page_bases[i / page_size] + (i % page_size)
//   init = image byte at addr (binary search over sorted `img_addr`/`img_val`, 0 if absent)
//   fini = snapshot byte at addr (binary search over sorted `snap_addr`/`snap_val`) else init
// This reproduces `build_page_bitwise_arrays` bit-for-bit: page_bases = pages(image ∪ snapshot); a
// touched byte's fini = snapshot value, an untouched byte's fini = its init (image or 0). Both search
// arrays are ascending (image sorted at upload; snapshot is the address-sorted walk output).
__device__ __forceinline__ uint64_t bh_bsearch(const uint64_t *keys, const uint64_t *vals,
                                               uint64_t n, uint64_t key, uint64_t dflt) {
    uint64_t lo = 0, hi = n, out = dflt;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t m = keys[mid];
        if (m == key) { out = vals[mid]; break; }
        else if (m < key) { lo = mid + 1; }
        else { hi = mid; }
    }
    return out;
}
extern "C" __global__ void bitwise_hist_page_snapshot(
    const uint64_t *page_bases, uint64_t num_pages, uint64_t page_size,
    const uint64_t *img_addr, const uint64_t *img_val, uint64_t img_n,
    const uint64_t *snap_addr, const uint64_t *snap_val, uint64_t snap_n,
    uint64_t num_rows, uint32_t num_copies, uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = num_pages * page_size;
    if (i >= total)
        return;
    uint64_t addr = page_bases[i / page_size] + (i % page_size);
    uint64_t iv = bh_bsearch(img_addr, img_val, img_n, addr, 0ull) & 0xFFull;
    uint64_t fv = bh_bsearch(snap_addr, snap_val, snap_n, addr, iv) & 0xFFull;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 3ull * num_rows + iv + fv * 256ull], 1ull);
}

// ---------------------------------------------------------------------------
// OP-VEC sources (P4a): each mirrors a CPU `collect_bitwise_from_X` / per-op
// `collect_bitwise_ops()` bit-for-bit, reading a chip's resident op vector.
// Lookup-type lanes (dense index): Msb8=0, Msb16=1, Zero=2, AreBytes=3,
// IsHalf=4, IsB20=5, Hwsl=6, ByteAluAnd=7, ByteAluOr=8, ByteAluXor=9.
// row_index(x,y,z) = x + y*256 + z*65536; a halfword `h` maps to `h` (x=h&0xff,
// y=h>>8, z=0); a single byte `b` to `b`; byte_op(a,b) to `a + b*256`.
// ---------------------------------------------------------------------------

// LT op-vec: per (lhs,rhs) op → 2 Msb16 (lhs[2], rhs[2]) + 4 IS_HALF over
// (lhs-rhs) halves + 2 IS_HALF (lhs[1], rhs[1]). Mirrors `collect_bitwise_from_lt`.
extern "C" __global__ void bitwise_hist_lt(uint64_t n, const uint64_t *lhs,
                                           const uint64_t *rhs, uint64_t num_rows,
                                           uint32_t num_copies, uint64_t copy_stride,
                                           unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t l = lhs[i], r = rhs[i];
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t msb16 = base + 1ull * num_rows;
    const uint64_t ish = base + 4ull * num_rows;
    atomicAdd(&hist[msb16 + ((l >> 48) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[msb16 + ((r >> 48) & 0xFFFFull)], 1ull);
    uint64_t sub = l - r;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        atomicAdd(&hist[ish + ((sub >> (k * 16)) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + ((l >> 32) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + ((r >> 32) & 0xFFFFull)], 1ull);
}

// STORE op-vec: per op → 8 ARE_BYTES (one per byte of `value`). Mirrors
// `StoreOperation::collect_bitwise_ops`.
extern "C" __global__ void bitwise_hist_store(uint64_t n, const uint64_t *value,
                                              uint64_t num_rows, uint32_t num_copies,
                                              uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t v = value[i];
    uint64_t are = (uint64_t)(blockIdx.x % num_copies) * copy_stride + 3ull * num_rows;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        atomicAdd(&hist[are + ((v >> (k * 8)) & 0xFFull)], 1ull);
}

// BYTEWISE op-vec: per op → 8 BYTE_ALU[kind] (one per byte pair), kind from
// `op` (AND=0→ByteAluAnd/7, OR=1→8, XOR=2→9). Mirrors `BytewiseOperation::collect_bitwise_ops`.
extern "C" __global__ void bitwise_hist_bytewise(uint64_t n, const uint64_t *a,
                                                 const uint64_t *b, const uint8_t *op,
                                                 uint64_t num_rows, uint32_t num_copies,
                                                 uint64_t copy_stride,
                                                 unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t av = a[i], bv = b[i];
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    uint64_t lane = base + (7ull + (uint64_t)op[i]) * num_rows;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        uint64_t x = (av >> (k * 8)) & 0xFFull;
        uint64_t y = (bv >> (k * 8)) & 0xFFull;
        atomicAdd(&hist[lane + x + y * 256ull], 1ull);
    }
}

// EQ op-vec: per (a,b) op → 4 IS_HALF over (a-b) halves + 1 ZERO[Σ halves].
// Mirrors `EqOperation::collect_bitwise_ops`.
extern "C" __global__ void bitwise_hist_eq(uint64_t n, const uint64_t *a, const uint64_t *b,
                                           uint64_t num_rows, uint32_t num_copies,
                                           uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t diff = a[i] - b[i];
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t ish = base + 4ull * num_rows;
    uint64_t sum = 0ull;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t half = (diff >> (k * 16)) & 0xFFFFull;
        sum += half;
        atomicAdd(&hist[ish + half], 1ull);
    }
    atomicAdd(&hist[base + 2ull * num_rows + sum], 1ull); // ZERO lane; sum < 2^18
}

// LOAD op-vec: per op → 1 MSB8[res[byte_idx]] when width∈{1,2,4} (skip width 8),
// byte_idx = 0/1/3. `res` is 8 u64 per op (byte value per limb). Mirrors
// `LoadOperation::collect_bitwise_ops`.
extern "C" __global__ void bitwise_hist_load(uint64_t n, const uint64_t *res,
                                             const uint32_t *width, uint64_t num_rows,
                                             uint32_t num_copies, uint64_t copy_stride,
                                             unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t w = width[i];
    int idx = (w == 1) ? 0 : (w == 2) ? 1 : (w == 4) ? 3 : -1;
    if (idx < 0)
        return;
    uint64_t byte = res[i * 8 + idx] & 0xFFull;
    uint64_t msb8 = (uint64_t)(blockIdx.x % num_copies) * copy_stride; // Msb8 lane = 0*num_rows
    atomicAdd(&hist[msb8 + byte], 1ull);
}

// FR4b — RESIDENT op-vec (STORE + EQ + BYTEWISE) from the resident cpu_ops: one thread per cpu_op,
// self-filter by the packed decode, read operands from the resident rv1/rv2/arg2 buffers (NO host SoA
// build/upload). Mirrors the host `store`/`eq`/`bytewise` op-vec sources bit-for-bit (same bump logic
// as `bitwise_hist_store`/`_eq`/`_bytewise`). Decode bits: WORD_INSTR=3, ALU=4, MEMORY=7,
// ALU_FLAGS byte@42 (alu_op = low 5 bits: AND0 OR1 XOR2 EQ3), MEM_FLAGS byte@50 (bit0=store).
// STORE value = rv2; EQ (a,b)=(rv1,arg2); BYTEWISE (a,b,op)=(rv1,arg2,alu_op). Branch EQ ops (ALU∧EQ,
// arg2=rv2) are included exactly as the host `is_eq` filter does.
extern "C" __global__ void bitwise_hist_opvec_packed(
    uint64_t n, const uint64_t *packed, const uint64_t *rv1, const uint64_t *rv2,
    const uint64_t *arg2, uint64_t num_rows, uint32_t num_copies, uint64_t copy_stride,
    unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool word = (pk >> 3) & 1ull;
    bool alu = (pk >> 4) & 1ull;
    bool memory = (pk >> 7) & 1ull;
    uint64_t mem_flags = (pk >> 50) & 0xFFull;
    uint64_t alu_op = ((pk >> 42) & 0xFFull) & 0x1Full;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;

    // STORE: 8 ARE_BYTES on rv2 (is_store = memory ∧ mem_flags bit0).
    if (memory && (mem_flags & 1ull)) {
        uint64_t v = rv2[i];
        uint64_t are = base + 3ull * num_rows;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            atomicAdd(&hist[are + ((v >> (k * 8)) & 0xFFull)], 1ull);
    }

    if (!word && alu) {
        if (alu_op == 3ull) {
            // EQ: 4 IS_HALF over (rv1 - arg2) halves + 1 ZERO[Σ].
            uint64_t diff = rv1[i] - arg2[i];
            const uint64_t ish = base + 4ull * num_rows;
            uint64_t sum = 0ull;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint64_t half = (diff >> (k * 16)) & 0xFFFFull;
                sum += half;
                atomicAdd(&hist[ish + half], 1ull);
            }
            atomicAdd(&hist[base + 2ull * num_rows + sum], 1ull); // ZERO lane
        } else if (alu_op <= 2ull) {
            // BYTEWISE: 8 BYTE_ALU[7+op] on (rv1, arg2) byte pairs (AND0/OR1/XOR2).
            uint64_t av = rv1[i], bv = arg2[i];
            uint64_t lane = base + (7ull + alu_op) * num_rows;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint64_t x = (av >> (k * 8)) & 0xFFull;
                uint64_t y = (bv >> (k * 8)) & 0xFFull;
                atomicAdd(&hist[lane + x + y * 256ull], 1ull);
            }
        }
    }
}

// S3 — RESIDENT op-vec (BRANCH + LOAD) from the resident cpu_ops seam: one thread per cpu_op,
// self-route by the packed decode + branch_cond flag, read pc/imm/rv1/rvd from the resident buffers
// (NO host SoA build/upload). Mirrors the host op-vec sources bit-for-bit:
//   BRANCH (branch_cond = `flags` bit0): next_pc from (pc/imm/rv1/jalr) exactly as
//     `collect_bitwise_from_branch` — ARE_BYTES[next_pc>>8] + ByteAluAnd[unmasked&0xff,254] + 3 IS_HALF
//     (next_pc high halfwords). jalr = mem_flags bit0; base = jalr?rv1:pc; unmasked = base+imm; npc = unmasked&~1.
//   LOAD (is_load = memory(bit7) ∧ !(mem_flags bit0)): 1 Msb8 of the loaded value's top significant
//     byte, exactly as `LoadOperation::collect_bitwise_ops` (skip width 8). loaded value = rvd; byte
//     index = width-1 (1→0, 2→1, 4→3); width from mem_flags bits 2/3/4.
extern "C" __global__ void bitwise_hist_branch_load_packed(
    uint64_t n, const uint64_t *packed, const uint8_t *flags, const uint64_t *pc,
    const uint64_t *imm, const uint64_t *rv1, const uint64_t *rvd, uint64_t num_rows,
    uint32_t num_copies, uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    uint64_t mem_flags = (pk >> 50) & 0xFFull;

    // BRANCH: branch_cond = flags bit0. Emit exactly `collect_bitwise_from_branch`.
    if (flags[i] & 1u) {
        bool jalr = (mem_flags & 1ull) != 0ull;
        uint64_t b = jalr ? rv1[i] : pc[i];
        uint64_t unmasked = b + imm[i]; // u64 wrapping add (offset already sign-extended)
        uint64_t npc = unmasked & ~1ull;
        atomicAdd(&hist[base + 3ull * num_rows + ((npc >> 8) & 0xFFull)], 1ull);           // ARE_BYTES
        atomicAdd(&hist[base + 7ull * num_rows + (unmasked & 0xFFull) + 254ull * 256ull], 1ull); // ByteAluAnd
        const uint64_t ish = base + 4ull * num_rows;
        atomicAdd(&hist[ish + ((npc >> 16) & 0xFFFFull)], 1ull);
        atomicAdd(&hist[ish + ((npc >> 32) & 0xFFFFull)], 1ull);
        atomicAdd(&hist[ish + ((npc >> 48) & 0xFFFFull)], 1ull);
    }

    // LOAD: Msb8 of the loaded value's top significant byte (width 1/2/4; skip width 8).
    bool memory = (pk >> 7) & 1ull;
    if (memory && ((mem_flags & 1ull) == 0ull)) {
        int byte_idx;
        if ((mem_flags >> 4) & 1ull)
            byte_idx = -1; // 8B: no sign extension → no lookup
        else if ((mem_flags >> 3) & 1ull)
            byte_idx = 3; // 4B
        else if ((mem_flags >> 2) & 1ull)
            byte_idx = 1; // 2B
        else
            byte_idx = 0; // 1B
        if (byte_idx >= 0) {
            uint64_t byte = (rvd[i] >> (byte_idx * 8)) & 0xFFull;
            atomicAdd(&hist[base + byte], 1ull); // Msb8 lane = 0*num_rows
        }
    }
}

// CPU32 op-vec: per op → 5 ARE_BYTES (hil,alu_flags,rs1,rs2,rd) + 8 IS_HALF
// (rv1[0..1], rv2[0..1], res[0..3]) + 1 BYTE_ALU[AND,32,alu_flags] +
// (signed? 2 MSB16: rv1[1], rv2[1]) + 1 MSB16 res[1]. signed = alu_flags bit5.
// Mirrors `collect_cpu32_bitwise`.
extern "C" __global__ void bitwise_hist_cpu32(
    uint64_t n, const uint8_t *hil, const uint8_t *alu_flags, const uint8_t *rs1,
    const uint8_t *rs2, const uint8_t *rd, const uint64_t *rv1, const uint64_t *rv2,
    const uint64_t *res, uint64_t num_rows, uint32_t num_copies, uint64_t copy_stride,
    unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t are = base + 3ull * num_rows;
    const uint64_t ish = base + 4ull * num_rows;
    const uint64_t and_ = base + 7ull * num_rows;
    const uint64_t msb16 = base + 1ull * num_rows;
    uint64_t af = alu_flags[i];
    atomicAdd(&hist[are + (uint64_t)hil[i]], 1ull);
    atomicAdd(&hist[are + af], 1ull);
    atomicAdd(&hist[are + (uint64_t)rs1[i]], 1ull);
    atomicAdd(&hist[are + (uint64_t)rs2[i]], 1ull);
    atomicAdd(&hist[are + (uint64_t)rd[i]], 1ull);
    uint64_t v1 = rv1[i], v2 = rv2[i], rr = res[i];
    uint64_t v1h1 = (v1 >> 16) & 0xFFFFull, v2h1 = (v2 >> 16) & 0xFFFFull;
    atomicAdd(&hist[ish + (v1 & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + v1h1], 1ull);
    atomicAdd(&hist[ish + (v2 & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + v2h1], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k)
        atomicAdd(&hist[ish + ((rr >> (k * 16)) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[and_ + 32ull + af * 256ull], 1ull);
    if (af & 0x20ull) {
        atomicAdd(&hist[msb16 + v1h1], 1ull);
        atomicAdd(&hist[msb16 + v2h1], 1ull);
    }
    atomicAdd(&hist[msb16 + ((rr >> 16) & 0xFFFFull)], 1ull);
}

// CPU32 op-vec from the PACKED device op rows (`pack_cpu32_op`: [ts,pc,rv1,rv2,imm,res,flags,bytes];
// bytes = rs1@0 rs2@8 rd@16 alu_flags@24 hil@32). Same bumps as `bitwise_hist_cpu32` but reads the
// device-built rows (res already computed on device by `build_cpu32_ops`, validated == build_cpu32_op),
// so the histogram reuses the resident CPU32 op-build with no host SoA. `n_ops` = compacted word rows.
extern "C" __global__ void bitwise_hist_cpu32_packed(
    uint64_t n_ops, const uint64_t *rows, uint64_t num_rows, uint32_t num_copies,
    uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ops)
        return;
    const uint64_t *r = rows + i * 8;
    uint64_t v1 = r[2], v2 = r[3], rr = r[5], bytes = r[7];
    uint64_t rs1 = bytes & 0xFFull, rs2 = (bytes >> 8) & 0xFFull, rd = (bytes >> 16) & 0xFFull;
    uint64_t af = (bytes >> 24) & 0xFFull, hil = (bytes >> 32) & 0xFFull;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t are = base + 3ull * num_rows;
    const uint64_t ish = base + 4ull * num_rows;
    const uint64_t and_ = base + 7ull * num_rows;
    const uint64_t msb16 = base + 1ull * num_rows;
    atomicAdd(&hist[are + hil], 1ull);
    atomicAdd(&hist[are + af], 1ull);
    atomicAdd(&hist[are + rs1], 1ull);
    atomicAdd(&hist[are + rs2], 1ull);
    atomicAdd(&hist[are + rd], 1ull);
    uint64_t v1h1 = (v1 >> 16) & 0xFFFFull, v2h1 = (v2 >> 16) & 0xFFFFull;
    atomicAdd(&hist[ish + (v1 & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + v1h1], 1ull);
    atomicAdd(&hist[ish + (v2 & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + v2h1], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k)
        atomicAdd(&hist[ish + ((rr >> (k * 16)) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[and_ + 32ull + af * 256ull], 1ull);
    if (af & 0x20ull) {
        atomicAdd(&hist[msb16 + v1h1], 1ull);
        atomicAdd(&hist[msb16 + v2h1], 1ull);
    }
    atomicAdd(&hist[msb16 + ((rr >> 16) & 0xFFFFull)], 1ull);
}

// BRANCH op-vec: per op → ARE_BYTES[next_pc[1..2 byte]] + BYTE_ALU[AND,
// next_pc_unmasked&0xff, 254] + 3 IS_HALF (next_pc high halfwords 0..2).
// Mirrors `collect_bitwise_from_branch`; `next_pc`/`next_pc_unmasked` precomputed
// (device branch op-gen computes them today).
extern "C" __global__ void bitwise_hist_branch(uint64_t n, const uint64_t *next_pc,
                                               const uint64_t *next_pc_unmasked,
                                               uint64_t num_rows, uint32_t num_copies,
                                               uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t npc = next_pc[i], unmasked = next_pc_unmasked[i];
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 3ull * num_rows + ((npc >> 8) & 0xFFull)], 1ull);           // ARE_BYTES
    atomicAdd(&hist[base + 7ull * num_rows + (unmasked & 0xFFull) + 254ull * 256ull], 1ull); // ByteAluAnd
    const uint64_t ish = base + 4ull * num_rows;
    atomicAdd(&hist[ish + ((npc >> 16) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + ((npc >> 32) & 0xFFFFull)], 1ull);
    atomicAdd(&hist[ish + ((npc >> 48) & 0xFFFFull)], 1ull);
}

// SHIFT op-vec (no dedup, μ=1/op): recomputes compute_aux (is_negative/bit_shift/zbs,
// bit-for-bit from `shift_fill`) then emits `collect_bitwise_from_shift`'s bumps:
// C14 Msb16[in[3]]|signed, C1/C2 BYTE_ALU[AND,·,15] (left/right), C3 ZERO[bit_shift],
// C4/C7 5×HWSL (unless zbs), C11 BYTE_ALU[AND,shift,mask], + ARE_BYTES×2 + IS_HALF×5.
// `value` packs in_halves; `flags` bit0=direction(right), bit1=signed, bit2=word_instr.
// Shared SHIFT bump emitter (compute_aux + `collect_bitwise_from_shift`), called by both the SoA
// `bitwise_hist_shift` and the packed `bitwise_hist_shift_packed`. `v` packs in_halves, `shift` =
// shift_amount low byte, `sa` = full shift_amount, `fl` bit0=direction(right)/bit1=signed/bit2=word.
__device__ __forceinline__ void bh_shift_emit(uint64_t v, uint32_t shift, uint64_t sa, uint32_t fl,
                                              uint64_t base, uint64_t num_rows,
                                              unsigned long long *hist) {
    uint32_t direction = fl & 1u, is_signed = (fl >> 1) & 1u, word_instr = (fl >> 2) & 1u;
    uint32_t left = 1u - direction, right = direction;
    uint16_t in_h[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
        in_h[k] = (uint16_t)((v >> (k * 16)) & 0xFFFFull);
    uint32_t is_negative = (is_signed && ((in_h[3] >> 15) & 1u)) ? 1u : 0u;
    uint16_t extension = is_negative ? (uint16_t)0xFFFFu : (uint16_t)0u;
    uint8_t bit_shift = left ? (uint8_t)(shift & 15u) : (uint8_t)((256u - shift) & 15u);
    uint32_t zbs = (bit_shift == 0u) ? 1u : 0u;

    const uint64_t msb16 = base + 1ull * num_rows;
    const uint64_t zero = base + 2ull * num_rows;
    const uint64_t are = base + 3ull * num_rows;
    const uint64_t ish = base + 4ull * num_rows;
    const uint64_t hwsl = base + 6ull * num_rows;
    const uint64_t and_ = base + 7ull * num_rows;

    if (is_signed)
        atomicAdd(&hist[msb16 + (uint64_t)in_h[3]], 1ull); // C14
    if (left)
        atomicAdd(&hist[and_ + (uint64_t)shift + 15ull * 256ull], 1ull); // C1
    if (right) {
        uint32_t complement = (256u - (zbs ? 16u : 0u) - shift) & 0xFFu; // C2
        atomicAdd(&hist[and_ + (uint64_t)complement + 15ull * 256ull], 1ull);
    }
    atomicAdd(&hist[zero + (uint64_t)bit_shift], 1ull); // C3
    if (!zbs) {                                         // C4/C7: HWSL, z=bit_shift
        uint64_t zshift = (uint64_t)bit_shift * 65536ull;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            atomicAdd(&hist[hwsl + (uint64_t)in_h[k] + zshift], 1ull);
        atomicAdd(&hist[hwsl + (uint64_t)extension + zshift], 1ull);
    }
    uint64_t mask = word_instr ? 16ull : 48ull; // C11
    atomicAdd(&hist[and_ + (uint64_t)shift + mask * 256ull], 1ull);
    atomicAdd(&hist[are + ((sa >> 8) & 0xFFull)], 1ull);  // ARE_BYTES[shift_amount[1]]
    atomicAdd(&hist[are + (uint64_t)shift], 1ull);        // ARE_BYTES[shift[0]]
    atomicAdd(&hist[ish + ((sa >> 16) & 0xFFFFull)], 1ull); // IS_HALF[shift_amount high half]
#pragma unroll
    for (int k = 0; k < 4; ++k)
        atomicAdd(&hist[ish + (uint64_t)in_h[k]], 1ull); // IS_HALF[in[k]]
}

extern "C" __global__ void bitwise_hist_shift(uint64_t n, const uint64_t *value,
                                              const uint8_t *shift_in, const uint64_t *shift_amount,
                                              const uint32_t *flags, uint64_t num_rows,
                                              uint32_t num_copies, uint64_t copy_stride,
                                              unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    bh_shift_emit(value[i], (uint32_t)shift_in[i], shift_amount[i], flags[i], base, num_rows, hist);
}

// SHIFT op-vec from the PACKED device op rows (`build_shift_ops`/`cpu32_shift_ops`: 3 u64/op =
// [value, shift_amount, flags]). shift = shift_amount low byte (ShiftOperation::new). Same bumps as
// `bitwise_hist_shift` via the shared emitter — the histogram reuses the resident SHIFT op-build
// (instruction ⊕ cpu32-derived) with no host SoA. `n_ops` = compacted shift rows.
extern "C" __global__ void bitwise_hist_shift_packed(uint64_t n_ops, const uint64_t *rows,
                                                     uint64_t num_rows, uint32_t num_copies,
                                                     uint64_t copy_stride,
                                                     unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ops)
        return;
    const uint64_t *o = rows + i * 3ull;
    uint64_t sa = o[1];
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    bh_shift_emit(o[0], (uint32_t)(sa & 0xFFull), sa, (uint32_t)o[2], base, num_rows, hist);
}

// MUL op-vec PER-OP part (one set per raw op, no chunk dedup): 16 IS_HALF (lhs/rhs input +
// lo/hi output halves) + 4 IS_B20 (carry range checks). Reuses the `mul_fill` 128-bit product
// + raw-product math and `mul::compute_carries`. `flags` bit0=lhs_signed, bit1=rhs_signed.
// The per-chunk-deduped MSB16 (signed sign bits) is NOT emitted here — it rides on the deduped
// MUL table rows in P4b assembly. Bit-identical to `collect_bitwise_from_mul` for UNSIGNED ops
// (where the MSB16 dedup contributes nothing).
// Shared MUL per-op emitter (16 IS_HALF + 4 IS_B20), called by both the SoA `bitwise_hist_mul_perop`
// and the packed `bitwise_hist_mul_perop_packed`. `fl` bit0=lhs_signed, bit1=rhs_signed.
__device__ __forceinline__ void bh_mul_perop_emit(uint64_t L, uint64_t R, uint32_t fl,
                                                  uint64_t base, uint64_t num_rows,
                                                  unsigned long long *hist) {
    uint32_t ls = fl & 1u, rs = (fl >> 1) & 1u;
    __int128 a = ls ? (__int128)(int64_t)L : (__int128)L;
    __int128 b = rs ? (__int128)(int64_t)R : (__int128)R;
    __int128 product = a * b;
    uint64_t lo = (uint64_t)product;
    uint64_t hi = (uint64_t)((unsigned __int128)product >> 64);

    uint64_t lhs_is_neg = (ls && ((int64_t)L < 0)) ? 1u : 0u;
    uint64_t rhs_is_neg = (rs && ((int64_t)R < 0)) ? 1u : 0u;
    uint64_t le[8], re[8];
    for (int j = 0; j < 4; ++j) {
        le[j] = (L >> (16 * j)) & 0xFFFFull;
        re[j] = (R >> (16 * j)) & 0xFFFFull;
    }
    for (int j = 4; j < 8; ++j) {
        le[j] = lhs_is_neg ? 0xFFFFull : 0ull;
        re[j] = rhs_is_neg ? 0xFFFFull : 0ull;
    }
    uint64_t raw[4];
    for (int k = 0; k < 4; ++k) {
        unsigned __int128 sum = 0;
        for (int c = 0; c <= 1; ++c) {
            int idx = 2 * k + c;
            if (idx < 8) {
                for (int j = 0; j <= idx; ++j) {
                    if (j < 8 && (idx - j) < 8) {
                        unsigned __int128 term = (unsigned __int128)le[j] * (unsigned __int128)re[idx - j];
                        sum += term << (16 * c);
                    }
                }
            }
        }
        raw[k] = (uint64_t)sum;
    }
    uint64_t res4[4] = {lo & 0xFFFFFFFFull, lo >> 32, hi & 0xFFFFFFFFull, hi >> 32};
    uint64_t carries[4];
    carries[0] = (raw[0] - res4[0]) >> 32;
    for (int k = 1; k < 4; ++k)
        carries[k] = (raw[k] + carries[k - 1] - res4[k]) >> 32;

    const uint64_t ish = base + 4ull * num_rows;
    const uint64_t isb20 = base + 5ull * num_rows;
    uint64_t words[4] = {L, R, lo, hi};
#pragma unroll
    for (int w = 0; w < 4; ++w)
        for (int k = 0; k < 4; ++k)
            atomicAdd(&hist[ish + ((words[w] >> (k * 16)) & 0xFFFFull)], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t c = carries[k];
        uint64_t row = (c & 0xFFull) + ((c >> 8) & 0xFFull) * 256ull + ((c >> 16) & 0xFull) * 65536ull;
        atomicAdd(&hist[isb20 + row], 1ull);
    }
}

extern "C" __global__ void bitwise_hist_mul_perop(uint64_t n, const uint64_t *lhs,
                                                  const uint64_t *rhs, const uint32_t *flags,
                                                  uint64_t num_rows, uint32_t num_copies,
                                                  uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    bh_mul_perop_emit(lhs[i], rhs[i], flags[i], base, num_rows, hist);
}

// MUL per-op from the MERGED device key stream (`k0`=flags lhs_signed|rhs_signed<<1, `k1`=lhs,
// `k2`=rhs) that `mul_full_resident_core` builds pre-dedup from all 4 sources (instruction ⊕
// instruction-dvrm→mul ⊕ cpu32 ⊕ cpu32-dvrm→mul). Same bumps as `bitwise_hist_mul_perop` via the
// shared emitter — the histogram reuses the resident MUL key gather with no host SoA. `n_ops` = total.
extern "C" __global__ void bitwise_hist_mul_perop_packed(uint64_t n_ops, const uint64_t *k0,
                                                         const uint64_t *k1, const uint64_t *k2,
                                                         uint64_t num_rows, uint32_t num_copies,
                                                         uint64_t copy_stride,
                                                         unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ops)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    bh_mul_perop_emit(k1[i], k2[i], (uint32_t)k0[i], base, num_rows, hist);
}

// (unused legacy body retained below for reference during refactor; compiled out)
#if 0
extern "C" __global__ void bitwise_hist_mul_perop_OLD(uint64_t n, const uint64_t *lhs,
                                                  const uint64_t *rhs, const uint32_t *flags,
                                                  uint64_t num_rows, uint32_t num_copies,
                                                  uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t L = lhs[i], R = rhs[i];
    uint32_t fl = flags[i];
    uint32_t ls = fl & 1u, rs = (fl >> 1) & 1u;
    __int128 a = ls ? (__int128)(int64_t)L : (__int128)L;
    __int128 b = rs ? (__int128)(int64_t)R : (__int128)R;
    __int128 product = a * b;
    uint64_t lo = (uint64_t)product;
    uint64_t hi = (uint64_t)((unsigned __int128)product >> 64);

    uint64_t lhs_is_neg = (ls && ((int64_t)L < 0)) ? 1u : 0u;
    uint64_t rhs_is_neg = (rs && ((int64_t)R < 0)) ? 1u : 0u;
    uint64_t le[8], re[8];
    for (int j = 0; j < 4; ++j) {
        le[j] = (L >> (16 * j)) & 0xFFFFull;
        re[j] = (R >> (16 * j)) & 0xFFFFull;
    }
    for (int j = 4; j < 8; ++j) {
        le[j] = lhs_is_neg ? 0xFFFFull : 0ull;
        re[j] = rhs_is_neg ? 0xFFFFull : 0ull;
    }
    uint64_t raw[4];
    for (int k = 0; k < 4; ++k) {
        unsigned __int128 sum = 0;
        for (int c = 0; c <= 1; ++c) {
            int idx = 2 * k + c;
            if (idx < 8) {
                for (int j = 0; j <= idx; ++j) {
                    if (j < 8 && (idx - j) < 8) {
                        unsigned __int128 term = (unsigned __int128)le[j] * (unsigned __int128)re[idx - j];
                        sum += term << (16 * c);
                    }
                }
            }
        }
        raw[k] = (uint64_t)sum;
    }
    uint64_t res4[4] = {lo & 0xFFFFFFFFull, lo >> 32, hi & 0xFFFFFFFFull, hi >> 32};
    uint64_t carries[4];
    carries[0] = (raw[0] - res4[0]) >> 32;
    for (int k = 1; k < 4; ++k)
        carries[k] = (raw[k] + carries[k - 1] - res4[k]) >> 32;

    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t ish = base + 4ull * num_rows;
    const uint64_t isb20 = base + 5ull * num_rows;
    uint64_t words[4] = {L, R, lo, hi};
#pragma unroll
    for (int w = 0; w < 4; ++w)
        for (int k = 0; k < 4; ++k)
            atomicAdd(&hist[ish + ((words[w] >> (k * 16)) & 0xFFFFull)], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t c = carries[k];
        uint64_t row = (c & 0xFFull) + ((c >> 8) & 0xFFull) * 256ull + ((c >> 16) & 0xFull) * 65536ull;
        atomicAdd(&hist[isb20 + row], 1ull);
    }
}
#endif // legacy bitwise_hist_mul_perop_OLD

// DVRM op-vec PER-OP part (one set per raw op, no chunk dedup): 20 IS_HALF (n,d,r,n_sub_r,q
// halves) + 2 ZERO (C8 overflow_sum, C20 d_sum). Reuses the `dvrm_fill`/`cpu32_dvrm` quotient/
// remainder special cases. `flags` bit0=signed. The per-chunk-deduped MSB16 + NEG-template ZERO
// (signed only) are NOT emitted here — they ride on the deduped DVRM rows in P4b. Bit-identical
// to `collect_bitwise_from_dvrm` for UNSIGNED ops.
// Shared DVRM per-op emitter (20 IS_HALF + 2 ZERO), called by both the SoA `bitwise_hist_dvrm_perop`
// and the packed `bitwise_hist_dvrm_perop_packed`. `fl` bit0=signed.
__device__ __forceinline__ void bh_dvrm_perop_emit(uint64_t N, uint64_t D, uint32_t fl,
                                                   uint64_t base, uint64_t num_rows,
                                                   unsigned long long *hist) {
    uint32_t is_signed = fl & 1u;
    uint32_t div_by_zero = (D == 0ull) ? 1u : 0u;
    uint32_t overflow =
        (is_signed && N == 0x8000000000000000ull && D == 0xFFFFFFFFFFFFFFFFull) ? 1u : 0u;
    uint64_t q, r;
    if (div_by_zero) {
        q = 0xFFFFFFFFFFFFFFFFull;
        r = N;
    } else if (overflow) {
        q = N;
        r = 0ull;
    } else if (is_signed) {
        q = (uint64_t)((int64_t)N / (int64_t)D);
        r = (uint64_t)((int64_t)N % (int64_t)D);
    } else {
        q = N / D;
        r = N % D;
    }
    uint64_t n_sub_r = N - r;

    const uint64_t ish = base + 4ull * num_rows;
    const uint64_t zero = base + 2ull * num_rows;
    uint64_t words[5] = {N, D, r, n_sub_r, q};
#pragma unroll
    for (int w = 0; w < 5; ++w)
        for (int k = 0; k < 4; ++k)
            atomicAdd(&hist[ish + ((words[w] >> (k * 16)) & 0xFFFFull)], 1ull);

    // C8: ZERO[Σn_halves + 262141 - 32769*sign_n - Σd_halves]; C20: ZERO[Σd_halves].
    uint32_t sign_n = (is_signed && ((N >> 63) & 1u)) ? 1u : 0u;
    uint64_t nsum = 0, dsum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        nsum += (N >> (k * 16)) & 0xFFFFull;
        dsum += (D >> (k * 16)) & 0xFFFFull;
    }
    uint64_t overflow_sum = nsum + 262141ull - 32769ull * (uint64_t)sign_n - dsum;
    atomicAdd(&hist[zero + (overflow_sum & 0xFFFFFull)], 1ull);
    atomicAdd(&hist[zero + (dsum & 0xFFFFFull)], 1ull);
}

extern "C" __global__ void bitwise_hist_dvrm_perop(uint64_t n_ops, const uint64_t *nn,
                                                   const uint64_t *dd, const uint32_t *flags,
                                                   uint64_t num_rows, uint32_t num_copies,
                                                   uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ops)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    bh_dvrm_perop_emit(nn[i], dd[i], flags[i], base, num_rows, hist);
}

// DVRM per-op from the MERGED device key stream (`k0`=flags(signed), `k1`=n, `k2`=d) that
// `dvrm_full_resident_core` builds pre-dedup (instruction ⊕ cpu32-derived). Same bumps as
// `bitwise_hist_dvrm_perop` via the shared emitter. `n_ops` = total.
extern "C" __global__ void bitwise_hist_dvrm_perop_packed(uint64_t n_ops, const uint64_t *k0,
                                                          const uint64_t *k1, const uint64_t *k2,
                                                          uint64_t num_rows, uint32_t num_copies,
                                                          uint64_t copy_stride,
                                                          unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ops)
        return;
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    bh_dvrm_perop_emit(k1[i], k2[i], (uint32_t)k0[i], base, num_rows, hist);
}

// Sum the `num_copies` replicated histograms into `out[i] = Σ_r hist[r*copy_stride + i]`.
// One thread per counter bin (`copy_stride` = num_rows * num_types total bins).
extern "C" __global__ void bitwise_hist_reduce(uint64_t copy_stride, uint32_t num_copies,
                                               const unsigned long long *hist,
                                               unsigned long long *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= copy_stride)
        return;
    unsigned long long s = 0ull;
    for (uint32_t r = 0; r < num_copies; ++r)
        s += hist[(uint64_t)r * copy_stride + i];
    out[i] = s;
}
