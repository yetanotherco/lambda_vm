// On-GPU CPU trace-table fill: one thread per row, row-major output
// `out[row*NCOLS + col]` (the layout the device-input LDE seam consumes).
//
// Mirrors `prover/src/tables/cpu.rs::generate_cpu_trace`. The `CpuOperation`
// already carries the resolved values (rv1/rv2/arg2/res/next_pc/branch_cond),
// so the fill is pure bit-slicing — every column limb is < 2^32 (DWordWL lo/hi,
// DWordHL 16-bit halves, bytes, bools), so NO Goldilocks reduction is needed.
//
// Packed input, stride STRIDE u64 per op:
//   [0]=timestamp [1]=pc [2]=imm [3]=next_pc [4]=rvd [5]=rv1 [6]=rv2 [7]=arg2
//   [8]=res [9]=flags [10]=bytes
// flags bits: 0 word_instr, 1 read_register1(raw), 2 read_register2(raw),
//   3 write_register(raw), 4 alu, 5 add, 6 sub, 7 memory, 8 branch, 9 ecall,
//   10 branch_cond.
// bytes: b0 rs1, b1 rs2, b2 rd, b3 half_instruction_length, b4 alu_flags,
//   b5 mem_flags.
//
// The output buffer MUST be pre-zeroed (alloc_zeros); the kernel writes only the
// non-zero cells of real rows, and (TIMESTAMP, PC_0=1, NEXT_PC_0=1) for padding.

#include <cstdint>

#define NCOLS 38u
#define STRIDE 11u

extern "C" __global__ void trace_cpu_fill(const uint64_t *ops, // n * STRIDE
                                          uint64_t n,          // real op count
                                          uint64_t num_rows,   // padded pow2
                                          uint64_t last_ts,    // last real ts (0 if n==0)
                                          uint64_t *out)       // num_rows * NCOLS, zeroed
{
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows) return;
    uint64_t base = r * NCOLS;

    // Padding rows: continue the +4 timestamp cadence; PC_0 = NEXT_PC_0 = 1
    // (odd, unreachable); every other column stays zero.
    if (r >= n) {
        uint64_t j = r - n + 1;
        out[base + 0] = last_ts + 4 * j; // TIMESTAMP
        out[base + 1] = 1;               // PC_0
        out[base + 21] = 1;              // NEXT_PC_0
        return;
    }

    const uint64_t *op = ops + r * STRIDE;
    uint64_t ts = op[0];
    uint64_t pc = op[1];
    uint64_t imm = op[2];
    uint64_t npc = op[3];
    uint64_t rvd = op[4];
    uint64_t rv1 = op[5];
    uint64_t rv2 = op[6];
    uint64_t arg2 = op[7];
    uint64_t res = op[8];
    uint64_t fl = op[9];
    uint64_t by = op[10];

    uint64_t word = fl & 1u;
    uint64_t rr1 = (fl >> 1) & 1u;
    uint64_t rr2 = (fl >> 2) & 1u;
    uint64_t wr = (fl >> 3) & 1u;
    uint64_t alu = (fl >> 4) & 1u;
    uint64_t add = (fl >> 5) & 1u;
    uint64_t sub = (fl >> 6) & 1u;
    uint64_t mem = (fl >> 7) & 1u;
    uint64_t br = (fl >> 8) & 1u;
    uint64_t ec = (fl >> 9) & 1u;
    uint64_t bcond = (fl >> 10) & 1u;

    uint64_t rs1 = by & 0xFFu;
    uint64_t rs2 = (by >> 8) & 0xFFu;
    uint64_t rd = (by >> 16) & 0xFFu;
    uint64_t hil = (by >> 24) & 0xFFu;
    uint64_t aluf = (by >> 32) & 0xFFu;
    uint64_t memf = (by >> 40) & 0xFFu;

    // `effective(flag) = !word && flag`. Word-delegate rows suppress operational
    // data (CPU32 owns it) — only the PC-advancing columns are set.
    uint64_t nw = 1u - word; // !word
    uint64_t rs1c = word ? 0u : rs1;
    uint64_t rs2c = word ? 0u : rs2;
    uint64_t rdc = word ? 0u : rd;
    uint64_t immc = word ? 0u : imm;
    uint64_t rvdc = word ? 0u : rvd;
    uint64_t rv1c = word ? 0u : rv1;
    uint64_t rv2c = word ? 0u : rv2;
    uint64_t arg2c = word ? 0u : arg2;
    uint64_t resc = word ? 0u : res;

    out[base + 0] = ts;                        // TIMESTAMP
    out[base + 1] = pc & 0xFFFFFFFFu;          // PC_0
    out[base + 2] = pc >> 32;                  // PC_1
    out[base + 3] = rs1c;                      // RS1
    out[base + 4] = rs2c;                      // RS2
    out[base + 5] = rdc;                       // RD
    out[base + 6] = nw & rr1 & (rs1 != 0u);    // READ_REGISTER1 = eff(rr1 && rs1!=0)
    out[base + 7] = nw & rr2 & (rs2 != 0u);    // READ_REGISTER2
    out[base + 8] = nw & wr & (rd != 0u);      // WRITE_REGISTER
    out[base + 9] = immc & 0xFFFFFFFFu;        // IMM_0
    out[base + 10] = immc >> 32;               // IMM_1
    out[base + 11] = hil;                      // HALF_INSTRUCTION_LENGTH (unmasked)
    out[base + 12] = word;                     // WORD_INSTR
    out[base + 13] = nw & alu;                 // ALU
    out[base + 14] = word ? 0u : aluf;         // ALU_FLAGS
    out[base + 15] = nw & add;                 // ADD
    out[base + 16] = nw & sub;                 // SUB
    out[base + 17] = nw & mem;                 // MEMORY
    out[base + 18] = word ? 0u : memf;         // MEM_FLAGS
    out[base + 19] = nw & br;                  // BRANCH
    out[base + 20] = nw & ec;                  // ECALL
    out[base + 21] = npc & 0xFFFFFFFFu;        // NEXT_PC_0
    out[base + 22] = npc >> 32;                // NEXT_PC_1
    out[base + 23] = rvdc & 0xFFFFFFFFu;       // RVD_0
    out[base + 24] = rvdc >> 32;               // RVD_1

    // Inline-PC coordination columns.
    uint64_t pcdr = nw & rr1 & (rs1 == 255u); // PC_DOUBLE_READ
    uint64_t ts_lo = ts & 0xFFFFFFFFu;
    uint64_t borrow = (pcdr == 0u && ts_lo < 3u) ? 1u : 0u; // PREV_PC_TIMESTAMP_BORROW
    out[base + 25] = borrow;
    out[base + 26] = pcdr;
    out[base + 27] = rv1c & 0xFFFFFFFFu;       // RV1_0
    out[base + 28] = rv1c >> 32;               // RV1_1
    out[base + 29] = rv2c & 0xFFFFFFFFu;       // RV2_0
    out[base + 30] = rv2c >> 32;               // RV2_1
    out[base + 31] = arg2c & 0xFFFFFFFFu;      // ARG2_0
    out[base + 32] = arg2c >> 32;              // ARG2_1
    out[base + 33] = resc & 0xFFFFu;           // RES_0
    out[base + 34] = (resc >> 16) & 0xFFFFu;   // RES_1
    out[base + 35] = (resc >> 32) & 0xFFFFu;   // RES_2
    out[base + 36] = (resc >> 48) & 0xFFFFu;   // RES_3
    out[base + 37] = bcond;                    // BRANCH_COND (unmasked)
}

// On-GPU MEMW_A (aligned memory) trace fill: one thread per row, row-major
// `out[row*MEMW_A_NCOLS + col]`. Mirrors
// `prover/src/tables/memw_aligned.rs::generate_memw_aligned_trace`. The op is
// already walked (old_value/old_timestamp filled by the memory-model walk), so
// this is pure bit-slicing — every limb is < 2^32 (halves/words/bytes/bools), no
// Goldilocks reduction. Padding rows (r >= n) stay all-zero (pre-zeroed buffer).
//
// Packed input, stride MEMW_A_STRIDE u64 per op:
//   [0]=flags (bit0 is_register, bit1 is_read, bits8..16 width)
//   [1]=base_address [2]=timestamp [3]=old_timestamp[0]
//   [4..8]=value[0..8] packed 2×u32/u64 (value[2i] | value[2i+1]<<32)
//   [8..12]=old[0..8]  packed 2×u32/u64
#define MEMW_A_NCOLS 29u
#define MEMW_A_STRIDE 12u

extern "C" __global__ void memw_aligned_fill(const uint64_t *ops, uint64_t n,
                                             uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * MEMW_A_NCOLS;
    if (r >= n)
        return; // padding rows all-zero

    const uint64_t *op = ops + r * MEMW_A_STRIDE;
    uint64_t fl = op[0];
    uint64_t is_register = fl & 1u;
    uint64_t is_read = (fl >> 1) & 1u;
    uint64_t width = (fl >> 8) & 0xFFu;
    uint64_t addr = op[1];
    uint64_t ts = op[2];
    uint64_t old_ts = op[3];

    out[base + 0] = is_register;               // IS_REGISTER
    out[base + 1] = addr & 0xFFFFu;             // BASE_ADDRESS[0] (low half)
    out[base + 2] = (addr >> 16) & 0xFFFFu;     // BASE_ADDRESS[1] (mid half)
    out[base + 3] = (addr >> 32) & 0xFFFFFFFFu; // BASE_ADDRESS[2] (high word)
    // VALUE[0..8] at cols 4..11.
    for (int i = 0; i < 4; ++i) {
        uint64_t p = op[4 + i];
        out[base + 4 + 2 * i] = p & 0xFFFFFFFFu;
        out[base + 4 + 2 * i + 1] = p >> 32;
    }
    out[base + 12] = ts & 0xFFFFFFFFu;          // TIMESTAMP_0
    out[base + 13] = ts >> 32;                  // TIMESTAMP_1
    out[base + 14] = (width == 2u) ? 1u : 0u;   // WRITE2
    out[base + 15] = (width == 4u) ? 1u : 0u;   // WRITE4
    out[base + 16] = (width == 8u) ? 1u : 0u;   // WRITE8
    // OLD[0..8] at cols 17..24.
    for (int i = 0; i < 4; ++i) {
        uint64_t p = op[8 + i];
        out[base + 17 + 2 * i] = p & 0xFFFFFFFFu;
        out[base + 17 + 2 * i + 1] = p >> 32;
    }
    out[base + 25] = old_ts & 0xFFFFFFFFu;      // OLD_TIMESTAMP_0
    out[base + 26] = old_ts >> 32;              // OLD_TIMESTAMP_1
    out[base + 27] = is_read;                   // MU_READ
    out[base + 28] = 1u - is_read;              // MU_WRITE
}

// On-GPU LOAD trace fill (18 cols). Mirrors
// `prover/src/tables/load.rs::generate_load_trace`. Packed stride LOAD_STRIDE:
//   [0]=flags (bit0 signed, bits8..16 width) [1]=base_address [2]=timestamp
//   [3..7]=res[0..8] packed 2×u32/u64. Padding rows (r>=n) all-zero.
#define LOAD_NCOLS 18u
#define LOAD_STRIDE 7u

extern "C" __global__ void load_fill(const uint64_t *ops, uint64_t n,
                                     uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * LOAD_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * LOAD_STRIDE;
    uint64_t fl = op[0];
    uint64_t is_signed = fl & 1u;
    uint64_t width = (fl >> 8) & 0xFFu;
    uint64_t addr = op[1];
    uint64_t ts = op[2];

    out[base + 0] = addr & 0xFFFFFFFFu;       // BASE_ADDRESS_0
    out[base + 1] = addr >> 32;               // BASE_ADDRESS_1
    out[base + 2] = ts & 0xFFFFFFFFu;         // TIMESTAMP_0
    out[base + 3] = ts >> 32;                 // TIMESTAMP_1
    out[base + 4] = (width == 2u) ? 1u : 0u;  // READ2
    out[base + 5] = (width == 4u) ? 1u : 0u;  // READ4
    out[base + 6] = (width == 8u) ? 1u : 0u;  // READ8
    out[base + 7] = is_signed;                // SIGNED

    uint64_t res[8];
    for (int i = 0; i < 4; ++i) {
        uint64_t p = op[3 + i];
        res[2 * i] = p & 0xFFFFFFFFu;
        res[2 * i + 1] = p >> 32;
    }
    for (int i = 0; i < 8; ++i)
        out[base + 8 + i] = res[i];           // RES[0..8]

    int bidx = (width == 8u) ? 7 : (width == 4u) ? 3 : (width == 2u) ? 1 : 0;
    out[base + 16] = (res[bidx] >> 7) & 1u;   // SIGN_BIT
    out[base + 17] = 1u;                      // MU (active row)
}

// On-GPU STORE trace fill (16 cols). Mirrors
// `prover/src/tables/store.rs::generate_store_trace`. Packed stride STORE_STRIDE:
//   [0]=flags (bit0 write2, bit1 write4, bit2 write8) [1]=base_address
//   [2]=timestamp [3]=value. Padding rows (r>=n) all-zero.
#define STORE_NCOLS 16u
#define STORE_STRIDE 4u

extern "C" __global__ void store_fill(const uint64_t *ops, uint64_t n,
                                      uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * STORE_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * STORE_STRIDE;
    uint64_t fl = op[0];
    uint64_t addr = op[1];
    uint64_t ts = op[2];
    uint64_t val = op[3];

    out[base + 0] = addr & 0xFFFFFFFFu;  // BASE_ADDRESS_0
    out[base + 1] = addr >> 32;          // BASE_ADDRESS_1
    out[base + 2] = ts & 0xFFFFFFFFu;    // TIMESTAMP_0
    out[base + 3] = ts >> 32;            // TIMESTAMP_1
    out[base + 4] = fl & 1u;             // WRITE2
    out[base + 5] = (fl >> 1) & 1u;      // WRITE4
    out[base + 6] = (fl >> 2) & 1u;      // WRITE8
    for (int i = 0; i < 8; ++i)
        out[base + 7 + i] = (val >> (i * 8)) & 0xFFu; // VALUE[0..8] (DWordBL)
    out[base + 15] = 1u;                 // MU
}

// On-GPU SHIFT trace fill (29 cols). Mirrors
// `prover/src/tables/shift.rs::{generate_shift_trace, compute_aux}` — the fill
// RECOMPUTES the aux (bit_shift, zbs, x/y half decomposition, limb_shift one-hot,
// shifted DWordHL) from the compact input, so only 3 u64/op are uploaded. No
// dedup: μ = 1 per op. Padding rows (r >= n) set ZBS = 1 (all else 0).
//
// Packed stride SHIFT_STRIDE: [0]=value (4×u16 in_halves) [1]=shift_amount
//   [2]=flags (bit0 direction(=right), bit1 signed, bit2 word_instr).

// HWSL: (halfword << z) & 0xFFFF   (z in [0,15]).
__device__ __forceinline__ uint16_t hwsl_(uint16_t h, uint32_t z) {
    return z == 0u ? h : (uint16_t)((uint32_t)h << z);
}
// HWSL carry: halfword >> (16 - z).
__device__ __forceinline__ uint16_t hwslc_(uint16_t h, uint32_t z) {
    return z == 0u ? (uint16_t)0 : (uint16_t)(h >> (16u - z));
}

#define SHIFT_NCOLS 29u
#define SHIFT_STRIDE 3u

extern "C" __global__ void shift_fill(const uint64_t *ops, uint64_t n,
                                      uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * SHIFT_NCOLS;
    if (r >= n) {
        out[base + 12] = 1u; // ZBS = 1 on padding rows
        return;
    }

    const uint64_t *op = ops + r * SHIFT_STRIDE;
    uint64_t value = op[0];
    uint64_t shift_amount = op[1];
    uint64_t fl = op[2];
    uint32_t direction = (uint32_t)(fl & 1u); // right
    uint32_t is_signed = (uint32_t)((fl >> 1) & 1u);
    uint32_t word_instr = (uint32_t)((fl >> 2) & 1u);

    uint16_t in_h[4];
    in_h[0] = (uint16_t)(value & 0xFFFFu);
    in_h[1] = (uint16_t)((value >> 16) & 0xFFFFu);
    in_h[2] = (uint16_t)((value >> 32) & 0xFFFFu);
    in_h[3] = (uint16_t)((value >> 48) & 0xFFFFu);
    uint8_t shift = (uint8_t)(shift_amount & 0xFFu);
    uint32_t left = 1u - direction;
    uint32_t right = direction;

    uint32_t is_negative = (is_signed && ((in_h[3] >> 15) & 1u)) ? 1u : 0u;
    uint16_t extension = is_negative ? (uint16_t)0xFFFF : (uint16_t)0;

    uint8_t bit_shift;
    if (left)
        bit_shift = (uint8_t)(shift & 15u);
    else
        bit_shift = (uint8_t)((256u - (uint32_t)shift) & 15u);
    uint32_t zbs = (bit_shift == 0u) ? 1u : 0u;

    uint16_t x[5] = {0, 0, 0, 0, 0};
    uint16_t y[4] = {0, 0, 0, 0};
    if (zbs) {
        for (int i = 0; i < 4; ++i) {
            if (left)
                x[i] = in_h[i];
            else
                y[i] = in_h[i];
        }
        x[4] = 0;
    } else {
        for (int i = 0; i < 4; ++i) {
            x[i] = hwsl_(in_h[i], bit_shift);
            y[i] = hwslc_(in_h[i], bit_shift);
        }
        x[4] = hwsl_(extension, bit_shift);
    }

    // limb_shift: one-hot of (shift >> 4) & mask.
    uint32_t limb_idx = word_instr ? (uint32_t)((shift >> 4) & 1u)
                                   : (uint32_t)((shift >> 4) & 3u);
    uint32_t ls[4] = {0, 0, 0, 0};
    ls[limb_idx] = 1u;

    // shifted[i] (DWordHL). intra_left(k)=x[0] if k==0 else x[k]+y[k-1];
    // intra_right(k)=y[k]+x[k+1].
    uint16_t shifted[4];
    for (int i = 0; i < 4; ++i) {
        uint16_t v = 0;
        if (left) {
            for (int j = 0; j <= i; ++j)
                if (ls[j]) {
                    int k = i - j;
                    v = (uint16_t)(v + (k == 0 ? x[0] : (uint16_t)(x[k] + y[k - 1])));
                }
        }
        if (right) {
            for (int j = 0; j <= 3 - i; ++j)
                if (ls[j]) {
                    int k = i + j;
                    v = (uint16_t)(v + (uint16_t)(y[k] + x[k + 1]));
                }
            for (int j = 4 - i; j < 4; ++j)
                if (ls[j])
                    v = (uint16_t)(v + extension);
        }
        shifted[i] = v;
    }
    uint32_t out0 = (uint32_t)shifted[0] | ((uint32_t)shifted[1] << 16);
    uint32_t out1 = (uint32_t)shifted[2] | ((uint32_t)shifted[3] << 16);

    out[base + 0] = in_h[0];  // IN_0
    out[base + 1] = in_h[1];
    out[base + 2] = in_h[2];
    out[base + 3] = in_h[3];
    out[base + 4] = shift;                  // SHIFT_AMOUNT
    out[base + 5] = direction;              // DIRECTION
    out[base + 6] = is_signed;              // SIGNED
    out[base + 7] = word_instr;             // WORD_INSTR
    out[base + 8] = out0;                   // OUT_0
    out[base + 9] = out1;                   // OUT_1
    out[base + 10] = is_negative;           // IS_NEGATIVE
    out[base + 11] = bit_shift;             // BIT_SHIFT
    out[base + 12] = zbs;                   // ZBS
    out[base + 13] = x[0];                  // X_0..X_4
    out[base + 14] = x[1];
    out[base + 15] = x[2];
    out[base + 16] = x[3];
    out[base + 17] = x[4];
    out[base + 18] = y[0];                  // Y_0..Y_3
    out[base + 19] = y[1];
    out[base + 20] = y[2];
    out[base + 21] = y[3];
    out[base + 22] = ls[0];                 // LIMB_SHIFT_RAW_0..2
    out[base + 23] = ls[1];
    out[base + 24] = ls[2];
    out[base + 25] = 1u;                    // MU
    out[base + 26] = (shift_amount >> 8) & 0xFFu;      // SHIFT_B1
    out[base + 27] = (shift_amount >> 16) & 0xFFFFu;   // SHIFT_H1
    out[base + 28] = (shift_amount >> 32) & 0xFFFFFFFFu; // SHIFT_HIGH
}

// On-GPU LT trace fill (17 cols). Mirrors the per-row compute of
// `prover/src/tables/lt.rs::generate_lt_trace`. Dedup is done on the HOST (the
// same per-chunk HashMap the CPU uses); this kernel receives already-unique ops
// with their summed multiplicity, one per row, and recomputes lt/out/sub/msbs.
// Order-independent (LogUp ALU bus) → validated by multiset/prove, not byte order.
// Padding rows (r >= n) stay all-zero.
//
// Packed stride LT_STRIDE: [0]=lhs [1]=rhs [2]=flags (bit0 signed, bit1 invert)
//   [3]=multiplicity.
#define LT_NCOLS 17u
#define LT_STRIDE 4u

extern "C" __global__ void lt_fill(const uint64_t *ops, uint64_t n,
                                   uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * LT_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * LT_STRIDE;
    uint64_t lhs = op[0];
    uint64_t rhs = op[1];
    uint64_t fl = op[2];
    uint64_t mult = op[3];
    uint64_t is_signed = fl & 1u;
    uint64_t invert = (fl >> 1) & 1u;

    uint64_t lt = is_signed ? ((long long)lhs < (long long)rhs ? 1u : 0u)
                            : (lhs < rhs ? 1u : 0u);
    uint64_t out_bit = lt ^ invert;
    uint64_t sub = lhs - rhs; // wrapping

    out[base + 0] = lhs & 0xFFFFFFFFu;    // LHS_0 (word)
    out[base + 1] = (lhs >> 32) & 0xFFFFu; // LHS_1 (half)
    out[base + 2] = (lhs >> 48) & 0xFFFFu; // LHS_2 (half)
    out[base + 3] = rhs & 0xFFFFFFFFu;    // RHS_0
    out[base + 4] = (rhs >> 32) & 0xFFFFu; // RHS_1
    out[base + 5] = (rhs >> 48) & 0xFFFFu; // RHS_2
    out[base + 6] = is_signed;            // SIGNED
    out[base + 7] = lt;                   // LT
    out[base + 8] = sub & 0xFFFFu;        // LHS_SUB_RHS_0 (DWordHL halves)
    out[base + 9] = (sub >> 16) & 0xFFFFu;
    out[base + 10] = (sub >> 32) & 0xFFFFu;
    out[base + 11] = (sub >> 48) & 0xFFFFu;
    out[base + 12] = (lhs >> 63) & 1u;    // LHS_MSB
    out[base + 13] = (rhs >> 63) & 1u;    // RHS_MSB
    out[base + 14] = invert;              // INVERT
    out[base + 15] = out_bit;             // OUT
    out[base + 16] = mult;                // MU
}

// On-GPU EQ trace fill (12 cols). Mirrors the per-row compute of
// `prover/src/tables/eq.rs::generate_eq_trace`. Dedup is done on the HOST (the
// same per-chunk HashMap the CPU uses); this kernel receives already-unique ops
// with their summed multiplicity, one per row, and recomputes diff/eq/res.
// Order-independent (LogUp ALU bus) → validated by multiset/prove, not byte order.
// Padding rows (r >= n) stay all-zero.
//
// Packed stride EQ_STRIDE: [0]=a [1]=b [2]=flags (bit0 invert) [3]=multiplicity.
#define EQ_NCOLS 12u
#define EQ_STRIDE 4u

extern "C" __global__ void eq_fill(const uint64_t *ops, uint64_t n,
                                   uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * EQ_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * EQ_STRIDE;
    uint64_t a = op[0];
    uint64_t b = op[1];
    uint64_t fl = op[2];
    uint64_t mult = op[3];
    uint64_t invert = fl & 1u;

    uint64_t eq = (a == b) ? 1u : 0u;
    uint64_t res = eq ^ invert;
    uint64_t diff = a - b; // wrapping

    out[base + 0] = a & 0xFFFFFFFFu;        // A_0 (DWordWL lo)
    out[base + 1] = a >> 32;                // A_1 (DWordWL hi)
    out[base + 2] = b & 0xFFFFFFFFu;        // B_0
    out[base + 3] = b >> 32;                // B_1
    out[base + 4] = invert;                 // INVERT
    out[base + 5] = res;                    // RES = eq ^ invert
    out[base + 6] = diff & 0xFFFFu;         // DIFF_0 (DWordHL halves)
    out[base + 7] = (diff >> 16) & 0xFFFFu; // DIFF_1
    out[base + 8] = (diff >> 32) & 0xFFFFu; // DIFF_2
    out[base + 9] = (diff >> 48) & 0xFFFFu; // DIFF_3
    out[base + 10] = eq;                    // EQ = (a == b)
    out[base + 11] = mult;                  // MU
}

// On-GPU BYTEWISE trace fill (26 cols). Mirrors the per-row compute of
// `prover/src/tables/bytewise.rs::generate_bytewise_trace`. Dedup is done on the
// HOST (the same per-chunk HashMap the CPU uses); this kernel receives unique ops
// + summed multiplicity, one per row, and recomputes res = a AND/OR/XOR b, then
// byte-splits a/b/res (DWordBL). Order-independent (LogUp ALU bus) → validated by
// multiset/prove, not byte order. Padding rows (r >= n) stay all-zero.
//
// Packed stride BYTEWISE_STRIDE: [0]=a [1]=b [2]=op (alu_op: 0 AND, 1 OR, 2 XOR)
//   [3]=multiplicity.
#define BYTEWISE_NCOLS 26u
#define BYTEWISE_STRIDE 4u

extern "C" __global__ void bytewise_fill(const uint64_t *ops, uint64_t n,
                                         uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * BYTEWISE_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * BYTEWISE_STRIDE;
    uint64_t a = op[0];
    uint64_t b = op[1];
    uint64_t opc = op[2];
    uint64_t mult = op[3];

    // op: 0 AND, 1 OR, 2 XOR (only these reach BYTEWISE).
    uint64_t res = (opc == 0u) ? (a & b) : (opc == 1u) ? (a | b) : (a ^ b);

    for (int i = 0; i < 8; ++i) {
        out[base + 0 + i] = (a >> (i * 8)) & 0xFFu;    // A[0..8]  (DWordBL)
        out[base + 8 + i] = (b >> (i * 8)) & 0xFFu;    // B[0..8]
        out[base + 17 + i] = (res >> (i * 8)) & 0xFFu; // RES[0..8]
    }
    out[base + 16] = opc;  // OP
    out[base + 25] = mult; // MU
}

// On-GPU MUL trace fill (26 cols). Mirrors the per-row compute of
// `prover/src/tables/mul.rs::generate_mul_trace` — the full 128-bit signed/unsigned
// product plus the sign-extended convolution `raw_product[0..4]`. Dedup is done on
// the HOST (the same per-chunk HashMap the CPU uses, keyed by op with split
// mu_lo/mu_hi from `wants_hi`); this kernel receives unique ops + both
// multiplicities, one per row. Order-independent (LogUp ALU bus) → validated by
// multiset/prove, not byte order. Padding rows (r >= n) stay all-zero.
//
// Packed stride MUL_STRIDE: [0]=lhs [1]=rhs [2]=flags (bit0 lhs_signed, bit1
//   rhs_signed) [3]=mu_lo [4]=mu_hi.
#define MUL_NCOLS 26u
#define MUL_STRIDE 5u

extern "C" __global__ void mul_fill(const uint64_t *ops, uint64_t n,
                                    uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * MUL_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * MUL_STRIDE;
    uint64_t lhs = op[0];
    uint64_t rhs = op[1];
    uint64_t fl = op[2];
    uint64_t mu_lo = op[3];
    uint64_t mu_hi = op[4];
    uint64_t lhs_signed = fl & 1u;
    uint64_t rhs_signed = (fl >> 1) & 1u;

    // Full 128-bit product. Signed operands sign-extend (int64 -> int128);
    // unsigned zero-extend (uint64 -> int128, value-preserving since < 2^64).
    __int128 a = lhs_signed ? (__int128)(int64_t)lhs : (__int128)lhs;
    __int128 b = rhs_signed ? (__int128)(int64_t)rhs : (__int128)rhs;
    __int128 product = a * b; // wrapping
    uint64_t lo = (uint64_t)product;
    uint64_t hi = (uint64_t)((unsigned __int128)product >> 64);

    uint64_t lhs_is_neg = (lhs_signed && ((int64_t)lhs < 0)) ? 1u : 0u;
    uint64_t rhs_is_neg = (rhs_signed && ((int64_t)rhs < 0)) ? 1u : 0u;

    // Sign-extended halfword arrays: [0..4] = halfwords, [4..8] = 0xFFFF*is_neg.
    uint64_t lfill = lhs_is_neg ? 0xFFFFull : 0ull;
    uint64_t rfill = rhs_is_neg ? 0xFFFFull : 0ull;
    uint64_t lhs_ext[8];
    uint64_t rhs_ext[8];
    for (int j = 0; j < 4; ++j) {
        lhs_ext[j] = (lhs >> (16 * j)) & 0xFFFFu;
        rhs_ext[j] = (rhs >> (16 * j)) & 0xFFFFu;
    }
    for (int j = 4; j < 8; ++j) {
        lhs_ext[j] = lfill;
        rhs_ext[j] = rfill;
    }

    // raw_product[i] = Σ_k 2^(16k) Σ_j lhs_ext[j]*rhs_ext[idx-j], idx = 2i+k.
    uint64_t raw[4];
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 sum = 0;
        for (int k = 0; k <= 1; ++k) {
            int idx = 2 * i + k;
            if (idx < 8) {
                for (int j = 0; j <= idx; ++j) {
                    if (j < 8 && (idx - j) < 8) {
                        unsigned __int128 term =
                            (unsigned __int128)lhs_ext[j] * (unsigned __int128)rhs_ext[idx - j];
                        sum += term << (16 * k);
                    }
                }
            }
        }
        raw[i] = (uint64_t)sum;
    }

    for (int j = 0; j < 4; ++j) {
        out[base + 0 + j] = (lhs >> (16 * j)) & 0xFFFFu;  // LHS_0..3 (DWordHL)
        out[base + 5 + j] = (rhs >> (16 * j)) & 0xFFFFu;  // RHS_0..3
        out[base + 10 + j] = (lo >> (16 * j)) & 0xFFFFu;  // LO_0..3
        out[base + 14 + j] = (hi >> (16 * j)) & 0xFFFFu;  // HI_0..3
    }
    out[base + 4] = lhs_signed;   // LHS_SIGNED
    out[base + 9] = rhs_signed;   // RHS_SIGNED
    out[base + 18] = lhs_is_neg;  // LHS_IS_NEGATIVE
    out[base + 19] = rhs_is_neg;  // RHS_IS_NEGATIVE
    out[base + 20] = raw[0];      // RAW_PRODUCT_0..3
    out[base + 21] = raw[1];
    out[base + 22] = raw[2];
    out[base + 23] = raw[3];
    out[base + 24] = mu_lo;       // MU_LO
    out[base + 25] = mu_hi;       // MU_HI
}

// On-GPU DVRM trace fill (34 cols). Mirrors the per-row compute of
// `prover/src/tables/dvrm.rs::generate_dvrm_trace` — RISC-V signed/unsigned
// division & remainder with the div-by-zero and MIN/-1 overflow special cases,
// plus the abs/sign aux columns and n_sub_r. Dedup is done on the HOST (per-chunk
// HashMap keyed by op with split mu_q/mu_r from `wants_remainder`); this kernel
// receives unique ops + both multiplicities, one per row. Order-independent
// (LogUp ALU bus) → validated by multiset/prove, not byte order. Padding rows
// (r >= n) stay all-zero.
//
// Packed stride DVRM_STRIDE: [0]=n [1]=d [2]=flags (bit0 signed) [3]=mu_q [4]=mu_r.
#define DVRM_NCOLS 34u
#define DVRM_STRIDE 5u

extern "C" __global__ void dvrm_fill(const uint64_t *ops, uint64_t n_ops,
                                     uint64_t num_rows, uint64_t *out) {
    uint64_t ri = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ri >= num_rows)
        return;
    uint64_t base = ri * DVRM_NCOLS;
    if (ri >= n_ops)
        return;

    const uint64_t *op = ops + ri * DVRM_STRIDE;
    uint64_t n = op[0];
    uint64_t d = op[1];
    uint64_t fl = op[2];
    uint64_t mu_q = op[3];
    uint64_t mu_r = op[4];
    uint64_t is_signed = fl & 1u;

    uint64_t div_by_zero = (d == 0ull) ? 1u : 0u;
    uint64_t overflow =
        (is_signed && n == 0x8000000000000000ull && d == 0xFFFFFFFFFFFFFFFFull) ? 1u : 0u;

    // Branch order matches the Rust: div-by-zero and overflow are handled before
    // the divide, so the signed path never hits INT64_MIN / -1 (UB) or /0.
    uint64_t q, rem;
    if (div_by_zero) {
        q = 0xFFFFFFFFFFFFFFFFull;
        rem = n;
    } else if (overflow) {
        q = n; // i64::MIN
        rem = 0ull;
    } else if (is_signed) {
        int64_t ni = (int64_t)n;
        int64_t di = (int64_t)d;
        q = (uint64_t)(ni / di);  // truncates toward zero, == wrapping_div
        rem = (uint64_t)(ni % di); // == wrapping_rem (sign follows dividend)
    } else {
        q = n / d;
        rem = n % d;
    }

    uint64_t sign_n = (is_signed && (n >> 63)) ? 1u : 0u;
    uint64_t sign_d = (is_signed && (d >> 63)) ? 1u : 0u;
    uint64_t sign_q = (is_signed && !overflow) ? 1u : 0u;
    uint64_t sign_r = (is_signed && (rem >> 63)) ? 1u : 0u;
    uint64_t n_sub_r = n - rem; // wrapping
    uint64_t sign_n_sub_r = (is_signed && (n_sub_r >> 63)) ? 1u : 0u;

    // abs(x) for a two's-complement negative is 0 - x (mod 2^64); correct for
    // i64::MIN too (matches Rust `unsigned_abs`). Non-negative passes through.
    uint64_t abs_r = sign_r ? (0ull - rem) : rem;
    uint64_t abs_d = sign_d ? (0ull - d) : d;

    for (int j = 0; j < 4; ++j) {
        out[base + 0 + j] = (n >> (16 * j)) & 0xFFFFu;        // N_0..3 (DWordHL)
        out[base + 4 + j] = (d >> (16 * j)) & 0xFFFFu;        // D_0..3
        out[base + 9 + j] = (q >> (16 * j)) & 0xFFFFu;        // Q_0..3
        out[base + 13 + j] = (rem >> (16 * j)) & 0xFFFFu;     // R_0..3
        out[base + 23 + j] = (n_sub_r >> (16 * j)) & 0xFFFFu; // N_SUB_R_0..3
    }
    out[base + 8] = is_signed;            // SIGNED
    out[base + 17] = div_by_zero;         // DIV_BY_ZERO
    out[base + 18] = overflow;            // OVERFLOW
    out[base + 19] = abs_r & 0xFFFFFFFFu; // ABS_R_0 (DWordWL)
    out[base + 20] = abs_r >> 32;         // ABS_R_1
    out[base + 21] = abs_d & 0xFFFFFFFFu; // ABS_D_0
    out[base + 22] = abs_d >> 32;         // ABS_D_1
    out[base + 27] = sign_n_sub_r;        // SIGN_N_SUB_R
    out[base + 28] = sign_n;              // SIGN_N
    out[base + 29] = sign_d;              // SIGN_D
    out[base + 30] = sign_q;              // SIGN_Q
    out[base + 31] = sign_r;              // SIGN_R
    out[base + 32] = mu_q;                // MU_Q
    out[base + 33] = mu_r;                // MU_R
}

// On-GPU MEMW_R (register fast-path) fill: write the 10 MEMW_R columns ROW-MAJOR
// from the host-walked rows. One thread per row (row_index is the identity for
// the fill-from-walked-rows path). Columns mirror
// `prover/src/tables/memw_register.rs::generate_memw_register_trace_from_rows`:
//   0 ADDRESS=reg_addr/2, 1 TS0=ts&0xffffffff, 2 TS1=ts>>32, 3 VAL0, 4 VAL1,
//   5 OLD0, 6 OLD1, 7 OLD_TS_LO=old_ts&0xffffffff, 8 MU_READ=is_read, 9 MU_WRITE=!is_read.
// All limbs < 2^32 (no Goldilocks reduction). Padding rows are pre-zeroed.
extern "C" __global__ void memw_register_fill(
    uint64_t n_acc, const uint32_t *reg_addr, const uint64_t *ts,
    const uint64_t *value, const uint8_t *is_read, const long long *row_index,
    const uint64_t *old_value, const uint64_t *old_ts, uint32_t ncols,
    uint64_t *buf) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_acc)
        return;
    long long row = row_index[i];
    if (row < 0)
        return;
    uint64_t base = (uint64_t)row * ncols;
    uint64_t v = value[i];
    uint64_t ov = old_value[i];
    uint64_t t = ts[i];
    uint64_t ot = old_ts[i];
    buf[base + 0] = (uint64_t)(reg_addr[i] / 2u);
    buf[base + 1] = t & 0xFFFFFFFFull;
    buf[base + 2] = t >> 32;
    buf[base + 3] = v & 0xFFFFFFFFull;
    buf[base + 4] = v >> 32;
    buf[base + 5] = ov & 0xFFFFFFFFull;
    buf[base + 6] = ov >> 32;
    buf[base + 7] = ot & 0xFFFFFFFFull;
    buf[base + 8] = is_read[i] ? 1ull : 0ull;
    buf[base + 9] = is_read[i] ? 0ull : 1ull;
}
