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

// On-GPU COMMIT (ECALL) trace fill (19 cols). Mirrors
// `prover/src/tables/commit.rs::generate_commit_trace`. One thread per row. Packed stride
// COMMIT_STRIDE: [0]=timestamp [1]=index [2]=address [3]=count [4]=first [5]=end [6]=value.
// Padding rows (r>=n) are NOT zero: they need count=1 + address_incr[0]=1 so the unconditional
// ADD/SUB template carries are valid (count_decr=0, address_incr=1).
#define COMMIT_NCOLS 19u
#define COMMIT_STRIDE 7u

extern "C" __global__ void commit_fill(const uint64_t *ops, uint64_t n, uint64_t num_rows,
                                       uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * COMMIT_NCOLS;
    if (r >= n) {
        out[base + 9] = 1u; // COUNT_0 = 1
        out[base + 5] = 1u; // ADDRESS_INCR_0 = 1 (address 0 → address+1 = 1)
        return;
    }
    const uint64_t *op = ops + r * COMMIT_STRIDE;
    uint64_t ts = op[0], index = op[1], addr = op[2], count = op[3];
    uint64_t first = op[4], end = op[5], value = op[6];

    out[base + 0] = ts & 0xFFFFFFFFu;  // TIMESTAMP_0
    out[base + 1] = ts >> 32;          // TIMESTAMP_1
    out[base + 2] = index;             // INDEX
    out[base + 3] = addr & 0xFFFFFFFFu; // ADDRESS_0
    out[base + 4] = addr >> 32;         // ADDRESS_1

    uint64_t ai = addr + 1ull; // address_incr = address + 1 (wrapping), as 4 halfwords
    out[base + 5] = ai & 0xFFFFu;
    out[base + 6] = (ai >> 16) & 0xFFFFu;
    out[base + 7] = (ai >> 32) & 0xFFFFu;
    out[base + 8] = (ai >> 48) & 0xFFFFu;

    out[base + 9] = count & 0xFFFFFFFFu;  // COUNT_0
    out[base + 10] = count >> 32;         // COUNT_1

    uint64_t cd = (count == 0ull) ? 0xFFFFFFFFFFFFFFFFull : (count - 1ull); // count_decr, 4 halfwords
    out[base + 11] = cd & 0xFFFFu;
    out[base + 12] = (cd >> 16) & 0xFFFFu;
    out[base + 13] = (cd >> 32) & 0xFFFFu;
    out[base + 14] = (cd >> 48) & 0xFFFFu;

    out[base + 15] = first; // FIRST
    out[base + 16] = end;   // END
    out[base + 17] = value; // VALUE
    out[base + 18] = 1u;    // MU
}

// On-GPU KECCAK (main permute) trace fill (511 cols). Mirrors
// `prover/src/tables/keccak.rs::generate_keccak_trace`. One thread per row. Packed stride
// KECCAK_TBL_STRIDE: [0]=timestamp [1]=state_addr [2..27]=input[25] [27..52]=output[25].
// Cols: TIMESTAMP(dword_wl,0..2) ADDR(8 bytes,2..10) INPUT_STATE(200,10..210)
// OUTPUT_STATE(200,210..410) STATE_PTR(25 lanes*4 hw,410..510) MU(510). Padding rows
// (r>=n) set state_ptr[lane][0] = 8*lane (per keccak.toml pad).
#define KECCAK_TBL_NCOLS 511u
#define KECCAK_TBL_STRIDE 52u

extern "C" __global__ void keccak_table_fill(const uint64_t *ops, uint64_t n, uint64_t num_rows,
                                             uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * KECCAK_TBL_NCOLS;
    if (r >= n) {
        for (int lane = 0; lane < 25; ++lane)
            out[base + 410 + lane * 4] = (uint64_t)(8 * lane);
        return;
    }
    const uint64_t *op = ops + r * KECCAK_TBL_STRIDE;
    uint64_t ts = op[0];
    uint64_t addr = op[1];

    out[base + 0] = ts & 0xFFFFFFFFu; // TIMESTAMP_0
    out[base + 1] = ts >> 32;         // TIMESTAMP_1
    for (int b = 0; b < 8; ++b)
        out[base + 2 + b] = (addr >> (8 * b)) & 0xFFu; // ADDR bytes

    for (int lane = 0; lane < 25; ++lane) {
        uint64_t v = op[2 + lane]; // input lane
        for (int b = 0; b < 8; ++b)
            out[base + 10 + lane * 8 + b] = (v >> (8 * b)) & 0xFFu;
    }
    for (int lane = 0; lane < 25; ++lane) {
        uint64_t v = op[27 + lane]; // output lane
        for (int b = 0; b < 8; ++b)
            out[base + 210 + lane * 8 + b] = (v >> (8 * b)) & 0xFFu;
    }
    for (int lane = 0; lane < 25; ++lane) {
        uint64_t ptr = addr + (uint64_t)(8 * lane); // state_ptr = addr + 8*lane, 4 halfwords
        out[base + 410 + lane * 4 + 0] = ptr & 0xFFFFu;
        out[base + 410 + lane * 4 + 1] = (ptr >> 16) & 0xFFFFu;
        out[base + 410 + lane * 4 + 2] = (ptr >> 32) & 0xFFFFu;
        out[base + 410 + lane * 4 + 3] = (ptr >> 48) & 0xFFFFu;
    }
    out[base + 510] = 1u; // MU
}

// On-GPU ECDAS (EC double-and-add step) trace fill (521 cols). Mirrors
// `prover/src/tables/ecdas.rs::generate_ecdas_trace` — pure FORMATTING of the precomputed
// witness (no EC/modular math; that ran on CPU during execution). Compact inputs:
//   bytes[r*326]: x_g[32] y_g[32] x_a[32] y_a[32] round op x_r[32] y_r[32] lambda[32]
//                 q0[33] q1[33] q2[33] next_op
//   carries[r*192] (i64): c0[64] c1[64] c2[64]   ts[r]: timestamp
// Signed carries → Goldilocks via ec_fe. Padding rows (r>=n): OP=1, rest 0.
#define ECDAS_NCOLS 521u
#define ECDAS_BSTRIDE 326u
#define ECDAS_CSTRIDE 192u
#define GOLDP 0xFFFFFFFF00000001ull
__device__ __forceinline__ uint64_t ec_fe(long long c) {
    return c >= 0 ? (uint64_t)c : (GOLDP - (uint64_t)(-c));
}

// -----------------------------------------------------------------------------
// On-GPU ECDAS per-step CARRY WITNESS (the `conv` limb-convolution math that the CPU
// `ecsm::witness::build_step` did — ~190ms/proof for ethrex_5tx, moved to device). The EC
// scalar-mult (`replay_double_and_add`, k256) and the tiny quotients stay on CPU; this kernel
// derives ONLY the per-step carries `c0/c1/c2` from the packed point+quotient bytes. Bit-exact
// with `carries_lambda/xr/yr` + `limb_carries` (int arithmetic, no field/curve ops). Input `bytes`
// layout matches `ecdas_fill` (ECDAS_BSTRIDE=326); output `out[r*192]` = c0[64] c1[64] c2[64] (i64).
// secp256k1 p and 3p as 64 zero-extended 8-bit limbs (little-endian); only [0..32]/[0..33] nonzero.
__device__ const long EC_PP[64] = {
    0x2F, 0xFC, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ const long EC_R3P[64] = {
    0x8D, 0xF4, 0xFF, 0xFF, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

typedef __int128 eci128;

// conv(a,b,i) = Σ_{j=0..i} a[j]·b[i-j]  (Rust `conv`).
__device__ __forceinline__ eci128 ec_conv(const long *a, const long *b, int i) {
    eci128 s = 0;
    for (int j = 0; j <= i; ++j)
        s += (eci128)a[j] * (eci128)b[i - j];
    return s;
}

// limb_carries: 2^8·c_i = c_{i-1} + terms_i, c_{-1}=0 (Rust `limb_carries`). For valid inputs each
// partial sum is divisible by 256, so the arithmetic shift equals the exact division.
__device__ __forceinline__ void ec_limb_carries(const eci128 *terms, long long *out) {
    eci128 carry = 0;
    for (int i = 0; i < 64; ++i) {
        eci128 s = carry + terms[i];
        carry = s >> 8;
        out[i] = (long long)carry;
    }
}

// Zero-extend `len` little-endian 8-bit limbs to 64 longs (Rust `ext64`).
__device__ __forceinline__ void ec_load(long *arr, const uint8_t *b, int len) {
    for (int i = 0; i < 64; ++i)
        arr[i] = (i < len) ? (long)b[i] : 0L;
}

extern "C" __global__ void ecdas_carries(const uint8_t *bytes, uint64_t n, long long *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n)
        return;
    const uint8_t *b = bytes + r * ECDAS_BSTRIDE;
    long xg[64], yg[64], xa[64], ya[64], xr[64], yr[64], lam[64], q0[64], q1[64], q2[64];
    ec_load(xg, b + 0, 32);
    ec_load(yg, b + 32, 32);
    ec_load(xa, b + 64, 32);
    ec_load(ya, b + 96, 32);
    uint8_t op = b[129];
    ec_load(xr, b + 130, 32);
    ec_load(yr, b + 162, 32);
    ec_load(lam, b + 194, 32);
    ec_load(q0, b + 226, 33);
    ec_load(q1, b + 259, 33);
    ec_load(q2, b + 292, 33);

    long long *c = out + r * ECDAS_CSTRIDE;
    eci128 terms[64];

    // c0 = carries_lambda: op·(Σ λ_j(xG−xA)_{i-j} + (yA−yG)_i) + (1−op)·Σ(2λ_j yA − 3xA_j xA)_{i-j}
    //      + conv(3p,p,i) − conv(q0,p,i)
    for (int i = 0; i < 64; ++i) {
        eci128 s;
        if (op == 1) {
            s = (eci128)ya[i] - (eci128)yg[i];
            for (int j = 0; j <= i; ++j)
                s += (eci128)lam[j] * ((eci128)xg[i - j] - (eci128)xa[i - j]);
        } else {
            s = 0;
            for (int j = 0; j <= i; ++j)
                s += (eci128)2 * lam[j] * ya[i - j] - (eci128)3 * xa[j] * xa[i - j];
        }
        terms[i] = s + ec_conv(EC_R3P, EC_PP, i) - ec_conv(q0, EC_PP, i);
    }
    ec_limb_carries(terms, c);

    // c1 = carries_xr: λ² − xA − xG − xR − (1−op)(xA−xG) + conv(3p,p,i) − conv(q1,p,i)
    for (int i = 0; i < 64; ++i) {
        eci128 op_term = (op == 0) ? ((eci128)xa[i] - (eci128)xg[i]) : (eci128)0;
        terms[i] = ec_conv(lam, lam, i) - (eci128)xa[i] - (eci128)xg[i] - (eci128)xr[i] - op_term +
                   ec_conv(EC_R3P, EC_PP, i) - ec_conv(q1, EC_PP, i);
    }
    ec_limb_carries(terms, c + 64);

    // c2 = carries_yr: Σ λ_j(xA−xR)_{i-j} − yA − yR + conv(3p,p,i) − conv(q2,p,i)
    for (int i = 0; i < 64; ++i) {
        eci128 conv_lam = 0;
        for (int j = 0; j <= i; ++j)
            conv_lam += (eci128)lam[j] * ((eci128)xa[i - j] - (eci128)xr[i - j]);
        terms[i] = conv_lam - (eci128)ya[i] - (eci128)yr[i] + ec_conv(EC_R3P, EC_PP, i) -
                   ec_conv(q2, EC_PP, i);
    }
    ec_limb_carries(terms, c + 128);
}

extern "C" __global__ void ecdas_fill(const uint8_t *bytes, const long long *carries,
                                      const uint64_t *ts, uint64_t n, uint64_t num_rows,
                                      uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * ECDAS_NCOLS;
    if (r >= n) {
        out[base + 131] = 1u; // OP = 1 (add) on padding rows
        return;
    }
    const uint8_t *b = bytes + r * ECDAS_BSTRIDE;
    const long long *c = carries + r * ECDAS_CSTRIDE;
    uint64_t t = ts[r];
    out[base + 0] = t & 0xFFFFFFFFu;
    out[base + 1] = t >> 32;
    int o = 0;
    for (int i = 0; i < 32; ++i) out[base + 2 + i] = b[o++];   // XG
    for (int i = 0; i < 32; ++i) out[base + 34 + i] = b[o++];  // YG
    for (int i = 0; i < 32; ++i) out[base + 66 + i] = b[o++];  // XA
    for (int i = 0; i < 32; ++i) out[base + 98 + i] = b[o++];  // YA
    out[base + 130] = b[o++];                                  // ROUND
    out[base + 131] = b[o++];                                  // OP
    for (int i = 0; i < 32; ++i) out[base + 132 + i] = b[o++]; // XR
    for (int i = 0; i < 32; ++i) out[base + 164 + i] = b[o++]; // YR
    for (int i = 0; i < 32; ++i) out[base + 196 + i] = b[o++]; // LAMBDA
    for (int i = 0; i < 33; ++i) out[base + 228 + i] = b[o++]; // Q0
    for (int i = 0; i < 33; ++i) out[base + 325 + i] = b[o++]; // Q1
    for (int i = 0; i < 33; ++i) out[base + 422 + i] = b[o++]; // Q2
    out[base + 519] = b[o++];                                  // NEXT_OP
    for (int i = 0; i < 64; ++i) out[base + 261 + i] = ec_fe(c[i]);        // C0
    for (int i = 0; i < 64; ++i) out[base + 358 + i] = ec_fe(c[64 + i]);   // C1
    for (int i = 0; i < 64; ++i) out[base + 455 + i] = ec_fe(c[128 + i]);  // C2
    out[base + 520] = 1u;                                      // MU
}

// On-GPU ECSM (k·G core) trace fill (667 cols). Mirrors
// `prover/src/tables/ecsm.rs::generate_ecsm_trace` — pure FORMATTING of the precomputed witness
// (no EC/modular math). Compact inputs:
//   bytes[r*354]: x_r[32] y_r[32] k[32] x_g[32] y_g[32] x2[32] q0[32] q1[33]
//                 x_g_sub_p[32] k_sub_n[32] x_r_sub_p[32] len_k
//   carries[r*128] (i64): c0[64] c1[64]   addrs[r*4]: ts, addr_xg, addr_k, addr_xr
// k → 256 bit columns; *_sub_p → 16 LE halfwords; signed carries → Goldilocks (ec_fe).
// Padding rows (r>=n): all zero (generate_ecsm_trace writes nothing for padding).
#define ECSM_NCOLS 667u
#define ECSM_BSTRIDE 354u
#define ECSM_CSTRIDE 128u
#define ECSM_ASTRIDE 4u

extern "C" __global__ void ecsm_fill(const uint8_t *bytes, const long long *carries,
                                     const uint64_t *addrs, uint64_t n, uint64_t num_rows,
                                     uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * ECSM_NCOLS;
    if (r >= n)
        return; // padding all-zero
    const uint8_t *b = bytes + r * ECSM_BSTRIDE;
    const long long *c = carries + r * ECSM_CSTRIDE;
    const uint64_t *a = addrs + r * ECSM_ASTRIDE;

    out[base + 0] = a[0] & 0xFFFFFFFFu;  // TIMESTAMP
    out[base + 1] = a[0] >> 32;
    out[base + 2] = a[1] & 0xFFFFFFFFu;  // ADDR_XG
    out[base + 3] = a[1] >> 32;
    out[base + 4] = a[2] & 0xFFFFFFFFu;  // ADDR_K
    out[base + 5] = a[2] >> 32;
    out[base + 6] = a[3] & 0xFFFFFFFFu;  // ADDR_XR
    out[base + 7] = a[3] >> 32;

    int o = 0;
    for (int i = 0; i < 32; ++i) out[base + 8 + i] = b[o++];  // XR
    for (int i = 0; i < 32; ++i) out[base + 40 + i] = b[o++]; // YR
    const uint8_t *kb = b + o;                                // K: 32 bytes → 256 bits
    o += 32;
    for (int bit = 0; bit < 256; ++bit)
        out[base + 72 + bit] = (uint64_t)((kb[bit >> 3] >> (bit & 7)) & 1u);
    for (int i = 0; i < 32; ++i) out[base + 329 + i] = b[o++]; // XG
    for (int i = 0; i < 32; ++i) out[base + 361 + i] = b[o++]; // YG
    for (int i = 0; i < 32; ++i) out[base + 393 + i] = b[o++]; // X2
    for (int i = 0; i < 32; ++i) out[base + 425 + i] = b[o++]; // Q0
    for (int i = 0; i < 33; ++i) out[base + 521 + i] = b[o++]; // Q1
    const uint8_t *xgsp = b + o;
    o += 32;
    for (int j = 0; j < 16; ++j) out[base + 618 + j] = (uint64_t)xgsp[2 * j] | ((uint64_t)xgsp[2 * j + 1] << 8);
    const uint8_t *ksn = b + o;
    o += 32;
    for (int j = 0; j < 16; ++j) out[base + 634 + j] = (uint64_t)ksn[2 * j] | ((uint64_t)ksn[2 * j + 1] << 8);
    const uint8_t *xrsp = b + o;
    o += 32;
    for (int j = 0; j < 16; ++j) out[base + 650 + j] = (uint64_t)xrsp[2 * j] | ((uint64_t)xrsp[2 * j + 1] << 8);
    out[base + 328] = b[o++]; // LEN_K

    for (int i = 0; i < 64; ++i) out[base + 457 + i] = ec_fe(c[i]);      // C0
    for (int i = 0; i < 64; ++i) out[base + 554 + i] = ec_fe(c[64 + i]); // C1
    out[base + 666] = 1u;                                                // MU
}

// On-GPU KECCAK_RND (per-round) trace fill (1480 cols, 24 rows/op). Mirrors
// `prover/src/tables/keccak_rnd.rs::generate_keccak_rnd_trace`. ONE THREAD PER OP: the state
// evolves round-to-round, so each thread runs all 24 rounds sequentially and writes 24 rows.
// Packed stride 26: [ts, input[25]] (output is recomputed as chi/iota, not read). Padding rows
// (row >= n*24) stay zero.
#define KRND_NCOLS 1480u
#define KRND_STRIDE 26u
__device__ __constant__ uint32_t KRND_RHO[25] = {
    0, 36, 3, 41, 18, 1, 44, 10, 45, 2, 62, 6, 43, 15, 61,
    28, 55, 25, 21, 56, 27, 20, 39, 8, 14};
__device__ __constant__ uint64_t KRND_RC[24] = {
    0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808Aull, 0x8000000080008000ull,
    0x000000000000808Bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
    0x000000000000008Aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000Aull,
    0x000000008000808Bull, 0x800000000000008Bull, 0x8000000000008089ull, 0x8000000000008003ull,
    0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800Aull, 0x800000008000000Aull,
    0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull};

__device__ __forceinline__ void hwsl_dev(uint16_t hw, uint32_t shift, uint16_t *shifted,
                                         uint16_t *carry) {
    if (shift == 0) {
        *shifted = hw;
        *carry = 0;
    } else {
        *shifted = (uint16_t)(hw << shift);
        *carry = (uint16_t)(hw >> (16 - shift));
    }
}

extern "C" __global__ void keccak_rnd_fill(const uint64_t *ops, uint64_t n, uint64_t num_rows,
                                           uint64_t *out) {
    uint64_t opi = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (opi >= n)
        return;
    const uint64_t *op = ops + opi * KRND_STRIDE;
    uint64_t ts = op[0];
    uint64_t state[25];
    for (int i = 0; i < 25; ++i)
        state[i] = op[1 + i];

    for (int round = 0; round < 24; ++round) {
        uint64_t base = (opi * 24 + (uint64_t)round) * KRND_NCOLS;
        out[base + 0] = ts & 0xFFFFFFFFu;
        out[base + 1] = ts >> 32;
        out[base + 2] = (uint64_t)round;
        for (int lane = 0; lane < 25; ++lane)
            for (int b = 0; b < 8; ++b)
                out[base + 3 + lane * 8 + b] = (state[lane] >> (8 * b)) & 0xFFu; // START

        // theta: Cxz chain → c_bytes
        uint8_t c_bytes[5][8];
        for (int x = 0; x < 5; ++x) {
            uint8_t cxz[4][8];
            for (int b = 0; b < 8; ++b)
                cxz[0][b] = (uint8_t)((state[x] >> (8 * b)) ^ (state[x + 5] >> (8 * b)));
            for (int b = 0; b < 8; ++b)
                out[base + 203 + (x * 4 + 0) * 8 + b] = cxz[0][b];
            for (int stage = 1; stage < 4; ++stage) {
                int y = stage + 1;
                for (int b = 0; b < 8; ++b)
                    cxz[stage][b] = (uint8_t)(cxz[stage - 1][b] ^ (state[x + 5 * y] >> (8 * b)));
                for (int b = 0; b < 8; ++b)
                    out[base + 203 + (x * 4 + stage) * 8 + b] = cxz[stage][b];
            }
            for (int b = 0; b < 8; ++b)
                c_bytes[x][b] = cxz[3][b];
        }
        // rotate C left 1 (HWSL) → cxz_left / cxz_right bits → rotated_c
        uint8_t rotated_c[5][8];
        for (int x = 0; x < 5; ++x) {
            uint8_t cxz_left[8];
            uint8_t cxz_right[4];
            for (int hw = 0; hw < 4; ++hw) {
                uint16_t halfword = (uint16_t)c_bytes[x][2 * hw] | ((uint16_t)c_bytes[x][2 * hw + 1] << 8);
                uint16_t sh, cy;
                hwsl_dev(halfword, 1, &sh, &cy);
                cxz_left[2 * hw] = sh & 0xFF;
                cxz_left[2 * hw + 1] = sh >> 8;
                cxz_right[hw] = (uint8_t)cy;
            }
            for (int b = 0; b < 8; ++b)
                out[base + 363 + x * 8 + b] = cxz_left[b]; // CXZ_LEFT
            for (int hw = 0; hw < 4; ++hw)
                out[base + 403 + x * 4 + hw] = cxz_right[hw]; // CXZ_RIGHT
            for (int b = 0; b < 8; ++b) {
                uint8_t rc_contrib = (b % 2 == 0) ? cxz_right[(b / 2 + 3) % 4] : 0;
                rotated_c[x][b] = (uint8_t)(cxz_left[b] + rc_contrib);
            }
        }
        // D
        uint8_t d_bytes[5][8];
        for (int x = 0; x < 5; ++x) {
            for (int b = 0; b < 8; ++b)
                d_bytes[x][b] = (uint8_t)(c_bytes[(x + 4) % 5][b] ^ rotated_c[(x + 1) % 5][b]);
            for (int b = 0; b < 8; ++b)
                out[base + 423 + x * 8 + b] = d_bytes[x][b]; // DXZ
        }
        // theta lanes
        uint64_t theta_lanes[25];
        for (int x = 0; x < 5; ++x) {
            uint64_t d_lane = 0;
            for (int b = 0; b < 8; ++b)
                d_lane |= ((uint64_t)d_bytes[x][b]) << (8 * b);
            for (int y = 0; y < 5; ++y) {
                theta_lanes[x + 5 * y] = state[x + 5 * y] ^ d_lane;
                for (int b = 0; b < 8; ++b)
                    out[base + 463 + (x + 5 * y) * 8 + b] = (theta_lanes[x + 5 * y] >> (8 * b)) & 0xFFu;
            }
        }
        // rho (HWSL by rnc = rho%16) → rot_left / rot_right
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y) {
                uint32_t rnc = KRND_RHO[x * 5 + y] % 16;
                uint64_t tl = theta_lanes[x + 5 * y];
                for (int hw = 0; hw < 4; ++hw) {
                    uint16_t halfword = (uint16_t)((tl >> (16 * hw)) & 0xFFFF);
                    uint16_t sh, cy;
                    hwsl_dev(halfword, rnc, &sh, &cy);
                    out[base + 663 + (x + 5 * y) * 8 + 2 * hw] = sh & 0xFF;
                    out[base + 663 + (x + 5 * y) * 8 + 2 * hw + 1] = sh >> 8;
                    out[base + 863 + (x + 5 * y) * 8 + 2 * hw] = cy & 0xFF;
                    out[base + 863 + (x + 5 * y) * 8 + 2 * hw + 1] = cy >> 8;
                }
            }
        // pi
        uint64_t pi_lanes[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y) {
                uint32_t rho = KRND_RHO[x * 5 + y];
                uint64_t t = theta_lanes[x + 5 * y];
                uint64_t rotated = (rho == 0) ? t : ((t << rho) | (t >> (64 - rho)));
                pi_lanes[y + 5 * ((2 * x + 3 * y) % 5)] = rotated;
            }
        // chi + iota
        uint64_t chi_lanes[25];
        for (int x = 0; x < 5; ++x)
            for (int y = 0; y < 5; ++y) {
                uint64_t and_val = (~pi_lanes[(x + 1) % 5 + 5 * y]) & pi_lanes[(x + 2) % 5 + 5 * y];
                chi_lanes[x + 5 * y] = pi_lanes[x + 5 * y] ^ and_val;
                for (int b = 0; b < 8; ++b)
                    out[base + 1063 + (x + 5 * y) * 8 + b] = (and_val >> (8 * b)) & 0xFFu;
                for (int b = 0; b < 8; ++b)
                    out[base + 1263 + (x + 5 * y) * 8 + b] = (chi_lanes[x + 5 * y] >> (8 * b)) & 0xFFu;
            }
        uint64_t rc_val = KRND_RC[round];
        uint64_t iota_lane = chi_lanes[0] ^ rc_val;
        for (int b = 0; b < 8; ++b)
            out[base + 1463 + b] = (rc_val >> (8 * b)) & 0xFFu;
        for (int b = 0; b < 8; ++b)
            out[base + 1471 + b] = (iota_lane >> (8 * b)) & 0xFFu;
        out[base + 1479] = 1u; // MU
        chi_lanes[0] = iota_lane;
        for (int i = 0; i < 25; ++i)
            state[i] = chi_lanes[i];
    }
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

// On-GPU BRANCH trace fill (14 cols). Mirrors the per-row compute of
// `prover/src/tables/branch.rs::generate_branch_trace`: next_pc = (base + offset)
// & ~1, where base = jalr ? register : pc, split into 3 high halfwords + 2 low
// bytes (LSB masked) plus the unmasked low byte. Dedup is done on the HOST (the
// same per-chunk HashMap the CPU uses); this kernel receives unique ops + summed
// multiplicity, one per row. Order-independent (LogUp lookup bus) → validated by
// multiset/prove, not byte order. Padding rows (r >= n) stay all-zero.
//
// Packed stride BRANCH_STRIDE: [0]=pc [1]=offset [2]=register [3]=flags (bit0
//   jalr) [4]=multiplicity.
#define BRANCH_NCOLS 14u
#define BRANCH_STRIDE 5u

extern "C" __global__ void branch_fill(const uint64_t *ops, uint64_t n,
                                       uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * BRANCH_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * BRANCH_STRIDE;
    uint64_t pc = op[0];
    uint64_t offset = op[1];
    uint64_t reg = op[2];
    uint64_t fl = op[3];
    uint64_t mult = op[4];
    uint64_t jalr = fl & 1u;

    uint64_t b = jalr ? reg : pc;
    uint64_t unmasked = b + offset; // wrapping
    uint64_t next_pc = unmasked & ~1ull;

    out[base + 0] = pc & 0xFFFFFFFFu;            // PC_0 (DWordWL)
    out[base + 1] = pc >> 32;                    // PC_1
    out[base + 2] = offset & 0xFFFFFFFFu;        // OFFSET_0
    out[base + 3] = offset >> 32;                // OFFSET_1
    out[base + 4] = reg & 0xFFFFFFFFu;           // REGISTER_0
    out[base + 5] = reg >> 32;                   // REGISTER_1
    out[base + 6] = jalr;                        // JALR
    out[base + 7] = (next_pc >> 16) & 0xFFFFu;   // NEXT_PC_HIGH_0 (halves)
    out[base + 8] = (next_pc >> 32) & 0xFFFFu;   // NEXT_PC_HIGH_1
    out[base + 9] = (next_pc >> 48) & 0xFFFFu;   // NEXT_PC_HIGH_2
    out[base + 10] = next_pc & 0xFFu;            // NEXT_PC_LOW_0 (masked LSB)
    out[base + 11] = (next_pc >> 8) & 0xFFu;     // NEXT_PC_LOW_1
    out[base + 12] = unmasked & 0xFFu;           // UNMASKED_LOW_BYTE
    out[base + 13] = mult;                       // MU
}

// On-GPU CPU32 trace fill (38 cols). Mirrors the per-row compute of
// `prover/src/tables/cpu32.rs::{generate_cpu32_trace, compute_aux}` — the delegated
// `*W` (32-bit word) instructions. Sign/zero-extends rv1/rv2 to arg1/arg2 and
// sign-extends the 32-bit result to rvd, per RV64 `*W` semantics. Per-row (μ=1, no
// dedup) → byte-identical to the CPU fill. Padding rows (r >= n) stay all-zero.
//
// Packed stride CPU32_STRIDE: [0]=timestamp [1]=pc [2]=rv1 [3]=rv2 [4]=imm [5]=res
//   [6]=flags (bit0 read_register1, bit1 read_register2, bit2 write_register,
//   bit3 alu, bit4 add, bit5 sub) [7]=bytes (b0 rs1, b1 rs2, b2 rd, b3 alu_flags,
//   b4 half_instruction_length).
#define CPU32_NCOLS 38u
#define CPU32_STRIDE 8u

extern "C" __global__ void cpu32_fill(const uint64_t *ops, uint64_t n,
                                      uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * CPU32_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * CPU32_STRIDE;
    uint64_t ts = op[0];
    uint64_t pc = op[1];
    uint64_t rv1 = op[2];
    uint64_t rv2 = op[3];
    uint64_t imm = op[4];
    uint64_t res = op[5];
    uint64_t fl = op[6];
    uint64_t by = op[7];

    uint64_t rr1 = fl & 1u;
    uint64_t rr2 = (fl >> 1) & 1u;
    uint64_t wr = (fl >> 2) & 1u;
    uint64_t alu = (fl >> 3) & 1u;
    uint64_t add = (fl >> 4) & 1u;
    uint64_t sub = (fl >> 5) & 1u;

    uint64_t rs1 = by & 0xFFu;
    uint64_t rs2 = (by >> 8) & 0xFFu;
    uint64_t rd = (by >> 16) & 0xFFu;
    uint64_t aluf = (by >> 24) & 0xFFu;
    uint64_t hil = (by >> 32) & 0xFFu;

    // signed = alu_flags bit 5 (ALU_FLAGS_SIGNED). rv1/rv2 sign bits are gated by
    // `signed`; the result is always sign-extended for *W.
    uint64_t is_signed = (aluf >> 5) & 1u;
    uint64_t rv1_sign = (is_signed && ((rv1 >> 31) & 1u)) ? 1u : 0u;
    uint64_t rv2_sign = (is_signed && ((rv2 >> 31) & 1u)) ? 1u : 0u;
    uint64_t res_sign = ((res >> 31) & 1u) ? 1u : 0u;

    uint64_t arg1_hi = rv1_sign ? 0xFFFFFFFFull : 0ull;
    uint64_t arg1 = (rv1 & 0xFFFFFFFFull) | (arg1_hi << 32);

    // By the decoding assumption exactly one of rv2 / imm is nonzero, so the
    // per-word sums do not overflow; the masks mirror the Rust exactly regardless.
    uint64_t arg2_lo = (rv2 & 0xFFFFFFFFull) + (imm & 0xFFFFFFFFull);
    uint64_t arg2_hi = (rv2_sign ? 0xFFFFFFFFull : 0ull) + (imm >> 32);
    uint64_t arg2 = (arg2_lo & 0xFFFFFFFFull) | (arg2_hi << 32);

    uint64_t rvd_hi = res_sign ? 0xFFFFFFFFull : 0ull;
    uint64_t rvd = (res & 0xFFFFFFFFull) | (rvd_hi << 32);

    out[base + 0] = ts & 0xFFFFFFFFu;           // TIMESTAMP_0 (DWordWL)
    out[base + 1] = ts >> 32;                   // TIMESTAMP_1
    out[base + 2] = pc & 0xFFFFFFFFu;           // PC_0
    out[base + 3] = pc >> 32;                   // PC_1
    out[base + 4] = rs1;                        // RS1
    out[base + 5] = rr1;                        // READ_REGISTER1
    out[base + 6] = rv1 & 0xFFFFu;              // RV1_0 (DWordWHH: half)
    out[base + 7] = (rv1 >> 16) & 0xFFFFu;      // RV1_1 (half)
    out[base + 8] = (rv1 >> 32) & 0xFFFFFFFFu;  // RV1_2 (word)
    out[base + 9] = rv1_sign;                   // RV1_SIGN
    out[base + 10] = arg1 & 0xFFFFFFFFu;        // ARG1_0 (DWordWL)
    out[base + 11] = arg1 >> 32;                // ARG1_1
    out[base + 12] = rs2;                       // RS2
    out[base + 13] = rr2;                       // READ_REGISTER2
    out[base + 14] = rv2 & 0xFFFFu;             // RV2_0
    out[base + 15] = (rv2 >> 16) & 0xFFFFu;     // RV2_1
    out[base + 16] = (rv2 >> 32) & 0xFFFFFFFFu; // RV2_2
    out[base + 17] = rv2_sign;                  // RV2_SIGN
    out[base + 18] = imm & 0xFFFFFFFFu;         // IMM_0
    out[base + 19] = imm >> 32;                 // IMM_1
    out[base + 20] = arg2 & 0xFFFFFFFFu;        // ARG2_0
    out[base + 21] = arg2 >> 32;                // ARG2_1
    out[base + 22] = res & 0xFFFFu;             // RES_0 (DWordHL: 4 halves)
    out[base + 23] = (res >> 16) & 0xFFFFu;     // RES_1
    out[base + 24] = (res >> 32) & 0xFFFFu;     // RES_2
    out[base + 25] = (res >> 48) & 0xFFFFu;     // RES_3
    out[base + 26] = res_sign;                  // RES_SIGN
    out[base + 27] = rd;                        // RD
    out[base + 28] = wr;                        // WRITE_REGISTER
    out[base + 29] = rvd & 0xFFFFFFFFu;         // RVD_0
    out[base + 30] = rvd >> 32;                 // RVD_1
    out[base + 31] = alu;                       // ALU
    out[base + 32] = aluf;                      // ALU_FLAGS
    out[base + 33] = add;                       // ADD
    out[base + 34] = sub;                       // SUB
    out[base + 35] = hil;                       // HALF_INSTRUCTION_LENGTH
    out[base + 36] = is_signed;                 // SIGNED
    out[base + 37] = 1u;                        // MU (active row)
}

// On-GPU MEMW (general / unaligned / split-timestamp) trace fill (49 cols). Mirrors
// `prover/src/tables/memw.rs::generate_memw_trace`. The op is already walked
// (old/old_timestamp filled), so this is bit-slicing + the carry[i] aux
// (base_addr_lo + (i+1) >= 2^32). Per-row (no dedup) → byte-identical to the CPU
// fill. Padding rows (r >= n) stay all-zero.
//
// Packed stride MEMW_STRIDE: [0]=flags (bit0 is_register, bit1 is_read, bits8..16
//   width) [1]=base_address [2]=timestamp [3..7]=value[0..8] packed 2×u32/u64
//   [7..11]=old[0..8] packed 2×u32/u64 [11..19]=old_timestamp[0..8].
#define MEMW_NCOLS 49u
#define MEMW_STRIDE 19u

extern "C" __global__ void memw_fill(const uint64_t *ops, uint64_t n,
                                     uint64_t num_rows, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rows)
        return;
    uint64_t base = r * MEMW_NCOLS;
    if (r >= n)
        return;

    const uint64_t *op = ops + r * MEMW_STRIDE;
    uint64_t fl = op[0];
    uint64_t is_register = fl & 1u;
    uint64_t is_read = (fl >> 1) & 1u;
    uint64_t width = (fl >> 8) & 0xFFu;
    uint64_t addr = op[1];
    uint64_t ts = op[2];

    out[base + 0] = is_register;                // IS_REGISTER
    out[base + 1] = addr & 0xFFFFFFFFu;         // BASE_ADDRESS_0 (DWordWL)
    out[base + 2] = addr >> 32;                 // BASE_ADDRESS_1
    // VALUE[0..8] at cols 3..11 (each column holds one full value limb).
    for (int i = 0; i < 4; ++i) {
        uint64_t p = op[3 + i];
        out[base + 3 + 2 * i] = p & 0xFFFFFFFFu;
        out[base + 3 + 2 * i + 1] = p >> 32;
    }
    out[base + 11] = ts & 0xFFFFFFFFu;          // TIMESTAMP_0
    out[base + 12] = ts >> 32;                  // TIMESTAMP_1
    out[base + 13] = (width == 2u) ? 1u : 0u;   // WRITE2
    out[base + 14] = (width == 4u) ? 1u : 0u;   // WRITE4
    out[base + 15] = (width == 8u) ? 1u : 0u;   // WRITE8
    // OLD[0..8] at cols 16..24.
    for (int i = 0; i < 4; ++i) {
        uint64_t p = op[7 + i];
        out[base + 16 + 2 * i] = p & 0xFFFFFFFFu;
        out[base + 16 + 2 * i + 1] = p >> 32;
    }
    // CARRY[0..7] at cols 24..31: carry when adding (i+1) to base_address low word.
    uint64_t base_lo = addr & 0xFFFFFFFFu;
    for (int i = 0; i < 7; ++i) {
        out[base + 24 + i] = (base_lo + (uint64_t)(i + 1) >= (1ull << 32)) ? 1u : 0u;
    }
    // OLD_TIMESTAMP[i] as DWordWL at cols 31 + 2i (8 timestamps → cols 31..47).
    for (int i = 0; i < 8; ++i) {
        uint64_t ot = op[11 + i];
        out[base + 31 + 2 * i] = ot & 0xFFFFFFFFu;
        out[base + 31 + 2 * i + 1] = ot >> 32;
    }
    out[base + 47] = is_read;                   // MU_READ
    out[base + 48] = 1u - is_read;              // MU_WRITE
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

// A1 — RESIDENT PAGE table fill: one thread per byte of a page, fill the 5 PAGE columns
// (OFFSET=0, INIT=1, FINI=2, TIMESTAMP_LO=3, TIMESTAMP_HI=4) directly on device from the sorted
// initial image + the device final-memory snapshot (with timestamps). Bit-identical to
// `generate_page_trace_from_dense` (exclude_touched=false): INIT = image byte (0 if absent); a byte
// present in the snapshot uses (snap_val, snap_ts); otherwise (init, ts=0). Row-major
// `buf[off*ncols + col]`, canonical-u64 field reprs (all values < 2^32). One kernel launch per page.
__device__ __forceinline__ uint64_t pf_bsearch(const uint64_t *keys, uint64_t n, uint64_t key,
                                               bool *found) {
    uint64_t lo = 0, hi = n, idx = 0;
    *found = false;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        uint64_t m = keys[mid];
        if (m == key) { idx = mid; *found = true; break; }
        else if (m < key) { lo = mid + 1; }
        else { hi = mid; }
    }
    return idx;
}
extern "C" __global__ void page_fill_snapshot(
    uint64_t page_base, uint64_t page_size, const uint64_t *img_addr, const uint64_t *img_val,
    uint64_t img_n, const uint64_t *snap_addr, const uint64_t *snap_val, const uint64_t *snap_ts,
    uint64_t snap_n, uint32_t ncols, uint64_t *buf) {
    uint64_t off = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (off >= page_size)
        return;
    uint64_t addr = page_base + off;
    bool f;
    uint64_t ii = pf_bsearch(img_addr, img_n, addr, &f);
    uint64_t iv = f ? (img_val[ii] & 0xFFull) : 0ull;
    uint64_t si = pf_bsearch(snap_addr, snap_n, addr, &f);
    uint64_t fv = f ? (snap_val[si] & 0xFFull) : iv;
    uint64_t ts = f ? snap_ts[si] : 0ull;
    // Match `generate_page_trace_from_dense` exactly: ts==0 emits (init, 0) — an untouched or
    // initial-image byte's final value equals its init value (NOT the stored snapshot value).
    if (ts == 0ull)
        fv = iv;
    uint64_t base = off * (uint64_t)ncols;
    buf[base + 0] = off;
    buf[base + 1] = iv;
    buf[base + 2] = fv;
    buf[base + 3] = ts & 0xFFFFFFFFull;
    buf[base + 4] = ts >> 32;
}
