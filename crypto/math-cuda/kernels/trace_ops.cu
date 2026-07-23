// Device CpuOperation builder (Phase 0 of full-GPU trace-gen).
//
// One thread per cycle reconstructs the per-cycle op record the prover's
// `CpuOperation::from_log` computes on the host — a stateless function of the cycle's
// executor Log (current_pc, next_pc, src1/src2/dst vals) and its decoded instruction
// (pc, imm, packed decode fields). Producing this SoA on-device is the seam every later
// trace-gen stage reads, so nothing returns to the CPU. Must match `from_log` bit-exact
// (byte-parity test in prover). See `prover/src/tables/cpu.rs::from_log`,
// `types.rs::{ShrunkDecode::pack, packed_decode_shrunk, alu_op}`, and
// `executor .../execution.rs` syscall numbers.

#include <cstdint>

// packed_decode_shrunk bit positions (types.rs).
#define PD_READ_REG1 0
#define PD_READ_REG2 1
#define PD_WORD_INSTR 3
#define PD_ALU 4
#define PD_ADD 5
#define PD_SUB 6
#define PD_MEMORY 7
#define PD_BRANCH 8
#define PD_ECALL 9
#define PD_RS1 10
#define PD_HIL 34
#define PD_ALU_FLAGS 42
#define PD_MEM_FLAGS 50

// Syscall numbers (executor execution.rs) — cpu.rs uses `execution::SyscallNumbers`.
#define SYS_COMMIT 64ull
#define SYS_KECCAK 0xFFFFFFFFFFFFFFFEull // u64::MAX - 1
#define SYS_ECSM 0xFFFFFFFFFFFFFFF5ull   // u64::MAX - 10

// alu_op (types.rs).
#define ALU_EQ 3
#define ALU_LT 4

__device__ __forceinline__ bool pd_bit(uint64_t p, uint32_t pos) {
    return ((p >> pos) & 1ull) != 0ull;
}
__device__ __forceinline__ uint8_t pd_byte(uint64_t p, uint32_t pos) {
    return (uint8_t)((p >> pos) & 0xFFull);
}

// Mirror of `CpuOperation::branch_taken`.
__device__ __forceinline__ bool branch_taken(uint8_t alu_flags, uint64_t rv1, uint64_t rv2) {
    uint8_t op = alu_flags & 0x1Fu;
    bool sgn = ((alu_flags >> 5) & 1u) != 0u;
    bool inv = ((alu_flags >> 6) & 1u) != 0u;
    bool cmp;
    if (op == ALU_EQ)
        cmp = (rv1 == rv2);
    else if (op == ALU_LT)
        cmp = sgn ? ((int64_t)rv1 < (int64_t)rv2) : (rv1 < rv2);
    else
        cmp = false;
    return cmp != inv; // cmp XOR inv
}

extern "C" __global__ void build_cpu_ops(
    uint64_t n, const uint64_t *current_pc, const uint64_t *next_pc_log,
    const uint64_t *src1_val, const uint64_t *src2_val, const uint64_t *dst_val,
    const uint64_t *pc, const uint64_t *imm, const uint64_t *packed,
    // outputs
    uint64_t *rv1_out, uint64_t *rv2_out, uint64_t *arg2_out, uint64_t *res_out,
    uint64_t *rvd_out, uint64_t *next_pc_out, uint8_t *flags_out,
    uint64_t *commit_buf_addr_out, uint64_t *commit_count_out,
    uint64_t *keccak_state_addr_out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t pk = packed[i];
    bool read_reg1 = pd_bit(pk, PD_READ_REG1);
    bool read_reg2 = pd_bit(pk, PD_READ_REG2);
    bool word_instr = pd_bit(pk, PD_WORD_INSTR);
    bool alu = pd_bit(pk, PD_ALU);
    bool add = pd_bit(pk, PD_ADD);
    bool sub = pd_bit(pk, PD_SUB);
    bool memory = pd_bit(pk, PD_MEMORY);
    bool branch = pd_bit(pk, PD_BRANCH);
    bool ecall = pd_bit(pk, PD_ECALL);
    uint8_t rs1 = pd_byte(pk, PD_RS1);
    uint8_t hil = pd_byte(pk, PD_HIL);
    uint8_t alu_flags = pd_byte(pk, PD_ALU_FLAGS);
    uint8_t mem_flags = pd_byte(pk, PD_MEM_FLAGS);

    uint64_t s1 = src1_val[i], s2 = src2_val[i], dv = dst_val[i], cpc = current_pc[i];
    uint64_t opc = pc[i], oimm = imm[i];
    uint64_t ilen = 2ull * (uint64_t)hil;

    // ECALL syscall classification (computed before the word-instr branch, like from_log;
    // word instructions never have ecall set, so these all resolve false for them).
    bool ec_commit = ecall && (s1 == SYS_COMMIT);
    bool ec_keccak = ecall && (s1 == SYS_KECCAK);
    bool ec_ecsm = ecall && (s1 == SYS_ECSM);
    commit_buf_addr_out[i] = ec_commit ? s2 : 0ull;
    commit_count_out[i] = ec_commit ? dv : 0ull;
    keccak_state_addr_out[i] = ec_keccak ? s2 : 0ull;

    uint64_t rv1, rv2, arg2 = 0, res = 0, rvd, npc;
    bool bcond = false;
    if (word_instr) {
        npc = opc + ilen;
        rv1 = s1;
        rv2 = read_reg2 ? s2 : 0ull;
        rvd = dv;
    } else {
        rv1 = (rs1 == 255) ? cpc : (read_reg1 ? s1 : 0ull);
        rv2 = read_reg2 ? s2 : 0ull;
        bool jalr = (mem_flags & 1u) != 0u;
        arg2 = memory ? oimm : (branch ? rv2 : (rv2 + oimm));
        bcond = branch ? (jalr ? true : branch_taken(alu_flags, rv1, rv2)) : false;
        if (add)
            res = rv1 + arg2;
        else if (sub)
            res = rv1 - arg2;
        else if (alu)
            res = branch ? (uint64_t)bcond : dv;
        else
            res = 0ull;
        bool store = memory && jalr;
        rvd = memory ? (store ? 0ull : dv) : (branch ? (opc + ilen) : res);
        npc = ecall ? (opc + ilen) : (bcond ? next_pc_log[i] : (opc + ilen));
    }

    rv1_out[i] = rv1;
    rv2_out[i] = rv2;
    arg2_out[i] = arg2;
    res_out[i] = res;
    rvd_out[i] = rvd;
    next_pc_out[i] = npc;
    flags_out[i] = (uint8_t)((bcond ? 1u : 0u) | (ec_commit ? 2u : 0u) | (ec_keccak ? 4u : 0u) |
                             (ec_ecsm ? 8u : 0u));
}

// --- Phase 3: state-free ALU chip-op classification + extraction ---
//
// The instruction-driven ALU chip ops derive purely from the resident cpu_op fields — no
// memory/register state, so they are the Phase-3 slice that validates on ethrex_5tx
// independent of the memory walk / precompiles. Six chips share the SAME raw gather triple
// (rv1, arg2, alu_flags byte): LT / SHIFT / EQ / BYTEWISE / MUL / DVRM. Each is a distinct
// route predicate on the decoded `alu_op` (all under `!word_instr && alu`). Must match the
// per-cycle `cpu_ops.iter().filter().map()` projections in trace_builder.rs (LT/SHIFT in
// `collect_ops_from_cpu`; EQ/BYTEWISE/MUL/DVRM in the generate pass). The host/fill
// reconstructs signed@bit5, signed2/invert@bit6, muldiv@bit7, alu_op@bits0-4 from alu_flags.
#define ALU_AND 0
#define ALU_OR 1
#define ALU_XOR 2
#define ALU_SHIFT 5
#define ALU_SHIFTW 6
#define ALU_MUL 7
#define ALU_DIVREM 8

// One thread per cycle: set the route flag (1 = emit) for each of the 6 state-free ALU chips.
extern "C" __global__ void chipop_alu_route(uint64_t n, const uint64_t *packed,
                                            uint32_t *flag_lt, uint32_t *flag_shift,
                                            uint32_t *flag_eq, uint32_t *flag_bytewise,
                                            uint32_t *flag_mul, uint32_t *flag_dvrm) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool nw_alu = !pd_bit(pk, PD_WORD_INSTR) && pd_bit(pk, PD_ALU);
    uint8_t op = pd_byte(pk, PD_ALU_FLAGS) & 0x1Fu;
    flag_lt[i] = (nw_alu && op == ALU_LT) ? 1u : 0u;
    flag_shift[i] = (nw_alu && (op == ALU_SHIFT || op == ALU_SHIFTW)) ? 1u : 0u;
    flag_eq[i] = (nw_alu && op == ALU_EQ) ? 1u : 0u;
    flag_bytewise[i] = (nw_alu && (op == ALU_AND || op == ALU_OR || op == ALU_XOR)) ? 1u : 0u;
    flag_mul[i] = (nw_alu && op == ALU_MUL) ? 1u : 0u;
    flag_dvrm[i] = (nw_alu && op == ALU_DIVREM) ? 1u : 0u;
}

// Gather matched cycles into a compacted SoA, in program order (excl = stable exclusive
// prefix of flag). out_a = rv1, out_b = arg2, out_f = alu_flags byte.
extern "C" __global__ void chipop_gather(uint64_t n, const uint64_t *in_a,
                                         const uint64_t *in_b, const uint64_t *packed,
                                         const uint32_t *flag, const uint64_t *excl,
                                         uint64_t *out_a, uint64_t *out_b, uint8_t *out_f) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (flag[i]) {
        uint64_t p = excl[i];
        out_a[p] = in_a[i];
        out_b[p] = in_b[i];
        out_f[p] = pd_byte(packed[i], PD_ALU_FLAGS);
    }
}

// BRANCH and STORE chip ops — also pure state-free projections of cpu_ops, but with their
// own field sets (so a separate 4-column gather). Match trace_builder.rs:
//   BRANCH: filter `op.branch_cond` → BranchOperation::new(pc, imm, rv1, jalr)
//   STORE:  filter `is_store()`     → StoreOperation::new(res, timestamp, rv2, mem_bytes)
// Route: `branch_cond` is a cpu_op output (flags bit 0, from build_cpu_ops); `is_store` =
// `memory && (mem_flags bit 0)`, a decode predicate.
extern "C" __global__ void chipop_branch_store_route(uint64_t n, const uint64_t *packed,
                                                     const uint8_t *cpu_flags,
                                                     uint32_t *flag_branch,
                                                     uint32_t *flag_store) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool memory = pd_bit(pk, PD_MEMORY);
    bool mem_op = (pd_byte(pk, PD_MEM_FLAGS) & 1u) != 0u;
    flag_branch[i] = (cpu_flags[i] & 1u) ? 1u : 0u; // branch_cond
    flag_store[i] = (memory && mem_op) ? 1u : 0u;
}

// Generic 4-column gather (BRANCH: pc/imm/rv1/packed; STORE: res/ts/rv2/packed). The
// consumer extracts jalr / mem_bytes from the gathered `packed` (col3) — pure decode logic.
extern "C" __global__ void chipop_gather4(uint64_t n, const uint64_t *in0, const uint64_t *in1,
                                          const uint64_t *in2, const uint64_t *in3,
                                          const uint32_t *flag, const uint64_t *excl,
                                          uint64_t *out0, uint64_t *out1, uint64_t *out2,
                                          uint64_t *out3) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (flag[i]) {
        uint64_t p = excl[i];
        out0[p] = in0[i];
        out1[p] = in1[i];
        out2[p] = in2[i];
        out3[p] = in3[i];
    }
}

// --- Phase 3c: CPU32 (word `*W`) chip-op generation on device ---
//
// CPU32 rows are state-free projections of cpu_ops, but the fill consumes the pre-computed
// `res` (pack_cpu32_op packs op.res), so the device builder must compute `res` too — via
// compute_aux (32-bit operand sign-extension) then cpu32_res (SHIFT/MUL/DVRM arithmetic).
// The result math below is ported bit-for-bit from the validated shift_fill / mul_fill /
// dvrm_fill kernels (trace_cpu.cu). Output is the 8-u64 pack_cpu32_op row, so the builder
// feeds cpu32_fill directly. Must match trace_builder.rs::{build_cpu32_op, cpu32_res,
// compute_aux}.
#define HI_FILL 0xFFFFFFFFull

// SHIFT compute_out (word_instr = true). Mirrors shift_fill's out0|(out1<<32).
__device__ uint64_t cpu32_shift_out(uint64_t value, uint64_t shift_amount, uint32_t direction,
                                    uint32_t is_signed) {
    uint16_t in_h[4];
    in_h[0] = (uint16_t)(value & 0xFFFFu);
    in_h[1] = (uint16_t)((value >> 16) & 0xFFFFu);
    in_h[2] = (uint16_t)((value >> 32) & 0xFFFFu);
    in_h[3] = (uint16_t)((value >> 48) & 0xFFFFu);
    uint8_t shift = (uint8_t)(shift_amount & 0xFFu);
    uint32_t left = 1u - (direction & 1u);
    uint32_t right = direction & 1u;
    uint32_t is_negative = (is_signed && ((in_h[3] >> 15) & 1u)) ? 1u : 0u;
    uint16_t extension = is_negative ? (uint16_t)0xFFFF : (uint16_t)0;
    uint8_t bit_shift = left ? (uint8_t)(shift & 15u) : (uint8_t)((256u - (uint32_t)shift) & 15u);
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
            x[i] = (uint16_t)((uint32_t)in_h[i] << bit_shift);
            y[i] = (uint16_t)(in_h[i] >> (16u - (uint32_t)bit_shift));
        }
        x[4] = (uint16_t)((uint32_t)extension << bit_shift);
    }
    // word_instr = true → limb_idx = (shift >> 4) & 1.
    uint32_t limb_idx = (uint32_t)((shift >> 4) & 1u);
    uint32_t ls[4] = {0, 0, 0, 0};
    ls[limb_idx] = 1u;
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
    uint64_t out0 = (uint64_t)((uint32_t)shifted[0] | ((uint32_t)shifted[1] << 16));
    uint64_t out1 = (uint64_t)((uint32_t)shifted[2] | ((uint32_t)shifted[3] << 16));
    return out0 | (out1 << 32);
}

// MUL compute_product().0 (low 64 bits). Mirrors mul_fill.
__device__ uint64_t cpu32_mul_lo(uint64_t lhs, uint32_t lhs_signed, uint64_t rhs,
                                 uint32_t rhs_signed) {
    __int128 a = lhs_signed ? (__int128)(int64_t)lhs : (__int128)lhs;
    __int128 b = rhs_signed ? (__int128)(int64_t)rhs : (__int128)rhs;
    __int128 product = a * b;
    return (uint64_t)product;
}

// DVRM quotient / remainder. Mirrors dvrm_fill (div-by-zero, signed overflow special cases).
__device__ void cpu32_dvrm(uint64_t n, uint64_t d, uint32_t is_signed, uint64_t *q,
                           uint64_t *rem) {
    uint32_t div_by_zero = (d == 0ull) ? 1u : 0u;
    uint32_t overflow =
        (is_signed && n == 0x8000000000000000ull && d == 0xFFFFFFFFFFFFFFFFull) ? 1u : 0u;
    if (div_by_zero) {
        *q = 0xFFFFFFFFFFFFFFFFull;
        *rem = n;
    } else if (overflow) {
        *q = n;
        *rem = 0ull;
    } else if (is_signed) {
        int64_t ni = (int64_t)n, di = (int64_t)d;
        *q = (uint64_t)(ni / di);
        *rem = (uint64_t)(ni % di);
    } else {
        *q = n / d;
        *rem = n % d;
    }
}

// cpu32_res: add/sub fast-path, else SHIFT/MUL/DVRM on the sign-extended (arg1, arg2).
__device__ uint64_t cpu32_res(uint32_t add, uint32_t sub, uint32_t alu, uint8_t alu_flags,
                              uint64_t arg1, uint64_t arg2) {
    if (add)
        return arg1 + arg2;
    if (sub)
        return arg1 - arg2;
    if (!alu)
        return 0ull;
    uint8_t op = alu_flags & 0x1Fu;
    uint32_t is_signed = (alu_flags >> 5) & 1u;
    uint32_t s2 = (alu_flags >> 6) & 1u;
    uint32_t muldiv = (alu_flags >> 7) & 1u;
    if (op == ALU_SHIFT || op == ALU_SHIFTW)
        return cpu32_shift_out(arg1, arg2, s2, is_signed);
    if (op == ALU_MUL)
        return cpu32_mul_lo(arg1, is_signed, arg2, s2);
    if (op == ALU_DIVREM) {
        uint64_t q, rem;
        cpu32_dvrm(arg1, arg2, is_signed, &q, &rem);
        return muldiv ? rem : q;
    }
    return 0ull;
}

// Route: flag word-instruction cycles (they delegate to CPU32).
extern "C" __global__ void cpu32_route(uint64_t n, const uint64_t *packed, uint32_t *flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    flag[i] = pd_bit(packed[i], PD_WORD_INSTR) ? 1u : 0u;
}

// Build the 8-u64 pack_cpu32_op row per word-instr cycle, compacted (excl = prefix of flag).
// ts = i*4+4 (matches from_log_and_instruction). `pc` is decode.pc (= current_pc).
extern "C" __global__ void build_cpu32_ops(uint64_t n, const uint64_t *packed,
                                           const uint64_t *rv1, const uint64_t *rv2,
                                           const uint64_t *imm, const uint64_t *pc,
                                           const uint32_t *flag, const uint64_t *excl,
                                           uint64_t *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint64_t pk = packed[i];
    uint32_t rr1 = pd_bit(pk, PD_READ_REG1) ? 1u : 0u;
    uint32_t rr2 = pd_bit(pk, PD_READ_REG2) ? 1u : 0u;
    uint32_t wr = pd_bit(pk, 2) ? 1u : 0u; // PD_WRITE_REG = 2
    uint32_t alu = pd_bit(pk, PD_ALU) ? 1u : 0u;
    uint32_t add = pd_bit(pk, PD_ADD) ? 1u : 0u;
    uint32_t sub = pd_bit(pk, PD_SUB) ? 1u : 0u;
    uint8_t rs1 = pd_byte(pk, PD_RS1);
    uint8_t rs2 = pd_byte(pk, 18);           // PD_RS2 = 18
    uint8_t rd = pd_byte(pk, 26);            // PD_RD = 26
    uint8_t alu_flags = pd_byte(pk, PD_ALU_FLAGS);
    uint8_t hil = pd_byte(pk, PD_HIL);

    uint64_t v1 = rv1[i], v2 = rv2[i], im = imm[i];
    // compute_aux (res field of the op is 0 here; arg1/arg2 don't depend on it).
    uint32_t is_signed = (alu_flags >> 5) & 1u;
    uint32_t rv1_sign = (is_signed && ((v1 >> 31) & 1u)) ? 1u : 0u;
    uint32_t rv2_sign = (is_signed && ((v2 >> 31) & 1u)) ? 1u : 0u;
    uint64_t arg1 = (v1 & 0xFFFFFFFFull) | (rv1_sign ? (HI_FILL << 32) : 0ull);
    uint64_t arg2_lo = (v2 & 0xFFFFFFFFull) + (im & 0xFFFFFFFFull);
    uint64_t arg2_hi = (rv2_sign ? HI_FILL : 0ull) + (im >> 32);
    uint64_t arg2 = (arg2_lo & 0xFFFFFFFFull) | (arg2_hi << 32);
    uint64_t res = cpu32_res(add, sub, alu, alu_flags, arg1, arg2);

    uint64_t flags = (uint64_t)rr1 | ((uint64_t)rr2 << 1) | ((uint64_t)wr << 2) |
                     ((uint64_t)alu << 3) | ((uint64_t)add << 4) | ((uint64_t)sub << 5);
    uint64_t bytes = (uint64_t)rs1 | ((uint64_t)rs2 << 8) | ((uint64_t)rd << 16) |
                     ((uint64_t)alu_flags << 24) | ((uint64_t)hil << 32);

    uint64_t *o = out + excl[i] * 8ull;
    o[0] = i * 4ull + 4ull; // timestamp
    o[1] = pc[i];
    o[2] = v1;
    o[3] = v2;
    o[4] = im;
    o[5] = res;
    o[6] = flags;
    o[7] = bytes;
}

// --- Phase 3d: LOAD chip-op generation on device ---
//
// The LOAD chip table row is a pure state-free projection of cpu_ops (base = res, ts, width,
// signed, res_bytes = the sign/zero-extended loaded value). Only the MEMW *read row*'s old_ts
// needs the memory walk (Phase 2), not this. Matches collect_load_op_from_cpu's LoadOperation
// + pack_load_op (LOAD_STRIDE = 7).
extern "C" __global__ void load_route(uint64_t n, const uint64_t *packed, uint32_t *flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool is_load = pd_bit(pk, PD_MEMORY) && ((pd_byte(pk, PD_MEM_FLAGS) & 1u) == 0u);
    flag[i] = is_load ? 1u : 0u;
}

extern "C" __global__ void build_load_ops(uint64_t n, const uint64_t *packed,
                                          const uint64_t *res_arr, const uint64_t *rvd,
                                          const uint32_t *flag, const uint64_t *excl,
                                          uint64_t *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t mf = pd_byte(packed[i], PD_MEM_FLAGS);
    // mem_bytes: MEM_FLAGS_8B=4, _4B=3, _2B=2 (default 1). mem_signed: bit 1.
    uint32_t width = ((mf >> 4) & 1u) ? 8u : ((mf >> 3) & 1u) ? 4u : ((mf >> 2) & 1u) ? 2u : 1u;
    uint32_t is_signed = (mf >> 1) & 1u;
    uint64_t loaded = rvd[i];
    uint32_t rb[8];
    for (uint32_t j = 0; j < 8u; ++j)
        rb[j] = (j < width) ? (uint32_t)((loaded >> (8u * j)) & 0xFFu) : 0u;
    if (width < 8u) {
        uint32_t sign_bit = (rb[width - 1] >> 7) & 1u;
        uint32_t fill = (is_signed && sign_bit) ? 0xFFu : 0u;
        for (uint32_t j = width; j < 8u; ++j)
            rb[j] = fill;
    }
    uint64_t flags = (uint64_t)is_signed | ((uint64_t)width << 8);
    uint64_t *o = out + excl[i] * 7ull;
    o[0] = flags;
    o[1] = res_arr[i];
    o[2] = i * 4ull + 4ull;
    o[3] = (uint64_t)rb[0] | ((uint64_t)rb[1] << 32);
    o[4] = (uint64_t)rb[2] | ((uint64_t)rb[3] << 32);
    o[5] = (uint64_t)rb[4] | ((uint64_t)rb[5] << 32);
    o[6] = (uint64_t)rb[6] | ((uint64_t)rb[7] << 32);
}

// --- Resident LT chain helpers: key gather (→ dedup) ---
// LT dedup key = (flags = signed|invert<<1, lhs=rv1, rhs=arg2) — the CANONICAL LtOperation
// discriminator (NOT the raw alu_flags byte), so instruction-driven LT ops merge correctly
// with derived LT ops (e.g. dvrm-derived, which have signed=invert=0 → flags=0). k0 = flags
// is already the final pack value, so the generic `dedup_pack_abf` produces the LT row.
extern "C" __global__ void lt_key_gather(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                         const uint64_t *arg2, const uint32_t *flag,
                                         const uint64_t *excl, uint64_t *k0, uint64_t *k1,
                                         uint64_t *k2) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint64_t p = excl[i];
    k0[p] = (uint64_t)(((af >> 5) & 1u) | (((af >> 6) & 1u) << 1)); // signed | invert<<1
    k1[p] = rv1[i];
    k2[p] = arg2[i];
}

// Derived LT ops from DVRM: for each is_divrem cycle, `LtOperation::new(abs_r, abs_d, false)`
// (signed=invert=false → flags=0). abs_r = |remainder|, abs_d = |d|, using the DVRM sign
// convention (matches trace_builder.rs line "lt_ops.push(LtOperation::new(op.abs_r(),
// op.abs_d(), false))"). Written at `out_base + excl[i]` so it can append to the
// instruction-driven LT key stream for a single merged dedup.
extern "C" __global__ void dvrm_lt_key_gather(uint64_t n, const uint64_t *packed,
                                              const uint64_t *rv1, const uint64_t *arg2,
                                              const uint32_t *flag, const uint64_t *excl,
                                              uint64_t out_base, uint64_t *k0, uint64_t *k1,
                                              uint64_t *k2) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint32_t is_signed = (af >> 5) & 1u;
    uint64_t nn = rv1[i], d = arg2[i];
    uint64_t q, rem;
    cpu32_dvrm(nn, d, is_signed, &q, &rem);
    uint32_t sign_r = (is_signed && (rem >> 63)) ? 1u : 0u;
    uint32_t sign_d = (is_signed && (d >> 63)) ? 1u : 0u;
    uint64_t abs_r = sign_r ? (0ull - rem) : rem;
    uint64_t abs_d = sign_d ? (0ull - d) : d;
    uint64_t p = out_base + excl[i];
    k0[p] = 0ull; // signed=false, invert=false
    k1[p] = abs_r;
    k2[p] = abs_d;
}

// memw→lt keys (LT-resident-table STEP 2B): append the memw→lt operand pairs (lhs=old_ts, rhs=ts)
// to the merged LT key stream at out_base+i, with k0=0 (unsigned, invert=0 — same convention as
// dvrm→lt). LT dedup key = (k0=signed|invert<<1, k1=lhs, k2=rhs).
extern "C" __global__ void lt_memw_key_write(uint64_t n_memw, const uint64_t *memw_lhs,
                                             const uint64_t *memw_rhs, uint64_t out_base,
                                             uint64_t *k0, uint64_t *k1, uint64_t *k2) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_memw)
        return;
    uint64_t p = out_base + i;
    k0[p] = 0ull;
    k1[p] = memw_lhs[i];
    k2[p] = memw_rhs[i];
}

// EQ dedup key = (invert only, a=rv1, b=arg2). EqOperation stores just `invert` (signedness
// is irrelevant for equality), so k0 must be the invert bit ALONE — keying on the full
// alu_flags would wrongly split ops that differ only in the (unused) signed bit.
extern "C" __global__ void eq_key_gather(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                         const uint64_t *arg2, const uint32_t *flag,
                                         const uint64_t *excl, uint64_t *k0, uint64_t *k1,
                                         uint64_t *k2) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint64_t p = excl[i];
    k0[p] = (uint64_t)((pd_byte(packed[i], PD_ALU_FLAGS) >> 6) & 1u); // invert
    k1[p] = rv1[i];
    k2[p] = arg2[i];
}

// BYTEWISE dedup key = (alu_op, a=rv1, b=arg2). BytewiseOperation stores `op` = alu_op (0-4
// = AND/OR/XOR), so k0 = the alu_op bits.
extern "C" __global__ void bytewise_key_gather(uint64_t n, const uint64_t *packed,
                                               const uint64_t *rv1, const uint64_t *arg2,
                                               const uint32_t *flag, const uint64_t *excl,
                                               uint64_t *k0, uint64_t *k1, uint64_t *k2) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint64_t p = excl[i];
    k0[p] = (uint64_t)(pd_byte(packed[i], PD_ALU_FLAGS) & 0x1Fu); // alu_op
    k1[p] = rv1[i];
    k2[p] = arg2[i];
}

// Generic dedup pack for chips whose fill stride is [a, b, flags, mult] (EQ, and any chip
// whose key-gather already emits k0 = the final flags value): out[r] = [uk1, uk2, uk0, mult].
extern "C" __global__ void dedup_pack_abf(uint64_t m, const uint64_t *uk0, const uint64_t *uk1,
                                          const uint64_t *uk2, const uint64_t *umult,
                                          uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m)
        return;
    uint64_t *o = out + r * 4ull;
    o[0] = uk1[r];
    o[1] = uk2[r];
    o[2] = uk0[r];
    o[3] = umult[r];
}

// Dual-multiplicity pack (MUL / DVRM stride 5): [a, b, flags, mult0, mult1].
// MUL: mult0=mu_lo, mult1=mu_hi. DVRM: mult0=mu_q, mult1=mu_r.
extern "C" __global__ void dedup_pack_abf2(uint64_t m, const uint64_t *uk0, const uint64_t *uk1,
                                           const uint64_t *uk2, const uint64_t *um0,
                                           const uint64_t *um1, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m)
        return;
    uint64_t *o = out + r * 5ull;
    o[0] = uk1[r];
    o[1] = uk2[r];
    o[2] = uk0[r];
    o[3] = um0[r];
    o[4] = um1[r];
}

// MUL dedup key = (flags = lhs_signed|rhs_signed<<1, lhs=rv1, rhs=arg2); selector = muldiv
// (alu_flags bit7 → mu_lo vs mu_hi). Matches MulOperation::new(rv1, alu_signed, arg2,
// alu_signed2_or_invert) + wants_hi = alu_muldiv().
extern "C" __global__ void mul_key_gather(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                          const uint64_t *arg2, const uint32_t *flag,
                                          const uint64_t *excl, uint64_t *k0, uint64_t *k1,
                                          uint64_t *k2, uint32_t *sel) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint64_t p = excl[i];
    k0[p] = (uint64_t)(((af >> 5) & 1u) | (((af >> 6) & 1u) << 1)); // lhs_signed|rhs_signed<<1
    k1[p] = rv1[i];
    k2[p] = arg2[i];
    sel[p] = (af >> 7) & 1u; // muldiv → wants_hi
}

// Derived MUL ops from DVRM: for each is_divrem cycle, `MulOperation::new(d, d_signed, q,
// q_signed)` contributes to BOTH mu_lo (C13) and mu_hi (C14) — so emit two entries (sel=0,
// sel=1) with the same key. d=arg2, d_signed=signed, q=compute_quotient, q_signed = signed &&
// !overflow. Written at `out_base + 2*excl[i]` so it appends to the instruction-MUL stream.
extern "C" __global__ void mul_dvrm_key_gather(uint64_t n, const uint64_t *packed,
                                               const uint64_t *rv1, const uint64_t *arg2,
                                               const uint32_t *flag, const uint64_t *excl,
                                               uint64_t out_base, uint64_t *k0, uint64_t *k1,
                                               uint64_t *k2, uint32_t *sel) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint32_t is_signed = (af >> 5) & 1u;
    uint64_t nn = rv1[i], d = arg2[i];
    uint64_t q, rem;
    cpu32_dvrm(nn, d, is_signed, &q, &rem);
    uint32_t overflow =
        (is_signed && nn == 0x8000000000000000ull && d == 0xFFFFFFFFFFFFFFFFull) ? 1u : 0u;
    uint32_t sign_q = (is_signed && !overflow) ? 1u : 0u;
    uint64_t flags = (uint64_t)is_signed | ((uint64_t)sign_q << 1); // d_signed | q_signed<<1
    uint64_t base = out_base + 2ull * excl[i];
    k0[base] = flags;
    k1[base] = d;
    k2[base] = q;
    sel[base] = 0u; // C13 lo
    k0[base + 1] = flags;
    k1[base + 1] = d;
    k2[base + 1] = q;
    sel[base + 1] = 1u; // C14 hi
}

// DVRM dedup key = (flags = signed, n=rv1, d=arg2); selector = muldiv (bit7 → mu_q vs mu_r).
extern "C" __global__ void dvrm_key_gather(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                           const uint64_t *arg2, const uint32_t *flag,
                                           const uint64_t *excl, uint64_t *k0, uint64_t *k1,
                                           uint64_t *k2, uint32_t *sel) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint64_t p = excl[i];
    k0[p] = (uint64_t)((af >> 5) & 1u); // signed
    k1[p] = rv1[i];
    k2[p] = arg2[i];
    sel[p] = (af >> 7) & 1u; // muldiv → wants_remainder
}

// SHIFT is per-row (no dedup). Build the 3-u64 pack_shift_op row per is_shift cycle:
// [value=rv1, shift_amount=arg2, flags=direction|signed<<1|word_instr<<2]. These are the
// !word instruction shifts, so word_instr=0; direction=alu_flags bit6, signed=bit5.
extern "C" __global__ void build_shift_ops(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                           const uint64_t *arg2, const uint32_t *flag,
                                           const uint64_t *excl, uint64_t *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint64_t flags = (uint64_t)((af >> 6) & 1u) | ((uint64_t)((af >> 5) & 1u) << 1);
    uint64_t *o = out + excl[i] * 3ull;
    o[0] = rv1[i];
    o[1] = arg2[i];
    o[2] = flags;
}

// cpu32-derived SHIFT source: word instructions that dispatch to the SHIFT chip
// (cpu32_chip_op). Route = word_instr && alu && !add && !sub && (SHIFT|SHIFTW). The op is
// ShiftOperation::new(arg1, arg2, s2_or_inv, signed, /*word=*/true) on the CPU32 aux operands.
extern "C" __global__ void cpu32_shift_route(uint64_t n, const uint64_t *packed, uint32_t *flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool word = pd_bit(pk, PD_WORD_INSTR);
    bool alu = pd_bit(pk, PD_ALU);
    bool addsub = pd_bit(pk, PD_ADD) || pd_bit(pk, PD_SUB);
    uint8_t op = pd_byte(pk, PD_ALU_FLAGS) & 0x1Fu;
    bool is_shift = (op == ALU_SHIFT || op == ALU_SHIFTW);
    flag[i] = (word && alu && !addsub && is_shift) ? 1u : 0u;
}

// Emit the pack_shift_op row for each cpu32-shift cycle, at out_base+excl[i]. value=arg1,
// shift_amount=arg2 (CPU32 32-bit sign-extended aux operands), flags = direction|signed<<1|
// word_instr(=1)<<2. compute_aux inlined (mirrors build_cpu32_ops).
extern "C" __global__ void cpu32_shift_ops(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                           const uint64_t *rv2, const uint64_t *imm,
                                           const uint32_t *flag, const uint64_t *excl,
                                           uint64_t out_base, uint64_t *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint32_t is_signed = (af >> 5) & 1u;
    uint64_t v1 = rv1[i], v2 = rv2[i], im = imm[i];
    uint32_t rv1_sign = (is_signed && ((v1 >> 31) & 1u)) ? 1u : 0u;
    uint32_t rv2_sign = (is_signed && ((v2 >> 31) & 1u)) ? 1u : 0u;
    uint64_t arg1 = (v1 & 0xFFFFFFFFull) | (rv1_sign ? (HI_FILL << 32) : 0ull);
    uint64_t arg2_lo = (v2 & 0xFFFFFFFFull) + (im & 0xFFFFFFFFull);
    uint64_t arg2_hi = (rv2_sign ? HI_FILL : 0ull) + (im >> 32);
    uint64_t arg2 = (arg2_lo & 0xFFFFFFFFull) | (arg2_hi << 32);
    uint32_t s2 = (af >> 6) & 1u; // direction
    uint64_t flags = (uint64_t)s2 | ((uint64_t)is_signed << 1) | (1ull << 2); // word_instr = 1
    uint64_t *o = out + (out_base + excl[i]) * 3ull;
    o[0] = arg1;
    o[1] = arg2;
    o[2] = flags;
}

// cpu32-derived MUL/DVRM routes + derive kernels (mirror the cpu32-shift pair). Route =
// word_instr && alu && !add && !sub && (MUL | DIVREM). Op = MulOperation::new(arg1, signed,
// arg2, s2_or_inv) / DvrmOperation::new(arg1, arg2, signed), selector = muldiv.
extern "C" __global__ void cpu32_mul_route(uint64_t n, const uint64_t *packed, uint32_t *flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool ok = pd_bit(pk, PD_WORD_INSTR) && pd_bit(pk, PD_ALU) && !pd_bit(pk, PD_ADD) &&
              !pd_bit(pk, PD_SUB);
    flag[i] = (ok && (pd_byte(pk, PD_ALU_FLAGS) & 0x1Fu) == ALU_MUL) ? 1u : 0u;
}
extern "C" __global__ void cpu32_dvrm_route(uint64_t n, const uint64_t *packed, uint32_t *flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool ok = pd_bit(pk, PD_WORD_INSTR) && pd_bit(pk, PD_ALU) && !pd_bit(pk, PD_ADD) &&
              !pd_bit(pk, PD_SUB);
    flag[i] = (ok && (pd_byte(pk, PD_ALU_FLAGS) & 0x1Fu) == ALU_DIVREM) ? 1u : 0u;
}

// Shared compute_aux for the cpu32-derived MUL/DVRM key gathers.
__device__ __forceinline__ void cpu32_aux(uint64_t v1, uint64_t v2, uint64_t im, uint8_t af,
                                          uint64_t *arg1, uint64_t *arg2) {
    uint32_t is_signed = (af >> 5) & 1u;
    uint32_t rv1_sign = (is_signed && ((v1 >> 31) & 1u)) ? 1u : 0u;
    uint32_t rv2_sign = (is_signed && ((v2 >> 31) & 1u)) ? 1u : 0u;
    *arg1 = (v1 & 0xFFFFFFFFull) | (rv1_sign ? (HI_FILL << 32) : 0ull);
    uint64_t lo = (v2 & 0xFFFFFFFFull) + (im & 0xFFFFFFFFull);
    uint64_t hi = (rv2_sign ? HI_FILL : 0ull) + (im >> 32);
    *arg2 = (lo & 0xFFFFFFFFull) | (hi << 32);
}

extern "C" __global__ void cpu32_mul_ops(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                         const uint64_t *rv2, const uint64_t *imm,
                                         const uint32_t *flag, const uint64_t *excl,
                                         uint64_t out_base, uint64_t *k0, uint64_t *k1,
                                         uint64_t *k2, uint32_t *sel) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint64_t arg1, arg2;
    cpu32_aux(rv1[i], rv2[i], imm[i], af, &arg1, &arg2);
    uint64_t p = out_base + excl[i];
    k0[p] = (uint64_t)(((af >> 5) & 1u) | (((af >> 6) & 1u) << 1)); // lhs_signed|rhs_signed<<1
    k1[p] = arg1;
    k2[p] = arg2;
    sel[p] = (af >> 7) & 1u; // muldiv
}

extern "C" __global__ void cpu32_dvrm_ops(uint64_t n, const uint64_t *packed, const uint64_t *rv1,
                                          const uint64_t *rv2, const uint64_t *imm,
                                          const uint32_t *flag, const uint64_t *excl,
                                          uint64_t out_base, uint64_t *k0, uint64_t *k1,
                                          uint64_t *k2, uint32_t *sel) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint64_t arg1, arg2;
    cpu32_aux(rv1[i], rv2[i], imm[i], af, &arg1, &arg2);
    uint64_t p = out_base + excl[i];
    k0[p] = (uint64_t)((af >> 5) & 1u); // signed
    k1[p] = arg1;
    k2[p] = arg2;
    sel[p] = (af >> 7) & 1u; // muldiv
}

// Derived MUL ops from cpu32-derived DVRM: the C13/C14 dvrm→mul derivation applied to the
// cpu32 word-dvrm ops (MUL's 4th source). For each cpu32-dvrm cycle: aux → (arg1, arg2), then
// MulOperation::new(d=arg2, d_signed=signed, q=quotient(arg1,arg2), q_signed=signed&&!overflow),
// contributing to both mu_lo and mu_hi (2 entries). Written at out_base + 2*excl[i].
extern "C" __global__ void cpu32_dvrm_mul_key_gather(uint64_t n, const uint64_t *packed,
                                                     const uint64_t *rv1, const uint64_t *rv2,
                                                     const uint64_t *imm, const uint32_t *flag,
                                                     const uint64_t *excl, uint64_t out_base,
                                                     uint64_t *k0, uint64_t *k1, uint64_t *k2,
                                                     uint32_t *sel) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t af = pd_byte(packed[i], PD_ALU_FLAGS);
    uint32_t is_signed = (af >> 5) & 1u;
    uint64_t arg1, arg2;
    cpu32_aux(rv1[i], rv2[i], imm[i], af, &arg1, &arg2);
    uint64_t q, rem;
    cpu32_dvrm(arg1, arg2, is_signed, &q, &rem);
    uint32_t overflow =
        (is_signed && arg1 == 0x8000000000000000ull && arg2 == 0xFFFFFFFFFFFFFFFFull) ? 1u : 0u;
    uint32_t sign_q = (is_signed && !overflow) ? 1u : 0u;
    uint64_t flags = (uint64_t)is_signed | ((uint64_t)sign_q << 1); // d_signed | q_signed<<1
    uint64_t base = out_base + 2ull * excl[i];
    k0[base] = flags;
    k1[base] = arg2; // lhs = d
    k2[base] = q;    // rhs = q
    sel[base] = 0u;  // C13 lo
    k0[base + 1] = flags;
    k1[base + 1] = arg2;
    k2[base + 1] = q;
    sel[base + 1] = 1u; // C14 hi
}

// BRANCH dedup key = (pc, offset=imm, register=rv1, jalr). 4 fields → dedup4. jalr = mem_flags
// bit0. Matches BranchOperation::new(pc, imm, rv1, jalr) routed by branch_cond.
extern "C" __global__ void branch_key_gather(uint64_t n, const uint64_t *packed,
                                             const uint64_t *pc, const uint64_t *imm,
                                             const uint64_t *rv1, const uint32_t *flag,
                                             const uint64_t *excl, uint64_t *k0, uint64_t *k1,
                                             uint64_t *k2, uint64_t *k3) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint64_t p = excl[i];
    k0[p] = (uint64_t)(pd_byte(packed[i], PD_MEM_FLAGS) & 1u); // jalr
    k1[p] = rv1[i];                                            // register
    k2[p] = imm[i];                                            // offset
    k3[p] = pc[i];
}

// BRANCH pack (stride 5): pack_branch_op = [pc, offset, register, jalr, mult].
// uk0=jalr, uk1=register, uk2=offset, uk3=pc.
extern "C" __global__ void branch_pack(uint64_t m, const uint64_t *uk0, const uint64_t *uk1,
                                       const uint64_t *uk2, const uint64_t *uk3,
                                       const uint64_t *umult, uint64_t *out) {
    uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m)
        return;
    uint64_t *o = out + r * 5ull;
    o[0] = uk3[r]; // pc
    o[1] = uk2[r]; // offset
    o[2] = uk1[r]; // register
    o[3] = uk0[r]; // jalr
    o[4] = umult[r];
}

// --- Phase 3e: STORE chip-op generation on device (per-row, state-free) ---
// Matches collect_store_op_from_cpu's StoreOperation + pack_store_op (STORE_STRIDE = 4):
// [0]=flags(write2|write4<<1|write8<<2), [1]=base(=res), [2]=timestamp, [3]=value(=rv2).
extern "C" __global__ void store_route(uint64_t n, const uint64_t *packed, uint32_t *flag) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t pk = packed[i];
    bool is_store = pd_bit(pk, PD_MEMORY) && ((pd_byte(pk, PD_MEM_FLAGS) & 1u) != 0u);
    flag[i] = is_store ? 1u : 0u;
}

extern "C" __global__ void build_store_ops(uint64_t n, const uint64_t *packed,
                                           const uint64_t *res_arr, const uint64_t *rv2,
                                           const uint32_t *flag, const uint64_t *excl,
                                           uint64_t *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flag[i])
        return;
    uint8_t mf = pd_byte(packed[i], PD_MEM_FLAGS);
    uint32_t width = ((mf >> 4) & 1u) ? 8u : ((mf >> 3) & 1u) ? 4u : ((mf >> 2) & 1u) ? 2u : 1u;
    uint64_t flags = (uint64_t)(width == 2u ? 1u : 0u) | ((uint64_t)(width == 4u ? 1u : 0u) << 1) |
                     ((uint64_t)(width == 8u ? 1u : 0u) << 2);
    uint64_t *o = out + excl[i] * 4ull;
    o[0] = flags;
    o[1] = res_arr[i];
    o[2] = i * 4ull + 4ull;
    o[3] = rv2[i];
}
