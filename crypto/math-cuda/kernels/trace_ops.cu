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
