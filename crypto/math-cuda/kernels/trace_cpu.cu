// GPU CPU-table trace generation.
//
// One thread per row. Reads the executor `Log` for the row plus the program's
// pre-decoded fields (`PackedDecode`, built once on the host from the ELF), and
// writes the 38 CPU-table columns directly in COLUMN-MAJOR layout
// (`cols[col*nrows + row]`) so the buffer feeds the LDE with no transpose.
//
// Mirrors `CpuOperation::from_log` + `generate_cpu_trace` in
// `prover/src/tables/cpu.rs` — same output bytes, fused into a single pass.
// Goldilocks here is direct canonical u64 (no Montgomery), and every CPU-table
// value is < p (32-bit words, 16-bit halves, bytes, bools, ts < 2^42), so no
// field reduction is needed: the kernel writes raw u64.

#include <cstdint>

// --- CPU table column indices (must match prover/src/tables/cpu.rs `cols`) ---
#define C_TIMESTAMP 0
#define C_PC_0 1
#define C_PC_1 2
#define C_RS1 3
#define C_RS2 4
#define C_RD 5
#define C_READ_REGISTER1 6
#define C_READ_REGISTER2 7
#define C_WRITE_REGISTER 8
#define C_IMM_0 9
#define C_IMM_1 10
#define C_HALF_INSTRUCTION_LENGTH 11
#define C_WORD_INSTR 12
#define C_ALU 13
#define C_ALU_FLAGS 14
#define C_ADD 15
#define C_SUB 16
#define C_MEMORY 17
#define C_MEM_FLAGS 18
#define C_BRANCH 19
#define C_ECALL 20
#define C_NEXT_PC_0 21
#define C_NEXT_PC_1 22
#define C_RVD_0 23
#define C_RVD_1 24
#define C_PREV_PC_TIMESTAMP_BORROW 25
#define C_PC_DOUBLE_READ 26
#define C_RV1_0 27
#define C_RV1_1 28
#define C_RV2_0 29
#define C_RV2_1 30
#define C_ARG2_0 31
#define C_ARG2_1 32
#define C_RES_0 33
#define C_RES_1 34
#define C_RES_2 35
#define C_RES_3 36
#define C_BRANCH_COND 37

// --- PackedDecode AoS layout (DEC_STRIDE u64 per PC; must match the Rust builder
//     in math-cuda/src/trace.rs and prover) ---
#define DEC_STRIDE 8
#define DEC_FLAGS 0
#define DEC_RS1 1
#define DEC_RS2 2
#define DEC_RD 3
#define DEC_HIL 4
#define DEC_ALU_FLAGS 5
#define DEC_MEM_FLAGS 6
#define DEC_IMM 7
// flag bit positions inside d[DEC_FLAGS]
#define F_READ_REGISTER1 0
#define F_READ_REGISTER2 1
#define F_WRITE_REGISTER 2
#define F_WORD_INSTR 3
#define F_ALU 4
#define F_ADD 5
#define F_SUB 6
#define F_MEMORY 7
#define F_BRANCH 8
#define F_ECALL 9

#define ALU_OP_EQ 3
#define ALU_OP_LT 4

__device__ __forceinline__ void setcol(uint64_t *cols, uint64_t nrows, int col,
                                        uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

extern "C" __global__ void
trace_cpu_kernel(const uint64_t *__restrict__ logs,   // 5 * n (this chunk)
                 const uint64_t *__restrict__ decode, // DEC_STRIDE * n_pc
                 uint64_t text_base,
                 uint64_t row_offset, // global index of this chunk's first row
                 uint64_t n, uint64_t nrows,
                 uint64_t *__restrict__ cols) { // 38 * nrows, zero-initialized
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;

  // TIMESTAMP is GLOBAL = 4*(row_offset + r) + 4, the same formula for real and
  // padding rows and across chunks: real global row g has ts = 4g+4; a chunk's
  // padding row g (>= chunk_start+n) has last_ts + 4*(local-n+1) which also
  // reduces to 4g+4. `row_offset` = chunk_start (0 for a single un-chunked CPU
  // table); the op-vector carries global timestamps, so chunks must match.
  uint64_t g = row_offset + r;
  uint64_t ts = 4 * g + 4;
  setcol(cols, nrows, C_TIMESTAMP, r, ts);

  if (r >= n) {
    // Padding row: pc = next_pc = 1 (odd, unreachable); everything else stays 0.
    setcol(cols, nrows, C_PC_0, r, 1);
    setcol(cols, nrows, C_NEXT_PC_0, r, 1);
    return;
  }

  // ---- real row ----
  const uint64_t *lg = logs + 5 * r;
  uint64_t current_pc = lg[0];
  uint64_t log_next_pc = lg[1];
  uint64_t src1_val = lg[2];
  uint64_t src2_val = lg[3];
  uint64_t dst_val = lg[4];

  const uint64_t *d = decode + DEC_STRIDE * ((current_pc - text_base) >> 1);
  uint64_t flags = d[DEC_FLAGS];
  uint64_t rs1 = d[DEC_RS1];
  uint64_t rs2 = d[DEC_RS2];
  uint64_t rd = d[DEC_RD];
  uint64_t hil = d[DEC_HIL];
  uint64_t alu_flags = d[DEC_ALU_FLAGS];
  uint64_t mem_flags = d[DEC_MEM_FLAGS];
  uint64_t imm = d[DEC_IMM];

  int read_register1 = (flags >> F_READ_REGISTER1) & 1;
  int read_register2 = (flags >> F_READ_REGISTER2) & 1;
  int write_register = (flags >> F_WRITE_REGISTER) & 1;
  int word = (flags >> F_WORD_INSTR) & 1;
  int alu = (flags >> F_ALU) & 1;
  int add_ = (flags >> F_ADD) & 1;
  int sub_ = (flags >> F_SUB) & 1;
  int memory = (flags >> F_MEMORY) & 1;
  int branch = (flags >> F_BRANCH) & 1;
  int ecall = (flags >> F_ECALL) & 1;

  uint64_t pc = current_pc;
  uint64_t ilen = 2 * hil;

  uint64_t rv1, rv2, arg2, res, rvd, next_pc;
  int branch_cond = 0;

  if (word) {
    // Word (*W) delegate row: only PC-advancing + register values; CPU32 owns
    // the operational columns (zeroed below via the `word` masks).
    next_pc = pc + ilen;
    rv1 = src1_val;
    rv2 = read_register2 ? src2_val : 0;
    rvd = dst_val;
    arg2 = 0;
    res = 0;
  } else {
    rv1 = (rs1 == 255) ? pc : (read_register1 ? src1_val : 0); // x255 = PC
    rv2 = read_register2 ? src2_val : 0;
    int jalr = mem_flags & 1;
    arg2 = memory ? imm : (branch ? rv2 : (rv2 + imm));
    if (branch) {
      if (jalr) {
        branch_cond = 1;
      } else {
        uint64_t op = alu_flags & 0x1F;
        int is_signed = (alu_flags >> 5) & 1;
        int invert = (alu_flags >> 6) & 1;
        int cmp = 0;
        if (op == ALU_OP_EQ)
          cmp = (rv1 == rv2);
        else if (op == ALU_OP_LT)
          cmp = is_signed ? ((int64_t)rv1 < (int64_t)rv2) : (rv1 < rv2);
        branch_cond = cmp ^ invert;
      }
    }
    if (add_)
      res = rv1 + arg2;
    else if (sub_)
      res = rv1 - arg2;
    else if (alu)
      res = branch ? (uint64_t)branch_cond : dst_val;
    else
      res = 0;
    int store = memory && jalr;
    rvd = memory ? (store ? 0 : dst_val) : (branch ? (pc + ilen) : res);
    next_pc = ecall ? (pc + ilen) : (branch_cond ? log_next_pc : (pc + ilen));
  }

  // ---- column writes (column-major) ----
  setcol(cols, nrows, C_PC_0, r, pc & 0xFFFFFFFF);
  setcol(cols, nrows, C_PC_1, r, pc >> 32);

  setcol(cols, nrows, C_RS1, r, word ? 0 : rs1);
  setcol(cols, nrows, C_RS2, r, word ? 0 : rs2);
  setcol(cols, nrows, C_RD, r, word ? 0 : rd);

  setcol(cols, nrows, C_READ_REGISTER1, r,
         (!word && read_register1 && rs1 != 0) ? 1 : 0);
  setcol(cols, nrows, C_READ_REGISTER2, r,
         (!word && read_register2 && rs2 != 0) ? 1 : 0);
  setcol(cols, nrows, C_WRITE_REGISTER, r,
         (!word && write_register && rd != 0) ? 1 : 0);

  uint64_t imm_eff = word ? 0 : imm;
  setcol(cols, nrows, C_IMM_0, r, imm_eff & 0xFFFFFFFF);
  setcol(cols, nrows, C_IMM_1, r, imm_eff >> 32);

  setcol(cols, nrows, C_HALF_INSTRUCTION_LENGTH, r, hil); // not masked by word
  setcol(cols, nrows, C_WORD_INSTR, r, (uint64_t)word);

  setcol(cols, nrows, C_ALU, r, (!word && alu) ? 1 : 0);
  setcol(cols, nrows, C_ALU_FLAGS, r, word ? 0 : alu_flags);
  setcol(cols, nrows, C_ADD, r, (!word && add_) ? 1 : 0);
  setcol(cols, nrows, C_SUB, r, (!word && sub_) ? 1 : 0);
  setcol(cols, nrows, C_MEMORY, r, (!word && memory) ? 1 : 0);
  setcol(cols, nrows, C_MEM_FLAGS, r, word ? 0 : mem_flags);
  setcol(cols, nrows, C_BRANCH, r, (!word && branch) ? 1 : 0);
  setcol(cols, nrows, C_ECALL, r, (!word && ecall) ? 1 : 0);

  setcol(cols, nrows, C_NEXT_PC_0, r, next_pc & 0xFFFFFFFF);
  setcol(cols, nrows, C_NEXT_PC_1, r, next_pc >> 32);

  uint64_t rvd_eff = word ? 0 : rvd;
  setcol(cols, nrows, C_RVD_0, r, rvd_eff & 0xFFFFFFFF);
  setcol(cols, nrows, C_RVD_1, r, rvd_eff >> 32);

  int pc_double_read = (!word && read_register1 && rs1 == 255) ? 1 : 0;
  uint64_t ts_lo = ts & 0xFFFFFFFF;
  int prev_pc_ts_borrow = (!pc_double_read && ts_lo < 3) ? 1 : 0;
  setcol(cols, nrows, C_PC_DOUBLE_READ, r, (uint64_t)pc_double_read);
  setcol(cols, nrows, C_PREV_PC_TIMESTAMP_BORROW, r, (uint64_t)prev_pc_ts_borrow);

  uint64_t rv1_eff = word ? 0 : rv1;
  uint64_t rv2_eff = word ? 0 : rv2;
  uint64_t arg2_eff = word ? 0 : arg2;
  setcol(cols, nrows, C_RV1_0, r, rv1_eff & 0xFFFFFFFF);
  setcol(cols, nrows, C_RV1_1, r, rv1_eff >> 32);
  setcol(cols, nrows, C_RV2_0, r, rv2_eff & 0xFFFFFFFF);
  setcol(cols, nrows, C_RV2_1, r, rv2_eff >> 32);
  setcol(cols, nrows, C_ARG2_0, r, arg2_eff & 0xFFFFFFFF);
  setcol(cols, nrows, C_ARG2_1, r, arg2_eff >> 32);

  uint64_t res_eff = word ? 0 : res;
  setcol(cols, nrows, C_RES_0, r, (res_eff >> 0) & 0xFFFF);
  setcol(cols, nrows, C_RES_1, r, (res_eff >> 16) & 0xFFFF);
  setcol(cols, nrows, C_RES_2, r, (res_eff >> 32) & 0xFFFF);
  setcol(cols, nrows, C_RES_3, r, (res_eff >> 48) & 0xFFFF);

  setcol(cols, nrows, C_BRANCH_COND, r, (uint64_t)branch_cond);
}
