// GPU CPU32-table fill (delegated *W word instructions). One thread per op (no
// dedup, mu=1). Mirrors generate_cpu32_trace + Cpu32Operation::compute_aux in
// prover/src/tables/cpu32.rs. Padding rows (r >= n) stay zero.
//
// Per-op input is interleaved, stride C32_STRIDE:
//   [timestamp, pc, rs1, read_register1, rv1, rs2, read_register2, rv2, imm,
//    res, rd, write_register, alu, alu_flags, add, sub, half_instruction_length]

#include <cstdint>

#define T_TIMESTAMP_0 0
#define T_PC_0 2
#define T_RS1 4
#define T_READ_REGISTER1 5
#define T_RV1_0 6 // DWordWHH (half, half, word)
#define T_RV1_SIGN 9
#define T_ARG1_0 10
#define T_RS2 12
#define T_READ_REGISTER2 13
#define T_RV2_0 14 // DWordWHH
#define T_RV2_SIGN 17
#define T_IMM_0 18
#define T_ARG2_0 20
#define T_RES_0 22 // DWordHL (4 halves)
#define T_RES_SIGN 26
#define T_RD 27
#define T_WRITE_REGISTER 28
#define T_RVD_0 29
#define T_ALU 31
#define T_ALU_FLAGS 32
#define T_ADD 33
#define T_SUB 34
#define T_HALF_INSTRUCTION_LENGTH 35
#define T_SIGNED 36
#define T_MU 37

#define C32_STRIDE 17
#define HI_FILL 0xFFFFFFFFULL
#define ALU_FLAGS_SIGNED 5

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}
// DWordWL: low/high 32-bit words.
__device__ __forceinline__ void dwwl(uint64_t *cols, uint64_t nrows, int col,
                                      uint64_t r, uint64_t v) {
  sc(cols, nrows, col + 0, r, v & 0xFFFFFFFF);
  sc(cols, nrows, col + 1, r, v >> 32);
}
// DWordWHH: [low half, mid half, high word].
__device__ __forceinline__ void dwwhh(uint64_t *cols, uint64_t nrows, int col,
                                       uint64_t r, uint64_t v) {
  sc(cols, nrows, col + 0, r, v & 0xFFFF);
  sc(cols, nrows, col + 1, r, (v >> 16) & 0xFFFF);
  sc(cols, nrows, col + 2, r, (v >> 32) & 0xFFFFFFFF);
}
// DWordHL: 4 little-endian 16-bit halves.
__device__ __forceinline__ void dwhl(uint64_t *cols, uint64_t nrows, int col,
                                      uint64_t r, uint64_t v) {
  sc(cols, nrows, col + 0, r, v & 0xFFFF);
  sc(cols, nrows, col + 1, r, (v >> 16) & 0xFFFF);
  sc(cols, nrows, col + 2, r, (v >> 32) & 0xFFFF);
  sc(cols, nrows, col + 3, r, (v >> 48) & 0xFFFF);
}

extern "C" __global__ void
trace_cpu32_kernel(const uint64_t *__restrict__ in, uint64_t n, uint64_t nrows,
                   uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return; // padding rows already zeroed

  const uint64_t *op = in + r * C32_STRIDE;
  uint64_t timestamp = op[0];
  uint64_t pc = op[1];
  uint64_t rs1 = op[2];
  uint64_t read_register1 = op[3];
  uint64_t rv1 = op[4];
  uint64_t rs2 = op[5];
  uint64_t read_register2 = op[6];
  uint64_t rv2 = op[7];
  uint64_t imm = op[8];
  uint64_t res = op[9];
  uint64_t rd = op[10];
  uint64_t write_register = op[11];
  uint64_t alu = op[12];
  uint64_t alu_flags = op[13];
  uint64_t add = op[14];
  uint64_t sub = op[15];
  uint64_t half_len = op[16];

  // compute_aux
  uint64_t signed_flag = (alu_flags >> ALU_FLAGS_SIGNED) & 1;
  uint64_t rv1_sign = (signed_flag && ((rv1 >> 31) & 1)) ? 1 : 0;
  uint64_t rv2_sign = (signed_flag && ((rv2 >> 31) & 1)) ? 1 : 0;
  uint64_t res_sign = (res >> 31) & 1;
  uint64_t arg1 = (rv1 & 0xFFFFFFFF) | ((rv1_sign ? HI_FILL : 0ULL) << 32);
  uint64_t arg2_lo = (rv2 & 0xFFFFFFFF) + (imm & 0xFFFFFFFF);
  uint64_t arg2_hi = (rv2_sign ? HI_FILL : 0ULL) + (imm >> 32);
  uint64_t arg2 = (arg2_lo & 0xFFFFFFFF) | (arg2_hi << 32);
  uint64_t rvd = (res & 0xFFFFFFFF) | ((res_sign ? HI_FILL : 0ULL) << 32);

  dwwl(cols, nrows, T_TIMESTAMP_0, r, timestamp);
  dwwl(cols, nrows, T_PC_0, r, pc);
  sc(cols, nrows, T_RS1, r, rs1);
  sc(cols, nrows, T_READ_REGISTER1, r, read_register1);
  dwwhh(cols, nrows, T_RV1_0, r, rv1);
  sc(cols, nrows, T_RV1_SIGN, r, rv1_sign);
  dwwl(cols, nrows, T_ARG1_0, r, arg1);
  sc(cols, nrows, T_RS2, r, rs2);
  sc(cols, nrows, T_READ_REGISTER2, r, read_register2);
  dwwhh(cols, nrows, T_RV2_0, r, rv2);
  sc(cols, nrows, T_RV2_SIGN, r, rv2_sign);
  dwwl(cols, nrows, T_IMM_0, r, imm);
  dwwl(cols, nrows, T_ARG2_0, r, arg2);
  dwhl(cols, nrows, T_RES_0, r, res);
  sc(cols, nrows, T_RES_SIGN, r, res_sign);
  sc(cols, nrows, T_RD, r, rd);
  sc(cols, nrows, T_WRITE_REGISTER, r, write_register);
  dwwl(cols, nrows, T_RVD_0, r, rvd);
  sc(cols, nrows, T_ALU, r, alu);
  sc(cols, nrows, T_ALU_FLAGS, r, alu_flags);
  sc(cols, nrows, T_ADD, r, add);
  sc(cols, nrows, T_SUB, r, sub);
  sc(cols, nrows, T_HALF_INSTRUCTION_LENGTH, r, half_len);
  sc(cols, nrows, T_SIGNED, r, signed_flag);
  sc(cols, nrows, T_MU, r, 1);
}
