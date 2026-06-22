// CPU32 (delegated `*W`) table main-column generation. One thread per
// row, 38 columns. Kernel computes signs / arg1 / arg2 / rvd on-device.
//
// Per-row inputs:
//   timestamps[row]  = u64
//   pcs[row]         = u64
//   rv1s[row]        = u64
//   rv2s[row]        = u64
//   imms[row]        = u64 (already sign-extended)
//   ress[row]        = u64 (raw 64-bit ALU result)
//   flags[row] bits:
//     bits  0.. 7   rs1
//     bits  8..15   rs2
//     bits 16..23   rd
//     bits 24..31   half_instruction_length
//     bits 32..39   alu_flags
//     bit 40        read_register1
//     bit 41        read_register2
//     bit 42        write_register
//     bit 43        alu
//     bit 44        add
//     bit 45        sub
//     bit 46        active (0 = padding row → all zeros)
//
// (`signed` is bit 5 of alu_flags, decoded inside the kernel.)
//
// Column layout (matches `prover/src/tables/cpu32.rs::cols`):
//   0  TIMESTAMP_0           19 IMM_1
//   1  TIMESTAMP_1           20 ARG2_0
//   2  PC_0                  21 ARG2_1
//   3  PC_1                  22 RES_0
//   4  RS1                   23 RES_1
//   5  READ_REGISTER1        24 RES_2
//   6  RV1_0                 25 RES_3
//   7  RV1_1                 26 RES_SIGN
//   8  RV1_2                 27 RD
//   9  RV1_SIGN              28 WRITE_REGISTER
//   10 ARG1_0                29 RVD_0
//   11 ARG1_1                30 RVD_1
//   12 RS2                   31 ALU
//   13 READ_REGISTER2        32 ALU_FLAGS
//   14 RV2_0                 33 ADD
//   15 RV2_1                 34 SUB
//   16 RV2_2                 35 HALF_INSTRUCTION_LENGTH
//   17 RV2_SIGN              36 SIGNED
//   18 IMM_0                 37 MU

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define HI_FILL 0xFFFFFFFFULL
// alu_flags bit 5 = signed (packed_decode_shrunk::ALU_FLAGS_SIGNED)
#define SIGNED_BIT 5

extern "C" __global__ void generate_cpu32_trace_rows(
    uint64_t num_rows,
    const uint64_t *timestamps,
    const uint64_t *pcs,
    const uint64_t *rv1s,
    const uint64_t *rv2s,
    const uint64_t *imms,
    const uint64_t *ress,
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols       // expected = 38
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t f = flags[row];
    uint64_t active = (f >> 46) & 1ULL;

    if (!active) {
        for (uint64_t c = 0; c < num_cols; ++c) {
            table_data[base + c] = 0;
        }
        return;
    }

    uint64_t ts = timestamps[row];
    uint64_t pc = pcs[row];
    uint64_t rv1 = rv1s[row];
    uint64_t rv2 = rv2s[row];
    uint64_t imm = imms[row];
    uint64_t res = ress[row];

    uint64_t alu_flags     = (f >> 32) & 0xFFULL;
    uint64_t is_signed     = (alu_flags >> SIGNED_BIT) & 1ULL;

    // Sign bits: rv1/rv2 are gated by `signed`; res is always sign-extended for *W.
    uint64_t rv1_sign = (is_signed && ((rv1 >> 31) & 1ULL)) ? 1ULL : 0ULL;
    uint64_t rv2_sign = (is_signed && ((rv2 >> 31) & 1ULL)) ? 1ULL : 0ULL;
    uint64_t res_sign = (res >> 31) & 1ULL;

    // arg1 = ext(rv1 low word)
    uint64_t arg1_hi = rv1_sign ? HI_FILL : 0ULL;
    uint64_t arg1 = (rv1 & 0xFFFFFFFFULL) | (arg1_hi << 32);

    // arg2 = ext(rv2 low word) + imm. By assumption only one of rv2/imm is
    // non-zero so per-word sums stay below 2^33.
    uint64_t arg2_lo = (rv2 & 0xFFFFFFFFULL) + (imm & 0xFFFFFFFFULL);
    uint64_t rv2_ext_hi = rv2_sign ? HI_FILL : 0ULL;
    uint64_t arg2_hi = rv2_ext_hi + (imm >> 32);
    uint64_t arg2 = (arg2_lo & 0xFFFFFFFFULL) | ((arg2_hi & 0xFFFFFFFFULL) << 32);

    // rvd = sign-extend(res low word)
    uint64_t rvd_hi = res_sign ? HI_FILL : 0ULL;
    uint64_t rvd = (res & 0xFFFFFFFFULL) | (rvd_hi << 32);

    table_data[base +  0] = ts & 0xFFFFFFFFULL;          // TIMESTAMP_0
    table_data[base +  1] = ts >> 32;                    // TIMESTAMP_1
    table_data[base +  2] = pc & 0xFFFFFFFFULL;          // PC_0
    table_data[base +  3] = pc >> 32;                    // PC_1
    table_data[base +  4] = (f >>  0) & 0xFFULL;         // RS1
    table_data[base +  5] = (f >> 40) & 1ULL;            // READ_REGISTER1
    table_data[base +  6] = rv1 & 0xFFFFULL;             // RV1_0 (Half)
    table_data[base +  7] = (rv1 >> 16) & 0xFFFFULL;     // RV1_1 (Half)
    table_data[base +  8] = rv1 >> 32;                   // RV1_2 (Word)
    table_data[base +  9] = rv1_sign;                    // RV1_SIGN
    table_data[base + 10] = arg1 & 0xFFFFFFFFULL;        // ARG1_0
    table_data[base + 11] = arg1 >> 32;                  // ARG1_1
    table_data[base + 12] = (f >>  8) & 0xFFULL;         // RS2
    table_data[base + 13] = (f >> 41) & 1ULL;            // READ_REGISTER2
    table_data[base + 14] = rv2 & 0xFFFFULL;             // RV2_0
    table_data[base + 15] = (rv2 >> 16) & 0xFFFFULL;     // RV2_1
    table_data[base + 16] = rv2 >> 32;                   // RV2_2
    table_data[base + 17] = rv2_sign;                    // RV2_SIGN
    table_data[base + 18] = imm & 0xFFFFFFFFULL;         // IMM_0
    table_data[base + 19] = imm >> 32;                   // IMM_1
    table_data[base + 20] = arg2 & 0xFFFFFFFFULL;        // ARG2_0
    table_data[base + 21] = arg2 >> 32;                  // ARG2_1
    table_data[base + 22] = res & 0xFFFFULL;             // RES_0
    table_data[base + 23] = (res >> 16) & 0xFFFFULL;     // RES_1
    table_data[base + 24] = (res >> 32) & 0xFFFFULL;     // RES_2
    table_data[base + 25] = (res >> 48) & 0xFFFFULL;     // RES_3
    table_data[base + 26] = res_sign;                    // RES_SIGN
    table_data[base + 27] = (f >> 16) & 0xFFULL;         // RD
    table_data[base + 28] = (f >> 42) & 1ULL;            // WRITE_REGISTER
    table_data[base + 29] = rvd & 0xFFFFFFFFULL;         // RVD_0
    table_data[base + 30] = rvd >> 32;                   // RVD_1
    table_data[base + 31] = (f >> 43) & 1ULL;            // ALU
    table_data[base + 32] = alu_flags;                   // ALU_FLAGS
    table_data[base + 33] = (f >> 44) & 1ULL;            // ADD
    table_data[base + 34] = (f >> 45) & 1ULL;            // SUB
    table_data[base + 35] = (f >> 24) & 0xFFULL;         // HALF_INSTRUCTION_LENGTH
    table_data[base + 36] = is_signed;                   // SIGNED
    table_data[base + 37] = 1ULL;                        // MU = 1
}
