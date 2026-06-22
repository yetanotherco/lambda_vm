// CPU table main-column generation. One thread per row, 38 columns.
//
// Per-row inputs (caller pre-masks word/padding overrides so the kernel
// does pure row layout with no branching):
//   timestamps[row]   = u64
//   pcs[row]          = u64 (split into [lo32, hi32])
//   next_pcs[row]     = u64 (split into [lo32, hi32])
//   imms[row]         = u64 (0 on word/padding rows)
//   rvds[row]         = u64 (0 on word/padding rows)
//   rv1s[row]         = u64 (0 on word/padding rows)
//   rv2s[row]         = u64 (0 on word/padding rows)
//   arg2s[row]        = u64 (0 on word/padding rows)
//   ress[row]         = u64 (0 on word/padding rows; split into 4 halfwords)
//   flags[row] bit-packed (all pre-masked for word/padding):
//     bits  0.. 7  rs1
//     bits  8..15  rs2
//     bits 16..23  rd
//     bits 24..31  half_instruction_length
//     bits 32..39  alu_flags
//     bits 40..47  mem_flags
//     bit 48       word_instr
//     bit 49       alu
//     bit 50       add
//     bit 51       sub
//     bit 52       memory
//     bit 53       branch
//     bit 54       ecall
//     bit 55       read_register1
//     bit 56       read_register2
//     bit 57       write_register
//     bit 58       branch_cond
//     bit 59       pc_double_read
//     bit 60       prev_pc_timestamp_borrow
//     bit 61       active (informational; not written as a column)
//
// Column layout (matches `prover/src/tables/cpu.rs::cols`):
//   0   TIMESTAMP
//   1   PC_0       (lo32)
//   2   PC_1       (hi32)
//   3   RS1
//   4   RS2
//   5   RD
//   6   READ_REGISTER1
//   7   READ_REGISTER2
//   8   WRITE_REGISTER
//   9   IMM_0
//   10  IMM_1
//   11  HALF_INSTRUCTION_LENGTH
//   12  WORD_INSTR
//   13  ALU
//   14  ALU_FLAGS
//   15  ADD
//   16  SUB
//   17  MEMORY
//   18  MEM_FLAGS
//   19  BRANCH
//   20  ECALL
//   21  NEXT_PC_0
//   22  NEXT_PC_1
//   23  RVD_0
//   24  RVD_1
//   25  PREV_PC_TIMESTAMP_BORROW
//   26  PC_DOUBLE_READ
//   27  RV1_0
//   28  RV1_1
//   29  RV2_0
//   30  RV2_1
//   31  ARG2_0
//   32  ARG2_1
//   33  RES_0   (bits  0-15)
//   34  RES_1   (bits 16-31)
//   35  RES_2   (bits 32-47)
//   36  RES_3   (bits 48-63)
//   37  BRANCH_COND

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_cpu_trace_rows(
    uint64_t num_rows,
    const uint64_t *timestamps,
    const uint64_t *pcs,
    const uint64_t *next_pcs,
    const uint64_t *imms,
    const uint64_t *rvds,
    const uint64_t *rv1s,
    const uint64_t *rv2s,
    const uint64_t *arg2s,
    const uint64_t *ress,
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols     // expected = 38
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;

    uint64_t ts = timestamps[row];
    uint64_t pc = pcs[row];
    uint64_t npc = next_pcs[row];
    uint64_t imm = imms[row];
    uint64_t rvd = rvds[row];
    uint64_t rv1 = rv1s[row];
    uint64_t rv2 = rv2s[row];
    uint64_t arg2 = arg2s[row];
    uint64_t res = ress[row];
    uint64_t f = flags[row];

    table_data[base +  0] = ts;
    table_data[base +  1] = pc & 0xFFFFFFFFULL;
    table_data[base +  2] = pc >> 32;
    table_data[base +  3] = (f >>  0) & 0xFFULL;
    table_data[base +  4] = (f >>  8) & 0xFFULL;
    table_data[base +  5] = (f >> 16) & 0xFFULL;
    table_data[base +  6] = (f >> 55) & 1ULL;
    table_data[base +  7] = (f >> 56) & 1ULL;
    table_data[base +  8] = (f >> 57) & 1ULL;
    table_data[base +  9] = imm & 0xFFFFFFFFULL;
    table_data[base + 10] = imm >> 32;
    table_data[base + 11] = (f >> 24) & 0xFFULL;
    table_data[base + 12] = (f >> 48) & 1ULL;
    table_data[base + 13] = (f >> 49) & 1ULL;
    table_data[base + 14] = (f >> 32) & 0xFFULL;
    table_data[base + 15] = (f >> 50) & 1ULL;
    table_data[base + 16] = (f >> 51) & 1ULL;
    table_data[base + 17] = (f >> 52) & 1ULL;
    table_data[base + 18] = (f >> 40) & 0xFFULL;
    table_data[base + 19] = (f >> 53) & 1ULL;
    table_data[base + 20] = (f >> 54) & 1ULL;
    table_data[base + 21] = npc & 0xFFFFFFFFULL;
    table_data[base + 22] = npc >> 32;
    table_data[base + 23] = rvd & 0xFFFFFFFFULL;
    table_data[base + 24] = rvd >> 32;
    table_data[base + 25] = (f >> 60) & 1ULL;
    table_data[base + 26] = (f >> 59) & 1ULL;
    table_data[base + 27] = rv1 & 0xFFFFFFFFULL;
    table_data[base + 28] = rv1 >> 32;
    table_data[base + 29] = rv2 & 0xFFFFFFFFULL;
    table_data[base + 30] = rv2 >> 32;
    table_data[base + 31] = arg2 & 0xFFFFFFFFULL;
    table_data[base + 32] = arg2 >> 32;
    table_data[base + 33] = (res >>  0) & 0xFFFFULL;
    table_data[base + 34] = (res >> 16) & 0xFFFFULL;
    table_data[base + 35] = (res >> 32) & 0xFFFFULL;
    table_data[base + 36] = (res >> 48) & 0xFFFFULL;
    table_data[base + 37] = (f >> 58) & 1ULL;
}
