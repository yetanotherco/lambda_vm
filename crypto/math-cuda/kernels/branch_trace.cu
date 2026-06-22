// BRANCH table main-column generation. One thread per row, 14 columns.
//
// Per-row inputs:
//   pcs[row]            = u64 program counter
//   offsets[row]        = u64 sign-extended offset (DWordWL)
//   registers[row]      = u64 register value (for JALR base)
//   flags[row] bits:
//     bit 0: jalr     (1 = register base, 0 = pc base)
//     bit 1: active   (0 = padding row → all zeros)
//   multiplicities[row] = u64 (μ column)
//
// Column layout (matches `prover/src/tables/branch.rs::cols`):
//   0  PC_0
//   1  PC_1
//   2  OFFSET_0
//   3  OFFSET_1
//   4  REGISTER_0
//   5  REGISTER_1
//   6  JALR
//   7  NEXT_PC_HIGH_0   (bits 16..31)
//   8  NEXT_PC_HIGH_1   (bits 32..47)
//   9  NEXT_PC_HIGH_2   (bits 48..63)
//   10 NEXT_PC_LOW_0    (bits 0..7, LSB-masked)
//   11 NEXT_PC_LOW_1    (bits 8..15)
//   12 UNMASKED_LOW_BYTE
//   13 MU

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_branch_trace_rows(
    uint64_t num_rows,
    const uint64_t *pcs,
    const uint64_t *offsets,
    const uint64_t *registers,
    const uint64_t *flags,
    const uint64_t *multiplicities,
    uint64_t *table_data,
    uint64_t num_cols          // expected = 14
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t f = flags[row];
    uint64_t active = (f >> 1) & 1ULL;

    if (!active) {
        for (uint64_t c = 0; c < num_cols; ++c) {
            table_data[base + c] = 0;
        }
        return;
    }

    uint64_t pc = pcs[row];
    uint64_t off = offsets[row];
    uint64_t reg = registers[row];
    uint64_t jalr = (f >> 0) & 1ULL;

    uint64_t base_v = jalr ? reg : pc;
    uint64_t unmasked = base_v + off;       // wrapping
    uint64_t next_pc = unmasked & ~1ULL;    // RISC-V LSB clear

    table_data[base +  0] = pc & 0xFFFFFFFFULL;
    table_data[base +  1] = pc >> 32;
    table_data[base +  2] = off & 0xFFFFFFFFULL;
    table_data[base +  3] = off >> 32;
    table_data[base +  4] = reg & 0xFFFFFFFFULL;
    table_data[base +  5] = reg >> 32;
    table_data[base +  6] = jalr;
    table_data[base +  7] = (next_pc >> 16) & 0xFFFFULL;
    table_data[base +  8] = (next_pc >> 32) & 0xFFFFULL;
    table_data[base +  9] = (next_pc >> 48) & 0xFFFFULL;
    table_data[base + 10] = next_pc & 0xFFULL;
    table_data[base + 11] = (next_pc >> 8) & 0xFFULL;
    table_data[base + 12] = unmasked & 0xFFULL;
    table_data[base + 13] = multiplicities[row];
}
