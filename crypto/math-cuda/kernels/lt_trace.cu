// LT (Less-Than) main-column generation. One thread per row, 17 columns.
//
// Per-row inputs:
//   lhs_values[row]      = u64 left operand
//   rhs_values[row]      = u64 right operand
//   flags[row] bits:
//     bit 0: signed   (1 = signed comparison; 0 = unsigned)
//     bit 1: invert   (BGE/BGEU dispatch: out = lt XOR invert)
//     bit 2: active   (0 = padding row → all zeros)
//   multiplicities[row]  = u64 (counts in the receiver — μ column)
//
// Column layout (matches `prover/src/tables/lt.rs::cols`):
//   0..2   LHS_0..LHS_2     (DWordHHW: Word, Half, Half)
//   3..5   RHS_0..RHS_2     (DWordHHW)
//   6      SIGNED
//   7      LT               (raw less-than)
//   8..11  LHS_SUB_RHS_0..3 (DWordHL: 4 halfwords of lhs - rhs)
//   12     LHS_MSB          (bit 63 of lhs)
//   13     RHS_MSB          (bit 63 of rhs)
//   14     INVERT
//   15     OUT              (lt XOR invert)
//   16     MU

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_lt_trace_rows(
    uint64_t num_rows,
    const uint64_t *lhs_values,
    const uint64_t *rhs_values,
    const uint64_t *flags,
    const uint64_t *multiplicities,
    uint64_t *table_data,
    uint64_t num_cols           // expected = 17
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t f = flags[row];
    uint64_t active = (f >> 2) & 1ULL;

    if (!active) {
        for (uint64_t c = 0; c < num_cols; ++c) {
            table_data[base + c] = 0;
        }
        return;
    }

    uint64_t lhs = lhs_values[row];
    uint64_t rhs = rhs_values[row];
    uint64_t is_signed = (f >> 0) & 1ULL;
    uint64_t invert    = (f >> 1) & 1ULL;

    // LT result.
    uint64_t lt_unsigned = (lhs < rhs) ? 1ULL : 0ULL;
    uint64_t lt_signed   = ((int64_t)lhs < (int64_t)rhs) ? 1ULL : 0ULL;
    uint64_t lt = is_signed ? lt_signed : lt_unsigned;
    uint64_t out = lt ^ invert;

    // lhs - rhs (wrapping)
    uint64_t sub = lhs - rhs;

    table_data[base +  0] = lhs & 0xFFFFFFFFULL;          // LHS_0 (Word)
    table_data[base +  1] = (lhs >> 32) & 0xFFFFULL;      // LHS_1 (Half)
    table_data[base +  2] = (lhs >> 48) & 0xFFFFULL;      // LHS_2 (Half, contains MSB)
    table_data[base +  3] = rhs & 0xFFFFFFFFULL;          // RHS_0
    table_data[base +  4] = (rhs >> 32) & 0xFFFFULL;      // RHS_1
    table_data[base +  5] = (rhs >> 48) & 0xFFFFULL;      // RHS_2
    table_data[base +  6] = is_signed;                    // SIGNED
    table_data[base +  7] = lt;                           // LT
    table_data[base +  8] = sub & 0xFFFFULL;              // LHS_SUB_RHS_0
    table_data[base +  9] = (sub >> 16) & 0xFFFFULL;      // LHS_SUB_RHS_1
    table_data[base + 10] = (sub >> 32) & 0xFFFFULL;      // LHS_SUB_RHS_2
    table_data[base + 11] = (sub >> 48) & 0xFFFFULL;      // LHS_SUB_RHS_3
    table_data[base + 12] = (lhs >> 63) & 1ULL;           // LHS_MSB
    table_data[base + 13] = (rhs >> 63) & 1ULL;           // RHS_MSB
    table_data[base + 14] = invert;                       // INVERT
    table_data[base + 15] = out;                          // OUT
    table_data[base + 16] = multiplicities[row];          // MU
}
