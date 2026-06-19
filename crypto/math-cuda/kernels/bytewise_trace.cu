// BYTEWISE table main-column generation. One thread per row, 26 columns.
//
// Per-row columns (matches `prover/src/tables/bytewise.rs::cols`):
//   col 0..7    A[0..7]   = byte decomp of a_values[row]
//   col 8..15   B[0..7]   = byte decomp of b_values[row]
//   col 16      OP        = ops[row]
//   col 17..24  RES[0..7] = byte decomp of res_values[row]
//   col 25      MU        = multiplicities[row]
//
// Dedup (HashMap merge with summed multiplicities) is done on the CPU
// side; this kernel only handles the byte breakdown and row layout.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_bytewise_trace_rows(
    uint64_t num_rows,
    const uint64_t *a_values,
    const uint64_t *b_values,
    const uint64_t *res_values,
    const uint64_t *ops,
    const uint64_t *multiplicities,
    uint64_t *table_data,
    uint64_t num_cols    // expected = 26
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t a = a_values[row];
    uint64_t b = b_values[row];
    uint64_t r = res_values[row];

    for (int i = 0; i < 8; ++i) {
        table_data[base + i]      = (a >> (8 * i)) & 0xFFULL;
        table_data[base + 8 + i]  = (b >> (8 * i)) & 0xFFULL;
        table_data[base + 17 + i] = (r >> (8 * i)) & 0xFFULL;
    }
    table_data[base + 16] = ops[row];
    table_data[base + 25] = multiplicities[row];
}
