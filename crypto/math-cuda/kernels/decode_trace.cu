// DECODE-table main-column generation: one thread per row, writes the 6
// columns in row-major layout.
//
// Per-row columns (matches `prover/src/tables/decode.rs`):
//   col 0  PC_0           = pcs[row] & 0xFFFFFFFF
//   col 1  PC_1           = pcs[row] >> 32
//   col 2  PACKED_DECODE  = packed_decodes[row]
//   col 3  IMM_0          = imms[row] & 0xFFFFFFFF
//   col 4  IMM_1          = imms[row] >> 32
//   col 5  MU             = 0  (multiplicities filled by a separate pass on host)
//
// Caller is responsible for pre-filling the three input arrays to length
// `num_rows`, including the CPU-padding row and any trailing padding rows.
// All padding-row pcs/imms/packed_decodes are baked in by the caller.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_decode_trace_rows(
    uint64_t num_rows,
    const uint64_t *pcs,                // length num_rows
    const uint64_t *packed_decodes,     // length num_rows
    const uint64_t *imms,               // length num_rows
    uint64_t *table_data,               // length num_rows * num_cols, row-major
    uint64_t num_cols                   // expected = 6
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t pc = pcs[row];
    uint64_t imm = imms[row];
    table_data[base + 0] = pc & 0xFFFFFFFFULL;
    table_data[base + 1] = pc >> 32;
    table_data[base + 2] = packed_decodes[row];
    table_data[base + 3] = imm & 0xFFFFFFFFULL;
    table_data[base + 4] = imm >> 32;
    table_data[base + 5] = 0ULL;
}
