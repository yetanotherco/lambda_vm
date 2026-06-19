// BITWISE table main-column generation. Preprocessed table — each row is a
// pure function of its index. No execution-dependent input.
//
// Row layout (matches `prover/src/tables/bitwise.rs::cols`):
//   col 0  X       = row_idx & 0xFF
//   col 1  Y       = (row_idx >> 8) & 0xFF
//   col 2  Z       = (row_idx >> 16) & 0xF
//   col 3  AND     = X & Y
//   col 4  OR      = X | Y
//   col 5  XOR     = X ^ Y
//   col 6  MSB8    = (X >> 7) & 1
//   col 7  MSB16   = ((X + Y*256) >> 15) & 1
//   col 8  ZERO    = (X==0 && Y==0 && Z==0) ? 1 : 0
//   col 9  SLL     = (Z == 0) ? halfword : ((halfword << Z) & 0xFFFF)
//   col 10 SLLC    = (Z == 0) ? 0 : (halfword >> (16 - Z))
//   col 11..20    multiplicities — left at 0 (filled by update_multiplicities on host)
//
// num_rows is always 2^20 (256 * 256 * 16). One thread per row.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_bitwise_trace_rows(
    uint64_t num_rows,
    uint64_t *table_data,    // length num_rows * num_cols, row-major
    uint64_t num_cols        // expected = 21
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;

    uint64_t x = row & 0xFFULL;
    uint64_t y = (row >> 8) & 0xFFULL;
    uint64_t z = (row >> 16) & 0xFULL;

    uint64_t halfword = x + y * 256ULL;
    uint64_t msb8 = (x >> 7) & 1ULL;
    uint64_t msb16 = (halfword >> 15) & 1ULL;
    uint64_t is_zero = (x == 0 && y == 0 && z == 0) ? 1ULL : 0ULL;
    uint64_t sll = (z == 0) ? halfword : ((halfword << z) & 0xFFFFULL);
    uint64_t sllc = (z == 0) ? 0ULL : (halfword >> (16 - z));

    uint64_t base = row * num_cols;
    table_data[base + 0] = x;
    table_data[base + 1] = y;
    table_data[base + 2] = z;
    table_data[base + 3] = x & y;
    table_data[base + 4] = x | y;
    table_data[base + 5] = x ^ y;
    table_data[base + 6] = msb8;
    table_data[base + 7] = msb16;
    table_data[base + 8] = is_zero;
    table_data[base + 9] = sll;
    table_data[base + 10] = sllc;
    // Columns 11..20 are multiplicities; left at 0 (kernel writes them
    // explicitly so we don't rely on the caller pre-zeroing the buffer).
    for (uint64_t c = 11; c < num_cols; ++c) {
        table_data[base + c] = 0ULL;
    }
}
