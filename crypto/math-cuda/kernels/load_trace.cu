// LOAD table main-column generation. One thread per row, 18 columns.
//
// Per-row columns (matches `prover/src/tables/load.rs::cols`):
//   col 0  BASE_ADDRESS_0 = base_addresses[row] & 0xFFFFFFFF
//   col 1  BASE_ADDRESS_1 = base_addresses[row] >> 32
//   col 2  TIMESTAMP_0    = timestamps[row] & 0xFFFFFFFF
//   col 3  TIMESTAMP_1    = timestamps[row] >> 32
//   col 4  READ2          = flags[row] bit 0
//   col 5  READ4          = flags[row] bit 1
//   col 6  READ8          = flags[row] bit 2
//   col 7  SIGNED         = flags[row] bit 3
//   col 8..15  RES[0..7]  = res_bytes[row * 8 + i]
//   col 16 SIGN_BIT       = flags[row] bit 4
//   col 17 MU             = flags[row] bit 5 (1 for active rows, 0 for padding)
//
// Caller pads all four input arrays to `num_rows` (next power of two of
// the operations count, min 4); padding rows have all fields = 0.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_load_trace_rows(
    uint64_t num_rows,
    const uint64_t *base_addresses,   // length num_rows
    const uint64_t *timestamps,       // length num_rows
    const uint64_t *flags,            // length num_rows, bit-packed
    const uint64_t *res_bytes,        // length 8 * num_rows (interleaved per row)
    uint64_t *table_data,             // length num_rows * num_cols, row-major
    uint64_t num_cols                 // expected = 18
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t addr = base_addresses[row];
    uint64_t ts = timestamps[row];
    uint64_t f = flags[row];

    table_data[base + 0]  = addr & 0xFFFFFFFFULL;
    table_data[base + 1]  = addr >> 32;
    table_data[base + 2]  = ts & 0xFFFFFFFFULL;
    table_data[base + 3]  = ts >> 32;
    table_data[base + 4]  = (f >> 0) & 1ULL;  // READ2
    table_data[base + 5]  = (f >> 1) & 1ULL;  // READ4
    table_data[base + 6]  = (f >> 2) & 1ULL;  // READ8
    table_data[base + 7]  = (f >> 3) & 1ULL;  // SIGNED
    const uint64_t *r = res_bytes + row * 8;
    table_data[base + 8]  = r[0];
    table_data[base + 9]  = r[1];
    table_data[base + 10] = r[2];
    table_data[base + 11] = r[3];
    table_data[base + 12] = r[4];
    table_data[base + 13] = r[5];
    table_data[base + 14] = r[6];
    table_data[base + 15] = r[7];
    table_data[base + 16] = (f >> 4) & 1ULL;  // SIGN_BIT
    table_data[base + 17] = (f >> 5) & 1ULL;  // MU
}
