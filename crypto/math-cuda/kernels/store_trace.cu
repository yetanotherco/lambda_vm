// STORE table main-column generation. One thread per row, 16 columns.
//
// Per-row columns (matches `prover/src/tables/store.rs::cols`):
//   col 0  BASE_ADDRESS_0 = base_addresses[row] & 0xFFFFFFFF
//   col 1  BASE_ADDRESS_1 = base_addresses[row] >> 32
//   col 2  TIMESTAMP_0    = timestamps[row] & 0xFFFFFFFF
//   col 3  TIMESTAMP_1    = timestamps[row] >> 32
//   col 4  WRITE2         = flags[row] bit 0
//   col 5  WRITE4         = flags[row] bit 1
//   col 6  WRITE8         = flags[row] bit 2
//   col 7..14  VALUE[0..7] = byte decomposition of values[row] (LE)
//   col 15 MU             = flags[row] bit 3 (1 for active, 0 for padding)

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_store_trace_rows(
    uint64_t num_rows,
    const uint64_t *base_addresses,
    const uint64_t *timestamps,
    const uint64_t *values,
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols    // expected = 16
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t addr = base_addresses[row];
    uint64_t ts = timestamps[row];
    uint64_t v = values[row];
    uint64_t f = flags[row];

    table_data[base + 0]  = addr & 0xFFFFFFFFULL;
    table_data[base + 1]  = addr >> 32;
    table_data[base + 2]  = ts & 0xFFFFFFFFULL;
    table_data[base + 3]  = ts >> 32;
    table_data[base + 4]  = (f >> 0) & 1ULL;
    table_data[base + 5]  = (f >> 1) & 1ULL;
    table_data[base + 6]  = (f >> 2) & 1ULL;
    table_data[base + 7]  = (v >>  0) & 0xFFULL;
    table_data[base + 8]  = (v >>  8) & 0xFFULL;
    table_data[base + 9]  = (v >> 16) & 0xFFULL;
    table_data[base + 10] = (v >> 24) & 0xFFULL;
    table_data[base + 11] = (v >> 32) & 0xFFULL;
    table_data[base + 12] = (v >> 40) & 0xFFULL;
    table_data[base + 13] = (v >> 48) & 0xFFULL;
    table_data[base + 14] = (v >> 56) & 0xFFULL;
    table_data[base + 15] = (f >> 3) & 1ULL;
}
