// MEMW_R (Memory Write/Read -- Register) main-column generation.
// One thread per row, 10 columns.
//
// Per-row inputs:
//   base_addresses[row]  = 2 * register_index (CPU contract)
//   timestamps[row]      = u64 (split into [lo32, hi32])
//   old_timestamps[row]  = u64 (op.old_timestamp[0]; lo32 used, hi32 shared with TIMESTAMP_1)
//   values[2*row + i]    = u64 value[i], i=0,1
//   olds[2*row + i]      = u64 old[i],   i=0,1
//   flags[row] bits:
//     bit 0: is_read    (mu_read; mu_write = 1 - is_read when active)
//     bit 1: active     (0 = padding row → all zeros)
//
// Column layout (matches `prover/src/tables/memw_register.rs::cols`):
//   0  ADDRESS           = base_address / 2
//   1  TIMESTAMP_0       = timestamp & 0xFFFFFFFF
//   2  TIMESTAMP_1       = timestamp >> 32
//   3  VAL_0             = value[0]
//   4  VAL_1             = value[1]
//   5  OLD_0             = old[0]
//   6  OLD_1             = old[1]
//   7  OLD_TIMESTAMP_LO  = old_timestamp & 0xFFFFFFFF
//   8  MU_READ
//   9  MU_WRITE

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_memw_register_trace_rows(
    uint64_t num_rows,
    const uint64_t *base_addresses,
    const uint64_t *timestamps,
    const uint64_t *old_timestamps,
    const uint64_t *values,    // 2 * num_rows
    const uint64_t *olds,      // 2 * num_rows
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols          // expected = 10
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

    uint64_t addr = base_addresses[row];
    uint64_t ts = timestamps[row];
    uint64_t old_ts = old_timestamps[row];
    uint64_t is_read = (f >> 0) & 1ULL;

    uint64_t v_off = row * 2;

    table_data[base + 0] = addr >> 1;            // ADDRESS = base / 2
    table_data[base + 1] = ts & 0xFFFFFFFFULL;   // TIMESTAMP_0
    table_data[base + 2] = ts >> 32;             // TIMESTAMP_1
    table_data[base + 3] = values[v_off + 0];    // VAL_0
    table_data[base + 4] = values[v_off + 1];    // VAL_1
    table_data[base + 5] = olds[v_off + 0];      // OLD_0
    table_data[base + 6] = olds[v_off + 1];      // OLD_1
    table_data[base + 7] = old_ts & 0xFFFFFFFFULL; // OLD_TIMESTAMP_LO
    table_data[base + 8] = is_read;              // MU_READ
    table_data[base + 9] = 1ULL - is_read;       // MU_WRITE
}
