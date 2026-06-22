// MEMW (unaligned / split-timestamp path) main-column generation.
// One thread per row, 49 columns. GPU computes carry[7] from
// base_address_lo + (i+1), splits timestamps + values + olds + the 8
// old_timestamps into the column layout.
//
// Per-row inputs:
//   base_addresses[row]    = u64
//   timestamps[row]        = u64
//   values[8*row + i]      = u64 value[i], i=0..7
//   olds[8*row + i]        = u64 old[i],   i=0..7
//   old_timestamps[8*row+i]= u64 old_timestamp[i], i=0..7
//   flags[row] bits:
//     bit 0: is_register
//     bit 1: is_read   (mu_read; mu_write = 1 - is_read when active)
//     bit 2: write2
//     bit 3: write4
//     bit 4: write8
//     bit 5: active    (0 = padding row → all zeros)
//
// Column layout (matches `prover/src/tables/memw.rs::cols`):
//   0       IS_REGISTER
//   1..2    BASE_ADDRESS_0/_1
//   3..10   VALUE[0..7]
//   11..12  TIMESTAMP_0/_1
//   13..15  WRITE2/4/8
//   16..23  OLD[0..7]
//   24..30  CARRY[0..6]
//   31..46  OLD_TIMESTAMP[0..7] each lo,hi  (16 cols)
//   47      MU_READ
//   48      MU_WRITE

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_memw_trace_rows(
    uint64_t num_rows,
    const uint64_t *base_addresses,
    const uint64_t *timestamps,
    const uint64_t *values,         // 8 * num_rows
    const uint64_t *olds,           // 8 * num_rows
    const uint64_t *old_timestamps, // 8 * num_rows
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols               // expected = 49
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t f = flags[row];
    uint64_t active = (f >> 5) & 1ULL;

    if (!active) {
        for (uint64_t c = 0; c < num_cols; ++c) {
            table_data[base + c] = 0;
        }
        return;
    }

    uint64_t addr = base_addresses[row];
    uint64_t ts = timestamps[row];
    uint64_t addr_lo = addr & 0xFFFFFFFFULL;

    uint64_t is_register = (f >> 0) & 1ULL;
    uint64_t is_read     = (f >> 1) & 1ULL;
    uint64_t w2          = (f >> 2) & 1ULL;
    uint64_t w4          = (f >> 3) & 1ULL;
    uint64_t w8          = (f >> 4) & 1ULL;

    table_data[base +  0] = is_register;
    table_data[base +  1] = addr_lo;
    table_data[base +  2] = addr >> 32;

    uint64_t v_off = row * 8;
    table_data[base +  3] = values[v_off + 0];
    table_data[base +  4] = values[v_off + 1];
    table_data[base +  5] = values[v_off + 2];
    table_data[base +  6] = values[v_off + 3];
    table_data[base +  7] = values[v_off + 4];
    table_data[base +  8] = values[v_off + 5];
    table_data[base +  9] = values[v_off + 6];
    table_data[base + 10] = values[v_off + 7];

    table_data[base + 11] = ts & 0xFFFFFFFFULL;
    table_data[base + 12] = ts >> 32;
    table_data[base + 13] = w2;
    table_data[base + 14] = w4;
    table_data[base + 15] = w8;

    table_data[base + 16] = olds[v_off + 0];
    table_data[base + 17] = olds[v_off + 1];
    table_data[base + 18] = olds[v_off + 2];
    table_data[base + 19] = olds[v_off + 3];
    table_data[base + 20] = olds[v_off + 4];
    table_data[base + 21] = olds[v_off + 5];
    table_data[base + 22] = olds[v_off + 6];
    table_data[base + 23] = olds[v_off + 7];

    // carry[i] = 1 iff (addr_lo + i+1) >= 2^32
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        uint64_t overflows = (addr_lo + (uint64_t)(i + 1) >= (1ULL << 32)) ? 1ULL : 0ULL;
        table_data[base + 24 + i] = overflows;
    }

    // OLD_TIMESTAMP[i] as DWordWL (lo, hi), starting at col 31
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint64_t ots = old_timestamps[v_off + i];
        table_data[base + 31 + 2 * i + 0] = ots & 0xFFFFFFFFULL;
        table_data[base + 31 + 2 * i + 1] = ots >> 32;
    }

    table_data[base + 47] = is_read;
    table_data[base + 48] = 1ULL - is_read;
}
