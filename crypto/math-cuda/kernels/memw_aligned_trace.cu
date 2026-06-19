// MEMW_A (Memory Write/Read — Aligned) main-column generation.
// One thread per row, 29 columns.
//
// Per-row inputs:
//   base_addresses[row]   = u64 (DWordWHH: [Half, Half, Word])
//   timestamps[row]       = u64 (DWordWL)
//   old_timestamps[row]   = u64 (DWordWL, single — shared across all bytes)
//   values[8 * row + i]   = u64 value[i] for byte i (0..7)
//   olds[8 * row + i]     = u64 old[i] for byte i (0..7)
//   flags[row] bits:
//     bit 0: is_register
//     bit 1: is_read    (mu_read column)
//     bit 2: write2
//     bit 3: write4
//     bit 4: write8
//     bit 5: active     (1 = active row; 0 = padding, write all zeros)
// (mu_write = !is_read when active, 0 otherwise.)
//
// Column layout (matches `prover/src/tables/memw_aligned.rs::cols`):
//   0       IS_REGISTER
//   1..3    BASE_ADDRESS[0..2]   ([Half(0..15), Half(16..31), Word(32..63)])
//   4..11   VALUE[0..7]
//   12..13  TIMESTAMP_0/_1       (lo32, hi32)
//   14..16  WRITE2 / WRITE4 / WRITE8
//   17..24  OLD[0..7]
//   25..26  OLD_TIMESTAMP_0/_1   (lo32, hi32)
//   27      MU_READ
//   28      MU_WRITE

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_memw_aligned_trace_rows(
    uint64_t num_rows,
    const uint64_t *base_addresses,
    const uint64_t *timestamps,
    const uint64_t *old_timestamps,
    const uint64_t *values,    // 8 * num_rows
    const uint64_t *olds,      // 8 * num_rows
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols          // expected = 29
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
    uint64_t old_ts = old_timestamps[row];

    uint64_t is_register = (f >> 0) & 1ULL;
    uint64_t is_read     = (f >> 1) & 1ULL;
    uint64_t w2          = (f >> 2) & 1ULL;
    uint64_t w4          = (f >> 3) & 1ULL;
    uint64_t w8          = (f >> 4) & 1ULL;

    table_data[base + 0]  = is_register;
    table_data[base + 1]  = addr & 0xFFFFULL;
    table_data[base + 2]  = (addr >> 16) & 0xFFFFULL;
    table_data[base + 3]  = addr >> 32;

    uint64_t v_off = row * 8;
    table_data[base + 4]  = values[v_off + 0];
    table_data[base + 5]  = values[v_off + 1];
    table_data[base + 6]  = values[v_off + 2];
    table_data[base + 7]  = values[v_off + 3];
    table_data[base + 8]  = values[v_off + 4];
    table_data[base + 9]  = values[v_off + 5];
    table_data[base + 10] = values[v_off + 6];
    table_data[base + 11] = values[v_off + 7];

    table_data[base + 12] = ts & 0xFFFFFFFFULL;
    table_data[base + 13] = ts >> 32;

    table_data[base + 14] = w2;
    table_data[base + 15] = w4;
    table_data[base + 16] = w8;

    table_data[base + 17] = olds[v_off + 0];
    table_data[base + 18] = olds[v_off + 1];
    table_data[base + 19] = olds[v_off + 2];
    table_data[base + 20] = olds[v_off + 3];
    table_data[base + 21] = olds[v_off + 4];
    table_data[base + 22] = olds[v_off + 5];
    table_data[base + 23] = olds[v_off + 6];
    table_data[base + 24] = olds[v_off + 7];

    table_data[base + 25] = old_ts & 0xFFFFFFFFULL;
    table_data[base + 26] = old_ts >> 32;

    table_data[base + 27] = is_read;
    table_data[base + 28] = 1ULL - is_read; // mu_write
}
