// Page-table main-column generation: one thread per byte in the page,
// writes the 5-column row in row-major layout.
//
// Per-row columns (matches `prover/src/tables/page.rs`):
//   col 0  OFFSET        = byte index 0..page_size-1
//   col 1  INIT          = init_values[offset]
//   col 2  FINI          = final_values[offset]
//   col 3  TIMESTAMP_LO  = final_timestamps[offset] & 0xFFFFFFFF
//   col 4  TIMESTAMP_HI  = final_timestamps[offset] >> 32
//
// The caller has already flattened the `FinalStateMap` HashMap into the
// three parallel u64 arrays (length = page_size each), resolving
// "never accessed" → (timestamp=0, fini=init_value). This kernel only
// does the row layout, which is the bulk of the work for large pages.

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_page_trace_rows(
    uint64_t page_size,
    const uint64_t *init_values,        // length page_size
    const uint64_t *final_values,       // length page_size
    const uint64_t *final_timestamps,   // length page_size
    uint64_t *table_data,               // length page_size * num_cols, row-major
    uint64_t num_cols                   // expected = 5
) {
    uint64_t offset = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (offset >= page_size) return;
    uint64_t base = offset * num_cols;
    uint64_t ts = final_timestamps[offset];
    table_data[base + 0] = offset;
    table_data[base + 1] = init_values[offset];
    table_data[base + 2] = final_values[offset];
    table_data[base + 3] = ts & 0xFFFFFFFFULL;
    table_data[base + 4] = ts >> 32;
}
