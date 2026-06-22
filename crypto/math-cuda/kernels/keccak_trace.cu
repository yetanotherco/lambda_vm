// KECCAK core chip main-column generation. One thread per row, 511 columns.
// Kernel does the byte/halfword decomposition of the address, 25 input
// lanes (5×5×8 bytes), 25 output lanes, and the 25 state_ptr DWordHL
// addresses.
//
// Per-row inputs:
//   timestamps[row]       = u64
//   state_addrs[row]      = u64
//   inputs[25*row + lane] = u64 input[lane]   (lane = x + 5*y, 0..24)
//   outputs[25*row + lane]= u64 output[lane]
//   flags[row] bits:
//     bit 0: active (0 = padding row → only state_ptr[lane][0] = 8*lane_idx)
//
// Column layout (matches `prover/src/tables/keccak.rs::cols`):
//   0..1    TIMESTAMP_0/_1
//   2..9    addr[0..7]                   (DWordBL: 8 bytes of state_addr)
//   10..209 INPUT_STATE  [x][y][byte]    (5*5*8 = 200 bytes; idx (x+5y)*8+b)
//   210..409 OUTPUT_STATE [x][y][byte]   (200 bytes)
//   410..509 STATE_PTR    [lane][hw]     (25 lanes × 4 halfwords)
//   510     MU

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 64

extern "C" __global__ void generate_keccak_trace_rows(
    uint64_t num_rows,
    const uint64_t *timestamps,
    const uint64_t *state_addrs,
    const uint64_t *inputs,    // 25 * num_rows
    const uint64_t *outputs,   // 25 * num_rows
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols          // expected = 511
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t f = flags[row];
    uint64_t active = (f >> 0) & 1ULL;

    // Zero the entire row first; we'll overwrite the active fields below.
    for (uint64_t c = 0; c < num_cols; ++c) {
        table_data[base + c] = 0;
    }

    if (!active) {
        // Padding row: state_ptr[lane][0] = 8 * lane_idx (per spec).
        #pragma unroll
        for (int lane = 0; lane < 25; ++lane) {
            table_data[base + 410 + lane * 4 + 0] = (uint64_t)(lane * 8);
        }
        return;
    }

    uint64_t ts = timestamps[row];
    uint64_t addr = state_addrs[row];

    // Timestamp split
    table_data[base + 0] = ts & 0xFFFFFFFFULL;
    table_data[base + 1] = ts >> 32;

    // addr[0..7]: bytes of state_addr
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        table_data[base + 2 + b] = (addr >> (b * 8)) & 0xFFULL;
    }

    uint64_t lane_off = row * 25;

    // input_state and output_state: 25 lanes, 8 bytes each
    #pragma unroll
    for (int lane = 0; lane < 25; ++lane) {
        uint64_t in_lane  = inputs[lane_off + lane];
        uint64_t out_lane = outputs[lane_off + lane];
        // Layout matches cols::input_state(x,y,b) where lane = x + 5*y,
        // so the contiguous index per lane is lane*8 + b.
        for (int b = 0; b < 8; ++b) {
            table_data[base + 10  + lane * 8 + b] = (in_lane  >> (b * 8)) & 0xFFULL;
            table_data[base + 210 + lane * 8 + b] = (out_lane >> (b * 8)) & 0xFFULL;
        }
    }

    // state_ptr[lane] = addr + 8*lane as DWordHL (4 halfwords). Wraps
    // silently on u64 overflow, where the CPU path (`checked_add().expect()`)
    // would panic. The executor's keccak-syscall handler rejects addresses
    // near u64::MAX before the witness reaches the prover, so the
    // precondition `addr + 8*24 <= u64::MAX` holds and the two paths agree.
    // If that contract is ever broken, this kernel produces wrapped (wrong)
    // bus tokens silently — surface the precondition explicitly there.
    #pragma unroll
    for (int lane = 0; lane < 25; ++lane) {
        uint64_t ptr = addr + (uint64_t)(lane * 8);
        table_data[base + 410 + lane * 4 + 0] = ptr & 0xFFFFULL;
        table_data[base + 410 + lane * 4 + 1] = (ptr >> 16) & 0xFFFFULL;
        table_data[base + 410 + lane * 4 + 2] = (ptr >> 32) & 0xFFFFULL;
        table_data[base + 410 + lane * 4 + 3] = (ptr >> 48) & 0xFFFFULL;
    }

    // mu = 1
    table_data[base + 510] = 1ULL;
}
