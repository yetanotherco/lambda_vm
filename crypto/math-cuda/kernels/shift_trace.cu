// SHIFT table main-column generation. One thread per row, 29 columns.
//
// Per-row inputs:
//   in_values[row]      = the 64-bit operand (decomposed into 4 halfwords)
//   shift_amounts[row]  = the full arg2 (decomposed into [byte,byte,half,word])
//   flags[row] bits:
//     bit 0: direction (0 = left, 1 = right)
//     bit 1: signed    (arithmetic right shift)
//     bit 2: word_instr (32-bit shift)
//     bit 3: active (mu = 1; 0 = pure padding row)
//
// Column layout (matches `prover/src/tables/shift.rs::cols`):
//   0..3   IN_0..IN_3         (halfwords)
//   4      SHIFT_AMOUNT       (low byte of shift_amount)
//   5      DIRECTION
//   6      SIGNED
//   7      WORD_INSTR
//   8..9   OUT_0, OUT_1       (word halves of the shifted output)
//   10     IS_NEGATIVE
//   11     BIT_SHIFT
//   12     ZBS                (= bit_shift == 0; also set on padding rows)
//   13..17 X_0..X_4
//   18..21 Y_0..Y_3
//   22..24 LIMB_SHIFT_RAW_0..2
//   25     MU
//   26     SHIFT_B1           (shift_amount bits  8..15)
//   27     SHIFT_H1           (shift_amount bits 16..31)
//   28     SHIFT_HIGH         (shift_amount bits 32..63)

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256

extern "C" __global__ void generate_shift_trace_rows(
    uint64_t num_rows,
    const uint64_t *in_values,
    const uint64_t *shift_amounts,
    const uint64_t *flags,
    uint64_t *table_data,
    uint64_t num_cols   // expected = 29
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;

    uint64_t f = flags[row];
    uint64_t active = (f >> 3) & 1ULL;

    if (!active) {
        // Padding row: ZBS=1, all other columns 0.
        for (uint64_t c = 0; c < num_cols; ++c) {
            table_data[base + c] = 0;
        }
        table_data[base + 12] = 1ULL; // ZBS
        return;
    }

    uint64_t v = in_values[row];
    uint64_t sa = shift_amounts[row];
    uint64_t direction  = (f >> 0) & 1ULL;
    uint64_t signed_f   = (f >> 1) & 1ULL;
    uint64_t word_instr = (f >> 2) & 1ULL;
    uint64_t left  = 1ULL - direction;
    uint64_t right = direction;

    uint32_t in_h[4];
    in_h[0] = (uint32_t)((v >>  0) & 0xFFFFULL);
    in_h[1] = (uint32_t)((v >> 16) & 0xFFFFULL);
    in_h[2] = (uint32_t)((v >> 32) & 0xFFFFULL);
    in_h[3] = (uint32_t)((v >> 48) & 0xFFFFULL);

    uint32_t shift = (uint32_t)(sa & 0xFFULL);

    // is_negative: gated by signed
    uint32_t is_negative = (signed_f && ((in_h[3] >> 15) & 1U)) ? 1U : 0U;
    uint32_t extension = is_negative ? 0xFFFFU : 0U;

    // bit_shift = left ? shift & 15 : (256 - shift) & 15
    uint32_t bit_shift;
    if (left) {
        bit_shift = shift & 15U;
    } else {
        bit_shift = (256U - shift) & 15U;
    }
    uint32_t zbs = (bit_shift == 0) ? 1U : 0U;

    uint32_t x_arr[5] = {0, 0, 0, 0, 0};
    uint32_t y_arr[4] = {0, 0, 0, 0};
    if (zbs) {
        // bit_shift = 0 override: x[i] = in_h[i] (left), y[i] = in_h[i] (right)
        for (int i = 0; i < 4; ++i) {
            if (left) x_arr[i] = in_h[i];
            else      y_arr[i] = in_h[i];
        }
        x_arr[4] = 0;
    } else {
        uint32_t inv = 16U - bit_shift;
        for (int i = 0; i < 4; ++i) {
            x_arr[i] = (in_h[i] << bit_shift) & 0xFFFFU;
            y_arr[i] = in_h[i] >> inv;
        }
        x_arr[4] = (extension << bit_shift) & 0xFFFFU;
    }

    // limb_shift[4] one-hot: limb_idx = word_instr ? (shift>>4)&1 : (shift>>4)&3
    uint32_t limb_idx = word_instr ? ((shift >> 4) & 1U) : ((shift >> 4) & 3U);
    uint32_t ls[4] = {0, 0, 0, 0};
    ls[limb_idx] = 1U;

    // shifted[4]:
    //   left  -> sum_{j=0..i}   ls[j] * intra_left(i-j)
    //   right -> sum_{j=0..3-i} ls[j] * intra_right(i+j)
    //          + extension * sum_{j=4-i..3} ls[j]
    // intra_left(0) = x[0], intra_left(k) = x[k] + y[k-1]   (k >= 1)
    // intra_right(k) = y[k] + x[k+1]
    uint32_t shifted[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        uint32_t val = 0;
        if (left) {
            for (int j = 0; j <= i; ++j) {
                if (ls[j]) {
                    int k = i - j;
                    uint32_t intra = (k == 0) ? x_arr[0] : (x_arr[k] + y_arr[k - 1]);
                    val = (val + intra) & 0xFFFFU;
                }
            }
        }
        if (right) {
            for (int j = 0; j <= 3 - i; ++j) {
                if (ls[j]) {
                    uint32_t intra = y_arr[i + j] + x_arr[i + j + 1];
                    val = (val + intra) & 0xFFFFU;
                }
            }
            // extension contribution: j in [4-i, 3]
            for (int j = 4 - i; j < 4; ++j) {
                if (j >= 0 && ls[j]) {
                    val = (val + extension) & 0xFFFFU;
                }
            }
        }
        shifted[i] = val;
    }

    uint32_t out_0 = shifted[0] | (shifted[1] << 16);
    uint32_t out_1 = shifted[2] | (shifted[3] << 16);

    // Row layout (29 cols).
    table_data[base +  0] = (uint64_t)in_h[0];
    table_data[base +  1] = (uint64_t)in_h[1];
    table_data[base +  2] = (uint64_t)in_h[2];
    table_data[base +  3] = (uint64_t)in_h[3];
    table_data[base +  4] = (uint64_t)shift;
    table_data[base +  5] = direction;
    table_data[base +  6] = signed_f;
    table_data[base +  7] = word_instr;
    table_data[base +  8] = (uint64_t)out_0;
    table_data[base +  9] = (uint64_t)out_1;
    table_data[base + 10] = (uint64_t)is_negative;
    table_data[base + 11] = (uint64_t)bit_shift;
    table_data[base + 12] = (uint64_t)zbs;
    table_data[base + 13] = (uint64_t)x_arr[0];
    table_data[base + 14] = (uint64_t)x_arr[1];
    table_data[base + 15] = (uint64_t)x_arr[2];
    table_data[base + 16] = (uint64_t)x_arr[3];
    table_data[base + 17] = (uint64_t)x_arr[4];
    table_data[base + 18] = (uint64_t)y_arr[0];
    table_data[base + 19] = (uint64_t)y_arr[1];
    table_data[base + 20] = (uint64_t)y_arr[2];
    table_data[base + 21] = (uint64_t)y_arr[3];
    table_data[base + 22] = (uint64_t)ls[0];
    table_data[base + 23] = (uint64_t)ls[1];
    table_data[base + 24] = (uint64_t)ls[2];
    table_data[base + 25] = 1ULL; // MU = 1 for active rows
    table_data[base + 26] = (sa >>  8) & 0xFFULL;
    table_data[base + 27] = (sa >> 16) & 0xFFFFULL;
    table_data[base + 28] = (sa >> 32) & 0xFFFFFFFFULL;

    // LIMB_SHIFT_RAW only stores ls[0..2]; ls[3] is virtual.
    // No further writes needed.
}
