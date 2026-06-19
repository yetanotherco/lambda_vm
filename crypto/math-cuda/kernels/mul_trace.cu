// MUL table main-column generation. One thread per row, 26 columns.
//
// Per-row inputs:
//   lhs_values[row]      = u64 left operand
//   rhs_values[row]      = u64 right operand
//   flags[row] bits:
//     bit 0: lhs_signed
//     bit 1: rhs_signed
//     bit 2: active (0 = padding row → all zeros)
//   mu_lo[row], mu_hi[row] = per-row multiplicities (ALU lo / hi)
//
// Column layout (matches `prover/src/tables/mul.rs::cols`):
//   0..3   LHS_0..LHS_3       (DWordHL: 4 halfwords)
//   4      LHS_SIGNED
//   5..8   RHS_0..RHS_3
//   9      RHS_SIGNED
//   10..13 LO_0..LO_3         (lower 64-bit product)
//   14..17 HI_0..HI_3         (upper 64-bit product)
//   18     LHS_IS_NEGATIVE
//   19     RHS_IS_NEGATIVE
//   20..23 RAW_PRODUCT_0..3   (convolution intermediates, fit in B51)
//   24     MU_LO
//   25     MU_HI

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define SIGN_FILL 0xFFFFULL

extern "C" __global__ void generate_mul_trace_rows(
    uint64_t num_rows,
    const uint64_t *lhs_values,
    const uint64_t *rhs_values,
    const uint64_t *flags,
    const uint64_t *mu_lo,
    const uint64_t *mu_hi,
    uint64_t *table_data,
    uint64_t num_cols           // expected = 26
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;
    uint64_t f = flags[row];
    uint64_t active = (f >> 2) & 1ULL;

    if (!active) {
        for (uint64_t c = 0; c < num_cols; ++c) {
            table_data[base + c] = 0;
        }
        return;
    }

    uint64_t lhs = lhs_values[row];
    uint64_t rhs = rhs_values[row];
    uint64_t lhs_signed = (f >> 0) & 1ULL;
    uint64_t rhs_signed = (f >> 1) & 1ULL;

    uint64_t lhs_is_neg = (lhs_signed && ((int64_t)lhs < 0)) ? 1ULL : 0ULL;
    uint64_t rhs_is_neg = (rhs_signed && ((int64_t)rhs < 0)) ? 1ULL : 0ULL;

    // 128-bit signed-aware product via __int128 (nvcc supports it).
    __int128 a = lhs_signed ? (__int128)(int64_t)lhs : (__int128)(unsigned __int128)lhs;
    __int128 b = rhs_signed ? (__int128)(int64_t)rhs : (__int128)(unsigned __int128)rhs;
    __int128 product = a * b;
    uint64_t lo = (uint64_t)product;
    uint64_t hi = (uint64_t)((unsigned __int128)product >> 64);

    // Sign-extended 8-halfword operands for the convolution.
    uint64_t lhs_ext[8];
    uint64_t rhs_ext[8];
    lhs_ext[0] = lhs & 0xFFFFULL;
    lhs_ext[1] = (lhs >> 16) & 0xFFFFULL;
    lhs_ext[2] = (lhs >> 32) & 0xFFFFULL;
    lhs_ext[3] = (lhs >> 48) & 0xFFFFULL;
    rhs_ext[0] = rhs & 0xFFFFULL;
    rhs_ext[1] = (rhs >> 16) & 0xFFFFULL;
    rhs_ext[2] = (rhs >> 32) & 0xFFFFULL;
    rhs_ext[3] = (rhs >> 48) & 0xFFFFULL;
    uint64_t lhs_fill = lhs_is_neg ? SIGN_FILL : 0ULL;
    uint64_t rhs_fill = rhs_is_neg ? SIGN_FILL : 0ULL;
    for (int j = 4; j < 8; ++j) {
        lhs_ext[j] = lhs_fill;
        rhs_ext[j] = rhs_fill;
    }

    // raw_product[i] = sum_{k=0..1} 2^(16k) * sum_{j=0..2i+k} lhs_ext[j] * rhs_ext[2i+k-j]
    uint64_t raw[4];
    for (int i = 0; i < 4; ++i) {
        // u64 holds the sum: each lhs_ext[j]*rhs_ext[m] is ≤ 32 bits, summed
        // ≤ 8 ways then shifted ≤ 16 → fits in ~52 bits, well within u64.
        uint64_t sum = 0;
        for (int k = 0; k <= 1; ++k) {
            int idx = 2 * i + k;
            if (idx < 8) {
                uint64_t inner = 0;
                for (int j = 0; j <= idx; ++j) {
                    if (j < 8 && (idx - j) < 8) {
                        inner += lhs_ext[j] * rhs_ext[idx - j];
                    }
                }
                sum += inner << (16 * k);
            }
        }
        raw[i] = sum;
    }

    table_data[base +  0] = lhs & 0xFFFFULL;
    table_data[base +  1] = (lhs >> 16) & 0xFFFFULL;
    table_data[base +  2] = (lhs >> 32) & 0xFFFFULL;
    table_data[base +  3] = (lhs >> 48) & 0xFFFFULL;
    table_data[base +  4] = lhs_signed;
    table_data[base +  5] = rhs & 0xFFFFULL;
    table_data[base +  6] = (rhs >> 16) & 0xFFFFULL;
    table_data[base +  7] = (rhs >> 32) & 0xFFFFULL;
    table_data[base +  8] = (rhs >> 48) & 0xFFFFULL;
    table_data[base +  9] = rhs_signed;
    table_data[base + 10] = lo & 0xFFFFULL;
    table_data[base + 11] = (lo >> 16) & 0xFFFFULL;
    table_data[base + 12] = (lo >> 32) & 0xFFFFULL;
    table_data[base + 13] = (lo >> 48) & 0xFFFFULL;
    table_data[base + 14] = hi & 0xFFFFULL;
    table_data[base + 15] = (hi >> 16) & 0xFFFFULL;
    table_data[base + 16] = (hi >> 32) & 0xFFFFULL;
    table_data[base + 17] = (hi >> 48) & 0xFFFFULL;
    table_data[base + 18] = lhs_is_neg;
    table_data[base + 19] = rhs_is_neg;
    table_data[base + 20] = raw[0];
    table_data[base + 21] = raw[1];
    table_data[base + 22] = raw[2];
    table_data[base + 23] = raw[3];
    table_data[base + 24] = mu_lo[row];
    table_data[base + 25] = mu_hi[row];
}
