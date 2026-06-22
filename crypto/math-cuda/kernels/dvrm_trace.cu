// DVRM (divide/remainder) table main-column generation. One thread per
// row, 34 columns. Kernel computes q, r, abs values, and sign aux
// directly from (n, d, signed) per the RISC-V spec.
//
// Per-row inputs:
//   ns[row]              = u64 numerator
//   ds[row]              = u64 denominator
//   flags[row] bits:
//     bit 0: signed
//     bit 1: active (0 = padding row → all zeros)
//   mu_qs[row], mu_rs[row] = per-row multiplicities
//
// Column layout (matches `prover/src/tables/dvrm.rs::cols`):
//   0..3   N_0..N_3         (DWordHL, 4 halfwords of n)
//   4..7   D_0..D_3         (DWordHL, 4 halfwords of d)
//   8      SIGNED
//   9..12  Q_0..Q_3
//   13..16 R_0..R_3
//   17     DIV_BY_ZERO
//   18     OVERFLOW
//   19..20 ABS_R_0/_1       (DWordWL: 2 words)
//   21..22 ABS_D_0/_1
//   23..26 N_SUB_R_0..N_SUB_R_3
//   27     SIGN_N_SUB_R
//   28     SIGN_N
//   29     SIGN_D
//   30     SIGN_Q
//   31     SIGN_R
//   32     MU_Q
//   33     MU_R

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define INT64_MIN_U  ((uint64_t)0x8000000000000000ULL)
#define UINT64_MAX_U ((uint64_t)0xFFFFFFFFFFFFFFFFULL)

// abs_value: if is_negative != 0 return (-(int64)v) as u64, else v
__device__ static inline uint64_t abs_value(uint64_t v, int is_negative) {
    if (is_negative) {
        // Two's-complement absolute value: works for INT64_MIN too (returns 2^63).
        return (uint64_t)(-(int64_t)v);
    }
    return v;
}

extern "C" __global__ void generate_dvrm_trace_rows(
    uint64_t num_rows,
    const uint64_t *ns,
    const uint64_t *ds,
    const uint64_t *flags,
    const uint64_t *mu_qs,
    const uint64_t *mu_rs,
    uint64_t *table_data,
    uint64_t num_cols     // expected = 34
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

    uint64_t n = ns[row];
    uint64_t d = ds[row];
    uint64_t is_signed = (f >> 0) & 1ULL;

    int div_by_zero = (d == 0) ? 1 : 0;
    int overflow = (is_signed && n == INT64_MIN_U && d == UINT64_MAX_U) ? 1 : 0;

    uint64_t q;
    uint64_t r;
    if (div_by_zero) {
        q = UINT64_MAX_U;
        r = n;
    } else if (overflow) {
        q = n;   // i64::MIN
        r = 0;
    } else if (is_signed) {
        int64_t ns_v = (int64_t)n;
        int64_t ds_v = (int64_t)d;
        // RISC-V signed div/rem semantics match Rust's wrapping_div/wrapping_rem
        // for non-overflow cases (overflow handled above).
        int64_t qs = ns_v / ds_v;
        int64_t rs = ns_v % ds_v;
        q = (uint64_t)qs;
        r = (uint64_t)rs;
    } else {
        q = n / d;
        r = n % d;
    }

    uint64_t n_sub_r = n - r;  // wrapping
    int sign_n = (is_signed && ((n >> 63) & 1ULL)) ? 1 : 0;
    int sign_d = (is_signed && ((d >> 63) & 1ULL)) ? 1 : 0;
    int sign_r = (is_signed && ((r >> 63) & 1ULL)) ? 1 : 0;
    // sign_q = signed * (1 - overflow)
    int sign_q = (is_signed && !overflow) ? 1 : 0;
    int sign_n_sub_r = (is_signed && ((n_sub_r >> 63) & 1ULL)) ? 1 : 0;

    uint64_t abs_r = abs_value(r, sign_r);
    uint64_t abs_d = abs_value(d, sign_d);

    table_data[base +  0] = n & 0xFFFFULL;
    table_data[base +  1] = (n >> 16) & 0xFFFFULL;
    table_data[base +  2] = (n >> 32) & 0xFFFFULL;
    table_data[base +  3] = (n >> 48) & 0xFFFFULL;
    table_data[base +  4] = d & 0xFFFFULL;
    table_data[base +  5] = (d >> 16) & 0xFFFFULL;
    table_data[base +  6] = (d >> 32) & 0xFFFFULL;
    table_data[base +  7] = (d >> 48) & 0xFFFFULL;
    table_data[base +  8] = is_signed;
    table_data[base +  9] = q & 0xFFFFULL;
    table_data[base + 10] = (q >> 16) & 0xFFFFULL;
    table_data[base + 11] = (q >> 32) & 0xFFFFULL;
    table_data[base + 12] = (q >> 48) & 0xFFFFULL;
    table_data[base + 13] = r & 0xFFFFULL;
    table_data[base + 14] = (r >> 16) & 0xFFFFULL;
    table_data[base + 15] = (r >> 32) & 0xFFFFULL;
    table_data[base + 16] = (r >> 48) & 0xFFFFULL;
    table_data[base + 17] = (uint64_t)div_by_zero;
    table_data[base + 18] = (uint64_t)overflow;
    table_data[base + 19] = abs_r & 0xFFFFFFFFULL;
    table_data[base + 20] = abs_r >> 32;
    table_data[base + 21] = abs_d & 0xFFFFFFFFULL;
    table_data[base + 22] = abs_d >> 32;
    table_data[base + 23] = n_sub_r & 0xFFFFULL;
    table_data[base + 24] = (n_sub_r >> 16) & 0xFFFFULL;
    table_data[base + 25] = (n_sub_r >> 32) & 0xFFFFULL;
    table_data[base + 26] = (n_sub_r >> 48) & 0xFFFFULL;
    table_data[base + 27] = (uint64_t)sign_n_sub_r;
    table_data[base + 28] = (uint64_t)sign_n;
    table_data[base + 29] = (uint64_t)sign_d;
    table_data[base + 30] = (uint64_t)sign_q;
    table_data[base + 31] = (uint64_t)sign_r;
    table_data[base + 32] = mu_qs[row];
    table_data[base + 33] = mu_rs[row];
}
