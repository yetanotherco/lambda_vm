// GPU MUL (and later DVRM) trace generation. Heavy 128-bit compute via CUDA's
// built-in __int128. One thread per deduped op. Dedup (+ dual multiplicity
// counters) is on the host. Mirrors generate_mul_trace + MulOperation in
// prover/src/tables/mul.rs. Padding rows (r >= n) stay zero.
//
// RAW_PRODUCT columns hold a truncated-128-bit u64 that may exceed p, so they
// are reduced with from_u64 (matching the CPU's FE::from). All other columns
// are 16-bit halves / bits / counts, written raw.

#include "goldilocks.cuh"
#include <cstdint>

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}
// DWordHL: 4 little-endian 16-bit halves.
__device__ __forceinline__ void dwhl(uint64_t *cols, uint64_t nrows, int col,
                                     uint64_t r, uint64_t v) {
  sc(cols, nrows, col + 0, r, v & 0xFFFF);
  sc(cols, nrows, col + 1, r, (v >> 16) & 0xFFFF);
  sc(cols, nrows, col + 2, r, (v >> 32) & 0xFFFF);
  sc(cols, nrows, col + 3, r, (v >> 48) & 0xFFFF);
}
// DWordWL: 2 little-endian 32-bit words.
__device__ __forceinline__ void dwwl(uint64_t *cols, uint64_t nrows, int col,
                                     uint64_t r, uint64_t v) {
  sc(cols, nrows, col + 0, r, v & 0xFFFFFFFF);
  sc(cols, nrows, col + 1, r, v >> 32);
}
// Reduce a u64 into [0, p) — matches GoldilocksField from_u64 / FE::from.
__device__ __forceinline__ uint64_t from_u64(uint64_t x) {
  return x >= goldilocks::PRIME ? x - goldilocks::PRIME : x;
}

// MUL columns (must match prover/src/tables/mul.rs `cols`).
#define M_LHS_0 0
#define M_LHS_SIGNED 4
#define M_RHS_0 5
#define M_RHS_SIGNED 9
#define M_LO_0 10
#define M_HI_0 14
#define M_LHS_IS_NEGATIVE 18
#define M_RHS_IS_NEGATIVE 19
#define M_RAW_0 20
#define M_MU_LO 24
#define M_MU_HI 25

extern "C" __global__ void
trace_mul_kernel(const uint64_t *__restrict__ lhs,
                 const uint64_t *__restrict__ rhs,
                 const uint64_t *__restrict__ flags, // bit0=lhs_signed,bit1=rhs_signed
                 const uint64_t *__restrict__ mu_lo,
                 const uint64_t *__restrict__ mu_hi, uint64_t n, uint64_t nrows,
                 uint64_t *__restrict__ cols) { // 26 * nrows, zeroed
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n)
    return; // padding rows stay zero

  uint64_t l = lhs[r], rr = rhs[r], f = flags[r];
  int ls = f & 1, rs = (f >> 1) & 1;

  // 128-bit signed product (computed unsigned to avoid signed-overflow UB;
  // bit-identical to Rust i128::wrapping_mul).
  unsigned __int128 ua =
      ls ? (unsigned __int128)(__int128)(int64_t)l : (unsigned __int128)l;
  unsigned __int128 ub =
      rs ? (unsigned __int128)(__int128)(int64_t)rr : (unsigned __int128)rr;
  unsigned __int128 p = ua * ub;
  uint64_t lo = (uint64_t)p;
  uint64_t hi = (uint64_t)(p >> 64);

  int lneg = ls && ((int64_t)l < 0);
  int rneg = rs && ((int64_t)rr < 0);

  uint64_t lhs_ext[8], rhs_ext[8];
  uint64_t lfill = lneg ? 0xFFFF : 0;
  uint64_t rfill = rneg ? 0xFFFF : 0;
  for (int i = 0; i < 4; i++) {
    lhs_ext[i] = (l >> (16 * i)) & 0xFFFF;
    rhs_ext[i] = (rr >> (16 * i)) & 0xFFFF;
  }
  for (int i = 4; i < 8; i++) {
    lhs_ext[i] = lfill;
    rhs_ext[i] = rfill;
  }

  uint64_t raw[4];
  for (int i = 0; i < 4; i++) {
    unsigned __int128 sum = 0;
    for (int k = 0; k <= 1; k++) {
      int idx = 2 * i + k;
      if (idx < 8) {
        for (int j = 0; j <= idx; j++) {
          if (j < 8 && (idx - j) < 8) {
            sum += ((unsigned __int128)lhs_ext[j] * rhs_ext[idx - j])
                   << (16 * k);
          }
        }
      }
    }
    raw[i] = (uint64_t)sum;
  }

  dwhl(cols, nrows, M_LHS_0, r, l);
  sc(cols, nrows, M_LHS_SIGNED, r, (uint64_t)ls);
  dwhl(cols, nrows, M_RHS_0, r, rr);
  sc(cols, nrows, M_RHS_SIGNED, r, (uint64_t)rs);
  dwhl(cols, nrows, M_LO_0, r, lo);
  dwhl(cols, nrows, M_HI_0, r, hi);
  sc(cols, nrows, M_LHS_IS_NEGATIVE, r, (uint64_t)lneg);
  sc(cols, nrows, M_RHS_IS_NEGATIVE, r, (uint64_t)rneg);
  for (int i = 0; i < 4; i++)
    sc(cols, nrows, M_RAW_0 + i, r, from_u64(raw[i]));
  sc(cols, nrows, M_MU_LO, r, mu_lo[r]);
  sc(cols, nrows, M_MU_HI, r, mu_hi[r]);
}

// DVRM columns (must match prover/src/tables/dvrm.rs `cols`).
#define DV_N_0 0
#define DV_D_0 4
#define DV_SIGNED 8
#define DV_Q_0 9
#define DV_R_0 13
#define DV_DIV_BY_ZERO 17
#define DV_OVERFLOW 18
#define DV_ABS_R_0 19
#define DV_ABS_D_0 21
#define DV_N_SUB_R_0 23
#define DV_SIGN_N_SUB_R 27
#define DV_SIGN_N 28
#define DV_SIGN_D 29
#define DV_SIGN_Q 30
#define DV_SIGN_R 31
#define DV_MU_Q 32
#define DV_MU_R 33

extern "C" __global__ void
trace_dvrm_kernel(const uint64_t *__restrict__ nn,
                  const uint64_t *__restrict__ dd,
                  const uint64_t *__restrict__ flags, // bit0=signed
                  const uint64_t *__restrict__ mu_q,
                  const uint64_t *__restrict__ mu_r, uint64_t n_ops,
                  uint64_t nrows,
                  uint64_t *__restrict__ cols) { // 34 * nrows, zeroed
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_ops)
    return; // padding rows stay zero

  uint64_t nv = nn[r], dv = dd[r];
  int is_signed = flags[r] & 1;

  int div0 = (dv == 0);
  int overflow = is_signed && (nv == 0x8000000000000000ULL) &&
                 (dv == 0xFFFFFFFFFFFFFFFFULL);

  // Quotient / remainder (RISC-V). The signed / and % branches run only when
  // !div0 && !overflow, so no division overflow / div-by-zero (no UB).
  uint64_t q, rem;
  if (div0) {
    q = 0xFFFFFFFFFFFFFFFFULL;
    rem = nv;
  } else if (overflow) {
    q = nv; // i64::MIN
    rem = 0;
  } else if (is_signed) {
    q = (uint64_t)((int64_t)nv / (int64_t)dv);
    rem = (uint64_t)((int64_t)nv % (int64_t)dv);
  } else {
    q = nv / dv;
    rem = nv % dv;
  }

  int sign_n = is_signed && ((nv >> 63) & 1);
  int sign_d = is_signed && ((dv >> 63) & 1);
  int sign_q = is_signed && !overflow;
  int sign_r = is_signed && ((rem >> 63) & 1);

  // unsigned_abs of a negative i64 = two's-complement negate (~v + 1).
  uint64_t abs_r = sign_r ? (~rem + 1) : rem;
  uint64_t abs_d = sign_d ? (~dv + 1) : dv;
  uint64_t n_sub_r = nv - rem;
  int sign_n_sub_r = is_signed && ((n_sub_r >> 63) & 1);

  dwhl(cols, nrows, DV_N_0, r, nv);
  dwhl(cols, nrows, DV_D_0, r, dv);
  sc(cols, nrows, DV_SIGNED, r, (uint64_t)is_signed);
  dwhl(cols, nrows, DV_Q_0, r, q);
  dwhl(cols, nrows, DV_R_0, r, rem);
  sc(cols, nrows, DV_DIV_BY_ZERO, r, (uint64_t)div0);
  sc(cols, nrows, DV_OVERFLOW, r, (uint64_t)overflow);
  dwwl(cols, nrows, DV_ABS_R_0, r, abs_r);
  dwwl(cols, nrows, DV_ABS_D_0, r, abs_d);
  dwhl(cols, nrows, DV_N_SUB_R_0, r, n_sub_r);
  sc(cols, nrows, DV_SIGN_N_SUB_R, r, (uint64_t)sign_n_sub_r);
  sc(cols, nrows, DV_SIGN_N, r, (uint64_t)sign_n);
  sc(cols, nrows, DV_SIGN_D, r, (uint64_t)sign_d);
  sc(cols, nrows, DV_SIGN_Q, r, (uint64_t)sign_q);
  sc(cols, nrows, DV_SIGN_R, r, (uint64_t)sign_r);
  sc(cols, nrows, DV_MU_Q, r, mu_q[r]);
  sc(cols, nrows, DV_MU_R, r, mu_r[r]);
}
