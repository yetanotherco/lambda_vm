// GPU LT-table trace generation (per-row compute over already-deduped ops).
//
// One thread per unique op. Dedup (operand tuple -> summed multiplicity) is done
// on the host; this kernel just fills the 17 LT columns, column-major. Mirrors
// the fill loop in `generate_lt_trace` (prover/src/tables/lt.rs). Padding rows
// (r >= n) stay zero (alloc_zeros), matching the CPU's FE::zero padding.
//
// Raw canonical u64 throughout (every value < p): 32-bit words, 16-bit halves,
// bits, and multiplicities all fit, so no field reduction.

#include <cstdint>

// LT column indices (must match prover/src/tables/lt.rs `cols`).
#define LT_LHS_0 0
#define LT_LHS_1 1
#define LT_LHS_2 2
#define LT_RHS_0 3
#define LT_RHS_1 4
#define LT_RHS_2 5
#define LT_SIGNED 6
#define LT_LT 7
#define LT_SUB_0 8
#define LT_SUB_1 9
#define LT_SUB_2 10
#define LT_SUB_3 11
#define LT_LHS_MSB 12
#define LT_RHS_MSB 13
#define LT_INVERT 14
#define LT_OUT 15
#define LT_MU 16

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

extern "C" __global__ void
trace_lt_kernel(const uint64_t *__restrict__ lhs,   // n unique ops
                const uint64_t *__restrict__ rhs,   // n
                const uint64_t *__restrict__ flags, // n: bit0=signed, bit1=invert
                const uint64_t *__restrict__ mult,  // n: summed multiplicity
                uint64_t n, uint64_t nrows,
                uint64_t *__restrict__ cols) { // 17 * nrows, zero-initialized
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n)
    return; // padding rows stay zero

  uint64_t l = lhs[r];
  uint64_t rr = rhs[r];
  uint64_t f = flags[r];
  uint64_t mu = mult[r];
  int is_signed = f & 1;
  int invert = (f >> 1) & 1;

  // LHS / RHS as DWordHHW: [word(32), half(16), half(16)].
  sc(cols, nrows, LT_LHS_0, r, l & 0xFFFFFFFF);
  sc(cols, nrows, LT_LHS_1, r, (l >> 32) & 0xFFFF);
  sc(cols, nrows, LT_LHS_2, r, (l >> 48) & 0xFFFF);
  sc(cols, nrows, LT_RHS_0, r, rr & 0xFFFFFFFF);
  sc(cols, nrows, LT_RHS_1, r, (rr >> 32) & 0xFFFF);
  sc(cols, nrows, LT_RHS_2, r, (rr >> 48) & 0xFFFF);

  sc(cols, nrows, LT_SIGNED, r, (uint64_t)is_signed);

  int lt = is_signed ? ((int64_t)l < (int64_t)rr) : (l < rr);
  sc(cols, nrows, LT_LT, r, (uint64_t)lt);

  // lhs - rhs (wrapping) as DWordHL: 4 x 16-bit halves.
  uint64_t sub = l - rr;
  sc(cols, nrows, LT_SUB_0, r, sub & 0xFFFF);
  sc(cols, nrows, LT_SUB_1, r, (sub >> 16) & 0xFFFF);
  sc(cols, nrows, LT_SUB_2, r, (sub >> 32) & 0xFFFF);
  sc(cols, nrows, LT_SUB_3, r, (sub >> 48) & 0xFFFF);

  sc(cols, nrows, LT_LHS_MSB, r, (l >> 63) & 1);
  sc(cols, nrows, LT_RHS_MSB, r, (rr >> 63) & 1);

  sc(cols, nrows, LT_INVERT, r, (uint64_t)invert);
  sc(cols, nrows, LT_OUT, r, (uint64_t)(lt ^ invert));
  sc(cols, nrows, LT_MU, r, mu);
}
