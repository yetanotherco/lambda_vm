// GPU ALU-table trace generation (per-row compute over already-deduped ops).
//
// Shared file for the simple ALU tables. Dedup (operand tuple -> summed
// multiplicity) is done on the host; each kernel fills its table's columns
// column-major, one thread per unique op. Padding rows (r >= n) stay zero
// (alloc_zeros), matching the CPU's FE::zero padding. Raw canonical u64
// throughout (all values < p) — no field reduction.
//
// Mirrors generate_eq_trace / generate_bytewise_trace in prover/src/tables/.

#include <cstdint>

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

// ===================== EQ (12 columns) =====================
// cols: A_0,A_1, B_0,B_1, INVERT, RES, DIFF_0..3, EQ, MU
#define EQ_A_0 0
#define EQ_A_1 1
#define EQ_B_0 2
#define EQ_B_1 3
#define EQ_INVERT 4
#define EQ_RES 5
#define EQ_DIFF_0 6
#define EQ_DIFF_1 7
#define EQ_DIFF_2 8
#define EQ_DIFF_3 9
#define EQ_EQ 10
#define EQ_MU 11

extern "C" __global__ void
trace_eq_kernel(const uint64_t *__restrict__ a, const uint64_t *__restrict__ b,
                const uint64_t *__restrict__ flags, // bit0 = invert
                const uint64_t *__restrict__ mult, uint64_t n, uint64_t nrows,
                uint64_t *__restrict__ cols) { // 12 * nrows, zero-initialized
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n)
    return;
  uint64_t av = a[r], bv = b[r];
  int invert = flags[r] & 1;

  sc(cols, nrows, EQ_A_0, r, av & 0xFFFFFFFF);
  sc(cols, nrows, EQ_A_1, r, av >> 32);
  sc(cols, nrows, EQ_B_0, r, bv & 0xFFFFFFFF);
  sc(cols, nrows, EQ_B_1, r, bv >> 32);

  int eq = (av == bv);
  sc(cols, nrows, EQ_INVERT, r, (uint64_t)invert);
  sc(cols, nrows, EQ_RES, r, (uint64_t)(eq ^ invert));

  uint64_t diff = av - bv;
  sc(cols, nrows, EQ_DIFF_0, r, diff & 0xFFFF);
  sc(cols, nrows, EQ_DIFF_1, r, (diff >> 16) & 0xFFFF);
  sc(cols, nrows, EQ_DIFF_2, r, (diff >> 32) & 0xFFFF);
  sc(cols, nrows, EQ_DIFF_3, r, (diff >> 48) & 0xFFFF);

  sc(cols, nrows, EQ_EQ, r, (uint64_t)eq);
  sc(cols, nrows, EQ_MU, r, mult[r]);
}

// ===================== BYTEWISE (26 columns) =====================
// cols: A[0..8], B[8..16], OP(16), RES[17..25], MU(25). op in {AND=0,OR=1,XOR=2}.
extern "C" __global__ void
trace_bytewise_kernel(const uint64_t *__restrict__ a,
                      const uint64_t *__restrict__ b,
                      const uint64_t *__restrict__ op,
                      const uint64_t *__restrict__ mult, uint64_t n,
                      uint64_t nrows,
                      uint64_t *__restrict__ cols) { // 26 * nrows, zeroed
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n)
    return;
  uint64_t av = a[r], bv = b[r], o = op[r];
  uint64_t res = (o == 0) ? (av & bv) : (o == 1) ? (av | bv) : (av ^ bv);

  // A[0..8], B[8..16], RES[17..25] are little-endian byte decompositions.
  for (int i = 0; i < 8; i++) {
    sc(cols, nrows, /*A*/ i, r, (av >> (i * 8)) & 0xFF);
    sc(cols, nrows, /*B*/ 8 + i, r, (bv >> (i * 8)) & 0xFF);
    sc(cols, nrows, /*RES*/ 17 + i, r, (res >> (i * 8)) & 0xFF);
  }
  sc(cols, nrows, /*OP*/ 16, r, o);
  sc(cols, nrows, /*MU*/ 25, r, mult[r]);
}
