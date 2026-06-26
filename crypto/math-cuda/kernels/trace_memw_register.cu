// GPU MEMW_R (register memory) trace generation. One thread per row, no dedup
// (multiplicity comes from is_read). Mirrors generate_memw_register_trace in
// prover/src/tables/memw_register.rs. Padding rows (r >= n) stay all-zero,
// matching the CPU builder (which only fills `operations.len()` rows).
//
// Per-op input is interleaved with stride MR_STRIDE:
//   [base_address, timestamp, value0, value1, old0, old1, old_timestamp0, is_read]
// Register value/old words are < 2^32 but reduced via from_u64 anyway, matching
// the CPU's set_u64 (= FE::from). Timestamp limbs are < 2^32, written raw.

#include "goldilocks.cuh"
#include <cstdint>

// MEMW_R column indices (must match prover/src/tables/memw_register.rs `cols`).
#define MR_ADDRESS 0
#define MR_TIMESTAMP_0 1
#define MR_TIMESTAMP_1 2
#define MR_VAL_0 3
#define MR_VAL_1 4
#define MR_OLD_0 5
#define MR_OLD_1 6
#define MR_OLD_TIMESTAMP_LO 7
#define MR_MU_READ 8
#define MR_MU_WRITE 9
#define MR_STRIDE 8

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}
// Reduce a u64 into [0, p) — matches GoldilocksField from_u64 / FE::from.
__device__ __forceinline__ uint64_t from_u64(uint64_t x) {
  return x >= goldilocks::PRIME ? x - goldilocks::PRIME : x;
}

extern "C" __global__ void
trace_memw_register_kernel(const uint64_t *__restrict__ in, uint64_t n,
                           uint64_t nrows, uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return; // padding rows already zeroed by alloc_zeros

  const uint64_t *op = in + r * MR_STRIDE;
  uint64_t base = op[0];
  uint64_t ts = op[1];
  uint64_t val0 = op[2];
  uint64_t val1 = op[3];
  uint64_t old0 = op[4];
  uint64_t old1 = op[5];
  uint64_t old_ts0 = op[6];
  uint64_t is_read = op[7];

  // ADDRESS = base_address / 2 (CPU sends 2 * register_index).
  sc(cols, nrows, MR_ADDRESS, r, from_u64(base / 2));
  // Timestamp split into lo/hi 32-bit words (DWordWL).
  sc(cols, nrows, MR_TIMESTAMP_0, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, MR_TIMESTAMP_1, r, ts >> 32);
  // Value (2 register words).
  sc(cols, nrows, MR_VAL_0, r, from_u64(val0));
  sc(cols, nrows, MR_VAL_1, r, from_u64(val1));
  // Old value.
  sc(cols, nrows, MR_OLD_0, r, from_u64(old0));
  sc(cols, nrows, MR_OLD_1, r, from_u64(old1));
  // Old timestamp low (upper limb shared with TIMESTAMP_1).
  sc(cols, nrows, MR_OLD_TIMESTAMP_LO, r, old_ts0 & 0xFFFFFFFF);
  // Multiplicity: read vs write.
  sc(cols, nrows, MR_MU_READ, r, is_read);
  sc(cols, nrows, MR_MU_WRITE, r, is_read ^ 1);
}
