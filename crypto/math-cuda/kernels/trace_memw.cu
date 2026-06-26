// GPU MEMW_A (aligned) and MEMW (general) trace generation. One thread per row,
// no dedup (multiplicity from is_read). Mirrors generate_memw_aligned_trace /
// generate_memw_trace in prover/src/tables/{memw_aligned,memw}.rs. The ops
// already carry old/old_timestamp from the (CPU) memory-model walk; these
// kernels only move the per-row fill to the device. Padding rows (r >= n) stay
// all-zero, matching the CPU builders.

#include "goldilocks.cuh"
#include <cstdint>

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}
__device__ __forceinline__ uint64_t from_u64(uint64_t x) {
  return x >= goldilocks::PRIME ? x - goldilocks::PRIME : x;
}

// ---------------------------------------------------------------------------
// MEMW_A (aligned): 29 columns. Per-op input stride MA_STRIDE:
//   [is_register, base_address, value0..value7, timestamp, width,
//    old0..old7, old_timestamp0, is_read]   (1+1+8+1+1+8+1+1 = 22)
// ---------------------------------------------------------------------------
#define MA_IS_REGISTER 0
#define MA_BASE_ADDRESS 1 // DWordWHH: cols 1,2,3
#define MA_VALUE 4        // value[0..8] at cols 4..12
#define MA_TIMESTAMP_0 12
#define MA_WRITE2 14
#define MA_WRITE4 15
#define MA_WRITE8 16
#define MA_OLD 17 // old[0..8] at cols 17..25
#define MA_OLD_TIMESTAMP_0 25
#define MA_MU_READ 27
#define MA_MU_WRITE 28
#define MA_STRIDE 22

extern "C" __global__ void
trace_memw_aligned_kernel(const uint64_t *__restrict__ in, uint64_t n,
                          uint64_t nrows, uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return;

  const uint64_t *op = in + r * MA_STRIDE;
  uint64_t is_register = op[0];
  uint64_t base = op[1];
  const uint64_t *value = op + 2;
  uint64_t ts = op[10];
  uint64_t width = op[11];
  const uint64_t *old = op + 12;
  uint64_t old_ts0 = op[20];
  uint64_t is_read = op[21];

  sc(cols, nrows, MA_IS_REGISTER, r, is_register);
  // base_address DWordWHH = [low16, mid16, high32].
  sc(cols, nrows, MA_BASE_ADDRESS, r, base & 0xFFFF);
  sc(cols, nrows, MA_BASE_ADDRESS + 1, r, (base >> 16) & 0xFFFF);
  sc(cols, nrows, MA_BASE_ADDRESS + 2, r, (base >> 32) & 0xFFFFFFFF);
  for (int i = 0; i < 8; i++)
    sc(cols, nrows, MA_VALUE + i, r, from_u64(value[i]));
  sc(cols, nrows, MA_TIMESTAMP_0, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, MA_TIMESTAMP_0 + 1, r, ts >> 32);
  sc(cols, nrows, MA_WRITE2, r, width == 2);
  sc(cols, nrows, MA_WRITE4, r, width == 4);
  sc(cols, nrows, MA_WRITE8, r, width == 8);
  for (int i = 0; i < 8; i++)
    sc(cols, nrows, MA_OLD + i, r, from_u64(old[i]));
  sc(cols, nrows, MA_OLD_TIMESTAMP_0, r, old_ts0 & 0xFFFFFFFF);
  sc(cols, nrows, MA_OLD_TIMESTAMP_0 + 1, r, old_ts0 >> 32);
  sc(cols, nrows, MA_MU_READ, r, is_read);
  sc(cols, nrows, MA_MU_WRITE, r, is_read ^ 1);
}

// ---------------------------------------------------------------------------
// MEMW (general): 49 columns. Per-op input stride MW_STRIDE:
//   [is_register, base_address, value0..value7, timestamp, width,
//    old0..old7, old_timestamp0..old_timestamp7, is_read]
//   (1+1+8+1+1+8+8+1 = 29)
// ---------------------------------------------------------------------------
#define MW_IS_REGISTER 0
#define MW_BASE_ADDRESS_0 1 // DWordWL: cols 1,2
#define MW_VALUE 3          // value[0..8] at cols 3..11
#define MW_TIMESTAMP_0 11
#define MW_WRITE2 13
#define MW_WRITE4 14
#define MW_WRITE8 15
#define MW_OLD 16            // old[0..8] at cols 16..24
#define MW_CARRY 24          // carry[0..7] at cols 24..31
#define MW_OLD_TIMESTAMP_START 31 // old_timestamp[i] DWordWL at 31+2i
#define MW_MU_READ 47
#define MW_MU_WRITE 48
#define MW_STRIDE 29

extern "C" __global__ void
trace_memw_kernel(const uint64_t *__restrict__ in, uint64_t n, uint64_t nrows,
                  uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return;

  const uint64_t *op = in + r * MW_STRIDE;
  uint64_t is_register = op[0];
  uint64_t base = op[1];
  const uint64_t *value = op + 2;
  uint64_t ts = op[10];
  uint64_t width = op[11];
  const uint64_t *old = op + 12;
  const uint64_t *old_ts = op + 20;
  uint64_t is_read = op[28];

  sc(cols, nrows, MW_IS_REGISTER, r, is_register);
  uint64_t base_lo = base & 0xFFFFFFFF;
  sc(cols, nrows, MW_BASE_ADDRESS_0, r, base_lo);
  sc(cols, nrows, MW_BASE_ADDRESS_0 + 1, r, base >> 32);
  for (int i = 0; i < 8; i++)
    sc(cols, nrows, MW_VALUE + i, r, from_u64(value[i]));
  sc(cols, nrows, MW_TIMESTAMP_0, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, MW_TIMESTAMP_0 + 1, r, ts >> 32);
  sc(cols, nrows, MW_WRITE2, r, width == 2);
  sc(cols, nrows, MW_WRITE4, r, width == 4);
  sc(cols, nrows, MW_WRITE8, r, width == 8);
  for (int i = 0; i < 8; i++)
    sc(cols, nrows, MW_OLD + i, r, from_u64(old[i]));
  // carry[i] = 1 if (base_lo + i+1) >= 2^32, i = 0..7.
  for (int i = 0; i < 7; i++)
    sc(cols, nrows, MW_CARRY + i, r,
       (base_lo + (uint64_t)(i + 1)) >= (1ULL << 32));
  // old_timestamp[i] as DWordWL (lo, hi), i = 0..8.
  for (int i = 0; i < 8; i++) {
    sc(cols, nrows, MW_OLD_TIMESTAMP_START + 2 * i, r, old_ts[i] & 0xFFFFFFFF);
    sc(cols, nrows, MW_OLD_TIMESTAMP_START + 2 * i + 1, r, old_ts[i] >> 32);
  }
  sc(cols, nrows, MW_MU_READ, r, is_read);
  sc(cols, nrows, MW_MU_WRITE, r, is_read ^ 1);
}
