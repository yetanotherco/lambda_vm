// GPU COMMIT-table fill. One thread per row (no dedup; mu=1 per real row).
// Mirrors generate_commit_trace in prover/src/tables/commit.rs. Padding rows
// (r >= n) are NOT all-zero: spec needs count=1 and address_incr=[1,0,0,0] so
// the unconditional ADD/SUB templates have valid carries.
//
// Per-op input is interleaved, stride C_STRIDE:
//   [timestamp, index, address, count, first, end, value]

#include "goldilocks.cuh"
#include <cstdint>

#define C_TIMESTAMP_0 0
#define C_INDEX 2
#define C_ADDRESS_0 3
#define C_ADDRESS_INCR_0 5 // DWordHL: 4 halves
#define C_COUNT_0 9
#define C_COUNT_DECR_0 11 // DWordHL: 4 halves
#define C_FIRST 15
#define C_END 16
#define C_VALUE 17
#define C_MU 18
#define C_STRIDE 7

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}
__device__ __forceinline__ uint64_t from_u64(uint64_t x) {
  return x >= goldilocks::PRIME ? x - goldilocks::PRIME : x;
}
// DWordHL: 4 little-endian 16-bit halves.
__device__ __forceinline__ void dwhl(uint64_t *cols, uint64_t nrows, int col,
                                     uint64_t r, uint64_t v) {
  sc(cols, nrows, col + 0, r, v & 0xFFFF);
  sc(cols, nrows, col + 1, r, (v >> 16) & 0xFFFF);
  sc(cols, nrows, col + 2, r, (v >> 32) & 0xFFFF);
  sc(cols, nrows, col + 3, r, (v >> 48) & 0xFFFF);
}

extern "C" __global__ void
trace_commit_kernel(const uint64_t *__restrict__ in, uint64_t n, uint64_t nrows,
                    uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n) {
    // Padding: count=1 (→ count_decr=0), address_incr[0]=1 (address=0 → +1).
    sc(cols, nrows, C_COUNT_0, r, 1);
    sc(cols, nrows, C_ADDRESS_INCR_0, r, 1);
    return;
  }

  const uint64_t *op = in + r * C_STRIDE;
  uint64_t ts = op[0];
  uint64_t index = op[1];
  uint64_t address = op[2];
  uint64_t count = op[3];
  uint64_t first = op[4];
  uint64_t end = op[5];
  uint64_t value = op[6];

  sc(cols, nrows, C_TIMESTAMP_0, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, C_TIMESTAMP_0 + 1, r, ts >> 32);
  sc(cols, nrows, C_INDEX, r, from_u64(index));
  sc(cols, nrows, C_ADDRESS_0, r, address & 0xFFFFFFFF);
  sc(cols, nrows, C_ADDRESS_0 + 1, r, address >> 32);
  dwhl(cols, nrows, C_ADDRESS_INCR_0, r, address + 1);
  sc(cols, nrows, C_COUNT_0, r, count & 0xFFFFFFFF);
  sc(cols, nrows, C_COUNT_0 + 1, r, count >> 32);
  uint64_t count_decr = (count == 0) ? ~0ULL : (count - 1);
  dwhl(cols, nrows, C_COUNT_DECR_0, r, count_decr);
  sc(cols, nrows, C_FIRST, r, first);
  sc(cols, nrows, C_END, r, end);
  sc(cols, nrows, C_VALUE, r, value);
  sc(cols, nrows, C_MU, r, 1);
}
